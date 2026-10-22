"""Explicit Windows process timer lease and interruptible, non-spinning waits."""

import ctypes
from dataclasses import dataclass
import sys
from time import perf_counter_ns


class _TimeCaps(ctypes.Structure):
    _fields_ = [("wPeriodMin", ctypes.c_uint), ("wPeriodMax", ctypes.c_uint)]


def _load_winmm():
    if sys.platform != "win32":
        raise RuntimeError("windows_event_wait_1ms requires Windows")
    version = sys.getwindowsversion()
    if version.major < 10 or version.build < 19041:
        raise RuntimeError("Process-scoped timer resolution requires Windows 10 2004 or newer")
    api = ctypes.WinDLL("winmm", use_last_error=True)
    api.timeGetDevCaps.argtypes = [ctypes.POINTER(_TimeCaps), ctypes.c_uint]
    api.timeGetDevCaps.restype = ctypes.c_uint
    for name in ("timeBeginPeriod", "timeEndPeriod"):
        function = getattr(api, name)
        function.argtypes, function.restype = [ctypes.c_uint], ctypes.c_uint
    return api, version.build


class WindowsTimerResolution:
    """One balanced timeBeginPeriod/timeEndPeriod pair per actual episode."""

    def __init__(self, period_ms):
        if type(period_ms) is not int or period_ms != 1:
            raise ValueError("windows_event_wait_1ms requires explicit integer timer_resolution_ms=1")
        self.period_ms, self._active = period_ms, False
        self.metadata = dict(strategy="windows_event_wait_1ms", requested_period_ms=period_ms,
                             scope="calling_process", status="not_acquired")

    def __enter__(self):
        if self._active:
            raise RuntimeError("Timer-resolution lease is already active")
        self._api, build = _load_winmm()
        caps = _TimeCaps()
        result = self._api.timeGetDevCaps(ctypes.byref(caps), ctypes.sizeof(caps))
        if result != 0:
            raise RuntimeError(f"timeGetDevCaps failed: MMRESULT={result}")
        if not caps.wPeriodMin <= self.period_ms <= caps.wPeriodMax:
            raise RuntimeError("The explicit 1ms timer resolution is outside device capabilities")
        self.metadata.update(windows_build=build, minimum_period_ms=caps.wPeriodMin,
                             maximum_period_ms=caps.wPeriodMax)
        result = self._api.timeBeginPeriod(self.period_ms)
        self.metadata["begin_result"] = result
        if result != 0:
            raise RuntimeError(f"timeBeginPeriod({self.period_ms}) failed: MMRESULT={result}")
        self._active = True
        self.metadata.update(status="active", acquired_at_monotonic_ns=perf_counter_ns())
        return self

    def __exit__(self, *_):
        result = self._api.timeEndPeriod(self.period_ms)
        self._active = False
        self.metadata.update(end_result=result, released_at_monotonic_ns=perf_counter_ns(),
                             status="released" if result == 0 else "release_failed")
        if result != 0:
            raise RuntimeError(f"timeEndPeriod({self.period_ms}) failed: MMRESULT={result}")


@dataclass(frozen=True)
class DeadlineWait:
    interrupted: bool
    requested_ns: int
    elapsed_ns: int

    @property
    def explicit_wait_ns(self):
        return min(self.requested_ns, self.elapsed_ns)

    @property
    def overshoot_ns(self):
        return max(0, self.elapsed_ns-self.requested_ns)


def wait_until(event, deadline_ns):
    started_ns = perf_counter_ns()
    requested_ns = max(0, deadline_ns-started_ns)
    if not requested_ns:
        return DeadlineWait(event.is_set(), 0, 0)
    interrupted = event.wait(requested_ns/1e9)
    return DeadlineWait(interrupted, requested_ns, perf_counter_ns()-started_ns)
