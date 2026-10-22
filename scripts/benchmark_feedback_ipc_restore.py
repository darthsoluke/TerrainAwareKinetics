"""CPU comparison of archived and current IPC restoration on saved feedback."""

import argparse
from dataclasses import fields, is_dataclass
import gc
import json
from pathlib import Path
import pickle
from time import perf_counter_ns
import tracemalloc

import numpy as np

from motionbricks.mb_sonic.planner_process import restore_ipc_value
from motionbricks.mb_sonic.scene import sha256
from scripts.probe_mb_actual_feedback import SavedEpisode


def arrays(value):
    if isinstance(value, np.ndarray):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from arrays(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from arrays(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--archived-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recording = SavedEpisode(args.recording)
    model_report = args.recording.parent / "episode_0001_000_loop_diagnosis_v1.json"
    # The scene's actual recorded body order is retained in the forensic source.
    probe = args.recording.parents[1] / "actual_feedback_preparation_v1/closed_v13_error_state_probe_v1.json"
    body_names = tuple(json.loads(probe.read_text())["body_names"])
    payload, receipts = recording.compact_history(30., body_names)
    wire = pickle.dumps(payload, protocol=5)
    received = pickle.loads(wire)
    source = args.archived_source.read_text(encoding="utf-8")
    function = source[source.index("def restore_ipc_value("):source.index("\ndef _send(")]
    namespace = dict(np=np, fields=fields, is_dataclass=is_dataclass)
    exec(compile(function, str(args.archived_source), "exec"), namespace)
    results = {}
    baseline = None
    for name, restore in (("archived", namespace["restore_ipc_value"]), ("memoized", restore_ipc_value)):
        gc.collect()
        times = []
        for _ in range(30):
            start = perf_counter_ns()
            result = restore(received)
            times.append((perf_counter_ns()-start)/1e6)
        values = [(array.dtype.str, array.shape, array.tobytes()) for array in arrays(result)]
        if baseline is None:
            baseline = values
        elif values != baseline:
            raise ValueError("Restored numeric values changed")
        tracemalloc.start()
        result = restore(received)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[name] = dict(p50_ms=float(np.percentile(times, 50)), p99_ms=float(np.percentile(times, 99)),
            peak_traced_bytes=peak, array_visits=len(values), unique_arrays=len({id(a) for a in arrays(result)}),
            all_readonly=all(not a.flags.writeable for a in arrays(result)))
    report = dict(schema="motionbricks.feedback_ipc_restore_comparison.v1", optimizer_updates=0,
        new_physics_steps=0, gpu_seconds=0, wire_bytes=len(wire), numeric_values_equal=True,
        recorded_sample_count=len(payload.samples), original_receipts=receipts, results=results,
        archived_source_sha256=sha256(args.archived_source), current_source_sha256=sha256(Path(__file__).parents[1]/"motionbricks/mb_sonic/planner_process.py"))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(wire_bytes=len(wire), results=results), indent=2))


if __name__ == "__main__":
    main()
