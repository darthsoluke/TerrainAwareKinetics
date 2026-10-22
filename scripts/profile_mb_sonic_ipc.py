"""CPU-only cached-payload IPC sidecar; never constructs a planner or physics."""

import argparse
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from queue import Queue
import sys
from threading import Event, Thread, local
from time import perf_counter_ns, thread_time_ns

import numpy as np

from motionbricks.mb_sonic import planner_process as production
from motionbricks.mb_sonic.mb_reference import MBProposal
from motionbricks.mb_sonic.reference import G1Reference, NativeSomaHistory, RawMBBlock, SomaReference
from motionbricks.mb_sonic.runtime import MBReferenceUpdate, ReferencePacket


MAXIMUM_BYTES = 16 * 1024 * 1024
TIMEOUT_S = 5.
MEASUREMENT = local()


def identity(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def arrays(value, prefix=""):
    if isinstance(value, np.ndarray):
        yield prefix, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from arrays(getattr(value, field.name), prefix + "." + field.name)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from arrays(item, prefix + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from arrays(item, prefix + "." + str(index))


def assert_equal(expected, actual):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert expected.dtype == actual.dtype and expected.shape == actual.shape
        assert expected.tobytes() == actual.tobytes()
        assert not actual.flags.writeable and actual.flags.owndata
        assert not np.shares_memory(expected, actual)
    elif is_dataclass(expected):
        assert type(expected) is type(actual)
        for field in fields(expected):
            assert_equal(getattr(expected, field.name), getattr(actual, field.name))
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual) and len(expected) == len(actual)
        for before, after in zip(expected, actual):
            assert_equal(before, after)
    else:
        assert expected == actual


def load_cache(record_path, recording_path, expected_sha256):
    file = identity(record_path)
    if file["sha256"] != expected_sha256:
        raise ValueError("Cached plan SHA256 differs from the explicit pin")
    with Path(recording_path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    logged = next(row for row in rows if row["kind"] == "numeric_artifact_written"
                  and Path(row["path"]).resolve() == Path(record_path).resolve())
    if logged["sha256"] != expected_sha256 or logged["bytes"] != file["bytes"]:
        raise ValueError("Cached plan identity differs from its actual recording")
    with np.load(record_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata["schema"] != "motionbricks.mb_sonic.numeric_record.v1":
            raise ValueError("Expected the actual lossless numeric-record schema")

        def decode(value):
            if isinstance(value, dict):
                if "array" in value:
                    array = archive[value["array"]]
                    if list(array.shape) != value["shape"] or array.dtype.str != value["dtype"]:
                        raise ValueError("Numeric member differs from its recorded descriptor")
                    if array.dtype.hasobject or not np.isfinite(array).all():
                        raise ValueError("Cached numeric arrays must be finite and non-object")
                    return array.copy()
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value

        record = decode(metadata["record"])
    if record["kind"] != "mb_raw_and_adapted_reference" or metadata["record_id"] != logged["record_id"]:
        raise ValueError("Expected a recorded MB update, not a tick or generated fixture")
    used = next(row for row in rows if row["kind"] == "reference_packet_used"
                and row["plan_id"] == record["plan_id"])
    source_values = dict(record["raw_source"], bone_names=tuple(record["raw_source"]["bone_names"]))
    source = SomaReference(**source_values)
    reference_values = dict(record["adapted_reference"])
    for name in ("body_names", "joint_names"):
        reference_values[name] = tuple(reference_values[name])
    reference = G1Reference(**reference_values)
    history = NativeSomaHistory(**record["conditioning_history"])
    estimated = record["estimated_actual_history_diagnostic"]
    if estimated is not None:
        raise ValueError("This bounded sidecar requires the recorded B1 diagnostic-disabled update")
    blocks = tuple(RawMBBlock(**block) for block in record["raw_blocks"])
    proposal = MBProposal(source, blocks, history, record["proposal_timing_ns"])
    packet_values = dict(record["packet"], joint_names=tuple(record["packet"]["joint_names"]))
    packet = ReferencePacket(**packet_values)
    assert np.array_equal(packet.future_timestamps, reference.timestamps_s)
    assert np.array_equal(packet.root_pose, reference.qpos[:, :7])
    assert np.array_equal(packet.body_joint_positions, reference.joint_pos)
    assert np.array_equal(packet.body_joint_velocities, reference.joint_vel)
    update = MBReferenceUpdate(record["intent_revision"], record["command"], record["snapshot_time_s"],
        used["derived_heading_rad"], history, estimated, proposal, reference,
        record["timing_ns"], record["clocks_ns"], packet)
    update = production.restore_ipc_value(update)
    scope = dict(scene_manifest_sha256=packet.scene_manifest_sha256, episode_id=packet.episode_id,
        session_id=packet.session_id, command_seq=packet.generated_from_command_seq,
        intent_revision=record["intent_revision"], state_version=packet.state_version)
    response = dict(kind="result", request_id=0, scope=scope, value=update, evidence=None,
        received_ns=record["clocks_ns"]["ipc_child_received"],
        reply_started_ns=record["clocks_ns"]["ipc_child_reply_started"])
    provenance = dict(file=file, recording=identity(recording_path), record_id=record["plan_id"],
        recorded_array_count=logged["array_count"], metadata=metadata,
        reconstruction="Actual saved values rebuilt into original dataclasses; no generation/retarget/FK. "
        "Logged post-IPC timing metadata is retained. Original Python alias graph and original Pipe bytes were not recorded.")
    return response, provenance


@contextmanager
def span(stats, name):
    wall, cpu = perf_counter_ns(), thread_time_ns()
    try:
        yield
    finally:
        stats[name] = dict(wall_ns=perf_counter_ns()-wall, thread_cpu_ns=thread_time_ns()-cpu)


def measured_send(connection, message, maximum_bytes):
    stats = MEASUREMENT.stats
    with span(stats, "request_pickle"):
        payload = pickle.dumps(message, protocol=5)
        if len(payload) > maximum_bytes:
            raise ValueError("Explicit IPC byte bound exceeded")
    with span(stats, "request_send"):
        connection.send_bytes(payload)


def measured_receive(connection, maximum_bytes, timeout_s):
    stats = MEASUREMENT.stats
    with span(stats, "response_poll"):
        if not connection.poll(max(0., timeout_s)):
            raise TimeoutError("Cached exchange exceeded5s")
    with span(stats, "response_receive_bytes"):
        payload = connection.recv_bytes(maxlength=maximum_bytes)
    stats["response_bytes"] = len(payload)
    if MEASUREMENT.raw:
        return payload
    with span(stats, "response_unpickle"):
        decoded = pickle.loads(payload)
    with span(stats, "response_restore_validation_owned_copies"):
        return production.restore_ipc_value(decoded)


def cached_child(connection, record, recording, sha):
    try:
        cached, provenance = load_cache(record, recording, sha)
        raw = pickle.dumps(cached, protocol=5)
        connection.send_bytes(pickle.dumps(dict(kind="ready", pid=os.getpid(), raw_bytes=len(raw),
            source_sha256=provenance["file"]["sha256"], torch_imported="torch" in sys.modules), protocol=5))
        while connection.poll(TIMEOUT_S):
            request = production.restore_ipc_value(pickle.loads(connection.recv_bytes(MAXIMUM_BYTES)))
            if request["operation"] == "close":
                return
            if request["scope"] != cached["scope"]:
                raise ValueError("Cached request scope differs")
            stats = {}
            received = perf_counter_ns()
            if request["operation"] == "raw":
                payload = raw
            else:
                response = dict(cached, request_id=request["request_id"], received_ns=received,
                                reply_started_ns=perf_counter_ns())
                with span(stats, "child_response_pickle"):
                    payload = pickle.dumps(response, protocol=5)
            if len(payload) > MAXIMUM_BYTES:
                raise ValueError("Cached response exceeds explicit bound")
            with span(stats, "child_response_send_bytes"):
                connection.send_bytes(payload)
            connection.send_bytes(pickle.dumps(dict(kind="telemetry", stages=stats,
                child_received_ns=received, child_send_finished_ns=perf_counter_ns()), protocol=5))
    finally:
        connection.close()


def validate_response(response, scope, request_id):
    if response["kind"] != "result" or response["scope"] != scope or response["request_id"] != request_id:
        raise ValueError("Cached result scope/request identity differs")


class PersistentExchange:
    """Diagnostic-only single CPU thread; one request and one full validated reply."""

    def __init__(self, connection):
        self.connection, self.queue = connection, Queue(maxsize=1)
        self.thread = Thread(target=self.run, name="ipc-profile-persistent", daemon=True)
        self.thread.start()

    def run(self):
        while True:
            task = self.queue.get()
            if task is None:
                return
            request, done, outcome, raw = task
            stats = {}
            MEASUREMENT.stats, MEASUREMENT.raw = stats, raw
            try:
                stats["exchange_entered_ns"] = perf_counter_ns()
                measured_send(self.connection, request, MAXIMUM_BYTES)
                value = measured_receive(self.connection, MAXIMUM_BYTES, TIMEOUT_S)
                if not raw:
                    validate_response(value, request["scope"], request["request_id"])
                outcome["value"], outcome["stats"] = value, stats
            except Exception as error:
                outcome["error"] = error
            finally:
                stats["exchange_finished_ns"] = perf_counter_ns()
                done.set()

    def call(self, request, *, raw=False):
        done, outcome = Event(), {}
        started = perf_counter_ns()
        self.queue.put_nowait((request, done, outcome, raw))
        if not done.wait(TIMEOUT_S):
            self.connection.close()
            raise TimeoutError("Persistent exchange exceeded5s")
        returned = perf_counter_ns()
        if "error" in outcome:
            raise outcome["error"]
        stats = outcome["stats"]
        stats["thread_dispatch_ns"] = stats.pop("exchange_entered_ns")-started
        stats["caller_resume_ns"] = returned-stats.pop("exchange_finished_ns")
        stats["call_wall_ns"] = returned-started
        return outcome["value"], stats

    def close(self):
        self.queue.put_nowait(None)
        self.thread.join(TIMEOUT_S)
        if self.thread.is_alive():
            raise TimeoutError("Persistent CPU thread did not exit")


def instrumented_restore_cost(value):
    """Separate diagnostic pass; recursive timing is excluded from RPC comparisons."""
    original = production.restore_ipc_value
    stack, totals = [], {}

    def timed(item):
        frame = dict(children=0)
        stack.append(frame)
        started = perf_counter_ns()
        try:
            return original(item)
        finally:
            elapsed = perf_counter_ns()-started
            stack.pop()
            if stack:
                stack[-1]["children"] += elapsed
            category = ("array_finite_check_and_owned_copy" if isinstance(item, np.ndarray) else
                        "dataclass_constructor_validation_and_internal_copies" if is_dataclass(item) else
                        "container_and_scalar_recursion")
            entry = totals.setdefault(category, dict(exclusive_wall_ns=0, calls=0))
            entry["exclusive_wall_ns"] += elapsed-frame["children"]
            entry["calls"] += 1

    production.restore_ipc_value = timed
    try:
        restored = timed(value)
    finally:
        production.restore_ipc_value = original
    assert_equal(value, restored)
    return totals


def distribution(values):
    return dict(n=len(values), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), p99=float(np.percentile(values, 99)), max=max(values))


def summarize(rows):
    result = {}
    for mode in ("raw", "fresh_thread_rpc", "persistent_thread_rpc"):
        selected = [row for row in rows if row["mode"] == mode and not row["warmup"]]
        scalars = ("call_wall_ns", "thread_dispatch_ns", "caller_resume_ns")
        result[mode] = {name.removesuffix("_ns")+"_ms": distribution([row["parent"][name]/1e6 for row in selected])
                        for name in scalars}
        for owner in ("parent", "child"):
            names = sorted({name for row in selected for name, value in row[owner].items() if isinstance(value, dict)})
            for name in names:
                for clock in ("wall_ns", "thread_cpu_ns"):
                    values = [row[owner][name][clock]/1e6 for row in selected if name in row[owner]]
                    result[mode][owner+"_"+name+"_"+clock.removesuffix("_ns")+"_ms"] = distribution(values)
    return result


def run_profile(args):
    cached, provenance = load_cache(args.record, args.recording, args.sha256)
    source_paths = [Path(production.__file__), Path(__file__).resolve().parents[1]/"motionbricks/mb_sonic/runtime.py"]
    sources_before = [identity(path) for path in source_paths]
    local_rows, restore_attribution = [], []
    for _ in range(20):
        stats = {}
        with span(stats, "response_pickle"):
            wire = pickle.dumps(cached, protocol=5)
        with span(stats, "response_unpickle"):
            decoded = pickle.loads(wire)
        with span(stats, "restore_validation_owned_copies"):
            restored = production.restore_ipc_value(decoded)
        assert_equal(cached, restored)
        local_rows.append(stats)
    for _ in range(5):
        restore_attribution.append(instrumented_restore_cost(pickle.loads(wire)))
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=cached_child,
        args=(child, str(args.record), str(args.recording), args.sha256), name="cached-mb-ipc-profile", daemon=False)
    rows, persistent = [], None
    original_send, original_receive = production._send, production._receive
    per_thread_stats = {}

    def fresh_send(connection, message, maximum):
        stats = {"exchange_entered_ns": perf_counter_ns()}
        MEASUREMENT.stats, MEASUREMENT.raw = stats, False
        per_thread_stats[message["request_id"]] = stats
        return measured_send(connection, message, maximum)

    try:
        process.start()
        child.close()
        if not parent.poll(TIMEOUT_S):
            raise TimeoutError("Single cached CPU child startup exceeded5s")
        ready = pickle.loads(parent.recv_bytes(MAXIMUM_BYTES))
        if ready["kind"] != "ready" or ready["source_sha256"] != args.sha256 or ready["torch_imported"]:
            raise ValueError("Cached CPU child identity/import scope differs")
        persistent = PersistentExchange(parent)
        rpc = production.PlannerRPC(parent, maximum_bytes=MAXIMUM_BYTES, timeout_s=TIMEOUT_S)
        production._send, production._receive = fresh_send, measured_receive
        counts = {mode: 0 for mode in ("raw", "fresh_thread_rpc", "persistent_thread_rpc")}
        for index in range(args.exchanges):
            mode = tuple(counts)[index % 3]
            request_id = 0 if mode == "raw" else index+1000
            request = dict(kind="request", request_id=request_id, operation="raw" if mode == "raw" else "cached",
                           arguments={}, scope=cached["scope"])
            if mode == "fresh_thread_rpc":
                response = rpc.call("cached", {}, scope=cached["scope"])
                returned = perf_counter_ns()
                stats = per_thread_stats.pop(response["request_id"])
                stats["thread_dispatch_ns"] = stats.pop("exchange_entered_ns")-response["ipc_parent_started_ns"]
                stats["caller_resume_ns"] = returned-response["ipc_parent_received_ns"]
                stats["call_wall_ns"] = returned-response["ipc_parent_started_ns"]
            else:
                response, stats = persistent.call(request, raw=mode == "raw")
                if mode == "raw":
                    response = production.restore_ipc_value(pickle.loads(response))
                validate_response(response, cached["scope"], request_id)
            assert_equal(cached["value"], response["value"])
            if not parent.poll(TIMEOUT_S):
                raise TimeoutError("Cached child telemetry exceeded5s")
            telemetry = pickle.loads(parent.recv_bytes(MAXIMUM_BYTES))
            assert telemetry["kind"] == "telemetry"
            rows.append(dict(mode=mode, warmup=counts[mode] < 2, parent=stats, child=telemetry["stages"],
                             bitwise_owned_readonly=True))
            counts[mode] += 1
        parent.send_bytes(pickle.dumps(dict(operation="close"), protocol=5))
        process.join(TIMEOUT_S)
        if process.is_alive() or process.exitcode != 0:
            raise RuntimeError("Cached CPU child did not exit cleanly")
    finally:
        production._send, production._receive = original_send, original_receive
        if persistent is not None:
            persistent.close()
        parent.close(); child.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(TIMEOUT_S)
    sources_after = [identity(path) for path in source_paths]
    assert sources_before == sources_after
    return dict(schema="motionbricks.mb_sonic.cached_ipc_profile.v1", status="PASS",
        scope="CPU cached data transport only, not model generation/physics or live timing causality",
        provenance=provenance, source_before=sources_before, source_after=sources_after,
        environment=dict(python=sys.version, numpy=np.__version__, platform=sys.platform,
                         thread_switch_interval_s=sys.getswitchinterval(), torch_imported="torch" in sys.modules),
        process=dict(start_method="spawn", pid=process.pid, exitcode=process.exitcode, spawned_children=1,
                     timed_exchanges=len(rows), timeout_s=TIMEOUT_S, alive_after_cleanup=process.is_alive()),
        payload=dict(pickle_protocol=5, response_bytes=len(wire), array_occurrences=len(list(arrays(cached))),
                     array_bytes=sum(array.nbytes for _, array in arrays(cached)), metadata_dropped=False,
                     evidence_field="None, matching successful production RPC; all recorded provenance remains"),
        local_phase_samples=local_rows, restore_attribution_samples=restore_attribution,
        attribution_scope="Separate instrumented passes; exclusive recursion timing includes instrumentation overhead; not RPC timings",
        exchanges=rows, summary=summarize(rows), optimizer_updates=0, gpu_seconds=0, physical_steps=0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exchanges", type=int, default=48)
    args = parser.parse_args(argv)
    if not 20 <= args.exchanges <= 50:
        parser.error("Use20..50 total cached exchanges in one bounded CPU child")
    if not args.output.name.startswith("ipc") or args.output.suffix != ".json":
        parser.error("Diagnostic output must be ipc*.json")
    return args


def main():
    args = parse_args()
    result = run_profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
    print(json.dumps(dict(output=str(args.output), payload=result["payload"],
                          process=result["process"], summary=result["summary"]), indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
