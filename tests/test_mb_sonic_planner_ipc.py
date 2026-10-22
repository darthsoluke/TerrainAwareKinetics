"""Independent CPU spawn/IPC checks; no frozen neural provider is constructed."""

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
import json
import multiprocessing
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pytest

from motionbricks.mb_sonic.planner_process import (
    GeneratorMetadata, PlannerRPC, ProcessMBProvider, _planner_main,
    _provider_metadata, _receive, _send, serve_provider,
)
from motionbricks.mb_sonic.reference import (
    G1Reference, NativeSomaHistory, ReferenceCoverageError, SomaReference,
)
from motionbricks.mb_sonic.runtime import MBGenerationSnapshot, MBReferenceUpdate, ReferencePacket
from motionbricks.mb_sonic.scene import load_scene, sha256


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "out/mb_sonic_e0_e3_20261018"
MAXIMUM_BYTES = 16 * 1024 * 1024


def _arrays(value, prefix=""):
    if isinstance(value, np.ndarray):
        yield prefix, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _arrays(getattr(value, field.name), prefix + "." + field.name)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _arrays(item, prefix + "." + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _arrays(item, prefix + "." + str(index))


class CPUProviderError(ReferenceCoverageError):
    """Importable exception used only to verify typed child-error transport."""


@dataclass(frozen=True)
class CPUProposal:
    source: SomaReference


class CPUProvider:
    def __init__(self, manifest_hash, robot_hash):
        self.scene_manifest_sha256, self.robot_model_sha256 = manifest_hash, robot_hash
        self.rig_id, self.execution, self.conditioning_mode = "cpu-ipc-test", "cpu-test", "B1"
        self.consumer_horizon_s, self.request_evidence = 1.12, None
        self.reference_stream_policy = None
        self.prepared_count = 0
        self.generator = GeneratorMetadata(
            dict(kind="explicit_cpu_test_provider", factory=make_cpu_provider.__module__ + ".make_cpu_provider",
                 factory_pid=os.getpid(), nested=dict(value=7)), dict(scene="000"))

    def cache_description(self):
        return dict(nested=dict(prepared_count=self.prepared_count))

    def prepare(self, history, *, actual_root_xyz_m, desired_heading_rad):
        self.prepared_count += 1
        return dict(history=history, root=actual_root_xyz_m, heading=desired_heading_rad)

    def bind_scene(self, scene):
        self.scene_manifest_sha256 = sha256(scene.manifest_path)
        self.generator.scene_query_receipt["scene"] = scene.manifest.scene_id

    def propose(self, snapshot, *, command, intent_revision, requested_at_ns, current_sim_time):
        started = perf_counter_ns()
        source, reference = command["payload"]["source"], command["payload"]["reference"]
        self.request_evidence = dict(source=source, reference=reference, child_pid=os.getpid(),
            received_readonly=all(not array.flags.writeable for _, array in _arrays((source, reference, snapshot))))
        assert self.request_evidence["received_readonly"], "Spawned provider inputs must be read-only"
        self.request_evidence["clock_times_s"] = (current_sim_time(), current_sim_time())
        if command.get("raise_error"):
            raise CPUProviderError("explicit CPU provider rejected the requested interval")
        clocks = dict(requested=requested_at_ns, reference_ready=perf_counter_ns())
        packet = ReferencePacket(
            schema_version=2, rig_id=self.rig_id, robot_model_sha256=self.robot_model_sha256,
            scene_manifest_sha256=self.scene_manifest_sha256, joint_names=reference.joint_names,
            units=dict(length="m", time="s", angle="rad"), frame="mujoco_world_z_up",
            issued_from_sim_time=snapshot.simulation_time_s, generated_from_command_seq=command["command_seq"],
            plan_id="cpu-ipc-test", state_version=snapshot.state_version,
            future_timestamps=reference.timestamps_s, root_pose=reference.qpos[:, :7],
            body_joint_positions=reference.joint_pos, body_joint_velocities=reference.joint_vel,
            valid_until=reference.trusted_until_s, conditioning_mode="B1",
            conditioning_history_times_s=snapshot.reference_history.timestamps_s,
            query_root_xyz_m=snapshot.actual_root_xyz_m, query_origin_soma_xyz=source.root_positions_m[0],
            query_origin_frame="soma_world_y_up", morphology_root_z_offset_m=0., clocks_ns=clocks,
            episode_id=command["episode_id"], session_id=command["session_id"])
        return MBReferenceUpdate(intent_revision, command, snapshot.history_time_s, snapshot.derived_heading_rad,
            snapshot.reference_history, None, CPUProposal(source), reference,
            dict(provider_total=perf_counter_ns() - started,
                 cpu_test_received_readonly=self.request_evidence["received_readonly"],
                 cpu_test_clock_times_s=self.request_evidence["clock_times_s"]), clocks, packet)


def make_cpu_provider(manifest_hash, robot_hash):
    """Top-level importable factory, executed inside the actual spawned child."""
    return CPUProvider(manifest_hash, robot_hash)


def _serve_cpu_factory(connection, manifest_hash, robot_hash):
    provider = make_cpu_provider(manifest_hash, robot_hash)
    _send(connection, dict(kind="cpu_initialized", metadata=_provider_metadata(provider)), MAXIMUM_BYTES)
    serve_provider(connection, provider, maximum_bytes=MAXIMUM_BYTES, callback_timeout_s=2.)


@pytest.fixture(scope="module")
def real_payload():
    path = ASSETS / "wasd_v3/episode_0001_numeric/19af187647d0857013f1f95c75aa0fd75bb3faf165f2be4c5d7955fa57734fd0.npz"
    with np.load(path, allow_pickle=False) as archive:
        record = json.loads(str(archive["metadata_json"]))["record"]

        def decode(value):
            if isinstance(value, dict):
                if "array" in value:
                    return archive[value["array"]].copy()
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value

        source_values = decode(record["raw_source"])
        source_values["bone_names"] = tuple(source_values["bone_names"])
        reference_values = decode(record["adapted_reference"])
        for key in ("body_names", "joint_names"):
            reference_values[key] = tuple(reference_values[key])
        source, reference = SomaReference(**source_values), G1Reference(**reference_values)
    return source, reference


def test_repeated_paired_feedback_references_restore_once_without_sharing_input(real_payload):
    from motionbricks.mb_sonic.planner_process import restore_ipc_value

    source, reference = real_payload
    payload = dict(samples=[dict(native_reference_source=source, robot_reference_source=reference) for _ in range(13)])
    restored = restore_ipc_value(payload)
    first = restored["samples"][0]
    assert first["native_reference_source"] is not source
    assert first["robot_reference_source"] is not reference
    for row in restored["samples"]:
        assert row["native_reference_source"] is first["native_reference_source"]
        assert row["robot_reference_source"] is first["robot_reference_source"]
    for key, original in (("native_reference_source", source), ("robot_reference_source", reference)):
        original_arrays = dict(_arrays(original))
        for name, array in _arrays(first[key]):
            assert not array.flags.writeable
            assert not np.shares_memory(array, original_arrays[name])
            assert array.tobytes() == original_arrays[name].tobytes()


@contextmanager
def spawned_cpu_provider():
    scene = load_scene(ASSETS / "scenes/000/scene_manifest.json")
    robot_hash = sha256(ROOT.parent / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_serve_cpu_factory,
        args=(child, sha256(scene.manifest_path), robot_hash), daemon=False)
    provider = None
    try:
        process.start()
        child.close()
        initialized = _receive(parent, MAXIMUM_BYTES, 10.)
        assert initialized["kind"] == "cpu_initialized"
        metadata = initialized["metadata"]
        assert metadata["identity"]["factory_pid"] == process.pid != os.getpid()
        assert metadata["identity"]["factory"].endswith(".make_cpu_provider")
        settings = dict(estimate_actual_diagnostics=False,
                        planner_process=dict(startup_timeout_s=5., rpc_timeout_s=2., shutdown_timeout_s=3.))
        provider = ProcessMBProvider(settings, scene, None, process,
            PlannerRPC(parent, maximum_bytes=MAXIMUM_BYTES, timeout_s=2.), metadata)
        yield provider
        provider.close()
        provider.close()
        assert process.exitcode == 0 and not process.is_alive()
        assert process.pid not in {item.pid for item in multiprocessing.active_children()}
        assert parent.closed
    finally:
        try:
            if provider is not None:
                provider.close()
        finally:
            parent.close()
            child.close()
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(3.)
                assert not process.is_alive(), "CPU test child must never be orphaned"


def _request(provider, payload, *, raise_error=False):
    source, reference = payload
    history = NativeSomaHistory(source.timestamps_s[:4], source.raw_features[:4], source.scene_id,
                                dict(kind="source_seed"))
    snapshot = MBGenerationSnapshot(float(reference.timestamps_s[0]), float(history.timestamps_s[-1]),
        37, 9, reference.qpos[0, :3], .12, history, None, perf_counter_ns(), {})
    command = dict(scene_id=source.scene_id, scene_manifest_sha256=provider.scene_manifest_sha256,
        episode_id=4, session_id="cpu-ipc-session", command_seq=8,
        payload=dict(source=source, reference=reference), raise_error=raise_error)
    return snapshot, command


def _assert_arrays_roundtrip(actual, expected):
    actual_arrays, expected_arrays = dict(_arrays(actual)), dict(_arrays(expected))
    assert actual_arrays.keys() == expected_arrays.keys() and actual_arrays
    for key, before in expected_arrays.items():
        after = actual_arrays[key]
        assert after.dtype == before.dtype and after.shape == before.shape
        assert after.tobytes() == before.tobytes(), key
        assert not after.flags.writeable and not np.shares_memory(after, before), key


def test_actual_spawn_preserves_real_reference_arrays_and_requested_clock_values(real_payload):
    with spawned_cpu_provider() as provider:
        snapshot, command = _request(provider, real_payload)
        requested_times = [snapshot.simulation_time_s + .02, snapshot.simulation_time_s + .04]
        clock = iter(requested_times)
        update = provider.propose(snapshot, command=command, intent_revision=9,
            requested_at_ns=perf_counter_ns(), current_sim_time=lambda: next(clock))
        assert type(update.proposal.source) is SomaReference
        assert type(update.adapted_reference) is G1Reference
        _assert_arrays_roundtrip(update.proposal.source, real_payload[0])
        _assert_arrays_roundtrip(update.adapted_reference, real_payload[1])
        _assert_arrays_roundtrip(update.packet.root_pose, real_payload[1].qpos[:, :7])
        assert provider.request_evidence is None
        assert update.timing_ns["cpu_test_received_readonly"]
        assert update.timing_ns["cpu_test_clock_times_s"] == tuple(requested_times)
        assert update.packet.state_version == 37 and update.packet.generated_from_command_seq == 8


def test_actual_spawn_preserves_typed_error_trace_and_evidence_then_remains_usable(real_payload):
    with spawned_cpu_provider() as provider:
        snapshot, command = _request(provider, real_payload, raise_error=True)
        with pytest.raises(CPUProviderError, match="rejected the requested interval") as caught:
            provider.propose(snapshot, command=command, intent_revision=9,
                requested_at_ns=perf_counter_ns(), current_sim_time=lambda: snapshot.simulation_time_s)
        assert "CPUProviderError" in caught.value.remote_traceback and "in propose" in caught.value.remote_traceback
        assert provider.failure_evidence["child_pid"] == provider._process.pid
        assert provider.failure_evidence["scope"]["state_version"] == 37
        assert "CPUProviderError" in json.dumps(provider.failure_evidence)
        _assert_arrays_roundtrip(provider.request_evidence["source"], real_payload[0])
        command["raise_error"] = False
        assert provider.propose(snapshot, command=command, intent_revision=9,
            requested_at_ns=perf_counter_ns(), current_sim_time=lambda: snapshot.simulation_time_s).packet.state_version == 37


def test_actual_spawn_prepare_and_bind_metadata_are_independent(real_payload):
    with spawned_cpu_provider() as provider:
        snapshot, _ = _request(provider, real_payload)
        response = provider._call("prepare", dict(history=snapshot.reference_history,
            actual_root_xyz_m=snapshot.actual_root_xyz_m, desired_heading_rad=.12))
        _assert_arrays_roundtrip(response["value"]["history"], snapshot.reference_history)
        identity = provider.generator.identity
        identity["nested"]["value"] = -1
        cache = provider.cache_description()
        cache["nested"]["prepared_count"] = -1
        cache["planner_process"]["source_sha256"]["planner_process.py"] = "tampered"
        assert provider.generator.identity["nested"]["value"] == 7
        assert provider.cache_description()["nested"]["prepared_count"] == 1
        assert provider.cache_description()["planner_process"]["source_sha256"]["planner_process.py"] != "tampered"
        scene = load_scene(ASSETS / "scenes/001/scene_manifest.json")
        provider.bind_scene(scene)
        assert provider.scene_manifest_sha256 == sha256(scene.manifest_path)
        assert provider.generator.scene_query_receipt["scene"] == "001"


def test_actual_spawn_rejects_changed_source_identity_before_any_provider_load():
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_planner_main, args=(child, {},
        str(ASSETS / "scenes/000/scene_manifest.json"), {}, MAXIMUM_BYTES, 2.), daemon=False)
    try:
        process.start()
        child.close()
        response = _receive(parent, MAXIMUM_BYTES, 10.)
        assert response["kind"] == "initialization_error"
        assert type(response["error"]) is ValueError
        assert "frozen sources" in str(response["error"])
        assert "_planner_main" in response["traceback"]
        process.join(3.)
        assert process.exitcode == 0 and not process.is_alive()
        assert process.pid not in {item.pid for item in multiprocessing.active_children()}
    finally:
        parent.close()
        child.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(3.)
            assert not process.is_alive()
