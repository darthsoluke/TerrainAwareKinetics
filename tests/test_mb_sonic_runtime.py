"""Observation equations and real CPU MuJoCo/ONNX integration, without training."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import re
from time import perf_counter_ns
from threading import Event, Lock, Thread
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from motionbricks.mb_sonic.reference import OFFICIAL_BODY_NAMES, load_official_reference
from motionbricks.mb_sonic.runtime import (BodyOnsetObserver, IntentMailbox, PhysicsRecorder,
                                          MBReferenceUpdate, NumericArtifactWriter, NumericLogBackpressure,
                                          FrozenMBProvider, MBGenerationSnapshot, OnlineMBWorker,
                                          ReferencePacket, SonicLabRuntime,
                                          RecordingBudget, RuntimeTick, PhysicsState, ContactForce,
                                          SonicPhysicsRuntime, YawIntentIntegrator, load_mb_seed_history,
                                          _reference_body_parents,
                                          calibrated_native_query_origin, consumer_horizon_contract, create_lab_runtime,
                                          load_physics_scene, next_controller_grid)
from motionbricks.mb_sonic.scene import export_scene
from motionbricks.mb_sonic import windows_clock
from motionbricks.mb_sonic.sonic_policy import (
    ACTION_SCALE, DECODER_FIELDS, DEFAULT_ANGLES, ENCODER_FIELDS, FrozenSonicPolicy,
    ISAAC_TO_MJ, KD, KP, MJ_TO_ISAAC, ObservationHistory, POLICY_DT, Proprioception,
    SonicObservationBuilder,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / "motionbricks/out/mb_sonic_e0_e3_20261018/sonic_6733128"
ROBOT_XML = ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
REFERENCE_DIR = ROOT / "gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428"
ENCODER_HASH = "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"
DECODER_HASH = "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"


@pytest.fixture(scope="module")
def physics_scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("sonic_physics")
    source = root / "flat.npz"
    np.savez(source, heightfield_y_m=np.zeros((5, 5), np.float32),
             min_point_world_xyz_m=np.array([-10., 0., -10.]), dx_m=np.array(5.),
             metadata_json=np.array(json.dumps(dict(coordinate_frame="soma_world_y_up"))))
    manifest = export_scene(source, root / "canonical", scene_id="runtime_test_flat",
                            friction=(1., .005, .0001), friction_source="explicit_test_fixture")
    return load_physics_scene(manifest, ROBOT_XML)


@pytest.fixture(scope="module")
def policy():
    return FrozenSonicPolicy(POLICY_DIR / "model_encoder.onnx", POLICY_DIR / "model_decoder.onnx",
                             POLICY_DIR / "observation_config.yaml",
                             encoder_sha256=ENCODER_HASH, decoder_sha256=DECODER_HASH)


@pytest.fixture(scope="module")
def source(physics_scene):
    return load_official_reference(REFERENCE_DIR, scene_id=physics_scene[0].manifest.scene_id)


def initialized_runtime(physics_scene, policy=None, **kwargs):
    scene, model, _ = physics_scene
    runtime = SonicPhysicsRuntime(scene, model, policy, **kwargs)
    qpos = model.qpos0.copy()
    qpos[runtime.qpos_indices] = DEFAULT_ANGLES
    runtime.reset(qpos, np.zeros(model.nv))
    return runtime


def test_joint_permutations_match_official_cpp():
    path = ROOT / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"
    text = path.read_text(encoding="utf-8")
    for name, expected in (("isaaclab_to_mujoco", ISAAC_TO_MJ), ("mujoco_to_isaaclab", MJ_TO_ISAAC)):
        body = re.search(rf"{name}\s*=\s*\{{([^}}]+)\}}", text).group(1)
        np.testing.assert_array_equal(np.fromstring(body, sep=",", dtype=int), expected)
    np.testing.assert_array_equal(MJ_TO_ISAAC[ISAAC_TO_MJ], np.arange(29))


def test_official_pd_and_action_scaling():
    omega = 20 * 3.1415926535
    assert KP[0] == pytest.approx(.025101925 * omega**2)
    assert KD[0] == pytest.approx(4 * .025101925 * omega)
    assert KP[4] == pytest.approx(2 * .003609725 * omega**2)
    assert ACTION_SCALE[4] == pytest.approx(.25 * 25 / (.003609725 * omega**2))
    assert ACTION_SCALE[0] == pytest.approx(.25 * 139 / (.025101925 * omega**2))
    np.testing.assert_allclose(DEFAULT_ANGLES[[0, 3, 4, 15, 16, 18, 23]],
                               [-.312, .669, -.363, .2, .2, .6, -.2])


def test_history_has_no_padding_and_uses_actual_oldest_first():
    history = ObservationHistory()
    with pytest.raises(RuntimeError, match="ten actual"):
        history.decoder_input(np.zeros(64))
    for frame in range(10):
        history.append(Proprioception(
            frame * POLICY_DT, DEFAULT_ANGLES+np.arange(29)+frame,
            np.arange(29)*2+frame, [1, 0, 0, 0], [frame, frame+1, frame+2],
            np.arange(29)+3*frame))
    obs = history.decoder_input(np.arange(64))[0]
    assert obs.shape == (994,)
    np.testing.assert_array_equal(obs[:64], np.arange(64))
    np.testing.assert_array_equal(obs[64:94].reshape(10, 3)[[0, 9]], [[0, 1, 2], [9, 10, 11]])
    np.testing.assert_allclose(obs[94:384].reshape(10, 29),
                               (np.arange(29)[None]+np.arange(10)[:, None])[:, MJ_TO_ISAAC])
    np.testing.assert_allclose(obs[384:674].reshape(10, 29)[0], 2*np.arange(29)[MJ_TO_ISAAC])
    np.testing.assert_array_equal(obs[-30:].reshape(10, 3), np.tile([0, 0, -1], (10, 1)))
    with pytest.raises(ValueError, match="exactly 50Hz"):
        history.append(Proprioception(1., DEFAULT_ANGLES, np.zeros(29), [1, 0, 0, 0],
                                      np.zeros(3), np.zeros(29)))


def test_encoder_mode_zero_mask_and_official_joint_positions(source):
    builder = SonicObservationBuilder(POLICY_DIR / "observation_config.yaml")
    window = source.sonic_window(0)
    builder.reset_heading(window.qpos[0, 3:7], window.qpos[0, 3:7])
    obs = builder.encoder_input(window, window.qpos[0, 3:7])[0]
    assert obs.shape == (1762,)
    assert sum(width for _, width in ENCODER_FIELDS) == 1762
    assert sum(width for _, width in DECODER_FIELDS) == 994
    np.testing.assert_array_equal(obs[:4], 0)
    np.testing.assert_allclose(obs[4:294].reshape(10, 29), window.joint_pos_isaaclab, atol=1e-7)
    np.testing.assert_allclose(obs[294:584].reshape(10, 29), window.joint_vel_isaaclab, atol=1e-6)
    for name, _ in ENCODER_FIELDS:
        if name not in ("motion_joint_positions_10frame_step5", "motion_joint_velocities_10frame_step5",
                        "motion_anchor_orientation_10frame_step5"):
            np.testing.assert_array_equal(obs[builder.encoder_slices[name]], 0)


def test_orientation_first_two_columns_are_rowwise(source):
    builder = SonicObservationBuilder(POLICY_DIR / "observation_config.yaml")
    window = source.sonic_window(0)
    quats = window.qpos.copy()
    theta = .3
    quats[:, 3:7] = [np.cos(theta/2), np.sin(theta/2), 0, 0]
    window = replace(window, qpos=quats)
    builder.reset_heading([1, 0, 0, 0], quats[0, 3:7])
    obs = builder.encoder_input(window, [1, 0, 0, 0])[0]
    expected = [1, 0, 0, np.cos(theta), 0, np.sin(theta)]
    np.testing.assert_allclose(obs[builder.encoder_slices["motion_anchor_orientation_10frame_step5"]]
                               .reshape(10, 6), np.tile(expected, (10, 1)), atol=1e-7)


def test_physics_keeps_official_hands_and_only_canonical_ground(physics_scene):
    scene, model, metadata = physics_scene
    assert (model.nq, model.nv, model.nu) == (50, 49, 43)
    assert "original_collision" in metadata["hands"]
    assert model.nhfield == 1
    assert all(model.geom(index).name != "floor" for index in range(model.ngeom))
    assert model.geom("mb_ground").type == mujoco.mjtGeom.mjGEOM_HFIELD
    assert model.joint("left_hand_thumb_0_joint").id > 0
    assert scene.query_height([[0, 0]]).height_m[0] == 0


def test_real_history_warmup_preserves_free_root_and_native_rates(physics_scene):
    runtime = initialized_runtime(physics_scene)
    start = runtime.data.qpos.copy()
    frames = runtime.warmup_history()
    assert len(frames) == 36
    assert runtime.time_s == pytest.approx(.18)
    assert len(runtime.history.entries) == 10
    assert runtime.model.opt.timestep == .005
    assert runtime.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    assert not np.array_equal(runtime.data.qpos, start)
    np.testing.assert_array_equal(runtime.data.qfrc_applied, 0)
    np.testing.assert_array_equal(runtime.data.xfrc_applied, 0)
    np.testing.assert_array_equal(runtime.data.qfrc_actuator[:6], 0)
    np.testing.assert_allclose(np.diff([frame.time_s for frame in frames]), .005, atol=1e-12)
    assert len({entry.time_s for entry in runtime.history.entries}) == 10


def test_pd_clips_real_torques_not_joint_state(physics_scene):
    runtime = initialized_runtime(physics_scene)
    target = np.full(29, 100.)
    states = runtime._advance_pd(target)
    assert len(states) == 4
    np.testing.assert_allclose(states[0].actuator_controls[runtime.actuator_ids], runtime.torque_upper)
    assert not np.allclose(states[-1].joint_positions, target)
    assert not runtime.terminal_reason


def test_contacts_are_actual_mujoco_wrenches(physics_scene):
    runtime = initialized_runtime(physics_scene)
    runtime.warmup_history()
    state = runtime.snapshot()
    assert state.contacts
    for index, contact in enumerate(state.contacts):
        expected = np.empty(6)
        mujoco.mj_contactForce(runtime.model, runtime.data, index, expected)
        np.testing.assert_array_equal(contact.wrench_contact, expected)
    assert state.root_ground_known
    assert state.root_ground_height_m == 0
    np.testing.assert_array_equal(state.qfrc_applied, 0)
    np.testing.assert_array_equal(state.xfrc_applied, 0)
    np.testing.assert_array_equal(state.root_qfrc_actuator, 0)
    assert all(warning["number"] == 0 for warning in state.solver_warnings.values())


def test_termination_preserves_state_and_requires_explicit_reset(physics_scene):
    runtime = initialized_runtime(physics_scene)
    runtime._advance_pd(DEFAULT_ANGLES)
    qpos, qvel, time = runtime.data.qpos.copy(), runtime.data.qvel.copy(), runtime.time_s
    state = runtime.terminate("explicit_test_failure")
    with pytest.raises(RuntimeError, match="Explicit reset"):
        runtime._advance_pd(DEFAULT_ANGLES)
    assert state.terminal_reason == "explicit_test_failure"
    assert runtime.time_s == time
    np.testing.assert_array_equal(runtime.data.qpos, qpos)
    np.testing.assert_array_equal(runtime.data.qvel, qvel)


def test_core_has_no_root_height_fall_threshold(physics_scene):
    runtime = initialized_runtime(physics_scene)
    qpos = runtime.data.qpos.copy()
    qpos[2] = .15
    runtime.reset(qpos, np.zeros(runtime.model.nv))
    runtime._advance_pd(DEFAULT_ANGLES)
    assert runtime.time_s > 0
    assert runtime.terminal_reason != "fall"
    assert runtime.data.qpos[2] != pytest.approx(runtime.model.qpos0[2])


def test_pinned_actual_ort_pair_and_real_rollout(physics_scene, policy, source):
    assert policy.encoder.get_providers() == ["CPUExecutionProvider"]
    assert policy.decoder.get_inputs()[0].shape == [1, 994]
    runtime = initialized_runtime(physics_scene, policy)
    runtime.warmup_history()
    shifted = replace(source, timestamps_s=source.timestamps_s+runtime.time_s,
                      trusted_until_s=source.trusted_until_s+runtime.time_s)
    tick = runtime.step(shifted, raw_reference=dict(kind="official_csv"),
                        issued_at_monotonic_ns=perf_counter_ns(), phase="official_walking_reference")
    assert len(tick.physics_states) == 4
    assert tick.actual.time_s == pytest.approx(.2)
    assert tick.policy_command.decoder_input.shape == (1, 994)
    assert np.isfinite(tick.policy_command.action_isaac).all()
    np.testing.assert_allclose(tick.policy_command.joint_target,
                               DEFAULT_ANGLES+tick.policy_command.action_isaac[ISAAC_TO_MJ]*ACTION_SCALE)
    assert not np.array_equal(tick.actual.joint_positions, tick.policy_command.joint_target)
    assert tick.timing_ns["chain_total"] > tick.timing_ns["encoder"] > 0
    with pytest.raises(ValueError, match="same canonical"):
        runtime.step(replace(shifted, scene_id="different"), raw_reference={},
                     issued_at_monotonic_ns=perf_counter_ns(), phase="walking")


def test_recording_retains_physics_and_reference_without_overwrite(tmp_path, physics_scene):
    runtime = initialized_runtime(physics_scene)
    path = tmp_path / "actual.jsonl"
    with PhysicsRecorder(path, dict(scene_id=runtime.scene_id)) as recorder:
        recorder.write(dict(kind="actual", state=asdict(runtime.snapshot())))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "metadata"
    assert len(rows[1]["state"]["qpos"]) == 50
    with pytest.raises(FileExistsError):
        PhysicsRecorder(path, {})


@pytest.fixture(scope="module")
def lab_runtime(tmp_path_factory, physics_scene):
    root = tmp_path_factory.mktemp("sonic_lab")
    scene = physics_scene[0]
    config = dict(schema="motionbricks.mb_sonic.runtime_config.v1", mode="official_reference",
                  scene_root=str(scene.root.parent), initial_scene_id=scene.manifest.scene_id,
                  enabled_scene_ids=[scene.manifest.scene_id], robot_model=str(ROBOT_XML),
                  reference_dir=str(REFERENCE_DIR), policy_dir=str(POLICY_DIR),
                  encoder_sha256=ENCODER_HASH, decoder_sha256=DECODER_HASH,
                  observation_config_sha256="466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b",
                  policy_threads=1, physics_dt_s=.005, duration_s=.2,
                  clock=dict(strategy="windows_event_wait_1ms", timer_resolution_ms=1),
                  recording_dir=str(root / "recordings"))
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    runtime = create_lab_runtime(path, allocation_id="cpu_unit_config_assets_only")
    yield runtime
    runtime.stop()


def test_lab_config_and_polls_do_not_advance_or_start_physics(lab_runtime):
    before = lab_runtime.physics.data.qpos.copy()
    config = lab_runtime.config()
    assert config["control_mode"] == "official_reference_read_only"
    assert config["intent_enabled"] is False
    assert config["replan_clock_owner"] is None
    assert lab_runtime._thread is None
    assert config["reference"]["parents"][0] == -1
    assert len(config["reference"]["parents"]) == 14
    for _ in range(20):
        snapshot = lab_runtime.snapshot()
        assert snapshot["sim_time_s"] == 0
        assert snapshot["command"] is None
        assert snapshot["actual"]["body_onset_command_id"] is None
    np.testing.assert_array_equal(lab_runtime.physics.data.qpos, before)


@pytest.mark.parametrize("mode,pending", [("mb", .1), ("mb", .2), ("official_reference", None)])
def test_admission_limits_public_config_and_recording_keep_actual_contract(tmp_path, mode, pending):
    from motionbricks.mb_sonic.runtime import _admission_limits
    settings = (dict(max_generation_state_age_s=.1, max_plan_wall_age_s=.1,
                     max_pending_intent_age_s=pending) if mode == "mb" else {})
    expected = dict(settings) if mode == "mb" else None
    lab = SonicLabRuntime.__new__(SonicLabRuntime)
    lab.hardware_execution_profile = 'normal_dynamic_gpu_power'
    lab._lock = Lock()
    lab._configuration = dict(admission_limits=_admission_limits(mode, settings))
    public = lab.config()
    assert public["admission_limits"] == expected
    path = tmp_path/"admission.jsonl"
    with PhysicsRecorder(path, dict(admission_limits=public["admission_limits"])):
        pass
    metadata = json.loads(path.read_text().splitlines()[0])
    assert metadata["admission_limits"] == expected
    if mode == "mb":
        public["admission_limits"]["max_pending_intent_age_s"] = 99
        assert lab.config()["admission_limits"] == expected
        assert settings["max_pending_intent_age_s"] == pending


def test_lab_assets_are_real_packed_content_addressed_meshes(lab_runtime):
    import hashlib

    config = lab_runtime.config()
    for geometry in config["scene"]["geometries"]+config["robot"]["geometries"]:
        description = geometry["mesh"]
        packed = lab_runtime.asset(description["asset_id"])
        assert len(packed) == (description["vertex_count"]*3+description["index_count"])*4
        assert hashlib.sha256(packed).hexdigest() == description["asset_id"]
        vertices = np.frombuffer(packed, dtype="<f4", count=description["vertex_count"]*3)
        indices = np.frombuffer(packed, dtype="<u4", offset=description["vertex_count"]*12)
        assert np.isfinite(vertices).all()
        assert indices.max() < description["vertex_count"]
    state = lab_runtime.snapshot()
    assert len(state["actual"]["geom_poses"]) == len(config["robot"]["geometries"])
    json.dumps(config, allow_nan=False)
    json.dumps(state, allow_nan=False)


def test_lab_reference_mode_never_claims_joystick_consumption(lab_runtime):
    with pytest.raises(ValueError, match="read-only"):
        lab_runtime.submit_intent(dict(command_id=1, velocity_xy_mps=[1, 0], heading_rad=0))
    response = lab_runtime.submit_intent(dict(source="disconnect_stop"))
    assert response["consumed"] is False
    assert lab_runtime.snapshot()["command"] is None
    assert lab_runtime.physics.time_s == 0


def test_lab_start_only_arms_until_explicit_geometry_ready_gate(lab_runtime):
    lab_runtime.start()
    for _ in range(20):
        assert lab_runtime.snapshot()["sim_time_s"] == 0
        assert lab_runtime.config()["start_required"] is True
    assert not lab_runtime._begin_event.is_set()
    assert lab_runtime._thread.is_alive()
    lab_runtime.stop()
    assert not lab_runtime._thread.is_alive()
    assert lab_runtime.physics.time_s == 0


def _snapshot_subscription_lab():
    lab = object.__new__(SonicLabRuntime)
    lab._lock, lab._snapshot_subscription_lock = Lock(), Lock()
    lab._snapshot_subscriber = None
    lab._latest, lab._failure = dict(frame_id=1, status="awaiting_start"), None
    return lab


def test_snapshot_subscription_is_exclusive_immediate_and_never_copies(monkeypatch):
    import motionbricks.mb_sonic.runtime as module
    lab, calls = _snapshot_subscription_lab(), []
    monkeypatch.setattr(module.copy, "deepcopy", lambda value: pytest.fail("Wakeups must not copy snapshots"))
    unsubscribe = lab.subscribe_snapshots(lambda: calls.append(1))
    assert calls == [1]
    with pytest.raises(RuntimeError, match="already owns"):
        lab.subscribe_snapshots(lambda: None)
    lab._notify_snapshots()
    assert calls == [1, 1]
    unsubscribe()
    unsubscribe()
    lab._notify_snapshots()
    assert calls == [1, 1]
    with pytest.raises(TypeError, match="callable"):
        lab.subscribe_snapshots(None)


def test_snapshot_unsubscribe_race_never_removes_a_new_owner():
    lab, calls = _snapshot_subscription_lab(), []
    entered, release = Event(), Event()

    def old_callback():
        calls.append("old")
        if len(calls) > 1:
            entered.set()
            assert release.wait(2)

    old_unsubscribe = lab.subscribe_snapshots(old_callback)
    thread = Thread(target=lab._notify_snapshots)
    thread.start()
    try:
        assert entered.wait(2)
        old_unsubscribe()
        new_unsubscribe = lab.subscribe_snapshots(lambda: calls.append("new"))
        old_unsubscribe()
        lab._notify_snapshots()
        assert calls == ["old", "old", "new", "new"]
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    new_unsubscribe()


def test_snapshot_callback_exceptions_and_reentrant_unsubscribe_do_not_affect_runtime():
    lab, calls = _snapshot_subscription_lab(), []

    def broken_callback():
        assert lab._lock.acquire(blocking=False)
        lab._lock.release()
        assert lab._snapshot_subscription_lock.acquire(blocking=False)
        lab._snapshot_subscription_lock.release()
        calls.append(1)
        raise RuntimeError("UI event loop closed")

    unsubscribe = lab.subscribe_snapshots(broken_callback)
    lab._notify_snapshots()
    assert lab._failure is None and lab._latest["status"] == "awaiting_start"
    assert calls == [1, 1]
    unsubscribe()
    handle = [None]

    def cancel_callback():
        calls.append(2)
        if handle[0] is not None:
            handle[0]()

    handle[0] = lab.subscribe_snapshots(cancel_callback)
    lab._notify_snapshots()
    lab._notify_snapshots()
    assert calls == [1, 1, 2, 2]
    assert lab._snapshot_subscriber is None


def test_snapshot_subscription_survives_manual_reset_without_starting_physics():
    lab, frames = _snapshot_subscription_lab(), []
    lab._catalog, lab._recording_budget = [dict(id="000", available=True)], None
    lab.stop, lab.start = lambda: None, lambda: None
    lab.config = lambda: dict(episode=2)
    lab._errors = ["retained_previous_episode_failure"]

    def load(scene_id):
        assert lab._errors == []
        lab._latest = dict(frame_id=2, status="awaiting_start")

    lab._load_episode = load
    unsubscribe = lab.subscribe_snapshots(lambda: frames.append(lab.snapshot()["frame_id"]))
    assert lab.select_scene("000") == dict(episode=2)
    assert frames == [1, 2]
    unsubscribe()


def test_snapshot_runtime_failure_wakes_before_fallible_logging(monkeypatch):
    import motionbricks.mb_sonic.runtime as module
    lab, notices, reasons = _snapshot_subscription_lab(), [], []
    lab._status, lab._errors = "running", []
    lab.physics = SimpleNamespace(terminate=reasons.append, snapshot=lambda: None)

    def failed_write(record):
        raise OSError("disk unavailable")

    lab._recorder = SimpleNamespace(write=failed_write)
    monkeypatch.setattr(module, "asdict", lambda state: {})
    unsubscribe = lab.subscribe_snapshots(lambda: notices.append(lab._failure))
    with pytest.raises(OSError, match="disk unavailable"):
        lab._record_runtime_failure(ValueError("coverage exhausted"))
    assert notices == [None, "ValueError: coverage exhausted"]
    assert lab._latest["status"] == "runtime_failure"
    assert reasons == ["ValueError: coverage exhausted"]
    with pytest.raises(RuntimeError, match="coverage exhausted"):
        lab.snapshot()
    unsubscribe()


def test_snapshot_warmup_publication_notifies_after_releasing_state_lock(monkeypatch):
    import motionbricks.mb_sonic.runtime as module
    # Publication plumbing only: the physics step is an analytic unit-test input.
    lab, notices = _snapshot_subscription_lab(), []
    lab.mode, lab.run_mode, lab.settings = "official_reference", "finite", {}
    lab._stop_event, lab._errors, lab._physics_steps = Event(), [], 0
    lab._mb_worker, lab._recording_budget = None, None
    state = SimpleNamespace(time_s=.005)
    lab.physics = SimpleNamespace(physics_dt=.005, substeps=4, needs_warmup=True,
        warmup_joint_target=DEFAULT_ANGLES,
        history=SimpleNamespace(entries=[]), terminal_reason="unit_stop",
        _advance_pd=lambda *args, **kwargs: [state])
    lab._recorder = SimpleNamespace(write=lambda row: None)
    lab._close_numeric_logs = lambda: None
    lab._publish = lambda published: setattr(lab, "_latest", dict(frame_id=2, status=lab._status))
    monkeypatch.setattr(module, "wait_until", lambda *args: SimpleNamespace(interrupted=False))
    monkeypatch.setattr(module, "asdict", lambda value: {})

    def notification():
        assert lab._lock.acquire(blocking=False)
        lab._lock.release()
        notices.append(lab.snapshot())

    unsubscribe = lab.subscribe_snapshots(notification)
    lab._run_paced()
    assert notices == [dict(frame_id=1, status="awaiting_start"),
                       dict(frame_id=2, status="terminal_failure")]
    assert lab._physics_steps == 1
    unsubscribe()


def test_intent_mailbox_keeps_only_latest_and_snapshots_are_private():
    mailbox = IntentMailbox()
    assert mailbox.latest() == (0, None)
    command = dict(command_id=1, velocity_xy_mps=[1., 0.], server_received_ns=10)
    mailbox.replace(command)
    command["velocity_xy_mps"][0] = 9.
    mailbox.replace(dict(command_id=2, velocity_xy_mps=[0., 1.], server_received_ns=20))
    revision, latest = mailbox.latest()
    assert revision == 2 and latest["command_id"] == 2
    latest["velocity_xy_mps"][1] = 7.
    assert mailbox.latest()[1]["velocity_xy_mps"] == [0., 1.]


def test_pending_intent_deadline_cannot_be_extended_by_continuous_input():
    mailbox = IntentMailbox()
    for sequence in range(1, 101):
        mailbox.replace(dict(command_id=sequence, server_received_ns=sequence*10_000_000))
    assert mailbox.pending_since_ns(0) == 10_000_000
    assert mailbox.pending_since_ns(80) == 810_000_000
    assert mailbox.pending_since_ns(100) is None
    assert not mailbox._unapplied


@pytest.mark.parametrize("elapsed_ns,expired", [(199_999_999, False), (200_000_000, False),
                                               (200_000_001, True), (220_000_000, True)])
def test_pending_deadline_is_checked_against_last_policy_use_not_new_adoption(elapsed_ns, expired):
    lab = object.__new__(SonicLabRuntime)
    lab._intents, lab._consumed_intent_revision = IntentMailbox(), 1
    lab.settings = dict(max_pending_intent_age_s=.2)
    lab._intents.replace(dict(command_id=1, server_received_ns=1))
    received_ns = 1_000_000_000
    lab._intents.replace(dict(command_id=2, server_received_ns=received_ns))
    lab._active_update = SimpleNamespace(intent_revision=2, timing_ns=dict(provider_total=80_000_000))
    if expired:
        with pytest.raises(RuntimeError, match="pending-intent age"):
            lab._check_mb_input_deadline(received_ns+elapsed_ns)
    else:
        lab._check_mb_input_deadline(received_ns+elapsed_ns)
    assert lab._intents.pending_since_ns(1) == received_ns


def test_pending_deadline_catches_policy_compute_crossing_the_limit():
    lab = object.__new__(SonicLabRuntime)
    lab._intents, lab._consumed_intent_revision = IntentMailbox(), 0
    lab.settings = dict(max_pending_intent_age_s=.2)
    lab._intents.replace(dict(command_id=1, server_received_ns=1_000_000_000))
    lab._check_mb_input_deadline(1_195_000_000)
    with pytest.raises(RuntimeError, match="pending-intent age"):
        lab._check_mb_input_deadline(1_205_000_000)


def _onset_measurement(time_s, velocity_xy, yaw=0.):
    # Analytic observer inputs only; these are not physical-run evidence.
    qpos = np.zeros(7)
    qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    return SimpleNamespace(time_s=time_s, qpos=qpos, qvel=np.array([*velocity_xy, 0, 0, 0, 0]),
                           captured_at_monotonic_ns=round(time_s*1e9)+1)


def test_body_onset_requires_consumption_direction_and_sustained_actual_change():
    observer = BodyOnsetObserver(linear_threshold_mps=.05, heading_threshold_rad=.02,
                                 sustained_samples=3, sample_hz=200)
    assert observer.observe(_onset_measurement(.1, [1, 0])) is None
    command = dict(session_id="s", command_id=1, velocity_xy_mps=[1, 0], yaw_rate_radps=0)
    observer.consume(command, _onset_measurement(.1, [0, 0]), desired_heading_rad=0,
                     consumed_at_monotonic_ns=100)
    assert observer.observe(_onset_measurement(.105, [-.2, 0])) is None
    assert observer.observe(_onset_measurement(.11, [.06, 0])) is None
    assert observer.observe(_onset_measurement(.115, [.01, 0])) is None
    assert observer.observe(_onset_measurement(.12, [.06, 0])) is None
    assert observer.observe(_onset_measurement(.125, [.07, 0])) is None
    onset = observer.observe(_onset_measurement(.13, [.08, 0]))
    assert onset["sim_time_s"] == .12
    assert onset["command_id"] == 1
    assert onset["sample_hz"] == 200
    assert onset["sustained_span_s"] == pytest.approx(.01)
    assert onset["detected_at_monotonic_ns"] > onset["sample_monotonic_ns"]
    observer.consume(dict(command, command_id=2), _onset_measurement(.14, [.08, 0]),
                     desired_heading_rad=0, consumed_at_monotonic_ns=200)
    assert observer.onset is None


def test_body_onset_stationary_identical_target_remains_unknown_and_turn_wraps():
    observer = BodyOnsetObserver(linear_threshold_mps=.05, heading_threshold_rad=.02,
                                 sustained_samples=2, sample_hz=200)
    command = dict(session_id="s", command_id=1, velocity_xy_mps=[0, 0], yaw_rate_radps=0)
    observer.consume(command, _onset_measurement(.1, [0, 0]), desired_heading_rad=0,
                     consumed_at_monotonic_ns=100)
    assert observer.observe(_onset_measurement(.2, [.2, .1], yaw=.2)) is None
    observer.consume(dict(command, command_id=2, yaw_rate_radps=1),
                     _onset_measurement(.3, [0, 0], yaw=np.pi-.1),
                     desired_heading_rad=-np.pi+.1, consumed_at_monotonic_ns=200)
    assert observer.observe(_onset_measurement(.305, [0, 0], yaw=np.pi-.05)) is None
    onset = observer.observe(_onset_measurement(.31, [0, 0], yaw=-np.pi+.01))
    assert onset["directed_heading_change_rad"] > 0


@pytest.mark.parametrize("yaw_rate", [-1., 1.])
@pytest.mark.parametrize("first_target_delta", [0., .001, -.001])
def test_body_onset_pure_yaw_uses_raw_rate_sign_and_fixed_threshold(yaw_rate, first_target_delta):
    observer = BodyOnsetObserver(linear_threshold_mps=.05, heading_threshold_rad=.02,
                                 sustained_samples=12, sample_hz=200)
    command = dict(session_id="pure_yaw", command_id=1, velocity_xy_mps=[0., 0.], yaw_rate_radps=yaw_rate)
    observer.consume(command, _onset_measurement(.1, [0., 0.], yaw=.3),
                     desired_heading_rad=.3+first_target_delta, consumed_at_monotonic_ns=100)
    assert observer.observe(_onset_measurement(.105, [0., 0.], yaw=.3-.03*yaw_rate)) is None
    assert observer.observe(_onset_measurement(.11, [0., 0.], yaw=.3+.01*yaw_rate)) is None
    for index in range(11):
        assert observer.observe(_onset_measurement(.115+index*.005, [0., 0.], yaw=.3+.03*yaw_rate)) is None
    onset = observer.observe(_onset_measurement(.17, [0., 0.], yaw=.3+.03*yaw_rate))
    assert onset["sim_time_s"] == pytest.approx(.115)
    assert onset["sustained_samples"] == 12
    assert onset["sample_hz"] == 200
    assert onset["sustained_span_s"] == pytest.approx(.055)
    assert onset["heading_threshold_rad"] == .02
    assert onset["directed_heading_change_rad"] == pytest.approx(.03)
    assert onset["requested_yaw_rate_radps"] == yaw_rate


def test_yaw_intent_integrates_native_rate_without_auto_facing_velocity():
    integrator = YawIntentIntegrator(.3)
    command = dict(command_id=1, velocity_xy_mps=[-1, 2], yaw_rate_radps=0)
    original = json.loads(json.dumps(command))
    assert integrator.for_tick(command, .18) == pytest.approx(.3)
    assert command == original
    pure_turn = dict(command, velocity_xy_mps=[0, 0], yaw_rate_radps=1.)
    for index in range(50):
        integrator.for_tick(pure_turn, .2+index*.02)
    assert integrator.desired_heading_rad == pytest.approx(1.3)
    assert integrator.tick == 51


def test_yaw_same_tick_newest_rate_replaces_without_double_integration():
    integrator = YawIntentIntegrator(0.)
    command = dict(command_seq=1, yaw_rate_radps=1.)
    assert integrator.for_tick(command, .18) == pytest.approx(.02)
    assert integrator.for_tick(dict(command, command_seq=2, yaw_rate_radps=2.), .18) == pytest.approx(.04)
    assert integrator.tick == 1
    assert integrator.for_tick(command, .2) == pytest.approx(.06)
    with pytest.raises(ValueError, match="native 50Hz"):
        integrator.for_tick(command, .205)


def test_reference_packet_uses_only_covered_native_grid_and_keeps_snapshot_identity():
    timestamps = .22+np.arange(81)/50
    packet = ReferencePacket(1, "unit_test_rig", "robot_hash", "scene_hash", tuple(),
        dict(length="m", time="s", angle="rad"), "mujoco_world_z_up", .205, 7,
        "plan_7_41", 41, timestamps, np.zeros((81, 7)), np.zeros((81, 29)),
        np.zeros((81, 29)), float(timestamps[-1]), "B1",
        np.array([.1, .133333333, .166666667, .2]), np.array([1., 2., .8]), np.array([1., 1.01, -2.]),
        "soma_world_y_up", -.21, dict(), 1, "unit_session")
    receipt = packet.use_receipt(.24, policy_action_ready_ns=101, first_motor_control_ns=102)
    assert receipt["state_version"] == 41
    assert receipt["generated_from_command_seq"] == 7
    np.testing.assert_allclose(receipt["used_time_window"], .24+np.arange(10)*.1)
    assert receipt["state_age_sim_s"] == pytest.approx(.035)
    with pytest.raises(ValueError, match="real 50Hz"):
        packet.use_receipt(.245, policy_action_ready_ns=101, first_motor_control_ns=102)
    with pytest.raises(ValueError, match="actual future"):
        packet.use_receipt(1.0, policy_action_ready_ns=101, first_motor_control_ns=102)


def test_next_controller_grid_does_not_anchor_reference_to_mid_physics_step():
    assert next_controller_grid(.205) == pytest.approx(.22)
    assert next_controller_grid(.2+1e-12) == pytest.approx(.2)
    assert next_controller_grid(283.7400000002023+.02) == pytest.approx(283.76, abs=1e-10)
    time_s = .18
    for _ in range(250):
        assert next_controller_grid(time_s) == pytest.approx(time_s, abs=1e-9)
        time_s += .02


def test_physics_timestamp_uses_exact_step_index_without_changing_native_time():
    physics = object.__new__(SonicPhysicsRuntime)
    physics.physics_dt = .001
    physics.data = SimpleNamespace(time=599.9999999927238)
    assert physics.time_s == 600.
    assert physics.data.time == 599.9999999927238
    assert next_controller_grid(physics.time_s+.02) == pytest.approx(600.02, abs=1e-10)


def test_reference_calibration_robot_hash_is_checked_before_inference():
    runtime = object.__new__(SonicPhysicsRuntime)
    runtime.initialized, runtime.terminal_reason = True, None
    runtime.scene_id, runtime.robot_model_sha256 = "scene", "actual_robot"
    reference = SimpleNamespace(scene_id="scene", provenance=dict(
        kind="calibrated_soma_reference", calibration=dict(robot_xml_sha256="different_robot")))
    with pytest.raises(ValueError, match="actual physics robot"):
        runtime.begin_control(reference)


def _pure_mb_worker_slot():
    # Scheduler-only fixtures; no model, GPU, simulator or service is constructed.
    intents, rows = IntentMailbox(), []
    revision = intents.replace(dict(command_id=2, command_seq=2, server_received_ns=1))
    worker = OnlineMBWorker(SimpleNamespace(request_evidence=None, failure_evidence=None), intents, lambda: None,
        SimpleNamespace(write=rows.append), numeric_writer=SimpleNamespace(check=lambda: None),
        simulation_time=lambda: .24, replan_interval_s=.1,
        max_state_age_s=.1, max_plan_wall_age_s=.1)
    clocks = dict(requested=perf_counter_ns(), reference_ready=perf_counter_ns())
    packet = ReferencePacket(1, "unit", "robot", "scene", tuple(), {}, "mujoco_world_z_up",
        .205, 2, "test_plan", 41, np.array([.22, 1.22]), np.zeros((2, 7)), np.zeros((2, 29)),
        np.zeros((2, 29)), 1.22, "B1", np.arange(4)/30, np.zeros(3), np.array([0., .21, 0.]),
        "soma_world_y_up", -.21, clocks, 1, "unit_session")
    update = MBReferenceUpdate(intent_revision=revision, command=dict(command_id=2, command_seq=2, server_received_ns=1),
        history_time_s=.2, derived_heading_rad=0., conditioning_history=None,
        estimated_actual_history=None, proposal=SimpleNamespace(source=None, raw_blocks=(), timing_ns={}),
        adapted_reference=SimpleNamespace(timestamps_s=np.array([.22, 1.22]), latest_window_start_s=.32),
        packet=packet, clocks_ns=clocks, timing_ns=dict(provider_total=1000))
    worker._published_ns = perf_counter_ns()
    return worker, update, rows


def test_online_mb_slot_accepts_fresh_completed_plan_when_newer_input_is_pending():
    worker, update, rows = _pure_mb_worker_slot()
    worker.intents.replace(dict(command_id=3, command_seq=3, server_received_ns=2))
    worker._ready = update
    assert worker.take_ready(.24).command["command_seq"] == 2
    assert rows == []
    assert worker._ready is None
    assert worker.intents.latest()[1]["command_seq"] == 3
    assert worker.intents.pending_since_ns(update.intent_revision) == 2


def test_online_short_tracker_coverage_does_not_use_the_sonic_point9_window():
    worker, update, rows = _pure_mb_worker_slot()
    times = np.array([.22, .82])
    update = replace(update, packet=replace(update.packet, future_timestamps=times, valid_until=.82),
        adapted_reference=SimpleNamespace(timestamps_s=times, latest_window_start_s=-.08))
    worker._ready = update
    assert worker.take_ready(.24) is None
    assert rows[-1]["reason"] == "trusted_future_coverage_expired"
    worker.required_future_s = .16
    worker._ready = update
    assert worker.take_ready(.24).packet.plan_id == update.packet.plan_id


def test_online_mb_keeps_a_fresh_future_result_until_its_real_control_grid():
    worker, update, rows = _pure_mb_worker_slot()
    times = .28+np.array([0., 1.])
    worker._ready = replace(update, packet=replace(update.packet, issued_from_sim_time=.2,
        future_timestamps=times, valid_until=times[-1]),
        adapted_reference=SimpleNamespace(timestamps_s=times, latest_window_start_s=.38))
    worker.request()
    original = worker._ready
    assert worker.take_ready(.26) is None
    assert worker._ready is original and rows == []
    assert not worker.control_tick(.26, intent=worker.intents.latest())
    assert worker.take_ready(.28).packet.plan_id == update.packet.plan_id
    assert worker._ready is None


def test_online_mb_future_result_still_expires_at_original_wall_bound(monkeypatch):
    import motionbricks.mb_sonic.runtime as module
    worker, update, rows = _pure_mb_worker_slot()
    requested = 1_000_000_000
    update.clocks_ns["requested"] = requested
    monkeypatch.setattr(module, "perf_counter_ns", lambda: requested+100_000_001)
    worker._ready = replace(update,
        adapted_reference=SimpleNamespace(timestamps_s=np.array([.28, 1.28]), latest_window_start_s=.38))
    assert worker.take_ready(.26) is None
    assert worker._ready is None
    assert rows[-1]["reason"] == "maximum_plan_wall_age_exceeded"


def test_explicit_thread_schedule_is_process_local_and_does_not_change_control_period(monkeypatch):
    import motionbricks.mb_sonic.runtime as module
    intervals = []
    monkeypatch.setattr(module.sys, "setswitchinterval", intervals.append)
    monkeypatch.setattr(module.sys, "getswitchinterval", lambda: .001)
    assert module.configure_thread_schedule(dict(python_thread_switch_interval_s=.001)) == .001
    assert intervals == [.001]
    module.configure_thread_schedule({})
    assert intervals == [.001]
    assert module.POLICY_DT == .02


@pytest.mark.parametrize("interval", [0., -.001, .01, float("nan")])
def test_invalid_thread_schedule_is_not_applied(monkeypatch, interval):
    import motionbricks.mb_sonic.runtime as module
    monkeypatch.setattr(module.sys, "setswitchinterval", lambda value: pytest.fail("Invalid thread policy applied"))
    with pytest.raises(ValueError, match="thread interval"):
        module.configure_thread_schedule(dict(python_thread_switch_interval_s=interval))


@pytest.mark.parametrize("initial,expected_start", [(True, .2), (False, .28)])
def test_constrained_provider_uses_native_grid_lead_without_clock_rpc(initial, expected_start):
    _, old_update, _ = _pure_mb_worker_slot()
    calls = []
    source = SimpleNamespace(trusted_until_s=2.)
    proposal = SimpleNamespace(source=source)

    def retarget(source, *, start_time_s, **kwargs):
        calls.append((start_time_s, kwargs))
        return SimpleNamespace(timestamps_s=start_time_s+np.array([0., 1.3]),
            qpos=np.zeros((2, 36)), joint_pos=np.zeros((2, 29)), joint_vel=np.zeros((2, 29)),
            trusted_until_s=start_time_s+1.3)

    provider = FrozenMBProvider(SimpleNamespace(generate=lambda *args, **kwargs: proposal),
        SimpleNamespace(root_z_offset_m=-.21, retarget_packet=retarget),
        conditioning_mode="B1", estimate_actual_diagnostics=False, robot_model_sha256="robot",
        scene_manifest_sha256="scene", rig_id="unit", reference_history_builder=None,
        consumer_horizon_s=1.3, constraint_solver=SimpleNamespace(
            identity=dict(schema='motionbricks.mb_sonic.constrained_reference.v1')),
        maximum_adoption_age_s=.1, planning_lead_s=.08)
    provider.preparation_report = {}
    history = SimpleNamespace(provenance=dict(kind="source_seed"), timestamps_s=.2-np.arange(3, -1, -1)/30)
    snapshot = MBGenerationSnapshot(.2, .2, 40, 1, np.zeros(3), 0., history, None, 1, {},
        previous_reference_packet=None if initial else old_update.packet)
    command = dict(command_seq=2, command_id=2, episode_id=1, session_id="unit_session",
                   velocity_xy_mps=[1., 0.], yaw_rate_radps=0.)
    update = provider.propose(snapshot, command=command, intent_revision=1, requested_at_ns=1,
        current_sim_time=lambda: pytest.fail("Fixed-grid reference planning must not wait on a clock RPC"))
    assert update.packet.future_timestamps[0] == expected_start
    assert calls[0][0] == expected_start
    boundary = calls[0][1]["boundary"]
    if initial:
        assert boundary is None
    else:
        assert boundary["plan_id"] == old_update.packet.plan_id
        assert boundary["adoption_until_s"] == pytest.approx(.3)


def test_yaw_rate_sets_future_heading_goal_without_rotating_translation():
    calls = []
    proposal = SimpleNamespace(source=SimpleNamespace(trusted_until_s=2.))
    def generate(*args, **kwargs):
        calls.append(kwargs)
        return proposal
    def retarget(source, **kwargs):
        return SimpleNamespace(timestamps_s=np.array([.2, 1.5]), qpos=np.zeros((2, 36)),
            joint_pos=np.zeros((2, 29)), joint_vel=np.zeros((2, 29)), trusted_until_s=1.5)
    provider = FrozenMBProvider(SimpleNamespace(generate=generate),
        SimpleNamespace(root_z_offset_m=-.21, retarget_packet=retarget),
        conditioning_mode='B1', estimate_actual_diagnostics=False, robot_model_sha256='robot',
        scene_manifest_sha256='scene', rig_id='unit', reference_history_builder=None,
        consumer_horizon_s=1.3, yaw_goal_lead_s=28/30)
    provider.preparation_report = {}
    history = SimpleNamespace(provenance=dict(kind='source_seed'), timestamps_s=np.arange(4)/30+.1)
    snapshot = MBGenerationSnapshot(.2, .2, 40, 1, np.zeros(3), -np.pi/2, history, None, 1, {})
    command = dict(command_seq=1, command_id=1, episode_id=1, session_id='unit',
                   velocity_xy_mps=[.7, -.2], yaw_rate_radps=1.5)
    provider.propose(snapshot, command=command, intent_revision=1, requested_at_ns=1, current_sim_time=lambda: .2)
    np.testing.assert_allclose(calls[0]['velocity_world_xz_mps'], [.7, .2], atol=1e-12)
    assert calls[0]['heading_world_rad'] == pytest.approx(1.4)


@pytest.mark.parametrize('velocity', [[.7, 0.], [0., 0.]])
def test_stationary_and_moving_feedback_use_the_same_fixed_observer(velocity):
    captured = []
    history = SimpleNamespace(provenance=dict(kind='actual_g1_error_state'), timestamps_s=np.arange(4)/30+.1)
    def observe(*args, **kwargs):
        captured.append(kwargs)
        return history
    proposal = SimpleNamespace(source=SimpleNamespace(trusted_until_s=2.))
    def retarget(source, **kwargs):
        return SimpleNamespace(timestamps_s=np.array([.2, 1.5]), qpos=np.zeros((2, 36)),
            joint_pos=np.zeros((2, 29)), joint_vel=np.zeros((2, 29)), trusted_until_s=1.5)
    provider = FrozenMBProvider(SimpleNamespace(generate=lambda *a, **kw: proposal),
        SimpleNamespace(root_z_offset_m=-.21, retarget_packet=retarget),
        conditioning_mode='B2', estimate_actual_diagnostics=False, robot_model_sha256='robot',
        scene_manifest_sha256='scene', rig_id='unit', reference_history_builder=None,
        consumer_horizon_s=1.3)
    provider.preparation_report = {}
    provider._history_observer = object()
    provider._state_observer = SimpleNamespace(history=observe)
    snapshot = MBGenerationSnapshot(.2, .2, 40, 1, np.zeros(3), 0., history, object(), 1, {})
    command = dict(command_seq=1, command_id=1, episode_id=1, session_id='unit',
                   velocity_xy_mps=velocity, yaw_rate_radps=1.5)
    provider.propose(snapshot, command=command, intent_revision=1, requested_at_ns=1, current_sim_time=lambda: .2)
    assert set(captured[0]) == {'end_time_s', 'observer'}
    assert captured[0]['end_time_s'] == .2


def test_constant_yaw_rate_integrates_at50hz_without_restarting_pose_each_tick():
    lab = object.__new__(SonicLabRuntime)
    lab.settings = dict(replan_on_integrated_heading=False)
    command = dict(yaw_rate_radps=1.5)
    latest = [1, command]
    lab._intents = SimpleNamespace(latest=lambda: tuple(latest))
    lab._lock = Lock()
    lab._yaw = YawIntentIntegrator(0.)
    lab._integrated_intent_revision = 1
    requested = []
    lab._mb_worker = SimpleNamespace(request=lambda: requested.append(True))
    lab.physics = SimpleNamespace(time_s=0.)
    for i in range(5):
        lab.physics.time_s = i*.02
        lab._integrate_mb_intent()
    assert lab._yaw.desired_heading_rad == pytest.approx(.15)
    assert not requested
    latest[0] = 2
    command['yaw_rate_radps'] = -.7
    lab.physics.time_s = .1
    lab._integrate_mb_intent()
    assert requested == [True]


def test_phase_cadence_never_defers_a_changed_translation_or_yaw_command():
    from motionbricks.mb_sonic.runtime import reference_adoption_due
    active = SimpleNamespace(command=dict(session_id='unit', command_seq=1), packet=SimpleNamespace(issued_from_sim_time=0.))
    same = SimpleNamespace(command=dict(session_id='unit', command_seq=1))
    changed = SimpleNamespace(command=dict(session_id='unit', command_seq=2))
    assert not reference_adoption_due(same, active, .08, .26)
    assert reference_adoption_due(changed, active, .08, .26)
    assert reference_adoption_due(same, active, .26, .26)
    assert reference_adoption_due(same, active, .08, 0.)
    active.packet.issued_from_sim_time = 283.7400000002023
    assert 284.24-283.82 < .48
    assert reference_adoption_due(same, active, 284.24, .48)
    contract = consumer_horizon_contract(dict(tracker_backend='protomotions', consumer_horizon_s=.74,
        mb_replan_interval_s=.06, mb_adopt_interval_s=.26, max_plan_wall_age_s=.1))
    assert contract['minimum_required_horizon_s'] == pytest.approx(.54)


def test_online_mb_does_not_dispatch_over_unconsumed_ready_result():
    worker, update, _ = _pure_mb_worker_slot()
    assert worker.take_ready(.24) is None
    worker._ready = update
    worker.request()
    assert not worker.control_tick(.24, intent=worker.intents.latest())
    assert worker._ready is update and worker._pending


@pytest.mark.parametrize("elapsed_ns,accepted", [(80_000_000, True), (99_000_000, True),
                                                (100_000_000, True), (100_000_001, False)])
def test_online_mb_slow_fresh_plan_preserves_exact_wall_deadline(monkeypatch, elapsed_ns, accepted):
    import motionbricks.mb_sonic.runtime as module
    worker, update, rows = _pure_mb_worker_slot()
    requested = 1_000_000_000
    clocks = dict(update.clocks_ns, requested=requested, reference_ready=requested+elapsed_ns)
    worker._published_ns = requested+elapsed_ns
    worker._ready = replace(update, clocks_ns=clocks,
        packet=replace(update.packet, issued_from_sim_time=.2, clocks_ns=clocks))
    monkeypatch.setattr(module, "perf_counter_ns", lambda: requested+elapsed_ns)
    assert (worker.take_ready(.3) is not None) is accepted
    assert not rows if accepted else rows[-1]["reason"] == "maximum_plan_wall_age_exceeded"


def test_online_mb_slot_checks_snapshot_age_wall_age_and_real_future_coverage():
    worker, update, rows = _pure_mb_worker_slot()
    worker._ready = update
    assert worker.take_ready(.24).packet.plan_id == update.packet.plan_id
    assert worker.take_ready(.24) is None
    worker._ready = update
    assert worker.take_ready(.31) is None
    assert rows[-1]["reason"] == "maximum_generation_snapshot_age_exceeded"
    update = replace(update, packet=replace(update.packet, issued_from_sim_time=.24))
    update.clocks_ns["requested"] = perf_counter_ns()-1_000_000_000
    worker._ready = update
    assert worker.take_ready(.24) is None
    assert rows[-1]["reason"] == "maximum_plan_wall_age_exceeded"
    update.clocks_ns["requested"] = perf_counter_ns()
    update = replace(update, packet=replace(update.packet, issued_from_sim_time=.3))
    worker._ready = update
    assert worker.take_ready(.34) is None
    assert rows[-1]["reason"] == "trusted_future_coverage_expired"


def test_online_mb_request_cap_rejects_period_faster_than_native_controller():
    with pytest.raises(ValueError, match="50Hz request cap"):
        OnlineMBWorker(None, IntentMailbox(), lambda: None, None, simulation_time=lambda: 0., replan_interval_s=.01,
                       numeric_writer=None,
                       max_state_age_s=.1, max_plan_wall_age_s=.1)


@pytest.mark.parametrize("offset_ns,accepted", [(-1, True), (0, True), (1, False)])
def test_online_mb_actual_v2_snapshot_boundary_is_exact_nanoseconds(offset_ns, accepted):
    worker, update, rows = _pure_mb_worker_slot()
    times = .4+np.arange(57)/50
    packet = replace(update.packet, issued_from_sim_time=.3200000000000002,
                     state_version=64, future_timestamps=times, valid_until=1.52)
    reference = SimpleNamespace(timestamps_s=times, latest_window_start_s=.62)
    worker._ready = replace(update, packet=packet, adapted_reference=reference)
    current = .42000000000000026+offset_ns/1e9
    if offset_ns == 0:
        assert current-packet.issued_from_sim_time > .1
        assert round(current*1e9)-round(packet.issued_from_sim_time*1e9) == 100_000_000
    result = worker.take_ready(current)
    assert (result is not None) is accepted
    if accepted:
        assert rows == []
    else:
        assert rows[-1]["reason"] == "maximum_generation_snapshot_age_exceeded"


def test_online_mb_publishes_before_numeric_snapshot_or_recording():
    worker, update, rows = _pure_mb_worker_slot()
    worker.history_snapshot = lambda: SimpleNamespace(integrated_intent_revision=update.intent_revision,
        history_time_s=.2, state_version=41, captured_at_monotonic_ns=100, timing_ns=dict(reference_history=10))
    worker.provider.propose = lambda *args, **kwargs: update

    def submit(record_id, record):
        assert worker._ready.packet.plan_id == update.packet.plan_id
        assert worker._published_ns >= update.clocks_ns["reference_ready"]
        assert record["clocks_ns"]["reference_published"] == worker._published_ns
        assert record["packet"].clocks_ns["reference_published"] == worker._published_ns
        assert record["timing_ns"]["history_snapshot"] > 0
        assert record["timing_ns"]["history_snapshot_stages"]["reference_history"] == 10
        assert record["timing_ns"]["provider_total"] == (1000+record["timing_ns"]["history_snapshot"]
                                                         +record["timing_ns"]["worker_dispatch_wait"])
        worker._stop.set()

    worker.numeric_writer.submit = submit
    worker.request()
    assert worker.control_tick(.2, intent=worker.intents.latest())
    worker._run()
    assert worker._failure is None
    assert rows == []


def test_online_mb_continuous_input_publishes_each_real_plan_and_queues_latest():
    worker, update, rows = _pure_mb_worker_slot()
    worker.history_snapshot = lambda: SimpleNamespace(integrated_intent_revision=update.intent_revision,
        history_time_s=.2, state_version=41, captured_at_monotonic_ns=100, timing_ns={})
    records = []

    def propose(*args, **kwargs):
        for sequence in range(3, 9):
            worker.intents.replace(dict(command_id=sequence, command_seq=sequence, server_received_ns=sequence))
        return update

    worker.provider.propose = propose
    worker.numeric_writer.submit = lambda record_id, record: records.append(record)
    worker.request()
    assert worker.control_tick(.2, intent=worker.intents.latest())
    dispatch, worker._dispatch = worker._dispatch, None
    worker._wake.clear()
    worker._process_dispatch(dispatch)
    worker._busy = False
    assert worker._ready.command["command_seq"] == 2
    assert records[-1]["publication_status"] == "published"
    assert rows[-1]["kind"] == "mb_newer_intent_pending"
    assert rows[-1]["latest_intent_revision"] == 7
    assert worker.take_ready(.24).command["command_seq"] == 2
    worker.history_snapshot = lambda: SimpleNamespace(integrated_intent_revision=worker.intents.latest()[0])
    assert worker.control_tick(.24, intent=worker.intents.latest())
    assert worker._dispatch["command"]["command_seq"] == 8


def test_online_mb_obsolete_failure_keeps_newest_request_alive():
    worker, update, rows = _pure_mb_worker_slot()
    records, calls = [], []
    worker.history_snapshot = lambda: SimpleNamespace(
        integrated_intent_revision=worker.intents.latest()[0], history_time_s=.2,
        state_version=41, simulation_time_s=.205, captured_at_monotonic_ns=100, timing_ns={})

    def propose(*args, **kwargs):
        calls.append(kwargs["intent_revision"])
        if len(calls) == 1:
            worker.intents.replace(dict(command_id=3, command_seq=3, server_received_ns=2))
            raise ValueError("obsolete request failed")
        return replace(update, intent_revision=kwargs["intent_revision"], command=kwargs["command"],
                       packet=replace(update.packet, generated_from_command_seq=3, plan_id="newest_plan"))

    def submit(record_id, record):
        records.append(record)
        if record["kind"] == "mb_raw_and_adapted_reference":
            worker._stop.set()

    worker.provider.propose, worker.numeric_writer.submit = propose, submit
    worker.request()
    assert worker.control_tick(.2, intent=worker.intents.latest())
    dispatch, worker._dispatch = worker._dispatch, None
    worker._wake.clear()
    worker._process_dispatch(dispatch)
    worker._busy = False
    assert calls == [1] and worker._pending and not worker._wake.is_set()
    assert worker.control_tick(.22, intent=worker.intents.latest())
    worker._run()
    assert worker._failure is None
    assert calls == [1, 2]
    assert records[0]["kind"] == "obsolete_mb_request_failure"
    assert worker._ready.command["command_seq"] == 3
    assert rows[0]["reason"] == "obsolete_request_failed"


def test_online_mb_current_request_failure_remains_terminal():
    worker, update, rows = _pure_mb_worker_slot()
    worker.history_snapshot = lambda: SimpleNamespace(integrated_intent_revision=update.intent_revision)

    def propose(*args, **kwargs):
        raise ValueError("current window illegal")

    worker.provider.propose = propose
    worker.request()
    assert worker.control_tick(.2, intent=worker.intents.latest())
    worker._run()
    assert worker._failure == "ValueError: current window illegal"
    assert rows[0]["kind"] == "mb_provider_failure"


def _native_clock_worker():
    # Scheduling-only inputs; no model, physical rollout or GPU task.
    worker, update, rows = _pure_mb_worker_slot()
    cursor, captures, calls, records = dict(time=.2, version=40), [], [], []

    def capture():
        revision = worker.intents.latest()[0]
        history = MBGenerationSnapshot(cursor["time"], cursor["time"], cursor["version"], revision,
            np.array([cursor["time"], 0., .79]), 0., None, None, 123+cursor["version"], {})
        captures.append(history)
        return history

    def propose(history, **kwargs):
        calls.append((history, kwargs))
        clocks = dict(update.clocks_ns, requested=kwargs["requested_at_ns"], reference_ready=perf_counter_ns())
        times = history.simulation_time_s+np.array([0., 1.])
        return replace(update, command=kwargs["command"], intent_revision=kwargs["intent_revision"],
            adapted_reference=SimpleNamespace(timestamps_s=times, latest_window_start_s=times[0]+.1),
            packet=replace(update.packet, issued_from_sim_time=history.simulation_time_s,
                state_version=history.state_version, generated_from_command_seq=kwargs["command"]["command_seq"],
                future_timestamps=times, valid_until=times[-1],
                clocks_ns=clocks), clocks_ns=clocks)

    worker.history_snapshot, worker.provider.propose = capture, propose
    worker.numeric_writer.submit = lambda record_id, record: records.append(record)
    return worker, cursor, captures, calls, records


def _complete_native_dispatch(worker):
    dispatch, worker._dispatch = worker._dispatch, None
    worker._wake.clear()
    try:
        worker._process_dispatch(dispatch)
    finally:
        worker._busy = False


def test_native_clock_requests_only_mark_pending_and_capture_at_boundary():
    worker, cursor, captures, calls, records = _native_clock_worker()
    for _ in range(100):
        worker.request()
    assert worker._pending and not worker._wake.is_set() and captures == []
    assert worker.control_tick(.2, intent=worker.intents.latest())
    assert len(captures) == 1 and captures[0].simulation_time_s == .2
    cursor.update(time=.205, version=41)
    _complete_native_dispatch(worker)
    assert calls[0][0].simulation_time_s == .2 and calls[0][0].state_version == 40
    assert calls[0][0].captured_at_monotonic_ns == 163
    np.testing.assert_array_equal(calls[0][0].actual_root_xyz_m, [.2, 0., .79])
    assert records[0]["control_tick"] == 10
    assert records[0]["clocks_ns"]["snapshot_captured"] == 163
    assert records[0]["clocks_ns"]["worker_started"] >= records[0]["clocks_ns"]["snapshot_ready"]


def test_native_clock_uses_integrated_boundary_intent_despite_new_arrival():
    worker, update, _ = _pure_mb_worker_slot()
    integrated = worker.intents.latest()
    worker.intents.replace(dict(command_id=3, command_seq=3, server_received_ns=2))
    worker.history_snapshot = lambda: SimpleNamespace(integrated_intent_revision=integrated[0])
    worker.request()
    assert worker.control_tick(.2, intent=integrated)
    assert worker._dispatch["command"]["command_seq"] == 2
    assert worker._dispatch["history"].integrated_intent_revision == integrated[0]
    assert worker.intents.latest()[1]["command_seq"] == 3


def test_native_clock_continuous_100hz_input_keeps_50hz_control_supplied(monkeypatch):
    import motionbricks.mb_sonic.runtime as module
    worker, cursor, _, _, records = _native_clock_worker()
    worker.intents = IntentMailbox()
    now = [1_000_000_000]
    monkeypatch.setattr(module, "perf_counter_ns", lambda: now[0])
    complete_tick, applied_revision, adopted, pending_ages = None, 0, [], []
    worker.request()
    for tick in range(200):
        now[0] = 1_000_000_000+tick*20_000_000
        cursor.update(time=.2+tick*.02, version=40+tick*4)
        for offset in (-10_000_000, 0):
            sequence = tick*2+offset//10_000_000+2
            worker.intents.replace(dict(command_id=sequence, command_seq=sequence,
                server_received_ns=now[0]+offset))
            worker.request()
        if tick == complete_tick:
            _complete_native_dispatch(worker)
            complete_tick = None
        update = worker.take_ready(cursor["time"])
        if update is not None:
            applied_revision = update.intent_revision
            adopted.append((cursor["time"], applied_revision))
        pending = worker.intents.pending_since_ns(applied_revision)
        if adopted and pending is not None:
            pending_ages.append(now[0]-pending)
        if worker.control_tick(cursor["time"], intent=worker.intents.latest()):
            complete_tick = tick+3
    assert len(adopted) >= 60
    assert max(np.diff([time for time, _ in adopted])) < .081
    assert max(pending_ages) <= 100_000_000
    assert all(record["publication_status"] == "published" for record in records)


def test_native_clock_periodic_dispatch_is_five_ticks_without_wall_timer():
    worker, cursor, captures, calls, records = _native_clock_worker()
    assert worker.control_tick(.2, intent=worker.intents.latest())
    _complete_native_dispatch(worker)
    assert worker.take_ready(.2) is not None
    for tick in range(11, 15):
        cursor.update(time=tick*.02, version=tick*4)
        assert not worker.control_tick(cursor["time"], intent=worker.intents.latest())
    assert len(captures) == len(calls) == 1
    cursor.update(time=.3, version=60)
    assert worker.control_tick(.3, intent=worker.intents.latest())
    _complete_native_dispatch(worker)
    assert len(captures) == len(calls) == 2
    assert [record["control_tick"] for record in records] == [10, 15]


def test_native_clock_busy_coalesces_and_completion_requires_next_fresh_boundary():
    worker, cursor, captures, calls, records = _native_clock_worker()
    assert worker.control_tick(.2, intent=worker.intents.latest())
    for tick in range(11, 16):
        worker.intents.replace(dict(command_id=tick, command_seq=tick, server_received_ns=tick))
        worker.request()
        cursor.update(time=tick*.02, version=tick*4)
        assert not worker.control_tick(cursor["time"], intent=worker.intents.latest())
    assert len(captures) == 1 and worker._dispatch["history"] is captures[0]
    _complete_native_dispatch(worker)
    assert worker._pending and not worker._wake.is_set()
    assert worker.take_ready(.3).command["command_seq"] == 2
    assert len(calls) == 1
    cursor.update(time=.32, version=64)
    assert worker.control_tick(.32, intent=worker.intents.latest())
    _complete_native_dispatch(worker)
    assert len(calls) == 2 and calls[-1][0].simulation_time_s == .32
    assert calls[-1][1]["command"]["command_seq"] == 15
    assert worker._ready.command["command_seq"] == 15


def test_native_clock_same_tick_never_restarts_without_explicit_initial_wait():
    worker, cursor, captures, calls, records = _native_clock_worker()
    assert worker.control_tick(.2, intent=worker.intents.latest())
    _complete_native_dispatch(worker)
    assert worker.take_ready(.2) is not None
    worker.request()
    assert not worker.control_tick(.2, intent=worker.intents.latest())
    cursor.update(time=.22, version=44)
    assert worker.control_tick(.22, intent=worker.intents.latest())
    assert len(captures) == 2


def test_native_clock_held_initial_wait_keeps_original_state_and_50hz_cap(monkeypatch):
    import motionbricks.mb_sonic.runtime as module
    worker, cursor, captures, calls, records = _native_clock_worker()
    cursor.update(time=.18, version=36)
    now = [1_000_000_000]
    monkeypatch.setattr(module, "perf_counter_ns", lambda: now[0])
    assert worker.control_tick(.18, intent=worker.intents.latest(), initial_wait=True)
    _complete_native_dispatch(worker)
    assert worker.take_ready(.18) is not None
    worker.request()
    now[0] += 19_999_999
    assert not worker.control_tick(.18, intent=worker.intents.latest(), initial_wait=True)
    now[0] += 1
    assert worker.control_tick(.18, intent=worker.intents.latest(), initial_wait=True)
    assert [item.state_version for item in captures] == [36, 36]
    assert [item.captured_at_monotonic_ns for item in captures] == [159, 159]


def test_native_clock_rejects_off_grid_and_backwards_ticks_and_fractional_period():
    worker, cursor, captures, calls, records = _native_clock_worker()
    with pytest.raises(ValueError, match="boundary"):
        worker.control_tick(.205, intent=worker.intents.latest())
    assert worker.control_tick(.2, intent=worker.intents.latest())
    with pytest.raises(ValueError, match="backwards"):
        worker.control_tick(.18, intent=worker.intents.latest())
    with pytest.raises(ValueError, match="whole native"):
        OnlineMBWorker(None, None, None, None, numeric_writer=None, simulation_time=None,
            replan_interval_s=.03, max_state_age_s=.1, max_plan_wall_age_s=.1)


def test_native_clock_actual_worker_wakes_only_for_control_dispatch_and_stops():
    worker, cursor, captures, calls, records = _native_clock_worker()
    published = Event()

    def log(record_id, record):
        records.append(record)
        published.set()

    worker.numeric_writer.submit = log
    worker.start()
    try:
        worker.request()
        assert captures == [] and not worker._wake.is_set()
        assert worker.control_tick(.2, intent=worker.intents.latest())
        assert published.wait(2)
        assert len(calls) == len(captures) == 1
    finally:
        worker.stop()
    assert not worker._thread.is_alive() and not worker._busy


def test_numeric_log_owns_readonly_arrays_and_writes_numpy_not_array_json(tmp_path):
    rows = []
    writer = NumericArtifactWriter(tmp_path / "binary", SimpleNamespace(write=rows.append), pending_plans=2)
    values = np.arange(4096, dtype=np.float32).reshape(1024, 4)
    writer.submit("plan:1", dict(values=values, nested=dict(clock_ns=123)))
    queued = writer._queue.queue[0][1]["values"]
    assert not queued.flags.writeable
    assert not np.shares_memory(queued, values)
    values[:] = -1
    writer.start()
    writer.close()
    writer.check()
    assert len(rows) == 1
    with np.load(rows[0]["path"], allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        description = metadata["record"]["values"]
        assert description["shape"] == [1024, 4]
        np.testing.assert_array_equal(archive[description["array"]], np.arange(4096).reshape(1024, 4))
        assert "0.0,1.0,2.0" not in str(archive["metadata_json"])


def test_numeric_compression_preserves_float_payloads_and_reduces_repeated_data(tmp_path):
    rows = []
    writer = NumericArtifactWriter(tmp_path / "compressed", SimpleNamespace(write=rows.append), pending_plans=1)
    bits = np.array([0, 0x8000000000000000, 0x7ff8000000000042,
                     0x7ff0000000000000, 0xfff0000000000000], dtype=np.uint64)
    value = np.tile(bits, 4096).view(np.float64)
    writer._persist(("float_bits", dict(values=value), 0))
    with np.load(rows[0]["path"], allow_pickle=False) as archive:
        description = json.loads(str(archive["metadata_json"]))["record"]["values"]
        assert archive[description["array"]].tobytes() == value.tobytes()
    assert rows[0]["bytes"] < value.nbytes / 10


def test_numeric_spawn_compressor_roundtrips_owned_arrays_and_shuts_down(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from motionbricks.mb_sonic.numeric_compression import encode_npz

    rows = []
    values = np.arange(4096, dtype=np.float64).reshape(512, 8)
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as executor:
        executor.submit(encode_npz, {}).result(timeout=10)
        writer = NumericArtifactWriter(tmp_path / "spawn_numeric", SimpleNamespace(write=rows.append),
            pending_plans=2, compression_executor=executor)
        writer.submit("spawn:1", dict(values=values))
        writer.start()
        writer.close()
    with np.load(rows[0]["path"], allow_pickle=False) as archive:
        description = json.loads(str(archive["metadata_json"]))["record"]["values"]
        assert archive[description["array"]].tobytes() == values.tobytes()


def test_bounded_tick_log_absorbs_eight_ticks_during_persistence_pause(tmp_path):
    rows=[]
    writer=NumericArtifactWriter(tmp_path/'ticks',SimpleNamespace(write=rows.append),pending_plans=16)
    entered,release=Event(),Event()
    persist=writer._persist
    def blocked(item):
        if item[0]=='tick:0':
            entered.set()
            assert release.wait(3)
        persist(item)
    writer._persist=blocked
    writer.start()
    try:
        writer.submit('tick:0',dict(qpos=np.array([0.])))
        assert entered.wait(3)
        for index in range(1,9):
            writer.submit(f'tick:{index}',dict(qpos=np.array([float(index)])))
        writer.check()
    finally:
        release.set()
        writer.close()
    assert [r['record_id'] for r in rows]==[f'tick:{i}' for i in range(9)]
    assert len(list((tmp_path/'ticks').glob('*.npz')))==9


def test_numeric_log_backpressure_is_fatal_and_preserves_overflow_record(tmp_path):
    rows = []
    writer = NumericArtifactWriter(tmp_path / "binary", SimpleNamespace(write=rows.append), pending_plans=1)
    writer.submit("first", dict(value=np.arange(4)))
    with pytest.raises(NumericLogBackpressure, match="overflow_record_retained"):
        writer.submit("overflow", dict(value=np.arange(5)))
    assert writer._overflow[0] == "overflow"
    writer.start()
    with pytest.raises(RuntimeError, match="backpressure"):
        writer.close()
    assert {row["record_id"] for row in rows} == {"first", "overflow"}
    with pytest.raises(RuntimeError, match="backpressure"):
        writer.check()


def test_numeric_log_and_recording_existing_bytes_are_never_overwritten(tmp_path):
    rows = []
    recorder = SimpleNamespace(write=rows.append)
    directory = tmp_path / "exclusive_numeric"
    writer = NumericArtifactWriter(directory, recorder, pending_plans=2)
    writer._persist(("same_plan", dict(values=np.arange(8)), 0))
    artifact = next(directory.glob("*.npz"))
    original = artifact.read_bytes()
    with pytest.raises(FileExistsError):
        writer._persist(("same_plan", dict(values=np.arange(16)), 0))
    assert artifact.read_bytes() == original
    with pytest.raises(FileExistsError):
        NumericArtifactWriter(directory, recorder, pending_plans=2)
    assert artifact.read_bytes() == original
    recording = tmp_path / "exclusive.jsonl"
    with PhysicsRecorder(recording, dict(test="exclusive_output")):
        pass
    original = recording.read_bytes()
    with pytest.raises(FileExistsError):
        PhysicsRecorder(recording, dict(test="replacement"))
    assert recording.read_bytes() == original


def _fake_timer_api(calls, *, caps_result=0, begin_result=0, end_result=0, minimum_ms=1):
    def capabilities(pointer, size):
        calls.append(("caps", size))
        pointer._obj.wPeriodMin, pointer._obj.wPeriodMax = minimum_ms, 1000
        return caps_result

    def begin(period):
        calls.append(("begin", period))
        return begin_result

    def end(period):
        calls.append(("end", period))
        return end_result

    return SimpleNamespace(timeGetDevCaps=capabilities, timeBeginPeriod=begin, timeEndPeriod=end)


@pytest.mark.parametrize("body_raises", [False, True])
def test_windows_clock_pairs_same_period_on_normal_and_exception_exit(monkeypatch, body_raises):
    calls = []
    api = _fake_timer_api(calls)
    monkeypatch.setattr(windows_clock, "_load_winmm", lambda: (api, 26200))
    timer = windows_clock.WindowsTimerResolution(1)

    def episode():
        with timer:
            assert timer.metadata["status"] == "active"
            if body_raises:
                raise ValueError("episode failed")

    if body_raises:
        with pytest.raises(ValueError, match="episode failed"):
            episode()
    else:
        episode()
    assert calls == [("caps", 8), ("begin", 1), ("end", 1)]
    assert timer.metadata["status"] == "released"
    assert timer.metadata["end_result"] == 0


@pytest.mark.parametrize("kwargs,expected", [
    (dict(caps_result=257), [("caps", 8)]),
    (dict(minimum_ms=2), [("caps", 8)]),
    (dict(begin_result=257), [("caps", 8), ("begin", 1)]),
])
def test_windows_clock_acquisition_failure_never_falls_back_or_calls_end(monkeypatch, kwargs, expected):
    calls = []
    api = _fake_timer_api(calls, **kwargs)
    monkeypatch.setattr(windows_clock, "_load_winmm", lambda: (api, 26200))
    with pytest.raises(RuntimeError):
        with windows_clock.WindowsTimerResolution(1):
            pytest.fail("A failed timer request must not start the episode")
    assert calls == expected


def test_windows_clock_release_failure_is_explicit(monkeypatch):
    calls = []
    api = _fake_timer_api(calls, end_result=257)
    monkeypatch.setattr(windows_clock, "_load_winmm", lambda: (api, 26200))
    timer = windows_clock.WindowsTimerResolution(1)
    with pytest.raises(RuntimeError, match="timeEndPeriod"):
        with timer:
            pass
    assert calls == [("caps", 8), ("begin", 1), ("end", 1)]
    assert timer.metadata["status"] == "release_failed"


@pytest.mark.parametrize("period", [0, 2, 1.0, True])
def test_windows_clock_requires_explicit_one_millisecond(period):
    with pytest.raises(ValueError, match="explicit integer"):
        windows_clock.WindowsTimerResolution(period)


def test_windows_clock_rejects_globally_scoped_old_windows(monkeypatch):
    monkeypatch.setattr(windows_clock, "sys", SimpleNamespace(platform="win32",
        getwindowsversion=lambda: SimpleNamespace(major=10, build=19040)))
    with pytest.raises(RuntimeError, match="Process-scoped"):
        windows_clock._load_winmm()


@pytest.mark.parametrize("elapsed_ns,interrupted", [(15_000_000, False), (100_000, True)])
def test_windows_clock_wait_keeps_overshoot_in_scheduler_without_spinning(monkeypatch, elapsed_ns, interrupted):
    times = iter([100, 100+elapsed_ns])
    monkeypatch.setattr(windows_clock, "perf_counter_ns", lambda: next(times))
    calls = []
    event = SimpleNamespace(wait=lambda seconds: calls.append(seconds) or interrupted)
    outcome = windows_clock.wait_until(event, 1_000_100)
    assert calls == [.001]
    assert outcome.interrupted is interrupted
    assert outcome.requested_ns == 1_000_000
    assert outcome.elapsed_ns == elapsed_ns
    assert outcome.explicit_wait_ns == min(1_000_000, elapsed_ns)
    assert outcome.overshoot_ns == max(0, elapsed_ns-1_000_000)
    assert elapsed_ns-outcome.explicit_wait_ns == outcome.overshoot_ns


def test_windows_clock_past_deadline_does_not_sleep_or_busy_spin(monkeypatch):
    monkeypatch.setattr(windows_clock, "perf_counter_ns", lambda: 100)
    event = SimpleNamespace(is_set=lambda: True,
        wait=lambda seconds: pytest.fail("Past deadlines must not enter another wait"))
    assert windows_clock.wait_until(event, 99) == windows_clock.DeadlineWait(True, 0, 0)


@pytest.mark.parametrize("body_raises", [False, True])
def test_windows_clock_lab_lifecycle_is_balanced_without_running_physics(monkeypatch, body_raises):
    calls, rows, errors = [], [], []
    api = _fake_timer_api(calls)
    monkeypatch.setattr(windows_clock, "_load_winmm", lambda: (api, 26200))
    lab = object.__new__(SonicLabRuntime)
    lab._begin_event, lab._stop_event = Event(), Event()
    lab._begin_event.set()
    lab.settings = dict(clock=dict(strategy="windows_event_wait_1ms", timer_resolution_ms=1))
    lab._recorder = SimpleNamespace(write=rows.append)
    lab._record_runtime_failure = errors.append

    def paced():
        calls.append(("paced", 1))
        if body_raises:
            raise ValueError("paced unit failure")

    lab._run_paced = paced
    lab._run()
    assert calls == [("caps", 8), ("begin", 1), ("paced", 1), ("end", 1)]
    assert rows[0]["kind"] == "timer_resolution_acquired"
    assert rows[-1]["kind"] == "timer_resolution_lifecycle"
    assert rows[-1]["status"] == "released"
    assert len(errors) == int(body_raises)
    calls.clear()
    rows.clear()
    lab._stop_event.set()
    lab._run()
    assert calls == rows == []


def test_actual_root_query_origin_applies_only_declared_morphology_offset():
    actual = np.array([2., -3., .79])
    query = calibrated_native_query_origin(actual, -.21)
    np.testing.assert_allclose(query, [2., 1., 3.])
    np.testing.assert_array_equal(actual, [2., -3., .79])


def test_body_onset_does_not_invent_yaw_when_root_forward_is_vertical():
    observer = BodyOnsetObserver(linear_threshold_mps=.05, heading_threshold_rad=.02,
                                 sustained_samples=1, sample_hz=200)
    command = dict(session_id="s", command_id=1, velocity_xy_mps=[0, 0], yaw_rate_radps=1)
    observer.consume(command, _onset_measurement(.1, [0, 0]), desired_heading_rad=.5,
                     consumed_at_monotonic_ns=100)
    sample = _onset_measurement(.105, [0, 0])
    sample.qpos[3:7] = [np.sqrt(.5), 0, np.sqrt(.5), 0]
    assert observer.observe(sample) is None


def test_numeric_log_drain_failure_promotes_completed_lab_to_terminal():
    lab = object.__new__(SonicLabRuntime)
    rows, reasons = [], []
    lab._lock = Lock()
    lab._snapshot_subscription_lock, lab._snapshot_subscriber = Lock(), None
    lab._failure, lab._status, lab._errors = None, "completed_mb_legacy_reference_history_B1", []
    lab._latest = dict(frame_id=7, sim_time_s=.3, status=lab._status)
    lab.physics = SimpleNamespace(time_s=.3, terminate=reasons.append)
    lab._recorder = SimpleNamespace(write=rows.append)

    def failed_close():
        raise OSError("disk full during final numeric drain")

    lab._numeric_writer = SimpleNamespace(close=failed_close)
    lab._tick_writer = None
    lab._compression_executor = None
    notices = []
    unsubscribe = lab.subscribe_snapshots(lambda: notices.append(lab._failure))
    lab._close_numeric_logs()
    assert lab._status == "runtime_failure"
    assert lab._latest["frame_id"] == 7
    assert "disk full" in lab._failure
    assert reasons == [lab._failure]
    assert rows[0]["kind"] == "required_numeric_log_failure"
    assert notices == [None, lab._failure]
    unsubscribe()


def test_mb_snapshot_keeps_real_capture_clock_and_source_history_separate():
    from motionbricks.mb_sonic.reference import NativeSomaHistory

    lab = object.__new__(SonicLabRuntime)
    lab.settings, lab._active_update = {}, None
    lab._lock, lab._seed_clock_origin_s = Lock(), .18
    lab._mb_planner_backend = "thread"
    qpos = np.arange(36, dtype=float)
    lab._executed = SimpleNamespace(samples=[dict(timestamp_s=.2, qpos=qpos,
        captured_at_monotonic_ns=123, state_version=40)], body_names=("world", "pelvis"))
    lab.physics = SimpleNamespace(scene_id="unit", physics_dt=.005,
        history=SimpleNamespace(entries=[SimpleNamespace(time_s=.2)]))
    lab._reference_cache, lab._integrated_intent_revision = None, 5
    lab._yaw = SimpleNamespace(desired_heading_rad=.4)
    lab._seed_history = NativeSomaHistory((np.arange(4)-3)/30, np.zeros((4, 366)), "unit",
                                         dict(kind="source_seed"))
    snapshot = lab._mb_generation_snapshot()
    assert snapshot.captured_at_monotonic_ns == 123
    assert snapshot.state_version == 40
    assert snapshot.history_time_s == pytest.approx(.18)
    np.testing.assert_array_equal(snapshot.actual_root_xyz_m, qpos[:3])
    np.testing.assert_array_equal(snapshot.reference_history.features, 0)
    assert snapshot.reference_history.provenance["kind"] == "source_seed"
    assert set(snapshot.timing_ns) == {"lock_wait", "state_snapshot", "reference_history", "total"}
    assert snapshot.timing_ns["total"] == sum(snapshot.timing_ns[key]
        for key in ("lock_wait", "state_snapshot", "reference_history"))
    source, representation = object(), object()
    lab._reference_cache = source

    def build_history(received, *, end_time_s, representation):
        assert received is source and end_time_s == .2
        assert lab._lock.acquire(blocking=False)
        lab._lock.release()
        return replace(lab._seed_history, timestamps_s=(np.arange(4)-3)/30+end_time_s,
                       provenance=dict(kind="legacy_reference_history"))

    lab._provider = SimpleNamespace(reference_history_builder=build_history,
                                     retargeter=SimpleNamespace(rep=representation))
    snapshot = lab._mb_generation_snapshot()
    assert snapshot.reference_history.provenance["kind"] == "legacy_reference_history"
    assert snapshot.captured_at_monotonic_ns == 123
    assert snapshot.timing_ns["reference_history"] > 0
    lab._mb_planner_backend = "spawn"
    lab._provider.reference_history_builder = lambda *args, **kwargs: pytest.fail("Spawn parent must not build B1 history")
    snapshot = lab._mb_generation_snapshot()
    assert snapshot.reference_history is None and snapshot.reference_source is source
    assert snapshot.history_time_s == .2 and snapshot.simulation_time_s == .2
    assert snapshot.captured_at_monotonic_ns == 123 and snapshot.state_version == 40
    np.testing.assert_array_equal(snapshot.actual_root_xyz_m, qpos[:3])

    lab.settings = dict(mb_conditioning_mode="B2")
    snapshot = lab._mb_generation_snapshot()
    assert snapshot.history_time_s == .2 and snapshot.simulation_time_s == .2
    assert snapshot.reference_history is lab._seed_history
    assert snapshot.reference_history.timestamps_s[-1] == 0.
    assert snapshot.reference_source is None
    assert snapshot.executed_history.samples[-1]["state_version"] == 40


def _executed_state_input(time_s, version):
    # Analytic history-contract input, not physical-run evidence.
    return SimpleNamespace(time_s=time_s, captured_at_monotonic_ns=12345+version,
        state_version=version, qpos=np.arange(50.)+version, qvel=np.arange(49.)-version,
        body_positions_world=np.arange(135.).reshape(45, 3),
        body_quat_wxyz=np.tile([1., 0., 0., 0.], (45, 1)),
        contacts=(ContactForce((2, 4), (1, 3), np.arange(3.), np.eye(3), np.arange(6.)-2, -.01),))


def test_executed_history_reuses_measured_state_without_native_requeries(monkeypatch):
    from motionbricks.mb_sonic.reference import ExecutedHistory
    lab = object.__new__(SonicLabRuntime)
    lab.mode = "official_reference"
    lab.physics = SimpleNamespace(physics_dt=.005)
    body_names = tuple(f"body_{index}" for index in range(45))
    lab._executed = ExecutedHistory("unit", body_names=body_names)

    def forbidden(*args, **kwargs):
        pytest.fail("An already captured PhysicsState must not query MuJoCo again")

    monkeypatch.setattr(mujoco, "mj_contactForce", forbidden)
    monkeypatch.setattr(mujoco, "mj_id2name", forbidden)
    state = _executed_state_input(.005, 1)
    lab._append_executed(state)
    sample = lab._executed.samples[-1]
    assert lab._executed.body_names is body_names
    assert sample["timestamp_s"] == state.time_s
    assert sample["captured_at_monotonic_ns"] == state.captured_at_monotonic_ns
    assert sample["state_version"] == state.state_version
    for key, actual in (("qpos", state.qpos), ("qvel", state.qvel),
                        ("body_pos_w", state.body_positions_world),
                        ("body_quat_w", state.body_quat_wxyz)):
        np.testing.assert_array_equal(sample[key], actual)
        assert not sample[key].flags.writeable and not np.shares_memory(sample[key], actual)
    contact = sample["contacts"][0]
    assert contact["body_ids"] == state.contacts[0].body_ids
    for key, actual in (("position_world", state.contacts[0].position_world),
                        ("wrench_contact", state.contacts[0].wrench_contact)):
        np.testing.assert_array_equal(contact[key], actual)
        assert not contact[key].flags.writeable and not np.shares_memory(contact[key], actual)
    state.qpos[:] = -99
    state.contacts[0].wrench_contact[:] = 99
    np.testing.assert_array_equal(sample["qpos"], np.arange(50.)+1)
    np.testing.assert_array_equal(contact["wrench_contact"], np.arange(6.)-2)
    with pytest.raises(ValueError, match="must advance"):
        lab._append_executed(state)
    assert len(lab._executed.samples) == 1


def test_executed_history_keeps_every_200hz_sample_with_existing_bound():
    from motionbricks.mb_sonic.reference import ExecutedHistory
    lab = object.__new__(SonicLabRuntime)
    lab.mode = "official_reference"
    lab.physics, lab._executed = SimpleNamespace(physics_dt=.005), ExecutedHistory("unit")
    for version in range(1, 71):
        lab._append_executed(_executed_state_input(version*.005, version))
    assert len(lab._executed.samples) == 52
    assert [sample["state_version"] for sample in lab._executed.samples] == list(range(19, 71))
    np.testing.assert_allclose(np.diff([s["timestamp_s"] for s in lab._executed.samples]), .005)
    assert all(len(sample["contacts"]) == 1 for sample in lab._executed.samples)


def test_explicit_scene_reset_rearms_completed_episode_without_starting_physics():
    lab = object.__new__(SonicLabRuntime)
    calls = []
    lab._catalog, lab._lock = [dict(id="000", available=True)], Lock()
    lab._snapshot_subscription_lock, lab._snapshot_subscriber = Lock(), None
    lab._recording_budget = None
    lab._thread = SimpleNamespace(is_alive=lambda: False)
    lab.stop = lambda: calls.append("stop_completed")
    lab._load_episode = lambda scene_id: calls.append("load_"+scene_id)
    lab.start = lambda: calls.append("arm_start_gate")
    lab.config = lambda: dict(start_required=True, scene=dict(manifest_id="000"))
    config = lab.select_scene("000")
    assert calls == ["stop_completed", "load_000", "arm_start_gate"]
    assert config["start_required"] is True


def test_online_mb_mismatch_caps_attempts_without_dispatching_or_busy_retry():
    worker, update, rows = _pure_mb_worker_slot()

    def history_snapshot():
        worker._stop.set()
        return SimpleNamespace(integrated_intent_revision=999)

    worker.history_snapshot = history_snapshot
    worker.provider.propose = lambda *args, **kwargs: pytest.fail("A mismatched intent must not generate")
    worker.request()
    worker.control_tick(.2, intent=worker.intents.latest())
    worker._run()
    assert worker._last_attempt_ns is not None
    assert worker._last_request_ns is None
    assert worker._failure is None


def test_online_mb_obsolete_history_builder_failure_does_not_kill_newest():
    worker, update, rows = _pure_mb_worker_slot()
    snapshots, records = [], []

    def history_snapshot():
        snapshots.append(1)
        if len(snapshots) == 1:
            worker.intents.replace(dict(command_id=3, command_seq=3, server_received_ns=2))
            raise ValueError("obsolete history request")
        return SimpleNamespace(integrated_intent_revision=worker.intents.latest()[0],
            state_version=41, simulation_time_s=.205, history_time_s=.2, captured_at_monotonic_ns=100, timing_ns={})

    def propose(*args, **kwargs):
        return replace(update, command=kwargs["command"], intent_revision=kwargs["intent_revision"])

    def submit(record_id, record):
        records.append(record)
        if record["kind"] == "mb_raw_and_adapted_reference":
            worker._stop.set()

    worker.history_snapshot, worker.provider.propose = history_snapshot, propose
    worker.numeric_writer.submit = submit
    worker.request()
    assert worker.control_tick(.2, intent=worker.intents.latest())
    dispatch, worker._dispatch = worker._dispatch, None
    worker._wake.clear()
    worker._process_dispatch(dispatch)
    worker._busy = False
    assert worker._pending and worker._ready is None
    assert worker.control_tick(.22, intent=worker.intents.latest())
    worker._run()
    assert worker._failure is None
    assert records[0]["state_version"] is None
    assert records[0]["error"] == "ValueError:obsolete history request"
    assert worker._ready.command["command_seq"] == 3


def test_provider_matches_public_b1_and_uniform_consumer_api_signatures():
    import inspect
    from motionbricks.mb_sonic.mb_reference import TrainedMBReferenceAdapter, reference_history_b1
    from motionbricks.mb_sonic.retarget import CalibratedSomaG1Adapter

    for method in (TrainedMBReferenceAdapter.prepare_compiled, TrainedMBReferenceAdapter.generate):
        parameters = inspect.signature(method).parameters
        assert "conditioning_mode" in parameters
        assert "query_origin_soma_xyz" in parameters
        assert "actual_root_mujoco_xyz" not in parameters
    assert "representation" in inspect.signature(reference_history_b1).parameters
    assert "start_time_s" in inspect.signature(CalibratedSomaG1Adapter.retarget_packet).parameters
    assert "consumer_horizon_s" in inspect.signature(CalibratedSomaG1Adapter.retarget_packet).parameters


@pytest.mark.parametrize("execution", ["native", "full_plan"])
def test_provider_prepares_cpu_history_once_before_native_or_full_plan(monkeypatch, execution):
    from motionbricks.data import native_mb_data
    from motionbricks.mb_sonic import mb_reference
    from motionbricks.mb_sonic.retarget import CalibratedSomaG1Adapter

    # Admission/forwarding test only: no real model loader or GPU call is executed.
    settings = json.loads((ROOT / "motionbricks/configs/mb_sonic_b1_fullplan_v3.json").read_text())
    settings["mb_execution"] = execution
    calls, preparation_calls = {}, []
    representation = object()
    observer = SimpleNamespace(representation=representation,
        preparation=dict(execution="cpu_observer_unit_fixture", optimizer_steps=0))

    def prepare_cpu(rep, features):
        preparation_calls.append((rep, features))
        return observer

    def build_history(source, *, end_time_s, representation, observer):
        return (source, end_time_s, representation, observer)

    def prepare(history, **kwargs):
        calls["history"], calls["prepare"] = history, kwargs
        return dict(execution=kwargs["execution"], conditioning_mode=kwargs["conditioning_mode"])

    generator = SimpleNamespace(prepare_compiled=prepare)
    monkeypatch.setattr(native_mb_data, "build_native_representation", lambda: representation)
    monkeypatch.setattr(CalibratedSomaG1Adapter, "load", lambda *args, **kwargs:
                        SimpleNamespace(root_z_offset_m=-.21, rep=representation))
    monkeypatch.setattr(mb_reference.TrainedMBReferenceAdapter, "load", lambda *args, **kwargs: generator)
    monkeypatch.setattr(mb_reference, "prepare_reference_history_b1", prepare_cpu)
    monkeypatch.setattr(mb_reference, "reference_history_b1", build_history)
    scene = SimpleNamespace(manifest_path=Path(settings["scene_root"]) / "000/scene_manifest.json")
    provider = FrozenMBProvider.load(settings, canonical_scene=scene)
    history = SimpleNamespace(features=np.arange(4*366, dtype=np.float32).reshape(4, 366))
    report = provider.prepare(history, actual_root_xyz_m=[2., 3., .79], desired_heading_rad=.4)
    assert provider.execution == report["execution"] == execution
    assert provider.consumer_horizon_s == 1.12
    assert provider.estimate_actual_diagnostics is False
    assert len(preparation_calls) == 1
    assert preparation_calls[0][0] is representation
    assert preparation_calls[0][1] is history.features
    assert report["cpu_history_observer"] == observer.preparation
    assert report["cpu_history_observer"] is not observer.preparation
    assert report["cpu_history_observer_prepared_now"] is True
    json.dumps(report, allow_nan=False)
    source = object()
    assert provider.reference_history_builder(source, end_time_s=.2, representation=representation) == (
        source, .2, representation, observer)
    if execution == "full_plan":
        assert calls["history"] is history
        assert calls["prepare"]["conditioning_mode"] == "B1"
        np.testing.assert_allclose(calls["prepare"]["query_origin_soma_xyz"], [2., 1., -3.])
        assert calls["prepare"]["heading_world_rad"] == pytest.approx(.4+np.pi/2)
    else:
        assert calls == {}
    again = provider.prepare(history, actual_root_xyz_m=[2., 3., .79], desired_heading_rad=.4)
    assert len(preparation_calls) == 1
    assert again["cpu_history_observer_prepared_now"] is False
    if execution == "full_plan":
        def failed_neural_prepare(*args, **kwargs):
            raise RuntimeError("neural deployment preparation failed")

        generator.prepare_compiled = failed_neural_prepare
        provider._scene_preparations.clear()
        with pytest.raises(RuntimeError, match="neural deployment preparation failed"):
            provider.prepare(history, actual_root_xyz_m=[2., 3., .79], desired_heading_rad=.4)
        assert provider.preparation_report is None
        assert len(preparation_calls) == 1


def _wasd_settings():
    return json.loads((ROOT / "motionbricks/configs/mb_sonic_wasd_v1.json").read_text())


@pytest.mark.parametrize("scene_id", ["000", "001", "002", "003", "004", "005", "006", "007",
                                      "009", "010", "011", "012", "013", "014", "urban"])
def test_wasd_seed_is_exact_canonical_four_frames_without_root_reanchor(scene_id):
    cfg = _wasd_settings()
    path = Path(cfg["scene_root"])/scene_id/"scene_manifest.json"
    manifest = json.loads(path.read_text())
    scene = SimpleNamespace(root=path.parent, manifest_path=path,
                              manifest=SimpleNamespace(scene_id=scene_id, source=manifest["source"]))
    history = load_mb_seed_history(cfg, scene)
    with np.load(cfg["scene_seeds"][scene_id]["path"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(history.features, archive["initial_history"][0])
    np.testing.assert_allclose(history.timestamps_s, [-.1, -2/30, -1/30, 0.], atol=1e-15)
    assert history.scene_id == scene_id and history.provenance["spatial_transform"] == "none"
    assert history.provenance["sha256"] == manifest["source"]["scene"]["sha256"]
    assert history.provenance["scene_manifest_sha256"] == cfg["scene_manifest_sha256"][scene_id]
    assert not history.features.flags.writeable


def test_wasd_seed_never_substitutes_another_map_or_legacy_flat_seed():
    cfg = _wasd_settings()
    path = Path(cfg["scene_root"])/"001/scene_manifest.json"
    manifest = json.loads(path.read_text())
    scene = SimpleNamespace(root=path.parent, manifest_path=path,
        manifest=SimpleNamespace(scene_id="001", source=manifest["source"]))
    cfg["mb_seed_path"], cfg["mb_seed_sha256"] = cfg["scene_seeds"]["000"]["path"], cfg["scene_seeds"]["000"]["sha256"]
    cfg["scene_seeds"]["001"] = cfg["scene_seeds"]["000"]
    with pytest.raises(ValueError, match="canonical manifest"):
        load_mb_seed_history(cfg, scene)
    cfg["scene_seeds"].pop("001")
    with pytest.raises(KeyError):
        load_mb_seed_history(cfg, scene)


def _scoped_intent_lab():
    lab = object.__new__(SonicLabRuntime)
    rows = []
    lab.mode, lab._status = "mb", "awaiting_explicit_start_mb"
    lab._lock, lab._intents = Lock(), IntentMailbox()
    lab.physics = SimpleNamespace(scene_id="001", manifest_sha256="canonical")
    lab._episode, lab._controller_session_id, lab._last_intent_seq = 2, None, 0
    lab._recording_budget = None
    lab._recorder = SimpleNamespace(write=rows.append)
    lab._mb_worker = SimpleNamespace(request=lambda: None)
    command = dict(scene_id="001", scene_manifest_sha256="canonical", episode_id=2, session_id="human",
        command_id=1, command_seq=1, velocity_xy_mps=[0., 0.], yaw_rate_radps=0., server_received_ns=1)
    return lab, command, rows


@pytest.mark.parametrize("field,value", [("scene_id", "000"), ("scene_manifest_sha256", "old"),
                                        ("episode_id", 1), ("episode_id", "2")])
def test_wasd_intent_scope_rejects_old_scene_hash_or_episode_before_accepting(field, value):
    lab, command, rows = _scoped_intent_lab()
    with pytest.raises(ValueError, match="stale"):
        lab.submit_intent(dict(command, **{field: value}))
    assert lab._intents.latest() == (0, None) and rows == []
    assert lab._controller_session_id is None


def test_wasd_intent_session_and_sequence_are_independent_of_manual_start():
    lab, command, rows = _scoped_intent_lab()
    assert lab.submit_intent(command)["consumed"] is False
    assert lab._controller_session_id == "human"
    with pytest.raises(ValueError, match="strictly increasing"):
        lab.submit_intent(command)
    with pytest.raises(ValueError, match="manual Reset"):
        lab.submit_intent(dict(command, session_id="stale_socket", command_id=2, command_seq=2))
    with pytest.raises(ValueError, match="finite"):
        lab.submit_intent(dict(command, command_id=2, command_seq=2, yaw_rate_radps=np.nan))
    assert len(rows) == 1
    assert lab._intents.latest()[1]["velocity_xy_mps"] == [0., 0.]
    lab._status = "completed_interactive_run_budget"
    with pytest.raises(ValueError, match="episode ended"):
        lab.submit_intent(dict(command, command_id=2, command_seq=2))


def test_wasd_run_budget_is_explicit_finite_and_never_resets_physics(monkeypatch):
    import motionbricks.mb_sonic.runtime as module
    lab = object.__new__(SonicLabRuntime)
    rows = []
    lab.run_mode, lab._budget_finished = "interactive", False
    lab._wall_start_ns, lab._run_deadline_ns = 10, 300_000_000_010
    lab._lock, lab._episode = Lock(), 4
    lab._snapshot_subscription_lock, lab._snapshot_subscriber = Lock(), None
    lab._latest, lab._status = dict(actual=dict(root_position=[1., 2., 3.])), "running"
    lab.physics = SimpleNamespace(time_s=299.)
    lab.settings, lab._recorder = dict(max_run_wall_s=300), SimpleNamespace(write=rows.append)
    monkeypatch.setattr(module, "perf_counter_ns", lambda: 300_000_000_009)
    assert not lab._run_budget_exhausted()
    monkeypatch.setattr(module, "perf_counter_ns", lambda: 300_000_000_010)
    assert lab._run_budget_exhausted() and lab._run_budget_exhausted()
    assert lab._latest["actual"]["root_position"] == [1., 2., 3.]
    assert lab._status == "completed_interactive_run_budget"
    assert len(rows) == 1 and rows[0]["automatic_reset"] is False


def test_wasd_shared_storage_budget_reserves_drain_bytes_and_enforces_hard_cap():
    budget = RecordingBudget(1000, 200, 100)
    budget.reserve(800)
    budget.check()
    budget.reserve(1)
    with pytest.raises(RuntimeError, match="pending records are draining"):
        budget.check()
    budget.reserve(199)
    assert budget.used_bytes == 1000
    with pytest.raises(RuntimeError, match="hard byte cap"):
        budget.reserve(1)


def test_wasd_binary_tick_retains_every_state_array_with_small_json_receipt(tmp_path):
    # Analytic serialization inputs only, not physical-run evidence.
    budget = RecordingBudget(2**20, 2**17, 2**16)
    recorder = PhysicsRecorder(tmp_path/"record.jsonl", dict(test="binary_full_tick"), storage_budget=budget)
    writer = NumericArtifactWriter(tmp_path/"ticks", recorder, pending_plans=2,
                                    storage_budget=budget, max_record_bytes=2**18)
    recorder.tick_writer = writer
    states = []
    contact = ContactForce((1, 2), (0, 1), np.zeros(3), np.eye(3), np.arange(6.), -.001)
    for step in range(1, 5):
        states.append(PhysicsState(step*.005, step*5_000_000, step,
            np.arange(50.)+step, np.arange(49.), np.arange(29.), np.arange(29.),
            np.zeros((45, 3)), np.tile([1., 0., 0., 0.], (45, 1)),
            np.arange(43.), np.arange(43.), np.arange(49.), np.zeros(6), np.zeros(49),
            np.zeros((45, 6)), {"warning": dict(number=0)}, True, 0., (contact,), None))
    tick = RuntimeTick("001", 7, "unit_binary", .18, 1, dict(source="unit"),
                       np.zeros((10, 36)), dict(action=np.arange(29.)), tuple(states), dict(chain_total=100))
    recorder.tick(tick)
    states[-1].qpos[:] = -99
    writer.start()
    writer.close()
    recorder.close()
    rows = [json.loads(line) for line in (tmp_path/"record.jsonl").read_text().splitlines()]
    receipt = next(row for row in rows if row["kind"] == "policy_tick_binary")
    assert len(json.dumps(receipt)) < 1024
    assert not any(row["kind"] == "policy_tick" for row in rows)
    artifact = next(row for row in rows if row["kind"] == "numeric_artifact_written")
    with np.load(artifact["path"], allow_pickle=False) as archive:
        value = json.loads(str(archive["metadata_json"]))["record"]
        recorded = value["tick"]["physics_states"]
        assert len(recorded) == 4 and set(recorded[-1]) == set(PhysicsState.__dataclass_fields__)
        np.testing.assert_array_equal(archive[recorded[-1]["qpos"]["array"]], np.arange(50.)+4)
        np.testing.assert_array_equal(archive[recorded[-1]["contacts"][0]["wrench_contact"]["array"]], np.arange(6.))
    assert budget.used_bytes == sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())


def test_wasd_scene_query_proofs_and_catalog_match_all_fifteen_canonical_sources():
    from motionbricks.mb_sonic.runtime import _scene_query_options
    cfg = _wasd_settings()
    catalog = json.loads(Path(cfg["scene_catalog"]["path"]).read_text())
    assert cfg["enabled_scene_ids"] == [row["id"] for row in catalog["entries"]]
    assert len(cfg["enabled_scene_ids"]) == 15
    assert "008" not in cfg["enabled_scene_ids"] and "rigid_boxes" not in cfg["enabled_scene_ids"]
    for key in cfg["enabled_scene_ids"]:
        scene = SimpleNamespace(manifest=SimpleNamespace(scene_id=key))
        options = _scene_query_options(cfg["scene_query_backend"], cfg["scene_query_proofs"], scene)
        proof = json.loads(options["scene_query_proof"].read_text())
        assert proof["passed"] and proof["scene_sha256"] == cfg["scene_seeds"][key]["sha256"]
    cfg["scene_query_proofs"]["000"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="proof SHA256"):
        _scene_query_options("warp_bvh_exact", cfg["scene_query_proofs"],
                              SimpleNamespace(manifest=SimpleNamespace(scene_id="000")))


def test_wasd_scene_adapter_cache_reuses_graphs_shared_weights_and_releases_before_eviction(monkeypatch):
    import gc
    import weakref
    import torch
    import motionbricks.mb_sonic.mb_reference as reference
    from motionbricks.mb_sonic.scene import sha256
    cfg, events = _wasd_settings(), []
    controller, rep = SimpleNamespace(inference=SimpleNamespace(device="cpu")), object()
    graph_refs = {}

    class Plan:
        def __init__(self, adapter):
            self.adapter = adapter
            self.query = SimpleNamespace(native=self.query_native)
            self.call = SimpleNamespace(function=self.forward)

        def query_native(self):
            pass

        def forward(self):
            pass

    class Adapter:
        def __init__(self, controller, *, canonical_scene, identity, **kwargs):
            self.controller, self.identity, self.scene_id = controller, identity, canonical_scene.manifest.scene_id
            self.accounted_bytes, self.prepares, self.released = 100, 0, False
            self._full_plan = Plan(self)
            graph_refs[self.scene_id] = weakref.ref(self._full_plan)
            events.append(("construct", self.scene_id))

        def activate_scene(self):
            assert not self.released
            events.append(("activate", self.scene_id))

        def release_scene(self):
            self.released = True
            self._full_plan = None
            events.append(("release", self.scene_id))

        def prepare_compiled(self, history, **kwargs):
            self.prepares += 1
            return dict(execution="full_plan", scene_id=self.scene_id)

    def scene(key):
        return SimpleNamespace(manifest_path=Path(cfg["scene_root"])/key/"scene_manifest.json",
                               manifest=SimpleNamespace(scene_id=key))

    first_scene = scene("000")
    first = Adapter(controller, canonical_scene=first_scene, identity=dict(unit=True))
    original_collect = gc.collect
    cuda_initialized = torch.cuda.is_initialized()

    def collect():
        events.append(("gc", None))
        result = original_collect()
        assert graph_refs["001"]() is None
        assert graph_refs["000"]() is not None
        return result

    monkeypatch.setattr(gc, "collect", collect)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append(("empty_cache", None)))
    monkeypatch.setattr(reference, "TrainedMBReferenceAdapter", Adapter)
    observer = SimpleNamespace(representation=rep, preparation=dict(unit=True))
    monkeypatch.setattr(reference, "prepare_reference_history_b1", lambda *args: observer)
    provider = FrozenMBProvider(first, SimpleNamespace(rep=rep, root_z_offset_m=-.21),
        conditioning_mode="B1", estimate_actual_diagnostics=False,
        robot_model_sha256="unit", scene_manifest_sha256=sha256(first_scene.manifest_path),
        rig_id="unit", reference_history_builder=None, consumer_horizon_s=1.12)
    provider.execution, provider._scene_cache_capacity = "full_plan", 2
    history = SimpleNamespace(features=np.zeros((4, 366), np.float32))
    provider.prepare(history, actual_root_xyz_m=[0., 0., .79], desired_heading_rad=0.)
    provider.bind_scene(scene("001"))
    second = provider.generator
    provider.prepare(history, actual_root_xyz_m=[0., 0., .79], desired_heading_rad=0.)
    provider.bind_scene(first_scene)
    hit = provider.prepare(history, actual_root_xyz_m=[0., 0., .79], desired_heading_rad=0.)
    assert provider.generator is first and hit["scene_preparation_cache_hit"] is True
    assert first.prepares == second.prepares == 1
    assert not any(event[0] in ("gc", "empty_cache") for event in events)
    provider.bind_scene(scene("002"))
    assert second.released and not first.released
    assert events.index(("release", "001")) < events.index(("construct", "002"))
    assert events.index(("release", "001")) < events.index(("gc", None))
    assert events.index(("gc", None)) < events.index(("empty_cache", None)) < events.index(("construct", "002"))
    assert graph_refs["001"]() is None and graph_refs["000"]() is not None
    assert all(adapter.controller is controller for adapter in provider._scene_adapters.values())
    description = provider.cache_description()
    assert len(description["manifests"]) == 2 and description["accounted_cuda_torch_storage_bytes"] == 200
    assert description["warp_bvh_bytes"] is description["graph_arena_bytes"] is None
    assert description["torch_cuda_allocator"] is None
    controller.inference.device = "cuda:7"
    queried = []
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: queried.append(("allocated", str(device))) or 123)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: queried.append(("reserved", str(device))) or 456)
    allocator = provider.cache_description()["torch_cuda_allocator"]
    assert queried == [("allocated", "cuda:7"), ("reserved", "cuda:7")]
    assert allocator["memory_allocated_bytes"] == 123 and allocator["memory_reserved_bytes"] == 456
    assert allocator["device"] == "cuda:7" and isinstance(allocator["captured_at_monotonic_ns"], int)
    assert torch.cuda.is_initialized() == cuda_initialized


@pytest.mark.parametrize("field", ["packet_episode", "packet_session", "command_episode", "command_scene"])
def test_wasd_adoption_rejects_old_packet_epoch_session_and_map(field):
    import motionbricks.mb_sonic.runtime as module
    lab, command, _ = _scoped_intent_lab()
    lab._controller_session_id = command["session_id"]
    lab.physics = SimpleNamespace(scene_id="001", manifest_sha256="canonical", robot_model_sha256="robot", time_s=.24)
    _, update, _ = _pure_mb_worker_slot()
    packet = replace(update.packet, robot_model_sha256="robot", scene_manifest_sha256="canonical",
        joint_names=module.BODY_JOINT_NAMES, episode_id=2, session_id="human", generated_from_command_seq=1)
    update = replace(update, packet=packet, command=command,
        adapted_reference=SimpleNamespace(body_names=OFFICIAL_BODY_NAMES))
    if field == "packet_episode": update = replace(update, packet=replace(packet, episode_id=1))
    if field == "packet_session": update = replace(update, packet=replace(packet, session_id="old"))
    if field == "command_episode": update = replace(update, command=dict(command, episode_id=1))
    if field == "command_scene": update = replace(update, command=dict(command, scene_id="000"))
    with pytest.raises(ValueError, match="stale controller session or episode"):
        lab._adopt_mb_update(update)


def test_wasd_plan_and_tick_queues_are_independent_and_overflow_is_retained(tmp_path):
    rows = []
    recorder = SimpleNamespace(write=rows.append)
    plans = NumericArtifactWriter(tmp_path/"plans", recorder, pending_plans=1)
    ticks = NumericArtifactWriter(tmp_path/"ticks", recorder, pending_plans=2)
    plans.submit("plan", dict(value=np.arange(4)))
    with pytest.raises(NumericLogBackpressure):
        plans.submit("plan_overflow", dict(value=np.arange(5)))
    ticks.submit("tick_1", dict(value=np.arange(6)))
    ticks.submit("tick_2", dict(value=np.arange(7)))
    plans.start()
    ticks.start()
    with pytest.raises(RuntimeError, match="backpressure"):
        plans.close()
    ticks.close()
    assert {row["record_id"] for row in rows} == {"plan", "plan_overflow", "tick_1", "tick_2"}


def test_provider_native_wiring_uses_frozen_cpu_observer_real_seed_without_cuda():
    import torch
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.mb_sonic.mb_reference import decode_soma_reference, reference_history_b1_oracle
    from motionbricks.mb_sonic.reference import NativeSomaHistory

    settings = json.loads((ROOT / "motionbricks/configs/mb_sonic_b1_fullplan_v3.json").read_text())
    with np.load(settings["mb_seed_path"], allow_pickle=False) as archive:
        features = archive["initial_history"][0].copy()
    original = features.copy()
    old_threads = torch.get_num_threads()
    assert not torch.cuda.is_initialized()
    try:
        torch.set_num_threads(4)
        representation = build_native_representation()
        history = NativeSomaHistory((np.arange(4)-3)/30, features, "000", dict(kind="source_seed"))
        rng = torch.random.get_rng_state().clone()
        provider = FrozenMBProvider(SimpleNamespace(), SimpleNamespace(rep=representation),
            conditioning_mode="B1", estimate_actual_diagnostics=False, robot_model_sha256="unit",
            scene_manifest_sha256="unit", rig_id="unit", reference_history_builder=None, consumer_horizon_s=1.12)
        report = provider.prepare(history, actual_root_xyz_m=[0., 0., .79], desired_heading_rad=0.)
        assert report["execution"] == "native"
        assert report["cpu_history_observer"]["execution"] == "prepared_native_cpu_trace"
        assert report["cpu_history_observer"]["initial_all_channels_equal"] is True
        json.dumps(report, allow_nan=False)
        source = decode_soma_reference(features, history.timestamps_s, representation,
            scene_id="000", command_id="actual_frozen_seed_cpu_wiring_test", trusted_until_s=0.,
            provenance=dict(kind="source_seed"))
        actual = provider.reference_history_builder(source, end_time_s=0., representation=representation)
        expected = reference_history_b1_oracle(source, end_time_s=0., representation=representation)
        np.testing.assert_array_equal(actual.features, expected.features)
        np.testing.assert_array_equal(actual.timestamps_s, expected.timestamps_s)
        np.testing.assert_array_equal(history.features, original)
        np.testing.assert_array_equal(features, original)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(old_threads)


def test_provider_rejects_unadmitted_execution_before_loading_assets():
    with pytest.raises(ValueError, match="full_plan"):
        FrozenMBProvider.load(dict(mb_execution="implicit_fast_mode"), canonical_scene=None)


def test_consumer_horizon_contract_includes_processing_deadline_and_native_tick():
    settings = dict(consumer_horizon_s=1.12, mb_replan_interval_s=.1, max_plan_wall_age_s=.1)
    contract = consumer_horizon_contract(settings)
    assert contract["required_future_s"] == .9
    assert contract["minimum_required_horizon_s"] == pytest.approx(.9+.1+.1+.02)
    assert contract["usable_window_s"] == pytest.approx(.22)
    assert contract["dense_sample_count"] == 57
    for short in (1., 1.1):
        with pytest.raises(ValueError, match="scheduling requirement"):
            consumer_horizon_contract(dict(settings, consumer_horizon_s=short))
    for off_grid in (1.13, np.inf, np.nan):
        with pytest.raises(ValueError, match="native-50Hz"):
            consumer_horizon_contract(dict(settings, consumer_horizon_s=off_grid))
    with pytest.raises(ValueError, match="scheduling requirement"):
        consumer_horizon_contract(dict(settings, mb_replan_interval_s=.12))


def test_continuation_reserve_covers_recorded_three_expired_attempts():
    from motionbricks.mb_sonic.runtime import continuous_reference_replan_due
    active=SimpleNamespace(intent_revision=1,adapted_reference=SimpleNamespace(trusted_until_s=1.6))
    assert not continuous_reference_replan_due(active,1,0.,.16,.1)
    assert continuous_reference_replan_due(active,2,0.,.16,.1)
    assert continuous_reference_replan_due(None,1,0.,.16,.1)
    dispatch=next(i for i in range(50) if continuous_reference_replan_due(active,1,i*.02,.16,.1))
    # v49 rejected three consecutive GPU-delayed renewals at91.48/91.58/91.70s.
    # Retain those strict100ms bounds and leave time for a fresh fourth attempt.
    adoption=dispatch*.02+.12+.10+.12+.10
    assert adoption+.16<=active.adapted_reference.trusted_until_s+1e-8


def test_reference_body_parents_match_actual_official_g1_hierarchy_without_rollout():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    parents = _reference_body_parents(model, OFFICIAL_BODY_NAMES)
    assert parents == [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12]
    assert len(parents) == 14 and parents.count(-1) == 1
    assert all(parent < index for index, parent in enumerate(parents))


def _display_protocol_fixture(time_s):
    # Analytic protocol inputs, not generated or measured physical-run evidence.
    times = np.array([.18, .2, .22])
    qpos = np.zeros((3, 36))
    qpos[:, :3] = [[0., 0., .79], [.1, .2, .79], [.3, .4, .79]]
    qpos[:, 3] = 1
    body_positions = np.arange(3*14*3).reshape(3, 14, 3) / 100
    body_positions[:, 0] = qpos[:, :3]
    adapted = SimpleNamespace(timestamps_s=times, qpos=qpos, body_pos_w=body_positions,
                               trusted_until_s=.22, body_names=OFFICIAL_BODY_NAMES)
    command = dict(session_id="used_session", command_id=7, command_seq=7,
                   velocity_xy_mps=[.3, 0.], yaw_rate_radps=0.)
    update = SimpleNamespace(adapted_reference=adapted, command=command, derived_heading_rad=0.,
                              packet=SimpleNamespace(plan_id="used_plan", generated_from_command_seq=7))
    runtime = SonicLabRuntime.__new__(SonicLabRuntime)
    runtime.mode, runtime._active_update = "mb", update
    runtime.physics = SimpleNamespace(scene_id="000", manifest_sha256="scene",
        data=SimpleNamespace(geom_xmat=np.eye(3).reshape(1, 9), geom_xpos=np.zeros((1, 3))))
    runtime._robot_geom_ids = np.array([0])
    runtime._source_clock_origin_s, runtime._wall_start_ns, runtime._onset = None, None, None
    runtime._reference_version, runtime._consumed_command = "frozen_mb_test", command
    runtime._frame_id, runtime._episode = 0, 1
    runtime._status, runtime._errors = "protocol_fixture", []
    runtime._last_chain_ms, runtime._physics_steps, runtime._controller_steps = None, 1, 1
    state = _onset_measurement(time_s, [0., 0.])
    state.qpos[:3], state.state_version, state.contacts = [.1, .21, .79], 41, ()
    return runtime, state, adapted


@pytest.mark.parametrize("time_s,index", [(.18, 0), (.205, 1), (.22, 2)])
def test_mb_display_uses_adapted_g1_same_time_as_root_error_and_used_plan(time_s, index):
    runtime, state, adapted = _display_protocol_fixture(time_s)
    initial_qpos = state.qpos.copy()
    runtime._publish(state)
    frame = runtime._latest
    reference = frame["reference"]
    assert reference["kind"] == "adapted_g1"
    assert reference["version"] == "frozen_mb_test"
    assert reference["time_s"] == adapted.timestamps_s[index]
    assert reference["plan_id"] == "used_plan"
    assert reference["command_seq"] == reference["command_id"] == frame["command"]["command_seq"] == 7
    assert reference["session_id"] == frame["command"]["session_id"] == "used_session"
    np.testing.assert_array_equal(reference["joint_positions"], adapted.body_pos_w[index])
    assert len(reference["joint_positions"]) == 14
    assert frame["timing"]["tracking_error_m"] == pytest.approx(
        np.linalg.norm(initial_qpos[:3]-adapted.qpos[index, :3]))
    np.testing.assert_array_equal(state.qpos, initial_qpos)
    assert frame["actual"]["root_heading_rad"] == pytest.approx(0.)


@pytest.mark.parametrize("time_s", [.179, .225])
def test_mb_display_rejects_uncovered_adapted_g1_time(time_s):
    from motionbricks.mb_sonic.reference import ReferenceCoverageError

    runtime, state, _ = _display_protocol_fixture(time_s)
    with pytest.raises(ReferenceCoverageError, match="adapted G1"):
        runtime._publish(state)


def test_provider_b1_uses_reference_history_and_actual_morphology_query_without_rebase():
    # Pure contract fixture: no neural model, physical state, GPU or service execution.
    calls = {}
    features = np.arange(4*366).reshape(4, 366)
    history = SimpleNamespace(features=features, timestamps_s=np.arange(4)/30,
                              provenance=dict(kind="legacy_reference_history"))
    source = SimpleNamespace(trusted_until_s=1.8)
    proposal = SimpleNamespace(source=source)

    def generate(received_history, **kwargs):
        assert received_history is history
        calls["generate"] = kwargs
        return proposal

    def retarget_packet(full_source, *, start_time_s, consumer_horizon_s):
        assert full_source is source and source.trusted_until_s == 1.8
        calls["consumer_start"] = start_time_s
        calls["consumer_horizon_s"] = consumer_horizon_s
        qpos, qvel = np.zeros((57, 36)), np.zeros((57, 35))
        return SimpleNamespace(timestamps_s=start_time_s+np.arange(57)/50, qpos=qpos,
            joint_pos=qpos[:, 7:], joint_vel=qvel[:, 6:], trusted_until_s=start_time_s+consumer_horizon_s)

    provider = FrozenMBProvider(SimpleNamespace(generate=generate),
        SimpleNamespace(root_z_offset_m=-.21, retarget_packet=retarget_packet),
        conditioning_mode="B1", estimate_actual_diagnostics=False,
        robot_model_sha256="robot", scene_manifest_sha256="scene", rig_id="unit",
        reference_history_builder=None, consumer_horizon_s=1.12)
    command = dict(session_id="s", episode_id=1, command_id=7, command_seq=7, velocity_xy_mps=[1., 2.],
                   yaw_rate_radps=.7, server_received_ns=1)
    snapshot = MBGenerationSnapshot(.2, .2, 40, 3, np.array([2., 3., .79]), .4, history, None, 10, {})
    with pytest.raises(RuntimeError, match="Explicit MB deployment preparation"):
        provider.propose(snapshot, command=command, intent_revision=3,
                          requested_at_ns=20, current_sim_time=lambda: .205)
    provider.preparation_report = dict(execution="native", cpu_history_observer="unit_contract_already_prepared")
    update = provider.propose(snapshot, command=command, intent_revision=3,
                              requested_at_ns=20, current_sim_time=lambda: .205)
    assert calls["consumer_start"] == pytest.approx(.22)
    assert calls["consumer_horizon_s"] == 1.12
    np.testing.assert_allclose(calls["generate"]["query_origin_soma_xyz"], [2., 1., -3.])
    np.testing.assert_allclose(calls["generate"]["velocity_world_xz_mps"], [1., -2.])
    assert calls["generate"]["heading_world_rad"] == pytest.approx(.4+np.pi/2)
    assert calls["generate"]["conditioning_mode"] == "B1"
    assert calls["generate"]["blocks"] == 2
    assert update.packet.issued_from_sim_time == .2
    assert update.packet.valid_until == pytest.approx(1.34)
    assert len(update.packet.future_timestamps) == 57
    assert update.packet.generated_from_command_seq == 7
    assert update.estimated_actual_history is None
    assert "heading_rad" not in update.command
    np.testing.assert_array_equal(features, np.arange(4*366).reshape(4, 366))
    source.trusted_until_s = 1.3
    calls.pop("consumer_start")
    with pytest.raises(ValueError, match="explicit consumer horizon"):
        provider.propose(snapshot, command=command, intent_revision=3,
                          requested_at_ns=30, current_sim_time=lambda: .205)
    assert "consumer_start" not in calls
    assert provider.request_evidence["consumer_horizon_s"] == 1.12
    assert provider.request_evidence["consumer_end_s"] == pytest.approx(1.34)


def _saved_process_update():
    from motionbricks.mb_sonic.mb_reference import MBProposal
    from motionbricks.mb_sonic.reference import G1Reference, NativeSomaHistory, RawMBBlock, SomaReference
    path = ROOT/"motionbricks/out/mb_sonic_e0_e3_20261018/wasd_v3/episode_0001_numeric/19af187647d0857013f1f95c75aa0fd75bb3faf165f2be4c5d7955fa57734fd0.npz"
    with np.load(path, allow_pickle=False) as archive:
        def decode(value):
            if isinstance(value, dict):
                if "array" in value and "dtype" in value:
                    return archive[value["array"]].copy()
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return tuple(decode(item) for item in value)
            return value

        record = decode(json.loads(str(archive["metadata_json"]))["record"])
    history = NativeSomaHistory(**record["conditioning_history"])
    source = SomaReference(**record["raw_source"])
    blocks = tuple(RawMBBlock(**block) for block in record["raw_blocks"])
    proposal = MBProposal(source, blocks, history, record["proposal_timing_ns"])
    return MBReferenceUpdate(record["intent_revision"], record["command"], record["snapshot_time_s"],
        0., history, None, proposal, G1Reference(**record["adapted_reference"]),
        record["timing_ns"], record["clocks_ns"], ReferencePacket(**record["packet"]))


def test_process_ipc_real_saved_packet_is_bitwise_owned_readonly_and_revalidated():
    import pickle
    from motionbricks.mb_sonic.planner_process import restore_ipc_value
    original = _saved_process_update()
    transported = pickle.loads(pickle.dumps(original, protocol=5))
    object.__setattr__(transported.adapted_reference, "qpos", transported.adapted_reference.qpos.copy())
    restored = restore_ipc_value(transported)
    for before, after in ((original.proposal.source.raw_features, restored.proposal.source.raw_features),
                          (original.proposal.raw_blocks[0].raw_features, restored.proposal.raw_blocks[0].raw_features),
                          (original.adapted_reference.qpos, restored.adapted_reference.qpos),
                          (original.adapted_reference.qvel, restored.adapted_reference.qvel),
                          (original.packet.root_pose, restored.packet.root_pose),
                          (original.packet.future_timestamps, restored.packet.future_timestamps)):
        np.testing.assert_array_equal(before, after)
        assert before.dtype == after.dtype and not after.flags.writeable
        assert not np.shares_memory(before, after)
    transported.adapted_reference.qpos[:] = 99
    np.testing.assert_array_equal(restored.adapted_reference.qpos, original.adapted_reference.qpos)
    corrupted = pickle.loads(pickle.dumps(original, protocol=5))
    object.__setattr__(corrupted.adapted_reference, "qpos", corrupted.adapted_reference.qpos.copy())
    corrupted.adapted_reference.qpos[:, 3:7] = 0
    with pytest.raises(ValueError, match="quaternions"):
        restore_ipc_value(corrupted)


class _CPUProcessProtocolProvider:
    """Protocol test double only; never used by the production child entrypoint."""

    def __init__(self, update):
        self.update, self.request_evidence = update, None
        self.reference_stream_policy = None
        self.scene_manifest_sha256 = update.packet.scene_manifest_sha256

    def propose(self, snapshot, *, command, intent_revision, requested_at_ns, current_sim_time):
        from motionbricks.mb_sonic.reference import ReferenceCoverageError
        actual_time = current_sim_time()
        self.request_evidence = dict(proposal=self.update.proposal, conditioning_history=snapshot.reference_history,
            command=command, snapshot=snapshot, consumer_start_s=actual_time,
            consumer_end_s=actual_time+1.12, consumer_horizon_s=1.12, requested_at_ns=requested_at_ns)
        if command.get("unit_raise"):
            raise ReferenceCoverageError("unit genuine-evidence coverage rejection")
        clocks = dict(self.update.clocks_ns, requested=requested_at_ns, reference_ready=perf_counter_ns())
        packet = replace(self.update.packet, issued_from_sim_time=snapshot.simulation_time_s,
            state_version=snapshot.state_version, generated_from_command_seq=command["command_seq"],
            episode_id=command["episode_id"], session_id=command["session_id"], clocks_ns=clocks)
        return replace(self.update, command=command, packet=packet, intent_revision=intent_revision,
            conditioning_history=snapshot.reference_history, history_time_s=snapshot.history_time_s,
            clocks_ns=clocks, timing_ns=dict(provider_total=1, actual_callback_time_s=actual_time))


def _cpu_planner_child(connection, update):
    from motionbricks.mb_sonic.planner_process import restore_ipc_value, serve_provider
    serve_provider(connection, _CPUProcessProtocolProvider(restore_ipc_value(update)),
                   maximum_bytes=8*1024*1024, callback_timeout_s=2.)


def _cpu_history_planner_child(connection, update):
    from functools import partial
    import torch
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.mb_sonic.mb_reference import prepare_reference_history_b1, reference_history_b1
    from motionbricks.mb_sonic.planner_process import restore_ipc_value, serve_provider
    torch.set_num_threads(4)
    update = restore_ipc_value(update)
    provider = _CPUProcessProtocolProvider(update)
    representation = build_native_representation()
    observer = prepare_reference_history_b1(representation, update.conditioning_history.features)
    provider.retargeter = SimpleNamespace(rep=representation)
    provider.reference_history_builder = partial(reference_history_b1, observer=observer)
    serve_provider(connection, provider, maximum_bytes=8*1024*1024, callback_timeout_s=2.)


def test_process_child_b1_history_is_bitwise_and_success_preserves_full_update_without_evidence():
    import multiprocessing
    import torch
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.mb_sonic.mb_reference import prepare_reference_history_b1, reference_history_b1
    from motionbricks.mb_sonic.planner_process import PlannerRPC
    torch.set_num_threads(4)
    update = _saved_process_update()
    representation = build_native_representation()
    observer = prepare_reference_history_b1(representation, update.conditioning_history.features)
    source = update.proposal.source
    times = source.timestamps_s[[4, 11, 24, 46]]
    expected = [reference_history_b1(source, end_time_s=float(time), representation=representation,
                                     observer=observer) for time in times]
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_cpu_history_planner_child, args=(child, update))
    process.start()
    child.close()
    rpc = PlannerRPC(parent, maximum_bytes=8*1024*1024, timeout_s=20.)
    command = dict(update.command)
    scope = dict(scene_manifest_sha256=update.packet.scene_manifest_sha256,
        episode_id=command["episode_id"], session_id=command["session_id"], command_seq=command["command_seq"])
    try:
        for time, native_history in zip(times, expected):
            requested = perf_counter_ns()
            snapshot = MBGenerationSnapshot(float(time), float(time), 36, update.intent_revision,
                update.packet.query_root_xyz_m, 0., None, None, 123, {}, reference_source=source)
            response = rpc.call("propose", dict(snapshot=snapshot, command=command,
                intent_revision=update.intent_revision, requested_at_ns=requested),
                scope=scope, current_sim_time=lambda: .26)
            assert response["kind"] == "result" and response["evidence"] is None
            actual = response["value"]
            np.testing.assert_array_equal(actual.conditioning_history.features, native_history.features)
            np.testing.assert_array_equal(actual.conditioning_history.timestamps_s, native_history.timestamps_s)
            assert not actual.conditioning_history.features.flags.writeable
            np.testing.assert_array_equal(actual.proposal.source.raw_features, source.raw_features)
            for field in ("qpos", "qvel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                np.testing.assert_array_equal(getattr(actual.adapted_reference, field),
                                              getattr(update.adapted_reference, field))
            for before, after in zip(update.proposal.raw_blocks, actual.proposal.raw_blocks):
                np.testing.assert_array_equal(before.raw_features, after.raw_features)
            assert actual.packet.issued_from_sim_time == snapshot.simulation_time_s
            assert actual.packet.state_version == 36 and actual.clocks_ns["requested"] == requested
            assert actual.clocks_ns["child_history_ready"] >= actual.clocks_ns["child_history_started"]
            assert actual.timing_ns["provider_total"] >= actual.timing_ns["child_reference_history"] > 0
        rpc.call("close", {}, scope={})
    finally:
        parent.close()
        process.join(5.)
        if process.is_alive():
            process.terminate()
            process.join(5.)
    assert process.exitcode == 0


def test_process_actual_spawn_callback_timing_exception_evidence_and_shutdown():
    import multiprocessing
    from motionbricks.mb_sonic.planner_process import PlannerRPC, ProcessMBProvider
    from motionbricks.mb_sonic.reference import ReferenceCoverageError
    update = _saved_process_update()
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_cpu_planner_child, args=(child, update))
    process.start()
    child.close()
    provider = ProcessMBProvider.__new__(ProcessMBProvider)
    provider._process = process
    provider._rpc = PlannerRPC(parent, maximum_bytes=8*1024*1024, timeout_s=5.)
    provider.settings = dict(planner_process=dict(startup_timeout_s=5., shutdown_timeout_s=2.))
    provider.scene_manifest_sha256 = update.packet.scene_manifest_sha256
    provider.robot_model_sha256 = update.packet.robot_model_sha256
    provider.estimate_actual_diagnostics, provider._closed = False, False
    provider.conditioning_mode = "B1"
    provider._metadata, provider.failure_evidence = {}, None
    snapshot = MBGenerationSnapshot(.18, .18, 36, update.intent_revision,
        update.packet.query_root_xyz_m, 0., update.conditioning_history, None, 123, {})
    command = dict(update.command)
    calls = []

    def actual_time():
        calls.append(.26)
        return .26

    try:
        requested = perf_counter_ns()
        result = provider.propose(snapshot, command=command, intent_revision=update.intent_revision,
            requested_at_ns=requested, current_sim_time=actual_time)
        assert calls == [.26] and result.timing_ns["actual_callback_time_s"] == .26
        assert provider.request_evidence is None
        assert result.clocks_ns["requested"] == requested
        assert result.clocks_ns["ipc_parent_started"] >= requested
        assert result.timing_ns["provider_total"] >= result.timing_ns["child_provider_total"]
        assert result.timing_ns["ipc_and_parent_work"] > 0
        assert not result.proposal.source.raw_features.flags.writeable
        assert not result.adapted_reference.qpos.flags.writeable
        assert not result.packet.future_timestamps.flags.writeable
        np.testing.assert_array_equal(result.adapted_reference.qpos, update.adapted_reference.qpos)
        with pytest.raises(ReferenceCoverageError, match="coverage rejection"):
            provider.propose(snapshot, command=dict(command, unit_raise=True), intent_revision=update.intent_revision,
                requested_at_ns=perf_counter_ns(), current_sim_time=actual_time)
        assert "unit genuine-evidence" in provider.failure_evidence["child_traceback"]
        assert provider.failure_evidence["request_id"] == 2
        assert provider.failure_evidence["scope"]["session_id"] == command["session_id"]
        assert provider.request_evidence["consumer_start_s"] == .26
        assert not provider.request_evidence["proposal"].source.raw_features.flags.writeable
    finally:
        provider.close()
    assert not process.is_alive() and process.exitcode == 0
    provider.close()
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        provider._call("prepare", {})


def test_process_provider_uses_operation_specific_rpc_deadlines():
    from motionbricks.mb_sonic.planner_process import ProcessMBProvider
    provider = ProcessMBProvider.__new__(ProcessMBProvider)
    calls = []
    provider._closed, provider.scene_manifest_sha256 = False, "scene"
    provider.settings = dict(planner_process=dict(startup_timeout_s=180., shutdown_timeout_s=3.))
    provider._rpc = SimpleNamespace(call=lambda op, args, **kwargs: calls.append((op, kwargs["timeout_s"]))
        or dict(kind="result", evidence=None, value=None))
    for operation in ("prepare", "bind_scene", "propose", "close"):
        provider._call(operation, {})
    assert calls == [("prepare", 180.), ("bind_scene", 180.), ("propose", None), ("close", 3.)]


def test_process_rpc_stalled_send_is_bounded_and_never_retried(monkeypatch):
    from time import perf_counter
    import motionbricks.mb_sonic.planner_process as module
    entered, release, finished = Event(), Event(), Event()
    calls = []

    def stalled_send(*args):
        calls.append(1)
        entered.set()
        try:
            release.wait(2)
            raise OSError("unit released blocked send")
        finally:
            finished.set()

    connection = SimpleNamespace(close=release.set)
    monkeypatch.setattr(module, "_send", stalled_send)
    rpc = module.PlannerRPC(connection, maximum_bytes=1024, timeout_s=.02)
    started = perf_counter()
    with pytest.raises(TimeoutError, match="deadline"):
        rpc.call("propose", {}, scope=dict(scene="unit"))
    assert perf_counter()-started < .5
    assert entered.is_set() and finished.wait(2)
    with pytest.raises(RuntimeError, match="planner_process_transport"):
        rpc.call("propose", {}, scope=dict(scene="unit"))
    assert calls == [1]


def test_process_rpc_rejects_wrong_response_scope_without_consuming_it():
    import multiprocessing
    from motionbricks.mb_sonic.planner_process import PlannerRPC, _receive, _send
    parent, child = multiprocessing.get_context("spawn").Pipe(duplex=True)

    def wrong_reply():
        try:
            request = _receive(child, 4096, 2.)
            _send(child, dict(kind="result", request_id=request["request_id"],
                scope=dict(episode_id=99), value=None), 4096)
        finally:
            child.close()

    thread = Thread(target=wrong_reply)
    thread.start()
    try:
        with pytest.raises(ValueError, match="identity differs"):
            PlannerRPC(parent, maximum_bytes=4096, timeout_s=2.).call(
                "propose", {}, scope=dict(episode_id=1))
    finally:
        parent.close()
        thread.join(2)
    assert not thread.is_alive()


def test_process_rpc_child_eof_is_terminal_without_retry():
    import multiprocessing
    from motionbricks.mb_sonic.planner_process import PlannerRPC
    parent, child = multiprocessing.get_context("spawn").Pipe(duplex=True)
    child.close()
    rpc = PlannerRPC(parent, maximum_bytes=4096, timeout_s=.1)
    with pytest.raises((OSError, EOFError)):
        rpc.call("propose", {}, scope={})
    assert rpc.failure is not None
    with pytest.raises(RuntimeError, match="planner_process_transport"):
        rpc.call("propose", {}, scope={})


def test_online_mb_process_traceback_is_persisted_in_failure_json_and_raw_evidence():
    from motionbricks.mb_sonic.reference import ReferenceCoverageError
    worker, update, rows = _pure_mb_worker_slot()
    receipts = []
    failure = dict(request_id=7, scope=dict(episode_id=1, session_id="unit", command_seq=2),
                   child_traceback="Traceback: child real coverage failure")
    worker.provider.failure_evidence = failure
    worker.history_snapshot = lambda: SimpleNamespace(integrated_intent_revision=update.intent_revision)
    worker.numeric_writer.submit = lambda record_id, record: receipts.append(record)

    def rejected(*args, **kwargs):
        worker.provider.request_evidence = dict(requested_at_ns=1, proposal=update.proposal,
            conditioning_history=None, command=update.command, consumer_horizon_s=1.12,
            consumer_start_s=.2, consumer_end_s=1.32,
            snapshot=SimpleNamespace(state_version=40, simulation_time_s=.2))
        raise ReferenceCoverageError("unit child coverage failure")

    worker.provider.propose = rejected
    worker.request()
    assert worker.control_tick(.2, intent=worker.intents.latest())
    worker._run()
    assert worker._failure.startswith("ReferenceCoverageError:")
    assert rows[0]["planner_process_failure"] == failure
    assert receipts[0]["planner_process_failure"] == failure


@pytest.mark.parametrize("stop_raises", [False, True])
def test_lab_shutdown_closes_process_even_on_stop_failure_but_manual_stop_does_not(stop_raises):
    lab = SonicLabRuntime.__new__(SonicLabRuntime)
    calls = []
    lab._mb_planner_backend, lab._provider = "spawn", SimpleNamespace(close=lambda: calls.append("close"))

    def stop():
        calls.append("stop")
        if stop_raises:
            raise OSError("unit final drain failure")

    lab.stop = stop
    if stop_raises:
        with pytest.raises(OSError, match="drain failure"):
            lab.shutdown()
    else:
        lab.stop()
        assert calls == ["stop"]
        calls.clear()
        lab.shutdown()
    assert calls == ["stop", "close"]
