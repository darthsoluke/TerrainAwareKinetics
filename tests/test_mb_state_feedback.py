"""Measured feedback geometry and causality, using CPU FK only."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.mb_sonic.mb_reference import decode_soma_reference, prepare_reference_history_b1, validate_conditioning
from motionbricks.mb_sonic.reference import ExecutedHistory, NativeSomaHistory, ReferenceCoverageError
from motionbricks.mb_sonic.retarget import CalibratedSomaG1Adapter, _quat
from motionbricks.mb_sonic.scene import SOMA_TO_MUJOCO
from motionbricks.mb_sonic.state_feedback import error_state_history, FixedPairStateObserver


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def feedback_assets():
    import torch

    torch.set_num_threads(2)
    settings = json.loads((ROOT/"configs/mb_protomotions_wasd_v13.json").read_text())
    rep = build_native_representation()
    adapter = CalibratedSomaG1Adapter.load(settings["calibration_path"],
        robot_model_path=settings["robot_model"], representation=rep)
    with np.load(settings["scene_seeds"]["000"]["path"]) as archive:
        raw = archive["initial_history"][0].copy()
    seed = NativeSomaHistory((np.arange(4)-3)/30, raw, "000", dict(kind="source_seed"))
    observer = prepare_reference_history_b1(rep, raw)
    source = decode_soma_reference(np.repeat(raw[-1:], 32, axis=0), np.arange(32)/30,
        rep, scene_id="000", command_id="static", trusted_until_s=31/30, provenance=dict(kind="test_prior"))
    reference = adapter.retarget(source)
    return adapter, seed, observer, source, reference


def samples(assets, *, speed=0., turn=0., knee=0., bootstrap=False):
    adapter, seed, observer, source, reference = assets
    kin = adapter.kinematics
    result = ExecutedHistory("000", body_names=kin.body_names)
    for time in np.arange(11)/50:
        q = reference.qpos[round(time*50)].copy()
        q[:3] += [speed*time, 0., 0.]
        r = Rotation.from_rotvec([0., 0., turn*time]).as_matrix() @ Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix()
        q[3:7] = _quat(r)
        q[10] += knee*time
        p, rotations = kin.fk(q[None])
        result.samples.append(dict(timestamp_s=time, qpos=q, qvel=np.zeros(35),
            body_pos_w=p[0], body_quat_w=_quat(rotations[0]), contact_columns=dict(
                body_ids=np.empty((0, 2), int), position_world=np.empty((0, 3)), wrench_contact=np.empty((0, 6))),
            state_version=round(time*1000), bootstrap_until_s=.1,
            native_reference_source=None if bootstrap else source,
            reference_qpos=None if bootstrap else reference.qpos[round(time*50)].copy(),
            robot_reference_source=None if bootstrap else reference, reference_plan_id="paired_prior"))
    return result


def test_zero_error_preserves_full_native_pose_not_neutral_filling(feedback_assets):
    adapter, seed, observer, source, _ = feedback_assets
    history = error_state_history(seed, samples(feedback_assets), end_time_s=.2, adapter=adapter, observer=observer)
    decoded = decode_soma_reference(history.features, history.timestamps_s, adapter.rep,
        scene_id="000", command_id="feedback", trusted_until_s=.2, provenance=dict(history.provenance))
    np.testing.assert_allclose(decoded.global_rot_mats, source.global_rot_mats[[0]*4], atol=3e-6, rtol=0)
    np.testing.assert_allclose(decoded.root_positions_m, source.root_positions_m[[0]*4], atol=3e-6, rtol=0)
    assert history.provenance["kind"] == "actual_g1_error_state"


def test_measured_translation_heading_and_joint_errors_enter_history(feedback_assets):
    adapter, seed, observer, source, reference = feedback_assets
    executed = samples(feedback_assets, speed=.7, turn=.4, knee=.1)
    history = error_state_history(seed, executed, end_time_s=.2, adapter=adapter, observer=observer)
    expected = source.root_positions_m[0]+np.stack((.7*history.timestamps_s, np.zeros(4), np.zeros(4)), -1)
    np.testing.assert_allclose(history.features[:, adapter.rep.indices["global_root_pos"]], expected, atol=3e-6)
    baseline = error_state_history(seed, samples(feedback_assets, speed=.7, turn=.4), end_time_s=.2,
        adapter=adapter, observer=observer)
    assert np.max(abs(history.features-baseline.features)) > 1e-4
    assert all(row["actual_brackets_s"][1] <= .2+1e-9 for row in history.provenance["samples"])
    assert validate_conditioning(history, "B2", expected[-1]).shape == (3,)
    with pytest.raises(ValueError, match="B1"):
        validate_conditioning(history, "B1", expected[-1])


def test_bootstrap_uses_measured_frames_and_cannot_extend_initial_prior(feedback_assets):
    adapter, seed, observer, _, _ = feedback_assets
    executed = samples(feedback_assets, bootstrap=True)
    history = error_state_history(seed, executed, end_time_s=.1, adapter=adapter, observer=observer)
    np.testing.assert_allclose(history.timestamps_s, np.arange(4)/30, atol=1e-12)
    with pytest.raises(ReferenceCoverageError, match="Initial pose prior"):
        error_state_history(seed, executed, end_time_s=.2, adapter=adapter, observer=observer)


def test_fixed_observer_does_not_reinterpret_history_when_plan_identity_changes(feedback_assets):
    adapter, seed, observer, source, reference = feedback_assets
    state_observer = FixedPairStateObserver(seed, reference.qpos[0], adapter)
    actual = samples(feedback_assets, speed=.4, turn=.6, knee=.1)
    original = state_observer.history(actual, end_time_s=.2, observer=observer)
    for row in actual.samples:
        row.update(native_reference_source=None, robot_reference_source=None, reference_plan_id='new_plan')
    changed = state_observer.history(actual, end_time_s=.2, observer=observer)
    np.testing.assert_array_equal(original.features, changed.features)
    assert changed.provenance['reference_dependent'] is False


def test_fixed_observer_preserves_seed_pose_and_observes_actual_leg_motion(feedback_assets):
    adapter, seed, observer, source, reference = feedback_assets
    state_observer = FixedPairStateObserver(seed, reference.qpos[0], adapter)
    baseline = state_observer.history(samples(feedback_assets), end_time_s=.2, observer=observer)
    decoded = decode_soma_reference(baseline.features, baseline.timestamps_s, adapter.rep,
        scene_id='000', command_id='fixed', trusted_until_s=.2, provenance=dict(baseline.provenance))
    np.testing.assert_allclose(decoded.global_rot_mats, source.global_rot_mats[[0]*4], atol=3e-6, rtol=0)
    measured = state_observer.history(samples(feedback_assets, knee=.4), end_time_s=.2, observer=observer)
    assert np.max(np.abs(measured.features-baseline.features)) > .01
    assert measured.provenance['command_dependent'] is False
    assert all(s['prior_plan_id'] == 'fixed_initial_pair' for s in measured.provenance['samples'])


def test_reference_phase_uses_full_g1_joint_hierarchy_and_immutable_causal_samples(feedback_assets):
    adapter, seed, observer, source, reference = feedback_assets
    state_observer = FixedPairStateObserver(seed, reference.qpos[0], adapter, limb_source='reference')
    actual = samples(feedback_assets, speed=.4, turn=.6, knee=.4)
    first = state_observer.history(actual, end_time_s=.2, observer=observer)
    for sample in actual.samples:
        sample.update(native_reference_source=None, robot_reference_source=None, reference_plan_id='new_metadata')
    second = state_observer.history(actual, end_time_s=.2, observer=observer)
    np.testing.assert_array_equal(first.features, second.features)
    assert second.provenance['raw_native_plan_dependent'] is False
    assert second.provenance['limb_source'] == 'reference'
    decoded = decode_soma_reference(second.features, second.timestamps_s, adapter.rep,
        scene_id='000', command_id='phase', trusted_until_s=.2, provenance=dict(second.provenance))
    knee = adapter.names.index('LeftShin')
    np.testing.assert_allclose(decoded.local_rot_mats[:, knee], source.local_rot_mats[[0]*4, knee], atol=3e-6)


def test_later_measurements_cannot_change_an_existing_history_window(feedback_assets):
    adapter, seed, observer, _, reference = feedback_assets
    fixed = FixedPairStateObserver(seed, reference.qpos[0], adapter, limb_source='reference')
    actual = samples(feedback_assets, speed=.4, turn=.6, knee=.2)
    before = fixed.history(actual, end_time_s=.18, observer=observer)
    actual.samples[-1]['qpos'] = actual.samples[-1]['qpos']+0.1
    actual.samples[-1]['reference_qpos'] = actual.samples[-1]['reference_qpos']+0.2
    after = fixed.history(actual, end_time_s=.18, observer=observer)
    np.testing.assert_array_equal(before.features, after.features)


def test_unobserved_local_pose_and_physical_input_arrays_are_preserved(feedback_assets):
    adapter, seed, observer, source, reference = feedback_assets
    actual = samples(feedback_assets, speed=.4, turn=.6, knee=.1)
    original_q = [row["qpos"].copy() for row in actual.samples]
    original_features = source.raw_features.copy()
    history = error_state_history(seed, actual, end_time_s=.2, adapter=adapter, observer=observer)
    decoded = decode_soma_reference(history.features.copy(), history.timestamps_s, adapter.rep,
        scene_id="000", command_id="feedback", trusted_until_s=.2, provenance=dict(history.provenance))
    bone = adapter.names.index("Neck1")
    np.testing.assert_allclose(decoded.local_rot_mats[:, bone], source.local_rot_mats[[0]*4, bone], atol=3e-6)
    for row, original in zip(actual.samples, original_q):
        np.testing.assert_array_equal(row["qpos"], original)
    np.testing.assert_array_equal(source.raw_features, original_features)


def test_feedback_uses_actual_contact_force_without_distance_labels(feedback_assets):
    adapter, seed, observer, _, _ = feedback_assets
    actual = samples(feedback_assets)
    body = actual.body_names.index("left_ankle_roll_link")
    for row in actual.samples:
        rotation = Rotation.from_quat(row["body_quat_w"][body, [1, 2, 3, 0]]).as_matrix()
        row["contact_columns"] = dict(body_ids=np.array([[body, 0]]),
            wrench_contact=np.array([[20., 0, 0, 0, 0, 0]]),
            position_world=(row["body_pos_w"][body]+rotation@np.array([.3, 0., 0.]))[None])
    history = error_state_history(seed, actual, end_time_s=.2, adapter=adapter, observer=observer)
    expected = np.repeat([[0., 1., 0., 0.]], 4, axis=0)
    np.testing.assert_array_equal(history.features[:, adapter.rep.indices["foot_contacts"]], expected)


def test_root_torso_feedback_preserves_reference_limb_phase(feedback_assets):
    adapter, seed, observer, source, _ = feedback_assets
    executed = samples(feedback_assets, speed=.4, turn=.3, knee=.2)
    history = error_state_history(seed, executed, end_time_s=.2, adapter=adapter, observer=observer,
        correction_segments=("Hips", "Chest"))
    decoded = decode_soma_reference(history.features.copy(), history.timestamps_s, adapter.rep,
        scene_id="000", command_id="feedback", trusted_until_s=.2, provenance=dict(history.provenance))
    bone = adapter.names.index("LeftShin")
    np.testing.assert_allclose(decoded.local_rot_mats[:, bone], source.local_rot_mats[[0]*4, bone], atol=3e-6)
    assert history.provenance["measured_correction_segments"] == ["Hips", "Chest"]


def test_reference_constraints_do_not_hide_real_orientation_from_native_feedback(feedback_assets):
    adapter, seed, observer, source, reference = feedback_assets
    actual = samples(feedback_assets, speed=.4, turn=.6, knee=.1)
    poses = reference.qpos.copy()
    for row in actual.samples:
        poses[round(row['timestamp_s']*50)] = row['qpos']
    constrained = replace(reference, qpos=poses)
    for row in actual.samples:
        row['robot_reference_source'] = constrained
    history = error_state_history(seed, actual, end_time_s=.2, adapter=adapter, observer=observer,
        correction_segments=('Hips', 'Chest'), robot_prior='native_anatomical')
    decoded = decode_soma_reference(history.features, history.timestamps_s, adapter.rep,
        scene_id='000', command_id='feedback', trusted_until_s=.2, provenance=dict(history.provenance))
    measured_turn = Rotation.from_rotvec(history.timestamps_s[:, None]*[0., 0., .6]).as_matrix()
    expected = SOMA_TO_MUJOCO.T@measured_turn@SOMA_TO_MUJOCO@source.global_rot_mats[0, 0]
    np.testing.assert_allclose(decoded.global_rot_mats[:, 0], expected, atol=3e-6, rtol=0)
    old_history = error_state_history(seed, actual, end_time_s=.2, adapter=adapter, observer=observer,
        correction_segments=('Hips', 'Chest'), robot_prior='consumed_reference')
    old = decode_soma_reference(old_history.features, old_history.timestamps_s, adapter.rep,
        scene_id='000', command_id='old', trusted_until_s=.2, provenance=dict(old_history.provenance))
    assert np.max(np.abs(old.global_rot_mats[:, 0]-expected)) > .05
    assert history.provenance['robot_pose_prior'] == 'native_anatomical'


def test_recorded_reference_boundary_roundoff_is_solved_with_original_limits(feedback_assets):
    from scripts.audit_mb_tracker_binary import decode
    from motionbricks.mb_sonic.reference import SomaReference
    from motionbricks.mb_sonic.constrained_reference import ConstrainedReferenceSolver

    path = ROOT / "out/mb_sonic_e0_e3_20261018/wasd_protomotions_v16_feedback/episode_0001_000.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    artifacts = {r["record_id"]: r for r in rows if r["kind"] == "numeric_artifact_written"}
    failed = next(key for key in artifacts if key.startswith("failed:"))
    requested = int(failed.split(":")[1])

    def load(key):
        from motionbricks.mb_sonic.scene import sha256

        artifact = artifacts[key]
        assert sha256(artifact["path"]) == artifact["sha256"]
        with np.load(artifact["path"], allow_pickle=False) as archive:
            return decode(json.loads(str(archive["metadata_json"]))["record"], archive)

    failure = load(failed)
    prior = max((r for r in rows if r["kind"] == "reference_packet_adopted"
        and r["reference_adopted_monotonic_ns"] <= requested), key=lambda r: r["reference_adopted_monotonic_ns"])
    old = load(prior["plan_id"])["adapted_reference"]
    values = failure["raw_source"]
    values["bone_names"] = tuple(values["bone_names"])
    source = SomaReference(**values)
    adapter = feedback_assets[0]
    boundary = dict(timestamps_s=old["timestamps_s"], joint_positions=old["qpos"][:, 7:],
        plan_id=prior["plan_id"], adoption_until_s=failure["issued_from_sim_time"]+.1)
    reference = adapter.retarget_packet(source, start_time_s=failure["consumer_start_s"],
        consumer_horizon_s=.6, required_future_s=.16, boundary=boundary,
        constraints=ConstrainedReferenceSolver(adapter.kinematics))
    assert reference.adaptation["dense_constraint_audit"]["status"] == "PASS"
    report = reference.adaptation["reference_constraints"]
    assert report["root_exact"]
    assert report["maximum_velocity_ratio"] <= 1+1e-9


@pytest.mark.parametrize("end", [.08, .23])
def test_feedback_never_extrapolates_physical_history(feedback_assets, end):
    adapter, seed, observer, _, _ = feedback_assets
    with pytest.raises(ReferenceCoverageError, match="factual covered"):
        error_state_history(seed, samples(feedback_assets), end_time_s=end, adapter=adapter, observer=observer)
