"""CPU kinematic tests on official paired data and recorded actual E1 physics."""

import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from motionbricks.mb_sonic.reference import ExecutedHistory, ReferenceCoverageError, SomaReference, load_official_reference, load_soma_reference
from motionbricks.mb_sonic.retarget import (BODY_MAP, SEGMENTS, CalibratedSomaG1Adapter,
                                           G1Kinematics, _load_pair, _quat, anatomical_frames,
                                           native_limb_material_frames, native_rest_frames)
from motionbricks.mb_sonic.scene import sha256


ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = ROOT.parent / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
SAMPLES = ROOT / "out/mb_sonic_e0_e3_20261018/sonic_6733128/sample_data"
CLIP = "walk_forward_amateur_001__A001"
V3_NUMERIC = ROOT / "out/mb_sonic_e0_e3_20261018/e3_b1_fullplan_v3/episode_0001_numeric"


def load_numeric_source(path):
    with np.load(path, allow_pickle=False) as archive:
        record = json.loads(str(archive["metadata_json"]))["record"]
        if "raw_source" not in record:
            return None
        return SomaReference(**{key: archive[value["array"]].copy()
                               if isinstance(value, dict) and "array" in value else value
                               for key, value in record["raw_source"].items()})


def legacy_root_at_times(adapter, source, times):
    legacy = adapter._map_rotations(source.root_positions_m, source.global_rot_mats)
    positions = np.stack([np.interp(times, source.timestamps_s, legacy[:, i]) for i in range(3)], axis=-1)
    rotations = Slerp(source.timestamps_s, Rotation.from_quat(legacy[:, [4, 5, 6, 3]]))(times).as_matrix()
    return np.concatenate((positions, _quat(rotations)), axis=-1)




@pytest.fixture(scope="module")
def representation():
    import torch
    from motionbricks.data.native_mb_data import build_native_representation

    torch.set_num_threads(2)
    return build_native_representation()


@pytest.fixture(scope="module")
def adapter(representation):
    return CalibratedSomaG1Adapter(ROBOT_XML, SAMPLES / "soma_filtered/210531" / (CLIP + ".pkl"),
                                   SAMPLES / "robot_filtered/210531" / (CLIP + ".pkl"), representation=representation)


def actual_history(adapter):
    path = ROOT / "out/mb_sonic_e0_e3_20261018/e1_official_walk_v2/recording.jsonl"
    history = ExecutedHistory("000", body_names=adapter.kinematics.body_names)
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["kind"] != "policy_tick":
                continue
            for state in record["physics_states"]:
                contacts = tuple(dict(body_ids=tuple(c["body_ids"]), position_world=np.array(c["position_world"]),
                                      wrench_contact=np.array(c["wrench_contact"])) for c in state["contacts"])
                history.samples.append(dict(timestamp_s=state["time_s"], qpos=np.array(state["qpos"]),
                                             qvel=np.array(state["qvel"]), body_pos_w=np.array(state["body_positions_world"]),
                                             body_quat_w=np.array(state["body_quat_wxyz"]), contacts=contacts))
            if history.samples[-1]["timestamp_s"] >= .3:
                break
    return history


def test_pure_fk_matches_official_csv_and_native_xml():
    kin = G1Kinematics(ROBOT_XML)
    source = load_official_reference(ROOT.parent / "gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428", scene_id="000")
    q = source.qpos[::30]
    positions, rotations = kin.fk(q)
    np.testing.assert_allclose(positions[:, kin.output_ids], source.body_pos_w[::30], atol=3e-6, rtol=0)
    # This MjData is a separate, static test FK oracle, never a physical rollout.
    data = mujoco.MjData(kin.model)
    for i, frame in enumerate(q):
        data.qpos[:7] = frame[:7]
        for j, name in enumerate(kin.joint_ids):
            data.qpos[kin.model.jnt_qposadr[kin.joint_ids[name]]] = frame[7 + j]
        mujoco.mj_forward(kin.model, data)
        np.testing.assert_allclose(positions[i], data.xpos, atol=1e-12, rtol=0)
        np.testing.assert_allclose(rotations[i], data.xmat.reshape(-1, 3, 3), atol=1e-12, rtol=0)


def test_native_rest_frames_are_finite_and_right_handed(representation):
    s = representation.skeleton
    frames = native_rest_frames(np.asarray(s.neutral_joints), s.bone_order_names)
    assert frames.shape == (14, 3, 3)
    np.testing.assert_allclose(np.linalg.det(frames), 1, atol=1e-10, rtol=0)


@pytest.mark.parametrize("angles", [(0., 0., 0.), (.2, -.3, .4)])
def test_native_foot_material_frames_preserve_neutral_calibration_gauge(representation, angles):
    skeleton = representation.skeleton
    names = tuple(skeleton.bone_order_names)
    neutral = np.asarray(skeleton.neutral_joints)
    rotation = Rotation.from_euler("xyz", angles).as_matrix()
    globals_ = np.tile(rotation, (1, len(names), 1, 1))
    positions = (neutral @ rotation.T + [.4, .2, -.1])[None]
    material = native_limb_material_frames(globals_, neutral, names)
    frames = anatomical_frames(positions, names, limb_frames=material)
    rest = native_rest_frames(neutral, names)
    for side in ("Left", "Right"):
        index = SEGMENTS.index(side + "Foot")
        np.testing.assert_allclose(frames[0, index], rotation @ rest[index], atol=1e-12, rtol=0)
        np.testing.assert_allclose(frames[0, index] @ rest[index].T, rotation, atol=1e-12, rtol=0)


@pytest.mark.parametrize("side,bank_deg", [("Left", 10.), ("Right", -10.)])
def test_native_foot_bank_reaches_supported_g1_ankle_without_state_writes(adapter, monkeypatch, side, bank_deg):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    roots = np.repeat(source.root_positions_m[12:13], 2, axis=0)
    positions = np.repeat(source.world_positions_m[12:13], 2, axis=0)
    rotations = np.repeat(source.global_rot_mats[12:13].astype(float), 2, axis=0)
    foot, toe = (source.bone_names.index(side + name) for name in ("Foot", "ToeBase"))
    axis = adapter.neutral[toe] - adapter.neutral[foot]
    axis /= np.linalg.norm(axis)
    bank = Rotation.from_rotvec(axis * np.deg2rad(bank_deg)).as_matrix()
    # Banking around this axis leaves Foot and ToeBase positions unchanged.
    np.testing.assert_allclose(bank @ axis, axis, atol=1e-12, rtol=0)
    rotations[1, foot] = rotations[0, foot] @ bank
    for value in (roots, positions, rotations):
        value.flags.writeable = False
    model_initial = adapter.kinematics.model.qpos0.copy()

    def forbid_physics(*args, **kwargs):
        pytest.fail("Foot mapping must not allocate or step physical state")

    monkeypatch.setattr(mujoco, "MjData", forbid_physics)
    monkeypatch.setattr(mujoco, "mj_step", forbid_physics)
    monkeypatch.setattr(adapter, "limb_rotation_source", "markers")
    marker, _ = adapter._map_anatomical(roots, positions, source.bone_names,
        root_rotations=rotations[:, 0], global_rotations=rotations)
    np.testing.assert_array_equal(marker[0], marker[1])
    monkeypatch.setattr(adapter, "limb_rotation_source", "native_material_axes")
    mapped, _ = adapter._map_anatomical(roots, positions, source.bone_names,
        root_rotations=rotations[:, 0], global_rotations=rotations)
    ankle = [adapter.kinematics.column[adapter.kinematics.joint_ids[side.lower() + name]]
             for name in ("_ankle_pitch_joint", "_ankle_roll_joint")]
    delta_deg = np.rad2deg(mapped[1, ankle[1]] - mapped[0, ankle[1]])
    assert delta_deg == pytest.approx(bank_deg, abs=1.)
    unchanged = np.setdiff1d(np.arange(36), ankle)
    np.testing.assert_array_equal(mapped[0, unchanged], mapped[1, unchanged])
    assert (mapped[:, 7:] >= adapter.kinematics.limits[:, 0]).all()
    assert (mapped[:, 7:] <= adapter.kinematics.limits[:, 1]).all()
    _, body_rotations = adapter.kinematics.fk(mapped)
    body = adapter.kinematics.body_names.index(side.lower() + "_ankle_roll_link")
    foot_delta = Rotation.from_matrix(body_rotations[1, body] @ body_rotations[0, body].T)
    assert np.rad2deg(foot_delta.magnitude()) == pytest.approx(abs(bank_deg), abs=1.)
    np.testing.assert_array_equal(adapter.kinematics.model.qpos0, model_initial)


def test_calibration_has_actual_hashes_no_mirror_fit_or_optimizer(adapter):
    calibration = adapter.calibration
    assert calibration["clip"] == CLIP and not calibration["clip"].endswith("_M")
    assert calibration["source_soma_sha256"] == sha256(SAMPLES / "soma_filtered/210531" / (CLIP + ".pkl"))
    assert calibration["optimizer_updates"] == 0
    assert np.isfinite(list(calibration["segment_rotation_rmse_rad"].values())).all()
    assert len(calibration["identity"]) == 64
    np.testing.assert_allclose(np.linalg.det(adapter.offset), 1, atol=1e-10, rtol=0)


def test_saved_calibration_identity_roundtrip(adapter, representation, tmp_path):
    path = tmp_path / "calibration.json"
    adapter.save(path)
    loaded = CalibratedSomaG1Adapter.load(path, robot_model_path=ROBOT_XML, representation=representation)
    assert loaded.calibration == adapter.calibration
    np.testing.assert_array_equal(loaded.offset, adapter.offset)
    corrupted = json.loads(path.read_text())
    corrupted["root_z_offset_m"] += .1
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="identity mismatch"):
        CalibratedSomaG1Adapter.load(path, robot_model_path=ROBOT_XML, representation=representation)


def test_legacy_rotation_gauge_algebra_roundtrip_is_not_native_forward_validation(adapter):
    _, _, q = _load_pair(SAMPLES / "soma_filtered/210531" / (CLIP + "_M.pkl"),
                         SAMPLES / "robot_filtered/210531" / (CLIP + "_M.pkl"))
    q = q[::30]
    _, rotations = adapter.kinematics.fk(q)
    estimate = adapter.inverse(q[:, :3], rotations)
    reconstructed = adapter._map_rotations(estimate.root_positions_m, estimate.global_rot_mats)
    difference = (reconstructed[:, 7:] - q[:, 7:] + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(difference, 0., atol=2e-5, rtol=0)
    np.testing.assert_allclose(reconstructed[:, :3], q[:, :3], atol=1e-12, rtol=0)
    p, r = adapter.kinematics.fk(reconstructed)
    expected_p, expected_r = adapter.kinematics.fk(q)
    np.testing.assert_allclose(p[:, adapter.kinematics.output_ids], expected_p[:, adapter.kinematics.output_ids], atol=1e-5, rtol=0)
    assert Rotation.from_matrix((r @ expected_r.swapaxes(-1, -2)).reshape(-1, 3, 3)).magnitude().max() < 2e-5


def test_actual_physics_history_has_inferred_masks_measured_contacts_and_no_state_writes(adapter):
    history = actual_history(adapter)
    original = [sample["qpos"].copy() for sample in history.samples]
    result = adapter.estimate_history(history, end_time_s=.3)
    assert result.features.shape == (4, 366) and np.isfinite(result.features).all()
    assert result.provenance["kind"] == "calibrated_executed_g1"
    assert result.provenance["inferred_rotation_dof_count"] == 58
    assert sum(result.provenance["native_rotation_observed_rank_in_bone_order"]) == 32
    assert result.provenance["physical_state_writes"] == 0
    contacts = result.features[:, adapter.rep.indices["foot_contacts"]]
    assert np.isin(contacts, [0., 1.]).all() and contacts.any()
    times = np.array([sample["timestamp_s"] for sample in history.samples])
    pelvis = history.body_names.index("pelvis")
    actual_roots = np.stack([sample["body_pos_w"][pelvis] for sample in history.samples])
    native_roots = np.stack([np.interp(result.timestamps_s, times, actual_roots[:, i]) for i in range(3)], -1)
    native_roots = (native_roots - [0., 0., adapter.root_z_offset_m]) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(result.features[:, :3], native_roots, atol=2e-6, rtol=0)
    for before, sample in zip(original, history.samples):
        np.testing.assert_array_equal(sample["qpos"], before)
    with pytest.raises(ReferenceCoverageError, match="actual executed coverage"):
        adapter.estimate_history(history, end_time_s=.05)


def test_force_contacts_follow_actual_force_and_spatial_split(adapter):
    history = actual_history(adapter)
    sample = next(sample for sample in history.samples if adapter._contacts(sample, history.body_names).any())
    expected = np.zeros(4)
    for side_index, side in enumerate(("left", "right")):
        body = history.body_names.index(side + "_ankle_roll_link")
        rotation = Rotation.from_quat(sample["body_quat_w"][body][[1, 2, 3, 0]]).as_matrix()
        for contact in sample["contacts"]:
            if set(contact["body_ids"]) == {0, body} and contact["wrench_contact"][0] > 1:
                point = rotation.T @ (contact["position_world"] - sample["body_pos_w"][body])
                expected[2 * side_index + int(point[0] >= adapter.heel_toe_split_x_m[side])] = 1
    np.testing.assert_array_equal(adapter._contacts(sample, history.body_names), expected)


@pytest.mark.parametrize("suffix,joint_rmse", [("", .05251242179), ("_M", .05190319479)])
def test_pair_diagnostic_uses_actual_markers_and_retains_root_path_scale_difference(adapter, suffix, joint_rmse):
    report = adapter.diagnose_pair(SAMPLES / "soma_filtered/210531" / (CLIP + suffix + ".pkl"),
                                   SAMPLES / "robot_filtered/210531" / (CLIP + suffix + ".pkl"))
    assert report["held_out"] == bool(suffix) and report["status"] == "REHEARSAL_ONLY"
    assert report["forward_input"].startswith("actual_paired_markers")
    assert report["inverse_roundtrip_scope"].startswith("legacy_rotation_gauge_algebra")
    assert report["joint_rmse_rad"] == pytest.approx(joint_rmse, abs=1e-9)
    assert report["root_position_rms_m"] > 1.
    assert report["root_relative_body_position_rms_m"] < .02
    assert .77 < report["paired_root_xy_displacement_gain_diagnostic_only"] < .78
    assert not report["runtime_root_xy_scaling"]


def test_genuine_full_mb_proposal_matches_calibration_gauge_without_limit_clipping(adapter):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    original = source.global_rot_mats.copy()
    reference = adapter.retarget(source, start_time_s=0.)
    assert reference.qpos.shape == (81, 36)
    assert reference.adaptation["source_constraint_audit"]["status"] == "PASS"
    assert reference.adaptation["source_constraint_audit"]["sample_count"] == 52
    assert adapter.audit_joint_limits(source)["violation_count"] == 11
    np.testing.assert_array_equal(source.global_rot_mats, original)


def test_explicit_short_covered_proposal_resamples_without_clock_rebase(adapter):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    # A declared unit-test prefix, not an accepted full E3 motion or physical run.
    selected = replace(source, trusted_until_s=.5)
    reference = adapter.retarget(selected, start_time_s=0.)
    assert reference.qpos.shape == (26, 36)
    np.testing.assert_allclose(np.diff(reference.timestamps_s), .02, atol=1e-12, rtol=0)
    assert reference.trusted_until_s == reference.timestamps_s[-1] == .5
    assert reference.provenance["selected_trusted_until_s"] == .5
    assert reference.provenance["source_time_bounds_s"][-1] == 1.6
    assert reference.adaptation["physical_state_writes"] == 0




def test_quaternion_signs_remain_continuous_through_pi():
    matrices = Rotation.from_euler("z", [170., 179., 181., 190.], degrees=True).as_matrix()
    quats = _quat(matrices)
    assert ((quats[1:] * quats[:-1]).sum(-1) > 0).all()
    np.testing.assert_allclose(Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix(), matrices, atol=1e-15, rtol=0)


@pytest.mark.parametrize("start", [0., .18])
def test_uniform_one_second_packet_is_legal_and_preserves_failed_raw_tail(adapter, start):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    reference = adapter.retarget_packet(source, start_time_s=start)
    assert reference.qpos.shape == (51, 36)
    assert reference.timestamps_s[0] == start
    assert reference.trusted_until_s == pytest.approx(start + 1.)
    assert reference.latest_window_start_s == pytest.approx(start + .1)
    assert reference.sonic_window(start).qpos.shape == (10, 36)
    assert reference.sonic_window(start + .1).qpos.shape == (10, 36)
    with pytest.raises(ReferenceCoverageError):
        reference.sonic_window(start + .12)
    assert reference.adaptation["consumer_horizon_s"] == 1.
    assert reference.adaptation["scope"] == "uniform_all_commands_and_maps"
    audit = reference.adaptation["raw_source_joint_limit_audit"]
    assert audit["status"] == "FAIL" and audit["violation_count"] == 11
    assert audit["sample_count"] == 52 and audit["source_time_bounds_s"][-1] == 1.6
    assert source.trusted_until_s == 1.6


def test_fixed_packet_rejects_insufficient_guard_and_preserves_legacy_failure(adapter):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    with pytest.raises(ReferenceCoverageError, match="0.9s future plus 0.1s guard"):
        adapter.retarget_packet(replace(source, trusted_until_s=.9), start_time_s=0.)
    reference = adapter.retarget_packet(source, start_time_s=.3)
    assert reference.adaptation["source_constraint_audit"]["status"] == "PASS"
    assert reference.adaptation["raw_source_joint_limit_audit"]["status"] == "FAIL"


def test_public_audit_does_not_drop_tail_when_view_has_shorter_trust(adapter):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    full = adapter.audit_joint_limits(source)
    view = adapter.audit_joint_limits(replace(source, trusted_until_s=1.))
    assert full["violations"] == view["violations"]
    assert view["scope"] == "every_raw_native_sample; no_consumer_horizon_filter"


def test_short_packet_requires_explicit_short_tracker_contract_and_retains_raw_audit(adapter):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    with pytest.raises(ValueError):
        adapter.retarget_packet(source, start_time_s=.3, consumer_horizon_s=.6)
    reference = adapter.retarget_packet(source, start_time_s=.3, consumer_horizon_s=.6, required_future_s=.16)
    assert reference.qpos.shape == (31, 36)
    assert reference.adaptation["tracker_required_future_s"] == .16
    assert reference.adaptation["raw_source_joint_limit_audit"]["sample_count"] == 52
    assert source.trusted_until_s == 1.6
    assert not reference.adaptation["automatic_shortening"]
    with pytest.raises(ReferenceCoverageError):
        reference.sonic_window(.3)


def test_explicit_one_second_horizon_preserves_default_v1_contract(adapter):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    default = adapter.retarget_packet(source, start_time_s=0.)
    explicit = adapter.retarget_packet(source, start_time_s=0., consumer_horizon_s=1.)
    np.testing.assert_equal(default.adaptation, explicit.adaptation)
    assert explicit.provenance["consumer_contract"] == "motionbricks.mb_sonic.fixed_1s_consumer.v1"
    assert "scheduler_reserve_s" not in explicit.adaptation
    np.testing.assert_array_equal(default.qpos, explicit.qpos)


@pytest.mark.parametrize("start", [0., .005, .095, .1])
def test_explicit_reserved_packet_is_legal_and_keeps_full_failed_tail(adapter, start):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    original = source.global_rot_mats.copy()
    reference = adapter.retarget_packet(source, start_time_s=start, consumer_horizon_s=1.12)
    assert reference.qpos.shape == (57, 36)
    assert reference.timestamps_s[0] == start
    assert reference.trusted_until_s == pytest.approx(start + 1.12)
    np.testing.assert_allclose(np.diff(reference.timestamps_s), .02, atol=1e-12, rtol=0)
    assert reference.latest_window_start_s == pytest.approx(start + .22)
    assert reference.sonic_window(start + .22).qpos.shape == (10, 36)
    with pytest.raises(ReferenceCoverageError):
        reference.sonic_window(start + .24)
    assert reference.adaptation["consumer_contract"] == "motionbricks.mb_sonic.explicit_consumer_horizon.v2"
    assert reference.adaptation["consumer_horizon_s"] == 1.12
    assert reference.adaptation["replan_guard_s"] == .1
    assert reference.adaptation["scheduler_reserve_s"] == pytest.approx(.12)
    assert not reference.adaptation["automatic_shortening"]
    audit = reference.adaptation["raw_source_joint_limit_audit"]
    assert audit["status"] == "FAIL" and audit["violation_count"] == 11
    assert audit["sample_count"] == 52 and audit["source_time_bounds_s"][-1] == 1.6
    assert source.trusted_until_s == 1.6
    np.testing.assert_array_equal(source.global_rot_mats, original)


def test_reserved_packet_rejects_insufficient_coverage_and_preserves_legacy_failure(adapter):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    with pytest.raises(ReferenceCoverageError, match="Explicit 1.12s consumer packet"):
        adapter.retarget_packet(replace(source, trusted_until_s=1.1), start_time_s=0., consumer_horizon_s=1.12)
    reference = adapter.retarget_packet(source, start_time_s=.12, consumer_horizon_s=1.12)
    assert reference.adaptation["source_constraint_audit"]["status"] == "PASS"
    assert reference.adaptation["raw_source_joint_limit_audit"]["status"] == "FAIL"


@pytest.mark.parametrize("horizon", [.9, 0., np.nan, np.inf, 1.121])
def test_consumer_horizon_requires_finite_full_50hz_interval_contract(adapter, horizon):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    with pytest.raises(ValueError, match="Consumer horizon must"):
        adapter.retarget_packet(source, start_time_s=0., consumer_horizon_s=horizon)


def test_actual_seed_has_calibrated_hip_gauge_and_exact_legacy_root(adapter):
    from motionbricks.mb_sonic.mb_reference import decode_soma_reference

    with np.load(ROOT / "out/pfnn_mixed_20261015/final_acceptance_flat_v1/scene.npz", allow_pickle=False) as archive:
        features = archive["initial_history"][0].copy()
    source = decode_soma_reference(features, (np.arange(4) - 3) / 30, adapter.rep, scene_id="000",
                                   command_id="seed", trusted_until_s=0., provenance=dict(kind="source_seed"))
    old = adapter._map_rotations(source.root_positions_m, source.global_rot_mats)
    raw, _ = adapter._map_anatomical(source.root_positions_m, source.world_positions_m, source.bone_names,
                                    root_rotations=source.global_rot_mats[:, 0])
    np.testing.assert_array_equal(raw[:, :7], old[:, :7])
    np.testing.assert_allclose(old[-1, [9, 15]], [-1.43893567, -1.79181681], atol=2e-6, rtol=0)
    np.testing.assert_allclose(raw[-1, [9, 15]], [.131196, -.18034694], atol=2e-6, rtol=0)
    reference = adapter.retarget(source, start_time_s=-.1)
    assert reference.adaptation["source_constraint_audit"]["status"] == "PASS"
    assert reference.adaptation["dense_constraint_audit"]["status"] == "PASS"
    np.testing.assert_array_equal(reference.qpos[:, :7], legacy_root_at_times(adapter, source, reference.timestamps_s))


def test_unobserved_raw_global_twist_does_not_reenter_runtime_mapping(adapter):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    original_positions = source.world_positions_m.copy()
    perturbed = source.global_rot_mats.copy()
    perturbed[:, 1:] = perturbed[:, 1:] @ Rotation.from_euler("xyz", [.8, -.5, .2]).as_matrix()
    # Deliberate unit-test gauge perturbation, not a generated or physical frame.
    changed = replace(source, global_rot_mats=perturbed)
    before = adapter.retarget(source, start_time_s=0.)
    after = adapter.retarget(changed, start_time_s=0.)
    np.testing.assert_array_equal(before.qpos, after.qpos)
    np.testing.assert_array_equal(before.qvel, after.qvel)
    np.testing.assert_array_equal(source.world_positions_m, original_positions)


@pytest.mark.parametrize("path", sorted(V3_NUMERIC.glob("*.npz")), ids=lambda path: path.stem[:8])
def test_all_saved_v3_raw_sources_use_real_markers_and_pass_full_constraints(adapter, path):
    source = load_numeric_source(path)
    assert source is not None
    original = source.global_rot_mats.copy()
    reference = adapter.retarget(source, start_time_s=source.timestamps_s[0])
    audit = reference.adaptation["source_constraint_audit"]
    assert audit["status"] == "PASS" and audit["sample_count"] == 52
    assert audit["minimum_position_margin_rad"] > .16
    assert audit["maximum_velocity_ratio"] < 1.
    assert reference.adaptation["dense_constraint_audit"]["status"] == "PASS"
    assert reference.adaptation["model_mapping_residual"]["rms_rad"] < .12
    assert reference.adaptation["physical_state_writes"] == 0
    np.testing.assert_array_equal(reference.qpos[:, :7], legacy_root_at_times(adapter, source, reference.timestamps_s))
    np.testing.assert_array_equal(source.global_rot_mats, original)


@pytest.mark.parametrize("proposal", ["mb_proposal_v1", "mb_proposal_compiled_v2", "mb_proposal_full_plan_v1"])
def test_all_preserved_mb_proposal_frames_and_velocities_are_checked(adapter, proposal):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018" / proposal / "proposal.npz")
    first = adapter.retarget(source, start_time_s=source.timestamps_s[0])
    second = adapter.retarget(source, start_time_s=source.timestamps_s[0])
    np.testing.assert_array_equal(first.qpos, second.qpos)
    assert first.adaptation["source_constraint_audit"]["sample_count"] == 52
    assert first.adaptation["source_constraint_audit"]["status"] == "PASS"
    assert first.adaptation["dense_constraint_audit"]["status"] == "PASS"


def test_genuine_failed_f07d_packet_retains_legacy_ankle_failure(adapter):
    path = V3_NUMERIC / "f07d781c2545f8cfb0685d64067d941f175ff62235b9e55680cfa8289bb71a3d.npz"
    source = load_numeric_source(path)
    old = adapter._map_rotations(source.root_positions_m, source.global_rot_mats)
    old_ankle = np.interp(1.96, source.timestamps_s, old[:, 18])
    assert old_ankle < adapter.kinematics.limits[11, 0]
    packet = adapter.retarget_packet(source, start_time_s=1.16, consumer_horizon_s=1.12)
    assert packet.qpos.shape == (57, 36)
    assert packet.adaptation["source_constraint_audit"]["status"] == "PASS"
    assert packet.adaptation["raw_source_joint_limit_audit"]["source_gauge"].startswith("legacy_raw_global")


def test_true_velocity_violation_is_rejected_without_time_stretch(adapter):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    compressed = replace(source, timestamps_s=source.timestamps_s * .02, trusted_until_s=source.trusted_until_s * .02)
    with pytest.raises(ValueError, match="'kind': 'velocity'"):
        adapter.retarget(compressed, start_time_s=0.)


def test_true_anatomical_position_violation_is_rejected_without_clipping(adapter):
    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    positions = source.world_positions_m.copy()
    foot, toe = (source.bone_names.index(name) for name in ("LeftFoot", "LeftToeBase"))
    positions[:, toe] = 2 * positions[:, foot] - positions[:, toe]
    with pytest.raises(ValueError, match="'kind': 'position'"):
        adapter.retarget(replace(source, world_positions_m=positions), start_time_s=0.)


def test_authorized_constrained_reference_repairs_real_waist_failure_and_retains_raw_audit(adapter):
    from motionbricks.mb_sonic.constrained_reference import ConstrainedReferenceSolver

    path = ROOT / "out/mb_sonic_e0_e3_20261018/wasd_v13/episode_0003_numeric/e44758570cbf01c506a803d51339bf4b843fd595a5f5568d9d06c35b64a655ac.npz"
    source = load_numeric_source(path)
    original = source.raw_features.copy()
    with pytest.raises(ValueError, match="waist_pitch_joint"):
        adapter.retarget_packet(source, start_time_s=54.58, consumer_horizon_s=1.3)
    reference = adapter.retarget_packet(source, start_time_s=54.58, consumer_horizon_s=1.3,
        constraints=ConstrainedReferenceSolver(adapter.kinematics))
    assert len(reference.timestamps_s) == 66
    assert reference.adaptation["source_constraint_audit"]["status"] == "FAIL"
    assert reference.adaptation["dense_constraint_audit"]["status"] == "PASS"
    assert reference.adaptation["reference_constraints"]["root_exact"]
    assert reference.adaptation["reference_constraints"]["qp_solves"] > 0
    np.testing.assert_array_equal(reference.qpos[:, :7], legacy_root_at_times(adapter, source, reference.timestamps_s))
    np.testing.assert_array_equal(source.raw_features, original)


def test_authorized_constraints_do_not_change_a_healthy_dense_reference(adapter):
    from motionbricks.mb_sonic.constrained_reference import ConstrainedReferenceSolver

    source = load_soma_reference(ROOT / "out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz")
    original = adapter.retarget_packet(source, start_time_s=0., consumer_horizon_s=1.3)
    constrained = adapter.retarget_packet(source, start_time_s=0., consumer_horizon_s=1.3,
        constraints=ConstrainedReferenceSolver(adapter.kinematics))
    assert constrained.adaptation["reference_constraints"]["qp_solves"] == 0
    for name in ("timestamps_s", "qpos", "qvel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        np.testing.assert_array_equal(getattr(original, name), getattr(constrained, name))
