import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from motionbricks.data.bones_seed import (
    BonesMotionDataset, build_representation, calibration_crop, estimate_contacts, official_sole_offsets,
    recording_group, resample_rotations_root, split_for_group,
    validate_planar_posture,
)


def test_recordings_keep_actors_takes_and_mirrors_together():
    names = ["neutral_walk_180_R_003__A121", "neutral_walk_180_R_002__A122_M"]
    assert recording_group(names[0]) == recording_group(names[1])
    assert split_for_group(recording_group(names[0])) == split_for_group(recording_group(names[1]))


def test_actual_timestamps_and_constant_rotations():
    frames, source_fps = 240, 120.00480019200768
    rotations = np.broadcast_to(np.eye(3), (frames, 2, 3, 3)).copy()
    time = np.arange(frames) / source_fps
    root = np.stack([time, np.ones_like(time), 2 * time], -1)
    sampled, sampled_root = resample_rotations_root(rotations, root, source_fps)
    assert len(sampled) == int(np.floor(time[-1] * 60)) + 1
    np.testing.assert_allclose(sampled, np.broadcast_to(np.eye(3), sampled.shape), atol=1e-6)
    np.testing.assert_allclose(sampled_root[10:-10, 0], np.arange(len(sampled))[10:-10] / 60, atol=2e-5)
    np.testing.assert_allclose(sampled_root[:, 1], 1, atol=1e-6)


def test_rotation_wraparound_remains_a_rotation():
    angles = np.linspace(170, 210, 240)
    rotations = Rotation.from_euler("y", angles[:, None], degrees=True).as_matrix()[:, None]
    sampled, _ = resample_rotations_root(rotations, np.zeros((240, 3)), 120)
    np.testing.assert_allclose(np.linalg.det(sampled), 1, atol=1e-6)
    delta = Rotation.from_matrix(sampled[1:, 0] @ sampled[:-1, 0].transpose(0, 2, 1)).magnitude()
    assert delta.max() < 0.02


def test_static_support_and_airborne_clearance():
    sole = torch.zeros(120, 4, 3)
    sole[:, 2:, 1] = 0.2
    contacts, confidence, plane, _ = estimate_contacts(sole)
    assert abs(plane) < 1e-6
    assert contacts[10:-10, :2].all()
    assert not contacts[:, 2:].any()
    assert confidence[10:-10].all()


def test_no_support_is_rejected():
    sole = torch.zeros(120, 4, 3)
    sole[:, :, 0] = torch.arange(120)[:, None] / 60
    with pytest.raises(ValueError, match="stationary support"):
        estimate_contacts(sole)


def test_soma_contract_and_sole_geometry():
    representation = build_representation()
    skeleton = representation.skeleton
    assert skeleton.nbjoints == 30
    assert skeleton.bone_order_names[0] == "Hips"
    assert skeleton.foot_joint_names == ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]
    assert len(representation.indices["all"]) == 366
    assert len(representation.dual_rep.indices["all"]) == 370
    offsets, digest = official_sole_offsets(skeleton)
    assert offsets.shape == (4, 3) and len(digest) == 64
    assert (offsets[:, 1] < 0).all()


def test_upright_geometry_rejects_wrong_support_and_inversion():
    skeleton = build_representation().skeleton
    offsets, _ = official_sole_offsets(skeleton)
    reference_height = -(skeleton.neutral_joints[skeleton.foot_joint_idx] + offsets)[:, 1].median()
    positions = skeleton.neutral_joints[None].expand(180, -1, -1).clone()
    positions[..., 1] += reference_height
    assert validate_planar_posture(positions, skeleton, 0.0, offsets)["hip_height_median_m"] > 0.8
    with pytest.raises(ValueError, match="support plane"):
        validate_planar_posture(positions, skeleton, 2.0, offsets)
    positions[:, skeleton.bone_index["Head"], 1] -= 1.0
    with pytest.raises(ValueError, match="Inverted"):
        validate_planar_posture(positions, skeleton, 0.0, offsets)


def test_locomotion_filter_excludes_handstands_and_interactions():
    import importlib.util
    from pathlib import Path
    project = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("prepare_bones_filter", project / "scripts/prepare_bones_locomotion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.behavior({}, "turn_handstand_270_R_002__A305") is None
    assert module.behavior({}, "walk_forward_grab_relax_002__A006") is None
    assert module.behavior({}, "jog_high_knees_R_stop_003__A359") is None
    assert module.behavior({}, "idle_right_R_point_dir_horizontal_002__A476") is None
    assert module.behavior({}, "walk_hands_on_back_loop_001__A093") is None
    assert module.behavior({}, "neutral_walk_180_R_003__A121") == "walk"
    assert module.behavior({}, "walk_backward_stop_002__A030") == "backward"


def test_worker_statistics_do_not_alias_model_buffers(tmp_path):
    rep = build_representation()
    from motionbricks.motionlib.core.utils.stats import Stats
    stats = Stats()
    stats.register_from_tensors(torch.zeros(366), torch.ones(366))
    np.savez(tmp_path / "clip.npz", global_features=np.zeros((140, 366), dtype=np.float32),
             contact_confidence=np.ones((140, 4), dtype=np.float32))
    dataset = BonesMotionDataset(tmp_path, [{"shard": "clip.npz"}], stats)
    stats.mean.fill_(100)
    assert dataset[0]["motion"].abs().max() == 0
    assert dataset.stats.mean.data_ptr() != stats.mean.data_ptr()


def test_boundary_tpose_is_removed_but_locomotion_preserved():
    skeleton = build_representation().skeleton
    neutral = skeleton.neutral_joints[None].expand(300, -1, -1).clone()
    neutral[..., 1] += 1
    # After calibration, lower the wrists and keep the root trajectory unchanged.
    neutral[45:, skeleton.bone_index["LeftHand"], 1] -= 0.5
    neutral[45:, skeleton.bone_index["RightHand"], 1] -= 0.5
    start, end = calibration_crop(neutral, skeleton)
    assert start >= 45 and end == 300


def test_no_future_body_pose_or_ground_truth_height_conditioning():
    from types import SimpleNamespace
    from motionbricks.data.bones_models import BonesRootModel
    dummy = SimpleNamespace(backbone_net=SimpleNamespace(get_num_frames_per_token=lambda: 4))
    body = torch.randn(2, 128, 268)
    values, mask = BonesRootModel._sample_the_conditions(dummy, body, 128)
    assert mask[:, :4].all() and not mask[:, 4:].any()
    assert not values[:, 4:].any()
    root = torch.randn(2, 128, 5)
    values, _ = BonesRootModel._sample_the_conditions(dummy, root, 128)
    torch.testing.assert_close(values[:, 4:, 1], root[:, 3:4, 1].expand(-1, 4))


def test_rotation_error_includes_half_turn():
    from motionbricks.data.bones_evaluation import rotation_error_degrees
    identity = torch.eye(3).expand(3, -1, -1)
    rotations = torch.from_numpy(Rotation.from_euler("y", [[0], [90], [180]], degrees=True).as_matrix()).float()
    torch.testing.assert_close(rotation_error_degrees(rotations, identity), torch.tensor([0., 90., 180.]))


def test_root_drift_and_body_pose_metrics_are_separate():
    from motionbricks.data.bones_evaluation import reconstruction_metrics
    from motionbricks.motionlib.core.utils.stats import Stats
    from kimodo.exports.motion_io import complete_motion_dict
    stats = Stats()
    stats.register_from_tensors(torch.zeros(365), torch.ones(365))
    rep = build_representation()
    local = rep.dual_rep.local_motion_rep
    local.stats = stats
    frames = 128
    rotations = torch.eye(3).expand(frames, 30, 3, 3).clone()
    root = torch.zeros(frames, 3)
    root[..., 1] = 1
    motion = complete_motion_dict(rotations, root, rep.skeleton, 60)
    target = local({"posed_joints": motion["posed_joints"],
                    "local_joint_rots": motion["local_rot_mats"],
                    "global_joint_rots": motion["global_rot_mats"],
                    "translation": root, "foot_contacts": torch.ones(frames, 4)}, to_normalize=True)[None]
    prediction = target.clone()
    prediction[..., local.indices["local_root_vel"][0]] += 0.2
    offsets, _ = official_sole_offsets(rep.skeleton)
    metrics, _, _ = reconstruction_metrics(prediction, target, local, offsets)
    assert metrics["root_xz_error_m"] > 0.1
    assert metrics["body_root_relative_mpjpe_m"] < 1e-5
    assert metrics["oracle_root_mpjpe_m"] < 1e-5


def test_real_bones_bvh_roundtrip_when_fixture_is_available():
    import importlib.util
    import os
    from pathlib import Path
    from kimodo.skeleton.registry import build_skeleton
    project = Path(__file__).resolve().parents[1]
    fixture = Path(os.environ.get("MOTIONBRICKS_BONES_BVH_FIXTURE",
                                  project / "out/bones_pilot_source/neutral_walk_180_R_003__A121.bvh"))
    if not fixture.exists():
        pytest.skip("Local BONES integration fixture is not present")
    spec = importlib.util.spec_from_file_location("prepare_bones", project / "scripts/prepare_bones_locomotion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rep = build_representation()
    offsets, _ = official_sole_offsets(rep.skeleton)
    arrays, report = module.convert({"path": str(fixture)}, build_skeleton(77), rep, offsets)
    assert arrays["global_features"].shape[1] == 366
    assert report["body_subset_fk_error_m"] < 0.002
    assert report["feature_roundtrip_error_m"] < 0.002
