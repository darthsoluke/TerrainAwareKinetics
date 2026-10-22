"""Causal, Root-only PARC height review contract tests."""

from pathlib import Path
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from motionbricks.data.native_mb_online_conditions import (
    ONLINE_PROTOCOL,
    enable_online_conditioning,
    sparse_planar_conditions,
)
from motionbricks.data.native_mb_models import build_native_model
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.review_parc_root_height import (
    CONTEXT_FRAMES,
    NUM_TOKENS,
    _height_summary,
    _planar_goal,
    build_current_terrain_observation,
    load_input_bundle,
    prepare_root_inputs,
    run_root_forecast,
    stats_digest,
    tensor_digest,
)
from test_native_mb_stage_training import _motion_and_representation
from test_native_mb_terrain_integration import small_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _raw_history_and_model():
    representation, batch = _motion_and_representation(frame_count=25)
    history = representation.unnormalize(batch["motion"][:, :4]).clone()
    # Exercise XZ-only recentering and nonzero heading without changing height.
    history[..., 0] += 2.3
    history[..., 2] -= 1.7
    angle = torch.tensor(0.63)
    history = representation.change_first_heading(history, angle, False, False)
    config = enable_online_conditioning(small_config("root"))
    model = build_native_model("root", representation, config).eval()
    return representation, history, model


def _scene():
    return {
        "heightfield_y_m": torch.zeros(121, 121),
        "min_point_world_xyz_m": torch.tensor([-6.0, 0.0, -6.0]),
        "dx_m": torch.tensor(0.1),
    }


def test_planar_goal_rotates_local_endpoint_and_keeps_heading_and_height_out():
    roots = torch.zeros(1, 4, 5)
    roots[:, :, 1] = 1.4
    roots[:, :, 0] = 3.0
    roots[:, :, 2] = -2.0
    roots[:, :, 3] = torch.cos(torch.tensor(0.7))
    roots[:, :, 4] = torch.sin(torch.tensor(0.7))
    goal = _planar_goal(roots, (0.0, 1.6))
    torch.testing.assert_close(goal[:, 2:4], roots[:, -1, 3:5])
    expected_xz = torch.stack((3.0 + 1.6 * torch.sin(torch.tensor(0.7)),
                               -2.0 + 1.6 * torch.cos(torch.tensor(0.7)))).reshape(1, 2)
    torch.testing.assert_close(goal[:, :2], expected_xz)
    roots[..., 1] = 99.0
    torch.testing.assert_close(_planar_goal(roots, (0.0, 1.6)), goal)


def test_prepared_inputs_match_online_masks_and_predict_normalization():
    representation, history, model = _raw_history_and_model()
    prepared = prepare_root_inputs(model, history, (0.0, 1.6))
    inputs = prepared["inputs"]
    mask = inputs["global_root_component_mask"]
    assert mask[:, :4].all()
    assert mask[:, 7, [0, 2, 3, 4]].all()
    assert not mask[:, 4:7].any()
    assert not mask[:, 7, 1].any()
    assert inputs["has_local_root_values"][0].tolist() == [True, True, True, False, False, False, False, False]
    assert inputs["has_poses"][0].tolist() == [True, True, True, True, False, False, False, False]
    assert inputs["num_tokens"].tolist() == [[NUM_TOKENS]]
    assert inputs["poses"].shape[-1] == 268
    assert torch.isfinite(inputs["poses"]).all()

    expected_xz = prepared["history_root"][:, :1, [0, 2]]
    torch.testing.assert_close(prepared["initial_xz"], expected_xz)
    history_root = extract_feature_from_motion_rep(history, model.global_motion_rep, "root")
    local = representation.dual_rep.global_to_local(
        history, is_normalized=False, to_normalize=False, lengths=torch.tensor([4]))
    local_root = extract_feature_from_motion_rep(local, model.local_motion_rep, "root")
    external_pose = extract_feature_from_motion_rep(
        local, model.local_motion_rep, "joint_positions_and_rotations")
    goal = _planar_goal(history_root, (0.0, 1.6))
    conditions = sparse_planar_conditions(history_root, local_root, external_pose, goal)

    component = conditions["global_root_component_mask"]
    recentered = conditions["global_root_values"].clone()
    xz_indices = torch.as_tensor(model.global_motion_rep.indices["global_root_pos_2d"])
    recentered[..., xz_indices] -= recentered[:, :1, xz_indices].clone()
    expected_global = model.global_motion_rep.normalize(recentered)
    expected_global = torch.where(component, expected_global, 0.0)
    torch.testing.assert_close(inputs["global_root_values"], expected_global)

    local_mask = conditions["has_local_root_values"]
    expected_local = model.local_motion_rep.normalize(conditions["local_root_values"])
    expected_local = torch.where(local_mask[..., None], expected_local, 0.0)
    torch.testing.assert_close(inputs["local_root_values"], expected_local)

    pose_mask = conditions["has_local_poses"]
    pose_indices = torch.as_tensor(extract_feature_from_motion_rep(
        torch.zeros((1, 1, len(model.local_motion_rep.indices["all"]))),
        model.local_motion_rep, "joint_positions_and_rotations_and_hip_height", fetch_feat_idx=True,
    ), dtype=torch.long)
    internal_pose = torch.cat((recentered[..., 1:2], conditions["local_poses"]), dim=-1)
    mean = model.local_motion_rep.stats.mean[pose_indices]
    std = model.local_motion_rep.stats.std[pose_indices]
    expected_pose = (internal_pose - mean[None, None]) / torch.sqrt(std[None, None] ** 2 + 1e-5)
    expected_pose = torch.where(pose_mask[..., None], expected_pose, 0.0)
    torch.testing.assert_close(inputs["poses"], expected_pose)

    # The condition is recentered around history frame zero, but height and yaw are not transformed.
    goal = prepared["goal"]
    torch.testing.assert_close(goal[:, 2:4], prepared["history_root"][:, -1, 3:5])
    assert float(prepared["history_root"][:, -1, 1]) == pytest.approx(float(history[0, -1, 1]))
    assert len(prepared["input_sha256"]) == 64
    assert pose_indices.numel() == inputs["poses"].shape[-1]


def test_terrain_observation_is_current_and_changes_only_with_direction():
    _, history, model = _raw_history_and_model()
    scene = _scene()
    idle = prepare_root_inputs(model, history, (0.0, 0.0))
    left = prepare_root_inputs(model, history, (-0.8, 0.0))
    idle_obs, idle_direction = build_current_terrain_observation(idle["history_root"], idle["goal"], scene)
    left_obs, left_direction = build_current_terrain_observation(left["history_root"], left["goal"], scene)
    assert torch.count_nonzero(idle_direction) == 0
    assert left_direction[0, 0].item() == pytest.approx(-0.8)
    assert idle_obs.points.shape == (1, 2, 9, 9, 3)
    assert idle_obs.valid.shape == (1, 2, 9, 9)
    torch.testing.assert_close(idle_obs.points[:, 0], left_obs.points[:, 0])
    assert not torch.equal(idle_obs.points[:, 1], left_obs.points[:, 1])
    assert idle_obs.valid.all() and left_obs.valid.all()


def test_direct_root_forecast_is_eight_tokens_and_reports_non_support_lookups():
    _, history, model = _raw_history_and_model()
    scene = _scene()
    prepared = prepare_root_inputs(model, history, (0.0, 1.6))
    observation, _ = build_current_terrain_observation(prepared["history_root"], prepared["goal"], scene)
    result = run_root_forecast(model, prepared, observation, scene)
    assert result["pred_num_tokens"] == NUM_TOKENS
    assert len(result["root_world_xyz_heading_cos_sin"]) == NUM_TOKENS * 4
    assert len(result["terrain_at_predicted_root_points_diagnostic_only"]) == NUM_TOKENS * 4 - CONTEXT_FRAMES
    assert "no yaw rotation or final-root replacement" in result["postprocessing"]
    assert result["terrain_lookup_interpretation"].startswith("known heightfield values only")
    assert result["height"]["first_published_8_future_frames"][
        "prediction_frame_indices_inclusive"] == [4, 11]
    assert result["height"]["forecast_tail_8_future_frames"][
        "prediction_frame_indices_inclusive"] == [24, 31]
    assert all(row["known_heightfield_cell"] for row in
               result["terrain_at_predicted_root_points_diagnostic_only"])


def test_bundle_reader_binds_only_seed_scene_stats_and_rejects_future_references(tmp_path):
    history = torch.zeros(1, 4, 366)
    history[..., 3] = 1.0
    scene = _scene()
    mean, std = torch.zeros(366), torch.ones(366)
    bundle = {
        "schema": "motionbricks.parc_online_soma_bundle.v1",
        "protocol": ONLINE_PROTOCOL,
        "quality_approved": False,
        "post_processing": None,
        "metadata": {
            "parc_only": True,
            "root_checkpoint_sha256": "root-hash",
            "dataset_manifest_sha256": "manifest-hash",
            "pose_root_dependency": {
                "checkpoint_sha256": "root-hash",
                "frozen": True,
                "state_digest": "same-state",
                "end_state_digest": "same-state",
            },
        },
        "seed_provenance": {"split": "validation", "shard": "seed.npz", "start": 12, "frames": 4},
        "initial_history": history,
        "scene": scene,
        "stats_mean": mean,
        "stats_std": std,
        "stats_tensor_sha256": stats_digest(mean, std),
    }
    path = tmp_path / "portable.pt"
    torch.save(bundle, path)
    loaded = load_input_bundle(path)
    assert loaded["history"].shape == (1, 4, 366)
    assert loaded["seed_provenance"]["split"] == "validation"
    assert loaded["stats_tensor_sha256"] == stats_digest(mean, std)
    assert loaded["scene_tensor_sha256"] == tensor_digest(scene)

    bundle["reference_clips"] = {"future": torch.zeros(1, 64, 366)}
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="without future references"):
        load_input_bundle(path)


def test_height_summary_reports_future_delta_and_both_eight_frame_blocks():
    root = torch.zeros(32, 5)
    root[:, 1] = torch.linspace(1.0, 2.0, 32)
    summary = _height_summary(root, current_y=1.2)
    assert summary["future_prediction_frame_indices_inclusive"] == [4, 31]
    assert summary["first_published_8_future_frames"]["prediction_frame_indices_inclusive"] == [4, 11]
    assert summary["forecast_tail_8_future_frames"]["prediction_frame_indices_inclusive"] == [24, 31]
