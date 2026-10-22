"""Root-only scores do not conflate heading, height or context errors."""

import math
from types import SimpleNamespace

import pytest
import torch

from scripts.review_parc_root_terrain import root_inputs, root_metrics


def root():
    values = torch.zeros(1, 64, 5)
    values[..., 3] = 1
    return values


def test_root_height_and_horizontal_errors_are_separate():
    source, pred = root(), root()
    pred[..., 0] = .3
    pred[..., 1] = .4
    metrics = root_metrics(pred, source, 30)
    assert metrics["root_translation_error_m"] == pytest.approx(.5)
    assert metrics["root_xz_error_m"] == pytest.approx(.3)
    assert metrics["root_height_error_m"] == pytest.approx(.4)
    assert metrics["root_velocity_error_mps"] == 0


def test_context_and_endpoint_are_not_interior_scores():
    source, pred = root(), root()
    pred[:, :4, :3] = 100
    pred[:, 63, :3] = -100
    assert all(value == 0 for value in root_metrics(pred, source, 30).values())


def test_heading_error_wraps_at_pi():
    source, pred = root(), root()
    source[..., 3:] = torch.tensor([math.cos(math.pi - .1), math.sin(math.pi - .1)])
    pred[..., 3:] = torch.tensor([math.cos(-math.pi + .1), math.sin(-math.pi + .1)])
    assert root_metrics(pred, source, 30)["root_heading_error_rad"] == pytest.approx(.2)


def test_nonfinite_prediction_is_rejected():
    source, pred = root(), root()
    pred[:, 6, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        root_metrics(pred, source, 30)


def test_root_inputs_use_native_normalization_and_only_endpoint_conditions():
    from test_native_mb_stage_training import _motion_and_representation
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    rep, batch = _motion_and_representation(frame_count=65)
    raw = rep.unnormalize(batch["motion"])
    raw[..., 0] += torch.arange(65) / 30
    local_rep = rep.dual_rep.local_motion_rep
    model = SimpleNamespace(
        motion_rep=rep, global_motion_rep=rep, local_motion_rep=local_rep,
        args={"min_tokens": 6, "max_tokens": 16, "global_root_feature": "root",
              "local_root_feature": "root", "local_pose_feature": "joint_positions_and_rotations_and_hip_height"},
        backbone_net=SimpleNamespace(MASKED_NUM_TOKENS=18),
    )
    inputs, source = root_inputs(model, raw)
    canonical = rep.change_first_heading(raw, 0.0, False, False)
    normalized = rep.normalize(canonical)
    local = rep.dual_rep.global_to_local(normalized, True, True, lengths=torch.tensor([65]))
    expected = {
        "global_root_values": extract_feature_from_motion_rep(normalized, rep, "root"),
        "local_root_values": extract_feature_from_motion_rep(local, local_rep, "root"),
        "poses": extract_feature_from_motion_rep(local, local_rep, model.args["local_pose_feature"]),
    }
    for key, full in expected.items():
        torch.testing.assert_close(inputs[key][:, [3, 7]], full[:, [3, 63]])
        assert torch.count_nonzero(inputs[key][:, [0, 1, 2, 4, 5, 6]]) == 0
    for key in ("has_global_root_values", "has_local_root_values", "has_poses"):
        assert inputs[key].nonzero()[:, 1].tolist() == [3, 7]
    torch.testing.assert_close(source, extract_feature_from_motion_rep(canonical[:, :64], rep, "root"))
    assert inputs["num_tokens"].item() == 18
    assert inputs["allowed_pred_num_tokens"].nonzero()[:, 1].tolist() == [10]
