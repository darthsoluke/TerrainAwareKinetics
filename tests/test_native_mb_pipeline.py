"""Input and history contracts for the unmodified native inference pipeline."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionbricks.data.native_mb_data import build_native_representation
from scripts.evaluate_native_mb_pipeline import command_constraints, rollout


def _fixture():
    rep = build_native_representation()
    local_rotations = torch.eye(3).expand(4, rep.skeleton.nbjoints, 3, 3).clone()
    root = torch.zeros(4, 3)
    root[:, 0] = 2.0
    root[:, 1] = 1.0
    raw = rep({"local_joint_rots": local_rotations, "translation": root,
               "foot_contacts": torch.ones(4, 4)}, to_normalize=False,
              lengths=torch.tensor(4))[None]
    # Feature extraction canonicalizes the source origin. Exercise a translated
    # runtime history explicitly, while preserving its root-relative body fields.
    raw[:, :, 0] += 2.0
    infer = SimpleNamespace(motion_rep=rep, global_motion_rep=rep,
        local_motion_rep=rep.dual_rep.local_motion_rep,
        EXTERNAL_POSE_FEATURE_MODE="joint_positions_and_rotations",
        _args={"min_tokens": 6, "max_tokens": 16})
    return infer, raw


def test_commands_supply_no_future_body_or_future_local_velocity():
    infer, raw = _fixture()
    before = raw.clone()
    inputs = command_constraints(infer, raw, (1.0, 0.0), 0.0)
    assert inputs["has_local_poses"].tolist() == [[True] * 4 + [False] * 4]
    assert inputs["has_local_root_values"].tolist() == [[True] * 3 + [False] * 5]
    assert inputs["local_poses"][:, 4:].count_nonzero() == 0
    assert inputs["local_root_values"][:, 4:].count_nonzero() == 0
    assert inputs["num_tokens"] is None
    assert inputs["allowed_pred_num_tokens"].sum() == 1
    assert inputs["allowed_pred_num_tokens"][0, 2] == 1
    torch.testing.assert_close(raw, before, rtol=0, atol=0)


def test_goal_times_are_relative_to_last_context_frame():
    infer, raw = _fixture()
    inputs = command_constraints(infer, raw, (1.0, -0.5), 0.0)
    expected_offsets = torch.arange(25, 29) / 30
    torch.testing.assert_close(inputs["global_root_values"][0, 4:, 0], 2 + expected_offsets)
    torch.testing.assert_close(inputs["global_root_values"][0, 4:, 2], -0.5 * expected_offsets)
    assert torch.all(inputs["global_root_values"][0, 4:, 1] == 1)


@pytest.mark.parametrize("velocity,heading,tokens", [((float("nan"), 0), 0, 8),
    ((0, 0), float("inf"), 8), ((0, 0), 0, 5), ((0, 0, 0), 0, 8)])
def test_invalid_commands_rejected(velocity, heading, tokens):
    infer, raw = _fixture()
    with pytest.raises(ValueError):
        command_constraints(infer, raw, velocity, heading, tokens)


def test_rollout_uses_generated_history_and_never_replaces_generated_roots():
    infer, raw = _fixture()
    calls = []

    def predict(**kwargs):
        calls.append(kwargs)
        assert not kwargs["has_local_poses"][:, 4:].any()
        assert kwargs["config"]["final_root_pred_mode"] == "from_pose_module"
        assert not kwargs["config"]["smooth_root_traj"]
        # Deliberately ignore command goals to detect hidden root replacement.
        prediction = raw[:, :1].repeat(1, 64, 1)
        prediction[:, :, 0] = float(len(calls) + 10)
        return prediction, torch.tensor([8])

    infer.predict = predict
    features, _, plans, summary = rollout(infer, raw, plans_per_command=1)
    assert len(calls) == 5
    assert summary["generated_frames"] == 40
    assert len(plans) == 5
    torch.testing.assert_close(features[:, :4], raw, rtol=0, atol=0)
    for index, call in enumerate(calls):
        if index:
            assert torch.all(call["global_root_values"][:, :4, 0] == float(index + 10))
        assert torch.all(features[:, 4 + index * 8:12 + index * 8, 0] == float(index + 11))


def test_rollout_rejects_nonfinite_prediction():
    infer, raw = _fixture()
    infer.predict = lambda **kwargs: (torch.full((1, 64, raw.shape[-1]), float("nan")), torch.tensor([8]))
    with pytest.raises(ValueError, match="nonfinite"):
        rollout(infer, raw, plans_per_command=1)
