import numpy as np
import pytest
import torch

from scripts.evaluate_bones_control import command_at, distribution, free_metrics, sole_vertices
from test_bones_control import models, source_motion


def test_command_timeline_covers_all_sixty_seconds_and_real_turns():
    assert {command_at(frame / 60)[0] for frame in range(3600)} == {
        "idle", "walk", "run", "stop", "turn90", "turn180", "backward", "lateral", "start"}
    assert command_at(32)[2] - command_at(24)[2] == pytest.approx(np.pi)
    with pytest.raises(ValueError):
        command_at(60)


def test_no_measured_samples_are_null_not_zero_or_approved():
    assert distribution([]) == {"count": 0, "mean": None, "p95": None, "peak": None}
    with pytest.raises(FloatingPointError):
        distribution([float("nan")])


def test_free_metrics_use_physical_units_and_report_coverage():
    rep, _, _ = models()
    raw = rep.unnormalize(source_motion(rep, 65))[0, :64]
    raw[:, 0] = 0
    raw[:, 2] = torch.arange(64) / 60
    points = torch.zeros(64, 6, 3)
    indices = {"Left": np.arange(3), "Right": np.arange(3, 6)}
    initial = raw[0, :3].clone()
    initial[2] -= 1 / 60
    metrics = free_metrics(raw, torch.tensor([[0.0, 1.0]]).expand(64, -1), torch.zeros(64),
                           initial, rep, points, indices)
    assert metrics["velocity_error_mps"]["mean"] < 1e-5
    assert metrics["sole_penetration_m"]["peak"] == 0
    assert metrics["feet"]["Left"]["near_ground_material_speed_mps"]["mean"] == 0
    assert metrics["feet"]["Left"]["measured_vertex_interval_fraction"] == 1
    assert metrics["quality_approved"] is False
    assert metrics["boundary_joint_step_m"]["count"] > 0


def test_skinning_receives_full_joint_positions_not_only_root():
    rep, _, _ = models()
    raw = rep.unnormalize(source_motion(rep, 65))[0, :64]
    pose = rep.inverse(raw, False, return_all=True)

    class Skinner:
        def skin(self, rotations, positions, rot_is_global=False):
            assert positions.shape == (len(rotations), 30, 3)
            return torch.zeros(len(rotations), 6, 3)

    result = sole_vertices(pose, Skinner(), {"Left": np.arange(3), "Right": np.arange(3, 6)}, "cpu")
    assert result.shape == (64, 6, 3)
