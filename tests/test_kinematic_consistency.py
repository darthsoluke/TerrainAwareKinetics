import pytest
import torch

from motionbricks.data.kinematic_consistency import kinematic_consistency_losses


def linear_motion(fps):
    points = torch.zeros(1, 9, 2, 3)
    points[:, :, :, 0] = torch.arange(9)[None, :, None] / fps * 0.6
    velocity = torch.zeros_like(points)
    velocity[..., 0] = 0.6
    return points, velocity


@pytest.mark.parametrize("fps", [30.0, 60.0, 120.0])
def test_consistent_velocity_is_zero_independently_of_fps(fps):
    positions, velocity = linear_motion(fps)
    velocity[:, -1] = 1000
    result = kinematic_consistency_losses(positions, positions, velocity, fps)
    assert result["l_kinematic_position_consistency_m"] == 0
    assert result["l_kinematic_velocity_consistency_mps"] < 1e-6


def test_disagreement_has_physical_units_and_gradients_in_both_outputs():
    positions, velocity = linear_motion(60)
    fk = positions.clone().requires_grad_(True)
    feature = (positions + torch.tensor([0.02, 0, 0])).requires_grad_(True)
    velocity.zero_()
    velocity.requires_grad_(True)
    losses = kinematic_consistency_losses(fk, feature, velocity, 60)
    assert float(losses["l_kinematic_position_consistency_m"].detach()) == pytest.approx(0.02)
    assert float(losses["l_kinematic_velocity_consistency_mps"].detach()) == pytest.approx(0.6)
    sum(losses.values()).backward()
    for tensor in (fk, feature, velocity):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_consistency_is_translation_and_yaw_invariant():
    fk, velocity = linear_motion(60)
    feature = fk + torch.tensor([0.02, 0, 0.03])
    velocity *= 0.8
    rotation = torch.tensor([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    original = kinematic_consistency_losses(fk, feature, velocity, 60)
    transformed = kinematic_consistency_losses(fk @ rotation + 1.7, feature @ rotation + 1.7,
                                               velocity @ rotation, 60)
    for name in original:
        torch.testing.assert_close(original[name], transformed[name], atol=1e-5, rtol=1e-5)
