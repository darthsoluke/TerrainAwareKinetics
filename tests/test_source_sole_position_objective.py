import torch

from motionbricks.data.source_sole_position_objective import source_sole_position_loss


class Flat:
    def query(self, points):
        return points[..., 1], torch.ones(points.shape[:-1], dtype=torch.bool)


def test_source_xyz_supervision_catches_same_tread_slip_without_runtime_changes():
    source = torch.zeros(1, 4, 2, 3, requires_grad=True)
    prediction = source.detach().clone()
    prediction[..., 0] = .1
    prediction.requires_grad_()
    before = prediction.detach().clone()
    loss, active = source_sole_position_loss(prediction, source, Flat())
    torch.testing.assert_close(loss, torch.tensor(.1))
    gradient, source_gradient = torch.autograd.grad(loss, (prediction, source), allow_unused=True)
    assert active.all() and gradient[..., 0].abs().sum() > 0
    assert source_gradient is None
    torch.testing.assert_close(prediction, before, atol=0, rtol=0)


def test_airborne_and_fast_source_transitions_are_not_locked():
    source = torch.zeros(1, 4, 2, 3)
    source[:, :, 0, 1] = .3
    source[:, :, 1, 0] = torch.arange(4)*.1
    prediction = (source+.1).requires_grad_()
    loss, active = source_sole_position_loss(prediction, source, Flat())
    assert not active.any() and loss == 0
    assert torch.count_nonzero(torch.autograd.grad(loss, prediction)[0]) == 0


def test_real_slow_source_rolling_uses_measured_positions_not_forced_zero_velocity():
    source = torch.zeros(1, 4, 1, 3)
    source[0, :, 0, 0] = torch.arange(4)*.002
    prediction = source.clone().requires_grad_()
    loss, active = source_sole_position_loss(prediction, source, Flat())
    assert active.all() and loss == 0
    assert torch.isfinite(torch.autograd.grad(loss, prediction)[0]).all()
