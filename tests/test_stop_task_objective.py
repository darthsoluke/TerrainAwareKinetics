import torch

from motionbricks.data.stop_task_objective import published_root_velocities, scaled_zero_velocity_loss


def test_velocity_goal_includes_the_observed_publication_boundary():
    history = torch.zeros(1, 4, 5)
    prediction = torch.zeros(1, 32, 5, requires_grad=True)
    with torch.no_grad():
        prediction[:, 4:12, 0] = .06
    velocities = published_root_velocities(prediction, history)
    torch.testing.assert_close(velocities[0, 0], torch.tensor([1.8, 0.]))
    assert torch.count_nonzero(velocities[:, 1:]) == 0
    loss = scaled_zero_velocity_loss(velocities)
    loss.backward()
    assert prediction.grad[:, 4, 0].abs().sum() > 0
    assert torch.count_nonzero(prediction.grad[:, :4]) == 0


def test_dimensionless_scale_is_the_existing_stop_speed_not_a_cap():
    velocity = torch.tensor([[[.15, 0.]]], requires_grad=True)
    original = velocity.detach().clone()
    loss = scaled_zero_velocity_loss(velocity)
    torch.testing.assert_close(loss, torch.tensor(.5))
    torch.testing.assert_close(velocity.detach(), original)
    loss.backward()
    assert velocity.grad.norm() > 0
