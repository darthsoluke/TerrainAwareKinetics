import pytest
import torch

from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.terrain_skin_tail_loss import skin_tail_penetration


def plane():
    scene = dict(heightfield_y_m=torch.zeros(21, 21), min_point_world_xyz_m=[-10., 0., -10.], dx_m=1.)
    return scene, HeightfieldPrismSDF(scene)


def test_exact_plane_tail_gradient_is_upward_and_only_changes_future():
    scene, field = plane()
    future = torch.tensor([[[0., -.1, 0.], [1., .2, 0.]], [[0., -.2, 0.], [1., .2, 0.]]], requires_grad=True)
    history = torch.tensor([[[0., .1, 0.], [1., .2, 0.]]], requires_grad=True)
    before = future.detach().clone()
    loss, info = skin_tail_penetration(future, history, scene, field)
    loss.backward()
    assert history.grad is None
    assert future.grad[-1, 0, 1] < 0
    assert torch.count_nonzero(future.grad[..., [0, 2]]) == 0
    torch.testing.assert_close(future, before, atol=0, rtol=0)
    assert abs(info['peak_m'] - .2) < 1e-6
    torch.testing.assert_close(loss.detach(), torch.tensor(.195 ** 2))


def test_stair_side_gradient_and_finite_difference():
    scene = dict(heightfield_y_m=torch.tensor([[0., 1., 1.]]).expand(5, -1),
                 min_point_world_xyz_m=[0., 0., -2.], dx_m=1.)
    field = HeightfieldPrismSDF(scene)
    future = torch.tensor([[[.6, .5, 0.]]], requires_grad=True)
    history = torch.tensor([[[.4, .5, 0.]]])
    loss, _ = skin_tail_penetration(future, history, scene, field, substeps=1)
    gradient = torch.autograd.grad(loss, future)[0]
    assert gradient[0, 0, 0] > 0
    assert gradient[0, 0, 1] == 0
    delta = torch.zeros_like(future)
    delta[..., 0] = 1e-4
    positive = skin_tail_penetration(future.detach() + delta, history, scene, field, substeps=1)[0]
    negative = skin_tail_penetration(future.detach() - delta, history, scene, field, substeps=1)[0]
    torch.testing.assert_close(gradient[0, 0, 0], (positive - negative) / 2e-4, atol=4e-5, rtol=1e-4)


def test_no_penetration_is_zero_and_unknown_is_not_ignored():
    scene, field = plane()
    future = torch.tensor([[[0., .1, 0.]]], requires_grad=True)
    loss, info = skin_tail_penetration(future, future.detach(), scene, field)
    loss.backward()
    assert loss == 0 and future.grad.abs().sum() == 0 and info['peak_m'] == 0
    with pytest.raises(ValueError, match='leaves measured'):
        skin_tail_penetration(future.detach() + torch.tensor([100., 0., 0.]), future.detach(), scene, field)


def test_support_proxy_requires_two_slow_intervals_and_bottom_vertices():
    from motionbricks.data.terrain_skin_tail_loss import teacher_support_proxy, teacher_support_gap
    points = torch.zeros(1, 5, 4, 3)
    points[:, :, 1, 1] = .1
    points[:, :, 3, 1] = .1
    points[0, :, 2:, 0] = torch.tensor([0., .1, .1, .2, .3])[:, None]
    mask = teacher_support_proxy(points.requires_grad_(), [2, 2])
    assert mask[:, :, 0].all()
    assert not mask[:, :, 1:].any()
    assert not mask.requires_grad
    scene, _ = plane()
    candidate = (points.detach()[:, 1:] + torch.tensor([0., .2, 0.])).requires_grad_()
    loss, count = teacher_support_gap(candidate, mask[:, 1:], [scene])
    loss.backward()
    assert count == 4
    assert (candidate.grad[:, :, 0, 1] > 0).all()
    assert candidate.grad[:, :, 1:].abs().sum() == 0
    assert points.grad is None


def test_moving_phase_proxy_requires_backward_motion_and_lower_foot():
    from motionbricks.data.terrain_skin_tail_loss import teacher_phase_support_proxy
    roots = torch.zeros(1, 5, 3)
    roots[0, :, 0] = torch.arange(5) * .1
    points = torch.zeros(1, 5, 4, 3)
    points[:, :, 1, 1] = .1
    points[0, :, 2:, 0] = .5 + torch.arange(5)[:, None] * .2
    points[:, :, 2:, 1] = .15
    mask = teacher_phase_support_proxy(points.requires_grad_(), roots, torch.tensor([[1., 0.]]), [2, 2])
    assert mask[:, :, 0].all() and not mask[:, :, 1:].any()
    assert not mask.requires_grad
    reverse = teacher_phase_support_proxy(points, roots, torch.tensor([[-1., 0.]]), [2, 2])
    assert not reverse.any()


def test_direction_control_does_not_penalize_natural_slowdown():
    from motionbricks.data.terrain_skin_tail_loss import direction_without_speed_loss
    request = torch.tensor([[3., 0.]])
    for speed in (0., .05, .5, 1., 3.):
        velocity = torch.tensor([[[speed, 0.]]] * 4).transpose(0, 1).requires_grad_()
        loss = direction_without_speed_loss(velocity, request, torch.ones(2))
        loss.backward()
        assert loss == 0 and velocity.grad.abs().sum() == 0
    for value in ([1., .4], [-1., 0.]):
        velocity = torch.tensor([[value]], requires_grad=True)
        assert direction_without_speed_loss(velocity, request, torch.ones(2)) > 0
    assert direction_without_speed_loss(torch.tensor([[[.2, 0.]]]), torch.zeros_like(request), torch.ones(2)) > 0
