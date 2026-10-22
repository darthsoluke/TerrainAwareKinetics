import torch

from scripts.pilot_joint_sdf_full_context import command


def test_recorded_turn_preserves_body_travel_offset_without_future_pose():
    direction = torch.zeros(41, 2)
    direction[:, 0] = 1
    direction[11] = torch.tensor([0., -1.])
    world = torch.zeros(41, 5)
    world[:, 3] = 1
    case = {'name': 'stairs', 'world': world, 'command_velocity': [2., 0.],
            'recorded_direction': direction}
    straight = command(case, 0, device='cpu')
    turned = command(case, 8, device='cpu')
    torch.testing.assert_close(straight, torch.tensor([[2., 0., 1., 0.]]))
    torch.testing.assert_close(turned, torch.tensor([[0., -2., 0., 1.]]), atol=1e-6, rtol=0)
    case['world'][4:] = 1000
    torch.testing.assert_close(command(case, 8, device='cpu'), turned, atol=0, rtol=0)
    torch.testing.assert_close(command(case, 8, held=True, device='cpu'), straight, atol=0, rtol=0)


def test_direction_norm_is_not_used_as_speed_and_scale_only_changes_request():
    direction = torch.tensor([[.6, .8]]).expand(41, -1).clone()
    world = torch.zeros(41, 5)
    world[:, 3] = 1
    case = {'name': 'stairs', 'world': world, 'command_velocity': [3., 4.],
            'recorded_direction': direction}
    slow = command(case, 0, scale=.8, device='cpu')
    fast = command(case, 0, scale=1.2, device='cpu')
    torch.testing.assert_close(slow[:, :2].norm(dim=-1), torch.tensor([4.]))
    torch.testing.assert_close(fast[:, :2].norm(dim=-1), torch.tensor([6.]))
    torch.testing.assert_close(slow[:, 2:], fast[:, 2:], atol=0, rtol=0)
    torch.testing.assert_close(case['recorded_direction'], direction, atol=0, rtol=0)
