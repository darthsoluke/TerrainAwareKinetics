import math

import pytest
import torch

from motionbricks.data.local_terrain_observation import (
    LocalTerrainObservationConfig,
    extract_local_terrain_observation,
)


def config():
    return LocalTerrainObservationConfig(
        patch_height=9, patch_width=9, near_spacing_m=0.1,
        forward_spacing_m=0.1, forward_start_m=0.1,
    )


def observe(heightfield, *, root=(0.0, 1.0, 0.0), yaw=0.0, command=(0.0, 0.0),
            origin=(-2.0, 0.0, -2.0), dx=0.1, known=None, cfg=None):
    return extract_local_terrain_observation(
        torch.as_tensor(heightfield, dtype=torch.float32), origin, dx, root, yaw, command,
        heightfield_valid=known, config=cfg or config(),
    )


def test_flat_heightfield_model_contract_and_relative_xyz():
    obs = observe(torch.full((41, 41), 1.25))
    inputs = obs.as_model_inputs()
    assert set(inputs) == {"points", "valid"}
    assert inputs["points"].shape == (1, 2, 9, 9, 3)
    assert inputs["valid"].shape == (1, 2, 9, 9)
    assert inputs["valid"].all()
    torch.testing.assert_close(inputs["points"][..., 1], torch.full((1, 2, 9, 9), 0.25))
    assert torch.all(inputs["points"][0, 0, 4, 4] == torch.tensor([0.0, 0.25, 0.0]))
    assert obs.world_query_xz.shape == (1, 2, 9, 9, 2)
    assert obs.body_probe_positions_local_xyz_m is None


def test_stair_step_is_nearest_cell_not_bilinear():
    terrain = torch.zeros(41, 41)
    terrain[:, 20:] = 0.3
    obs = observe(terrain, root=(0.0, 0.0, 0.0), command=(0.0, 1.0))
    near_x = obs.world_query_xz[0, 0, :, :, 0]
    near_h = obs.points[0, 0, :, :, 1]
    assert (near_h == 0).any() and torch.isclose(near_h, torch.tensor(0.3)).any()
    assert not ((near_h > 0) & (near_h < 0.29)).any()
    assert (near_x < 0).any() and (near_x >= 0).any()


def test_command_reversal_rotates_intent_patch_but_preserves_near_patch():
    terrain = torch.zeros(81, 81)
    terrain[48, 40] = 0.7
    args = dict(origin=(-4.0, 0.0, -4.0), root=(0.0, 0.0, 0.0))
    forward = observe(terrain, command=(0.0, 1.0), **args)
    backward = observe(terrain, command=(0.0, -1.0), **args)
    torch.testing.assert_close(forward.points[:, 0], backward.points[:, 0])
    torch.testing.assert_close(forward.valid[:, 0], backward.valid[:, 0])
    assert forward.points[:, 1, :, :, 1].max() == 0.7
    assert backward.points[:, 1, :, :, 1].max() == 0.0


def test_left_right_commands_point_intent_patch_in_opposite_directions():
    terrain = torch.zeros(81, 81)
    args = dict(origin=(-4.0, 0.0, -4.0), root=(0.0, 0.0, 0.0))
    obs_left = observe(terrain, command=(-1.0, 0.0), **args)
    obs_right = observe(terrain, command=(1.0, 0.0), **args)
    left_mean = obs_left.world_query_xz[0, 1].mean(dim=(0, 1))
    right_mean = obs_right.world_query_xz[0, 1].mean(dim=(0, 1))
    assert left_mean[0] == pytest.approx(-0.5) and right_mean[0] == pytest.approx(0.5)
    assert left_mean[1] == pytest.approx(0.0, abs=1e-5)


def test_yaw_and_translation_equivariance():
    terrain = torch.full((81, 81), 0.7)
    origin = (-4.0, 0.0, -4.0)
    root = torch.tensor([0.0, 0.4, 0.0])
    yaw = 0.63
    command = torch.tensor([0.3, 0.8])
    base = extract_local_terrain_observation(terrain, origin, 0.1, root, yaw, command, config=config())
    delta = torch.tensor([1.1, 2.0, -0.7])
    angle = -0.41
    c, s = math.cos(angle), math.sin(angle)
    rotation = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    moved_origin = torch.tensor(origin) @ rotation.T + delta
    moved = extract_local_terrain_observation(terrain + delta[1], moved_origin, 0.1,
                                                  root @ rotation.T + delta, yaw + angle,
                                              command, config=config())
    torch.testing.assert_close(base.points, moved.points, atol=2e-5, rtol=0)
    torch.testing.assert_close(base.valid, moved.valid)
    rotation_xz_t = rotation[[0, 2]][:, [0, 2]].T
    expected_world = base.world_query_xz @ rotation_xz_t + delta[[0, 2]]
    torch.testing.assert_close(moved.world_query_xz, expected_world, atol=2e-5, rtol=0)


def test_unknown_cells_and_map_exterior_are_invalid_without_edge_clamping():
    terrain = torch.full((5, 5), 4.0)
    terrain[2, 2] = float("nan")
    known = torch.ones(5, 5, dtype=torch.bool)
    known[2, 3] = False
    obs = observe(terrain, root=(0.0, 3.0, 0.0), origin=(-0.2, 0.0, -0.2), dx=0.1, known=known)
    assert (~obs.valid).any()
    assert torch.isfinite(obs.points).all()
    assert (obs.points[..., 1][~obs.valid] == 0).all()
    far = observe(torch.full((5, 5), 4.0), root=(4.0, 3.0, 0.0),
                  origin=(-0.2, 0.0, -0.2), dx=0.1)
    assert not far.valid.any()
    assert (far.points[..., 1] == 0).all()
    assert (far.points[..., [0, 2]] != 0).any()


def test_far_terrain_changes_do_not_affect_local_observation():
    first = torch.zeros(101, 101)
    second = first.clone()
    second[:10, :] = 900.0
    second[90:, :] = -900.0
    args = dict(root=(0.0, 0.0, 0.0), command=(0.0, 1.0), origin=(-5.0, 0.0, -5.0))
    obs_a, obs_b = observe(first, **args), observe(second, **args)
    torch.testing.assert_close(obs_a.points, obs_b.points)
    torch.testing.assert_close(obs_a.valid, obs_b.valid)


def test_zero_command_uses_heading_and_stays_finite():
    obs = observe(torch.zeros(81, 81), root=(0.0, 0.0, 0.0), yaw=math.pi / 2,
                  command=(0.0, 0.0), origin=(-4.0, 0.0, -4.0))
    directional = obs.world_query_xz[0, 1].mean(dim=(0, 1))
    assert directional[0] == pytest.approx(0.5)
    assert abs(float(directional[1])) < 1e-5
    assert torch.isfinite(obs.points).all() and torch.isfinite(obs.world_query_xz).all()


def test_near_patch_retains_support_outside_command_facing_area():
    terrain = torch.zeros(81, 81)
    terrain[40, 36] = 0.5
    obs = observe(terrain, command=(1.0, 0.0), root=(0.0, 0.0, 0.0), origin=(-4.0, 0.0, -4.0))
    assert obs.points[0, 0, :, :, 1].max() == 0.5
    assert (obs.points[0, 0, :, :, 0] == 0).any()


def test_batched_queries_match_independent_maps():
    maps = torch.stack((torch.zeros(41, 41), torch.ones(41, 41) * 0.2))
    obs = extract_local_terrain_observation(
        maps, [(-2.0, 0.0, -2.0)] * 2, 0.1,
        [(0.0, 0.5, 0.0), (0.0, 0.5, 0.0)], [0.0, 0.0],
        [(0.0, 1.0), (0.0, 0.0)], config=config(),
    )
    assert obs.points.shape == (2, 2, 9, 9, 3)
    torch.testing.assert_close(obs.points[0, ..., 1], torch.full((2, 9, 9), -0.5))
    torch.testing.assert_close(obs.points[1, ..., 1], torch.full((2, 9, 9), -0.3))


def test_pose_batch_can_reuse_one_static_map():
    obs = extract_local_terrain_observation(
        torch.zeros(41, 41), (-2.0, 0.0, -2.0), 0.1,
        [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)], [0.0, 0.0],
        [(0.0, 1.0), (0.0, -1.0)], config=config(),
    )
    assert obs.points.shape == (2, 2, 9, 9, 3)


def test_three_distinct_roots_and_yaws_match_individual_evaluations():
    z, x = torch.meshgrid(torch.arange(41), torch.arange(41), indexing="ij")
    base_map = x.float() * 0.01 + z.float() * 0.02
    maps = torch.stack((base_map, base_map + 0.3, base_map - 0.2))
    origins = torch.tensor([[-2.0, 0.0, -2.0], [-1.8, 0.0, -2.1], [-2.1, 0.0, -1.9]])
    roots = torch.tensor([[-0.4, 0.7, 0.2], [0.3, 1.1, -0.5], [0.6, 0.4, 0.1]])
    yaws = torch.tensor([0.0, 0.73, -1.12])
    commands = torch.tensor([[0.0, 1.0], [-0.4, 0.7], [0.0, 0.0]])
    batched = extract_local_terrain_observation(
        maps, origins, 0.1, roots, yaws, commands, config=config())
    assert batched.points.shape == (3, 2, 9, 9, 3)
    for index in range(3):
        single = extract_local_terrain_observation(
            maps[index], origins[index], 0.1, roots[index], yaws[index], commands[index], config=config())
        torch.testing.assert_close(batched.points[index], single.points[0])
        torch.testing.assert_close(batched.valid[index], single.valid[0])
        torch.testing.assert_close(batched.world_query_xz[index], single.world_query_xz[0])


def test_invalid_metric_resolution_fails_loudly():
    with pytest.raises(ValueError):
        observe(torch.zeros(11, 11), dx=0.0)


def test_config_rejects_even_patch_dimensions():
    with pytest.raises(ValueError, match="patch_width"):
        LocalTerrainObservationConfig(patch_width=4)
