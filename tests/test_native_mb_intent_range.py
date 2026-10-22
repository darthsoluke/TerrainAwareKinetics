import pytest
import torch

from motionbricks.data.local_terrain_observation import extract_local_terrain_observation
from motionbricks.data.native_mb_online_terrain import gather_online_terrain_observation
from motionbricks.data.native_mb_terrain_protocol import INTENT_RANGE_POLICY, terrain_observation_config


def test_intent_range_changes_with_speed_horizon_without_losing_near_support():
    heightfield = torch.zeros(121, 121)
    root = torch.tensor([[0., 1., 0.]]).expand(3, -1)
    directions = torch.tensor([[0., 0.], [0., 1.6 * 28 / 30], [0., 3. * 28 / 30]])
    query = dict(heightfield_y_m=heightfield, min_point_world_xyz_m=[-6., 0., -6.], dx_m=.1,
                 root_position_world_xyz_m=root, root_heading_yaw_rad=torch.zeros(3),
                 control_direction_local_xz=directions)
    fixed = extract_local_terrain_observation(**query, config=terrain_observation_config())
    adaptive = extract_local_terrain_observation(**query, config=terrain_observation_config(INTENT_RANGE_POLICY))
    torch.testing.assert_close(fixed.points[:, 0], adaptive.points[:, 0], atol=0, rtol=0)
    assert adaptive.points.shape == fixed.points.shape == (3, 2, 9, 9, 3)
    torch.testing.assert_close(adaptive.points[:, 1, -1, 4, 2], torch.tensor([.8, 1.6*28/30+.4, 3.2]))
    torch.testing.assert_close(fixed.points[:, 1, -1, 4, 2], torch.full((3,), 3.4))
    assert torch.equal(fixed.valid, adaptive.valid)


def test_training_queries_match_online_geometry_at_sampled_current_frame():
    scene = {'heightfield_y_m': torch.arange(101).float()[None].expand(101, -1) * .01,
             'min_point_world_xyz_m': torch.tensor([-5., 0., -5.]), 'dx_m': torch.tensor(.1)}
    roots = torch.zeros(2, 24, 5)
    roots[..., 1] = 1.
    roots[..., 3] = 1.
    roots[1, :, 0] = 1.2
    roots[:, 15:, 1] = 99.  # Unobserved poses cannot affect the current query.
    sample = {'chosen_ids': torch.tensor([1, 0, 1]),
              'frame_indices': torch.tensor([[5, 6, 7, 8], [0, 1, 2, 3], [7, 8, 9, 10]])}
    commands = torch.tensor([[0., 0.], [0., 1.5], [2., 0.]])
    batch = {'terrain_scene': [scene, scene], 'terrain_root_world': roots}
    actual = gather_online_terrain_observation(batch, sample, commands, range_policy=INTENT_RANGE_POLICY)
    current = roots[sample['chosen_ids'], sample['frame_indices'][:, 3]]
    expected = extract_local_terrain_observation(**scene, root_position_world_xyz_m=current[:, :3],
        root_heading_yaw_rad=torch.zeros(3), control_direction_local_xz=commands,
        config=terrain_observation_config(INTENT_RANGE_POLICY))
    torch.testing.assert_close(actual['points'], expected.points, atol=0, rtol=0)
    assert torch.equal(actual['valid'], expected.valid)
    with pytest.raises(ValueError, match='range policy'):
        terrain_observation_config('untrained-guess')


@pytest.mark.parametrize('range_policy', [None, INTENT_RANGE_POLICY])
@pytest.mark.parametrize('packed_rows', [None, 3])
def test_batched_ragged_maps_equal_individual_queries(range_policy, packed_rows, monkeypatch):
    scenes = [
        {'heightfield_y_m': torch.arange(35).reshape(5, 7).float() * .03,
         'min_point_world_xyz_m': torch.tensor([-.6, 0., -.2]), 'dx_m': torch.tensor(.2)},
        {'heightfield_y_m': torch.full((11, 3), 1.5),
         'min_point_world_xyz_m': torch.tensor([0., 0., -.6]), 'dx_m': torch.tensor(.1)},
    ]
    scenes[0]['heightfield_y_m'][2, 3] = torch.nan
    roots = torch.zeros(2, 12, 5)
    roots[..., 1] = 1.
    roots[..., 3] = 1.
    roots[1, :, 0] = .7
    sample = {'chosen_ids': torch.tensor([1, 0, 1, 0]),
              'frame_indices': torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [0, 1, 2, 3], [6, 7, 8, 9]])}
    directions = torch.tensor([[0., .1], [1.5, 0.], [0., -2.], [0., 0.]])
    batch = {'terrain_scene': scenes, 'terrain_root_world': roots}
    expected = gather_online_terrain_observation(batch, sample, directions, range_policy=range_policy,
                                                  batch_queries=False)
    calls = []
    original = extract_local_terrain_observation
    def capture(*args, **kwargs):
        calls.append(args[0].shape[0])
        return original(*args, **kwargs)
    monkeypatch.setattr('motionbricks.data.local_terrain_observation.extract_local_terrain_observation', capture)
    if packed_rows:
        monkeypatch.setattr('motionbricks.data.native_mb_online_terrain.MAX_PACKED_TERRAIN_CELLS', 11*7*packed_rows)
    actual = gather_online_terrain_observation(batch, sample, directions, range_policy=range_policy)
    torch.testing.assert_close(actual['points'], expected['points'], atol=0, rtol=0)
    assert torch.equal(actual['valid'], expected['valid'])
    assert calls == ([3, 1] if packed_rows else [4])
