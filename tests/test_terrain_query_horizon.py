import pytest
import torch

from motionbricks.data.local_terrain_observation import (
    LocalTerrainObservationConfig, extract_local_terrain_observation,
)
from motionbricks.data.native_mb_terrain_protocol import (
    TERRAIN_PROTOCOL, terrain_observation_config,
)


def test_directional_patch_sees_landing_without_global_map_access():
    # Root 0.4 m before a 2.8 m gap: landing begins 3.2 m ahead.
    terrain = torch.zeros((101, 41))
    terrain[14:42] = -0.6
    observation = extract_local_terrain_observation(
        terrain, [-2., 0., -1.], .1, [0., 1., 0.], 0., [0., 1.],
        config=terrain_observation_config())
    forward = observation.points[0, 1]
    assert forward[-1, 4, 2].item() == pytest.approx(3.4)
    assert forward[-1, 4, 1].item() == pytest.approx(-1.)
    assert forward[3, 4, 1].item() == pytest.approx(-1.6)
    radius = observation.points[..., [0, 2]].norm(dim=-1)
    assert radius.max() <= TERRAIN_PROTOCOL['max_horizontal_query_radius_m']
    assert forward[..., 0].abs().max().item() == pytest.approx(.6)
    assert observation.points[0, 0, ..., [0, 2]].abs().max().item() == pytest.approx(.4)


def test_extended_queries_turn_with_current_control_not_future_path():
    config = terrain_observation_config()
    kwargs = dict(heightfield_y_m=torch.zeros((101, 101)),
                  min_point_world_xyz_m=[-5., 0., -5.], dx_m=.1,
                  root_position_world_xyz_m=[0., 1., 0.], root_heading_yaw_rad=0., config=config)
    forward = extract_local_terrain_observation(**kwargs, control_direction_local_xz=[0., 1.])
    right = extract_local_terrain_observation(**kwargs, control_direction_local_xz=[1., 0.])
    assert forward.world_query_xz[0, 1, -1, 4].tolist() == pytest.approx([0., 3.4])
    assert right.world_query_xz[0, 1, -1, 4].tolist() == pytest.approx([3.4, 0.], abs=1e-6)


def test_invalid_lateral_spacing_rejected():
    with pytest.raises(ValueError, match='lateral'):
        LocalTerrainObservationConfig(forward_lateral_spacing_m=0.)
