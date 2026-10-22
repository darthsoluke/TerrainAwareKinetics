"""Exact heightfield-grid rotations paired with native full-body coordinates."""

import torch

from .terrain_scene_frame import HorizontalSceneFrame


def quarter_matrix(turns, reference):
    if not isinstance(turns, int) or isinstance(turns, bool) or turns not in range(4):
        raise ValueError('Quarter turns must be an integer0..3')
    values = (((1., 0.), (0., 1.)), ((0., 1.), (-1., 0.)),
              ((-1., 0.), (0., -1.)), ((0., -1.), (1., 0.)))
    return reference.new_tensor(values[turns])


def rotate_heightfield(scene, turns):
    height = torch.as_tensor(scene['heightfield_y_m'])
    origin = torch.as_tensor(scene['min_point_world_xyz_m'])
    if height.ndim != 2 or origin.shape != (3,) or float(scene['dx_m']) <= 0:
        raise ValueError('A2Dheightfield, XYZ origin and positive cell size are required')
    matrix = quarter_matrix(turns, origin)
    if turns == 0:
        return scene
    dx = float(scene['dx_m'])
    x0, z0 = origin[0], origin[2]
    x1, z1 = x0 + (height.shape[1] - 1) * dx, z0 + (height.shape[0] - 1) * dx
    corners = torch.stack((torch.stack((x0, z0)), torch.stack((x0, z1)),
                           torch.stack((x1, z0)), torch.stack((x1, z1))))
    low = (corners @ matrix.T).amin(0)
    new_origin = torch.stack((low[0], origin[1], low[1]))
    result = dict(scene, heightfield_y_m=torch.rot90(height, turns, (0, 1)).contiguous(),
                  min_point_world_xyz_m=new_origin)
    if 'heightfield_valid' in scene:
        result['heightfield_valid'] = torch.rot90(torch.as_tensor(scene['heightfield_valid']), turns, (0, 1)).contiguous()
    return result


def rotate_motion(rep, world, turns):
    quarter_matrix(turns, world)
    if turns == 0:
        return world.clone()
    frame = HorizontalSceneFrame(world.new_full((len(world),), turns * torch.pi / 2),
                                 world.new_zeros(len(world), 2))
    return frame.motion(rep, world)
