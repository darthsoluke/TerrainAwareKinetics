import pytest
import torch

from motionbricks.data.terrain_quarter_turn import rotate_heightfield, quarter_matrix
from motionbricks.data.unified_scene_field import field_from_heightfield


@pytest.mark.parametrize('turns', [1, 2, 3])
def test_rigid_grid_rotation_preserves_sdf_and_rotates_normals(turns):
    scene = dict(heightfield_y_m=torch.tensor([[0., 0., .2, .2, .2],
        [0., 0., .2, .2, .2], [0., 0., .2, .2, .2], [0., 0., .2, .2, .2]]),
        min_point_world_xyz_m=torch.tensor([-1.2, 0., -1.4]), dx_m=.4)
    points = torch.tensor([[-.9, .03, -.9], [-.3, .25, -.9], [-.3, .17, -.8], [-.64, .12, -.8]])
    matrix = quarter_matrix(turns, points)
    changed_points = points.clone()
    changed_points[:, [0, 2]] = points[:, [0, 2]] @ matrix.T
    distance, normal, valid = field_from_heightfield(scene).query(points)
    changed = rotate_heightfield(scene, turns)
    new_distance, new_normal, new_valid = field_from_heightfield(changed).query(changed_points)
    expected_normal = normal.clone()
    expected_normal[:, [0, 2]] = normal[:, [0, 2]] @ matrix.T
    assert valid.all() and torch.equal(valid, new_valid)
    torch.testing.assert_close(new_distance, distance, atol=1e-6, rtol=0)
    torch.testing.assert_close(new_normal, expected_normal, atol=1e-5, rtol=0)


def test_unknown_cells_rotate_and_roundtrip_without_filling():
    scene = dict(heightfield_y_m=torch.tensor([[0., float('nan'), 1.], [2., 3., 4.]]),
        min_point_world_xyz_m=torch.tensor([-3., 0., 5.]), dx_m=.25,
        heightfield_valid=torch.tensor([[True, False, True], [True, True, False]]))
    restored = scene
    for _ in range(4):
        restored = rotate_heightfield(restored, 1)
    torch.testing.assert_close(restored['heightfield_y_m'], scene['heightfield_y_m'], atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(restored['min_point_world_xyz_m'], scene['min_point_world_xyz_m'], atol=0, rtol=0)
    assert torch.equal(restored['heightfield_valid'], scene['heightfield_valid'])
    with pytest.raises(ValueError):
        rotate_heightfield(scene, True)
