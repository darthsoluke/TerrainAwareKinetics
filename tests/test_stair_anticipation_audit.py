import pytest
import torch

from scripts.audit_unified_stair_anticipation import first_riser


def test_riser_uses_cell_boundary_not_cell_center():
    scene = dict(heightfield_y_m=torch.tensor([[0., 0.], [0., 0.], [.16, .16]]),
                 min_point_world_xyz_m=torch.tensor([-1., 0., -2.]), dx_m=torch.tensor(.2))
    result = first_riser(scene, -1., -2.)
    assert result['z_m'] == pytest.approx(-1.7)
    assert result['floor_y_m'] == 0
    assert result['first_tread_y_m'] == pytest.approx(.16)


def test_no_stair_does_not_invent_riser():
    scene = dict(heightfield_y_m=torch.zeros(3, 2),
                 min_point_world_xyz_m=torch.tensor([-1., 0., -2.]), dx_m=torch.tensor(.2))
    with pytest.raises(ValueError, match='No forward riser'):
        first_riser(scene, -1., -2.)
