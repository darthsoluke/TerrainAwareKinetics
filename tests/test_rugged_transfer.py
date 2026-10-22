import torch
from scripts.pilot_rugged_transfer import new_control,route_coverage


def test_command_uses_recorded_direction_not_future_body():
    case=dict(world=torch.zeros(40,12),recorded_direction=torch.tensor([[1.,0.]]).repeat(40,1),
              cruise_speed=torch.tensor(2.),heading_offset=torch.tensor(.15))
    before=new_control(case,0)
    case['world'][4:]=torch.randn_like(case['world'][4:])*50
    torch.testing.assert_close(before,new_control(case,0),atol=0,rtol=0)
    torch.testing.assert_close(before[:,:2],torch.tensor([[2.,0.]]))


def test_ordered_route_does_not_count_standing_at_the_goal_as_traversal():
    source=torch.zeros(81,3);source[:,0]=torch.linspace(0,4,81)
    assert route_coverage(source,source)['root_milestone_fraction']==1.
    repeated=source[-1:].repeat(81,1)
    report=route_coverage(repeated,source)
    assert report['endpoint_xz_error_m']==0.
    assert report['root_milestone_fraction']==0.
