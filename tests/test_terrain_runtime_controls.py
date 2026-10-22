import pytest
import torch

from motionbricks.data.terrain_runtime_controls import RUNTIME_DEFAULTS, validate_runtime_controls
from motionbricks.data.local_terrain_observation import extract_local_terrain_observation,trained_query_controls
from motionbricks.data.native_mb_terrain_protocol import terrain_observation_config, INTENT_RANGE_POLICY


def test_runtime_ranges_reject_unknown_nonfinite_and_unsafe_values():
    assert validate_runtime_controls(RUNTIME_DEFAULTS)==RUNTIME_DEFAULTS
    for update in [{'num_tokens':17},{'inference_steps':0},{'refresh_frames':9},
                   {'lookahead_m':float('nan')},{'lateral_halfwidth_m':6.5},
                   {'future_height':1},{'num_tokens':True}]:
        with pytest.raises(ValueError):
            validate_runtime_controls({**RUNTIME_DEFAULTS,**update})


def test_metric_query_controls_preserve_near_patch_and_expose_a_distant_wall():
    hf=torch.zeros(121,121);hf[80:]=1.2
    kwargs=dict(heightfield_y_m=hf,min_point_world_xyz_m=[-6,0,-6],dx_m=.1,
        root_position_world_xyz_m=[0,1,0],root_heading_yaw_rad=0,
        control_direction_local_xz=[0,1.49],config=terrain_observation_config(INTENT_RANGE_POLICY))
    original=extract_local_terrain_observation(**kwargs)
    default=extract_local_terrain_observation(**kwargs,query_controls=torch.tensor([[-1,.6]]))
    longer=extract_local_terrain_observation(**kwargs,query_controls=torch.tensor([[3.4,1.]]))
    torch.testing.assert_close(default.points,original.points,rtol=0,atol=0)
    torch.testing.assert_close(longer.points[:,0],original.points[:,0],rtol=0,atol=0)
    assert original.points[0,1,...,1].max()==-1
    assert longer.points[0,1,...,1].max()==pytest.approx(.2)
    assert longer.points[0,1,...,2].max()==pytest.approx(3.4)
    assert longer.points[0,1,...,0].abs().max()==pytest.approx(1.)
    scaled=extract_local_terrain_observation(**kwargs,query_controls=torch.tensor([[-1.,.6,3.,.4]]))
    assert scaled.points[0,1,...,2].max()==pytest.approx(1.49*3+.4)
    torch.testing.assert_close(scaled.points[:,0],original.points[:,0],rtol=0,atol=0)


def test_curved_decoded_path_scans_a_wall_missed_by_the_input_direction():
    hf=torch.zeros(101,101);hf[55:66,55:66]=1.2
    kwargs=dict(heightfield_y_m=hf,min_point_world_xyz_m=[-4,0,-4],dx_m=.1,
        root_position_world_xyz_m=[0,1,0],root_heading_yaw_rad=0,
        control_direction_local_xz=[0,3.],config=terrain_observation_config(INTENT_RANGE_POLICY),
        query_controls=torch.tensor([[3.,.6,1.,.4]]))
    straight=extract_local_terrain_observation(**kwargs)
    theta=torch.linspace(0,torch.pi/2,40)
    path=torch.stack((2*(1-torch.cos(theta)),2*torch.sin(theta)),-1)[None].requires_grad_(True)
    curved=extract_local_terrain_observation(**kwargs,query_path_world_xz=path,
                                           query_path_valid=torch.ones(1,40,dtype=torch.bool))
    assert straight.points[0,1,...,1].max()==-1
    assert curved.points[0,1,...,1].max()==pytest.approx(.2)
    torch.testing.assert_close(curved.points[:,0],straight.points[:,0],rtol=0,atol=0)
    assert curved.world_query_xz[0,1,:,4,0].max()>1.5
    assert not curved.points.requires_grad
    assert curved.as_model_inputs()['query_envelope_m'].numel()==1
    unavailable=extract_local_terrain_observation(**kwargs,query_path_world_xz=path,
        query_path_valid=torch.zeros(1,40,dtype=torch.bool))
    torch.testing.assert_close(unavailable.points,straight.points,atol=1e-6,rtol=0)


def test_far_survey_coordinates_cannot_expand_the_trained_motion_input():
    config=terrain_observation_config(INTENT_RANGE_POLICY)
    query=torch.tensor([[20.,6.,4.,10.],[-1.,6.,4.,10.]])
    direction=torch.tensor([[0.,0.],[0.,8.]])
    bounded=trained_query_controls(query,direction,config)
    torch.testing.assert_close(bounded[:,0],torch.full((2,),3.4))
    torch.testing.assert_close(bounded[:,1],torch.full((2,),.6))
    assert torch.equal(query[:,0],torch.tensor([20.,-1.]))
    near=trained_query_controls(torch.tensor([[1.9,.5,1.,.4]]),direction[:1],config)
    assert near[0,0]==pytest.approx(1.9) and near[0,1]==pytest.approx(.5)
