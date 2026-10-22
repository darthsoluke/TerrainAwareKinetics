import pytest
import torch

from motionbricks.data.terrain_response_loss import terrain_clearance,surface_training_losses


def test_flat_ground_penetration_and_upward_gradient_are_in_meters():
    scene={'heightfield_y_m':torch.zeros(9,9),'min_point_world_xyz_m':[-.8,0,-.8],'dx_m':.2}
    points=torch.tensor([[0.,-.3,0.],[0.,.5,0.]],requires_grad=True)
    sdf,known=terrain_clearance(points,scene,point_chunk=1,cell_chunk=7)
    assert known.all() and sdf[0].item()==pytest.approx(-.3)
    assert sdf[1]>0
    (-sdf[0]).backward()
    assert points.grad[0,1]<0 and torch.isfinite(points.grad).all()


def test_wall_interior_has_sideways_escape_gradient_not_cell_seams():
    hf=torch.zeros(21,21);hf[10:]=2.
    scene={'heightfield_y_m':hf,'min_point_world_xyz_m':[-1,0,-1],'dx_m':.1}
    point=torch.tensor([[0.,.8,.25]],requires_grad=True)
    sdf,known=terrain_clearance(point,scene)
    assert known.all() and sdf.item()==pytest.approx(-.3,abs=1e-6)
    (-sdf.sum()).backward()
    assert point.grad[0,2]>0 and abs(point.grad[0,1])<1e-6


def test_invalid_terrain_is_not_a_support_target_and_contact_cannot_reward_floating():
    hf=torch.zeros(9,9);hf[0,0]=float('nan')
    scene={'heightfield_y_m':hf,'min_point_world_xyz_m':[-.8,0,-.8],'dx_m':.2}
    source=torch.zeros(5,2,3);source[:,1,0]=-.8;source[:,1,2]=-.8
    pred=source.clone();pred[...,1]=.2;pred.requires_grad_(True)
    losses=surface_training_losses(pred,source,scene,2)
    assert losses['known_points']==5 and losses['unknown_points']==5
    assert losses['support_points']==5
    assert losses['source_support_gap_m'].item()==pytest.approx(.2)
    losses['source_support_gap_m'].backward()
    assert pred.grad[:,0,1].sum()>0 and pred.grad[:,1].abs().sum()==0
