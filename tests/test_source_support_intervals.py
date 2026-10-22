from types import SimpleNamespace

import pytest
import torch

from motionbricks.data.terrain_response_loss import source_support_masks
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.source_sole_position_objective import source_sole_position_loss


def plane(points):
    return points[...,1],torch.ones_like(points[...,1],dtype=torch.bool)


def test_support_endpoints_do_not_bridge_a_moving_interval():
    sole=torch.zeros(1,4,1,3);sole[:,2:,:,0]=.01
    active,intervals=source_support_masks(sole,*plane(sole))
    assert active.all()
    assert intervals[0,:,0].tolist()==[True,False,True]
    # Source position targets remain unchanged, even though velocity mask differs.
    loss,position_mask=source_sole_position_loss(torch.zeros_like(sole),sole,SimpleNamespace(query=plane))
    assert torch.equal(active,position_mask)
    torch.testing.assert_close(loss,torch.tensor(.005))


@pytest.mark.parametrize('kind',['prepared','legacy'])
def test_actual_velocity_objectives_use_certified_intervals(kind):
    objective=object.__new__(JointGeometryObjective);objective.sole_count=1
    objective.geometry=lambda value:(value,value)
    objective.rep=SimpleNamespace(inverse=lambda value,*a,**k:{'posed_joints':value})
    surface=SimpleNamespace(query=plane)
    source=torch.zeros(1,12,1,3);source[:,6:8,:,0]=.01
    prediction=torch.zeros_like(source,requires_grad=True)
    if kind=='prepared':
        losses=objective.prepared_losses(prediction,prediction[:,:4],
            dict(joints=source,points=source,volumes=surface,heights=surface),offset=0,diagnostics=False)
    else:
        objective.volume=lambda *a,**k:surface
        scene=dict(heightfield_y_m=torch.zeros(5,5),min_point_world_xyz_m=torch.tensor([-1.,0.,-1.]),dx_m=.5)
        losses=objective.losses(prediction,source,prediction[:,:4],[scene],['test'])
    assert losses['source_support_velocity_mps'].item()==0
    assert losses['support_vertices']==4
    losses['source_support_velocity_mps'].backward()
    assert torch.count_nonzero(prediction.grad)==0


@pytest.mark.parametrize('length',[1,2,5])
def test_stationary_and_unknown_boundaries(length):
    sole=torch.zeros(2,length,3,3);gap,known=plane(sole)
    active,intervals=source_support_masks(sole,gap,known)
    assert intervals.shape==(2,max(0,length-1),3)
    if length>1:assert intervals.all() and active.all()
    else:assert not active.any()
    known[0]=False
    active,intervals=source_support_masks(sole,gap,known)
    assert not active[0].any() and not intervals[0].any()
