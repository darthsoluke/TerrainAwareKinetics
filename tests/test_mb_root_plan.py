import pytest
import torch

from motionbricks.mb_sonic.root_plan import planar_root_reference


@pytest.mark.parametrize('initial,goal',[(.8,0.),(.8,-.8),(0.,1.2),(.5,.5)])
def test_planar_reference_brakes_or_reverses_without_moving_known_history(initial,goal):
    h=torch.zeros(1,4,5,dtype=torch.float64)
    h[0,:,0]=torch.arange(4)*initial/30
    h[:,:,1]=.8;h[:,:,3]=1
    prediction=torch.randn(1,32,5,dtype=torch.float64)
    result=planar_root_reference(prediction,h,torch.tensor([[goal,0.]],dtype=torch.float64))
    torch.testing.assert_close(result[:,:4],h,atol=0,rtol=0)
    torch.testing.assert_close(result[:,4:,[1,3,4]],prediction[:,4:,[1,3,4]],atol=0,rtol=0)
    velocity=result[0,:,[0,2]].diff(dim=0)*30
    acceleration=velocity.diff(dim=0)*30
    assert acceleration.norm(dim=-1).max()<=2.5+1e-5
    assert velocity[-1,0].item()==pytest.approx(goal,abs=1e-7)
    if goal==0:
        assert result[0,-1,0]-h[0,-1,0]==pytest.approx(initial**2/5,abs=1e-7)


def test_diagonal_reference_is_invariant_under_world_translation_and_rotation():
    h=torch.zeros(1,4,5,dtype=torch.float64)
    velocity=torch.tensor([.5,-.4],dtype=h.dtype)
    h[0,:,[0,2]]=torch.arange(4,dtype=h.dtype)[:,None]*velocity/30
    prediction=torch.zeros(1,32,5,dtype=h.dtype)
    goal=torch.tensor([[-.7,.8]],dtype=h.dtype)
    expected=planar_root_reference(prediction,h,goal)
    angle=torch.tensor(.83,dtype=h.dtype)
    rotation=torch.stack((torch.stack((angle.cos(),-angle.sin())),torch.stack((angle.sin(),angle.cos()))))
    shift=torch.tensor([13.,-5.],dtype=h.dtype)
    moved=h.clone();moved[:,:,[0,2]]=h[:,:,[0,2]]@rotation.T+shift
    actual=planar_root_reference(prediction,moved,goal@rotation.T)
    torch.testing.assert_close(actual[:,:,[0,2]],expected[:,:,[0,2]]@rotation.T+shift,atol=1e-12,rtol=0)
    speed=actual[:,:,[0,2]].diff(dim=1)*30
    assert speed.diff(dim=1).norm(dim=-1).max()*30<=2.5+1e-10
