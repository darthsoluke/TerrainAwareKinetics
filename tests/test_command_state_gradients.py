import torch
import pytest

from motionbricks.data.terrain_control import SoftCommandState


def history(x):
    base=torch.zeros(len(x),4,8,dtype=x.dtype,device=x.device)
    base[:,:,3]=1
    base[:,:,0]=x[:,None]
    return base


def sequence(xs,tracked):
    state=SoftCommandState(history_gradients=tracked);errors=[];tasks=[]
    speeds=((1.6,0.),(0.,1.6),(0.,0.),(0.,0.),(1.6,1.6))
    for step,speed in enumerate(speeds):
        h=history(xs[step]);v=xs.new_tensor([[speed[0],0.,1.,0.],[speed[1],0.,1.,0.]])
        task=state.from_controls(h,v,8);tasks.append(task)
        errors.append(torch.asinh(task[:,0]-xs[step]))
    return torch.stack(errors),torch.stack(tasks)


def test_full_mixed_batch_state_jacobian_matches_forward_finite_difference():
    xs=torch.tensor([[0.,.1],[.3,.2],[.4,.5],[.5,.6],[.7,.8]],dtype=torch.float64,requires_grad=True)
    jac=torch.autograd.functional.jacobian(lambda x:sequence(x,True)[0],xs)
    eps=1e-5;numeric=torch.empty_like(jac)
    for t in range(5):
        for b in range(2):
            plus=xs.detach().clone();minus=plus.clone();plus[t,b]+=eps;minus[t,b]-=eps
            numeric[:,:,t,b]=(sequence(plus,False)[0]-sequence(minus,False)[0])/(2*eps)
    torch.testing.assert_close(jac,numeric,atol=1e-8,rtol=1e-8)
    torch.testing.assert_close(sequence(xs,True)[1].detach(),sequence(xs,False)[1],atol=0,rtol=0)
    assert jac[3,0,1,0]>0 and jac[3,0,3,0]<0
    assert jac[3,1,2,1]>0 and jac[3,1,3,1]<0
    assert torch.count_nonzero(jac[4])==0


def test_explicit_goal_and_commands_remain_external_not_learned_targets():
    x=torch.tensor([.4],dtype=torch.float64,requires_grad=True)
    controls=torch.tensor([[1.,0.,1.,0.]],dtype=torch.float64,requires_grad=True)
    goal=torch.tensor([[2.,0.]],dtype=torch.float64,requires_grad=True)
    task=SoftCommandState(history_gradients=True).from_controls(history(x),controls,8,goal_xz=goal)
    value=torch.asinh(task[0,0]-x[0])
    dx,dc,dg=torch.autograd.grad(value,(x,controls,goal),allow_unused=True)
    torch.testing.assert_close(dx,-torch.ones_like(x)/torch.sqrt(1+(2-x.detach()).square()))
    assert dc is None and dg is None


@pytest.mark.parametrize('tokens',[6,8,16])
def test_idle_anchor_keeps_its_capture_graph_and_default_state_detaches(tokens):
    a=torch.tensor([0.],requires_grad=True);b=torch.tensor([.4],requires_grad=True)
    state=SoftCommandState(history_gradients=True)
    state.intent(history(a),[0.,0.],0.,tokens)
    task=state.intent(history(b),[0.,0.],0.,tokens)
    da,db=torch.autograd.grad(task[0,0]-b[0],(a,b))
    assert da.item()==1 and db.item()==-1
    live=SoftCommandState();out=live.intent(history(b),[0.,0.],0.,tokens)
    assert not out.requires_grad and not live.stop_goal.requires_grad
