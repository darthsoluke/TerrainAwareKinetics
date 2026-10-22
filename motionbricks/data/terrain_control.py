"""Shared causal task construction for offline supervision and online WASD."""

import torch


CONTROL_VERSION='explicit-planar-request-independent-facing-v2'
CONTROL_GRADIENT_VERSION='release-anchor-history-autograd-v1'


class SoftCommandState:
    """Retain a requested stop location; never alter the generated trajectory."""
    def __init__(self, *, history_gradients=False):
        if type(history_gradients) is not bool:
            raise TypeError('history_gradients must be boolean')
        self.history_gradients=history_gradients
        self.stop_goal=None
        self.was_moving=None

    def from_controls(self,history,controls,tokens,*,goal_xz=None):
        if history.ndim!=3 or history.shape[1]!=4 or history.shape[2]<5:
            raise ValueError('Control construction requires four observed root/body frames')
        if not isinstance(tokens,int) or not 6<=tokens<=16:
            raise ValueError('Use the native 6..16-token horizon')
        value=torch.as_tensor(controls,device=history.device,dtype=history.dtype).detach()
        if value.shape!=(len(history),4) or not torch.isfinite(value).all():
            raise ValueError('Controls are [B, vx, vz, cos(body_yaw), sin(body_yaw)]')
        facing=value[:,2:4]
        if not torch.allclose(facing.norm(dim=-1),torch.ones_like(facing[:,0]),atol=1e-4,rtol=0):
            raise ValueError('Body-facing heading must be a unit vector, independent of travel')
        current=history[:,-1,[0,2]]
        if not self.history_gradients:current=current.detach()
        moving=value[:,:2].norm(dim=-1)>1e-5
        if self.stop_goal is None:
            self.stop_goal=current.clone();self.was_moving=moving.clone()
        if self.stop_goal.shape!=current.shape:raise ValueError('Reset command state when the batch changes')
        reset=moving|self.was_moving
        # During differentiable rollouts, retain the release-time state graph,
        # not a new dependency on each later stopped position.
        self.stop_goal=torch.where(reset[:,None],current,self.stop_goal)
        if not self.history_gradients:self.stop_goal=self.stop_goal.detach()
        goal=torch.where(moving[:,None],current+value[:,:2]*((tokens*4-4)/30),self.stop_goal)
        if goal_xz is not None:
            requested=torch.as_tensor(goal_xz,device=history.device,dtype=history.dtype).detach()
            if requested.shape!=(len(history),2) or not torch.isfinite(requested).all():
                raise ValueError('Requested goal is finite planar XZ only; no future height or pose')
            goal=requested
        self.was_moving=moving.detach().clone()
        result=torch.cat((goal,value[:,:2],facing),-1)
        return result if self.history_gradients else result.detach()

    def intent(self,history,velocity,heading,tokens,*,goal_xz=None):
        velocity=torch.as_tensor(velocity,device=history.device,dtype=history.dtype).reshape(-1,2)
        yaw=torch.as_tensor(heading,device=history.device,dtype=history.dtype).reshape(-1)
        velocity=velocity.expand(len(history),-1);yaw=yaw.expand(len(history))
        control=torch.cat((velocity,yaw.cos()[:,None],yaw.sin()[:,None]),-1)
        return self.from_controls(history,control,tokens,goal_xz=goal_xz)


def demonstration_controls(world,frames,*,category=None,override=None,reference=None):
    """Infer weak task labels from real motion; these are not recorded key presses.

    Future demonstrated XZ motion estimates the requested speed/direction. Body
    heading remains the demonstrated heading. No future Y or pose is a control.
    Starts use the available real clip's cruise-speed estimate so a stationary
    prefix can be paired with a nonzero motion request before acceleration.
    """
    if world.ndim!=3 or world.shape[1]<frames+1:
        raise ValueError('Control labels require source interval endpoints')
    if override is not None:
        if override.shape!=(len(world),4):raise ValueError('Legacy fixed-goal weak labels are not supported')
        return override.detach()
    velocity=(world[:,frames-1,[0,2]]-world[:,3,[0,2]])/((frames-4)/30)
    if category=='idle':
        velocity=torch.zeros_like(velocity)
    elif category=='stop':
        # A clip-level "stop" label also contains its moving approach. Label
        # the actual terminal part of this window, not the whole file as idle.
        terminal=(world[:,frames-7:frames+1,[0,2]]-world[:,frames-8:frames,[0,2]])*30
        velocity=terminal.mean(1)
        resting=terminal.norm(dim=-1).amax(1)<=.12
        velocity=torch.where(resting[:,None],torch.zeros_like(velocity),velocity)
    elif category=='start':
        motion=world if reference is None else reference
        steps=(motion[:,1:,[0,2]]-motion[:,:-1,[0,2]])*30
        speed=torch.quantile(steps.norm(dim=-1),.9,dim=1)
        delta=motion[:,-1,[0,2]]-motion[:,0,[0,2]]
        direction=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        velocity=direction*speed[:,None]
    return torch.cat((velocity,world[:,frames-1,3:5]),-1).detach()
