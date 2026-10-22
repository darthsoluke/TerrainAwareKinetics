"""Plan planar reference motion before Pose conditioning, with bounded acceleration."""

import torch


def planar_root_reference(prediction, history, requested_velocity, *, acceleration_mps2=2.5, fps=30):
    """Keep terrain height/heading; connect known planar velocity to the latest intent."""
    if fps!=30 or not acceleration_mps2>0:
        raise ValueError('Root reference planning needs native30Hz and positive acceleration')
    dt=1/fps
    position=history[:,-1,[0,2]]
    velocity=(history[:,-1,[0,2]]-history[:,-2,[0,2]])/dt
    delta=requested_velocity-velocity
    duration=torch.linalg.vector_norm(delta,dim=-1,keepdim=True)/acceleration_mps2
    # Zero change has zero acceleration and keeps its existing velocity.
    acceleration=delta/duration.clamp_min(torch.finfo(prediction.dtype).eps)
    time=(torch.arange(prediction.shape[1]-4,device=prediction.device,dtype=prediction.dtype)+1)*dt
    elapsed=time[None,:,None]
    ramp=torch.minimum(elapsed,duration[:,None])
    travel=velocity[:,None]*elapsed+acceleration[:,None]*(ramp*elapsed-.5*ramp.square())
    future=prediction[:,4:].clone()
    future[:,:,[0,2]]=position[:,None]+travel
    return torch.cat((history[:,:4],future),dim=1)


def plan_root_conditions(inference,batch,root,*,acceleration_mps2=2.5):
    """Return mutually consistent normalized global/local Root conditions for Pose."""
    representation=inference.global_motion_rep
    predicted=representation.unnormalize(root)
    history=representation.unnormalize(batch['global_root_values'])[:,:4]
    requested=inference.local_motion_rep.unnormalize(batch['local_root_values'])[:,-1,1:3]
    planned=planar_root_reference(predicted,history,requested,acceleration_mps2=acceleration_mps2)
    global_values=representation.normalize(planned)
    local_values=inference.motion_rep.dual_rep.global_to_local(global_values,
        is_normalized=True,to_normalize=True,lengths=batch['pred_num_tokens']*4)
    from motionbricks.data.native_mb_online_conditions import apply_terminal_velocity
    local_values=apply_terminal_velocity(local_values,batch['local_root_values'],
        batch['local_root_component_mask'],batch['pred_num_tokens']*4)
    return global_values,local_values


def install_root_reference_plan(inference,*,acceleration_mps2):
    """Bind one explicit pre-Pose plan; its tensor arithmetic also runs in CUDA graphs."""
    if not 0<acceleration_mps2<=10:
        raise ValueError('Explicit walking-reference acceleration must be in (0,10]m/s2')
    native=inference._predict_root_trajectories
    def planned_root(batch,config):
        count,root,_=native(batch,config)
        root,local=plan_root_conditions(inference,dict(batch,pred_num_tokens=count),root,
            acceleration_mps2=acceleration_mps2)
        return count,root,local
    inference._predict_root_trajectories=planned_root
