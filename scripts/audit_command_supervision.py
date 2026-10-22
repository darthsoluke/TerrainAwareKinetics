"""Read-only episode-transition, motion-prior and support-interval audit."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.terrain_control import SoftCommandState
from scripts.train_joint_terrain_v3 import write_json


def interval_masks(sole, gap, known):
    speed=torch.diff(sole,dim=1).norm(dim=-1)*30
    certified=known[:,1:] & known[:,:-1] & (gap[:,1:].abs()<=.02) & (gap[:,:-1].abs()<=.02) & (speed<=.1)
    active=torch.zeros_like(known);active[:,:-1]|=certified;active[:,1:]|=certified
    reconstructed=active[:,1:] & active[:,:-1]
    return speed,certified,reconstructed


def actual_loss_counterexample():
    probe=object.__new__(JointGeometryObjective);probe.sole_count=1
    probe.geometry=lambda value:(value,value)
    probe.rep=SimpleNamespace(inverse=lambda value,*a,**k:{'posed_joints':value})
    plane=SimpleNamespace(query=lambda points:(points[...,1],torch.ones_like(points[...,1],dtype=torch.bool)))
    source=torch.zeros(1,12,1,3);source[:,6:8,:,0]=.01
    prediction=torch.zeros_like(source)
    prepared=dict(joints=source,points=source,volumes=plane,heights=plane)
    loss=probe.prepared_losses(prediction,prediction[:,:4],prepared,offset=0,diagnostics=False)
    target=source[:,4:8];gap,known=plane.query(target)
    speed,certified,reconstructed=interval_masks(target,gap,known)
    correct=(speed*certified).sum()/certified.sum().clamp_min(1)
    torch.testing.assert_close(loss['source_support_velocity_mps'],torch.tensor(.1),atol=1e-6,rtol=0)
    assert float(correct)==0
    return dict(source_x_m=target[0,:,0,0].tolist(),certified=certified[0,:,0].tolist(),
        actual_training_mask=reconstructed[0,:,0].tolist(),actual_slip_term=float(loss['source_support_velocity_mps']),
        certified_only_slip_term=float(correct),prediction='stationary point on zero plane',optimizer_updates=0)


def command_gradient_counterexample():
    def control_error(position, speed):
        base=torch.zeros(1,4,8,dtype=torch.float64);base[:,:,3]=1
        control=torch.tensor([[speed,0.,1.,0.]],dtype=torch.float64)
        driver=SoftCommandState();driver.from_controls(base,control,8)
        history=base.clone();history[:,:,0]=position
        intent=driver.from_controls(history.detach(),control,8)
        shift=history[:,3,[0,2]]-history[:,3,[0,2]].detach()
        intent=torch.cat((intent[:,:2]+shift,intent[:,2:]),-1)
        return torch.asinh(intent[:,:2]-history[:,3,[0,2]])[0,0]
    rows=[]
    with torch.enable_grad():
        for speed in (0.,1.6):
            position=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
            error=control_error(position,speed)
            gradient=float(torch.autograd.grad(error,position)[0])
            epsilon=1e-5
            finite=(control_error(torch.tensor(.4+epsilon,dtype=torch.float64),speed)
                -control_error(torch.tensor(.4-epsilon,dtype=torch.float64),speed))/(2*epsilon)
            rows.append(dict(request_speed=speed,control_feature=float(error.detach()),
                autograd_derivative=gradient,numerical_forward_derivative=float(finite)))
    assert abs(rows[0]['autograd_derivative'])<1e-12
    assert abs(rows[0]['numerical_forward_derivative']+1/(1+.4**2)**.5)<1e-8
    assert abs(rows[1]['numerical_forward_derivative'])<1e-8
    return dict(rows=rows,scope='Actual SoftCommandState and exact training shift/asinh expressions; retained stop anchor0, current X.4',
        issue='Unconditional zero-valued shift cancels the real stop-position feedback derivative')


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    cases=torch.load(args.cases,map_location='cpu',weights_only=False)
    anchors=torch.load(args.anchors,map_location='cpu',weights_only=False)
    rep=build_native_representation();geometry=JointGeometryObjective(rep,'cpu');rows=[]
    for case in cases+[c for c in anchors if c['pilot_split']=='train']:
        world=case['world'][None];segments=case['training_segments'];end=4+segments*8
        chunks=[geometry.geometry(world[:,start:start+32])[1] for start in range(0,end,32)]
        soles=torch.cat(chunks,1)[:,4:end,:geometry.sole_count]
        gap,known=heightfield_gap(soles,case['scene'])
        speed,certified,reconstructed=interval_masks(soles,gap,known)
        false=reconstructed & ~certified
        controls=case['controls'] if 'controls' in case else case['exact_constant_control'].repeat(segments,1)
        moving=controls[:,:2].norm(dim=-1)>1e-5
        rows.append(dict(label=case['label'],category=case.get('category','stairs'),segments=segments,
            native_actor_loss_calls_per_episode=1,later_plans_without_native_actor_loss=segments-1,
            moving_plans=int(moving.sum()),stopped_plans=int((~moving).sum()),
            command_change_count=int((moving[1:]!=moving[:-1]).sum()),
            stopped_to_moving_count=int((~moving[:-1]&moving[1:]).sum()),
            all_histories_reset_to_source_on_episode_entry=True,
            certified_support_material_intervals=int(certified.sum()),
            used_support_material_intervals=int(reconstructed.sum()),
            uncertified_bridged_intervals=int(false.sum()),
            uncertified_fraction_of_used=float(false.sum()/reconstructed.sum().clamp_min(1)),
            bridged_source_speed_max_mps=float(speed[false].max()) if false.any() else None))
    write_json(args.output,dict(cases=rows,counterexample=actual_loss_counterexample(),
        command_gradient=command_gradient_counterexample(),
        scope='Unrotated real source geometry; masks are kinematic proxies, not measured contacts',
        new_training_updates=0,model_and_losses_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cases','anchors'):p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gradient-only',action='store_true')
    args=p.parse_args()
    if args.gradient_only:write_json(args.output,command_gradient_counterexample())
    else:
        if args.cases is None or args.anchors is None:p.error('Full audit requires cases and anchors')
        main(args)
