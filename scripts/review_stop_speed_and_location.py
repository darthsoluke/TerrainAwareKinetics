"""Command-speed/location sensitivity, without limiting any generated movement."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.stop_command_adapter import StopCommandAdapter
from motionbricks.data.stop_stage_adapter import install_stop_stage
from motionbricks.data.stop_transition_adapter import enable_transition_gate, CommandTransitionState, set_transition_active
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_joint_sdf_contact_preserve import load_model
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.evaluate_joint_sdf import save_motion


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent=torch.load(args.parent,map_location='cpu',mmap=True,weights_only=False)
    model,_=load_model(parent);model.control=StopCommandAdapter(model.control)
    install_stop_stage(model,'pose');install_stop_stage(model,'decoder')
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(saved['model_state'],strict=True);model.requires_grad_(False);enable_transition_gate(model);model.eval()
    receipt=json.loads((args.artifact/'export.json').read_text())
    sources=torch.load(args.artifact/'offline-cases.pt',map_location='cpu',weights_only=False)
    rep=model.inference.global_motion_rep;geometry=JointGeometryObjective(rep,'cuda:0');rows=[]
    for entry,source in zip(receipt['entries'],sources):
        with np.load(args.artifact/(entry['entry']+'-scene.npz'),allow_pickle=False) as a:
            initial=torch.from_numpy(a['initial_history'].copy()).cuda()
            scene={k:torch.from_numpy(a[k].copy()).cuda() for k in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
        goal=source['world'][source['evaluation_goal_source_frame']-source['start_frame'],[0,2]].cuda()
        goal_point=source['world'][source['evaluation_goal_source_frame']-source['start_frame'],:3].cuda()
        initial_gap,_=heightfield_gap(initial[0,-1:,:3],scene)
        goal_gap,_=heightfield_gap(goal_point[None],scene)
        initial_floor=initial[0,-1,1]-initial_gap[0];goal_floor=goal_point[1]-goal_gap[0]
        direction=initial.new_tensor(entry['command_velocity_mps']);direction/=direction.norm()
        for requested_speed in (1.,1.5):
            for release_block in (2,3):
                history=initial.clone();driver=SoftCommandState();phase=CommandTransitionState()
                parts=[];stop_parts=[];stop_end=None;failure=None;commands=[];goal_reached=False;release_floor=None
                for block in range(64):
                    stopped=release_block<=block<release_block+10
                    velocity=direction*requested_speed if not stopped else direction*0
                    intent=driver.intent(history,velocity,entry['command_heading'],8)
                    observed_speed=float(((history[:,-1,[0,2]]-history[:,0,[0,2]])*10).norm())
                    set_transition_active(model,phase.update(velocity,observed_speed))
                    if block==release_block:
                        gap,known=heightfield_gap(history[0,-1:,:3],scene)
                        release_floor=float(history[0,-1,1]-gap[0]) if known.all() else None
                    try:result=model(history,intent,[scene])
                    except (ValueError,RuntimeError) as error:failure=str(error);break
                    future=result['world'][:,4:12].clone();parts.append(future)
                    history=reencode_played_history(result['world'][:,8:12],rep)
                    commands.append(dict(block=block,stopped=stopped,requested_speed_mps=float(velocity.norm())))
                    if stopped:stop_parts.append(future);stop_end=history[0,-1,:3].clone()
                    goal_reached=float((history[0,-1,[0,2]]-goal)@direction)>=0
                    if goal_reached and block>=release_block+14:break
                row=dict(entry=entry['entry'],requested_speed_mps=requested_speed,release_block=release_block,
                    frames=len(parts)*8,failure=failure,goal_reached=goal_reached,release_ground_height_m=release_floor,
                    no_output_speed_cap=True,quality_approved=False,commands=commands)
                if parts:
                    world=torch.cat(parts,1);row.update(exact_skin_metrics(geometry,world,scene))
                    if len(stop_parts)==10:
                        tail=torch.cat(stop_parts,1)[0,-15:,[0,2]]
                        speed=torch.diff(tail,dim=0).norm(dim=-1)*30
                        row.update(stop_tail_speed_p90_mps=float(torch.quantile(speed,.9)),
                            stop_tail_drift_m=float((tail[-1]-tail[0]).norm()),
                            resume_progress_m=float((world[0,-1,[0,2]]-stop_end[[0,2]])@direction))
                        gap,known=heightfield_gap(stop_end[None],scene);stop_floor=stop_end[1]-gap[0]
                        low,high=torch.minimum(initial_floor,goal_floor),torch.maximum(initial_floor,goal_floor)
                        inside=bool(known.all() and low+.05<stop_floor<high-.05)
                        row.update(stopped_inside_flight=inside,stop_resume_screen=bool(
                            torch.quantile(speed,.9)<=.15 and (tail[-1]-tail[0]).norm()<=.08
                            and inside and row['resume_progress_m']>=.25))
                    name=f"{entry['entry']}-speed{requested_speed:g}-release{release_block}"
                    save_motion(args.output/(name+'.npz'),world,scene,rep,row)
                rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='commands'}),flush=True)
                (args.output/'progress.json').write_text(json.dumps(dict(rows=rows),indent=2))
    (args.output/'complete.json').write_text(json.dumps(dict(rows=rows,updates=0,quality_approved=False,
        intervention='Different explicit user command speeds1/1.5mps and release times, NOT runtime clipping. Longer route budget accommodates genuine standing time.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','checkpoint','artifact','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
