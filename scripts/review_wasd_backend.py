"""Real online backend under explicit WASD/release inputs, without future reference."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.bones_seed_rehearsal import BonesSeedPool
from motionbricks.data.joint_terrain_live import JointTerrainOnlineBackend
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_runtime_controls import JOINT_RUNTIME_DEFAULTS
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_contact_preserve import load_model
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    summaries={}
    for arm in args.arms:
        label,path=arm.split('=',1);path=Path(path)
        model,_=load_model(torch.load(path,map_location='cpu',mmap=True,weights_only=False))
        rep=model.inference.global_motion_rep;geometry=JointGeometryObjective(rep,'cuda:0')
        pool=BonesSeedPool(args.bones_dataset,rep,args.stats)
        case=next(c for c in pool.evaluation if c['category']=='walk')
        initial=case['world'][None,:4].cuda();scene=case['scene']
        backend=JointTerrainOnlineBackend(model,initial,scene)
        backend.configure_runtime(dict(JOINT_RUNTIME_DEFAULTS))
        yaw=float(torch.atan2(initial[0,3,4],initial[0,3,3]));rows=[];all_parts=[]
        phases=[('W',yaw,1.6,6),('release_W',yaw,0.,12),
                ('A',yaw+torch.pi/2,1.6,6),('release_A',yaw+torch.pi/2,0.,6),
                ('D',yaw-torch.pi/2,1.6,6),('release_D',yaw-torch.pi/2,0.,6),
                ('S',yaw+torch.pi,1.6,6),('release_S',yaw+torch.pi,0.,6),('W_restart',yaw,1.6,6)]
        for name,heading,speed,blocks in phases:
            direction=initial.new_tensor([heading]).sin().item(),initial.new_tensor([heading]).cos().item()
            velocity=[v*speed for v in direction];parts=[];failure=None;start=backend.history[0,-1,:3].clone()
            for block in range(blocks):
                try:
                    result=backend.predict_plan(velocity,heading)
                    parts.append(result['planned_motion']);backend.history=result['published_history'].clone()
                except ValueError as error:failure=str(error);break
            value=dict(key=name,request_velocity_xz_mps=velocity,request_heading=heading,
                frames=len(parts)*8,requested_frames=blocks*8,failure=failure)
            if parts:
                world=torch.cat(parts,1);all_parts.append(world)
                velocities=torch.diff(world[0,:,[0,2]],dim=0)*30
                tail=velocities[-16:];delta=world[0,-1,[0,2]]-start[[0,2]]
                value.update(displacement_xz_m=delta.cpu().tolist(),
                    tail_speed_p90_mps=float(torch.quantile(tail.norm(dim=-1),.9)),
                    tail_speed_along_requested_direction_mps=float((tail*tail.new_tensor(direction)).sum(-1).mean()),
                    drift_after_one_second_m=float((world[0,-1,[0,2]]-world[0,min(30,world.shape[1]-1),[0,2]]).norm()))
            rows.append(value)
            if failure:break
        report=dict(source_id=case['row']['source_id'],source_start_frame=case['source_start_frame'],phases=rows,
            source_future_frames_used=False,recorded_motion_replay=False,actual_browser_keyboard_verified=False,
            backend='JointTerrainOnlineBackend, same native live defaults, serial fully-played8-frame boundary',
            prefetch_latency_not_measured=True,postprocessing=None,checkpoint_sha256=sha256(path))
        if all_parts:
            world=torch.cat(all_parts,1);report['geometry']=exact_skin_metrics(geometry,world,scene)
            save_motion(args.output/(label+'-wasd.npz'),world,scene,rep,report)
        summaries[label]=report;write_json(args.output/'progress.json',summaries)
        del backend,model,geometry,pool
        torch.cuda.empty_cache()
    write_json(args.output/'complete.json',dict(arms=summaries,optimizer_updates=0,quality_approved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bones-dataset','stats','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arms',nargs='+',required=True)
    main(p.parse_args())
