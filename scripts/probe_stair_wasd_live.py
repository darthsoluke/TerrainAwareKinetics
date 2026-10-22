"""Exercise the real online backend with stair commands, never reference playback."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.joint_terrain_live import JointTerrainOnlineBackend
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_runtime_controls import JOINT_RUNTIME_DEFAULTS
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.soma_a100_trial_server import A100TrialRuntime
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


@torch.inference_mode()
def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((args.artifact/'export.json').read_text())
    for name, digest in receipt['files'].items():
        if sha256(args.artifact/name)!=digest:
            raise ValueError('Private export hash mismatch: '+name)
    if sha256(args.artifact/'weights.pt')!=receipt['checkpoint_sha256']:
        raise ValueError('Not the approved model')
    runtime=A100TrialRuntime(args.bundle,receipt['source_bundle_sha256'],'cuda:0',
        args.artifact/'up-scene.npz','Approved stair gait: live probe','eager',True,args.artifact/'weights.pt')
    model,rep=runtime.joint_controller,runtime.inference.global_motion_rep
    geometry=JointGeometryObjective(rep,'cuda:0')
    cases=torch.load(args.artifact/'offline-cases.pt',map_location='cpu',weights_only=False)
    rows=[]
    for case, entry in zip(cases,receipt['entries']):
        with np.load(args.artifact/(entry['entry']+'-scene.npz'),allow_pickle=False) as a:
            initial=torch.from_numpy(a['initial_history'].copy()).cuda()
            scene={k:torch.from_numpy(a[k].copy()).cuda() for k in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
        velocity=entry['command_velocity_mps'];heading=entry['command_heading']
        goal=case['world'][case['evaluation_goal_source_frame']-case['start_frame'],[0,2]].cuda()
        goal_height=float(case['world'][case['evaluation_goal_source_frame']-case['start_frame'],1])
        axis=goal-initial[0,3,[0,2]];axis/=axis.norm()
        for profile in args.profiles:
            backend=JointTerrainOnlineBackend(model,initial,scene)
            backend.configure_runtime(dict(JOINT_RUNTIME_DEFAULTS))
            parts=[];times=[];commands=[];failure=None;stop_progress=[]
            for block in range(args.stop_blocks+24):
                speed=float(entry['command_speed_mps']) if profile=='exact_cruise' else 3.
                v=np.asarray(velocity)/np.linalg.norm(velocity)*speed
                yaw=heading if profile=='exact_cruise' else entry['native_heading_of_travel']
                stopped=(profile=='release_resume' and 3<=block<3+args.stop_blocks) or (profile=='idle_start' and block<args.stop_blocks)
                if stopped:
                    v=np.zeros(2)
                commands.append(dict(block=block,velocity=v.tolist(),heading=yaw))
                try:
                    if block==0 and profile=='exact_cruise':
                        intent=SoftCommandState().intent(initial,v,yaw,8)
                        plain=model(initial,intent,[scene])['world'][:,4:12].clone()
                    value=backend.predict_plan(v,yaw)
                    future=value['planned_motion']
                    commands[-1]['root_stage_endpoint']=value['debug']['root_stage_world_xyz_m'][-1]
                    commands[-1]['decoded_endpoint']=value['debug']['decoded_world_xyz_m'][-1]
                    if block==0 and profile=='exact_cruise':
                        parity=float((plain-future).abs().max())
                        torch.testing.assert_close(plain,future,atol=3e-5,rtol=1e-5)
                    torch.cuda.synchronize();started=time.perf_counter()
                    runtime.skin_features_with_joints(future,runtime.inference)
                    torch.cuda.synchronize()
                    times.append(dict(inference_ms=value['generation_seconds']*1000,skin_ms=(time.perf_counter()-started)*1000))
                    parts.append(future.clone())
                    backend.history=value['published_history'].clone()
                    if stopped:
                        stop_progress.append(backend.history[0,-1,:3].cpu().tolist())
                        commands[-1]['published_mean_speed_mps']=float(torch.diff(future[0,:,[0,2]],dim=0).norm(dim=-1).mean()*30)
                    reached=float((backend.history[0,-1,[0,2]]-goal) @ axis)>=0
                    resumed=profile not in ('release_resume','idle_start') or block>=(3+args.stop_blocks if profile=='release_resume' else args.stop_blocks)
                    if reached and resumed:
                        break
                except (ValueError,RuntimeError) as error:
                    failure=str(error);break
            row=dict(entry=entry['entry'],profile=profile,frames=len(parts)*8,failure=failure,
                commands=commands,latencies=times,quality_approved=False,postprocessing=None)
            if parts:
                world=torch.cat(parts,1)
                row.update(exact_skin_metrics(geometry,world,scene))
                row.update(goal_reached=float((world[0,-1,[0,2]]-goal) @ axis)>=0,
                    progress_m=float((world[0,-1,[0,2]]-initial[0,3,[0,2]]) @ axis),
                    root_height_change_m=float(world[0,-1,1]-initial[0,3,1]),
                    goal_height_error_m=abs(float(world[0,-1,1])-goal_height),
                    source_root_height_change_m=goal_height-float(initial[0,3,1]),
                    generation_median_ms=float(np.median([t['inference_ms'] for t in times])),
                    generation_p95_ms=float(np.quantile([t['inference_ms'] for t in times],.95)),
                    peak_cuda_bytes=torch.cuda.max_memory_allocated())
                if profile=='exact_cruise':row['live_default_first_block_error']=parity
                if stop_progress:
                    row['root_travel_during_stop_blocks_m']=float(np.linalg.norm(np.diff(np.asarray(stop_progress)[:,[0,2]],axis=0),axis=-1).sum())
                    row['last_stop_block_speed_mps']=[c['published_mean_speed_mps'] for c in commands if 'published_mean_speed_mps' in c][-1]
                save_motion(args.output/(entry['entry']+'-'+profile+'.npz'),world,scene,rep,row)
            rows.append(row)
            write_json(args.output/'progress.json',dict(rows=rows,quality_approved=False))
            print(json.dumps({k:v for k,v in row.items() if k not in ('commands','latencies')}),flush=True)
    write_json(args.output/'complete.json',dict(rows=rows,checkpoint_sha256=receipt['checkpoint_sha256'],
        keyboard_test='command-equivalent backend probe; actual browser keys still required',quality_approved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('artifact','bundle','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--profiles',nargs='+',choices=('exact_cruise','keyboard_cruise','release_resume','idle_start'),default=['exact_cruise','keyboard_cruise','release_resume'])
    parser.add_argument('--stop-blocks',type=int,default=4)
    main(parser.parse_args())
