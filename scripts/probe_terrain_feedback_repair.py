"""Matched full-rollout A/B for identifiable requested planar goals."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState,demonstration_controls,CONTROL_VERSION
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_model_v3 import JointTerrainController
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_terrain_overfit import select_cases,evaluate,save_motion
from scripts.train_joint_terrain_v3 import geometry_precision,intents,write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def rolling_intents(items,windows,frames,device,history=None,state=None):
    controls=torch.cat([demonstration_controls(w[None],frames,category=i['row'].get('control_category'),
        reference=i['world'][None].to(device)) for i,w in zip(items,windows)],0)
    return (state or SoftCommandState()).from_controls(windows[:,:4] if history is None else history,controls,frames//4)


@torch.no_grad()
def joystick_review(model,cases,geometry,output):
    rep=model.inference.global_motion_rep;rows=[]
    directory=output/'joystick';directory.mkdir()
    for name,speed in (('idle',0.),('walk',1.6)):
        case=next(i for i in cases if i['name']==name)
        source=case['world'][None].to('cuda:0');history=source[:,:4];origin=history[:,3,:3]
        yaw=torch.atan2(history[:,3,4],history[:,3,3]);state=SoftCommandState();parts=[]
        direction=torch.stack((yaw.sin(),yaw.cos()),-1)
        for _ in range(38):
            task=state.intent(history,direction*speed,yaw,8)
            result=model(history,task,[case['scene']],num_tokens=8)
            parts.append(result['world'][:,4:12])
            with geometry_precision():history=reencode_played_history(result['world'][:,8:12],rep)
        motion=torch.cat(parts,1);delta=motion[0,:,:3]-origin[0]
        metric=geometry.full_surface_metrics(motion,case['scene'],source,case['row']['source_id'])
        metric.update(requested_speed_mps=speed,
            progress_m=float((delta[-1,[0,2]]*direction[0]).sum()),
            root_xz_drift_max_m=float(delta[:,[0,2]].norm(dim=-1).max()),
            cross_track_endpoint_m=float((delta[-1,[0,2]]*torch.stack((direction[0,1],-direction[0,0]))).sum().abs()),
            height_range_m=float(motion[0,:,1].max()-motion[0,:,1].min()))
        row={'name':name,'input':'constant causal joystick, no source future goals or poses',
            'frames':motion.shape[1],'metrics':metric}
        rows.append(row);save_motion(directory/(name+'.npz'),motion,case['scene'],rep,row)
    write_json(directory/'metrics.json',{'cases':rows});return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dataset','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('rolling','requested'),required=True)
    p.add_argument('--updates',type=int,default=100)
    p.add_argument('--reset-every',type=int,default=0,help='Fixed teacher-history reset interval; zero uses the whole rollout')
    p.add_argument('--launch',action='store_true')
    args=p.parse_args()
    if args.output.exists():p.error('Use an isolated fresh A/B arm')
    if args.reset_every<0:p.error('reset-every must be nonnegative')
    if args.launch:
        from scripts.launch_joint_terrain_training import available_cards
        available_cards([2,3])
        with args.output.with_suffix('.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                *[value for value in sys.argv[1:] if value!='--launch']],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':process.pid,'output':str(args.output),'reset_every':args.reset_every}),flush=True);return
    args.output.mkdir();torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    torch.manual_seed(2333);np.random.seed(2333)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['schema']!='motionbricks.terrain_overfit_diagnostic.v1':raise ValueError('Use the declared same diagnostic initialization')
    if (sha256(args.source)!=checkpoint['contract']['source_sha256'] or
        sha256(args.dataset/'native_manifest.json')!=checkpoint['contract']['dataset_manifest_sha256']):
        raise ValueError('Reference corpus or source bundle changed')
    inference,bundle=load_terrain_bundle(args.source,'cuda:0');model=JointTerrainController(inference,bundle)
    model.set_learning_phase(False);model.load_state_dict(checkpoint['model_state'],strict=True)
    coefficients=checkpoint['contract']['coefficients']
    data=TerrainLearningDataset(args.dataset,build_native_representation(args.dataset/'stats'),'train')
    cases=select_cases(data);geometry=JointGeometryObjective(inference.global_motion_rep,'cuda:0')
    if [i['row']['source_id'] for i in cases]!=[i['source_id'] for i in checkpoint['contract']['cases']]:
        raise ValueError('A/B case identity changed')
    vocabulary={'encoder':state_digest(model.codec.encoder),'quantizer':state_digest(model.codec.quantizer)}
    contract={'schema':'motionbricks.feedback_goal_ab.v1','mode':args.mode,'updates':args.updates,
        'initial_weights_sha256':sha256(args.checkpoint),'initialization':str(args.checkpoint),
        'control_version':CONTROL_VERSION,'lr':5e-5,'optimizer':'fresh Adam, clip .5','seed':2333,
        'depth':13,'batch':4,'tokens':8,'coefficients':coefficients,'frozen_vocabulary':vocabulary,
        'reset_every':args.reset_every,
        'supervised_source_world_unchanged':True,'postprocessing':None,'future_height_pose_input':False,
        'source_contract':checkpoint['contract'],'comparison':'same updates, full-rollout depth, data, losses and optimizer; only requested goal semantics differ'}
    del checkpoint
    write_json(args.output/'contract.json',contract)
    builder=intents if args.mode=='requested' else rolling_intents
    initial=evaluate(model,cases,geometry,coefficients,args.output,'initial',task_builder=builder)
    optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=5e-5)
    parameters=[v for v in model.parameters() if v.requires_grad]
    source=torch.stack([i['world'] for i in cases]).to('cuda:0')
    scenes=[i['scene'] for i in cases];identities=[i['row']['source_id'] for i in cases]
    started=time.perf_counter()
    for step in range(args.updates):
        history=source[:,:4];state=SoftCommandState();optimizer.zero_grad(set_to_none=True);total=0.
        for segment in range(13):
            window=source[:,segment*8:segment*8+33]
            if args.reset_every and segment%args.reset_every==0:history=window[:,:4].detach()
            task=builder(cases,window,32,'cuda:0',history,state)
            result=model(history,task,scenes,source=window,num_tokens=8,straight_through=True)
            with geometry_precision():
                native=model.native_losses(result,window)
                physical=geometry.losses(result['world'],window,history,scenes,identities)
                loss=native['native_total']+sum(coefficients[key]*physical[key] for key in coefficients)
            if not torch.isfinite(loss):raise ValueError('Nonfinite A/B loss')
            (loss/13).backward();total+=float(loss.detach())/13
            with torch.no_grad(),geometry_precision():
                history=reencode_played_history(result['world'][:,8:12].detach(),inference.global_motion_rep)
        norm=torch.nn.utils.clip_grad_norm_(parameters,.5)
        if not torch.isfinite(norm):raise ValueError('Nonfinite A/B gradient')
        optimizer.step()
        if step%10==0 or step+1==args.updates:
            row={'mode':args.mode,'step':step+1,'loss':total,'gradient_norm':float(norm),
                'elapsed_seconds':time.perf_counter()-started}
            write_json(args.output/'progress.json',row);print(json.dumps(row),flush=True)
    if vocabulary!={'encoder':state_digest(model.codec.encoder),'quantizer':state_digest(model.codec.quantizer)}:
        raise ValueError('The token vocabulary changed')
    torch.save({'schema':contract['schema'],'model_state':model.state_dict(),'optimizer':optimizer.state_dict(),
        'contract':contract,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},args.output/'diagnostic.pt')
    final=evaluate(model,cases,geometry,coefficients,args.output,'final',task_builder=builder)
    joystick=joystick_review(model,cases,geometry,args.output)
    write_json(args.output/'complete.json',{'contract':contract,'initial':initial,'final':final,'joystick':joystick,
        'meaning':'small matched train-source experiment plus causal joystick probes; not a deployed model'})


if __name__=='__main__':main()
