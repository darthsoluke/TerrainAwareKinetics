"""Train-only memorization probe with unchanged joint-controller architecture/loss."""

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
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_model_v3 import JointTerrainController,SCHEMA
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_traversal_metrics import traversal_window
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import geometry_precision,intents,write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def select_cases(data):
    cases=[]
    for name in ('idle','walk','stairs','climbing'):
        indices=[i for i,r in enumerate(data.base.records) if (
            r['origin']=='bones_seed' and r.get('control_category')==name if name in ('idle','walk') else
            r['origin'].startswith('parc') and r.get('terrain_family')==name) and r['frames_30hz']>=129]
        if not indices:raise ValueError('No real 129-frame training clip for '+name)
        index=sorted(indices,key=lambda i:data.records[i]['source_id'])[0]
        item=data.full_reference_item(index);full=item['world']
        if name=='idle':
            speed=(full[1:,:3]-full[:-1,:3]).norm(dim=-1)*30
            start=int(speed.unfold(0,128,1).amax(-1).argmin())
        else:start,_,_=traversal_window(full[:,:3].numpy(),3.5)
        item['world']=full[start:start+129].clone().detach()
        if len(item['world'])!=129:raise ValueError('Do not pad an overfit reference')
        if name in ('idle','walk'):
            center=item['world'][3,:3]
            if torch.count_nonzero(item['scene']['heightfield_y_m']):raise ValueError('BONES flat assumption changed')
            item['scene']={'heightfield_y_m':torch.zeros(201,201),
                'min_point_world_xyz_m':torch.tensor([center[0]-50,0,center[2]-50]),'dx_m':torch.tensor(.5)}
        item.update(name=name,start_frame=start)
        cases.append(item)
    return cases


def save_motion(path,world,scene,rep,metadata):
    with torch.no_grad():pose=rep.inverse(world,False,return_all=True,joint_positions_from='global_rot_data')
    np.savez_compressed(path,fps=30,world=world[0].cpu().numpy(),roots=world[0,:,:3].cpu().numpy(),
        local_rot_mats=pose['local_joint_rots'][0].cpu().numpy(),
        heightfield_y_m=torch.as_tensor(scene['heightfield_y_m']).cpu().numpy(),
        min_point_world_xyz_m=torch.as_tensor(scene['min_point_world_xyz_m']).cpu().numpy(),
        dx_m=float(scene['dx_m']),metadata_json=np.asarray(json.dumps(metadata)))


@torch.no_grad()
def evaluate(model,cases,geometry,coefficients,output,label,*,task_builder=None):
    rep=model.inference.global_motion_rep;device=next(model.parameters()).device
    directory=output/label;directory.mkdir()
    report={'stage':label,'cases':[],'held_out':False,'postprocessing':None}
    for item in cases:
        source=item['world'][None].to(device);scene=item['scene'];identity=item['row']['source_id']
        source_metric=geometry.full_surface_metrics(source[:,4:108],scene,source,identity)
        for mode in ('teacher_history','generated_history'):
            history=source[:,:4];state=SoftCommandState();parts=[];losses=[];failure=None
            for offset in range(0,104,8):
                window=source[:,offset:offset+33]
                if mode=='teacher_history':history=window[:,:4];state=SoftCommandState()
                task=(task_builder or intents)([item],window,32,device,history,state)
                try:result=model(history,task,[scene],source=window,num_tokens=8)
                except ValueError as error:
                    if str(error)!='Current terrain origin is unobserved; do not invent a ground height' or not parts:raise
                    failure={'offset':offset,'reason':str(error)};break
                with geometry_precision():
                    native=model.native_losses(result,window)
                    physical=geometry.losses(result['world'],window,history,[scene],[identity])
                    losses.append({k:float(native[k]) for k in ('native_motion','native_root','token_ce','native_total')})
                    losses[-1]['weighted_total']=float(native['native_total']+sum(coefficients[k]*physical[k] for k in coefficients))
                    parts.append(result['world'][:,4:12])
                    history=reencode_played_history(result['world'][:,8:12],rep)
            prediction=torch.cat(parts,1);truth=source[:,4:4+prediction.shape[1]]
            with geometry_precision():
                pj=rep.inverse(prediction,False,return_all=False,joint_positions_from='global_rot_data')['posed_joints']
                sj=rep.inverse(truth,False,return_all=False,joint_positions_from='global_rot_data')['posed_joints']
                metric=geometry.full_surface_metrics(prediction,scene,source,identity)
            delta=prediction[...,:3]-truth[...,:3]
            metric.update(fk_mpjpe_m=float((pj-sj).norm(dim=-1).mean()),
                root_rmse_m=float(delta.square().sum(-1).mean().sqrt()),
                height_mae_m=float(delta[...,1].abs().mean()),generated_frames=prediction.shape[1],requested_frames=104,
                source_body_penetration_p99_m=source_metric['body_penetration_p99_m'],
                losses={key:float(np.mean([r[key] for r in losses])) for key in losses[0]})
            limit=.05 if mode=='teacher_history' else .15
            row={'name':item['name'],'mode':mode,'source_id':identity,'source_group':item['row']['source_group'],
                'start_frame':item['start_frame'],'split':'train','metrics':metric,'failure':failure,
                'requested_frames':104,'emitted_frames':prediction.shape[1],
                'memorization_screen_passed':failure is None and metric['fk_mpjpe_m']<=limit and metric['root_rmse_m']<=limit}
            report['cases'].append(row)
            save_motion(directory/(item['name']+'-'+mode+'.npz'),prediction,scene,rep,row)
            save_motion(directory/(item['name']+'-'+mode+'-source.npz'),source[:,4:108],scene,rep,
                        dict(row,role='unchanged_reference'))
            write_json(directory/'metrics.json',report)
    print(json.dumps({'review':label,'cases':[{'name':r['name'],'mode':r['mode'],
        'fk_m':r['metrics']['fk_mpjpe_m'],'root_m':r['metrics']['root_rmse_m'],
        'pass':r['memorization_screen_passed']} for r in report['cases']]}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dataset','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps-per-stage',type=int,default=600)
    p.add_argument('--launch',action='store_true')
    args=p.parse_args()
    if args.output.exists():p.error('Use a fresh diagnostic output')
    if args.launch:
        log=args.output.with_suffix('.log')
        with log.open('x') as handle:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                *[v for v in sys.argv[1:] if v!='--launch']],stdin=subprocess.DEVNULL,
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':process.pid,'log':str(log),'output':str(args.output)}),flush=True);return
    args.output.mkdir();torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    torch.manual_seed(2333);np.random.seed(2333)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['schema']!=SCHEMA or saved['contract']['source_bundle_sha256']!=sha256(args.source):
        raise ValueError('The source/checkpoint pair does not match the main run')
    if saved['contract']['dataset_manifest_sha256']!=sha256(args.dataset/'native_manifest.json'):
        raise ValueError('Use the unchanged main training corpus')
    inference,bundle=load_terrain_bundle(args.source,'cuda:0')
    model=JointTerrainController(inference,bundle);model.set_learning_phase(False)
    model.load_state_dict(saved['model_state'],strict=True)
    coefficients=saved['coefficients'];del saved
    vocabulary={'encoder':state_digest(model.codec.encoder),'quantizer':state_digest(model.codec.quantizer)}
    data=TerrainLearningDataset(args.dataset,build_native_representation(args.dataset/'stats'),'train')
    cases=select_cases(data);geometry=JointGeometryObjective(inference.global_motion_rep,'cuda:0')
    contract={'schema':'motionbricks.terrain_overfit_diagnostic.v1','checkpoint':str(args.checkpoint),
        'checkpoint_sha256':sha256(args.checkpoint),'source_sha256':sha256(args.source),
        'dataset_manifest_sha256':sha256(args.dataset/'native_manifest.json'),
        'coefficients':coefficients,'steps_per_stage':args.steps_per_stage,'seed':2333,
        'lr':5e-5,'optimizer':'fresh Adam, gradient clip 0.5','tokens':8,'batch':4,
        'source_motion_unchanged':True,'postprocessing':None,'held_out':False,
        'stages':['fixed-corpus teacher windows','same two-primitive generated-history exposure as main trainer'],
        'quality_approved':False,'vocabulary':vocabulary,
        'cases':[{'name':i['name'],'source_id':i['row']['source_id'],'source_group':i['row']['source_group'],
                  'split':i['row']['split'],'start_frame':i['start_frame'],'frames':len(i['world'])} for i in cases]}
    write_json(args.output/'contract.json',contract)
    reports=[evaluate(model,cases,geometry,coefficients,args.output,'initial')]
    optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=5e-5)
    parameters=[v for v in model.parameters() if v.requires_grad];device=next(model.parameters()).device
    scenes=[i['scene'] for i in cases];identities=[i['row']['source_id'] for i in cases]
    started=time.perf_counter()
    for stage,depth in (('teacher',1),('feedback',2)):
        for step in range(args.steps_per_stage):
            rng=np.random.default_rng(2333+step)
            starts=rng.integers(0,14-depth,size=len(cases))*8
            source=torch.stack([i['world'][start:start+33+(depth-1)*8] for i,start in zip(cases,starts)]).to(device)
            history=source[:,:4];state=SoftCommandState();optimizer.zero_grad(set_to_none=True);values=[]
            for segment in range(depth):
                window=source[:,segment*8:segment*8+33]
                task=intents(cases,window,32,device,history,state)
                result=model(history,task,scenes,source=window,num_tokens=8,straight_through=True)
                with geometry_precision():
                    native=model.native_losses(result,window)
                    physical=geometry.losses(result['world'],window,history,scenes,identities)
                    loss=native['native_total']+sum(coefficients[key]*physical[key] for key in coefficients)
                if not torch.isfinite(loss):raise ValueError('Nonfinite diagnostic loss')
                (loss/depth).backward();values.append(float(loss.detach()))
                with torch.no_grad(),geometry_precision():
                    history=reencode_played_history(result['world'][:,8:12].detach(),inference.global_motion_rep)
            norm=torch.nn.utils.clip_grad_norm_(parameters,.5)
            if not torch.isfinite(norm):raise ValueError('Nonfinite diagnostic gradient')
            optimizer.step()
            if step%50==0 or step+1==args.steps_per_stage:
                progress={'stage':stage,'step':step+1,'loss':float(np.mean(values)),
                    'gradient_norm':float(norm),'elapsed_seconds':time.perf_counter()-started}
                write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if vocabulary!={'encoder':state_digest(model.codec.encoder),'quantizer':state_digest(model.codec.quantizer)}:
            raise ValueError('Diagnostic changed the frozen vocabulary')
        torch.save({'schema':contract['schema'],'model_state':model.state_dict(),'contract':contract,'stage':stage},
                   args.output/(stage+'-diagnostic.pt'))
        reports.append(evaluate(model,cases,geometry,coefficients,args.output,stage))
    write_json(args.output/'complete.json',{'status':'complete','contract':contract,'reports':reports,
        'meaning':'train-source memorization diagnosis only; not held-out quality or a deployable model'})


if __name__=='__main__':main()
