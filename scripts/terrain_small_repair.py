"""Bounded matched source-event experiment; no live output corrections."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState, demonstration_controls
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_learning_episode import TerrainLearningEpisode
from motionbricks.data.terrain_model_v3 import JointTerrainController, SCHEMA
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.terrain_exact_interior import ExactTerrainInterior,PackedExactInteriors
from scripts.portable_parc_online_soma import load_terrain_bundle
from probe_terrain_overfit import save_motion
from scripts.train_joint_terrain_v3 import intents, geometry_precision, write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def initialize(checkpoint,detail=False):
    torch.set_num_threads(4)
    torch.manual_seed(2333);np.random.seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    contract=saved['contract']
    if sha256(Path(contract['source']))!=contract['source_bundle_sha256']:
        raise ValueError('Source bundle differs from checkpoint')
    infer,bundle=load_terrain_bundle(Path(contract['source']),'cuda:0')
    detail=detail or contract.get('small_repair',{}).get('detail_adapters',False)
    if detail:
        from motionbricks.data.terrain_detail_adapter import TerrainDetailController
        model=TerrainDetailController(infer,bundle)
    else:model=JointTerrainController(infer,bundle)
    model.set_learning_phase(False)
    if detail and not contract.get('small_repair',{}).get('detail_adapters',False):
        loaded=model.load_state_dict(saved['model_state'],strict=False)
        if loaded.unexpected_keys or any(not k.startswith(('root_detail','pose_detail','decoder_detail')) for k in loaded.missing_keys):
            raise ValueError('Only explicitly new zero-gated near adapters may be missing')
        model.initialize_detail_from_far()
    else:model.load_state_dict(saved['model_state'],strict=True)
    return model,saved


@torch.no_grad()
def source_screen(case,geometry):
    world=case['world'][None].to('cuda:0');rep=geometry.rep;points=[]
    for part in world[:,4:108].split(8,dim=1):
        pose=rep.inverse(part,False,return_all=True,joint_positions_from='global_rot_data')
        points.append(geometry.skin.skin(pose['local_joint_rots'][0],part[0,:,:3,None].transpose(1,2).expand(-1,30,-1),rot_is_global=False))
    points=torch.cat(points)
    volume=geometry.volume(case['scene'],world,case['row']['source_id'])
    distances,known=volume.query(points)
    deepest=torch.topk((-distances).masked_fill(~known,-torch.inf).reshape(-1),128).indices
    query=ExactTerrainInterior(case['scene'])
    exact,valid=query.query(points.reshape(-1,3)[deepest])
    report={'probe_vertices':128,'all_known':bool(valid.all()),'exact_max_penetration_m':float((-exact).clamp_min(0).max())}
    report['accepted']=report['all_known'] and report['exact_max_penetration_m']<=.03
    return report


def prepare(checkpoint,output,screen=False,stairs_first=False):
    torch.set_num_threads(4)
    output.mkdir(parents=True,exist_ok=False)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    dataset=Path(saved['contract']['dataset'])
    rep=build_native_representation(dataset/'stats')
    results=[];screened=[];geometry=None
    if screen:
        model,_=initialize(checkpoint)
        geometry=JointGeometryObjective(model.inference.global_motion_rep,'cuda:0')
    for split in ('train','validation'):
        data=TerrainLearningDataset(dataset,rep,split)
        for family in ('stairs','climbing','mixed','idle','walk'):
            if stairs_first and family=='climbing':continue
            ids=[i for i,r in enumerate(data.records) if r['frames_30hz']>=129 and
                 (family in str(r.get('terrain_family','')) if family not in ('idle','walk') else
                  r.get('origin')=='bones_seed' and r.get('control_category')==family)]
            ids=sorted(ids,key=lambda i:data.records[i]['source_id'])[:12]
            candidates=[]
            for index in ids:
                item=data.full_reference_item(index);world=item['world']
                if len(world)<129:continue
                if family in ('idle','walk'):
                    speed=(world[1:,[0,2]]-world[:-1,[0,2]]).norm(dim=-1)*30
                    starts=(speed.unfold(0,128,16).mean(1)-(.0 if family=='idle' else 1.6)).abs()
                    start=int(starts.argmin())*16
                    candidates.append((float(starts.min()),start,item))
                else:
                    delta=world[107:-21,1]-world[3:-125,1]
                    for direction,sign in [('up',1),('down',-1)]:
                        start=int((delta*sign).argmax());rise=float(delta[start])
                        if rise*sign>.2:candidates.append((direction,rise*sign,start,item))
            directions=['flat'] if family in ('idle','walk') else ['up','down']
            for direction in directions:
                options=candidates if direction=='flat' else [c for c in candidates if c[0]==direction]
                if not options:continue
                selected=None
                ranked=sorted(options,key=lambda c:c[0] if direction=='flat' else -c[1])
                for candidate in ranked:
                    if direction=='flat':_,start,item=candidate
                    else:_,_,start,item=candidate
                    possible=dict(item,world=item['world'][start:start+129].clone(),
                        start_frame=start,name=family if direction=='flat' else family+'_'+direction,split=split)
                    if geometry is not None and direction!='flat':
                        assessment=source_screen(possible,geometry)
                        screened.append(dict(assessment,name=possible['name'],split=split,source_id=item['row']['source_id'],start_frame=start))
                        print(json.dumps({'source_screen':screened[-1]}),flush=True)
                        if not assessment['accepted']:continue
                    selected=possible;break
                if selected is None:continue
                if family in ('idle','walk'):
                    center=selected['world'][3,:3]
                    selected['scene']={'heightfield_y_m':torch.zeros(201,201),
                        'min_point_world_xyz_m':torch.tensor([center[0]-50,0,center[2]-50]),'dx_m':torch.tensor(.5)}
                results.append(selected)
    train=[r for r in results if r['split']=='train']
    required={'stairs_up','stairs_down','climbing_up','climbing_down','idle','walk'}
    if stairs_first:required-={'climbing_up','climbing_down'}
    if not required.issubset({r['name'] for r in train}):
        raise ValueError('Missing real directional events: '+str(required-{r['name'] for r in train}))
    train_groups={r['row']['source_group'] for r in train}
    if any(r['row']['source_group'] in train_groups for r in results if r['split']=='validation'):
        raise ValueError('Source-family leakage')
    contract={'schema':'motionbricks.terrain_small_cases.v1','checkpoint_sha256':sha256(checkpoint),
        'dataset_manifest_sha256':saved['contract']['dataset_manifest_sha256'],
        'window_frames':129,'publish_frames':8,'feedback_depth':13,'future_height_pose_input':False,
        'source_screen':screened,
        'scope':'stairs and mixed, climbing excluded pending source repair' if stairs_first else 'stairs and climbing',
        'cases':[{'name':r['name'],'split':r['split'],'source_id':r['row']['source_id'],
            'source_group':r['row']['source_group'],'start_frame':r['start_frame'],
            'rise_104f_m':float(r['world'][107,1]-r['world'][3,1])} for r in results]}
    torch.save(results,output/'cases.pt');write_json(output/'cases.json',contract)
    print(json.dumps(contract),flush=True)


@torch.no_grad()
def review(model,cases,geometry,output):
    output.mkdir(parents=True,exist_ok=False)
    rep=model.inference.global_motion_rep;rows=[]
    for case in cases:
        source=case['world'][None].to('cuda:0');scene=case['scene'];identity=case['row']['source_id']
        source_body=geometry.full_surface_metrics(source[:,4:108],scene,source,identity)
        for mode in ('reference','rolling'):
            history=source[:,:4].clone();state=SoftCommandState();parts=[];failure=None
            for offset in range(0,104,8):
                window=source[:,offset:offset+33]
                if mode=='reference':task=intents([case],window,32,'cuda:0',history,state)
                else:
                    controls=demonstration_controls(window,32,category=case['row'].get('control_category'),reference=source)
                    task=state.from_controls(history,controls,8)
                try:result=model(history,task,[scene])
                except ValueError as error:
                    failure=str(error);break
                parts.append(result['world'][:,4:12])
                history=reencode_played_history(result['world'][:,8:12],rep)
            row={'name':case['name'],'split':case['split'],'mode':mode,'source_id':identity,
                'requested_frames':104,'emitted_frames':sum(x.shape[1] for x in parts),'failure':failure,
                'source_body_penetration_p99_m':source_body['body_penetration_p99_m']}
            if parts:
                prediction=torch.cat(parts,1);truth=source[:,4:4+prediction.shape[1]]
                pj,pp=geometry.geometry(prediction);sj,sp=geometry.geometry(truth)
                gap,known=heightfield_gap(pp[0,:,:geometry.sole_count],scene)
                source_gap,source_known=heightfield_gap(sp[0,:,:geometry.sole_count],scene)
                delta=prediction[...,:3]-truth[...,:3]
                relative=(pj-prediction[:,:,None,:3])-(sj-truth[:,:,None,:3])
                goal=truth[0,-1,[0,2]]-source[0,3,[0,2]]
                progress=prediction[0,-1,[0,2]]-source[0,3,[0,2]]
                row.update(root_rmse_m=float(delta.square().sum(-1).mean().sqrt()),
                    relative_mpjpe_m=float(relative.norm(dim=-1).mean()),
                    progress_ratio=float((progress*goal).sum()/goal.square().sum().clamp_min(.01)),
                    sole_vertical_p99_m=float(torch.quantile((-gap[known]).clamp_min(0),.99)) if known.any() else None,
                    source_sole_vertical_p99_m=float(torch.quantile((-source_gap[source_known]).clamp_min(0),.99)),
                    unknown_sole_fraction=float((~known).float().mean()))
                body=geometry.full_surface_metrics(prediction,scene,source,identity)
                row.update(body)
                row['passed']=failure is None and row['emitted_frames']==104 and row['unknown_body_fraction']==0 and (
                    row['root_rmse_m']<=.1 and row['relative_mpjpe_m']<=.08 and row['body_penetration_p99_m']<=.02
                    and row['sole_vertical_p99_m']<=.04 and (case['name']=='idle' or .8<=row['progress_ratio']<=1.2))
                name=case['split']+'-'+case['name']+'-'+mode
                save_motion(output/(name+'.npz'),prediction,scene,rep,row)
                save_motion(output/(name+'-source.npz'),truth,scene,rep,dict(row,role='unchanged_reference'))
            else:row['passed']=False
            rows.append(row);write_json(output/'metrics.json',{'cases':rows,'quality_approved':False})
            print(json.dumps({'review':str(output),'case':row}),flush=True)
    return rows


def train(args):
    args.output.mkdir(parents=True,exist_ok=False)
    model,saved=initialize(args.checkpoint,args.detail_adapters)
    if args.online_only:
        online_review(model,saved,args.output)
        return
    cases=torch.load(args.cases/'cases.pt',weights_only=False)
    case_contract=json.loads((args.cases/'cases.json').read_text())
    if sha256(args.checkpoint)!=case_contract['checkpoint_sha256']:raise ValueError('A/B initial weights changed')
    if args.far_only:
        model.requires_grad_(False)
        for key in ('root_far','pose_far','decoder_far'):getattr(model,key).requires_grad_(True)
        if args.detail_adapters:
            for key in ('root_detail','pose_detail','decoder_detail'):getattr(model,key).requires_grad_(True)
    geometry=JointGeometryObjective(model.inference.global_motion_rep,'cuda:0')
    coefficients=saved['coefficients'];train_cases=[r for r in cases if r['split']=='train']
    vocab={key:state_digest(getattr(model.codec,key)) for key in ('encoder','quantizer')}
    contract={'schema':'motionbricks.terrain_small_repair.v1','initial_checkpoint_sha256':sha256(args.checkpoint),
        'cases_sha256':sha256(args.cases/'cases.pt'),'reset_every':args.reset_every,'epochs':args.epochs,
        'updates':args.epochs*13,'batch':len(train_cases),'lr':5e-5,'seed':2333,'coefficients':coefficients,
        'frozen_vocabulary':vocab,'inherited_actual_updates':10709,'no_postprocessing':True,
        'selection':'complete real source events; train and validation source groups disjoint',
        'only_ab_difference':'factual-history reset interval within the same 13 source windows'}
    contract.update(exact_geometry=args.exact_geometry,accumulate_episode=args.accumulate_episode,
        updates=args.epochs if args.accumulate_episode else args.epochs*13,
        primitive_examples=args.epochs*13*len(train_cases))
    contract.update(far_only=args.far_only,source_motion_unchanged=case_contract.get('source_motion_unchanged',True),
        offline_hand_surface_retarget=case_contract.get('offline_hand_surface_retarget'))
    contract['detail_adapters']=args.detail_adapters
    if args.exact_geometry:contract['only_ab_difference']='exact target-side interior geometry versus voxel distance; same accepted sources and episode gradients'
    write_json(args.output/'contract.json',contract)
    if args.evaluate_only:
        review(model,cases,geometry,args.output/'baseline');return
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=5e-5)
    start=time.perf_counter();updates=0;counts=Counter()
    # Same eight full real events in each arm. No global minimum can shorten
    # another row: each selected event supplies exactly thirteen real windows.
    episode=TerrainLearningEpisode(train_cases,[0]*len(train_cases),32,13,model.inference.global_motion_rep,'cuda:0')
    prepared=geometry.prepare_episode(episode)
    if args.exact_geometry:
        prepared['volumes']=PackedExactInteriors(episode.scenes)
    for epoch in range(args.epochs):
        episode.position=0;episode.history=episode.source[:,:4].detach();episode.command_state=SoftCommandState()
        losses=[]
        if args.accumulate_episode:optimizer.zero_grad(set_to_none=True)
        for segment in range(13):
            teacher=segment==0 or bool(args.reset_every and segment%args.reset_every==0)
            with torch.no_grad():window,physical_window,physical_history,resets=episode.window(teacher)
            if resets:raise ValueError('Generated history left observed training terrain; inspect before proceeding')
            task=intents(train_cases,window,32,'cuda:0',episode.history,episode.command_state)
            if not args.accumulate_episode:optimizer.zero_grad(set_to_none=True)
            result=model(episode.history,task,episode.scenes,num_tokens=8,source=window,
                packed_scene=episode.packed_scene,straight_through=True)
            with geometry_precision():
                native=model.native_losses(result,window)
                physical=geometry.prepared_losses(result['world'],physical_history,prepared,offset=segment*8,diagnostics=False)
                loss=native['native_total']+sum(coefficients[k]*physical[k] for k in coefficients)
            if not torch.isfinite(loss):raise ValueError('Nonfinite pilot loss')
            (loss/(13 if args.accumulate_episode else 1)).backward()
            if not args.accumulate_episode or segment==12:
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5)
                if not torch.isfinite(norm):raise ValueError('Nonfinite pilot gradient')
                optimizer.step();updates+=1
            with torch.no_grad(),geometry_precision():
                episode.advance(reencode_played_history(result['world'][:,8:12].detach(),model.inference.global_motion_rep))
            counts['teacher' if teacher else 'generated']+=len(train_cases)
            counts['max_consecutive_generated']=max(counts['max_consecutive_generated'],segment%args.reset_every if args.reset_every else segment)
            losses.append(float(loss.detach()))
        progress={'epoch':epoch+1,'updates':updates,'loss_mean':float(np.mean(losses)),
            'elapsed_seconds':time.perf_counter()-start,'counts':dict(counts),'peak_memory_mb':torch.cuda.max_memory_allocated()/1024**2}
        write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if (epoch+1)%10==0 or epoch+1==args.epochs:
            save_contract=dict(saved['contract'],small_repair=contract)
            save_contract['source_motion_unchanged']=contract['source_motion_unchanged']
            save_contract.pop('sample_budget',None)
            save_contract['initialization']={'step':10709,'sha256':sha256(args.checkpoint)}
            torch.save({'schema':SCHEMA,'model_state':model.state_dict(),'step':updates,
                'contract':save_contract,'coefficients':coefficients},args.output/f'weights-{updates:06d}.pt')
    if vocab!={key:state_digest(getattr(model.codec,key)) for key in vocab}:raise ValueError('Vocabulary changed')
    rows=review(model,cases,geometry,args.output/'final')
    write_json(args.output/'complete.json',{'contract':contract,'results':rows,'all_passed':all(r['passed'] for r in rows),
        'quality_approved':False,'live_control_checked':False})


@torch.no_grad()
def online_review(model,saved,output):
    from terrain_test_world import build_world
    hf,origin,spawns=build_world()
    scene={'heightfield_y_m':torch.from_numpy(hf),'min_point_world_xyz_m':torch.from_numpy(origin),'dx_m':torch.tensor(.2)}
    bundle=torch.load(Path(saved['contract']['source']),map_location='cpu',weights_only=False)
    rep=model.inference.global_motion_rep
    seed=bundle['initial_history'].to('cuda:0')
    geometry=JointGeometryObjective(rep,'cuda:0');rows=[]
    for name,spawn,speed in [('idle','hub',0.),('walk','hub',1.6),('stairs16','zone-03',1.6),('stairs30','zone-06',1.6),('platform','zone-15',1.6)]:
        selected=next(s for s in spawns if s['id']==spawn)
        history=rep.change_first_heading(seed.clone(),0.,False,False)
        target=history.new_tensor(selected['spawn_xz'])
        if name=='platform':target[0]-=8.
        history[...,[0,2]]+=target-history[:,-1:,[0,2]]
        start=history[:,-1,:3].clone();parts=[];state=SoftCommandState();failure=None
        for _ in range(45):
            task=state.intent(history,[0.,speed],0.,8)
            try:prediction=model(history,task,[scene])['world']
            except ValueError as error:failure=str(error);break
            parts.append(prediction[:,4:12]);history=reencode_played_history(prediction[:,8:12],rep)
        row={'name':name,'requested_frames':360,'emitted_frames':sum(p.shape[1] for p in parts),'failure':failure,
            'conditions':'causal constant WASD; no source future goal, height or pose'}
        if parts:
            world=torch.cat(parts,1);joints,points=geometry.geometry(world)
            gap,known=heightfield_gap(points[0,:,:geometry.sole_count],scene)
            delta=world[0,:,:3]-start[0]
            row.update(progress_z_m=float(delta[-1,2]),cross_track_m=float(delta[-1,0].abs()),
                max_idle_drift_m=float(delta[:,[0,2]].norm(dim=-1).max()) if speed==0 else None,
                sole_vertical_p99_m=float(torch.quantile((-gap[known]).clamp_min(0),.99)),
                frames_sole_inside_2cm=int((gap.amin(-1)<-.02).sum()))
            row.update(geometry.full_surface_metrics(world,scene,world,'online-'+name))
            row['passed']=failure is None and len(world[0])==360 and row['sole_vertical_p99_m']<=.04 and row['body_penetration_p99_m']<=.02 and (
                row['max_idle_drift_m']<=.1 if speed==0 else row['progress_z_m']>=12 and row['cross_track_m']<=1.)
            save_motion(output/(name+'.npz'),world,scene,rep,row)
        else:row['passed']=False
        rows.append(row);write_json(output/'metrics.json',{'cases':rows,'all_passed':all(r['passed'] for r in rows)})
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--screen-sources',action='store_true')
    parser.add_argument('--stairs-first',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--online-only',action='store_true')
    parser.add_argument('--reset-every',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--accumulate-episode',action='store_true')
    parser.add_argument('--exact-geometry',action='store_true')
    parser.add_argument('--far-only',action='store_true')
    parser.add_argument('--detail-adapters',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args.checkpoint,args.output,args.screen_sources,args.stairs_first)
    else:train(args)
