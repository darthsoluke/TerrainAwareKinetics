"""Causal small-sample stairs pilot for the 15-joint SDF controller."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_control import SoftCommandState,demonstration_controls
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_batched_geometry import PackedHeightfields
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.joint_sdf_controller import JointSDFController,SDF_VERSION
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF,JOINT_SDF_NAMES
from motionbricks.data.joint_sdf_training import verified_lateral_offsets
from motionbricks.data.joint_sdf_codec_objective import native_codec_reconstruction
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest
from scripts.train_joint_terrain_v3 import write_json
from probe_terrain_overfit import save_motion


class FieldBatch:
    def __init__(self,fields):self.fields=fields
    def query(self,points):
        results=[field.query(row) for field,row in zip(self.fields,points)]
        return torch.stack([r[0] for r in results]),torch.stack([r[2] for r in results])


def load_model(path,device='cuda:0'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    source=Path(saved['contract']['source'])
    if sha256(source)!=saved['contract']['source_bundle_sha256']:raise ValueError('Source/checkpoint mismatch')
    infer,bundle=load_terrain_bundle(source,device)
    model=JointSDFController(infer,bundle)
    keys=model.load_state_dict(saved['model_state'],strict=False)
    if keys.unexpected_keys or any(not k.startswith('sdf_') for k in keys.missing_keys):raise ValueError('Unexpected missing base weights')
    model.freeze_base();model.eval()
    return model,saved,bundle


def select_data(model,saved):
    dataset=Path(saved['contract']['dataset'])
    data=TerrainLearningDataset(dataset,build_native_representation(dataset/'stats'),'train')
    candidates=sorted([i for i,r in enumerate(data.records) if 'stairs' in str(r.get('terrain_family','')) and r['frames_30hz']>=65],
        key=lambda i:(data.records[i]['source_id']!='official_df3f1541e287fcb3',data.records[i]['source_id']))
    stair=None
    for index in candidates:
        item=data.full_reference_item(index);world=item['world'];scene=item['scene']
        gap,known=heightfield_gap(world[:,:3],scene);ground=world[:,1]-gap
        for edge in torch.nonzero(ground[1:]-ground[:-1]>.05).flatten().tolist():
            edge+=1
            if edge<20 or edge+24>=len(world) or not known[edge-20:edge+24].all():continue
            if ground[edge-16:edge].max()-ground[edge-16:edge].min()>.03:continue
            start=max(0,edge-48)
            approach_end=edge-8
            delta=world[approach_end,[0,2]]-world[start+3,[0,2]]
            if delta.norm()<.06:continue
            direction=delta/delta.norm();side=torch.stack((direction[1],-direction[0]))
            end=min(start+257,len(world))
            horizontal=world[start:end,[0,2]]-world[start,[0,2]]
            turned=torch.nonzero((horizontal*side).sum(-1).abs()>.35).flatten()
            if len(turned):end=min(end,start+int(turned[0]))
            length=end-start
            if length<65 or ground[start+length-5]-ground[start]<.2:continue
            stair=dict(item,world=world[start:start+length].clone(),name='stairs',start_frame=start,
                edge_frame=edge-start,command_velocity=(delta/((approach_end-start-3)/30)).tolist(),
                command_heading=float(torch.atan2(world[start+3:approach_end,4].mean(),world[start+3:approach_end,3].mean())))
            break
        if stair is not None:break
    if stair is None:raise ValueError('No complete straight stair approach found; do not substitute a mid-stair window')
    cases=[stair]
    for category in ('idle','walk'):
        index=next(i for i,r in enumerate(data.records) if r.get('control_category')==category and r['frames_30hz']>=129)
        item=data.full_reference_item(index);item['world']=item['world'][:129]
        center=item['world'][3,:3]
        item['scene']={'heightfield_y_m':torch.zeros(201,201),'min_point_world_xyz_m':torch.tensor([center[0]-50,0,center[2]-50]),'dx_m':torch.tensor(.5)}
        cases.append(dict(item,name=category,start_frame=0))
    return cases


def request(history,case,window=None,state=None,speed_scale=1.):
    state=state or SoftCommandState()
    if case['name']=='stairs':return state.intent(history,history.new_tensor(case['command_velocity'])*speed_scale,case['command_heading'],8)
    controls=demonstration_controls(window,32,category=case['name'],reference=case['world'][None].to(history))
    return state.from_controls(history,controls,8)


@torch.no_grad()
def evaluate(model,cases,geometry,bundle,output):
    output.mkdir(parents=True,exist_ok=False);rep=model.inference.global_motion_rep;rows=[]
    for case in cases:
        modes=['source_controls'] if case['name']!='stairs' else ['source_history_start','idle_start','slow','fast']
        source=case['world'][None].to('cuda:0');scene=case['scene']
        for mode in modes:
            history=source[:,:4].clone();state=SoftCommandState();scale={'slow':.8,'fast':1.2}.get(mode,1.)
            blocks=(source.shape[1]-4)//8 if case['name']=='stairs' else 13
            if mode=='idle_start':
                initial=bundle['initial_history'].to(history)
                history=rep.change_first_heading(initial,case['command_heading'],False,False)
                history[...,[0,2]]+=source[:,3:4,[0,2]]-history[:,3:4,[0,2]]
                gap,known=heightfield_gap(history[0,3:4,:3],scene)
                if not known.all():raise ValueError('Unknown initial support')
                history=shift_global_motion(history,rep,history[0,3,1]-gap[0])
            initial=history.clone();parts=[];times=[];failure=None;diagnostics=[]
            for block in range(blocks):
                window=source[:,block*8:block*8+33]
                task=request(history,case,window,state,scale)
                torch.cuda.synchronize();start=time.perf_counter()
                try:result=model(history,task,[scene])
                except ValueError as error:failure=str(error);break
                torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
                parts.append(result['world'][:,4:12]);diagnostics.append(model.sdf_diagnostics)
                history=reencode_played_history(result['world'][:,8:12],rep)
            row={'name':case['name'],'mode':mode,'frames':sum(p.shape[1] for p in parts),'requested_frames':blocks*8,
                'failure':failure,'speed_scale':scale,'constant_command':case['name']=='stairs'}
            if parts:
                world=torch.cat(parts,1);_,surface=geometry.geometry(world)
                gap,known=heightfield_gap(surface[0,:,:geometry.sole_count],scene)
                delta=world[0,-1,:3]-initial[0,3,:3]
                direction=world.new_tensor(case.get('command_velocity',[0,1.]));direction=direction/direction.norm().clamp_min(1e-6)
                row.update(sole_vertical_p99_m=float(torch.quantile((-gap[known]).clamp_min(0),.99)),
                    frames_sole_inside_2cm=int((gap.min(-1).values<-.02).sum()),root_rise_m=float(delta[1]),
                    progress_m=float((delta[[0,2]]*direction).sum()),cross_track_m=float((delta[[0,2]]*torch.stack((direction[1],-direction[0]))).sum().abs()),
                    inference_median_ms=float(np.median(times)),sdf_valid_fraction_min=min(d['valid_fraction'] for d in diagnostics))
                if case['name']=='stairs':
                    truth=source[:,4:4+world.shape[1]];expected=float((truth[0,-1,[0,2]]-source[0,3,[0,2]]).dot(direction))
                    row['source_progress_m']=expected
                    row['passed']=failure is None and row['frames']==blocks*8 and row['sole_vertical_p99_m']<=.04 and row['cross_track_m']<=.35 and row['progress_m']>=.8*expected and abs(row['root_rise_m']-float(truth[0,-1,1]-source[0,3,1]))<=.2
                save_motion(output/(case['name']+'-'+mode+'.npz'),world,scene,rep,row)
                if case['name']=='stairs' and mode=='source_history_start':
                    save_motion(output/(case['name']+'-'+mode+'-source.npz'),source[:,4:4+world.shape[1]],scene,rep,dict(row,role='unchanged_reference'))
            rows.append(row);write_json(output/'metrics.json',{'cases':rows,'quality_approved':False})
            print(json.dumps(row),flush=True)
    return rows


def main(args):
    if args.native_codec_reconstruction and not args.unfreeze_decoder:
        raise ValueError('Source reconstruction probe requires the decoder adaptation arm')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(2333);np.random.seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,saved,bundle=load_model(args.checkpoint)
    if args.unfreeze_decoder:model.codec.decoder.requires_grad_(True)
    if args.unfreeze_root:model.root.requires_grad_(True)
    if args.unfreeze_pose:model.pose.requires_grad_(True)
    frozen_modules={name:getattr(model,name) for name in ('root','pose','control','root_far','pose_far','decoder_near','decoder_attention','decoder_far')}
    if args.unfreeze_root:frozen_modules.pop('root')
    if args.unfreeze_pose:frozen_modules.pop('pose')
    frozen_modules.update(codec_encoder=model.codec.encoder,codec_quantizer=model.codec.quantizer)
    if not args.unfreeze_decoder:frozen_modules['codec_decoder']=model.codec.decoder
    frozen={name:state_digest(module) for name,module in frozen_modules.items()}
    cases=select_data(model,saved);rep=model.inference.global_motion_rep
    torch.save(cases,args.output/'cases.pt')
    geometry=JointGeometryObjective(rep,'cuda:0')
    contract={'version':SDF_VERSION,'initial_sha256':sha256(args.checkpoint),'updates':args.updates,'batch':3,
        'command_builder':'same rolling SoftCommandState in training and inference; no source goal override',
        'stair_command_provenance':'constant weak intent from real approach XZ; not recorded keyboard input',
        'old_local_heightfield_branch_retained':True,'postprocessing':None,'lr':5e-5,'joints':JOINT_SDF_NAMES,
        'decoder_trainable':args.unfreeze_decoder,'frozen_modules':list(frozen_modules),
        'root_trainable':args.unfreeze_root,'pose_trainable':args.unfreeze_pose,
        'native_source_reconstruction':args.native_codec_reconstruction,
        'native_source_reconstruction_coefficient':1. if args.native_codec_reconstruction else 0.,
        'auxiliary_root_condition':'source Root only inside supervised codec reconstruction; actor Root remains predicted',
        'cases':[{'name':c['name'],'source_id':c['row']['source_id'],'start_frame':c['start_frame'],'frames':len(c['world']),
            'edge_frame':c.get('edge_frame'),'command_velocity':c.get('command_velocity')} for c in cases]}
    write_json(args.output/'contract.json',contract)
    baseline=evaluate(model,cases,geometry,bundle,args.output/'baseline')
    if args.reference_baseline:
        expected=json.loads(args.reference_baseline.read_text())['baseline']
        if len(baseline)!=len(expected):raise ValueError('Baseline case count differs')
        for actual,reference in zip(baseline,expected):
            if actual.keys()!=reference.keys():raise ValueError('Baseline fields differ')
            for key,value in actual.items():
                if key=='inference_median_ms':continue
                if isinstance(value,float):
                    if abs(value-reference[key])>1e-5:raise ValueError('Initial baseline differs: '+key)
                elif value!=reference[key]:raise ValueError('Initial baseline differs: '+key)
        write_json(args.output/'baseline_parity.json',{'passed':True,'reference':str(args.reference_baseline),
            'comparison':'all fields except timing, float absolute tolerance1e-5'})
    prepared=[];fields=[]
    with torch.no_grad():
        for case in cases:
            joints,points=geometry.geometry(case['world'][None].to('cuda:0'))
            prepared.append((joints[0],points[0]));fields.append(HeightfieldPrismSDF(case['scene']))
    offsets=[0.]
    if args.lateral_augmentation:
        offsets,admission=verified_lateral_offsets(prepared[0][1],cases[0]['scene'],[-.3,-.2,-.1,0.,.1,.2,.3])
        if len(offsets)<3:raise ValueError('Insufficient measured terrain-preserving translations')
        contract['lateral_augmentation']={'admitted_world_z_m':offsets,'screen':admission,
            'scope':'training reference only; source points, root and targets translate together; terrain is unchanged'}
        write_json(args.output/'contract.json',contract)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=5e-5)
    rng=np.random.default_rng(2333);started=time.perf_counter();noise=[];primitive_examples=0
    augmentation_rng=np.random.default_rng(74039)
    for step in range(args.updates):
        ids=[0,0,1+step%2];items=[cases[i] for i in ids]
        depth=1 if step<args.updates//2 else min(13,min(1+(len(item['world'])-33)//8 for item in items))
        starts=[int(rng.integers(0,len(item['world'])-(33+(depth-1)*8)+1)) for item in items]
        source=torch.stack([item['world'][start:start+33+(depth-1)*8] for item,start in zip(items,starts)]).to('cuda:0')
        shifts=source.new_zeros((len(items),3))
        if args.lateral_augmentation:
            for row,index in enumerate(ids):
                if index==0:shifts[row,2]=float(augmentation_rng.choice(offsets))
            source[:,:,2]+=shifts[:,None,2]
        scenes=[item['scene'] for item in items];history=source[:,:4];states=[SoftCommandState() for _ in items]
        optimizer.zero_grad(set_to_none=True);total=0.;source_reconstruction=0.
        for segment in range(depth):
            offset=segment*8;window=source[:,offset:offset+33]
            task=torch.cat([request(history[row:row+1],item,window[row:row+1],states[row]) for row,item in enumerate(items)])
            result=model(history,task,scenes,source=window,straight_through=True)
            native=model.native_losses(result,window)
            target={'joints':torch.stack([prepared[i][0][start+offset:start+offset+33] for i,start in zip(ids,starts)])+shifts[:,None,None],
                'points':torch.stack([prepared[i][1][start+offset:start+offset+33] for i,start in zip(ids,starts)])+shifts[:,None,None],
                'heights':PackedHeightfields(scenes,history.device),'volumes':FieldBatch([fields[i] for i in ids])}
            physical=geometry.prepared_losses(result['world'],history,target,offset=0,diagnostics=False)
            loss=native['native_total']+sum(saved['coefficients'][k]*physical[k] for k in saved['coefficients'])
            if args.native_codec_reconstruction:
                auxiliary=native_codec_reconstruction(model,result)['loss']
                loss=loss+auxiliary;source_reconstruction+=float(auxiliary.detach())/depth
            if not torch.isfinite(loss):raise ValueError('Nonfinite SDF pilot loss')
            (loss/depth).backward();total+=float(loss.detach())/depth;noise.append(native['sampled_mask_fraction'])
            history=reencode_played_history(result['world'][:,8:12].detach(),rep)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5)
        if not torch.isfinite(norm) or float(norm)==0:raise ValueError('Missing or nonfinite SDF adapter gradient')
        if step==0:
            for name,enabled in (('root',args.unfreeze_root),('pose',args.unfreeze_pose)):
                if enabled and not any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model,name).parameters()):
                    raise ValueError('Requested native backbone has no gradient: '+name)
        optimizer.step()
        primitive_examples+=len(items)*depth
        if step%20==0 or step+1==args.updates:
            progress={'step':step+1,'loss':total,'gradient_norm':float(norm),'elapsed_seconds':time.perf_counter()-started,
                'primitive_examples':primitive_examples,'batch32_equivalent_updates':primitive_examples/32,
                'native_source_reconstruction_loss':source_reconstruction,
                'sampled_mask_min':min(noise),'sampled_mask_max':max(noise),'sdf':model.sdf_diagnostics,
                'peak_memory_mb':torch.cuda.max_memory_allocated()/1024**2}
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    checkpoint={'schema':SDF_VERSION,'model_state':model.state_dict(),'step':args.updates,
        'contract':dict(saved['contract'],joint_sdf=contract),'coefficients':saved['coefficients']}
    if frozen!={name:state_digest(module) for name,module in frozen_modules.items()}:raise ValueError('Frozen base parameters changed')
    torch.save(checkpoint,args.output/'weights.pt')
    model.requires_grad_(False)
    final=evaluate(model,cases,geometry,bundle,args.output/'final')
    model.sdf_enabled=False
    ablation=evaluate(model,cases,geometry,bundle,args.output/'disabled_sdf')
    model.sdf_enabled=True
    write_json(args.output/'complete.json',{'baseline':baseline,'final':final,'disabled_sdf':ablation,
        'actual_updates':args.updates,'primitive_examples':primitive_examples,'decoder_trainable':args.unfreeze_decoder,
        'root_trainable':args.unfreeze_root,'pose_trainable':args.unfreeze_pose,
        'native_source_reconstruction':args.native_codec_reconstruction,
        'auxiliary_reconstruction_examples':primitive_examples if args.native_codec_reconstruction else 0,
        'ablation_semantics':'same final weights with SDF disabled; decoder fine-tuning is retained',
        'all_stair_modes_passed':all(row.get('passed',False) for row in final if row['name']=='stairs'),'quality_approved':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=200)
    p.add_argument('--unfreeze-decoder',action='store_true')
    p.add_argument('--lateral-augmentation',action='store_true')
    p.add_argument('--native-codec-reconstruction',action='store_true')
    p.add_argument('--unfreeze-root',action='store_true')
    p.add_argument('--unfreeze-pose',action='store_true')
    p.add_argument('--reference-baseline',type=Path)
    main(p.parse_args())
