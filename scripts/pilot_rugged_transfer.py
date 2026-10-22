"""Bounded new-PARC transfer plus unchanged accepted stair rehearsal."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import SDF_VERSION
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.joint_sdf_training import ExactPenetrationBatch
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_contact_preserve import load_model,control
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.pilot_joint_sdf_reciprocal_source import configure_trainable,frozen_state
from scripts.pilot_pure_parc_stairs import task_loss,review
from scripts.prepare_joint_sdf_context_pilot import recorded_direction
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest
from motionbricks.data.bones_seed_rehearsal import BonesSeedPool,bones_seed_controls


@torch.no_grad()
def review_bones(model,items,geometry,output):
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for case in items:
        source=case['world'][None].cuda();history=source[:,:4].clone();state=SoftCommandState();parts=[];failure=None
        count=(len(case['world'])-4)//8
        for block in range(count):
            offset=min(block*8,source.shape[1]-33)
            controls=bones_seed_controls(source[:,offset:offset+33],case['category'],case['reference'][None].cuda(),history)
            try:
                result=model(history,state.from_controls(history,controls,8),[case['scene']])['world']
                parts.append(result[:,4:12]);history=reencode_played_history(result[:,8:12],geometry.rep)
            except ValueError as error:failure=str(error);break
        name=case['category']+'-'+case['row']['source_id'].replace(':','_')
        row=dict(category=case['category'],source_id=case['row']['source_id'],split=case['evaluation_split'],
            scope=case['scope'],source_start_frame=case['source_start_frame'],frames=len(parts)*8,
            requested_frames=count*8,failure=failure,control='weak planar labels with causal observed facing',postprocessing=None)
        save_motion(output/(name+'-source.npz'),source,case['scene'],geometry.rep,dict(role='unmodified_BONES_reference'))
        if parts:
            world=torch.cat(parts,1);row.update(exact_skin_metrics(geometry,world,case['scene']))
            delta=world[0,-1,[0,2]]-source[0,3,[0,2]]
            target=source[0,3+count*8,[0,2]]-source[0,3,[0,2]]
            velocity=torch.diff(world[0,:,[0,2]],dim=0).norm(dim=-1)*30
            row.update(root_drift_m=float(delta.norm()),source_progress_m=float(target.norm()),
                root_endpoint_error_m=float((delta-target).norm()),
                tail_speed_p90_mps=float(torch.quantile(velocity[-8:],.9)),
                root_step_delta_v_max_mps=float(torch.diff(world[0,:,:3],n=2,dim=0).norm(dim=-1).max()*30))
            _,points=geometry.geometry(world);gap,known=heightfield_gap(points[0,:,:geometry.sole_count],case['scene'])
            row['both_soles_above5cm_fraction']=float((gap.amin(-1)>.05).float().mean())
            save_motion(output/(name+'-generated.npz'),world,case['scene'],geometry.rep,row)
        rows.append(row);write_json(output/'metrics.json',dict(rows=rows,quality_approved=False))
    return rows


def bone_gradient_check(model,source,controls,scenes,fields,geometry,coefficients):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        history=source[:,:4];intent=SoftCommandState().from_controls(history,controls,8)
        result=model(history,intent,scenes,source=source,straight_through=True)
        with torch.no_grad():joints,points=geometry.geometry(source)
        exact=ExactPenetrationBatch(scenes,source.device,fields)
        loss,_=task_loss(model,result,history,source,dict(joints=joints,points=points,heights=exact.heights,volumes=exact),geometry,coefficients)
        report={}
        for name in ('root','pose','codec.decoder'):
            parameters=tuple(p for p in model.get_submodule(name).parameters() if p.requires_grad)
            gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
            norm=sum(float(g.detach().square().sum()) for g in gradients if g is not None)**.5
            if not np.isfinite(norm) or norm<=0:raise ValueError('BONES has no finite gradient to '+name)
            report[name]=norm
        return dict(loss=float(loss.detach()),gradient_norms=report,optimizer_updates=0,
            inference_changed=False,base_predictors_trainable=True)


def new_control(case,offset):
    direction=case['recorded_direction'][min(offset+3,len(case['world'])-1)]
    yaw=torch.atan2(direction[0],direction[1])+case['heading_offset']
    return torch.cat((direction*case['cruise_speed'],yaw.cos()[None],yaw.sin()[None]))[None]


def route_coverage(actual,source):
    indices=[0];distance=0.
    for i in range(1,len(source)):
        distance+=float((source[i,[0,2]]-source[i-1,[0,2]]).norm())
        if distance>=.5:indices.append(i);distance=0.
    if indices[-1]!=len(source)-1:indices.append(len(source)-1)
    cursor=covered=0
    for i in indices:
        candidates=torch.nonzero((actual[cursor:,[0,2]]-source[i,[0,2]]).norm(dim=-1)<=.6).flatten()
        if not len(candidates):break
        cursor+=int(candidates[0])+1;covered+=1
    return dict(ordered_root_milestones=covered,required_root_milestones=len(indices),
        root_milestone_fraction=covered/len(indices),endpoint_xz_error_m=float((actual[-1,[0,2]]-source[-1,[0,2]]).norm()))


@torch.no_grad()
def review_new(model,cases,geometry,output):
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for case in cases:
        history=case['world'][None,:4].cuda();state=SoftCommandState();parts=[];failure=None
        maximum=(len(case['world'])-4)//8
        for block in range(maximum):
            try:
                intent=state.from_controls(history,new_control(case,block*8).cuda(),8)
                result=model(history,intent,[case['scene']])['world']
                parts.append(result[:,4:12]);history=reencode_played_history(result[:,8:12],geometry.rep)
            except ValueError as error:
                failure=str(error);break
        row=dict(source_id=case['row']['source_id'],split=case['row']['split'],family=case['family'],
            frames=len(parts)*8,requested_frames=maximum*8,failure=failure,source_exact_skin_pass=case['source_exact_skin_pass'],
            control='recorded direction schedule times inferred flat-approach cruise; offline replay, not free WASD',
            postprocessing=None)
        save_motion(output/(case['row']['source_id']+'-source.npz'),case['world'][None].cuda(),case['scene'],geometry.rep,
            dict(role='unmodified_teacher_source',source_exact_skin_pass=case['source_exact_skin_pass']))
        if parts:
            world=torch.cat(parts,1);row.update(exact_skin_metrics(geometry,world,case['scene']))
            row.update(route_coverage(world[0,:,:3],case['world'][4:4+maximum*8,:3].cuda()))
            _,points=geometry.geometry(world);gap,known=heightfield_gap(points[0,:,:geometry.sole_count],case['scene'])
            row['both_soles_above5cm_fraction']=float((gap.amin(-1)>.05).float().mean())
            row['root_step_delta_v_max_mps']=float(torch.diff(world[0,:,:3],n=2,dim=0).norm(dim=-1).max()*30) if world.shape[1]>2 else None
            save_motion(output/(case['row']['source_id']+'-generated.npz'),world,case['scene'],geometry.rep,row)
        rows.append(row);write_json(output/'metrics.json',dict(rows=rows,quality_approved=False))
    return rows


def main(args):
    if args.updates!=200:raise ValueError('This declared pilot is exactly200updates')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    expected='ae558ee88542191b6cbe72659bd96f1482711ea2d8beec6c1641a0ae5b814934'
    if sha256(args.checkpoint)!=expected:raise ValueError('Expected accepted live baseline')
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model,_=load_model(saved);configure_trainable(model,'coordinated');rep=model.inference.global_motion_rep
    geometry=JointGeometryObjective(rep,'cuda:0')
    terrain_names=('sdf_encoder','sdf_joint_embedding','sdf_root','sdf_pose','sdf_decoder')
    if args.train_terrain:
        for name in terrain_names:model.get_submodule(name).requires_grad_(True)
    def immutable_state():
        return {name:digest for name,digest in frozen_state(model,'coordinated').items()
            if not args.train_terrain or name not in terrain_names}
    frozen=immutable_state();terrain_gradient_max={name:0. for name in terrain_names}
    audits=json.loads(args.audit.read_text());new_ids={r['source_id']:r for r in audits['rows']}
    cases=[]
    for split in ('train','validation','test'):
        data=TerrainLearningDataset(args.dataset,rep,split)
        for i,row in enumerate(data.records):
            if row['source_id'] not in new_ids:continue
            item=data.full_reference_item(i);world=item['world'];direction=recorded_direction(data,row,len(world))
            gap,known=heightfield_gap(world[:,:3],item['scene']);ground=world[:,1]-gap
            moved=(ground-ground[3]).abs()>.03
            edge=torch.nonzero(moved[4:]).flatten();end=min(30,int(edge[0])+4 if len(edge) else 30)
            if end<9:raise ValueError('Insufficient real flat approach for a declared weak cruise label')
            speed=torch.diff(world[3:end,[0,2]],dim=0).norm(dim=-1).median()*30
            if speed<.2:raise ValueError('Weak approach speed is not a meaningful forward request')
            yaw=torch.atan2(world[3,4],world[3,3]);travel=torch.atan2(direction[3,0],direction[3,1])
            cases.append(dict(item,recorded_direction=direction,cruise_speed=speed,heading_offset=yaw-travel,
                source_exact_skin_pass=new_ids[row['source_id']]['source_exact_skin_pass'],family=new_ids[row['source_id']]['family']))
    new_train=[c for c in cases if c['row']['split']=='train' and c['source_exact_skin_pass']]
    anchors=torch.load(args.anchors,map_location='cpu',weights_only=False)
    old_train=[c for c in anchors if c['pilot_split']=='train']
    if len(new_train)!=3 or len(old_train)!=4:raise ValueError('Expected audited3new/4preserved sources')
    bones=BonesSeedPool(args.bones_dataset,rep,args.dataset/'stats') if args.bones_dataset else None
    batch=8 if bones else 4
    contract=dict(parent_sha256=expected,dataset_manifest_sha256=sha256(args.dataset/'native_manifest.json'),
        dataset=str(args.dataset),anchor_cases_sha256=sha256(args.anchors),
        audit_sha256=sha256(args.audit),updates=200,batch=batch,actor_primitives=200*batch,source_autoencoder_examples=200*batch,
        parc_primitives=800,bones_primitives=800 if bones else 0,
        sampling='2new PARC +2original PARC'+(' +4category-balanced BONES' if bones else '')+' source-history windows per update',
        new_sources=[c['row']['source_id'] for c in new_train],anchor_sources=[c['row']['source_id'] for c in old_train],
        trainable='Root,Pose,codec decoder only; SDF/codec encoder/vocabulary frozen',
        optimizer='fresh Adam5e-5,clip0.5',coefficients=saved['coefficients'],stop_task=False,
        command='captured unit direction times weak flat-approach speed; fixed initial facing offset; not measured WASD',
        source_history_only=True,generated_history_training=False,future_truth_in_actor=False,
        quality_approved=False,postprocessing=None)
    if bones:
        contract.update(bones_seed=bones.contract,comparison='PARC exposure and200updates retained; batch/sample/compute doubled, not optimizer-equivalent',
            flat_start_stop_included=True,terrain_stop_task=False)
    contract['train_terrain']=args.train_terrain
    if args.train_terrain:
        if not bones:raise ValueError('This coadaptation probe requires the matched BONES mixture')
        contract.update(trainable='Root/Pose/codec decoder AND SDF encoder/joint embedding/Root/Pose/decoder adapters; codec encoder/vocabulary/old control frozen',
            comparison='same200updates,batch8,source windows,losses/LR as frozen-SDF BONES arm; additional SDF trainable parameters only')
    write_json(args.output/'contract.json',contract)
    write_json(args.output/'state.json',dict(status='baseline'))
    initial_new=review_new(model,cases,geometry,args.output/'baseline_new')
    initial_old=review(model,anchors,geometry,args.output/'baseline_stairs')
    if args.reference_run:
        errors={}
        for directory in ('baseline_new','baseline_stairs'):
            for path in (args.output/directory).glob('*.npz'):
                with np.load(path,allow_pickle=False) as current,np.load(args.reference_run/directory/path.name,allow_pickle=False) as prior:
                    np.testing.assert_allclose(current['world'],prior['world'],atol=3e-5,rtol=1e-5)
                    errors[directory+'/'+path.name]=float(np.abs(current['world']-prior['world']).max())
        write_json(args.output/'baseline_parity.json',dict(reference=str(args.reference_run),errors=errors))
    initial_bones=review_bones(model,bones.evaluation,geometry,args.output/'baseline_bones') if bones else []
    if bones and args.reference_run and (args.reference_run/'baseline_bones').is_dir():
        errors={}
        for path in (args.output/'baseline_bones').glob('*.npz'):
            with np.load(path,allow_pickle=False) as current,np.load(args.reference_run/'baseline_bones'/path.name,allow_pickle=False) as prior:
                np.testing.assert_allclose(current['world'],prior['world'],atol=3e-5,rtol=1e-5)
                errors[path.name]=float(np.abs(current['world']-prior['world']).max())
        write_json(args.output/'baseline_bones_parity.json',errors)
    control_bones=[]
    if bones and args.reference_run:
        initial_digest=state_digest(model)
        control_saved=torch.load(args.reference_run/'weights.pt',map_location='cpu',mmap=True,weights_only=False)
        control_model,_=load_model(control_saved)
        control_bones=review_bones(control_model,bones.evaluation,geometry,args.output/'parc_only_control_bones')
        del control_model,control_saved
        torch.cuda.empty_cache()
        if state_digest(model)!=initial_digest:raise ValueError('Reference evaluation changed the trainable model')
    torch.manual_seed(2333);parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=5e-5);cursors=Counter();exposure=Counter();logs=[];bone_draws=[];started=time.time()
    fields={c['row']['source_id']:HeightfieldPrismSDF(c['scene']) for c in new_train+old_train}
    def checkpoint(step):
        stage=dict(schema='motionbricks.training_stage.v1',name='rugged_bones50_joint_v1' if args.train_terrain else ('rugged_bones50_v1' if bones else 'rugged_transfer_v1'),optimizer_updates=step,
            actor_primitives=step*batch,source_autoencoder_examples=step*batch,inherited_optimizer_updates=saved['step'])
        temporary=args.output/'weights.pending.pt'
        torch.save(dict(schema=SDF_VERSION,model_state=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
            contract=dict(saved['contract'],stair_training_stage=stage,rugged_transfer=contract),coefficients=saved['coefficients'],
            cursors=dict(cursors),exposure=dict(exposure),bones_draws=bone_draws,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temporary)
        temporary.replace(args.output/'weights.pt')
        if bones:write_json(args.output/'bones_exposure.json',dict(draws=bone_draws,category_counts=dict(Counter(r['category'] for r in bone_draws))))
    write_json(args.output/'state.json',dict(status='training'))
    for step in range(200):
        chosen=[(new_train[(step*2+j)%3],True) for j in range(2)]+[(old_train[(step*2+j)%4],False) for j in range(2)]
        windows=[];controls=[];scenes=[];batch_fields=[]
        for c,is_new in chosen:
            key=c['row']['source_id'];position=cursors[key]%((len(c['world'])-33)//8+1);offset=position*8
            windows.append(c['world'][offset:offset+33]);controls.append(new_control(c,offset) if is_new else control(c,position))
            scenes.append(c['scene']);batch_fields.append(fields[key]);cursors[key]+=1;exposure[key]+=1
        if bones:
            for window,value,scene,record in bones.sample(step):
                windows.append(window);controls.append(value);scenes.append(scene);batch_fields.append(HeightfieldPrismSDF(scene))
                exposure['bones:'+record['source_id']]+=1;bone_draws.append(dict(record,update=step+1))
        source=torch.stack(windows).cuda();history=source[:,:4]
        intent=SoftCommandState().from_controls(history,torch.cat(controls).cuda(),8)
        if step==0:
            if bones:write_json(args.output/'bones_gradient_preflight.json',bone_gradient_check(model,source[4:],torch.cat(controls[4:]).cuda(),scenes[4:],batch_fields[4:],geometry,saved['coefficients']))
            with torch.no_grad():
                reference=model(history,intent,scenes)['world'];changed=source.clone();changed[:,4:]+=2
                alternate=model(history,intent,scenes,source=changed)['world']
            with torch.inference_mode():runtime=model(history,intent,scenes)['world']
            torch.testing.assert_close(reference,alternate,atol=0,rtol=0)
            torch.testing.assert_close(reference,runtime,atol=3e-5,rtol=1e-5)
        result=model(history,intent,scenes,source=source,straight_through=True)
        if step==0:
            torch.testing.assert_close(reference,result['world'],atol=3e-5,rtol=1e-5)
            write_json(args.output/'preflight.json',dict(future_target_error=float((reference-alternate).abs().max()),
                runtime_error=float((reference-runtime).abs().max()),training_forward_error=float((reference-result['world']).abs().max())))
        with torch.no_grad():joints,points=geometry.geometry(source)
        exact=ExactPenetrationBatch(scenes,source.device,batch_fields)
        target=dict(joints=joints,points=points,heights=exact.heights,volumes=exact)
        loss,detail=task_loss(model,result,history,source,target,geometry,saved['coefficients'])
        if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(p.grad is not None for name in frozen for p in model.get_submodule(name).parameters()):raise ValueError('Frozen gradient changed')
        if args.train_terrain:
            for name in terrain_names:
                value=sum(float(p.grad.detach().square().sum()) for p in model.get_submodule(name).parameters() if p.grad is not None)**.5
                terrain_gradient_max[name]=max(terrain_gradient_max[name],value)
        norm=torch.nn.utils.clip_grad_norm_(parameters,.5)
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradients')
        optimizer.step()
        row=dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_seconds=time.time()-started,**detail)
        logs.append(row);write_json(args.output/'progress.json',row)
        if (step+1)%25==0:
            if immutable_state()!=frozen:raise ValueError('Frozen weights changed')
            checkpoint(step+1);print(json.dumps(row),flush=True)
    if args.train_terrain and max(terrain_gradient_max.values())<=0:raise ValueError('No actual terrain-branch gradient')
    model.requires_grad_(False);write_json(args.output/'state.json',dict(status='review'))
    final_new=review_new(model,cases,geometry,args.output/'final_new')
    final_old=review(model,anchors,geometry,args.output/'final_stairs')
    final_bones=review_bones(model,bones.evaluation,geometry,args.output/'final_bones') if bones else []
    if bones and args.reference_run:
        reference=json.loads((args.reference_run/'complete.json').read_text())['exposure']
        actual={k:exposure[k] for k in reference}
        if actual!=reference:raise ValueError('PARC sample exposure changed')
        write_json(args.output/'parc_exposure_parity.json',dict(actual=actual,reference=reference,matched=True))
        if (args.reference_run/'bones_exposure.json').exists():
            prior=json.loads((args.reference_run/'bones_exposure.json').read_text())['draws']
            if bone_draws!=prior:raise ValueError('BONES source/window schedule changed')
            write_json(args.output/'bones_window_parity.json',dict(matched=True,draw_count=len(bone_draws)))
    write_json(args.output/'complete.json',dict(initial_new=initial_new,final_new=final_new,
        initial_stairs=initial_old,final_stairs=final_old,initial_bones=initial_bones,parc_only_control_bones=control_bones,final_bones=final_bones,
        logs=logs,actual_updates=200,actor_primitives=200*batch,parc_primitives=800,bones_primitives=len(bone_draws),
        source_autoencoder_examples=200*batch,frozen_hashes_match=True,quality_approved=False,exposure=dict(exposure),terrain_gradient_max=terrain_gradient_max))
    write_json(args.output/'state.json',dict(status='complete',quality_approved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','dataset','audit','anchors','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bones-dataset',type=Path)
    p.add_argument('--reference-run',type=Path)
    p.add_argument('--train-terrain',action='store_true')
    p.add_argument('--updates',type=int,default=200);args=p.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.exists():write_json(args.output/'state.json',dict(status='failed',error=type(error).__name__+': '+str(error)))
        raise
