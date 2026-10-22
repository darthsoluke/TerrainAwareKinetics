"""Bounded continuous STOP/restart learning with command-event-gated neural branches."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.decoded_factual_history import decoded_factual_history_loss
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.stop_command_adapter import StopCommandAdapter
from motionbricks.data.stop_stage_adapter import install_stop_stage
from motionbricks.data.stop_transition_adapter import CommandTransitionState, enable_transition_gate, set_transition_active
from motionbricks.data.stop_task_objective import published_root_velocities
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_contact_preserve import load_model
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.stop_transition_reviews import prepare, review, write
from scripts.pilot_stop_feedback_v2 import task_loss
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def restart_cases(model, args, cases):
    data = TerrainLearningDataset(args.dataset, build_native_representation(args.dataset/'stats'), 'train')
    indexed = {row['source_id']:i for i,row in enumerate(data.records)}
    audit = json.loads(args.audit.read_text())
    geometry = JointGeometryObjective(model.inference.global_motion_rep, 'cuda:0')
    extra = []
    for case in cases:
        meta = case['metadata']
        if meta['phase'] != 'rest':
            continue
        event = next(e for e in audit['examples'] if e['source_id'] == meta['source_id']
            and e['rest_start'] == meta['source_start'])
        item = data.full_reference_item(indexed[meta['source_id']])
        start = event['rest_end']-4
        source = item['world'][None,start:start+33].cuda()
        if source.shape[1] != 33:
            raise ValueError('No fabricated restart tail')
        metrics = exact_skin_metrics(geometry, source, item['scene'])
        if metrics['body_prism_max_m'] > .02 or metrics['unknown_body_fraction'] or metrics['uncertain_inside_samples']:
            raise ValueError('Real restart window failed source screen')
        velocity = (source[:,-1,[0,2]]-source[:,20,[0,2]])/(12/30)
        if velocity.norm() < .3:
            raise ValueError('Restart reference lacks actual forward motion')
        yaw = torch.atan2(source[:,3,4],source[:,3,3])
        intent = SoftCommandState().intent(source[:,:4],velocity,yaw,8)
        extra.append(dict(world=source,scene=item['scene'],intent=intent,metadata=dict(
            source_id=meta['source_id'],source_start=start,source_end=start+33,phase='restart',
            provenance='Real observed rest followed by real acceleration; weak future planar command only',source_screen=metrics)))
    if len(extra) != 4:
        raise ValueError('Expected four screened genuine rest-to-move sources')
    return cases+extra


def resume_loss(model, result, history, scene, geometry, field, direction, resume_block):
    world = result['world']
    prefix = decoded_factual_history_loss(model.inference,model.last_local[:,:32],history,result['observation'],
        root_coefficient=float(model.inference._root_model.args['local_root_loss_coeff']))
    velocities = published_root_velocities(world,history)
    forward = velocities @ direction
    # A lower progress objective permits natural slow stair acceleration; no upper speed rule.
    progress = ((.3-forward).clamp_min(0)/.3).square().mean() if resume_block >= 2 else forward.sum()*0
    _, points = geometry.geometry(world[:,4:12])
    distance,_,certified = field.query(points)
    fallback,known = heightfield_gap(points,scene)
    if not known.all():
        raise ValueError('Restart left known terrain')
    distance = torch.where(certified,distance,fallback)
    excess = (-distance-.005).clamp_min(0).flatten()
    tail = excess.topk(max(1,int(np.ceil(excess.numel()*.01)))).values
    collision = (tail/.02).square().mean()
    return prefix+progress+collision,dict(prefix=float(prefix.detach()),progress=float(progress.detach()),
        collision=float(collision.detach()),forward_mean_mps=float(forward.mean().detach()))


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha256(args.parent) != 'ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf':
        raise ValueError('Approved moving weights differ')
    if sha256(args.checkpoint) != '2317ca38ab54f2cb3c6483aa6aead0ee7225cec26622d48490f2f2ae04d059c6':
        raise ValueError('Expected quality-scaled STOP initialization')
    parent=torch.load(args.parent,map_location='cpu',mmap=True,weights_only=False)
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model,_=load_model(parent);model.control=StopCommandAdapter(model.control)
    pose,_=install_stop_stage(model,'pose');decoder,_=install_stop_stage(model,'decoder')
    model.load_state_dict(saved['model_state'],strict=True);model.requires_grad_(False)
    model.control.stop.requires_grad_(True);pose.requires_grad_(True);decoder.requires_grad_(True)
    enable_transition_gate(model);model.eval()
    fixed=[k for k in model.state_dict() if not k.startswith(('control.stop.','stop_pose_condition.','stop_decoder_condition.'))]
    immutable=state_digest(model,fixed)
    set_transition_active(model,True)
    cases=restart_cases(model,args,prepare(model,args))
    write(args.output/'contract.json',dict(max_actual_updates=64,max_actor_examples=128,episodes=4,
        stop_primitives=10,resume_primitives=6,source_cases=[c['metadata'] for c in cases],
        transition='Causal STOP or restart; after at least two plans and observed speed>=.3m/s return to unchanged MOVE branch',
        stop_scales=dict(root_velocity_mps=.15,support_gap_m=.02,foot_velocity_mps=.1,penetration_m=.02),
        restart='After.533s,minimum forward progress speed.3m/s; slower than requested cruising is allowed',
        original_parameters_frozen=True,future_truth_in_actor=False,postprocessing=None,quality_approved=False))
    baseline=review(model,args,args.output/'baseline',transition=True)
    receipt=json.loads((args.artifact/'export.json').read_text());episodes=[]
    for entry in receipt['entries']:
        with np.load(args.artifact/(entry['entry']+'-scene.npz'),allow_pickle=False) as data:
            scene={k:torch.from_numpy(data[k].copy()).cuda() for k in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
        with np.load(args.checkpoint.parent/'final'/(entry['entry']+'-cruise.npz'),allow_pickle=False) as data:
            moving=torch.from_numpy(data['world'].copy())[None].cuda()
        for start in (2,3):
            history=reencode_played_history(moving[:,start*8-4:start*8],model.inference.global_motion_rep)
            episodes.append(dict(entry=entry,scene=scene,history=history,start=start,field=HeightfieldPrismSDF(scene)))
    parameters=list(model.control.stop.parameters())+list(pose.parameters())+list(decoder.parameters())
    optimizer=torch.optim.Adam(parameters,lr=5e-5,weight_decay=0.);torch.manual_seed(2333)
    geometry=JointGeometryObjective(model.inference.global_motion_rep,'cuda:0');logs=[];failures=[];updates=0
    for episode_index,episode in enumerate(episodes):
        history=episode['history'].clone();driver=SoftCommandState();phase=CommandTransitionState();parts=[]
        direction=history.new_tensor(episode['entry']['command_velocity_mps']);direction/=direction.norm()
        for block in range(16):
            optimizer.zero_grad(set_to_none=True);stopped=block<10
            velocity=[0.,0.] if stopped else episode['entry']['command_velocity_mps']
            observed_speed=float(((history[:,-1,[0,2]]-history[:,0,[0,2]])*10).norm())
            active=phase.update(velocity,observed_speed);set_transition_active(model,active)
            intent=driver.intent(history,velocity,episode['entry']['command_heading'],8)
            try:
                result=model(history,intent,[episode['scene']],straight_through=True)
                if stopped:
                    objective,detail,_=task_loss(model,result,history,episode['scene'],geometry,episode['field'],block,
                        parent['coefficients'],quality_normalized=True)
                else:
                    objective,detail=resume_loss(model,result,history,episode['scene'],geometry,episode['field'],direction,block-10)
            except (ValueError,RuntimeError) as error:
                failures.append(dict(episode=episode_index,block=block,actual_frames=len(parts)*8,error=str(error)));break
            published=result['world'][:,4:12].detach().clone()
            next_history=reencode_played_history(result['world'][:,8:12].detach(),model.inference.global_motion_rep)
            prior=cases[updates%len(cases)];source=prior['world'];set_transition_active(model,True)
            source_result=model(source[:,:4],prior['intent'],[prior['scene']],source=source,straight_through=True)
            source_prefix=decoded_factual_history_loss(model.inference,model.last_local[:,:32],source[:,:4],source_result['observation'],
                root_coefficient=float(model.inference._root_model.args['local_root_loss_coeff']))
            source_native=model.native_losses(source_result,source)['native_total'];loss=(objective+source_prefix+source_native)/2
            if not torch.isfinite(loss):raise ValueError('Nonfinite transition loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,.5)
            if not torch.isfinite(norm):raise ValueError('Nonfinite transition gradients')
            optimizer.step();updates+=1
            if state_digest(model,fixed)!=immutable:raise ValueError('Approved tensors changed')
            parts.append(published);history=next_history
            row=dict(step=updates,episode=episode_index,entry=episode['entry']['entry'],block=block,stopped=stopped,
                transition_active=active,loss=float(loss.detach()),source_phase=prior['metadata']['phase'],**detail)
            logs.append(row);write(args.output/'progress.json',dict(rows=logs,failures=failures));print(json.dumps(row),flush=True)
        if parts:save_motion(args.output/f'train-episode{episode_index}.npz',torch.cat(parts,1),episode['scene'],model.inference.global_motion_rep,
            dict(role='Actual generated continuous STOP/restart training episode',frames=len(parts)*8,quality_approved=False))
    torch.save(dict(schema='motionbricks.stop_transition_pilot.v1',model_state=model.state_dict(),optimizer=optimizer.state_dict(),
        contract=parent['contract'],actual_updates=updates,actor_examples=updates*2,quality_approved=False),args.output/'weights.pt')
    final=review(model,args,args.output/'final',reference=args.output/'baseline',transition=True)
    write(args.output/'complete.json',dict(baseline=baseline,final=final,logs=logs,failures=failures,actual_updates=updates,
        actor_examples=updates*2,approved_parameters_unchanged=True,candidate_sha256=sha256(args.output/'weights.pt'),quality_approved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','checkpoint','dataset','audit','artifact','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
