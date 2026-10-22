"""Eight-update matched causal command-interface probe with real full episodes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.bones_controller_episode import rotate_controls
from motionbricks.data.decoded_factual_history import decoded_factual_history_loss
from motionbricks.data.joint_sdf_codec_objective import native_codec_reconstruction
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.joint_sdf_training import ExactPenetrationBatch
from motionbricks.data.locomotion_command_adapter import COMMAND_VERSION, install_locomotion_command
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.source_sole_position_objective import source_sole_position_loss
from motionbricks.data.terrain_control import SoftCommandState,CONTROL_GRADIENT_VERSION
from motionbricks.data.terrain_response_loss import SOURCE_SUPPORT_MASK_VERSION
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_contact_preserve import load_model, gait_metrics
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.pilot_pure_parc_stairs import review as review_stairs
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def rotate_flat_case(case, rep, yaw):
    world = case['world'][None].cuda()
    previous = torch.atan2(world[0, 0, 4], world[0, 0, 3])
    rotated = rep.change_first_heading(world, yaw, False, False)
    rotated[..., [0, 2]] += world[:, :1, [0, 2]]
    return dict(case, world=rotated[0], controls=rotate_controls(case['controls'].cuda(), yaw-float(previous)), yaw=yaw)


def episode_loss(model, case, geometry, coefficients, step):
    source = case['world'][None].cuda()
    initial = history = source[:, :4]
    controls = case['controls'].cuda()
    scene = case['scene']; driver = SoftCommandState(history_gradients=True)
    parts, prefixes = [], []
    first = None
    for block in range(case['training_segments']):
        task = driver.from_controls(history, controls[block:block+1], 8)
        result = model(history, task, [scene], history_gradients=True, straight_through=True,
            source=source[:, :33] if block == 0 else None)
        world = result['world']
        if block == 0:
            first = world
            native = model.native_losses(result, source[:, :33])['native_total']
        prefixes.append(decoded_factual_history_loss(model.inference, model.last_local[:, :32], history,
            result['observation'], root_coefficient=float(model.inference._root_model.args['local_root_loss_coeff'])))
        parts.append(world[:, 4:12])
        history = reencode_played_history(world[:, 8:12], geometry.rep, differentiable=True)
    temporal = torch.autograd.grad(world[:, 4:12].square().mean(), first, retain_graph=True)[0]
    if not torch.isfinite(temporal).all() or temporal.norm() == 0:
        raise ValueError('Missing continuous temporal gradient')
    episode = torch.cat((initial, *parts, world[:, 12:16]), 1)
    field = HeightfieldPrismSDF(scene)
    exact = ExactPenetrationBatch([scene], source.device, [field])
    with torch.no_grad(): joints, points = geometry.geometry(source)
    prepared = dict(joints=joints, points=points, heights=exact.heights, volumes=exact)
    physical = geometry.prepared_losses(episode, initial, prepared, offset=0, diagnostics=False)
    _, predicted_points = geometry.geometry(episode)
    end = episode.shape[1]-4
    xyz, active = source_sole_position_loss(predicted_points[:, 4:end, :geometry.sole_count],
        points[:, 4:end, :geometry.sole_count], exact.heights)
    if int(active.sum()) != physical['support_vertices']:
        raise ValueError('Episode source support mask differs')
    offset = (step % case['training_segments'])*8
    window = source[:, offset:offset+33]
    with torch.no_grad():
        intent = SoftCommandState().from_controls(window[:, :4], controls[step % len(controls):step % len(controls)+1], 8)
        auxiliary = model(window[:, :4], intent, [scene], source=window)
    ae = native_codec_reconstruction(model, auxiliary)['loss']
    prefix = torch.stack(prefixes).mean()
    loss = native+ae+prefix+coefficients['fk_m']*xyz
    loss += sum(coefficients[k]*physical[k] for k in coefficients)
    return loss, dict(label=case['label'], segments=len(parts), native=float(native.detach()),
        source_ae=float(ae.detach()), prefix=float(prefix.detach()), sole_xyz_m=float(xyz.detach()),
        temporal_gradient=float(temporal.norm()), physical={k:float(physical[k].detach()) for k in coefficients})


@torch.no_grad()
def review_flat(model, seed_path, directory, geometry):
    directory.mkdir(parents=True, exist_ok=False)
    with np.load(seed_path, allow_pickle=False) as data:
        initial = torch.from_numpy(data['initial_history'].copy()).cuda()
    rep = model.inference.global_motion_rep
    initial = rep.change_first_heading(initial, 0., False, False)
    initial[..., [0,2]] -= initial[:, -1:, [0,2]].clone()
    scene = dict(heightfield_y_m=torch.zeros(201,201), min_point_world_xyz_m=torch.tensor([-50.,0.,-50.]), dx_m=torch.tensor(.5))
    cases = [(yaw,speed) for yaw in (0.,np.pi/2,np.pi,-np.pi/2) for speed in (1.6,5.)]
    all_rows = []
    for idle_blocks in (0,60):
        history = torch.cat([rep.change_first_heading(initial.clone(), yaw, False, False) for yaw,speed in cases])
        axes = history.new_tensor([[np.sin(yaw),np.cos(yaw)] for yaw,speed in cases])
        controls = torch.cat((axes*history.new_tensor([s for y,s in cases])[:,None],
            history.new_tensor([[np.cos(y),np.sin(y)] for y,s in cases])), -1)
        idle_start = history[:,3,[0,2]].clone(); driver = SoftCommandState(); parts = []
        idle_parts = []
        for block in range(idle_blocks+30):
            request = controls.clone()
            if block < idle_blocks: request[:,:2] = 0
            if block == idle_blocks: start = history[:,3,[0,2]].clone()
            world = model(history, driver.from_controls(history, request, 8), [scene]*len(cases))['world']
            (idle_parts if block < idle_blocks else parts).append(world[:,4:12].clone())
            history = reencode_played_history(world[:,8:12],rep)
        motion = torch.cat(parts,1)
        for index,(yaw,speed) in enumerate(cases):
            actual = motion[index:index+1]; delta=actual[0,-1,[0,2]]-start[index]
            forward = torch.diff(actual[0,:,[0,2]],dim=0)@axes[index]*30
            row = dict(idle_seconds=idle_blocks*8/30,yaw=yaw,request_speed=speed,
                frames=idle_blocks*8+240,movement_frames=240,
                idle_drift_m=float((start[index]-idle_start[index]).norm()),forward_m=float(delta@axes[index]),
                side_m=float(delta@torch.stack((-axes[index,1],axes[index,0]))),
                speed_last_second=float(forward[-30:].mean()),low_progress_fraction=float((forward<.2).float().mean()),
                **gait_metrics(rep,actual,axes[index]),**exact_skin_metrics(geometry,actual,scene))
            if idle_parts:
                row['idle_geometry']=exact_skin_metrics(geometry,torch.cat(idle_parts,1)[index:index+1],scene)
            all_rows.append(row)
            complete = torch.cat((*idle_parts,motion),1)[index:index+1] if idle_parts else actual
            save_motion(directory/f'idle{idle_blocks}-case{index}.npz',complete,scene,rep,row)
        write_json(directory/'metrics.json',dict(rows=all_rows,quality_approved=False))
    return all_rows


def main(args):
    if args.updates != 8: raise ValueError('This declared probe is exactly8updates')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha256(args.checkpoint) != 'eb631648f5144d5353384d74c85059d1ba2bbe05379492233f7a9a8a0bfb7cde':
        raise ValueError('Expected exact deployed candidate parent')
    if sha256(args.seed_scene) != 'e19876a349c54ba4a2c663b358f9e33bc0bab2e115f302e91ff3f63b7f819ffa':
        raise ValueError('Expected actual large-map entry seed')
    data_receipt=json.loads(args.cases.with_name('complete.json').read_text())
    if sha256(args.cases)!=data_receipt['cases_sha256']: raise ValueError('Source episode artifact changed')
    cases=torch.load(args.cases,map_location='cpu',weights_only=False)
    anchors=torch.load(args.anchors,map_location='cpu',weights_only=False)
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model,_=load_model(saved);rep=model.inference.global_motion_rep
    source=cases[0]['world'][None,:33].cuda();task=SoftCommandState().from_controls(source[:,:4],cases[0]['controls'][:1].cuda(),8)
    with torch.no_grad(): initial=model(source[:,:4],task,[cases[0]['scene']])['world'].clone()
    torch.manual_seed(2333);install_locomotion_command(model)
    with torch.no_grad():
        installed=model(source[:,:4],task,[cases[0]['scene']])['world'].clone()
        changed=source.clone();changed[:,4:]+=7
        isolated=model(source[:,:4],task,[cases[0]['scene']],source=changed)['world'].clone()
    torch.testing.assert_close(initial,installed,atol=0,rtol=0)
    torch.testing.assert_close(initial,isolated,atol=0,rtol=0)
    model.requires_grad_(False);model.control.requires_grad_(True)
    if args.arm=='all_stages':
        model.locomotion_pose_condition.requires_grad_(True)
        model.locomotion_decoder_condition.requires_grad_(True)
    trainable=[name for name,p in model.named_parameters() if p.requires_grad]
    trained=model(source[:,:4],task,[cases[0]['scene']],source=source,straight_through=True)['world']
    torch.testing.assert_close(initial,trained,atol=3e-5,rtol=1e-5)
    training_error=float((initial-trained.detach()).abs().max())
    with torch.inference_mode(): runtime=model(source[:,:4],task,[cases[0]['scene']])['world'].clone()
    torch.testing.assert_close(initial,runtime,atol=3e-5,rtol=1e-5)
    del trained
    fixed=[name for name in model.state_dict() if not name.startswith(('control.','locomotion_pose_condition.','locomotion_decoder_condition.'))]
    frozen=state_digest(model,fixed)
    geometry=JointGeometryObjective(rep,'cuda:0')
    groups={kind:[c for c in cases if c['category']==kind] for kind in ('idle','start','walk','run')}
    training_stairs=[dict(c,controls=c['exact_constant_control'].repeat(c['training_segments'],1)) for c in anchors if c['pilot_split']=='train']
    source_screen=[]
    for c in cases:
        metrics=exact_skin_metrics(geometry,c['world'][None].cuda(),c['scene'])
        source_screen.append(dict(label=c['label'],**metrics))
    write_json(args.output/'source_screen.json',dict(rows=source_screen))
    # Native BONES reconstruction is not a terrain-clearance certificate; preserve
    # every source measurement, and reject grossly intersecting supervision.
    if any(r['body_prism_max_m']>.02 or r['unknown_body_fraction'] or r['uncertain_inside_samples'] for r in source_screen):
        raise ValueError('A selected source fails the existing2cm exact-skin screen')
    contract=dict(version=COMMAND_VERSION,arm=args.arm,actual_updates=8,parent_sha256=sha256(args.checkpoint),
        command_gradient_version=CONTROL_GRADIENT_VERSION,support_mask_version=SOURCE_SUPPORT_MASK_VERSION,
        cases_sha256=sha256(args.cases),seed_scene_sha256=sha256(args.seed_scene),trainable=trainable,
        frozen_hash=frozen,command='existing7 Root features; desired VX/VZ,facing,horizon and MOVE flag to Pose/decoder; no future body',
        history='continuous factual generated/re-encoded history with temporal gradients and persistent SoftCommandState',
        augmentation='same deterministic random yaw inside each cardinal quadrant on BONES; paired zero plane; original stairs unchanged',
        losses='same native/sourceAE/factual-prefix/full-episode geometry and source-sole XYZ; no coefficient changes',
        optimizer='fresh Adam5e-5,clip.5',original_network_frozen=True,source_pose_keyframes=False,
        not_literal_original_MB_smart_primitive=True,postprocessing=None,quality_approved=False)
    write_json(args.output/'contract.json',contract)
    write_json(args.output/'preflight.json',dict(initial_error=float((initial-installed).abs().max()),
        future_target_error=float((initial-isolated).abs().max()),training_forward_error=training_error,
        runtime_error=float((initial-runtime).abs().max())))
    write_json(args.output/'state.json',dict(status='baseline'))
    baseline=review_flat(model,args.seed_scene,args.output/'baseline_flat',geometry)
    baseline_stairs=review_stairs(model,anchors,geometry,args.output/'baseline_stairs')
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=5e-5);schedule=[];exposure=Counter();logs=[]
    generator=torch.Generator().manual_seed(2333)
    started=time.time();write_json(args.output/'state.json',dict(status='training'))
    for step in range(8):
        chosen=[groups['idle'][step%2],groups['start'][step%2],groups['walk' if step%2==0 else 'run'][(step//2)%2]]
        chosen=[rotate_flat_case(c,rep,(step%4)*np.pi/2+(float(torch.rand((),generator=generator))-.5)*np.pi/4) for c in chosen]
        chosen.append(training_stairs[step%len(training_stairs)])
        optimizer.zero_grad(set_to_none=True);details=[]
        torch.manual_seed(2333+step)
        for case in chosen:
            loss,detail=episode_loss(model,case,geometry,saved['coefficients'],step)
            if not torch.isfinite(loss): raise ValueError('Nonfinite episode objective')
            (loss/len(chosen)).backward();details.append(detail)
            exposure[case['label']]+=case['training_segments']
            schedule.append(dict(step=step+1,label=case['label'],yaw=case.get('yaw'),segments=case['training_segments']))
            del loss
        norms={name:sum(float(p.grad.detach().square().sum()) for p in model.get_submodule(name).parameters() if p.grad is not None)**.5
            for name in ('control','locomotion_pose_condition','locomotion_decoder_condition')}
        if norms['control']<=0 or (args.arm=='all_stages' and min(norms.values())<=0): raise ValueError('Missing declared command gradient')
        if any(p.grad is not None for name,p in model.named_parameters() if name not in trainable): raise ValueError('Frozen parameter gradient')
        norm=torch.nn.utils.clip_grad_norm_(parameters,.5)
        if not torch.isfinite(norm): raise ValueError('Nonfinite gradient')
        optimizer.step()
        if state_digest(model,fixed)!=frozen: raise ValueError('Frozen model tensor changed')
        stage=dict(schema='motionbricks.training_stage.v1',name='bones_command_'+args.arm,optimizer_updates=step+1,
            actor_primitives=sum(exposure.values()),source_autoencoder_examples=4*(step+1),inherited_optimizer_updates=saved['step'])
        torch.save(dict(schema=saved['schema'],model_state=model.state_dict(),step=step+1,
            contract=dict(saved['contract'],stair_training_stage=stage,locomotion_command=contract),
            coefficients=saved['coefficients'],exposure=dict(exposure),optimizer=optimizer.state_dict()),args.output/'weights.pt')
        row=dict(step=step+1,details=details,gradient_norm=float(norm),gradient_groups=norms,
            elapsed_seconds=time.time()-started,peak_memory_bytes=torch.cuda.max_memory_allocated())
        logs.append(row);write_json(args.output/'progress.json',row);print(json.dumps(row),flush=True)
    model.requires_grad_(False);write_json(args.output/'state.json',dict(status='final_review'))
    final=review_flat(model,args.seed_scene,args.output/'final_flat',geometry)
    final_stairs=review_stairs(model,anchors,geometry,args.output/'final_stairs')
    write_json(args.output/'complete.json',dict(arm=args.arm,actual_updates=8,actor_primitives=sum(exposure.values()),
        source_autoencoder_examples=32,auxiliary_source_actor_calls=32,exposure=dict(exposure),schedule=schedule,
        baseline_flat=baseline,final_flat=final,baseline_stairs=baseline_stairs,final_stairs=final_stairs,
        logs=logs,frozen_hashes_match=True,quality_approved=False))
    write_json(args.output/'state.json',dict(status='complete',quality_approved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','cases','anchors','seed-scene','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=('root_only','all_stages'),required=True)
    parser.add_argument('--updates',type=int,default=8)
    args=parser.parse_args()
    try: main(args)
    except Exception as error:
        if args.output.exists(): write_json(args.output/'state.json',dict(status='failed',error=repr(error)))
        raise
