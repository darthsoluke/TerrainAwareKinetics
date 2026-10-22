"""Bounded matched restart supervision probe; no runtime motion correction."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.decoded_factual_history import decoded_factual_history_loss
from motionbricks.data.joint_sdf_codec_objective import native_codec_reconstruction
from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.joint_sdf_training import ExactPenetrationBatch
from motionbricks.data.locomotion_command_adapter import install_locomotion_command
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.source_episode_control import source_episode_task
from motionbricks.data.source_sole_position_objective import source_sole_position_loss
from motionbricks.data.terrain_control import SoftCommandState, CONTROL_GRADIENT_VERSION
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import SOURCE_SUPPORT_MASK_VERSION
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_bones_command_alignment import review_flat, rotate_flat_case
from scripts.pilot_joint_sdf_contact_preserve import gait_metrics
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.pilot_pure_parc_stairs import review as review_stairs
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def load_command_model(saved):
    infer, bundle = load_terrain_bundle(Path(saved['contract']['source']), 'cuda:0')
    model = JointSDFController(infer, bundle)
    install_locomotion_command(model)
    model.load_state_dict(saved['model_state'], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def episode_loss(model, case, geometry, coefficients, step, phase_prior):
    source = case['world'][None].cuda()
    initial = history = source[:, :4]
    controls = case['controls'][None].cuda()
    scene = case['scene']
    driver = SoftCommandState(history_gradients=True)
    parts, prefixes = [], []
    for block in range(case['training_segments']):
        task = driver.from_controls(history, controls[:, block], 8)
        result = model(history, task, [scene], history_gradients=True, straight_through=True,
                       source=source[:, :33] if block == 0 else None)
        world = result['world']
        if block == 0:
            first = world
            first_native = model.native_losses(result, source[:, :33])['native_total']
        prefixes.append(decoded_factual_history_loss(model.inference, model.last_local[:, :32], history,
            result['observation'], root_coefficient=float(model.inference._root_model.args['local_root_loss_coeff'])))
        parts.append(world[:, 4:12])
        history = reencode_played_history(world[:, 8:12], geometry.rep, differentiable=True)
    temporal = torch.autograd.grad(world[:, 4:12].square().mean(), first, retain_graph=True)[0]
    if not torch.isfinite(temporal).all() or temporal.norm() == 0:
        raise ValueError('Missing continuous temporal gradient')
    episode = torch.cat((initial, *parts, world[:, 12:16]), 1)
    exact = ExactPenetrationBatch([scene], source.device, [HeightfieldPrismSDF(scene)])
    with torch.no_grad():
        joints, points = geometry.geometry(source)
    physical = geometry.prepared_losses(episode, initial,
        dict(joints=joints, points=points, heights=exact.heights, volumes=exact), offset=0, diagnostics=False)
    _, predicted_points = geometry.geometry(episode)
    end = episode.shape[1]-4
    xyz, active = source_sole_position_loss(predicted_points[:, 4:end, :geometry.sole_count],
        points[:, 4:end, :geometry.sole_count], exact.heights)
    if int(active.sum()) != physical['support_vertices']:
        raise ValueError('Episode source support mask differs')

    block = case.get('restart_block', step % case['training_segments'])
    window = source[:, block*8:block*8+33]
    task = source_episode_task(source, controls, block)
    # Both arms execute the same source call and token corruption. Only the
    # allocation of native actor supervision at a genuine MOVE event differs.
    auxiliary = model(window[:, :4], task, [scene], source=window, straight_through=True)
    phase_native = model.native_losses(auxiliary, window)['native_total']
    ae = native_codec_reconstruction(model, auxiliary)['loss']
    native = (first_native+phase_native)/2 if phase_prior and 'restart_block' in case else first_native+phase_native*0
    prefix = torch.stack(prefixes).mean()
    loss = native+ae+prefix+coefficients['fk_m']*xyz
    loss += sum(coefficients[k]*physical[k] for k in coefficients)
    return loss, dict(label=case['label'], segments=len(parts), native=float(native.detach()),
        first_native=float(first_native.detach()), phase_native=float(phase_native.detach()),
        phase_block=block, phase_applied=bool(phase_prior and 'restart_block' in case),
        source_ae=float(ae.detach()), prefix=float(prefix.detach()), sole_xyz_m=float(xyz.detach()),
        temporal_gradient=float(temporal.norm()), physical={k:float(physical[k].detach()) for k in coefficients})


@torch.no_grad()
def review_restarts(model, cases, geometry, directory):
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        history = case['world'][None, :4].cuda()
        controls = case['controls'].cuda()
        initial = history[:, 3, [0, 2]].clone()
        driver = SoftCommandState()
        parts = []
        for block, control in enumerate(controls):
            if block == case['restart_block']:
                move_start = history[:, 3, [0, 2]].clone()
            world = model(history, driver.from_controls(history, control[None], 8), [case['scene']])['world']
            parts.append(world[:, 4:12])
            history = reencode_played_history(world[:, 8:12], geometry.rep)
        motion = torch.cat(parts, 1)
        moving = motion[:, case['restart_block']*8:]
        direction = controls[-1, :2]/controls[-1, :2].norm()
        delta = moving[0, -1, [0, 2]]-move_start[0]
        source_end = case['world'][3+len(parts)*8, [0, 2]].to(delta)
        row = dict(label=case['label'], split=case['pilot_split'], frames=motion.shape[1],
            restart_frame=case['restart_block']*8, idle_drift_m=float((move_start-initial).norm()),
            forward_m=float(delta@direction), side_m=float(delta@torch.stack((-direction[1], direction[0]))),
            source_endpoint_error_m=float((motion[0, -1, [0, 2]]-source_end).norm()),
            **gait_metrics(geometry.rep, moving, direction), **exact_skin_metrics(geometry, motion, case['scene']))
        rows.append(row)
        save_motion(directory/(case['label']+'.npz'), motion, case['scene'], geometry.rep, row)
    write_json(directory/'metrics.json', dict(rows=rows, quality_approved=False))
    return rows


def main(args):
    if args.updates not in (1, 16):
        raise ValueError('Only one-update preflight or the declared16-update probe')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(3117)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    restart_receipt = json.loads(args.restarts.with_name('complete.json').read_text())
    if sha256(args.restarts) != restart_receipt['cases_sha256']:
        raise ValueError('Restart artifact hash mismatch')
    restarts = torch.load(args.restarts, map_location='cpu', weights_only=False)
    retention = torch.load(args.retention, map_location='cpu', weights_only=False)
    anchors = torch.load(args.anchors, map_location='cpu', weights_only=False)
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model = load_command_model(saved)
    rep = model.inference.global_motion_rep
    geometry = JointGeometryObjective(rep, 'cuda:0')
    groups = {kind:[c for c in retention if c['category']==kind] for kind in ('idle', 'walk', 'run')}
    restart_train = [c for c in restarts if c['pilot_split']=='train']
    stairs = [dict(c, controls=c['exact_constant_control'].repeat(c['training_segments'], 1))
              for c in anchors if c['pilot_split']=='train']
    names = ('control', 'locomotion_pose_condition', 'locomotion_decoder_condition')
    for name in names:
        model.get_submodule(name).requires_grad_(True)
    trainable = [name for name,p in model.named_parameters() if p.requires_grad]
    fixed = [name for name in model.state_dict() if not name.startswith(tuple(n+'.' for n in names))]
    frozen = state_digest(model, fixed)
    c = restart_train[0]
    source = c['world'][None].cuda()
    block = c['restart_block']
    window = source[:, block*8:block*8+33]
    task = source_episode_task(source, c['controls'][None].cuda(), block)
    with torch.no_grad():
        initial = model(window[:, :4], task, [c['scene']])['world'].clone()
        changed = window.clone(); changed[:, 4:] += 7
        isolated = model(window[:, :4], task, [c['scene']], source=changed)['world'].clone()
    trained = model(window[:, :4], task, [c['scene']], source=window, straight_through=True)['world']
    with torch.inference_mode():
        runtime = model(window[:, :4], task, [c['scene']])['world'].clone()
    torch.testing.assert_close(initial, isolated, atol=0, rtol=0)
    torch.testing.assert_close(initial, trained.detach(), atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(initial, runtime, atol=3e-5, rtol=1e-5)
    write_json(args.output/'preflight.json', dict(future_target_error=float((initial-isolated).abs().max()),
        training_forward_error=float((initial-trained.detach()).abs().max()), runtime_error=float((initial-runtime).abs().max())))
    del trained
    contract = dict(version='genuine-rest-restart-phase-prior-v1', arm=args.arm,
        actual_updates=args.updates, parent_sha256=sha256(args.checkpoint),
        command_gradient_version=CONTROL_GRADIENT_VERSION, support_mask_version=SOURCE_SUPPORT_MASK_VERSION,
        restarts_sha256=sha256(args.restarts), retention_sha256=sha256(args.retention), anchors_sha256=sha256(args.anchors),
        seed_scene_sha256=sha256(args.seed_scene), trainable=trainable, frozen_hash=frozen,
        optimizer='fresh Adam5e-5,clip.5', source_control=restart_receipt['control'],
        history='full generated differentiable episode; persistent release anchor; no reset at MOVE',
        native_allocation='mean(first plan,real source restart plan) only in phase arm/restart episodes; first plan otherwise',
        source_ae='one same real window per episode in both arms; source future used as target only',
        retention='one genuine long idle,one restart,one old consistent walk/run,one original PARC anchor per update',
        old_start_cases_excluded=True, holdouts='two recording families unused in this probe; not unseen pretraining',
        loss_coefficients_unchanged=True, postprocessing=None, quality_approved=False)
    write_json(args.output/'contract.json', contract)
    baseline = {}
    if args.updates > 1:
        write_json(args.output/'state.json', dict(status='baseline'))
        baseline['restarts'] = review_restarts(model, restarts, geometry, args.output/'baseline_restarts')
        baseline['flat'] = review_flat(model, args.seed_scene, args.output/'baseline_flat', geometry)
        baseline['stairs'] = review_stairs(model, anchors, geometry, args.output/'baseline_stairs')
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    generator = torch.Generator().manual_seed(3117)
    schedule, logs = [], []
    exposure = Counter()
    started = time.time()
    write_json(args.output/'state.json', dict(status='training'))
    for step in range(args.updates):
        # Use only the second walk, whose direction/facing audit is consistent.
        locomotion = groups['walk'][1] if step%2==0 else groups['run'][(step//2)%2]
        chosen = [groups['idle'][step%2], restart_train[step%len(restart_train)], locomotion]
        chosen = [rotate_flat_case(c, rep, (step%4)*np.pi/2+(float(torch.rand((), generator=generator))-.5)*np.pi/4)
                  for c in chosen]
        chosen.append(stairs[step%len(stairs)])
        optimizer.zero_grad(set_to_none=True)
        details = []
        torch.manual_seed(3117+step)
        for case in chosen:
            loss, detail = episode_loss(model, case, geometry, saved['coefficients'], step, args.arm=='phase')
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite episode loss')
            (loss/len(chosen)).backward()
            details.append(detail)
            exposure[case['label']] += case['training_segments']
            schedule.append(dict(step=step+1, label=case['label'], yaw=case.get('yaw'), segments=case['training_segments']))
            del loss
        norms = {name:sum(float(p.grad.detach().square().sum()) for p in model.get_submodule(name).parameters()
                         if p.grad is not None)**.5 for name in names}
        if min(norms.values()) <= 0:
            raise ValueError('Missing declared command gradient')
        if any(p.grad is not None for name,p in model.named_parameters() if name not in trainable):
            raise ValueError('Frozen parameter gradient')
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite gradient')
        optimizer.step()
        if state_digest(model, fixed) != frozen:
            raise ValueError('Frozen tensor changed')
        stage = dict(schema='motionbricks.training_stage.v1', name='bones_restart_'+args.arm,
            optimizer_updates=step+1, actor_primitives=sum(exposure.values()),
            source_autoencoder_examples=4*(step+1), auxiliary_source_actor_calls=4*(step+1), inherited_optimizer_updates=saved['step'])
        torch.save(dict(schema=saved['schema'], model_state=model.state_dict(), step=step+1,
            contract=dict(saved['contract'], stair_training_stage=stage, restart_repair=contract),
            coefficients=saved['coefficients'], exposure=dict(exposure), optimizer=optimizer.state_dict()), args.output/'weights.pt')
        row = dict(step=step+1, details=details, gradient_norm=float(norm), gradient_groups=norms,
            elapsed_seconds=time.time()-started, peak_memory_bytes=torch.cuda.max_memory_allocated())
        logs.append(row)
        write_json(args.output/'progress.json', row)
        print(json.dumps(row), flush=True)
    model.requires_grad_(False)
    final = {}
    if args.updates > 1:
        write_json(args.output/'state.json', dict(status='final_review'))
        final['restarts'] = review_restarts(model, restarts, geometry, args.output/'final_restarts')
        final['flat'] = review_flat(model, args.seed_scene, args.output/'final_flat', geometry)
        final['stairs'] = review_stairs(model, anchors, geometry, args.output/'final_stairs')
    write_json(args.output/'complete.json', dict(arm=args.arm, actual_updates=args.updates,
        actor_primitives=sum(exposure.values()), source_autoencoder_examples=4*args.updates,
        auxiliary_source_actor_calls=4*args.updates, exposure=dict(exposure), schedule=schedule,
        baseline=baseline, final=final, logs=logs, frozen_hashes_match=True, quality_approved=False))
    write_json(args.output/'state.json', dict(status='complete', quality_approved=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'restarts', 'retention', 'anchors', 'seed-scene', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=('control', 'phase'), required=True)
    parser.add_argument('--updates', type=int, default=16)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output/'state.json', dict(status='failed', error=repr(error)))
        raise
