"""Short on-policy STOP fit with factual histories and terrain/task-only future goals."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.decoded_factual_history import decoded_factual_history_loss
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.sole_region_contact import sole_region_indices
from motionbricks.data.stop_command_adapter import StopCommandAdapter
from motionbricks.data.stop_stage_adapter import install_stop_stage, STOP_STAGE_VERSION
from motionbricks.data.stop_task_objective import published_root_velocities, scaled_zero_velocity_loss, STOP_TASK_SCALES
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_contact_preserve import load_model
from scripts.pilot_stair_stop_command import prepare, review, write
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def task_loss(model, result, history, scene, geometry, field, block, coefficients, quality_normalized=False):
    world = result['world']
    local = model.last_local[:, :32]
    local_rep = model.inference.local_motion_rep
    indices = local_rep.indices['local_root_vel']
    zeros = torch.zeros_like(local[:,23:31,indices])
    target = local_rep.stats.normalize(zeros, index=indices)
    horizon_velocity = F.mse_loss(local[:,23:31,indices], target)
    published = world[:,4:12]
    velocities = published_root_velocities(world, history)
    normalized = local_rep.stats.normalize(velocities, index=indices)
    published_velocity = F.mse_loss(normalized, local_rep.stats.normalize(torch.zeros_like(velocities), index=indices))
    root_coefficient = float(model.inference._root_model.args['local_root_loss_coeff'])
    prefix = decoded_factual_history_loss(model.inference, local, history, result['observation'],
        root_coefficient=root_coefficient)
    _, points = geometry.geometry(world[:,4:12])
    distances, _, certified = field.query(points)
    fallback, known = heightfield_gap(points, scene)
    if not known.all():
        raise ValueError('Generated published body leaves known terrain; no discarded points')
    distances = torch.where(certified, distances, fallback)
    collision = (-distances-.005).clamp_min(0).square().mean()
    surfaces, *_ = official_sole_surfaces(model.inference.global_motion_rep.skeleton)
    split = len(sole_region_indices(surfaces, 'Left'))
    sole_gap = fallback[..., :geometry.sole_count]
    # This is a declared standing task, NOT source-foot correspondence.
    support = torch.stack((sole_gap[..., :split].abs().amin(-1),
        sole_gap[..., split:].abs().amin(-1)), -1)[:, -4:].mean()
    foot_velocity = torch.diff(points[:,-4:,:geometry.sole_count], dim=1).norm(dim=-1).mean()*30
    settled_phase = block*8/30 >= 1.
    loss = prefix+root_coefficient*horizon_velocity+coefficients['collision_m2']*collision
    if settled_phase:
        loss = loss+root_coefficient*published_velocity+coefficients['support_gap_m']*support
        loss = loss+coefficients['source_support_velocity_mps']*foot_velocity
    native_scale_total = loss
    if quality_normalized:
        physical = local_rep.stats.unnormalize(local[:,23:31,indices], index=indices)
        excess = (-distances-.005).clamp_min(0).flatten()
        tail = excess.topk(max(1, int(np.ceil(excess.numel()*.01)))).values
        loss = prefix+scaled_zero_velocity_loss(physical)+(tail/STOP_TASK_SCALES['penetration_m']).square().mean()
        if settled_phase:
            loss = loss+scaled_zero_velocity_loss(velocities)+support/STOP_TASK_SCALES['support_gap_m']
            loss = loss+foot_velocity/STOP_TASK_SCALES['planted_foot_velocity_mps']
    return loss, dict(prefix=float(prefix.detach()), horizon_velocity=float(horizon_velocity.detach()),
        published_velocity=float(published_velocity.detach()), collision_m2=float(collision.detach()),
        stand_support_gap_m=float(support.detach()), stand_foot_velocity_mps=float(foot_velocity.detach()),
        settled_objective_active=settled_phase, observed_stop_seconds=block*8/30,
        quality_normalized=quality_normalized, native_scale_total=float(native_scale_total.detach())), native_scale_total


def main(args):
    if args.episodes != 4:
        raise ValueError('This probe is limited to four10-primitive episodes')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = torch.load(args.parent, map_location='cpu', mmap=True, weights_only=False)
    if sha256(args.parent) != 'ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf':
        raise ValueError('Approved parent differs')
    if sha256(args.checkpoint) != '8bd03d3d069c3b160f22f652fc7dc51a2bda1a73e86c8a10f692cc346fcb9571':
        raise ValueError('Expected matched Root-only stop initialization')
    model, _ = load_model(parent)
    model.control = StopCommandAdapter(model.control)
    initial = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(initial['model_state'], strict=True)
    model.requires_grad_(False)
    model.control.stop.requires_grad_(True)
    pose, _ = install_stop_stage(model, 'pose')
    decoder, _ = install_stop_stage(model, 'decoder')
    model.eval()
    trainable_prefixes = ('control.stop.', 'stop_pose_condition.', 'stop_decoder_condition.')
    fixed = [key for key in model.state_dict() if not key.startswith(trainable_prefixes)]
    immutable = state_digest(model, fixed)
    source_cases = prepare(model, args)
    baseline = review(model, args, args.output/'baseline')
    geometry = JointGeometryObjective(model.inference.global_motion_rep, 'cuda:0')
    receipt = json.loads((args.artifact/'export.json').read_text())
    episodes = []
    for entry in receipt['entries']:
        with np.load(args.artifact/(entry['entry']+'-scene.npz'), allow_pickle=False) as data:
            scene = {key:torch.from_numpy(data[key].copy()).cuda()
                for key in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
        with np.load(args.checkpoint.parent/'final'/(entry['entry']+'-cruise.npz'), allow_pickle=False) as data:
            moving = torch.from_numpy(data['world'].copy())[None].cuda()
        for starting_block in (2,3):
            history = reencode_played_history(moving[:,starting_block*8-4:starting_block*8], model.inference.global_motion_rep)
            episodes.append(dict(entry=entry, history=history, scene=scene, field=HeightfieldPrismSDF(scene), starting_block=starting_block))
    contract = dict(version=STOP_STAGE_VERSION, episodes=4, max_actual_updates=40,
        initialization_and_training_seed=2333,
        max_actor_examples=80, primitive_counts='One actual generated stair primitive and one real source primitive per update',
        history='Independent genuine prior MOVE states; continuous generated history within each STOP episode, no reset or retarget',
        on_policy_future='Requested velocity0 and actual terrain support/collision only. No source-time future pose/Root labels',
        stopping='One-second braking grace for published speed/support/slip task; forecast tail velocity goal remains0',
        source_prior='Same eight screened real braking/rest windows, with true source histories',
        coefficients=dict(parent['coefficients']), root_coefficient=float(model.inference._root_model.args['local_root_loss_coeff']),
        optimizer='Fresh Adam5e-5,clip.5; original parameters frozen',
        support_scope='Both feet have a supporting surface, not a force/contact certificate',
        quality_normalized=args.quality_normalized, quality_scales=dict(STOP_TASK_SCALES),
        velocity_target='Desired command goal, not a runtime speed cap or motion overwrite',
        future_truth_in_actor=False, postprocessing=None, quality_approved=False)
    write(args.output/'contract.json', contract)
    torch.manual_seed(2333)
    parameters = list(model.control.stop.parameters())+list(pose.parameters())+list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=5e-5, weight_decay=0.)
    logs, failures = [], []
    updates = 0
    for episode_index, episode in enumerate(episodes):
        history = episode['history'].clone()
        driver = SoftCommandState()
        parts = []
        for block in range(10):
            optimizer.zero_grad(set_to_none=True)
            task = driver.intent(history, [0.,0.], episode['entry']['command_heading'], 8)
            try:
                result = model(history, task, [episode['scene']], straight_through=True)
                task_total, detail, native_task_total = task_loss(model, result, history, episode['scene'], geometry,
                    episode['field'], block, parent['coefficients'], args.quality_normalized)
            except (ValueError, RuntimeError) as error:
                failures.append(dict(episode=episode_index, block=block, actual_frames=len(parts)*8, error=str(error)))
                break
            published = result['world'][:,4:12].detach().clone()
            next_history = reencode_played_history(result['world'][:,8:12].detach(), model.inference.global_motion_rep)
            prior = source_cases[updates % len(source_cases)]
            source = prior['world']
            source_result = model(source[:,:4], prior['intent'], [prior['scene']], source=source, straight_through=True)
            source_prefix = decoded_factual_history_loss(model.inference, model.last_local[:,:32], source[:,:4],
                source_result['observation'], root_coefficient=contract['root_coefficient'])
            source_native = model.native_losses(source_result, source)['native_total']
            gradient_audit = None
            if block in (0,4):
                gradients = {}
                for name, objective in [('task', task_total), ('native_task', native_task_total),
                                        ('source_prior', source_prefix+source_native)]:
                    values = torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
                    norms = []
                    offset = 0
                    for module in (model.control.stop, pose, decoder):
                        count = len(list(module.parameters()))
                        valid = [value for value in values[offset:offset+count] if value is not None]
                        norms.append(float(sum((g.detach().double().square().sum() for g in valid),
                            torch.zeros((), device='cuda:0')).sqrt()))
                        offset += count
                    gradients[name] = dict(zip(('root','pose','decoder'), norms))
                gradient_audit = gradients
            loss = (task_total+source_prefix+source_native)/2
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite on-policy task/source loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError('Invalid combined stop-branch gradient')
            optimizer.step()
            updates += 1
            if state_digest(model, fixed) != immutable:
                raise ValueError('Approved parameters changed')
            parts.append(published)
            history = next_history
            row = dict(step=updates, episode=episode_index, entry=episode['entry']['entry'],
                block=block, loss=float(loss.detach()), source_native=float(source_native.detach()),
                gradient=float(norm), gradient_audit=gradient_audit, **detail)
            logs.append(row)
            write(args.output/'progress.json', dict(rows=logs, failures=failures))
            print(json.dumps(row), flush=True)
        if parts:
            save_motion(args.output/f'train-episode{episode_index}.npz', torch.cat(parts,1), episode['scene'],
                model.inference.global_motion_rep, dict(role='actual changing-policy training recording',
                    entry=episode['entry']['entry'], initial_MOVE_blocks=episode['starting_block'],
                    frames=len(parts)*8, quality_approved=False, failures=failures))
    torch.save(dict(schema='motionbricks.stop_feedback_pilot.v1', version=STOP_STAGE_VERSION,
        model_state=model.state_dict(), optimizer=optimizer.state_dict(), contract=parent['contract'],
        actual_updates=updates, actor_examples=updates*2, quality_approved=False), args.output/'weights.pt')
    final = review(model, args, args.output/'final', reference=args.output/'baseline')
    write(args.output/'complete.json', dict(baseline=baseline, final=final, logs=logs, failures=failures,
        actual_updates=updates, actor_examples=updates*2, approved_parameters_unchanged=True,
        candidate_sha256=sha256(args.output/'weights.pt'), quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent','checkpoint','dataset','audit','artifact','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--quality-normalized', action='store_true')
    main(parser.parse_args())
