"""Small on-policy contact A/B preserving the accepted actor's gait and travel."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from motionbricks.data.decoded_factual_history import decoded_factual_history_loss
from motionbricks.data.decoded_factual_history import SEPARATE_FACTUAL_ROOT_VERSION
from motionbricks.data.joint_sdf_codec_objective import native_codec_reconstruction
from motionbricks.data.joint_sdf_controller import JointSDFController, SDF_VERSION
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.terrain_skin_tail_loss import SKIN_TAIL_VERSION, skin_tail_penetration
from motionbricks.data.terrain_skin_tail_loss import PHASE_PROXY_VERSION, teacher_phase_support_proxy, teacher_support_gap
from motionbricks.data.terrain_skin_tail_loss import direction_without_speed_loss
from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.sole_region_contact import sole_region_indices
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_factual_prefix import frozen_hashes
from scripts.pilot_joint_sdf_full_context import command, evaluate, exact_skin_metrics, load_cases
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.prepare_joint_sdf_context_pilot import recorded_direction
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def load_model(saved):
    infer, bundle = load_terrain_bundle(Path(saved['contract']['source']), 'cuda:0')
    model = JointSDFController(infer, bundle)
    model.load_state_dict(saved['model_state'], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, bundle


def prepare_cases(path, dataset):
    cases = load_cases(path, dataset)
    data = TerrainLearningDataset(dataset, build_native_representation(dataset / 'stats'), 'train')
    indexed = {row['source_id']: index for index, row in enumerate(data.records)}
    for case in cases[:3]:
        full = data.full_reference_item(indexed[case['row']['source_id']])
        start = case['start_frame']
        extended = full['world'][start:start + 105]
        if len(extended) != 105:
            raise ValueError('Need ten complete real33-frame windows; never pad the tail')
        torch.testing.assert_close(extended[:len(case['world'])], case['world'], atol=0, rtol=0)
        case['world'] = extended
        case['training_segments'] = 10
        case['recorded_direction'] = recorded_direction(data, case['row'], case['row']['frames_30hz'])[start:start + 105]
        world = extended[None].cuda()
        edge = case['edge_frame']
        velocity = (world[0, edge - 8, [0, 2]] - world[0, 3, [0, 2]]) / ((edge - 11) / 30)
        heading = torch.atan2(world[:, 3:edge - 8, 4].mean(1), world[:, 3:edge - 8, 3].mean(1))
        case['exact_constant_control'] = torch.cat((velocity, heading.cos(), heading.sin()))[None]
    return cases


def control(case, position):
    return case['exact_constant_control'] if case['name'] == 'stairs' else command(case, position * 8)


def skin_vertices(rep, geometry, world):
    pose = rep.inverse(world, False, return_all=True, joint_positions_from='global_rot_data')
    batch, frames = world.shape[:2]
    points = geometry.skin.skin(pose['local_joint_rots'].reshape(-1, 30, 3, 3),
        world[..., :3].reshape(-1, 1, 3).expand(-1, 30, -1), rot_is_global=False)
    return points.reshape(batch, frames, -1, 3), pose['posed_joints']


def gait_metrics(rep, world, direction):
    joints = rep.inverse(world, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints'][0]
    ids = [rep.skeleton.bone_order_names.index(side + 'Foot') for side in ('Left', 'Right')]
    feet = joints[:, ids] - world[0, :, None, :3]
    foreaft = (feet[..., [0, 2]] @ direction)
    difference = foreaft[:, 0] - foreaft[:, 1]
    correlation = np.corrcoef(torch.diff(feet[..., 1], dim=0).cpu().numpy().T)[0, 1]
    return dict(foreaft_rms_m=float(difference.square().mean().sqrt()),
        close_feet_fraction=float((difference.abs() < .1).float().mean()),
        last32_close_feet_fraction=float((difference[-32:].abs() < .1).float().mean()),
        foot_vertical_change_correlation=float(correlation) if np.isfinite(correlation) else None)


@torch.no_grad()
def exact_reviews(model, cases, geometry, directory, reference=None):
    directory.mkdir(parents=True, exist_ok=False)
    rep = model.inference.global_motion_rep
    rows = []
    for case in cases:
        source = case['world'][None].cuda()
        history = source[:, :4].clone()
        state = SoftCommandState()
        initial = history.clone()
        requested = 10 if case['name'] == 'stairs' else 13
        parts, failure = [], None
        for block in range(requested):
            intent = state.from_controls(history, control(case, block), 8)
            try:
                result = model(history, intent, [case['scene']])
            except ValueError as error:
                failure = str(error)
                break
            parts.append(result['world'][:, 4:12])
            history = reencode_played_history(result['world'][:, 8:12], rep)
        row = dict(label=case['label'], frames=len(parts) * 8, requested_frames=requested * 8,
                   failure=failure, quality_approved=False, postprocessing=None)
        if parts:
            world = torch.cat(parts, 1)
            direction = control(case, 0)[0, :2]
            direction = direction / direction.norm().clamp_min(1e-6)
            row.update(exact_skin_metrics(geometry, world, case['scene']))
            row.update(gait_metrics(rep, world, direction))
            delta = world[0, -1, [0, 2]] - initial[0, 3, [0, 2]]
            row.update(progress_m=float(delta @ direction), root_drift_m=float(delta.norm()),
                       cross_track_m=float((delta @ torch.stack((-direction[1], direction[0]))).abs()))
            _, points = geometry.geometry(world)
            gap, known = heightfield_gap(points[0, :, :geometry.sole_count], case['scene'])
            lowest = gap.amin(-1)
            row['all_soles_above10cm_fraction'] = float((lowest > .1).float().mean())
            longest = current = 0
            for airborne in (lowest > .05).tolist():
                current = current + 1 if airborne else 0
                longest = max(longest, current)
            row['longest_all_soles_above5cm_frames'] = longest
            row['sole_vertical_p99_m'] = float(torch.quantile((-gap[known]).clamp_min(0), .99))
            row['source_end_root_height_error_m'] = abs(float(world[0, -1, 1] - source[0, 3 + len(parts) * 8, 1]))
            if reference is not None and failure is None:
                with np.load(reference / (case['label'] + '.npz'), allow_pickle=False) as archive:
                    before = torch.from_numpy(archive['world'].copy()).to(world)[None]
                if before.shape == world.shape:
                    pj = rep.inverse(world, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
                    bj = rep.inverse(before, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
                    row['relative_body_change_m'] = float(((pj - world[:, :, None, :3]) - (bj - before[:, :, None, :3])).norm(dim=-1).mean())
                    row['root_xz_change_rms_m'] = float((world[..., [0, 2]] - before[..., [0, 2]]).square().sum(-1).mean().sqrt())
                    row['root_y_change_mean_m'] = float((world[..., 1] - before[..., 1]).mean())
            save_motion(directory / (case['label'] + '.npz'), world, case['scene'], rep, row)
            save_motion(directory / (case['label'] + '-source.npz'), source[:, 4:4 + len(parts) * 8],
                        case['scene'], rep, dict(role='unchanged_reference', quality_approved=False))
        rows.append(row)
        write_json(directory / 'metrics.json', dict(rows=rows, quality_approved=False))
    return rows


def main(args):
    if args.separate_root_preservation and args.direction_only_preservation:
        raise ValueError('Choose either speed preservation or direction-only control')
    if not 1 <= args.updates <= 200:
        raise ValueError('This matched pilot is limited to200 actual updates')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state_path = args.output / 'state.json'
    write_json(state_path, dict(status='initializing', quality_approved=False))
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if sha256(args.checkpoint) != 'b4faa6a64defb4d7f76320f5a4003ade7a88038a2a527a61f9ed72d064df0f35':
        raise ValueError('Must start from the user-accepted orange prefix checkpoint')
    if sha256(Path(saved['contract']['source'])) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source bundle identity mismatch')
    model, bundle = load_model(saved)
    teacher, _ = load_model(saved)
    rep = model.inference.global_motion_rep
    cases = prepare_cases(args.cases, Path(saved['contract']['dataset']))
    geometry = JointGeometryObjective(rep, 'cuda:0')
    surfaces, *_ = official_sole_surfaces(rep.skeleton)
    foot_sizes = [len(sole_region_indices(surfaces, side)) for side in ('Left', 'Right')]
    sole_indices = torch.as_tensor(geometry.indices[:geometry.sole_count], device='cuda:0', dtype=torch.long)
    fields = [HeightfieldPrismSDF(case['scene']) for case in cases]
    frozen = frozen_hashes(model)
    teacher_hash = state_digest(teacher)
    collision_coefficient = float(saved['coefficients']['collision_m2'])
    fk_coefficient = float(saved['coefficients']['fk_m'])
    contract = dict(arm=args.arm, actual_updates=args.updates, batch=4,
        source_checkpoint=str(args.checkpoint), source_sha256=sha256(args.checkpoint),
        trainable='codec.decoder only', frozen_hashes=frozen, teacher_sha256=teacher_hash,
        source_windows='3 unchanged real105-frame stair passages;10 published steps, plus alternating idle/walk13steps',
        exposure='each update:4 on-policy actor prefixes +4 frozen teacher calls +4 source-autoencoder examples',
        actor_history='actual current student generated history, detached each8frame block; no source reset within an episode',
        preservation='same-input frozen accepted actor: normalized rotation/root-XZ-velocity plus pelvis-relative FK; absolute Y not distilled',
        source_autoencoder='unchanged native loss; future source Root confined to separate source reconstruction',
        losses=dict(native_source=1., factual_prefix=1., native_rotation_and_planar=1.,
                    relative_fk_m=fk_coefficient, collision_m2=collision_coefficient if args.arm == 'contact' else 0.,
                    weak_support_gap_m=float(saved['coefficients']['support_gap_m']) if args.phase_support else 0.),
        support_proxy=PHASE_PROXY_VERSION if args.phase_support else None,
        separate_root_preservation=args.separate_root_preservation,
        direction_only_preservation=args.direction_only_preservation,
        separate_factual_root=args.separate_factual_root,
        factual_root_version=SEPARATE_FACTUAL_ROOT_VERSION if args.separate_factual_root else None,
        factual_root_coefficient=float(model.inference._root_model.args['local_root_loss_coeff']) if args.separate_factual_root else None,
        direction_policy='zero penalty at any nonnegative collinear pace, mean reverse/lateral penalty; idle zero velocity' if args.direction_only_preservation else None,
        root_preservation_coefficient=float(model.inference._root_model.args['local_root_loss_coeff']) if (args.separate_root_preservation or args.direction_only_preservation) else None,
        support_scope='frozen-teacher backward/lower-foot phase, not measured force contacts; actual student XY queries observed terrain',
        collision_version=SKIN_TAIL_VERSION, vertices=18056, swept_substeps=4,
        risk='worst1percent of squared exact penetration,5mm margin; chosen to avoid dilution by nonpenetrating body, no coefficient tuning',
        optimizer='fresh Adam5e-5, clip0.5', inference='unchanged four-pass/one-refinement eager actor',
        future_truth_in_actor=False, postprocessing=None, quality_approved=False)
    write_json(args.output / 'contract.json', contract)
    write_json(state_path, dict(status='baseline', quality_approved=False))
    baseline = exact_reviews(model, cases, geometry, args.output / 'baseline')
    if args.reference_trial:
        from scripts.summarize_joint_sdf_context_pair import assert_same
        prior_trial = json.loads(args.reference_trial.read_text())
        assert_same(baseline, prior_trial['baseline'])
        if prior_trial['actual_updates'] != args.updates:
            raise ValueError('Matched collision-only trial budget differs')
    baseline_case = args.output / 'baseline/generated_8030ae7a2a7ad640.npz'
    with np.load(baseline_case, allow_pickle=False) as a, np.load(args.accepted_recording, allow_pickle=False) as b:
        np.testing.assert_allclose(a['world'], b['world'], atol=3e-5, rtol=1e-5)
        recording_difference = float(np.abs(a['world'] - b['world']).max())
    model.codec.decoder.requires_grad_(True)
    parameters = list(model.codec.decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    positions, indices = [0] * 4, [0, 1, 2, 3]
    histories = [None] * 4
    states = [SoftCommandState() for _ in range(4)]
    exposure, schedule = Counter(), hashlib.sha256()
    logs = []
    started = time.perf_counter()
    write_json(state_path, dict(status='training', quality_approved=False))
    contact_gradient_seen = False
    # Local and global body columns differ by one Root component.
    local_rotation_indices = list(model.inference.local_motion_rep.indices['global_rot_data'])
    root_indices = [int(i) for i in model.inference.local_motion_rep.indices['root']
                    if i not in model.inference.local_motion_rep.indices['global_root_y']]
    preserved_indices = local_rotation_indices + root_indices
    velocity_indices = list(model.inference.local_motion_rep.indices['local_root_vel'])
    angular_indices = [index for index in root_indices if index not in velocity_indices]
    for step in range(args.updates):
        for row, index in enumerate(indices):
            if positions[row] == cases[index]['training_segments']:
                positions[row] = 0
                histories[row] = None
                states[row] = SoftCommandState()
                if row == 3:
                    indices[row] = 4 if index == 3 else 3
        selected = [cases[index] for index in indices]
        windows = torch.stack([case['world'][position * 8:position * 8 + 33]
            for case, position in zip(selected, positions)]).cuda()
        history = torch.stack([windows[row, :4] if histories[row] is None else histories[row] for row in range(4)])
        controls = [control(case, position) for case, position in zip(selected, positions)]
        intents = torch.cat([state.from_controls(history[row:row + 1], value, 8)
            for row, (state, value) in enumerate(zip(states, controls))])
        scenes = [case['scene'] for case in selected]
        schedule.update(json.dumps([indices, positions, [v.cpu().tolist() for v in controls]]).encode())
        with torch.no_grad():
            reference = teacher(history, intents, scenes)['world'].clone()
            reference_local = teacher.last_local[:, :32].detach().clone()
            reference_joints = rep.inverse(reference[:, 4:12], False, return_all=False,
                                           joint_positions_from='global_rot_data')['posed_joints']
            last_skin, _ = skin_vertices(rep, geometry, history[:, -1:])
            if args.phase_support:
                reference_skin, _ = skin_vertices(rep, geometry, reference[:, 4:12])
                reference_soles = torch.cat((last_skin, reference_skin), 1).index_select(2, sole_indices)
                support_mask = teacher_phase_support_proxy(reference_soles,
                    torch.cat((history[:, -1:, :3], reference[:, 4:12, :3]), 1), intents[:, 2:4], foot_sizes)[:, 1:]
        if step == 0:
            with torch.no_grad():
                plain = model(history, intents, scenes)['world'].clone()
                changed = windows.clone()
                changed[:, 4:, :3] += 3.
                altered = model(history, intents, scenes, source=changed)['world']
                torch.testing.assert_close(plain, altered, atol=0, rtol=0)
                torch.testing.assert_close(plain, reference, atol=0, rtol=0)
            with torch.inference_mode():
                runtime = model(history, intents, scenes)['world']
            torch.testing.assert_close(plain, runtime, atol=3e-5, rtol=1e-5)
            parity = dict(recording_difference=recording_difference, future_target_difference=float((plain - altered).abs().max()),
                          runtime_difference=float((plain - runtime).abs().max()), initial_teacher_difference=float((plain - reference).abs().max()))
        result = model(history, intents, scenes)
        local = model.last_local[:, :32]
        actual = result['world']
        prefix_loss = decoded_factual_history_loss(model.inference, local, history, result['observation'],
            root_coefficient=contract['factual_root_coefficient'])
        if args.separate_factual_root and step < 8:
            prefix_gradient = torch.autograd.grad(prefix_loss, local, retain_graph=True)[0]
            root_mask = torch.zeros_like(prefix_gradient, dtype=torch.bool)
            root_mask[:, :3, model.inference.local_motion_rep.indices['root']] = True
            root_mask[:, 3, model.inference.local_motion_rep.indices['global_root_y']] = True
            allowed = root_mask.clone()
            for feature in ('ric_data', 'global_rot_data'):
                allowed[:, :4, model.inference.local_motion_rep.indices[feature]] = True
            if prefix_gradient[~allowed].abs().max() != 0 or not torch.isfinite(prefix_gradient).all():
                raise ValueError('Factual prefix supervision escaped observed channels')
            if prefix_gradient[root_mask].norm() == 0:
                raise ValueError('Factual Root normalization has no active gradient')
            parity['factual_prefix_future_gradient_max'] = float(prefix_gradient[:, 4:].abs().max())
            parity['factual_root_gradient_norm'] = float(prefix_gradient[root_mask].norm())
        preservation = F.smooth_l1_loss(local[:, 4:12, preserved_indices], reference_local[:, 4:12, preserved_indices])
        preservation = preservation * len(preserved_indices) / local.shape[-1]
        root_preservation = F.smooth_l1_loss(local[:, 4:12, root_indices], reference_local[:, 4:12, root_indices])
        if args.separate_root_preservation:
            rotation_preservation = F.smooth_l1_loss(local[:, 4:12, local_rotation_indices], reference_local[:, 4:12, local_rotation_indices])
            preservation = rotation_preservation * len(local_rotation_indices) / local.shape[-1]
            preservation = preservation + contract['root_preservation_coefficient'] * root_preservation
        direction_loss = preservation.new_zeros(())
        if args.direction_only_preservation:
            rotation_and_heading = local_rotation_indices + angular_indices
            preservation = F.smooth_l1_loss(local[:, 4:12, rotation_and_heading], reference_local[:, 4:12, rotation_and_heading])
            preservation = preservation * len(rotation_and_heading) / local.shape[-1]
            raw_local = model.inference.local_motion_rep.unnormalize(local)
            if model.inference.global_motion_rep.removed_heading or len(velocity_indices) != 2:
                raise ValueError('Direction objective requires current native world-XZ velocities')
            direction_loss = direction_without_speed_loss(raw_local[:, 4:12, velocity_indices], intents[:, 2:4],
                model.inference.local_motion_rep.stats.std[velocity_indices].to(local))
            preservation = preservation + float(model.inference._root_model.args['local_root_loss_coeff']) * direction_loss
        full_skin, joints = skin_vertices(rep, geometry, actual[:, 4:12])
        relative = joints - actual[:, 4:12, None, :3]
        teacher_relative = reference_joints - reference[:, 4:12, None, :3]
        relative_fk = (relative - teacher_relative).norm(dim=-1).mean()
        losses, surface = [], []
        for row, index in enumerate(indices):
            value, diagnostic = skin_tail_penetration(full_skin[row], last_skin[row], scenes[row], fields[index])
            losses.append(value)
            surface.append(diagnostic)
        collision = torch.stack(losses).mean()
        support_loss, support_samples = (collision.new_zeros(()), 0)
        if args.phase_support:
            support_loss, support_samples = teacher_support_gap(full_skin.index_select(2, sole_indices), support_mask, scenes)
        if step < 8:
            gradient = torch.autograd.grad(collision, actual, retain_graph=True, allow_unused=True)[0]
            if gradient is None or not torch.isfinite(gradient).all():
                raise ValueError('No finite penetration gradient through actual skin to actor output')
            contact_gradient_seen |= bool(gradient.abs().sum() > 0)
            parity['collision_output_gradient_norm_max'] = max(parity.get('collision_output_gradient_norm_max', 0.), float(gradient.norm()))
            parity['collision_past_or_unpublished_gradient_max'] = max(parity.get('collision_past_or_unpublished_gradient_max', 0.),
                float(torch.cat((gradient[:, :4], gradient[:, 12:]), 1).abs().max()))
            if parity['collision_past_or_unpublished_gradient_max'] != 0:
                raise ValueError('Collision gradient escaped the published8 future frames')
        with torch.no_grad():
            source_intents = torch.cat([SoftCommandState().from_controls(windows[row:row + 1, :4], c, 8)
                                       for row, c in enumerate(controls)])
            source_result = model(windows[:, :4], source_intents, scenes, source=windows)
        source_loss = native_codec_reconstruction(model, source_result)['loss']
        loss = source_loss + prefix_loss + preservation + fk_coefficient * relative_fk
        if args.arm == 'contact':
            loss = loss + collision_coefficient * collision
        if args.phase_support:
            loss = loss + contract['losses']['weak_support_gap_m'] * support_loss
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite contact-preservation loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or float(norm) == 0:
            raise ValueError('No finite decoder gradient')
        if any(p.grad is not None for name, p in model.named_parameters() if not name.startswith('codec.decoder.')):
            raise ValueError('Gradient entered a frozen component')
        optimizer.step()
        played = reencode_played_history(actual[:, 8:12].detach(), rep)
        for row, (index, position) in enumerate(zip(indices, positions)):
            histories[row] = played[row]
            exposure[cases[index]['label'] + ':' + str(position)] += 1
            positions[row] += 1
        row = dict(step=step + 1, source_loss=float(source_loss.detach()), prefix_loss=float(prefix_loss.detach()),
            factual_past_xz_error_mean_m=float((actual[:, 3, [0, 2]] - history[:, 3, [0, 2]]).detach().norm(dim=-1).mean()),
            factual_past_xz_error_max_m=float((actual[:, 3, [0, 2]] - history[:, 3, [0, 2]]).detach().norm(dim=-1).max()),
            preservation=float(preservation.detach()), relative_fk_m=float(relative_fk.detach()),
            root_preservation=float(root_preservation.detach()),
            direction_loss=float(direction_loss.detach()),
            collision_tail_m2=float(collision.detach()), peak_m=max(v['peak_m'] for v in surface),
            weak_support_gap_m=float(support_loss.detach()), support_vertex_samples=support_samples,
            total_loss=float(loss.detach()), gradient_norm=float(norm), elapsed_seconds=time.perf_counter() - started)
        logs.append(row)
        if step == 7 or step + 1 == args.updates:
            if step >= 7 and not contact_gradient_seen:
                raise ValueError('Eight-step smoke never reached an active contact gradient')
            if frozen_hashes(model) != frozen or state_digest(teacher) != teacher_hash:
                raise ValueError('Frozen components or teacher changed')
            write_json(args.output / 'preflight.json', dict(**parity, checked_updates=step + 1,
                frozen_hashes_match=True, teacher_unchanged=True, gradients_decoder_only=True,
                active_contact_gradient=contact_gradient_seen, finite=True))
        if step == 0 or (step + 1) % 10 == 0:
            write_json(args.output / 'progress.json', dict(**row, schedule_sha256=schedule.hexdigest(),
                actor_exposures=(step + 1) * 4, source_examples=(step + 1) * 4, exposure=dict(exposure)))
            print(json.dumps(row), flush=True)
        if step == 7 or (step + 1) % 50 == 0 or step + 1 == args.updates:
            pending = args.output / 'weights.pending.pt'
            torch.save(dict(schema=SDF_VERSION, model_state=model.state_dict(), optimizer=optimizer.state_dict(),
                step=step + 1, contract=dict(saved['contract'], contact_preservation_pilot=contract),
                coefficients=saved['coefficients'], positions=positions, indices=indices,
                histories=[value.cpu() for value in histories], exposure=dict(exposure),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                command_states=[vars(state) for state in states], schedule_sha256=schedule.hexdigest()), pending)
            pending.replace(args.output / 'weights.pt')
    model.requires_grad_(False)
    if args.reference_trial and (prior_trial['schedule_sha256'] != schedule.hexdigest() or prior_trial['exposure'] != dict(exposure)):
        raise ValueError('Matched collision-only source/command exposure differs')
    write_json(args.output / 'training.json', dict(logs=logs, schedule_sha256=schedule.hexdigest(), contract=contract))
    write_json(state_path, dict(status='review', quality_approved=False))
    final = exact_reviews(model, cases, geometry, args.output / 'final', reference=args.output / 'baseline')
    varied = evaluate(model, bundle, cases, geometry, args.output / 'varied')
    write_json(args.output / 'complete.json', dict(arm=args.arm, actual_updates=args.updates,
        actor_exposures=args.updates * 4, source_examples=args.updates * 4, teacher_calls=args.updates * 4,
        schedule_sha256=schedule.hexdigest(), exposure=dict(exposure), baseline=baseline,
        final=final, varied=varied, quality_approved=False))
    write_json(state_path, dict(status='complete', quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('control', 'contact'), required=True)
    for name in ('checkpoint', 'cases', 'accepted-recording', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--phase-support', action='store_true')
    parser.add_argument('--reference-trial', type=Path)
    parser.add_argument('--separate-root-preservation', action='store_true')
    parser.add_argument('--direction-only-preservation', action='store_true')
    parser.add_argument('--separate-factual-root', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'state.json', dict(status='failed', error=str(error), quality_approved=False))
        raise
