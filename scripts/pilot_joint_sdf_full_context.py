"""Matched source-history / full-feedback pilot on factual complete stair passages."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController, SDF_VERSION
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.joint_sdf_training import ExactPenetrationBatch
from motionbricks.data.decoded_history_root_objective import DECODED_HISTORY_ROOT_VERSION, decoded_history_root_loss
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_control import SoftCommandState, demonstration_controls
from motionbricks.data.terrain_distance_volume import swept_surface_points
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.evaluate_joint_sdf import save_motion
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.prepare_joint_sdf_context_pilot import recorded_direction
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def load_cases(path, dataset):
    preparation = json.loads((path.parent / 'complete.json').read_text())
    if sha256(path) != preparation['cases_sha256']:
        raise ValueError('Prepared cases changed')
    cases = torch.load(path, map_location='cpu', weights_only=False)
    # The provenance reader needs only dataset_dir, not a resampled dataset.
    from types import SimpleNamespace
    data = SimpleNamespace(base=SimpleNamespace(dataset_dir=dataset))
    for case in cases:
        if case['name'] != 'stairs':
            continue
        direction = recorded_direction(data, case['row'], case['row']['frames_30hz'])
        start = case['start_frame']
        case['recorded_direction'] = direction[start:start + len(case['world'])].clone()
        if len(case['world']) != 33 + (case['training_segments'] - 1) * 8:
            raise ValueError('Every feedback position needs its own real 33-frame window')
        final_played = start + (case['training_segments'] - 1) * 8 + 11
        if final_played < case['evaluation_goal_source_frame']:
            raise ValueError('Prepared feedback schedule does not cover its goal')
    if [case['name'] for case in cases] != ['stairs', 'stairs', 'stairs', 'idle', 'walk']:
        raise ValueError('This bounded pilot requires exactly three stairs and two flat references')
    return cases


def command(case, offset, *, held=False, scale=1., device='cuda:0'):
    world = case['world']
    if case['name'] != 'stairs':
        offset = min(offset, len(world) - 33)
        return demonstration_controls(world[None, offset:offset + 33].to(device), 32,
                                      category=case['name'], reference=world[None].to(device))
    index = 3 if held else min(offset + 3, len(case['recorded_direction']) - 1)
    direction = case['recorded_direction'][index].to(device)
    initial = case['recorded_direction'][3].to(device)
    delta = torch.atan2(direction[1], direction[0]) - torch.atan2(initial[1], initial[0])
    # Native yaw rotates world XZ clockwise; preserve the initial body/travel
    # offset using only the recorded command, never a future source pose.
    yaw = world[3, 3:5].to(device)
    yaw = torch.atan2(yaw[1], yaw[0]) - delta
    speed = torch.as_tensor(case['command_velocity'], device=device).norm() * scale
    return torch.cat((direction * speed, yaw.cos()[None], yaw.sin()[None]))[None]


@torch.no_grad()
def exact_skin_metrics(geometry, world, scene):
    rep = geometry.rep
    field = HeightfieldPrismSDF(scene)
    depths = []
    previous = None
    unknown = uncertain = total = 0
    for start in range(0, world.shape[1], 16):
        part = world[:, start:start + 16]
        pose = rep.inverse(part, False, return_all=True, joint_positions_from='global_rot_data')
        points = geometry.skin.skin(pose['local_joint_rots'][0],
                                   part[0, :, :3, None].transpose(-1, -2).expand(-1, 30, -1), rot_is_global=False)
        if previous is not None:
            points = torch.cat((previous, points), 0)
        previous = points[-1:]
        points = swept_surface_points(points, 8)
        if start + 16 < world.shape[1]:
            points = points[:-1]
        gap, known = heightfield_gap(points, scene)
        inside = known & (gap < 0)
        depth = torch.zeros_like(gap)
        if inside.any():
            distance, _, certified = field.query(points[inside])
            depth[inside] = torch.where(certified, (-distance).clamp_min(0), -gap[inside])
            uncertain += int((~certified).sum())
        depths.append(depth[known].cpu().numpy())
        unknown += int((~known).sum())
        total += known.numel()
    values = np.concatenate(depths)
    return {'body_prism_p99_m': float(np.quantile(values, .99)) if len(values) else None,
            'body_prism_max_m': float(values.max()) if len(values) else None,
            'unknown_body_fraction': unknown / max(1, total), 'uncertain_inside_samples': uncertain,
            'vertices_per_frame': len(geometry.skin.bind_vertices), 'swept_substeps': 8}


@torch.no_grad()
def evaluate(model, bundle, cases, geometry, output):
    output.mkdir(parents=True, exist_ok=False)
    rep = model.inference.global_motion_rep
    rows = []
    for case in cases:
        source = case['world'][None].cuda()
        scene = case['scene']
        label = case['label']
        save_motion(output / (label + '-reference-source.npz'), source, scene, rep,
                    {'role': 'unchanged_reference', 'quality_approved': False})
        variants = ('recorded_direction', 'held_forward', 'slow', 'fast', 'idle') if case['name'] == 'stairs' else ('flat',)
        for variant in variants:
            scale = {'slow': .8, 'fast': 1.2}.get(variant, 1.)
            held = variant not in ('recorded_direction', 'flat')
            history = source[:, :4].clone()
            control = command(case, 0, held=held, scale=scale)
            if variant == 'idle':
                heading = torch.atan2(control[:, 3], control[:, 2])
                history = rep.change_first_heading(bundle['initial_history'].to(source), heading, False, False)
                history[..., [0, 2]] += source[:, 3:4, [0, 2]] - history[:, 3:4, [0, 2]]
                gap, known = heightfield_gap(source[0, 3:4, :3], scene)
                if not known.all():
                    raise ValueError('Idle spawn support is unknown')
                history = shift_global_motion(history, rep, source[0, 3, 1] - gap[0])
            initial = history.clone()
            state = SoftCommandState()
            if case['name'] == 'stairs':
                goal_index = case['evaluation_goal_source_frame'] - case['start_frame']
                goal = source[0, goal_index, :3]
                axis = (goal[[0, 2]] - initial[0, -1, [0, 2]])
                axis = axis / axis.norm()
                needed = float((goal[[0, 2]] - initial[0, -1, [0, 2]]).dot(axis))
                maximum = math.ceil((needed / float(control[0, :2].norm()) + 2.) * 30 / 8)
            else:
                maximum = 13
            parts = []
            failure = None
            reached = False
            for block in range(maximum):
                control = command(case, block * 8, held=held, scale=scale)
                task = state.from_controls(history, control, 8)
                try:
                    result = model(history, task, [scene])
                except ValueError as error:
                    failure = str(error)
                    break
                parts.append(result['world'][:, 4:12])
                history = reencode_played_history(result['world'][:, 8:12], rep)
                if case['name'] == 'stairs':
                    reached = float((history[0, -1, [0, 2]] - goal[[0, 2]]).dot(axis)) >= 0
                    if reached:
                        break
            row = {'name': label, 'variant': variant, 'frames': sum(p.shape[1] for p in parts),
                   'frame_limit': maximum * 8, 'failure': failure, 'goal_reached': reached,
                   'quality_approved': False, 'passed': False, 'postprocessing': None,
                   'command': 'recorded unit direction times weak approach speed' if not held else 'held direction and weak approach speed',
                   'evaluation_end': 'first published block reaching the fixed goal, or bounded timeout; no truncation or padding'}
            if parts:
                world = torch.cat(parts, 1)
                _, points = geometry.geometry(world)
                gap, known = heightfield_gap(points[0, :, :geometry.sole_count], scene)
                row.update(exact_skin_metrics(geometry, world, scene))
                row['sole_vertical_p99_m'] = float(torch.quantile((-gap[known]).clamp_min(0), .99)) if known.any() else None
                row['sole_vertical_max_m'] = float((-gap[known]).clamp_min(0).max()) if known.any() else None
                delta = world[0, -1, :3] - initial[0, -1, :3]
                if case['name'] == 'stairs':
                    row.update(progress_m=float(delta[[0, 2]].dot(axis)), required_progress_m=needed,
                               cross_track_m=float(delta[[0, 2]].dot(torch.stack((axis[1], -axis[0]))).abs()),
                               root_height_error_m=abs(float(world[0, -1, 1] - goal[1])))
                    traversed = reached and row['cross_track_m'] <= .35 and row['root_height_error_m'] <= .2
                else:
                    row['root_drift_m'] = float(delta[[0, 2]].norm())
                    expected = source[0, 107, [0, 2]] - source[0, 3, [0, 2]]
                    traversed = row['root_drift_m'] <= .1 if case['name'] == 'idle' else row['root_drift_m'] >= .8 * float(expected.norm())
                row['legacy_geometry_screen'] = bool(known.all() and row['unknown_body_fraction'] == 0
                    and row['sole_vertical_p99_m'] <= .04 and row['body_prism_p99_m'] <= .02)
                row['passed'] = bool(failure is None and traversed and row['legacy_geometry_screen']
                                     and row['body_prism_max_m'] <= .02 and row['uncertain_inside_samples'] == 0)
                # Goal-bounded trials can have different real durations across
                # arms. Distinct lengths remain distinct viewer recordings.
                save_motion(output / f'{label}-{variant}-{world.shape[1]}f.npz', world, scene, rep, row)
            rows.append(row)
            write_json(output / 'metrics.json', {'rows': rows, 'quality_approved': False})
            print(json.dumps(row), flush=True)
    return rows


def main(args):
    if not 1 <= args.updates <= 300:
        raise ValueError('This pilot is limited to 300 actual updates')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    np.random.seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    source_path = Path(saved['contract']['source'])
    if sha256(source_path) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source bundle identity changed')
    infer, bundle = load_terrain_bundle(source_path, 'cuda:0')
    model = JointSDFController(infer, bundle)
    model.load_state_dict(saved['model_state'], strict=True)
    model.freeze_base()
    model.eval()
    rep = infer.global_motion_rep
    cases = load_cases(args.cases, Path(saved['contract']['dataset']))
    frozen = {name: state_digest(getattr(model, name)) for name in
              ('root', 'pose', 'codec', 'control', 'root_far', 'pose_far', 'decoder_near', 'decoder_attention', 'decoder_far')}
    geometry = JointGeometryObjective(rep, 'cuda:0')
    prepared = []
    fields = []
    with torch.no_grad():
        for case in cases:
            joints, points = geometry.geometry(case['world'][None].cuda())
            prepared.append((joints[0], points[0]))
            fields.append(HeightfieldPrismSDF(case['scene']))
    contract = dict(arm=args.arm, updates=args.updates, batch=4, examples=args.updates * 4,
                    initial_checkpoint=str(args.checkpoint), initial_sha256=sha256(args.checkpoint),
                    cases_sha256=sha256(args.cases), history_policy=args.arm,
                    controls='current recorded direction times fixed weak approach speed; initial body/travel offset retained; flat source weak controls',
                    source_windows='independent deterministic full-passage cursors, three stairs plus alternating idle/walk',
                    trainable='SDF adapters only', frozen_hashes=frozen, learning_rate=5e-5,
                    losses=saved['coefficients'], postprocessing=None,
                    inference='archived default four denoising passes and one model-generated route refinement',
                    decoded_history_root=(DECODED_HISTORY_ROOT_VERSION if args.decoded_history_root else None),
                    decoded_history_root_coefficient=(float(infer._root_model.args['local_root_loss_coeff']) if args.decoded_history_root else 0.),
                    quality_approved=False)
    write_json(args.output / 'contract.json', contract)
    baseline = evaluate(model, bundle, cases, geometry, args.output / 'baseline')
    if args.reference_control:
        from scripts.summarize_joint_sdf_context_pair import assert_same
        reference = json.loads(args.reference_control.read_text())
        assert_same(baseline, reference['baseline'])
        if reference['actual_updates'] != args.updates or reference['primitive_examples'] != args.updates * 4:
            raise ValueError('Archived control budget differs')
        write_json(args.output / 'baseline_parity.json', {'passed': True, 'reference': str(args.reference_control)})
    torch.manual_seed(2333)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    positions = [0] * 4
    indices = [0, 1, 2, 3]
    histories = [None] * 4
    states = [SoftCommandState() for _ in range(4)]
    exposure = Counter()
    schedule = hashlib.sha256()
    started = time.perf_counter()
    for step in range(args.updates):
        for row, index in enumerate(indices):
            if positions[row] >= cases[index]['training_segments']:
                positions[row] = 0
                histories[row] = None
                states[row] = SoftCommandState()
                if row == 3:
                    indices[row] = 4 if index == 3 else 3
        selected = [cases[index] for index in indices]
        windows = torch.stack([case['world'][position * 8:position * 8 + 33]
                               for case, position in zip(selected, positions)]).cuda()
        if windows.shape[1] != 33:
            raise ValueError('Incomplete factual window')
        history = torch.stack([windows[row, :4] if args.arm == 'source' or histories[row] is None else histories[row]
                               for row in range(4)])
        controls = [command(case, position * 8) for case, position in zip(selected, positions)]
        intent = torch.cat([state.from_controls(history[row:row + 1], control, 8)
                            for row, (state, control) in enumerate(zip(states, controls))])
        scenes = [case['scene'] for case in selected]
        schedule.update(json.dumps([indices, positions, [control.cpu().tolist() for control in controls]]).encode())
        if step == 0:
            with torch.no_grad():
                plain = model(history, intent, scenes)['world'].clone()
                observed_target = model(history, intent, scenes, source=windows)['world'].clone()
                changed = windows.clone()
                changed[:, 4:, :3] += 3.
                altered = model(history, intent, scenes, source=changed)['world']
                torch.testing.assert_close(plain, observed_target, atol=0, rtol=0)
                torch.testing.assert_close(plain, altered, atol=0, rtol=0)
            with torch.inference_mode():
                runtime = model(history, intent, scenes)['world']
            torch.testing.assert_close(plain, runtime, atol=3e-5, rtol=1e-5)
            np.savez_compressed(args.output / 'initial-forward.npz', world=plain.cpu().numpy())
        result = model(history, intent, scenes, source=windows, straight_through=True)
        if step == 0:
            torch.testing.assert_close(plain, result['world'], atol=3e-5, rtol=1e-5)
            parity = {'future_target_max_difference': float((plain - altered).abs().max()),
                      'runtime_max_difference': float((plain - runtime).abs().max()),
                      'training_forward_max_difference': float((plain - result['world']).abs().max())}
        exact = ExactPenetrationBatch(scenes, history.device, [fields[index] for index in indices])
        target = {'joints': torch.stack([prepared[index][0][position * 8:position * 8 + 33]
                                        for index, position in zip(indices, positions)]),
                  'points': torch.stack([prepared[index][1][position * 8:position * 8 + 33]
                                        for index, position in zip(indices, positions)]),
                  'heights': exact.heights, 'volumes': exact}
        native = model.native_losses(result, windows)
        physical = geometry.prepared_losses(result['world'], history, target, offset=0, diagnostics=False)
        loss = native['native_total'] + sum(saved['coefficients'][key] * physical[key] for key in saved['coefficients'])
        history_root = loss.new_zeros(())
        if args.decoded_history_root:
            history_root = decoded_history_root_loss(infer, result['local_prediction'], history, result['observation'])
            loss = loss + contract['decoded_history_root_coefficient'] * history_root
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite pilot loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or float(norm) == 0:
            raise ValueError('No finite SDF gradient')
        if any(p.grad is not None for name in frozen for p in getattr(model, name).parameters()):
            raise ValueError('Gradient entered a frozen module')
        optimizer.step()
        with torch.no_grad():
            played = reencode_played_history(result['world'][:, 8:12].detach(), rep)
        for row, (index, position) in enumerate(zip(indices, positions)):
            histories[row] = played[row]
            exposure[cases[index]['label'] + ':position:' + str(position)] += 1
            positions[row] += 1
        if step == 7 or step + 1 == args.updates:
            current = {name: state_digest(getattr(model, name)) for name in frozen}
            if frozen != current:
                raise ValueError('Frozen base changed')
            write_json(args.output / 'preflight.json', dict(parity, passed=True, checked_updates=step + 1,
                       checked_examples=(step + 1) * 4, frozen_hashes_match=True, finite_sdf_gradient=True))
        if step % 20 == 0 or step + 1 == args.updates:
            write_json(args.output / 'progress.json', {'arm': args.arm, 'step': step + 1,
                       'examples': (step + 1) * 4, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                       'decoded_history_root_loss': float(history_root.detach()),
                       'elapsed_seconds': time.perf_counter() - started, 'exposure': dict(exposure),
                       'schedule_sha256': schedule.hexdigest(), 'quality_approved': False})
            print(json.dumps({'arm': args.arm, 'step': step + 1, 'loss': float(loss.detach()),
                              'elapsed_seconds': time.perf_counter() - started}), flush=True)
        if step == 7 or (step + 1) % 50 == 0 or step + 1 == args.updates:
            pending = args.output / 'weights.pending.pt'
            torch.save({'schema': SDF_VERSION, 'model_state': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'step': step + 1, 'contract': dict(saved['contract'], full_context_pilot=contract),
                        'coefficients': saved['coefficients'], 'positions': positions, 'indices': indices,
                        'histories': [value.cpu() for value in histories], 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state(), 'command_states': [vars(state) for state in states],
                        'exposure': dict(exposure), 'schedule_sha256': schedule.hexdigest()}, pending)
            pending.replace(args.output / 'weights.pt')
    model.requires_grad_(False)
    if args.reference_control:
        if reference['schedule_sha256'] != schedule.hexdigest() or reference['exposure'] != dict(exposure):
            raise ValueError('Archived control command/window/exposure schedule differs')
    rows = evaluate(model, bundle, cases, geometry, args.output / 'final')
    write_json(args.output / 'complete.json', {'arm': args.arm, 'actual_updates': args.updates,
               'primitive_examples': args.updates * 4, 'schedule_sha256': schedule.hexdigest(),
               'exposure': dict(exposure), 'baseline': baseline, 'final': rows, 'quality_approved': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('source', 'feedback'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--decoded-history-root', action='store_true')
    parser.add_argument('--reference-control', type=Path)
    main(parser.parse_args())
