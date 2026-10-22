"""Offline controlled-input comparison of the parent and fixed-64 prefix pilots.

Run as ``python -m scripts.evaluate_reactive_prefix_control``. All models execute
four native frames per call, with genuine emitted FK history and no correction.
The 30 Hz clock defines motion timestamps, not measured rendering performance.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


SCHEMA = 'motionbricks.reactive_prefix_control_evaluation.v2'
PILOT_SCHEMA = 'motionbricks.reactive_prefix_pilot.v1'
ARMS = ('source_history', 'generated_history')
FPS, STRIDE = 30, 4
SPEED, TURN_RATE = 1.2, 1.8
START_SPEED, STOP_SPEED, SETTLE_FRAMES = .3, .1, 6
CONTROL_SCREEN = dict(idle16_endpoint_m=.05, stop_confirmed_seconds=.5,
                      settled_speed_ratio=[.8, 1.2], moving_settle_frames=FPS,
                      orthogonal_max_m=.15, fixed_heading_p95_rad=math.radians(15))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def validate_endpoint(saved, parent_hash, bundle_hash):
    contract = saved.get('contract', {})
    if (saved.get('schema') != PILOT_SCHEMA or contract.get('schema') != PILOT_SCHEMA
            or type(saved.get('updates')) is not int or saved['updates'] != 64
            or type(saved.get('exposures')) is not int or saved['exposures'] != 512
            or contract.get('parent_sha256') != parent_hash
            or contract.get('bundle_sha256') != bundle_hash
            or contract.get('arm') not in ARMS
            or contract.get('eligible_for_automatic_export') is not False):
        raise ValueError('Need a fixed-64/512 pilot endpoint with matching parent and bundle')
    return contract


def select_idle(selection):
    """Select before evaluating, using only the supplied preflight validation set."""
    rows = [row for row in selection['validation'] if row['category'] == 'idle'
            and row['record']['domain'] == 'bones'
            and row['record']['split'] == 'validation' and row['record']['quarter'] == 0]
    if not rows:
        raise ValueError('Preflight selection has no BONES validation q0 idle seed')
    return min(rows, key=lambda row: (row['cache_index'], row['row_index']))


def plane_height(scene):
    heights = torch.as_tensor(scene['heightfield_y_m'])
    if (heights.ndim != 2 or min(heights.shape) < 2 or not torch.isfinite(heights).all()
            or not torch.equal(heights, heights.reshape(-1)[0].expand_as(heights))):
        raise ValueError('Selected factual scene must have finite, exactly constant heights')
    if 'heightfield_valid' in scene and not torch.as_tensor(scene['heightfield_valid']).bool().all():
        raise ValueError('Selected factual plane contains unknown cells')
    if not math.isfinite(float(scene['dx_m'])) or float(scene['dx_m']) <= 0:
        raise ValueError('Invalid factual plane cell size')
    return float(heights.reshape(-1)[0])


def control_arena(seed, height):
    """Explicit authored64m plane at the factual seed's existing ground height."""
    center = seed[0, -1, :3].detach().cpu()
    return dict(heightfield_y_m=torch.full((321, 321), float(height)),
        min_point_world_xyz_m=torch.tensor([float(center[0])-32., float(height), float(center[2])-32.]),
        dx_m=torch.tensor(.2))


def controlled_trials(seed):
    if seed.shape != (1, 4, 366) or not torch.isfinite(seed).all():
        raise ValueError('Need one factual finite [1,4,366] seed')
    heading = seed[0, -1, 3:5].double().cpu().numpy()
    if not np.isclose(np.linalg.norm(heading), 1., atol=1e-3, rtol=0):
        raise ValueError('Seed heading is not a unit cos/sin pair')
    yaw = math.atan2(heading[1], heading[0])
    forward = np.array([math.sin(yaw), math.cos(yaw)])
    right = np.array([math.cos(yaw), -math.sin(yaw)])
    definitions = (
        ('idle16', [('idle', 16, np.zeros(2), False)]),
        ('start_stop', [('idle', 2, np.zeros(2), False), ('forward', 6, forward, False),
                        ('stop', 8, np.zeros(2), False)]),
        ('independent_heading', [('forward', 3, forward, False), ('left', 3, -right, False),
                                 ('reverse', 3, -forward, False), ('right', 3, right, False),
                                 ('turn_in_place', 3, np.zeros(2), True),
                                 ('stop', 3, np.zeros(2), False)]),
    )
    trials = []
    for name, definitions_for_trial in definitions:
        phases, commands, current_yaw, cursor = [], [], yaw, 0
        for label, seconds, axis, turning in definitions_for_trial:
            count = seconds * FPS
            phase_commands = np.zeros((count, 4), dtype=np.float32)
            phase_commands[:, :2] = SPEED * axis
            angles = current_yaw + (np.arange(count) / FPS * TURN_RATE if turning else np.zeros(count))
            phase_commands[:, 2:] = np.stack((np.cos(angles), np.sin(angles)), -1)
            phases.append(dict(name=label, start_frame=cursor, end_frame=cursor + count,
                               velocity_world_xz_mps=(SPEED * axis).tolist(),
                               heading_rate_radps=TURN_RATE if turning else 0.))
            commands.append(phase_commands)
            cursor += count
            if turning:
                current_yaw += seconds * TURN_RATE
        requested = torch.from_numpy(np.concatenate(commands))
        # Inputs are sampled on the four-frame planning clock. Preserve nominal
        # phase boundaries and record the held inputs, including the 2-frame lag.
        issued = requested[(torch.arange(cursor) // STRIDE) * STRIDE].clone()
        trials.append(dict(name=name, frames=cursor, phases=phases, requested_controls=requested,
                           issued_controls=issued, initial_heading_rad=yaw))
    return trials


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return dict(count=0, mean=None, p50=None, p95=None, p99=None, maximum=None)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite measured metric')
    return dict(count=int(values.size), mean=float(values.mean()), p50=float(np.quantile(values, .5)),
                p95=float(np.quantile(values, .95)), p99=float(np.quantile(values, .99)),
                maximum=float(values.max()))


def sustained_crossing(values, threshold, *, below=False):
    mask = np.asarray(values) <= threshold if below else np.asarray(values) >= threshold
    for index in range(max(0, len(mask) - SETTLE_FRAMES + 1)):
        if mask[index:index + SETTLE_FRAMES].all():
            # Velocity interval i ends at (i+1)/30 relative to the input change.
            return dict(onset_seconds=(index + 1) / FPS,
                        confirmed_seconds=(index + SETTLE_FRAMES) / FPS)
    return dict(onset_seconds=None, confirmed_seconds=None)


def motion_metrics(world, joints, trial, ground_y, *, skeleton):
    """All derivatives include the genuine last-seed -> first-emitted interval."""
    from motionbricks.motionlib.core.motion_reps.tools.heading import calc_heading_from_joints_pos

    world, joints = world.detach().double().cpu(), joints.detach().double().cpu()
    if (world.ndim != 3 or world.shape[0] != 1 or world.shape[2] != 366
            or joints.shape != (*world.shape[:2], 30, 3)):
        raise ValueError('Metrics need native [1,T,366] and official FK [1,T,30,3]')
    count = world.shape[1] - 4
    if joints.shape[:2] != world.shape[:2] or count < 0 or count > trial['frames']:
        raise ValueError('Motion/FK clocks disagree')
    if not torch.isfinite(world).all() or not torch.isfinite(joints).all():
        raise ValueError('Nonfinite raw world/FK')
    roots = world[0, 3:, :3].numpy()
    velocity = np.diff(roots[:, [0, 2]], axis=0) * FPS
    relative = (joints - world[..., :3].unsqueeze(-2))[0, 3:]
    step = torch.diff(relative, dim=0).norm(dim=-1).numpy()
    root_step = np.linalg.norm(np.diff(roots[:, [0, 2]], axis=0), axis=-1)
    # Match native history re-encoding: hips_pos on the displayed global_rot_data
    # FK skeleton. Raw features 3:5 are a redundant prediction, not the display.
    fk_heading = calc_heading_from_joints_pos(joints, skeleton)
    if not torch.isfinite(fk_heading).all():
        raise ValueError('Displayed FK has an undefined native hips_pos heading')
    yaw = np.unwrap(fk_heading[0, 3:].numpy())
    raw_yaw = np.arctan2(world[0, 3:, 4].numpy(), world[0, 3:, 3].numpy())
    rows = []
    for phase in trial['phases']:
        start, end = phase['start_frame'], min(phase['end_frame'], count)
        if end <= start:
            continue
        target = np.asarray(phase['velocity_world_xz_mps'])
        speed = float(np.linalg.norm(target))
        axis = target / speed if speed else np.array([math.sin(trial['initial_heading_rad']),
                                                     math.cos(trial['initial_heading_rad'])])
        lateral = np.array([axis[1], -axis[0]])
        part = velocity[start:end]
        displacement = roots[start + 1:end + 1, [0, 2]] - roots[start, [0, 2]]
        heights = roots[start + 1:end + 1, 1]
        desired = trial['requested_controls'][start:end, 2:].double().numpy()
        target_heading = np.arctan2(desired[:, 1], desired[:, 0])
        heading_error = yaw[start + 1:end + 1] - target_heading
        raw_error = raw_yaw[start + 1:end + 1] - target_heading
        disagreement = raw_yaw[start + 1:end + 1] - yaw[start + 1:end + 1]
        row = dict(name=phase['name'], start_frame=start, measured_frames=end - start,
                   requested_frames=phase['end_frame'] - start, complete=end == phase['end_frame'],
                   target_velocity_world_xz_mps=target.tolist(),
                   mean_velocity_world_xz_mps=part.mean(0).tolist(),
                   mean_signed_velocity_mps=float((part @ axis).mean()),
                   mean_speed_mps=float(np.linalg.norm(part, axis=-1).mean()),
                   mean_abs_orthogonal_velocity_mps=float(np.abs(part @ lateral).mean()),
                   orthogonal_error_m=distribution(np.abs(displacement @ lateral)),
                   velocity_error_mps=distribution(np.linalg.norm(part - target, axis=-1)),
                   root_height_m=distribution(heights), root_height_above_plane_m=distribution(heights - ground_y),
                   body_relative_joint_step_m=distribution(step[start:end].reshape(-1)),
                   planar_root_step_m=distribution(root_step[start:end]),
                   absolute_heading_error_rad=distribution(np.abs(np.arctan2(np.sin(heading_error),
                                                                             np.cos(heading_error)))),
                   raw_feature_heading_error_rad=distribution(np.abs(np.arctan2(np.sin(raw_error),
                                                                                 np.cos(raw_error)))),
                   raw_vs_fk_heading_disagreement_rad=distribution(np.abs(np.arctan2(np.sin(disagreement),
                                                                                     np.cos(disagreement)))),
                   heading_change_rad=float(yaw[end] - yaw[start]))
        if speed:
            crossing = sustained_crossing(part @ axis, START_SPEED)
            row['start_threshold'] = dict(speed_mps=START_SPEED, sustained_frames=SETTLE_FRAMES, **crossing)
            row['settled_mean_signed_velocity_mps'] = (float((part[FPS:] @ axis).mean())
                                                       if len(part) > FPS else None)
        else:
            distances = np.linalg.norm(displacement, axis=-1)
            row.update(idle_endpoint_displacement_m=float(distances[-1]),
                       idle_max_displacement_m=float(distances.max()),
                       idle_path_length_m=float(root_step[start:end].sum()))
            if phase['name'] == 'stop':
                row['stop_delay'] = dict(speed_mps=STOP_SPEED, sustained_frames=SETTLE_FRAMES,
                                        **sustained_crossing(np.linalg.norm(part, axis=-1), STOP_SPEED, below=True))
        rows.append(row)
    return dict(emitted_frames=count, simulated_seconds=count / FPS, phases=rows,
                heading_source='native hips_pos from displayed global_rot_data FK',
                body_relative_joint_step_m=distribution(step.reshape(-1)),
                planar_root_step_m=distribution(root_step),
                endpoint_displacement_m=float(np.linalg.norm(roots[-1, [0, 2]] - roots[0, [0, 2]])))


def control_screen(motion, trial):
    """Fixed research screen, independent of the unchanged full-skin gate."""
    phases = {phase['name']: phase for phase in motion['phases']}
    checks = []

    def check(phase, criterion, measured, passed):
        checks.append(dict(phase=phase, criterion=criterion, measured=measured, passed=bool(passed)))

    for desired in trial['phases']:
        name = desired['name']
        phase = phases.get(name)
        complete = phase is not None and phase['complete']
        check(name, 'complete_phase', phase['measured_frames'] if phase else 0, complete)
        if not complete:
            continue
        if trial['name'] == 'idle16':
            distance = phase['idle_endpoint_displacement_m']
            check(name, 'idle16_endpoint_le_0.05m', distance, distance <= CONTROL_SCREEN['idle16_endpoint_m'])
        if name == 'stop':
            delay = phase['stop_delay']['confirmed_seconds']
            check(name, 'stop_confirmed_le_0.5s_at_0.1mps_for_6frames', delay,
                  delay is not None and delay <= CONTROL_SCREEN['stop_confirmed_seconds'])
        if np.linalg.norm(desired['velocity_world_xz_mps']) > 0:
            velocity = phase['settled_mean_signed_velocity_mps']
            low, high = (ratio * SPEED for ratio in CONTROL_SCREEN['settled_speed_ratio'])
            check(name, 'signed_mean_0.96_to_1.44mps_after_1s', velocity,
                  velocity is not None and low <= velocity <= high)
            error = phase['orthogonal_error_m']['maximum']
            check(name, 'orthogonal_max_le_0.15m_from_phase_entry', error,
                  error <= CONTROL_SCREEN['orthogonal_max_m'])
        if desired['heading_rate_radps'] == 0:
            error = phase['absolute_heading_error_rad']['p95']
            check(name, 'fixed_heading_p95_le_15deg', error,
                  error <= CONTROL_SCREEN['fixed_heading_p95_rad'])
    return dict(label='research_control_screen', passed=all(check['passed'] for check in checks),
                thresholds=CONTROL_SCREEN, checks=checks,
                failures=[check for check in checks if not check['passed']])


def check_observations(controller, observer):
    masks = dict(coarse_height=controller.observation['valid'],
                 predicted_path_height=observer.height.valid,
                 predicted_path_volume=controller.volume_observation['valid'])
    counts = {name: int((~mask.bool()).sum()) for name, mask in masks.items()}
    if any(not mask.flatten(1).any(-1).all() for mask in masks.values()):
        raise ValueError('No known scene observations: ' + json.dumps(counts))
    return counts


@torch.no_grad()
def run_rollout(controller, rep, seed, scene, trial, *, reencode, synchronize, observe):
    """Return partial raw data even when generation/encoding/observation fails."""
    history = None
    parts, forecasts, histories, controls, timings, observations, failures = [], [], [], [], [], [], []
    stage, call_start = 'initial_history', None
    try:
        history = reencode(seed.clone(), rep)
        for offset in range(0, trial['frames'], STRIDE):
            stage = 'forward'
            control = trial['issued_controls'][offset:offset + 1].to(seed)
            histories.append(history.detach().cpu().clone())
            controls.append(control.cpu().clone())
            synchronize()
            call_start = time.perf_counter()
            result = controller(history, control, scene)
            synchronize()
            forward_ms = (time.perf_counter() - call_start) * 1000
            predicted = result['world'].detach()
            forecasts.append(predicted.cpu().clone())
            if predicted.shape != (1, 32, 366) or not torch.isfinite(predicted).all():
                raise ValueError('Native forward must return finite [1,32,366]')
            emitted = predicted[:, 4:8].clone()
            # Retain the unmodified generated frames even if their re-encoding
            # fails, so the offending boundary remains in the partial world.
            parts.append(emitted.cpu())
            stage = 'history_reencoding'
            synchronize()
            encode_start = time.perf_counter()
            history = reencode(emitted, rep)
            synchronize()
            cycle_ms = forward_ms + (time.perf_counter() - encode_start) * 1000
            timings.append(dict(first_emitted_frame=offset, forward_ms=forward_ms,
                                forward_and_reencode_ms=cycle_ms))
            stage = 'observation_check'
            observations.append(observe())
            call_start = None
    except Exception as error:
        failures.append(dict(stage=stage, call_index=len(histories) - 1,
                             emitted_frames=sum(part.shape[1] for part in parts), error=repr(error),
                             failed_call_wall_ms=(time.perf_counter() - call_start) * 1000
                             if call_start is not None else None))
    raw = dict(schema=SCHEMA, world=torch.cat([seed.detach().cpu().clone(), *parts], 1),
               seed=seed.detach().cpu().clone(), scene={key: value.detach().cpu().clone()
               if torch.is_tensor(value) else value for key, value in scene.items()},
               trial=trial, conditioned_histories=histories, native_forecasts=forecasts,
               dispatched_controls=controls, timings=timings, observation_unknown_counts=observations,
               failures=failures, fps=FPS, postprocessing=None, optimizer_updates=0)
    raw['completed'] = not failures and raw['world'].shape[1] == trial['frames'] + 4
    return raw


def skin_failures(metrics):
    if (metrics.get('vertices_per_frame') != 18056 or metrics.get('swept_substeps') != 8
            or metrics.get('unknown_body_fraction') != 0
            or metrics.get('uncertain_inside_samples') != 0
            or metrics.get('body_prism_max_m') is None
            or not math.isfinite(metrics['body_prism_max_m'])
            or metrics['body_prism_max_m'] > .02):
        return [dict(stage='full_skin', error='Full-skin 2cm/known/certainty gate failed', metrics=metrics)]
    return []


def comparison(rows):
    """Signed raw differences; no checkpoint selection or composite score."""
    parent = {row['trial']: row for row in rows if row['model'] == 'parent'}
    result = []
    for row in rows:
        if row['model'] == 'parent' or row['trial'] not in parent:
            continue
        base = parent[row['trial']]
        original = {phase['name']: phase for phase in base.get('motion', {}).get('phases', [])}
        deltas = []
        for phase in row.get('motion', {}).get('phases', []):
            before = original.get(phase['name'])
            if before is None or not before['complete'] or not phase['complete']:
                continue
            keys = ('mean_signed_velocity_mps', 'mean_abs_orthogonal_velocity_mps',
                    'idle_endpoint_displacement_m', 'idle_max_displacement_m')
            values = {key: phase[key] - before[key] for key in keys if key in phase and key in before}
            values['body_relative_joint_step_p95_m'] = (phase['body_relative_joint_step_m']['p95']
                                                       - before['body_relative_joint_step_m']['p95'])
            for key in ('start_threshold', 'stop_delay'):
                if key in phase and key in before:
                    current, previous = phase[key]['onset_seconds'], before[key]['onset_seconds']
                    values[key] = dict(parent_seconds=previous, candidate_seconds=current,
                                       delta_seconds=current - previous
                                       if current is not None and previous is not None else None)
            deltas.append(dict(phase=phase['name'], candidate_minus_parent=values))
        result.append(dict(model=row['model'], trial=row['trial'], phases=deltas,
                           parent_completed=base['completed'], candidate_completed=row['completed']))
    return result


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, failures, controller = [], [], None
    report = dict(schema=SCHEMA, purpose='fixed_endpoint_offline_long_control_comparison',
                  optimizer_updates=0, postprocessing=None, quality_approved=False,
                  eligible_for_automatic_export=False, evaluation_script_sha256=sha256(Path(__file__)),
                  rows=rows, failures=failures)
    try:
        parent_hash, bundle_hash, selection_hash = map(sha256, (args.parent, args.bundle, args.selection))
        proof = read_json(args.selection.parent / 'complete.json')
        if (proof.get('passed') is not True or proof.get('optimizer_updates') != 0
                or proof.get('selection_sha256') != selection_hash
                or proof.get('parent_sha256') != parent_hash or proof.get('bundle_sha256') != bundle_hash):
            raise ValueError('Selection is not bound by the matching zero-update preflight')
        selection = read_json(args.selection)
        chosen = select_idle(selection)
        parent = torch.load(args.parent, map_location='cpu', mmap=True, weights_only=False)
        receipts = parent['contract']['cache_receipts']
        if len(receipts) != len(args.cache):
            raise ValueError('Cache count differs from the audited parent')
        for path, receipt in zip(args.cache, receipts):
            if (str(path.resolve()) != receipt['path']
                    or sha256(path / 'complete.json') != receipt['complete_sha256']
                    or sha256(path / 'data/records.jsonl') != receipt['records_sha256']):
                raise ValueError('Audited cache identity changed')
        models = [dict(name='parent', path=args.parent, sha256=parent_hash)]
        arm_contracts = []
        for path in args.candidates:
            saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            contract = validate_endpoint(saved, parent_hash, bundle_hash)
            complete, file_hash = read_json(path.parent / 'complete.json'), sha256(path)
            if (complete.get('completed') is not True or complete.get('optimizer_updates') != 64
                    or complete.get('source_window_exposures') != 512
                    or complete.get('checkpoint_sha256') != file_hash
                    or contract['sources'] != proof['sources']
                    or contract['plan_sha256'] != proof['plan_sha256']):
                raise ValueError('Candidate completion/source binding differs from preflight')
            arm_contracts.append(contract)
            models.append(dict(name=contract['arm'], path=path, sha256=file_hash,
                               updates=64, source_window_exposures=512))
            del saved
        if {value['name'] for value in models[1:]} != set(ARMS):
            raise ValueError('Need exactly one source_history and one generated_history endpoint')
        if arm_contracts[0]['plan'] != arm_contracts[1]['plan']:
            raise ValueError('The two endpoint plans differ')
        seed_number = arm_contracts[0]['plan']['seed']
        report.update(models=[dict(value, path=str(value['path'].resolve())) for value in models],
                      bundle_sha256=bundle_hash, parent_sha256=parent_hash, selection_sha256=selection_hash,
                      rng_seed=seed_number, selected_seed=chosen, runtime=dict(tokens=8, pose_passes=4,
                      history_frames=4, executed_frames=4, motion_clock_hz=FPS),
                      timing_scope='separately synchronized native forward and reencoding; '
                                   'archive copies, FK metrics and skin excluded; no render/FPS measurement',
                      thresholds=dict(start_speed_mps=START_SPEED, stop_speed_mps=STOP_SPEED,
                                      sustained_frames=SETTLE_FRAMES, skin_peak_m=.02),
                      control_screen=CONTROL_SCREEN)
        write_json(args.output / 'report.json', report)

        from motionbricks.data.native_mb_data import build_native_representation
        from motionbricks.data.native_mb_observed_history import reencode_played_history
        from motionbricks.data.reactive_training_data import ReactiveSourceBank, selection_receipt

        if selection_receipt(selection['validation']) != selection['validation_receipt']:
            raise ValueError('Validation selection receipt differs')
        rep_cpu = build_native_representation(args.mixed / 'stats')
        source = ReactiveSourceBank(args.cache, args.pfnn, args.mixed, rep_cpu).load(chosen)
        seed_cpu = source['world'][:, :4].clone()
        ground_y = plane_height(source['scene'])
        arena = control_arena(seed_cpu, ground_y)
        trials = controlled_trials(reencode_played_history(seed_cpu, rep_cpu))
        torch.save(dict(seed=seed_cpu, scene=arena, source_scene=source['scene'], selection=chosen, receipt=source['receipt'],
                        trials=trials, selection_sha256=selection_hash), args.output / 'inputs.pt')
        report.update(source_receipt=source['receipt'], factual_plane_height_m=ground_y,
                      scene_kind='authored64m_plane_at_source_ground_height',
                      observation_policy='mask_unknown_queries; require_known_inputs; zero_unknown_actual_skin',
                      inputs_sha256=sha256(args.output / 'inputs.pt'))

        from scripts.run_pfnn_parallel_workflow import gpu_snapshot

        snapshot, idle = gpu_snapshot()
        report.update(gpu_index=args.gpu, gpu_snapshot=snapshot)
        if args.gpu not in idle:
            raise ValueError('Selected GPU is occupied')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        from motionbricks.data.full_corpus_inference import inference_receipt, load_candidate, tensor_digest
        from motionbricks.data.reactive_prefix_training import TrainingRootPathObservation
        from motionbricks.data.terrain_joint_objective import JointGeometryObjective
        from motionbricks.data.unified_terrain import UnifiedTerrainController
        from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
        from scripts.portable_parc_online_soma import load_terrain_bundle

        construction_started = time.monotonic()
        inference, bundle = load_terrain_bundle(args.bundle, 'cuda:0')
        controller = UnifiedTerrainController(inference, bundle)
        torch.cuda.synchronize()
        report['controller_construction_seconds'] = time.monotonic() - construction_started
        parent_load_started = time.monotonic()
        load_candidate(controller, parent, inference_receipt(parent, parent_hash, parent_hash, bundle_hash), bundle_hash)
        torch.cuda.synchronize()
        parent_load_seconds = time.monotonic() - parent_load_started
        del parent
        observer = TrainingRootPathObservation(controller)
        rep = inference.global_motion_rep
        geometry = JointGeometryObjective(rep, 'cuda:0')
        if len(geometry.skin.bind_vertices) != 18056:
            raise ValueError('Official SOMA skin must have 18056 vertices')
        seed = seed_cpu.to('cuda:0')
        scene = {key: value.to('cuda:0') if torch.is_tensor(value) else value
                 for key, value in arena.items()}
        for model in models:
            load_started = time.monotonic()
            if model['name'] != 'parent':
                saved = torch.load(model['path'], map_location='cpu', mmap=True, weights_only=False)
                validate_endpoint(saved, parent_hash, bundle_hash)
                for value in saved['model_state'].values():
                    if not torch.is_tensor(value) or not torch.isfinite(value).all():
                        raise ValueError('Nonfinite or nontensor pilot state')
                controller.load_state_dict(saved['model_state'], strict=True)
                controller.codebook.copy_(controller.codec.get_codebook())
                del saved
            controller.eval().requires_grad_(False)
            inference.eval().requires_grad_(False)
            torch.cuda.synchronize()
            load_seconds = parent_load_seconds if model['name'] == 'parent' else time.monotonic() - load_started
            state_before = tensor_digest(controller.state_dict())
            for value in report['models']:
                if value['name'] == model['name']:
                    value.update(load_seconds=load_seconds, model_state_sha256=state_before)
            for trial in trials:
                random.seed(seed_number)
                np.random.seed(seed_number)
                torch.manual_seed(seed_number)
                raw = run_rollout(controller, rep, seed, scene, trial, reencode=reencode_played_history,
                                  synchronize=torch.cuda.synchronize,
                                  observe=lambda: check_observations(controller, observer))
                directory = args.output / model['name'] / trial['name']
                directory.mkdir(parents=True, exist_ok=False)
                raw.update(model=model['name'], checkpoint_sha256=model['sha256'], rng_seed=seed_number,
                           source_receipt=source['receipt'])
                # Persist before FK/skin evaluation, including rejected forecasts.
                torch.save(raw, directory / 'rollout.pt')
                row = dict(model=model['name'], trial=trial['name'], completed=raw['completed'],
                           requested_frames=trial['frames'], emitted_frames=raw['world'].shape[1] - 4,
                           raw_world_path=str((directory / 'rollout.pt').resolve()),
                           raw_sha256=sha256(directory / 'rollout.pt'), failures=list(raw['failures']),
                           forward_ms=distribution([value['forward_ms'] for value in raw['timings']]),
                           forward_and_reencode_ms=distribution([value['forward_and_reencode_ms']
                                                               for value in raw['timings']]))
                stage = 'fk_metrics'
                try:
                    with torch.no_grad():
                        world = raw['world'].to(seed)
                        pose = rep.inverse(world, False, return_all=False, joint_positions_from='global_rot_data')
                        row['motion'] = motion_metrics(raw['world'], pose['posed_joints'], trial, ground_y,
                                                       skeleton=rep.skeleton)
                        row['control_screen'] = control_screen(row['motion'], trial)
                        row['failures'].extend(dict(stage='control_screen', **failure)
                                               for failure in row['control_screen']['failures'])
                        stage = 'full_skin'
                        skin_started = time.monotonic()
                        row['skin'] = exact_skin_metrics(geometry, world, scene)
                        row['skin_seconds'] = time.monotonic() - skin_started
                        row['failures'].extend(skin_failures(row['skin']))
                except Exception as error:
                    row['failures'].append(dict(stage=stage, error=repr(error)))
                row['passed'] = row['completed'] and not row['failures']
                rows.append(row)
                failures.extend(dict(model=model['name'], trial=trial['name'], **failure) for failure in row['failures'])
                report['comparison'] = comparison(rows)
                write_json(directory / 'metrics.json', row)
                write_json(args.output / 'report.json', report)
                print(json.dumps(dict(model=model['name'], trial=trial['name'], completed=row['completed'],
                                      emitted_frames=row['emitted_frames'], failures=len(row['failures']))), flush=True)
            if tensor_digest(controller.state_dict()) != state_before:
                raise ValueError('Frozen evaluation changed model tensors')
            for value in report['models']:
                if value['name'] == model['name']:
                    value['weights_unchanged'] = True
            if sha256(model['path']) != model['sha256']:
                raise ValueError('Checkpoint file changed during evaluation')
        report.update(completed=True, passed=not failures and len(rows) == 9,
                      elapsed_seconds=time.monotonic() - started)
        write_json(args.output / 'report.json', report)
        write_json(args.output / 'complete.json', report)
        return 0 if report['passed'] else 1
    except Exception as error:
        failures.append(dict(stage='evaluation', error=repr(error)))
        report.update(completed=False, passed=False, elapsed_seconds=time.monotonic() - started)
        write_json(args.output / 'report.json', report)
        write_json(args.output / 'failure.json', report)
        raise
    finally:
        if controller is not None:
            controller.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'parent', 'selection', 'pfnn', 'mixed', 'output'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--candidates', type=Path, nargs=2, required=True,
                        metavar=('SOURCE_HISTORY_FINAL64', 'GENERATED_HISTORY_FINAL64'))
    result.add_argument('--cache', type=Path, nargs=2, required=True)
    result.add_argument('--gpu', type=int, required=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main(parser().parse_args()))
