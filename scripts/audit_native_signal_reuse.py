"""Offline height-signal tracing and matched buffer-reuse policy experiment."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.unified_terrain import UnifiedTerrainController, UnifiedTerrainBackend, PROFILE
from motionbricks.data.unified_training_goals import source_terminal_goal
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.audit_unified_stair_gait import generate
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


TRAINED_SHA = '03580db6f19c407b605fb43a6dd23514f8d5f17c5ad7b869c7ec95059ca68ae1'


@torch.no_grad()
def signal_trace(controller, history, command, scene):
    captured = {}
    codec = controller.codec
    original = codec.forward_decoder

    def capture_decoder(*args, **kwargs):
        captured['external'] = kwargs['external_cond'].detach().clone()
        return original(*args, **kwargs)

    def capture_projection(_module, _args, output):
        captured['height_projection'] = output.detach().clone()

    handle = controller.pose._terrain_root_height_projection.register_forward_hook(capture_projection)
    codec.forward_decoder = capture_decoder
    try:
        result = generate(controller, history, command, scene)
    finally:
        codec.forward_decoder = original
        handle.remove()
    inference = controller.inference
    origin = vertical_origin(inference, history, result['observation'])[:, None]
    root = inference.global_motion_rep.unnormalize(result['root_output']['pred_global_root_values'][:, :32])
    root_y = root[..., 1]
    pose_height = controller.last_pose_request[1]['root_height_trajectory'][:, :32, 0]
    expected_pose = root_y-root_y[:, 3:4]
    torch.testing.assert_close(pose_height, expected_pose, atol=2e-6, rtol=0)
    rep = inference.local_motion_rep
    indices = np.asarray(extract_feature_from_motion_rep(None, codec.motion_rep,
        codec.decoder_external_cond_feature_mode, fetch_feat_idx=True)).reshape(-1)
    y_index = int(np.asarray(rep.indices['global_root_y']).reshape(-1)[0])
    if y_index not in indices:
        raise ValueError('This approved checkpoint should expose decoder Root height')
    full = root.new_zeros(1, captured['external'].shape[1], len(rep.indices['all']))
    full[..., indices] = captured['external']
    external_y = rep.unnormalize(full)[:, :32, y_index]
    torch.testing.assert_close(external_y, root_y, atol=2e-6, rtol=0)
    decoded_y = rep.unnormalize(result['local_prediction'])[:, :32, y_index]+origin
    world_y = result['world'][:, :32, 1]
    torch.testing.assert_close(decoded_y, world_y, atol=2e-6, rtol=0)
    gate = controller.pose._terrain_root_height_gate.detach()
    return dict(planner_y=root_y+origin, pose_height=pose_height,
        pose_height_expected=expected_pose, decoder_external_y=external_y+origin,
        decoder_output_y=decoded_y, world_y=world_y, tokens=result['tokens'].clone(),
        gated_height_embedding=captured['height_projection'][:, :8]*gate,
        pose_height_gate=float(gate), checks=dict(
            pose_height_delivery_max_error_m=float((pose_height-expected_pose).abs().max()),
            decoder_height_delivery_max_error_m=float((external_y-root_y).abs().max()),
            decoded_to_world_height_max_error_m=float((decoded_y-world_y).abs().max())))


@torch.no_grad()
def trace_pair(controller, case, start):
    source = case['world'][None, start:start+33].to(controller.inference.device)
    history = source[:, :4]
    command = source_terminal_goal(source[..., :5])
    floor = float(history[0, -1, 1]-heightfield_gap(history[:, -1:, :3], case['scene'])[0][0, 0])
    height = torch.as_tensor(case['scene']['heightfield_y_m'])
    plane = dict(case['scene'], heightfield_y_m=torch.where(torch.isfinite(height), torch.full_like(height, floor), height))
    actual = signal_trace(controller, history, command, case['scene'])
    changed = signal_trace(controller, history, command, plane)
    differences = {name: (actual[name]-changed[name]).detach().cpu() for name in
        ('planner_y', 'pose_height', 'decoder_external_y', 'decoder_output_y', 'world_y')}
    return dict(label=case['label'], start=start, actual_checks=actual['checks'], plane_checks=changed['checks'],
        max_difference_m={name: float(value[:, 4:28].abs().max()) for name, value in differences.items()},
        difference_series_m={name: value[0, 4:28].tolist() for name, value in differences.items()},
        changed_token_head_entries=int((actual['tokens'] != changed['tokens']).sum()),
        pose_height_gate=actual['pose_height_gate'], gated_pose_height_embedding_difference_rms=float(
            (actual['gated_height_embedding']-changed['gated_height_embedding']).square().mean().sqrt()),
        interpretation='Relative Pose height uses predicted frame3; embedding RMS is not a distance unit. '
            'local_prediction is already decoder output. No maxima are treated as additive error components.')


def should_refresh(buffer_remaining, last_sample_position, current_position, threshold):
    return buffer_remaining >= 8 and last_sample_position is not None \
        and float((current_position-last_sample_position).norm()) > threshold


def observation_trace(debug, scene, current_history):
    snapshot = debug['unified_sdf']
    points = torch.tensor(snapshot['world_xyz_m'], device=current_history.device).reshape(1, 3, -1, 3)
    valid = torch.tensor(snapshot['valid'], device=current_history.device, dtype=torch.bool).reshape(points.shape[:-1])
    gap, known = heightfield_gap(points[:, 1], scene)
    levels = points[:, 1, :, 1]-gap
    sample_root = current_history.new_tensor(snapshot['root_world_xyz_m'])
    floor = sample_root[1]-heightfield_gap(sample_root.reshape(1, 1, 3), scene)[0][0, 0]
    active = valid[:, 1] & known
    delta = levels[active]-floor
    return dict(sampled_root_world_xyz_m=sample_root.tolist(),
        position_drift_from_sampling_m=float((current_history[0, -1, [0, 2]]-sample_root[[0, 2]]).norm()),
        age_frames=int(debug.get('observation_lag_frames', 0)),
        intent_known_points=int(active.sum()),
        intent_top_height_change_min_m=float(delta.min()) if len(delta) else None,
        intent_top_height_change_max_m=float(delta.max()) if len(delta) else None,
        intent_has_height_contrast=bool((delta.abs() > .04).any()),
        scope='Diagnostic height contrast in actual cached model-input points, not a runtime terrain classifier.')


@torch.no_grad()
def rollout(controller, geometry, case, seed, command, policy, max_blocks):
    backend = UnifiedTerrainBackend(controller, seed.clone(), case['scene'])
    backend.configure_runtime(dict(PROFILE))
    source = case['world'].to(seed.device)
    begin, target = source[3, [0, 2]], source[-1, [0, 2]]
    axis = target-begin
    length = float(axis.norm())
    axis = axis/axis.norm()
    side = torch.stack((axis[1], -axis[0]))
    heading = float(torch.atan2(command[0, 3], command[0, 2]))
    pieces, trace, failure = [seed.clone()], [], None
    sampled_at = None
    for block in range(max_blocks):
        history = backend.history.clone()
        remaining = 0 if backend._future_buffer is None else backend._future_buffer.shape[1]
        forced = policy == 'drift_0.4' and should_refresh(remaining, sampled_at, history[0, -1, [0, 2]], .4)
        if forced:
            # Offline policy intervention only; never replace an already played frame.
            backend._future_buffer = None
            backend._last_command = None
        try:
            started = time.perf_counter()
            result = backend.predict_plan(command[0, :2].tolist(), heading)
            elapsed = (time.perf_counter()-started)*1000
        except (RuntimeError, ValueError) as error:
            failure = repr(error)
            break
        neural = bool(backend.last_generation['neural_generation'])
        if neural:
            sampled_at = history[0, -1, [0, 2]].clone()
        observation = observation_trace(result['debug'], case['scene'], history)
        progress = float((history[0, -1, [0, 2]]-begin)@axis)
        trace.append(dict(block=block, current_history_end_frame=3+block*8, neural_generation=neural,
            forced_by_threshold=forced, buffer_remaining_before=remaining, compute_ms=elapsed,
            root_progress_m=progress, root_world_xyz_m=history[0, -1, :3].tolist(), observation=observation))
        pieces.append(result['planned_motion'].clone())
        backend.history = result['published_history'].clone()
        if float((backend.history[0, -1, [0, 2]]-begin)@axis) >= length:
            break
    world = torch.cat(pieces, 1)
    skin = exact_skin_metrics(geometry, world, case['scene'])
    progress = float((world[0, -1, [0, 2]]-begin)@axis)
    lateral = float(((world[0, :, [0, 2]]-begin)@side).abs().max())
    reached = progress >= length
    gate = reached and lateral <= .2 and skin['body_prism_max_m'] <= .02 \
        and skin['unknown_body_fraction'] == 0 and skin['uncertain_inside_samples'] == 0 and failure is None
    return dict(policy=policy, actual_frames=world.shape[1], trace=trace, failure=failure,
        neural_generations=sum(row['neural_generation'] for row in trace),
        forced_generations=sum(row['forced_by_threshold'] for row in trace),
        progress_m=progress, goal_distance_m=length, lateral_m=lateral, reached_end_plane=reached,
        skin=skin, route_geometry_gate_pass=gate), world


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17121)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(args.pose) != TRAINED_SHA:
        raise ValueError('Wrong approved native checkpoint')
    saved = torch.load(args.pose, map_location='cpu', weights_only=False)
    if sha256(args.cases) != saved['contract']['cases_sha256'] or sha256(args.bundle) != saved['contract']['bundle_sha256']:
        raise ValueError('Source contract changed')
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True)
    controller.eval().requires_grad_(False)
    before = state_digest(controller, list(controller.state_dict()))
    anchors = [window for window in saved['contract']['capacity_anchors'] if window['label'].endswith(('-up', '-down'))]
    lookup = {case['label']: case for case in cases}
    signal_rows = []
    for anchor in anchors:
        case = lookup[anchor['label']]
        for start in (24, 32, 36):
            signal_rows.append(trace_pair(controller, case, start))
    write_json(args.output/'signals.json', dict(rows=signal_rows))
    write_json(args.output/'state.json', dict(stage='reuse', optimizer_updates=0))
    geometry = JointGeometryObjective(inference.global_motion_rep, args.device)
    idle = next(case for case in cases if case['name'] == 'idle')
    rows = []
    for anchor in anchors:
        case = lookup[anchor['label']]
        source = case['world'][None].to(args.device)
        command = source_terminal_goal(source[:, anchor['start']:anchor['start']+33, :5])
        heading = float(torch.atan2(command[0, 3], command[0, 2]))
        idle_history = inference.global_motion_rep.change_first_heading(idle['world'][None, :4].to(args.device), heading, False, False)
        old_floor = idle_history[..., 1]-heightfield_gap(idle_history[..., :3], idle['scene'])[0]
        idle_history[..., [0, 2]] += source[:, 3:4, [0, 2]]-idle_history[:, -1:, [0, 2]].clone()
        new_floor = idle_history[..., 1]-heightfield_gap(idle_history[..., :3], case['scene'])[0]
        if float(new_floor.max()-new_floor.min()) > 1e-5:
            raise ValueError('Idle placement requires a measured flat approach')
        idle_history = shift_global_motion(idle_history, inference.global_motion_rep, new_floor[:, -1]-old_floor[:, -1])
        slow = command.clone()
        slow[:, :2] *= 1.6/command[:, :2].norm(dim=-1, keepdim=True)
        for mode, seed, control in (('source_anchor_command', source[:, :4], command),
                                  ('idle_anchor_command', idle_history, command), ('idle_1.6', idle_history, slow)):
            values = []
            for policy in ('default24', 'drift_0.4'):
                result, world = rollout(controller, geometry, case, seed, control, policy, args.max_blocks)
                values.append(result)
                folder = args.output/'recordings'
                folder.mkdir(exist_ok=True)
                torch.save(dict(world=world.cpu(), scene=case['scene'], role='offline_policy_diagnostic',
                    fps=30, actual_frames=world.shape[1], no_postprocessing=True),
                    folder/(case['label']+'-'+mode+'-'+policy+'.pt'))
            rows.append(dict(label=case['label'], mode=mode, command=control[0].tolist(), policies=values))
            write_json(args.output/'reuse-progress.json', dict(rows=rows))
    if state_digest(controller, list(controller.state_dict())) != before:
        raise ValueError('Read-only diagnostic changed model tensors')
    complete = dict(signals=signal_rows, reuse=rows, optimizer_updates=0, weights_unchanged=True,
        model_sha256=sha256(args.pose), script_sha256=sha256(Path(__file__)), runtime_profile=dict(PROFILE),
        live_runtime_changed=False, threshold_scope='0.4m rule tested offline at fully played8-frame boundaries; '
            'not browser prefetch latency or command-age validation.', quality_approved=False)
    write_json(args.output/'complete.json', complete)
    write_json(args.output/'state.json', dict(stage='complete', optimizer_updates=0, weights_unchanged=True))
    controller.close()
    print(json.dumps(dict(complete=True, signal_pairs=len(signal_rows), matched_policy_cases=len(rows))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'pose', 'cases', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-blocks', type=int, default=60)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        if arguments.output.is_dir():
            write_json(arguments.output/'failure.json', dict(error=repr(error)))
        raise
