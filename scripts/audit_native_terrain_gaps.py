"""Read-only coverage, geometry intervention, Pose ablation and route audit."""

import argparse
from contextlib import contextmanager
import hashlib
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
from scripts.audit_unified_stair_gait import difference, gait_features, generate
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


TRAINED_SHA = '03580db6f19c407b605fb43a6dd23514f8d5f17c5ad7b869c7ec95059ca68ae1'


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return None if not values.size else np.quantile(values, [0, .25, .5, .75, 1]).tolist()


def angle_delta(a, b):
    return torch.atan2(torch.sin(a-b), torch.cos(a-b))


def scene_hash(scene):
    digest = hashlib.sha256()
    for key in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m'):
        value = torch.as_tensor(scene[key]).cpu().contiguous().numpy()
        digest.update(key.encode('ascii'))
        digest.update(str(value.shape).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def coverage(cases, windows):
    by_label = {case['label']: case for case in cases}
    rows = []
    for window in windows:
        case = by_label[window['label']]
        source = case['world'][None, window['start']:window['start']+33]
        command = source_terminal_goal(source[..., :5])
        velocity = torch.diff(source[0, :, [0, 2]], dim=0)*30
        history_speed = velocity[:3].norm(dim=-1).mean()
        future_velocity = velocity[3:27].mean(0)
        heading = torch.atan2(source[0, :, 4], source[0, :, 3])
        travel = torch.atan2(command[0, 0], command[0, 1])
        target_heading = torch.atan2(command[0, 3], command[0, 2])
        speed = float(command[0, :2].norm())
        rows.append(dict(**window, name=case['name'], entry=case.get('entry'),
            command=command[0].tolist(), speed_mps=speed,
            desired_heading_rad=float(target_heading), history_heading_rad=float(heading[3]),
            history_speed_mps=float(history_speed), future_mean_speed_mps=float(velocity[3:27].norm(dim=-1).mean()),
            terminal_vs_mean_future_velocity_error_mps=float((command[0, :2]-future_velocity).norm()),
            desired_minus_history_heading_deg=float(torch.rad2deg(angle_delta(target_heading, heading[3]))),
            travel_minus_desired_heading_deg=None if speed < .05 else
                float(torch.rad2deg(angle_delta(travel, target_heading))),
            future_height_change_m=float(source[0, 27, 1]-source[0, 3, 1]),
            low_speed_history_to_moving_command=bool(history_speed < .1 and speed > .5)))
    groups = []
    for label in dict.fromkeys(row['label'] for row in rows):
        selected = [row for row in rows if row['label'] == label]
        groups.append(dict(label=label, name=selected[0]['name'], entry=selected[0]['entry'],
            windows=len(selected), speed_quantiles_mps=quantiles([row['speed_mps'] for row in selected]),
            desired_heading_quantiles_rad=quantiles([row['desired_heading_rad'] for row in selected]),
            terminal_vs_mean_future_error_quantiles_mps=quantiles([
                row['terminal_vs_mean_future_velocity_error_mps'] for row in selected]),
            low_speed_history_to_moving_command_windows=sum(row['low_speed_history_to_moving_command'] for row in selected)))
    return dict(rows=rows, groups=groups, total_windows=len(rows), unique_clips=len(groups),
        quantile_order=['min', 'p25', 'median', 'p75', 'max'],
        phase_scope='Source-history foot pose is added after representation loading; not inferred intent.',
        command_source='Source interval31->32 velocity and heading31; no recorded human input.',
        startup_diagnostic_thresholds=dict(history_speed_below_mps=.1, command_speed_above_mps=.5))


@contextmanager
def pose_ablation(controller, arm):
    choices = {'none': [], 'legacy_pose_off': [controller.pose._terrain_cross_attention],
        'volume_pose_off': [controller.adaptation['pose']],
        'both_pose_off': [controller.pose._terrain_cross_attention, controller.adaptation['pose']]}
    if arm not in choices:
        raise ValueError('Unknown offline ablation arm')
    calls = [0]

    def bypass(_module, args, _output):
        calls[0] += 1
        return args[0]

    handles = [module.register_forward_hook(bypass) for module in choices[arm]]
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def prediction(controller, history, command, scene):
    result = generate(controller, history, command, scene)
    return dict(world=result['world'].detach().clone(), tokens=result['tokens'].detach().clone(),
        logits=result['pose_logits'][:, :8].detach().clone(),
        planner_y=controller.inference.global_motion_rep.unnormalize(
            result['root_output']['pred_global_root_values'][:, :32])[..., 1].detach().clone()
            + vertical_origin(controller.inference, history, result['observation'])[:, None],
        origin=vertical_origin(controller.inference, history, result['observation']).detach().clone())


def comparison(rep, baseline, changed):
    a, b = baseline['world'][:, 4:28], changed['world'][:, 4:28]
    errors = difference(rep, b, a)
    return dict(logit_max=float((baseline['logits']-changed['logits']).abs().max()),
        probability_mean_absolute=float((baseline['logits'].softmax(-1)-changed['logits'].softmax(-1)).abs().mean()),
        changed_tokens=int((baseline['tokens'] != changed['tokens']).sum()),
        total_tokens=baseline['tokens'].numel(),
        planner_height_max_difference_m=float((baseline['planner_y'][:, 4:28]-changed['planner_y'][:, 4:28]).abs().max()),
        decoded_height_max_difference_m=float((a[..., 1]-b[..., 1]).abs().max()),
        decoded_root_xyz_rms_difference_m=float((a[..., :3]-b[..., :3]).square().mean().sqrt()),
        relative_leg_difference_m=errors['leg_relative_mpjpe_m'],
        origin_difference_m=float((baseline['origin']-changed['origin']).abs().max()))


def save_world(folder, label, world, scene, role):
    folder.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha256(label.encode('utf-8')).hexdigest()[:16]
    torch.save(dict(label=label, role=role, global_features=world.detach().cpu(), scene=scene,
        fps=30, actual_frames=world.shape[1], postprocessing=None), folder/(stem+'.pt'))


@torch.no_grad()
def condition_audit(controller, cases, windows, report, output):
    rep = controller.inference.global_motion_rep
    lookup = {case['label']: case for case in cases}
    expected = {(row['label'], row['start']): row['values']['actor'] for row in report['final']}
    selected = [window for window in windows if lookup[window['label']].get('entry') in ('up', 'down')]
    selected += [next(window for window in windows if window['label'] == case['label'])
                 for case in cases if case['label'] in lookup and case.get('entry') is None
                 and any(window['label'] == case['label'] for window in windows)]
    rows = []
    for index, window in enumerate(selected):
        case = lookup[window['label']]
        source = case['world'][None, window['start']:window['start']+33].to(controller.inference.device)
        history = source[:, :4]
        command = source_terminal_goal(source[..., :5])
        baseline = prediction(controller, history, command, case['scene'])
        replay = prediction(controller, history, command, case['scene'])
        if not torch.equal(baseline['tokens'], replay['tokens']):
            raise ValueError('Baseline replay tokens changed')
        torch.testing.assert_close(baseline['world'], replay['world'], atol=1e-5, rtol=0)
        values = difference(rep, baseline['world'][:, 4:28], source[:, 4:28])
        root_rms = float((baseline['world'][:, 4:28, :3]-source[:, 4:28, :3]).square().mean().sqrt())
        previous = expected[(window['label'], window['start'])]
        parity = max(abs(values['leg_relative_mpjpe_m']-previous['errors']['leg_relative_mpjpe_m']),
                     abs(root_rms-previous['root_xyz_rms_m']))
        if parity > 5e-5:
            raise ValueError('Current actor differs from the checkpoint training-bank review')
        arms = {}
        for arm in ('legacy_pose_off', 'volume_pose_off', 'both_pose_off'):
            with pose_ablation(controller, arm) as calls:
                changed = prediction(controller, history, command, case['scene'])
            if calls[0] == 0 or not torch.equal(baseline['planner_y'], changed['planner_y']):
                raise ValueError('Pose intervention failed to preserve Root or never executed')
            arms[arm] = dict(comparison(rep, baseline, changed), hook_calls=calls[0])
        restored = prediction(controller, history, command, case['scene'])
        torch.testing.assert_close(baseline['world'], restored['world'], atol=1e-5, rtol=0)
        if not torch.equal(baseline['tokens'], restored['tokens']):
            raise ValueError('Pose intervention was not fully restored')
        floor = float(history[0, -1, 1]-heightfield_gap(history[:, -1:, :3], case['scene'])[0][0, 0])
        height = torch.as_tensor(case['scene']['heightfield_y_m'])
        plane = dict(case['scene'], heightfield_y_m=torch.where(torch.isfinite(height),
            torch.full_like(height, floor), height))
        flat = prediction(controller, history, command, plane)
        row = dict(**window, entry=case.get('entry'), native_bank_replay_metric_difference_m=parity,
            baseline_replay_max_feature_difference=float((baseline['world']-replay['world']).abs().max()),
            pose_ablations=arms, geometry_comparison=comparison(rep, baseline, flat),
            actual_planner_height_change_m=float(baseline['planner_y'][0, 27]-baseline['planner_y'][0, 3]),
            plane_planner_height_change_m=float(flat['planner_y'][0, 27]-flat['planner_y'][0, 3]),
            actual_decoded_height_change_m=float(baseline['world'][0, 27, 1]-history[0, 3, 1]),
            plane_decoded_height_change_m=float(flat['world'][0, 27, 1]-history[0, 3, 1]))
        rows.append(row)
        write_json(output/'conditions-progress.json', dict(rows=rows))
    return dict(rows=rows, same_history_and_command=True, ablation_root_preserved=True,
        all_interventions_restored=True, counterfactual_scope='Changes both scene-conditioning paths; '
        'stair history with flattened geometry is not a clean flat reference.',
        ablation_scope='Only Pose cross-attention residual output bypassed; Root, decoder and weights unchanged.')


def continuity(rep, world, boundaries):
    joints = rep.inverse(world, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints'][0]
    root_speed = torch.diff(world[0, :, :3], dim=0).norm(dim=-1)*30
    joint_speed = torch.diff(joints, dim=0).norm(dim=-1)*30
    acceleration = torch.diff(joints, n=2, dim=0).norm(dim=-1)*900
    rows = []
    for boundary in boundaries:
        frame = boundary['frame']
        if frame < 2 or frame >= len(joints):
            continue
        rows.append(dict(**boundary, root_speed_mps=float(root_speed[frame-1]),
            joint_speed_max_mps=float(joint_speed[frame-1].max()),
            joint_acceleration_max_mps2=float(acceleration[frame-2].max())))
    mask = torch.ones(len(root_speed), dtype=torch.bool, device=root_speed.device)
    for boundary in boundaries:
        if 0 < boundary['frame'] <= len(root_speed):
            mask[boundary['frame']-1] = False
    return dict(boundaries=rows, interior_joint_speed_max_quantiles_mps=quantiles(
        joint_speed[mask].amax(-1).cpu().numpy()),
        scope='Physical frame-to-frame derivatives; no smoothing, no invented discontinuity threshold.')


@torch.no_grad()
def route_audit(controller, cases, trained_windows, geometry, output, max_blocks):
    rep = controller.inference.global_motion_rep
    trained = {window['label'] for window in trained_windows}
    trained_scenes = {scene_hash(case['scene']) for case in cases if case['label'] in trained}
    idle = next(case for case in cases if case['name'] == 'idle')
    rows = []
    for case in cases:
        if case.get('entry') not in ('up', 'down'):
            continue
        source = case['world'][None].to(controller.inference.device)
        start = max(0, min(case['interval']['first_riser']-case['interval']['start']-4, len(case['world'])-33))
        command = source_terminal_goal(source[:, start:start+33, :5])
        history = source[:, start:start+4]
        output_window = prediction(controller, history, command, case['scene'])
        error = difference(rep, output_window['world'][:, 4:28], source[:, start+4:start+28])
        window_skin = exact_skin_metrics(geometry, torch.cat((history, output_window['world'][:, 4:28]), 1), case['scene'])
        rms = float((output_window['world'][:, 4:28, :3]-source[:, start+4:start+28, :3]).square().mean().sqrt())
        window_pass = error['leg_relative_mpjpe_m'] <= .015 and rms <= .02 and window_skin['body_prism_max_m'] <= .02 \
            and window_skin['unknown_body_fraction'] == 0 and window_skin['uncertain_inside_samples'] == 0
        heading = float(torch.atan2(command[0, 3], command[0, 2]))
        idle_history = rep.change_first_heading(idle['world'][None, :4].to(source.device), heading, False, False)
        old_floor = idle_history[..., 1]-heightfield_gap(idle_history[..., :3], idle['scene'])[0]
        idle_history[..., [0, 2]] += source[:, 3:4, [0, 2]]-idle_history[:, -1:, [0, 2]].clone()
        new_floor = idle_history[..., 1]-heightfield_gap(idle_history[..., :3], case['scene'])[0]
        if float(new_floor.max()-new_floor.min()) > 1e-5:
            raise ValueError('Idle seed placement requires a measured flat approach')
        idle_history = shift_global_motion(idle_history, rep, new_floor[:, -1]-old_floor[:, -1])
        begin = source[0, 3, [0, 2]]
        target = source[0, -1, [0, 2]]
        axis = target-begin
        length = float(axis.norm())
        axis = axis/axis.norm()
        side = torch.stack((axis[1], -axis[0]))
        row = dict(label=case['label'], entry=case['entry'], latest_bank_trained_source=case['label'] in trained,
            scene_present_in_latest_bank=scene_hash(case['scene']) in trained_scenes,
            earlier_training_exposure='Not exhaustively audited; this is not a globally unseen-data claim.',
            source_history_window=dict(start=start, leg_relative_mpjpe_m=error['leg_relative_mpjpe_m'],
                root_xyz_rms_m=rms, skin=window_skin, gate_pass=window_pass), rollouts=[])
        for mode, seed in (('source_start', source[:, :4]), ('idle_start', idle_history)):
            backend = UnifiedTerrainBackend(controller, seed.clone(), case['scene'])
            backend.configure_runtime(dict(PROFILE))
            pieces, boundaries, failure = [seed.clone()], [], None
            for block in range(max_blocks):
                try:
                    result = backend.predict_plan(command[0, :2].tolist(), heading)
                    boundaries.append(dict(frame=4+8*block, neural_generation=backend.last_generation['neural_generation']))
                    pieces.append(result['planned_motion'].clone())
                    backend.history = result['published_history'].clone()
                    if float((backend.history[0, -1, [0, 2]]-begin)@axis) >= length:
                        break
                except (ValueError, RuntimeError) as error:
                    failure = repr(error)
                    break
            world = torch.cat(pieces, 1)
            skin = exact_skin_metrics(geometry, world, case['scene'])
            progress = float((world[0, -1, [0, 2]]-begin)@axis)
            lateral = float(((world[0, :, [0, 2]]-begin)@side).abs().max())
            reached = progress >= length
            geometry_pass = skin['body_prism_max_m'] <= .02 and skin['unknown_body_fraction'] == 0 \
                and skin['uncertain_inside_samples'] == 0
            result_row = dict(mode=mode, command=command[0].tolist(), actual_frames=world.shape[1],
                failure=failure, progress_m=progress, route_length_m=length, lateral_m=lateral,
                endpoint_distance_m=float((world[0, -1, [0, 2]]-target).norm()), reached_end_plane=reached,
                geometry_gate_pass=geometry_pass, skin=skin,
                route_geometry_gate_pass=reached and geometry_pass and lateral <= .2 and failure is None,
                completion='reached_end_plane' if reached else 'incomplete_within_review_budget',
                continuity=continuity(rep, world, boundaries))
            save_world(output/'recordings', case['label']+'-'+mode, world, case['scene'], 'actual_generated_feedback')
            row['rollouts'].append(result_row)
        save_world(output/'recordings', case['label']+'-source', source, case['scene'], 'unchanged_source')
        rows.append(row)
        write_json(output/'routes-progress.json', dict(rows=rows))
    return dict(rows=rows, constant_command=True, goal_uses_real_final_source_frame=True,
        review_budget_seconds=max_blocks*8/30, latency_scope='Serial offline execution; not browser timing or command-age measurement.',
        quality_gate=dict(skin_peak_m=.02, lateral_m=.2, unknown_fraction=0, uncertain_inside_samples=0),
        route_scope='Not in latest bank does not establish never seen by inherited models. '
        'One route failure does not prove memorization; reaching a plane does not certify physical support.')


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17121)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(args.pose) != TRAINED_SHA:
        raise ValueError('Require the exact approved native full model')
    saved = torch.load(args.pose, map_location='cpu', weights_only=False)
    if sha256(args.cases) != saved['contract']['cases_sha256'] or sha256(args.bundle) != saved['contract']['bundle_sha256']:
        raise ValueError('Bundle or source cases differ from trained contract')
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    windows = saved['contract']['source_windows']
    if len(windows) != 88:
        raise ValueError('Require the exact88-window bank')
    report = json.loads(args.pose.with_name('complete.json').read_text())
    data = coverage(cases, windows)
    write_json(args.output/'coverage.json', data)
    write_json(args.output/'state.json', dict(stage='loading_model', optimizer_updates=0))
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True)
    controller.eval().requires_grad_(False)
    before = state_digest(controller, list(controller.state_dict()))
    rep = inference.global_motion_rep
    by_label = {case['label']: case for case in cases}
    for row in data['rows']:
        source = by_label[row['label']]['world'][None, row['start']:row['start']+33].to(args.device)
        feet = gait_features(rep, source[:, :4])['feet'][0, 3]
        heading = source[0, 3, 3:5]
        forward = torch.stack((heading[1], heading[0]))
        row['history_left_minus_right_foot_forward_m'] = float((feet[0]-feet[1])[[0, 2]]@forward)
        row['history_left_minus_right_foot_height_m'] = float(feet[0, 1]-feet[1, 1])
    write_json(args.output/'coverage.json', data)
    write_json(args.output/'state.json', dict(stage='conditions', optimizer_updates=0))
    conditions = condition_audit(controller, cases, windows, report, args.output)
    write_json(args.output/'conditions.json', conditions)
    write_json(args.output/'state.json', dict(stage='routes', optimizer_updates=0))
    geometry = JointGeometryObjective(rep, args.device)
    routes = route_audit(controller, cases, windows, geometry, args.output, args.max_blocks)
    write_json(args.output/'routes.json', routes)
    after = state_digest(controller, list(controller.state_dict()))
    if before != after:
        raise ValueError('Diagnostic changed model tensors')
    complete = dict(schema='motionbricks.native_terrain_gaps.audit.v1', optimizer_updates=0,
        weights_unchanged=True, model_sha256=sha256(args.pose), cases_sha256=sha256(args.cases),
        runtime_profile=dict(PROFILE), model_state_sha256=before, coverage=data, conditions=conditions, routes=routes,
        runtime_changed=False, future_truth_in_actor=False, quality_approved=False,
        script_sha256=sha256(Path(__file__)))
    write_json(args.output/'complete.json', complete)
    write_json(args.output/'state.json', dict(stage='complete', optimizer_updates=0, weights_unchanged=True))
    controller.close()
    print(json.dumps(dict(complete=True, windows=len(windows), condition_cases=len(conditions['rows']), routes=len(routes['rows']))), flush=True)


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
