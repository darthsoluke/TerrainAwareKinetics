"""True generated-history course review of the passed native-stage capacity model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.unified_terrain import UnifiedTerrainController, UnifiedTerrainBackend, PROFILE
from motionbricks.data.unified_training_goals import source_terminal_goal
from scripts.audit_unified_stair_gait import summarize
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def load_staged(a):
    saved = torch.load(a.pose, map_location='cpu', weights_only=False)
    if a.initial_baseline is not None:
        initial = torch.load(a.initial_baseline, map_location='cpu', weights_only=False)
        if (sha256(a.initial_baseline) != 'ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979'
                or initial['schema'] != 'motionbricks.unified_terrain.v1'
                or initial['contract']['bundle_sha256'] != sha256(a.bundle)
                or saved['schema'] != 'motionbricks.unified.native_route_curriculum.v1'
                or saved['contract']['bundle_sha256'] != sha256(a.bundle)):
            raise ValueError('Expected the preserved28941baseline and matched route specification')
        inference, bundle = load_terrain_bundle(a.bundle, a.device)
        controller = UnifiedTerrainController(inference, bundle)
        controller.adaptation.load_state_dict(initial['adaptation'], strict=True)
        controller.eval().requires_grad_(False)
        specification = dict(saved, contract=dict(saved['contract'], source_windows=saved['contract']['capacity_anchors']))
        return controller, specification, dict(contract=dict(cases_sha256=saved['contract']['cases_sha256']))
    if saved['schema'] == 'motionbricks.unified.native_route_curriculum.v1':
        report = json.loads(a.pose.with_name('complete.json').read_text())
        if (saved['contract']['stage'] != 'pose' or not report['training_bank_gate_pass']
                or saved['contract']['bundle_sha256'] != sha256(a.bundle)):
            raise ValueError('Whole-route native stages have not passed')
        inference, bundle = load_terrain_bundle(a.bundle, a.device)
        controller = UnifiedTerrainController(inference, bundle)
        controller.load_state_dict(saved['model_state'], strict=True)
        controller.eval().requires_grad_(False)
        review_saved = dict(saved, contract=dict(saved['contract'], source_windows=saved['contract']['capacity_anchors']))
        return controller, review_saved, dict(contract=dict(cases_sha256=saved['contract']['cases_sha256']))
    if a.codec is None:
        raise ValueError('Capacity-stage weights require their separately bound codec')
    codec = torch.load(a.codec, map_location='cpu', weights_only=False)
    if (saved['schema'] != 'motionbricks.unified.native_stage_capacity.v1'
            or saved['contract']['stage'] != 'pose' or saved['contract']['codec_sha256'] != sha256(a.codec)
            or saved['contract']['bundle_sha256'] != sha256(a.bundle)):
        raise ValueError('Native stage chain identity mismatch')
    report = json.loads(a.pose.with_name('complete.json').read_text())
    if not report['capacity_gate_pass']:
        raise ValueError('Native capacity stages have not passed')
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.codec.decoder.load_state_dict(codec['decoder'], strict=True)
    controller.root.load_state_dict(saved['root'], strict=True)
    controller.pose.load_state_dict(saved['pose'], strict=True)
    controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
    controller.eval().requires_grad_(False)
    return controller, saved, codec


def air_run(gap):
    longest = count = 0
    for value in (gap > .10).tolist():
        count = count + 1 if value else 0
        longest = max(longest, count)
    return longest


@torch.no_grad()
def support_diagnostics(geometry, world, scene):
    gaps = []
    for start in range(0, world.shape[1], 32):
        _, points = geometry.geometry(world[:, start:start + 32])
        gap, known = heightfield_gap(points[0, :, :geometry.sole_count], scene)
        gap = torch.where(known, gap, torch.nan).amin(-1)
        gaps.append(gap)
    clearance = torch.cat(gaps)
    finite = clearance[torch.isfinite(clearance)]
    return dict(minimum_sole_clearance_p95_m=float(torch.quantile(finite, .95)) if len(finite) else None,
        longest_both_feet_above_10cm_frames=air_run(clearance),
        foot_support_established=False)


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    controller, saved, codec = load_staged(a)
    rep = controller.inference.global_motion_rep
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    if sha256(a.cases) != codec['contract']['cases_sha256']:
        raise ValueError('Source cases differ from the capacity study')
    by_label = {c['label']: c for c in cases}
    idle = next(c for c in cases if c.get('name') == 'idle')
    geometry = JointGeometryObjective(rep, a.device)
    rows, counterfactuals = [], []

    def save(label, arm, world, scene, metadata):
        folder = a.output / 'recordings' / arm; folder.mkdir(parents=True, exist_ok=True)
        pose = rep.inverse(world, False, return_all=True, joint_positions_from='global_rot_data')
        name = f'{label}-{world.shape[1]}f'
        np.savez_compressed(folder / (name + '.npz'), local_rot_mats=pose['local_joint_rots'][0].cpu().numpy(),
            roots=world[0, :, :3].cpu().numpy(), heightfield_y_m=np.asarray(scene['heightfield_y_m']),
            min_point_world_xyz_m=np.asarray(scene['min_point_world_xyz_m']), dx_m=np.float32(scene['dx_m']),
            metadata_json=np.array(json.dumps(dict(metadata, role='unchanged_reference' if arm == 'source' else 'generated',
                actual_frames=world.shape[1], quality_approved=False, postprocessing=None))))
        return name

    for window in saved['contract']['source_windows']:
        case = by_label[window['label']]
        if case.get('entry') not in ('up', 'down'):
            continue
        scene = case['scene']; source = case['world'][None].to(a.device)
        command = source_terminal_goal(source[:, window['start']:window['start'] + 33, :5])
        start_xz, goal = source[0, 3, [0, 2]], source[0, -6, [0, 2]]
        axis = (goal - start_xz); length = float(axis.norm()); axis /= axis.norm()
        side = torch.stack((axis[1], -axis[0]))
        heading = float(torch.atan2(command[0, 3], command[0, 2]))
        flat_idle = rep.change_first_heading(idle['world'][None, :4].to(a.device), heading, False, False)
        old_floor = flat_idle[..., 1] - heightfield_gap(flat_idle[..., :3], idle['scene'])[0]
        flat_idle[..., [0, 2]] += start_xz - flat_idle[:, -1:, [0, 2]].clone()
        new_floor = flat_idle[..., 1] - heightfield_gap(flat_idle[..., :3], scene)[0]
        if float(new_floor.max() - new_floor.min()) > 1e-5:
            raise ValueError('Initial idle placement must be on a measured flat approach')
        flat_idle = shift_global_motion(flat_idle, rep, new_floor[:, -1] - old_floor[:, -1])
        reference_support = support_diagnostics(geometry, source, scene)
        save(case['label'], 'source', source, scene, dict(reference_support=reference_support))
        modes = [('source_start', source[:, :4], command[:, :2]),
                 ('trained_entry', source[:, window['start']:window['start'] + 4], command[:, :2]),
                 ('idle_same_command', flat_idle, command[:, :2]),
                 ('idle_0.8', flat_idle, axis[None] * .8), ('idle_1.6', flat_idle, axis[None] * 1.6)]
        if a.command_policy == 'source-window':
            if a.refresh_frames != 24:
                raise ValueError('Source-window timing control uses the original24frame publication')
            modes = modes[:1]
        for name, seed, velocity in modes:
            actor = UnifiedTerrainBackend(controller, seed.clone(), scene)
            actor.configure_runtime(dict(PROFILE, refresh_frames=a.refresh_frames))
            parts, failure = [seed.clone()], None
            schedule_exhausted = False
            command_log = []
            for block in range(60):
                active_heading = heading
                if a.command_policy == 'source-window':
                    source_start = (block // 3) * 24
                    if source_start + 33 > source.shape[1]:
                        schedule_exhausted = True
                        break
                    current_command = source_terminal_goal(source[:, source_start:source_start + 33, :5])
                    velocity = current_command[:, :2]
                    active_heading = float(torch.atan2(current_command[0, 3], current_command[0, 2]))
                command_log.append(dict(block=block, velocity_xz=velocity[0].tolist(), heading=active_heading))
                try:
                    prediction = actor.predict_plan(velocity[0].tolist(), active_heading)
                except (ValueError, RuntimeError) as error:
                    failure = repr(error); break
                parts.append(prediction['planned_motion'].clone())
                actor.history = prediction['published_history'].clone()
                if float((actor.history[0, -1, [0, 2]] - start_xz) @ axis) >= length:
                    break
            world = torch.cat(parts, 1)
            skin = exact_skin_metrics(geometry, world, scene)
            support = support_diagnostics(geometry, world, scene)
            progress = float((world[0, -1, [0, 2]] - start_xz) @ axis)
            lateral = float(((world[0, :, [0, 2]] - start_xz) @ side).abs().max())
            row = dict(case=case['label'], mode=name, actual_frames=world.shape[1], failure=failure,
                requested_velocity_xz=velocity[0].tolist(), goal_distance_m=length, final_progress_m=progress,
                reached_goal=progress >= length, lateral_m=lateral, skin=skin, support=support,
                reference_support=reference_support, quality_approved=False,
                review_budget_seconds=16, completion='goal_reached' if progress >= length else
                    ('source_schedule_exhausted' if schedule_exhausted else 'incomplete_in_review_budget'),
                output_speed_limits=None, command_policy=a.command_policy, command_log=command_log,
                refresh_frames=a.refresh_frames, offline_horizontal_command_schedule=a.command_policy == 'source-window')
            row['recording'] = save(case['label'] + '-' + name, name, world, scene, row)
            rows.append(row); write_json(a.output / 'progress.json', dict(rows=rows))
        for start in ((0, window['start']) if a.command_policy == 'constant' else ()):
            history = source[:, start:start + 4]
            actual = controller(history, command, scene)
            origin = vertical_origin(controller.inference, history, actual['observation'])
            floor = float(history[0, -1, 1] - heightfield_gap(history[:, -1:, :3], scene)[0][0, 0])
            plane = dict(scene, heightfield_y_m=torch.where(torch.isfinite(scene['heightfield_y_m']),
                torch.full_like(scene['heightfield_y_m'], floor), scene['heightfield_y_m']))
            flat = controller(history, command, plane)
            flat_origin = vertical_origin(controller.inference, history, flat['observation'])
            counterfactuals.append(dict(case=case['label'], start=start,
                origin_difference_m=float((origin - flat_origin).abs().max()),
                actual_rise_m=float(actual['world'][0, 27, 1] - history[0, 3, 1]),
                plane_rise_m=float(flat['world'][0, 27, 1] - history[0, 3, 1]),
                max_root_y_difference_m=float((actual['world'][:, 4:28, 1] - flat['world'][:, 4:28, 1]).abs().max()),
                caution='Same factual history, counterfactual full scene. Not a real flat reference trajectory.'))
    write_json(a.output / 'complete.json', dict(rows=rows, counterfactuals=counterfactuals,
        optimizer_updates=0, future_truth_in_actor=False, runtime_corrections=False, no_padding=True,
        pose_sha256=sha256(a.pose) if a.initial_baseline is None else None,
        initial_baseline_sha256=sha256(a.initial_baseline) if a.initial_baseline else None,
        route_specification_sha256=sha256(a.pose), codec_sha256=sha256(a.codec) if a.codec else None,
        full_model_includes_codec=a.codec is None, quality_approved=False,
        profile=dict(PROFILE, refresh_frames=a.refresh_frames), command_policy=a.command_policy,
        scope='Previously trained sources, true generated feedback, including observed flat idle starts. '
        'Finite16secondreview is not an output-speed gate. Extend slow but otherwise viable candidates separately. '
        'Support diagnostics are not a physical support certificate.'))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'pose', 'cases', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--codec', type=Path)
    p.add_argument('--initial-baseline', type=Path)
    p.add_argument('--refresh-frames', type=int, choices=(8, 24), default=24)
    p.add_argument('--command-policy', choices=('constant', 'source-window'), default='constant')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
