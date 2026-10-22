"""Route-completion review allowing natural stair slowdown, with original clocks."""

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_contact_preserve import load_model, prepare_cases, control, gait_metrics
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.audit_joint_sdf_stair_treadmill import measure as measure_step_progress


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for run in args.runs:
        arm, path = run.split('=', 1)
        checkpoint = Path(path)
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        model, _ = load_model(saved)
        rep = model.inference.global_motion_rep
        cases = prepare_cases(args.cases, Path(saved['contract']['dataset']))[:3]
        geometry = JointGeometryObjective(rep, 'cuda:0')
        directory = args.output / arm
        directory.mkdir()
        for case in cases:
            source = case['world'][None].cuda()
            history = source[:, :4].clone()
            initial = history[0, 3, :3].clone()
            goal = source[0, case['evaluation_goal_source_frame'] - case['start_frame'], :3]
            axis = goal[[0, 2]] - initial[[0, 2]]
            needed = float(axis.norm())
            axis /= axis.norm()
            side = torch.stack((-axis[1], axis[0]))
            state = SoftCommandState()
            parts, failure, reached = [], None, False
            maximum = math.ceil(args.seconds * 30 / 8)
            for block in range(maximum):
                request = state.from_controls(history, control(case, block), 8)
                try:
                    result = model(history, request, [case['scene']])
                except ValueError as error:
                    failure = str(error)
                    break
                parts.append(result['world'][:, 4:12])
                history = reencode_played_history(result['world'][:, 8:12], rep)
                reached = float((history[0, -1, [0, 2]] - goal[[0, 2]]) @ axis) >= 0
                if reached:
                    break
            row = dict(arm=arm, label=case['label'], frames=len(parts) * 8,
                seconds=len(parts) * 8 / 30, frame_limit=maximum * 8, goal_reached=reached,
                failure=failure, required_progress_m=needed, checkpoint_sha256=sha256(checkpoint),
                rejection_reasons=[], quality_approved=False, optimizer_updates=0,
                timing_policy='fixed15s safety horizon independent of requested speed; slowdown alone is not failure',
                same_time_pose_comparison=False, frame_padding=False, playback_retiming=False)
            if parts:
                world = torch.cat(parts, 1)
                filename = f'{case["label"]}-{arm}-{world.shape[1]}f.npz'
                save_motion(directory / filename, world, case['scene'], rep, row)
                progress = (world[0, :, [0, 2]] - initial[[0, 2]]) @ axis
                cross = ((world[0, :, [0, 2]] - initial[[0, 2]]) @ side).abs()
                row.update(progress_m=float(progress[-1]), cross_track_end_m=float(cross[-1]),
                    cross_track_peak_m=float(cross.max()), root_height_error_at_goal_m=abs(float(world[0, -1, 1] - goal[1])),
                    progress_clock_m=progress.tolist())
                row['four_second_stagnant_windows'] = int(((progress[120:] - progress[:-120]) < .05).sum()) if len(progress) > 120 else 0
                steps = torch.diff(torch.cat((initial[None, [0, 2]], world[0, :, [0, 2]])), dim=0).norm(dim=-1) * 30
                start_gap, _ = heightfield_gap(initial[None], case['scene'])
                gap, _ = heightfield_gap(world[0, :, :3], case['scene'])
                level = world[0, :, 1] - gap
                ascent = level > (initial[1] - start_gap[0] + .03)
                row['approach_speed_mps'] = float(steps[~ascent].mean()) if (~ascent).any() else None
                row['raised_terrain_speed_mps'] = float(steps[ascent].mean()) if ascent.any() else None
                row.update(gait_metrics(rep, world, axis))
                joints = rep.inverse(world, False, return_all=False,
                    joint_positions_from='global_rot_data')['posed_joints'][0]
                foot_ids = [rep.skeleton.bone_order_names.index(side + 'Foot') for side in ('Left', 'Right')]
                step_progress = measure_step_progress(world[0, :, :3].cpu().numpy(),
                    joints[:, foot_ids].cpu().numpy(), axis.cpu().numpy())
                row['step_progress'] = {key: value for key, value in step_progress.items() if key != 'series'}
                row.update(exact_skin_metrics(geometry, world, case['scene']))
                _, sole_points = geometry.geometry(world)
                sole_gap, known = heightfield_gap(sole_points[0, :, :geometry.sole_count], case['scene'])
                lowest = sole_gap.amin(-1)
                longest = current = 0
                for above in (lowest > .05).tolist():
                    current = current + 1 if above else 0
                    longest = max(longest, current)
                row['all_soles_above10cm_fraction'] = float((lowest > .1).float().mean())
                row['longest_all_soles_above5cm_seconds'] = longest / 30
                row['sole_vertical_p99_m'] = float(torch.quantile((-sole_gap[known]).clamp_min(0), .99)) if known.any() else None
                if not reached:
                    row['rejection_reasons'].append('route_not_completed_within_explicit_safety_horizon')
                if failure is not None:
                    row['rejection_reasons'].append('generation_failure')
                if row['body_prism_max_m'] > .02:
                    row['rejection_reasons'].append('penetration_over2cm')
                if row['unknown_body_fraction'] or row['uncertain_inside_samples']:
                    row['rejection_reasons'].append('unverified_terrain')
                if row['cross_track_end_m'] > .35:
                    row['rejection_reasons'].append('route_endpoint_lateral_error')
                if reached and row['root_height_error_at_goal_m'] > .2:
                    row['rejection_reasons'].append('wrong_height_at_route_endpoint')
                if longest >= 30:
                    row['rejection_reasons'].append('sustained_no_sole_near_surface')
                if step_progress['recurrent_backward_replanning']:
                    row['rejection_reasons'].append('recurrent_backward_replanning')
                row['local_route_screen_passed'] = not row['rejection_reasons']
                save_motion(directory / filename, world, case['scene'], rep, row)
            else:
                row['rejection_reasons'].append('no_frames_generated')
            rows.append(row)
            write_json(args.output / 'progress.json', dict(rows=rows, quality_approved=False))
            print(json.dumps({key: value for key, value in row.items() if key != 'progress_clock_m'}), flush=True)
        del model, saved, geometry
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / 'complete.json', dict(rows=rows, quality_approved=False, optimizer_updates=0,
        scope='Timing-policy correction requested by user. Collision/endpoint tolerances unchanged. '
              'Slower climbing alone is allowed; each recording retains its actual duration. '
              'A bound expiration is not proof of incapacity at arbitrarily slow speed.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=15.)
    main(parser.parse_args())
