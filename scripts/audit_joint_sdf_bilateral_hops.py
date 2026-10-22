"""Replay the reported comparison clip and separate gait collapse from block jumps."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.evaluate_joint_sdf import save_motion


def series_metrics(rep, world, initial, direction):
    positions = rep.inverse(world, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints'][0]
    names = rep.skeleton.bone_order_names
    left, right = (names.index(side + 'Foot') for side in ('Left', 'Right'))
    feet = positions[:, [left, right]]
    roots = world[0, :, :3]
    displacement = torch.diff(torch.cat((initial[0, -1:, :3], roots)), dim=0)
    speed = displacement[:, [0, 2]] @ direction * 30
    relative_feet = feet - roots[:, None]
    foot_foreaft = relative_feet[..., [0, 2]] @ direction
    foreaft_separation = foot_foreaft[:, 0] - foot_foreaft[:, 1]
    seam = torch.arange(len(roots), device=roots.device) % 8 == 0
    vertical_change = torch.diff(relative_feet[:, :, 1], dim=0)
    correlation = np.corrcoef(vertical_change.cpu().numpy().T)[0, 1]
    def group(mask):
        return {'root_planar_step_mean_m': float(displacement[mask][:, [0, 2]].norm(dim=-1).mean()),
                'root_planar_step_max_m': float(displacement[mask][:, [0, 2]].norm(dim=-1).max()),
                'forward_speed_mean_mps': float(speed[mask].mean()), 'forward_speed_max_mps': float(speed[mask].max())}
    knee_angles = []
    for side in ('Left', 'Right'):
        ids = [names.index(side + joint) for joint in ('Leg', 'Shin', 'Foot')]
        leg = positions[:, ids]
        upper, lower = leg[:, 0] - leg[:, 1], leg[:, 2] - leg[:, 1]
        angle = torch.rad2deg(torch.acos(((upper * lower).sum(-1) / (upper.norm(dim=-1) * lower.norm(dim=-1))).clamp(-1, 1)))
        knee_angles.append(180 - angle)
    knees = torch.stack(knee_angles, -1)
    summary = {'block_boundary': group(seam), 'inside_block': group(~seam),
               'left_right_relative_foot_vertical_change_correlation': float(correlation) if np.isfinite(correlation) else None,
               'foot_foreaft_separation_rms_m': float(foreaft_separation.square().mean().sqrt()),
               'close_foreaft_feet_under10cm_fraction': float((foreaft_separation.abs() < .1).float().mean()),
               'mean_knee_flexion_deg': float(knees.mean()),
               'max_root_acceleration_mps2': float(torch.diff(displacement * 30, dim=0).norm(dim=-1).max() * 30)}
    series = {'forward_speed_mps': speed.tolist(), 'root_y_m': roots[:, 1].tolist(),
              'foot_foreaft_relative_m': foot_foreaft.tolist(),
              'foot_height_relative_to_root_m': relative_feet[:, :, 1].tolist(),
              'knee_flexion_deg': knees.tolist(), 'root_step_xyz_m': displacement.tolist()}
    return summary, series


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    bundle_path = Path(saved['contract']['source'])
    if sha256(bundle_path) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source bundle identity mismatch')
    inference, bundle = load_terrain_bundle(bundle_path, 'cuda:0')
    model = JointSDFController(inference, bundle)
    model.load_state_dict(saved['model_state'], strict=True)
    model.requires_grad_(False)
    model.eval()
    rep = inference.global_motion_rep
    case = next(case for case in torch.load(args.cases, map_location='cpu', weights_only=False)
                if case['row']['source_id'] == 'generated_8030ae7a2a7ad640')
    source = case['world'][None].cuda()
    if args.flat_scene:
        gap, known = heightfield_gap(source[0, 3:4, :3], case['scene'])
        if not known.all():
            raise ValueError('Counterfactual flat control needs measured initial support')
        height = float(source[0, 3, 1] - gap[0])
        center = source[0, 3, [0, 2]].cpu()
        case = dict(case, scene={'heightfield_y_m': torch.full((201, 201), height),
                    'min_point_world_xyz_m': torch.tensor([float(center[0]) - 50., 0., float(center[1]) - 50.]),
                    'dx_m': .5})
    initial = source[:, :4].clone()
    history = initial.clone()
    state = SoftCommandState()
    edge = case['edge_frame']
    velocity = (source[0, edge - 8, [0, 2]] - source[0, 3, [0, 2]]) / ((edge - 11) / 30)
    heading = torch.atan2(source[:, 3:edge - 8, 4].mean(1), source[:, 3:edge - 8, 3].mean(1))
    direction = velocity / velocity.norm()
    parts, local_parts, rows = [], [], []
    prior = None
    failure = None
    for block in range(10):
        task = state.intent(history, velocity, heading, 8)
        try:
            result = model(history, task, [case['scene']])
        except ValueError as error:
            failure = str(error)
            break
        generated = result['world']
        local = inference.local_motion_rep.unnormalize(model.last_local[:, :32])
        predicted_velocity = local[:, :, inference.local_motion_rep.indices['local_root_vel']]
        if rep.removed_heading or predicted_velocity.shape[-1] != 2:
            raise ValueError('Audit requires the current native world-XZ velocity representation')
        observed_displacement = history[:, 3, [0, 2]] - history[:, 0, [0, 2]]
        integrated_past = predicted_velocity[:, :3].sum(1) / 30
        prefix_error = generated[:, 3, [0, 2]] - history[:, 3, [0, 2]]
        torch.testing.assert_close(prefix_error, integrated_past - observed_displacement, atol=1e-5, rtol=0)
        first_step = generated[:, 4, [0, 2]] - history[:, 3, [0, 2]]
        first_velocity_step = predicted_velocity[:, 3] / 30
        torch.testing.assert_close(first_step, prefix_error + first_velocity_step, atol=1e-5, rtol=0)
        row = {'block': block, 'first_played_frame': block * 8,
               'decoded_past_xz_offset_m': prefix_error[0].tolist(),
               'decoded_past_xz_offset_norm_m': float(prefix_error.norm()),
               'decoded_past_y_error_m': float(generated[0, 3, 1] - history[0, 3, 1]),
               'published_boundary_step_m': first_step[0].tolist(),
               'model_boundary_velocity_step_m': first_velocity_step[0].tolist(),
               'past_offset_along_command_m': float(prefix_error[0].dot(direction)),
               'published_step_along_command_m': float(first_step[0].dot(direction)),
               'boundary_velocity_along_command_mps': float(predicted_velocity[0, 3].dot(direction)),
               'observed_history_velocity_rms_mps': float((torch.diff(history[0, :, [0, 2]], dim=0) * 30).square().sum(-1).mean().sqrt()),
               'decoded_history_velocity_error_rms_mps': float((predicted_velocity[:, :3] - torch.diff(history[:, :, [0, 2]], dim=1) * 30).square().sum(-1).mean().sqrt())}
        if prior is not None:
            previous_joints = rep.inverse(prior[:, 12:20], False, return_all=False,
                                          joint_positions_from='global_rot_data')['posed_joints']
            revised_joints = rep.inverse(generated[:, 4:12], False, return_all=False,
                                         joint_positions_from='global_rot_data')['posed_joints']
            row['same_time_forecast_revision_mpjpe_m'] = float((previous_joints - revised_joints).norm(dim=-1).mean())
        rows.append(row)
        parts.append(generated[:, 4:12])
        local_parts.append(local)
        prior = generated.clone()
        history = reencode_played_history(generated[:, 8:12], rep)
    if not parts:
        write_json(args.output / 'complete.json', {'failure': failure, 'frames': 0, 'requested_frames': 80,
                                                  'quality_approved': False})
        return
    motion = torch.cat(parts, dim=1)
    with np.load(args.recording, allow_pickle=False) as archive:
        expected = torch.from_numpy(archive['world']).to(motion)
    if args.require_recording_parity:
        torch.testing.assert_close(motion[0], expected, atol=3e-5, rtol=1e-5)
    actual_frames = motion.shape[1]
    source_summary, source_series = series_metrics(rep, source[:, 4:4 + actual_frames], initial, direction)
    measured = expected[None] if args.require_recording_parity else motion
    generated_summary, generated_series = series_metrics(rep, measured, initial, direction)
    report = {'source': source_summary, 'generated': generated_summary, 'blocks': rows,
              'series': {'source': source_series, 'generated': generated_series},
              'recording_replay_max_difference': float((motion[0] - expected[:actual_frames]).abs().max()),
              'recording_root_replay_max_difference_m': float((motion[0, :, :3] - expected[:actual_frames, :3]).abs().max()),
              'same_checkpoint_replay_required': args.require_recording_parity,
              'counterfactual_flat_scene': args.flat_scene,
              'source_reference_scene_matches_generation': not args.flat_scene,
              'failure': failure, 'frames': actual_frames, 'requested_frames': 80,
              'command_velocity_mps': velocity.tolist(), 'checkpoint_sha256': sha256(args.checkpoint),
              'quality_approved': False, 'optimizer_updates': 0,
              'scope': ('Same-checkpoint replay checked with floating-point tolerance; gait metrics use the saved recording. '
                        if args.require_recording_parity else
                        'Alternative-checkpoint generation with the reported clip initial state and identical constant command. ')
                       + 'Past-velocity decomposition '
                       'does not alter/integrate a replacement motion. Foot/knee statistics describe gait; '
                       'they are not force contacts or an automatic naturalness pass.'}
    write_json(args.output / 'complete.json', report)
    save_motion(args.output / 'reported-stairs.npz', motion, case['scene'], rep, report)
    if not args.flat_scene:
        save_motion(args.output / 'reported-stairs-source.npz', source[:, 4:4 + actual_frames], case['scene'], rep,
                    {'role': 'unchanged_reference', 'quality_approved': False})
    np.savez_compressed(args.output / 'decoder-local-windows.npz', local=torch.cat(local_parts).cpu().numpy())
    print(json.dumps({key: value for key, value in report.items() if key not in ('series',)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-recording-parity', action='store_true')
    parser.add_argument('--flat-scene', action='store_true')
    main(parser.parse_args())
