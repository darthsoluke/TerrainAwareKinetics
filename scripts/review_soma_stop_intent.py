"""Paired stop-goal semantics; raw predictions, no root/foot output correction."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.soma_lab_server import load_terrain_live_bundle
from motionbricks.data.native_mb_online_live import NativeTerrainOnlineBackend
from motionbricks.data import native_mb_online_live as live
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner
from motionbricks.data.bones_seed import official_sole_surfaces


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, scene, initial = load_terrain_live_bundle(args.bundle, 'cuda:0', args.scene)
    scene = {key: torch.as_tensor(value, device='cuda:0') for key, value in scene.items()}
    skinner = move_skinner(load_official_skinner(inference.motion_rep.skeleton), torch.device('cuda:0'))
    skinner.skeleton_skin.float().cuda()
    surfaces, _, _, _ = official_sole_surfaces(inference.motion_rep.skeleton)
    indices = torch.tensor(sorted(set(np.concatenate([s['selected_indices'] for s in surfaces]).tolist())), device='cuda:0')
    heading = float(torch.atan2(initial[0, -1, 4], initial[0, -1, 3]))
    forward = [1.6 * np.sin(heading), 1.6 * np.cos(heading)]
    report = {'quality_approved': False, 'output_corrections': None, 'cases': {}}
    original_goal = live.planar_command_goal
    for task in ('rest', 'move_then_stop'):
        report['cases'][task] = {}
        for held in (False, True):
            stationary = None
            def goal(history_root, velocity, heading, tokens, fps=30):
                nonlocal stationary
                value = original_goal(history_root, velocity, heading, tokens, fps)
                if float(torch.as_tensor(velocity).abs().sum()) == 0:
                    if stationary is None:
                        stationary = history_root[:, -1, [0, 2]].clone()
                    value[:, :2] = stationary
                else:
                    stationary = None
                return value
            # Experimental input-only ablation, deliberately not a live option.
            live.planar_command_goal = goal if held else original_goal
            backend = NativeTerrainOnlineBackend(inference, initial, scene)
            roots, gaps, anchor, goals = [], [], None, []
            for block in range(90):
                moving = task == 'move_then_stop' and block < 12
                if not moving and anchor is None:
                    anchor = backend.history[0, -1, :3].cpu().numpy().copy()
                result = backend.predict_plan(forward if moving else [0., 0.], heading)
                motion = result['planned_motion']
                if not moving:
                    roots.append(motion[0, :, :3].cpu().numpy())
                    pose = inference.motion_rep.inverse(motion, False, return_all=False)
                    points = skinner.skin(pose['global_joint_rots'][0], pose['posed_joints'][0],
                                         rot_is_global=True)[:, indices]
                    cell = torch.floor((points[..., [0, 2]] - scene['min_point_world_xyz_m'][[0, 2]]) / scene['dx_m'] + .5).long()
                    hf = scene['heightfield_y_m']
                    known = (cell[..., 0] >= 0) & (cell[..., 0] < hf.shape[1]) & (cell[..., 1] >= 0) & (cell[..., 1] < hf.shape[0])
                    heights = hf[cell[..., 1].clamp(0, hf.shape[0]-1), cell[..., 0].clamp(0, hf.shape[1]-1)]
                    gap = torch.where(known, points[..., 1]-heights, torch.nan)
                    gaps.append(gap.cpu().numpy())
                    goals.append(result['debug']['input_intent']['endpoint_world_xz_m'])
                # Offline accelerated native playback, all eight frames committed.
                backend.history = result['published_history']
            trajectory, gap = np.concatenate(roots), np.concatenate(gaps)
            settled = trajectory[60:]
            report['cases'][task]['held' if held else 'receding'] = {
                'seconds': len(trajectory)/30, 'anchor': anchor.tolist(),
                'final_xz_error_m': float(np.linalg.norm(trajectory[-1, [0, 2]]-anchor[[0, 2]])),
                'max_xz_error_m': float(np.linalg.norm(trajectory[:, [0, 2]]-anchor[[0, 2]], axis=-1).max()),
                'settled_root_y_range_m': float(np.ptp(settled[:, 1])),
                'settled_root_speed_p95_mps': float(np.percentile(np.linalg.norm(np.diff(settled, axis=0), axis=-1)*30, 95)),
                'sole_penetration_p99_m': float(np.nanpercentile(np.maximum(-gap, 0), 99)),
                'root_world_xyz_m': trajectory.tolist(), 'submitted_goal_xz': goals}
    live.planar_command_goal = original_goal
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({task: {name: {k:v for k,v in row.items() if k not in ('root_world_xyz_m', 'submitted_goal_xz')}
                            for name,row in rows.items()} for task,rows in report['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()
