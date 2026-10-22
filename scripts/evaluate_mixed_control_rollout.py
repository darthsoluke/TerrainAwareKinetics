"""Held-out native closed-loop rest/start/stop, with official SOMA sole geometry.

The flat test plane is declared at the unchanged BONES source Y=0. These are
accelerated simulation-clock rollouts, never wall-clock/FPS benchmarks. No
height projection, IK, pose smoothing or root output replacement is applied.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import NativeMBMotionDataset
from motionbricks.data.native_mb_online_live import NativeTerrainOnlineBackend
from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner
from scripts.portable_parc_online_soma import load_terrain_bundle, world_history
from scripts.train_native_mb import sha256


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seconds', type=int, default=60)
    args = p.parse_args()
    if args.output.exists() or args.seconds < 4:
        p.error('Use a new review directory and at least four seconds')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    data = NativeMBMotionDataset(args.dataset, 'validation', min_frames=65, max_frames=10**9,
                                 require_terrain=True, require_online_terrain=True, online_only=True)
    seeds = []
    for index, row in enumerate(data.records):
        if row.get('origin') != 'bones_seed' or row.get('control_category') != 'idle':
            continue
        item = data[index]
        roots = item['terrain_root_world'][:, :3]
        speed = (roots[1:]-roots[:-1]).norm(dim=-1) * 30
        if len(speed) < 60:
            continue
        score = speed.unfold(0, 60, 1).amax(-1)
        start = int(score.argmin())
        if float(score[start]) <= .08:
            seeds.append((row, item, start, float(score[start])))
        if len(seeds) == 2:
            break
    if len(seeds) < 2:
        raise ValueError('Two genuine held-out quiet BONES seeds are required')
    skinner = move_skinner(load_official_skinner(inference.motion_rep.skeleton), torch.device(args.device))
    skinner.skeleton_skin.float().to(args.device)
    surfaces, _, _, mesh_hash = official_sole_surfaces(inference.motion_rep.skeleton)
    region = {side: torch.tensor(sorted(set(np.concatenate([row['selected_indices'] for row in surfaces
        if row['channel'].startswith(side)]).tolist())), device=args.device) for side in ('Left', 'Right')}
    args.output.mkdir(parents=True)
    report = {'bundle_sha256': sha256(args.bundle), 'dataset_manifest_sha256': sha256(args.dataset/'native_manifest.json'),
        'mesh_sha256': mesh_hash, 'protocol': bundle['protocol'], 'clock': 'accelerated native 30 Hz',
        'post_processing': None, 'cases': [], 'quality_approved': False,
        'criteria': {'settle_seconds': 2, 'settled_drift_max_m': .1, 'settled_y_range_max_m': .1,
                     'settled_speed_p95_max_mps': .1, 'sole_penetration_p99_max_m': .03,
                     'median_higher_sole_gap_max_m': .05},
        'scope': 'flat held-out rest and start/stop only; not parkour or force validation'}
    for seed_index, (row, item, start, quiet_speed) in enumerate(seeds):
        for behavior in ('rest', 'start_stop'):
            history = world_history(inference.global_motion_rep, item, start).to(args.device)
            center = history[0, -1, :3]
            scene = {'heightfield_y_m': torch.zeros(401, 401, device=args.device),
                'min_point_world_xyz_m': torch.stack((center[0]-40, center.new_tensor(0.), center[2]-40)),
                'dx_m': torch.tensor(.2, device=args.device)}
            backend = NativeTerrainOnlineBackend(inference, history, scene)
            heading = float(torch.atan2(history[0, -1, 4], history[0, -1, 3]))
            forward = np.array([np.sin(heading), np.cos(heading)]) * 1.6
            warm_blocks = 12 if behavior == 'start_stop' else 0
            blocks = (args.seconds * 30 + 7) // 8
            motion_parts, command_parts = [], []
            for block in range(warm_blocks + blocks):
                command = forward if block < warm_blocks else [0., 0.]
                generated = backend.predict_plan(command, heading)
                motion = generated['planned_motion']
                motion_parts.append(motion.cpu())
                command_parts.extend([list(command)] * 8)
                backend.history = generated['published_history']
            motion = torch.cat(motion_parts, 1).to(args.device)
            pose = inference.motion_rep.inverse(motion, False, return_all=False)
            foot_gaps = []
            for offset in range(0, motion.shape[1], 64):
                vertices = skinner.skin(pose['global_joint_rots'][0, offset:offset+64],
                    pose['posed_joints'][0, offset:offset+64], rot_is_global=True)
                foot_gaps.append(torch.stack([vertices[:, indices, 1].amin(-1) for indices in region.values()], -1).cpu())
            gaps = torch.cat(foot_gaps).numpy()
            roots = motion[0, :, :3].cpu().numpy()
            stopped = warm_blocks * 8
            settled = stopped + 60
            stable_root, stable_gap = roots[settled:], gaps[settled:]
            drift = np.linalg.norm(stable_root[:, [0, 2]]-stable_root[:1, [0, 2]], axis=-1)
            metrics = {'settled_drift_max_m': float(drift.max()),
                'settled_y_range_m': float(np.ptp(stable_root[:, 1])),
                'settled_speed_p95_mps': float(np.percentile(np.linalg.norm(np.diff(stable_root, axis=0), axis=-1)*30, 95)),
                'sole_penetration_p99_m': float(np.percentile(np.maximum(-stable_gap, 0), 99)),
                'median_higher_sole_gap_m': float(np.median(stable_gap.max(-1))),
                'braking_displacement_m': float(np.linalg.norm(roots[settled, [0, 2]]-roots[stopped, [0, 2]]))}
            passed = (metrics['settled_drift_max_m'] <= .1 and metrics['settled_y_range_m'] <= .1
                and metrics['settled_speed_p95_mps'] <= .1 and metrics['sole_penetration_p99_m'] <= .03
                and metrics['median_higher_sole_gap_m'] <= .05)
            name = f'{seed_index}-{behavior}'
            np.savez_compressed(args.output / (name+'.npz'), fps=np.float32(30),
                global_features=motion[0].cpu().numpy(), root_positions=roots,
                local_rot_mats=pose['local_joint_rots'][0].cpu().numpy(), sole_gaps_m=gaps,
                command_world_xz_mps=np.asarray(command_parts, dtype=np.float32),
                heightfield_y_m=scene['heightfield_y_m'].cpu().numpy(),
                min_point_world_xyz_m=scene['min_point_world_xyz_m'].cpu().numpy(), dx_m=np.float32(.2))
            entry = {'name': name, 'source_shard': row['shard'], 'source_start': start,
                     'source_quiet_max_root_speed_mps': quiet_speed,
                     'metrics': metrics, 'status': 'pass' if passed else 'fail'}
            report['cases'].append(entry)
            (args.output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps(entry), flush=True)
    report['flat_stop_gate_passed'] = all(row['status'] == 'pass' for row in report['cases'])
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
