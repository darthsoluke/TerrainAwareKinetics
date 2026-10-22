"""Compare recorded gait and flat restart stability without changing motion."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.audit_unified_stair_gait import gait_features, summarize
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json


def dynamics(rep, world):
    motion = world[:, 4:]
    pose = gait_features(rep, motion)
    velocity = torch.diff(motion[..., :3], dim=1) * 30
    acceleration = torch.diff(velocity, dim=1) * 30
    feet = pose['feet']
    changes = torch.diff(feet, n=2, dim=1)
    return dict(mean_knee_flexion_deg=float(pose['knees'].mean()),
        root_acceleration_p99_mps2=float(torch.quantile(acceleration.norm(dim=-1), .99)),
        root_acceleration_max_mps2=float(acceleration.norm(dim=-1).max()),
        foot_relative_second_difference_rms_m=float(changes.square().mean().sqrt()),
        foot_foreaft_range_m=(torch.quantile(feet[..., 2], .95, dim=1) - torch.quantile(feet[..., 2], .05, dim=1))[0].tolist(),
        last_second_speed_mps=float(velocity[:, -30:, [0, 2]].norm(dim=-1).mean()))


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    inference, _ = load_terrain_bundle(args.bundle, 'cpu')
    rep = inference.global_motion_rep
    with np.load(args.scene, allow_pickle=False) as archive:
        scene = {k: torch.from_numpy(archive[k].copy()) for k in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
    rows, flat = [], []
    runs = [('before', args.adapter, 'baseline'), ('adapter', args.adapter, 'final'), ('capacity', args.capacity, 'final')]
    if args.comparison == 'coverage':
        runs = [('zero', args.adapter, 'baseline'), ('control', args.adapter, 'final'), ('coverage', args.capacity, 'final')]
    if args.comparison == 'joint':
        runs = [('zero', args.adapter, 'baseline'), ('control', args.adapter, 'final'), ('joint', args.capacity, 'final')]
    if args.reference_run:
        runs.insert(0, ('original', args.reference_run, 'baseline'))
    if args.mask:
        runs.append(('mask', args.mask, 'final'))
    for label, run, stage in runs:
        report = json.loads((run / 'complete.json').read_text())
        for row in report[stage]['live_stairs']:
            speed = row['speed_request_mps']
            with np.load(run / stage / 'live_stairs' / f'forward-{speed:g}.npz', allow_pickle=False) as archive:
                world = torch.from_numpy(archive['global_features'].copy())[None]
            ground = world[0, :, 1] - heightfield_gap(world[0, :, :3], scene)[0]
            stair_indices = torch.nonzero((ground > .08) & (ground < 1.15)).flatten()
            selected = world[:, stair_indices] if len(stair_indices) else world[:, :4]
            rows.append(dict(arm=label, speed_request_mps=speed, frames=world.shape[1],
                reached=row['reached_after_stair_goal'], full_skin_metrics=row['metrics'],
                on_stair_frames=len(stair_indices), gait=summarize(rep, selected, scene), dynamics=dynamics(rep, world)))
        for row in report[stage]['flat']:
            with np.load(run / stage / 'flat' / (row['case'] + '.npz'), allow_pickle=False) as archive:
                world = torch.from_numpy(archive['global_features'].copy())[None]
            result = dict(arm=label, case=row['case'], frames=world.shape[1], dynamics=dynamics(rep, world))
            if row['case'] == 'continuous-wasd-release':
                release = world[:, 52:100, [0, 2]]
                result['release48frames_displacement_m'] = float((release[:, -1] - release[:, 0]).norm())
                result['release_last8_speed_mps'] = float(torch.diff(release[:, -9:], dim=1).norm(dim=-1).mean() * 30)
            flat.append(result)
    write_json(args.output / 'complete.json', dict(live_rows=rows, flat_rows=flat, optimizer_updates=0,
        scope='Recorded kinematics only. Knee angle, foot range and acceleration are diagnostics, not physical contact certification.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'adapter', 'capacity', 'scene', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--mask', type=Path)
    p.add_argument('--comparison', choices=('capacity', 'coverage', 'joint'), default='capacity')
    p.add_argument('--reference-run', type=Path)
    main(p.parse_args())
