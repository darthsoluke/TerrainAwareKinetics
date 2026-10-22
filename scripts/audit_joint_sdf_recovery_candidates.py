"""Screen offline candidates against the factual published boundary and full skin."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_distance_volume import swept_surface_points
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def load_world(path):
    with np.load(path, allow_pickle=False) as values:
        return torch.from_numpy(values['world'].copy())[None].cuda()


@torch.no_grad()
def exact_penetration(points, scene, field):
    gap, known = heightfield_gap(points, scene)
    interior = known & (gap < 0)
    penetration = torch.zeros_like(gap)
    uncertain = 0
    if interior.any():
        distance, _, valid = field.query(points[interior])
        uncertain = int((~valid).sum())
        penetration[interior] = torch.where(valid, (-distance).clamp_min(0), -gap[interior])
    return penetration, known, uncertain


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    rep = build_native_representation(args.stats).to('cuda:0')
    geometry = JointGeometryObjective(rep, 'cuda:0')
    case = next(c for c in torch.load(args.cases, map_location='cpu', weights_only=False)
                if c['row']['source_id'] == 'generated_8030ae7a2a7ad640')
    source = case['world'][None, :84].cuda()
    scene = case['scene']
    field = HeightfieldPrismSDF(scene)
    actor = load_world(args.root / 'gait_stair_v2/run-0/generated_history-actor.npz')
    factual = torch.cat((source[:, :4], actor), 1)
    files = dict(actor=args.root / 'gait_stair_v2/run-0/generated_history-actor.npz',
        source_tokens=args.root / 'gait_stair_v2/run-0/generated_history-source_tokens.npz',
        source_pose_to_pose=args.root / 'pose_condition_v2/generated_history/source_pose_to_pose.npz',
        source_root_to_pose=args.root / 'pose_condition_v2/generated_history/source_root_to_pose.npz',
        source_root_and_pose_to_pose=args.root / 'pose_condition_v2/generated_history/source_root_and_pose_to_pose.npz')
    candidates = {key: load_world(path) for key, path in files.items()}
    candidates.update(real_source_control=source[:, 4:84], fixed_time_source_target=source[:, 4:84])
    names = rep.skeleton.bone_order_names
    feet = [names.index('LeftFoot'), names.index('RightFoot')]
    direction = source[0, 15, [0, 2]] - source[0, 3, [0, 2]]
    direction /= direction.norm()
    report = dict(optimizer_updates=0, quality_approved=False, accepted_training_labels=0,
        files={key: dict(path=str(path), sha256=sha256(path)) for key, path in files.items()}, arms={},
        scope='Each candidate is independently joined to the LAST FACTUAL played frame for measurement only. '
              'No motion is altered, no oracle fed back, no future is admitted as a recovery label. '
              'Source control uses its own history. Exact heightfield-prism distances on18056 skin vertices '
              'and8 linear sweep substeps are a kinematic screen, not force/dynamics or triangle-mesh CCD certification.')
    for label, candidate in candidates.items():
        if candidate.shape != actor.shape:
            raise ValueError('Every candidate must retain its actual80-frame recording')
        rows = []
        all_future_penetration = []
        for block in range(10):
            start = block * 8
            history = source[:, start:start + 4] if label == 'real_source_control' else factual[:, start:start + 4]
            future = candidate[:, start:start + 8]
            whole = torch.cat((history, future), 1)
            pose = rep.inverse(whole, False, return_all=True, joint_positions_from='global_rot_data')
            joints = pose['posed_joints'][0]
            joint_velocity = torch.diff(joints, dim=0) * 30
            boundary_acceleration = (joint_velocity[3] - joint_velocity[2]).norm(dim=-1) * 30
            skins = geometry.skin.skin(pose['local_joint_rots'][0, 3:],
                whole[0, 3:, None, :3].expand(-1, 30, -1), rot_is_global=False)
            swept = swept_surface_points(skins, 8)
            penetration, known, uncertain = exact_penetration(swept, scene, field)
            all_future_penetration.append(penetration[8:][known[8:]].cpu().numpy())
            first_step = future[0, 0, :3] - history[0, 3, :3]
            progress = future[0, -1, [0, 2]] - history[0, 3, [0, 2]]
            foot_separation = (joints[4:, feet[0], [0, 2]] - joints[4:, feet[1], [0, 2]]) @ direction
            rows.append(dict(block=block, first_published_frame=start,
                boundary_root_step_m=float(first_step.norm()),
                boundary_joint_speed_mean_mps=float(joint_velocity[3].norm(dim=-1).mean()),
                boundary_joint_speed_max_mps=float(joint_velocity[3].norm(dim=-1).max()),
                boundary_joint_acceleration_mean_mps2=float(boundary_acceleration.mean()),
                boundary_joint_acceleration_max_mps2=float(boundary_acceleration.max()),
                progress_along_command_m=float(progress @ direction),
                close_feet_fraction=float((foot_separation.abs() < .1).float().mean()),
                inherited_skin_peak_m=float(penetration[0].max()),
                boundary_skin_peak_m=float(penetration[:9].max()),
                future_skin_peak_m=float(penetration[8:].max()),
                future_skin_p99_m=float(torch.quantile(penetration[8:][known[8:]], .99)),
                unknown_samples=int((~known).sum()), uncertain_inside_samples=uncertain))
        values = np.concatenate(all_future_penetration)
        summary = dict(rows=rows, future_skin_peak_m=float(values.max()),
            future_skin_p99_m=float(np.quantile(values, .99)),
            boundary_joint_speed_max_mps=max(r['boundary_joint_speed_max_mps'] for r in rows),
            boundary_joint_speed_mean_mps=float(np.mean([r['boundary_joint_speed_mean_mps'] for r in rows])),
            boundary_joint_acceleration_max_mps2=max(r['boundary_joint_acceleration_max_mps2'] for r in rows),
            blocks_future_peak_over2cm=sum(r['future_skin_peak_m'] > .02 for r in rows),
            blocks_boundary_peak_over2cm=sum(r['boundary_skin_peak_m'] > .02 for r in rows),
            blocks_backward=sum(r['progress_along_command_m'] <= 0 for r in rows),
            unknown_samples=sum(r['unknown_samples'] for r in rows),
            uncertain_inside_samples=sum(r['uncertain_inside_samples'] for r in rows))
        report['arms'][label] = summary
        write_json(args.output / 'progress.json', report)
        print(json.dumps(dict(arm=label, **{k: v for k, v in summary.items() if k != 'rows'})), flush=True)
    write_json(args.output / 'complete.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
