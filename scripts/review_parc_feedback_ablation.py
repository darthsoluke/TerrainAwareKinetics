"""Held-out teacher-history vs played-history rollouts with native model futures."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import NativeMBMotionDataset
from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from motionbricks.data.native_mb_online_live import predict_planar_terrain
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.native_mb_online_conditions import planar_goal, velocity_goal, VELOCITY_PROTOCOL
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner
from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.sole_region_contact import sole_region_indices
from motionbricks.data.sole_geometry_losses import region_geometry_from_motion
from scripts.evaluate_parc_native_terrain import _dataset_windows
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import select_gpu, sha256


def terrain_gap(y, xz, scene):
    heights = torch.as_tensor(scene['heightfield_y_m'], device=y.device)
    origin = torch.as_tensor(scene['min_point_world_xyz_m'], device=y.device)
    spacing = torch.as_tensor(scene['dx_m'], device=y.device)
    cell = torch.floor((xz - origin[[0, 2]]) / spacing + .5).long()
    x, z = cell[..., 0], cell[..., 1]
    valid = (x >= 0) & (x < heights.shape[1]) & (z >= 0) & (z < heights.shape[0])
    ground = heights[z.clamp(0, heights.shape[0] - 1), x.clamp(0, heights.shape[1] - 1)]
    return y - ground, valid & torch.isfinite(ground)


def mean_or_none(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def evaluation_goal(inference, world, offset, protocol_version):
    """Derive the offline planar command through the same dual rep as runtime."""
    if protocol_version == VELOCITY_PROTOCOL['version']:
        window = world[:, offset:offset + 33]
        if window.shape[1] != 33:
            raise ValueError('Terminal velocity requires both interval endpoints')
        local = inference.motion_rep.dual_rep.global_to_local(
            window, False, False,
            lengths=torch.full((len(window),), 33, dtype=torch.long, device=window.device))
        return velocity_goal(local[:, 31, :4], world[:, offset + 31, :5])
    return planar_goal(world[:, offset + 31, :5])


@torch.inference_mode()
def evaluation_windows(dataset, split, max_windows):
    if split == 'validation':
        directory, windows, _, excluded, identity, _, _, _ = _dataset_windows(dataset, max_windows)
        return directory, windows, excluded, identity
    if split != 'test':
        raise ValueError('Only held-out validation/test splits are supported')
    dataset = Path(dataset)
    manifest = json.loads((dataset / 'native_manifest.json').read_text())
    accepted_path = dataset / 'accepted.json'
    if sha256(accepted_path) != manifest['artifact_sha256']['accepted.json']:
        raise ValueError('Split manifest changed')
    windows, excluded = [], []
    for record in json.loads(accepted_path.read_text()):
        if record['split'] != 'test' or record['terrain_family'] not in {'stairs', 'jumping', 'mid_climbing'}:
            continue
        path = (dataset / record['shard']).resolve()
        if dataset.resolve() not in path.parents or sha256(path) != manifest['artifact_sha256'][record['shard']]:
            raise ValueError('Test shard identity changed')
        with np.load(path) as data:
            frames = len(data['global_features'])
        if frames < 65:
            excluded.append({'shard': record['shard'], 'reason': 'shorter than 65 frames'})
            continue
        for start in sorted(set(np.linspace(0, frames - 65, 3).astype(int).tolist())):
            windows.append({'shard': record['shard'], 'shard_sha256': sha256(path),
                            'start': start, 'terrain_family': record['terrain_family'],
                            'source_group': record['source_group'], 'split': 'test'})
    if not windows:
        raise ValueError('No held-out test skill windows')
    return dataset, windows, excluded, {'dataset_manifest_sha256': sha256(dataset / 'native_manifest.json')}


@torch.inference_mode()
def evaluate(inference, bundle, dataset, output, max_windows=8, split='validation'):
    rep = inference.global_motion_rep
    directory, windows, excluded, identity = evaluation_windows(dataset, split, max_windows)
    if identity['dataset_manifest_sha256'] != bundle['metadata']['dataset_manifest_sha256']:
        raise ValueError('Dataset differs from the artifact binding')
    loader = NativeMBMotionDataset(directory, split, min_frames=65, max_frames=10**9,
                                  require_terrain=True, require_online_terrain=True, online_only=True)
    lookup = {row['shard']: i for i, row in enumerate(loader.records)}
    skin = move_skinner(load_official_skinner(rep.skeleton), torch.device(inference.device))
    skin.skeleton_skin.float().to(inference.device)
    surfaces, _, _, _ = official_sole_surfaces(rep.skeleton)
    indices = np.concatenate([sole_region_indices(surfaces, side) for side in ('Left', 'Right')])
    rows = []
    for wi, window in enumerate(windows):
        item = loader[lookup[window['shard']]]
        start = window['start']
        raw = rep.unnormalize(item['motion'][start:start + 65].to(inference.device))[None]
        source_root = item['terrain_root_world'][start:start + 65].to(inference.device)[None]
        heading = torch.atan2(source_root[:, 0, 4], source_root[:, 0, 3])
        world = rep.change_first_heading(raw, heading, False, False)
        world[..., [0, 2]] += source_root[:, :1, [0, 2]]
        source = world[:, 4:36]
        source_pose = rep.inverse(source, False, return_all=True)
        source_joints = source_pose['posed_joints'][0]
        sy, sxz = region_geometry_from_motion(source_pose['local_joint_rots'][0], source[0, :, :3], skin, indices)
        sgap, svalid = terrain_gap(sy, sxz, item['terrain_scene'])
        source_speed = (sxz[1:] - sxz[:-1]).norm(dim=-1) * 30
        # Read-only candidate evidence on the fixed same material vertex. This
        # diagnostic gate is not a claim of measured contact or training truth.
        close = (sgap.abs() <= .02) & svalid
        intervals = close[:-1] & close[1:] & (source_speed <= .1)
        interval_count = int(intervals.sum())
        for mode in ('teacher_history', 'played_history'):
            history, outputs, blocks = world[:, :4], [], []
            for k in range(4):
                offset = k * 8
                if mode == 'teacher_history':
                    history = world[:, offset:offset + 4]
                # Same explicit planar endpoint command in both cases; never Y.
                goal = evaluation_goal(inference, world, offset, bundle['protocol']['version'])
                prediction, _ = predict_planar_terrain(inference, history, goal, item['terrain_scene'],
                    num_tokens=8, config={'num_inference_step': 4, 'pose_token_sampling_use_argmax': True})
                future = prediction[:, 4:12]
                target = world[:, offset + 4:offset + 12]
                blocks.append({'block': k, 'root_y_error_m': float((future[..., 1] - target[..., 1]).abs().mean()),
                               'root_y_bias_m': float((future[..., 1] - target[..., 1]).mean())})
                outputs.append(future)
                history = reencode_played_history(future[:, -4:], rep)
            generated = torch.cat(outputs, 1)
            pose = rep.inverse(generated, False, return_all=True)
            joints = pose['posed_joints'][0]
            py, pxz = region_geometry_from_motion(pose['local_joint_rots'][0], generated[0, :, :3], skin, indices)
            pgap, pvalid = terrain_gap(py, pxz, item['terrain_scene'])
            p_speed = (pxz[1:] - pxz[:-1]).norm(dim=-1) * 30
            assessed = close & pvalid
            row = {'window': window, 'history': mode, 'blocks': blocks,
                   'fk_mpjpe_m': float((joints - source_joints).norm(dim=-1).mean()),
                   'root_y_mae_m': float((generated[..., 1] - source[..., 1]).abs().mean()),
                   'final_y_error_m': float((generated[:, -1, 1] - source[:, -1, 1]).abs().mean()),
                   'near_source_vertex_count': int(close.sum()), 'assessed_vertex_count': int(assessed.sum()),
                   'source_near_gap_abs_m': mean_or_none(sgap.abs(), close),
                   'predicted_near_gap_abs_m': mean_or_none(pgap.abs(), assessed),
                   'unassessed_near_vertices': int((close & ~pvalid).sum()),
                   'source_slow_interval_count': interval_count,
                   'source_slow_point_speed_mps': mean_or_none(source_speed, intervals),
                   'predicted_slow_point_speed_mps': mean_or_none(p_speed, intervals),
                   'penetration_mean_m': mean_or_none(torch.relu(-pgap), pvalid)}
            rows.append(row)
            np.savez_compressed(output / f'window-{wi:02d}-{mode}.npz',
                source=source_joints.cpu().numpy(), prediction=joints.cpu().numpy(),
                roots_source=source[0, :, :3].cpu().numpy(), local_source=source_pose['local_joint_rots'][0].cpu().numpy(),
                roots_prediction=generated[0, :, :3].cpu().numpy(), local_prediction=pose['local_joint_rots'][0].cpu().numpy(),
                hf=item['terrain_scene']['heightfield_y_m'].numpy(),
                min_point=item['terrain_scene']['min_point_world_xyz_m'].numpy(),
                dx=item['terrain_scene']['dx_m'].numpy(), fps=30)
    keys = ('fk_mpjpe_m', 'root_y_mae_m', 'final_y_error_m', 'predicted_near_gap_abs_m',
            'predicted_slow_point_speed_mps', 'penetration_mean_m')
    summary = {}
    for mode in ('teacher_history', 'played_history'):
        cases = [r for r in rows if r['history'] == mode]
        summary[mode] = {key: float(np.mean([r[key] for r in cases if r[key] is not None]))
                         if any(r[key] is not None for r in cases) else None for key in keys}
    result = {'schema': 'parc.feedback_ablation.v1', 'quality_approved': False,
              'scope': 'four consecutive eight-frame blocks; teacher-history comparator is offline only',
              'command_source': 'same expert hindsight planar goal under bound protocol; no future Y/pose',
              'source_near_gate': {'abs_gap_m': .02, 'material_vertex_speed_mps': .1,
                                  'measured_physical_contact': False},
              'scene_geometry': 'top of box heightfield; vertical hand contacts not assessed',
              'postprocessing': None, 'summary': summary, 'rows': rows, 'dataset': identity,
              'split': split, 'excluded': excluded}
    return result


@full_precision_evaluation()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'dataset', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--max-windows', type=int, default=8)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new review directory')
    gpu = None if args.cpu else select_gpu(args.gpu)
    torch.set_num_threads(4)
    inference, bundle = load_terrain_bundle(args.bundle, 'cpu' if args.cpu else 'cuda:0')
    args.output.mkdir(parents=True)
    report = evaluate(inference, bundle, args.dataset, args.output, args.max_windows, args.split)
    report.update(bundle_sha256=sha256(args.bundle), gpu_at_launch=gpu)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
