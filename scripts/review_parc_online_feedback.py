"""Paired causal Root/Pose rollouts, separating raw feature and played-FK history."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.portable_parc_online_soma import load_terrain_bundle
from motionbricks.data.native_mb_online_live import NativeTerrainOnlineBackend
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.bones_contract import file_sha256


@torch.inference_mode()
def rollout(inference, bundle, *, observed_fk, blocks=24, native_timing=False):
    scene = {key: torch.as_tensor(value, device=inference.device) for key, value in bundle['scene'].items()}
    backend = NativeTerrainOnlineBackend(inference, bundle['initial_history'], scene,
                                         reencode_observed_history=observed_fk)
    if native_timing:
        backend.inference_config['online_predict_duration'] = True
    rows, prior = [], None
    for index in range(blocks):
        command = [0., 0.] if index < 6 else [0., 1.6]
        history = backend.history.clone()
        result = backend.predict_plan([command], [0.])
        motion = result['planned_motion']
        roots = motion[0, :, :3].cpu().numpy()
        heights = np.asarray(scene['heightfield_y_m'].cpu())
        origin = np.asarray(scene['min_point_world_xyz_m'].cpu())
        spacing = float(scene['dx_m'])
        cell = np.floor((roots[:, [0, 2]] - origin[[0, 2]]) / spacing + .5).astype(int)
        valid = (cell[:, 0] >= 0) & (cell[:, 0] < heights.shape[1]) & (cell[:, 1] >= 0) & (cell[:, 1] < heights.shape[0])
        gaps = [float(point[1] - heights[z, x]) if known else None
                for point, (x, z), known in zip(roots, cell, valid)]
        diagnostic = {}
        observation = reencode_played_history(motion[:, -4:], inference.global_motion_rep, diagnostic)
        original_ric = inference.global_motion_rep.inverse(history, False, joint_positions_from='ric_data',
                                                         return_all=False)['posed_joints']
        actual_fk = inference.global_motion_rep.inverse(history, False, joint_positions_from='global_rot_data',
                                                       return_all=False)['posed_joints']
        root_trace = result.get('debug', {}).get('root_stage_world_xyz_m', [])
        rows.append({'block': index + 1, 'command': command,
            'neural_generation': backend.last_generation['neural_generation'],
            'predicted_frames': backend.last_generation.get('predicted_frames'),
            'context_ric_fk_mean_m': float((original_ric - actual_fk).norm(dim=-1).mean()),
            'played_ric_fk_mean_m': diagnostic['original_ric_fk_mean_error_m'],
            'physical_state_reencoding_max_error_m': diagnostic['physical_state_reencoding_max_error_m'],
            'root_xyz_m': roots.tolist(),
            'pelvis_ground_gap_m': gaps,
            'root_stage_y_m': [point[1] for point in root_trace[:8]],
            'boundary_root_step_m': None if prior is None else float(np.linalg.norm(roots[0] - prior)),
            'generation_ms': result['generation_seconds'] * 1000})
        # All eight candidate frames are now considered played on the native clock.
        backend.history = observation if observed_fk else motion[:, -4:].clone()
        prior = roots[-1]
    roots = np.concatenate([np.asarray(row['root_xyz_m']) for row in rows])
    boundaries = [row['boundary_root_step_m'] for row in rows if row['boundary_root_step_m'] is not None]
    known_gaps = [gap for row in rows for gap in row['pelvis_ground_gap_m'] if gap is not None]
    return {'history_mode': 'actual_played_fk' if observed_fk else 'raw_decoder_features',
        'clock': 'offline accelerated native 30 Hz steps; not wall-clock performance',
        'initial_y_m': float(bundle['initial_history'][0, -1, 1]),
        'final_y_m': float(roots[-1, 1]), 'max_y_m': float(roots[:, 1].max()),
        'max_boundary_root_step_m': max(boundaries),
        'max_known_pelvis_ground_gap_m': max(known_gaps) if known_gaps else None,
        'mean_context_ric_fk_error_m': float(np.mean([r['context_ric_fk_mean_m'] for r in rows])),
        'rows': rows, 'quality_approved': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--blocks', type=int, default=24)
    parser.add_argument('--scene', choices=['source', 'flat'], default='source')
    parser.add_argument('--native-timing', action='store_true')
    parser.add_argument('--context4-diagnostic', action='store_true',
                        help='Offline constraint-coverage ablation; not a deployment approval')
    args = parser.parse_args()
    if args.output.exists() or args.blocks < 7:
        parser.error('Use a new output file and at least seven blocks')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    if args.context4_diagnostic:
        original_decoder = inference._vqvae_pose_model.forward_decoder
        def observed_context_decoder(*values, **kwargs):
            mask = kwargs['has_target_cond'].clone()
            mask[:, :4] = True
            kwargs['has_target_cond'] = mask
            return original_decoder(*values, **kwargs)
        inference._vqvae_pose_model.forward_decoder = observed_context_decoder
    if args.scene == 'flat':
        bundle = dict(bundle, scene={'heightfield_y_m': torch.zeros(201, 201),
            'min_point_world_xyz_m': torch.tensor([-10., 0., -10.]), 'dx_m': torch.tensor(.2)})
    result = {'bundle_sha256': file_sha256(args.bundle), 'metadata': bundle['metadata'],
        'test_scene': args.scene, 'context4_diagnostic': args.context4_diagnostic, 'native_timing': args.native_timing,
        'future_height_source': 'model', 'future_pose_source': 'model',
        'ik_smoothing_root_overwrite': False,
        'raw': rollout(inference, bundle, observed_fk=False, blocks=args.blocks, native_timing=args.native_timing),
        'observed': rollout(inference, bundle, observed_fk=True, blocks=args.blocks, native_timing=args.native_timing),
        'quality_approved': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({name: {key: value for key, value in result[name].items() if key != 'rows'}
                     for name in ('raw', 'observed')}, indent=2))


if __name__ == '__main__':
    main()
