"""Held-out native MB generation from history and planar goals, not full endpoints."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_native_mb_control import load_control_checkpoint
from scripts.evaluate_parc_native_terrain import (
    _dataset_windows, _metric_arrays, _world_transform, _window_world_transform,
    SCORE_START, SCORE_STOP,
)
from motionbricks.data.native_mb_data import NativeMBMotionDataset, collate_native_batch
from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from motionbricks.data.native_mb_online_conditions import (
    ONLINE_PROTOCOL, sparse_planar_conditions, planar_goal, goal_direction_local,
)
from motionbricks.data.native_mb_online_terrain import gather_online_terrain_observation
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from motionbricks.motion_backbone.inference.motion_inference import motion_inference


def stage_metrics(rep, source, prediction, predicted_root):
    """Read-only decomposition; no traced tensor becomes an inference condition."""
    source_root = extract_feature_from_motion_rep(source, rep, 'root')
    decoded_root = extract_feature_from_motion_rep(prediction, rep, 'root')
    score = slice(SCORE_START, SCORE_STOP)
    difference = predicted_root[:, score, :3] - source_root[:, score, :3]
    decoded_difference = decoded_root[:, score, :3] - predicted_root[:, score, :3]
    return {
        'root_stage_xyz_error_m': float(difference.norm(dim=-1).mean()),
        'root_stage_xz_error_m': float(difference[..., [0, 2]].norm(dim=-1).mean()),
        'root_stage_y_error_m': float(difference[..., 1].abs().mean()),
        'decoder_vs_root_xyz_m': float(decoded_difference.norm(dim=-1).mean()),
        'decoder_vs_root_y_m': float(decoded_difference[..., 1].abs().mean()),
        'root_stage_endpoint_xz_error_m': float(
            (predicted_root[:, 63, [0, 2]] - source_root[:, 63, [0, 2]]).norm(dim=-1).mean()),
        'decoded_endpoint_xz_error_m': float(
            (decoded_root[:, 63, [0, 2]] - source_root[:, 63, [0, 2]]).norm(dim=-1).mean()),
        'decoded_current_frame_root_error_m': float(
            (decoded_root[:, 3, :3] - source_root[:, 3, :3]).norm(dim=-1).mean()),
    }


@full_precision_evaluation()
def evaluate(dataset, root_run, pose_run, output, device='cuda', max_windows=8,
             inference_steps=1, terrain_mode='observed', bundle_path=None, decoder_review=False):
    if (not isinstance(inference_steps, int) or isinstance(inference_steps, bool)
            or not 1 <= inference_steps <= 16 or terrain_mode not in {'observed', 'invalid'}):
        raise ValueError('Use 1..16 native iterations and observed/invalid terrain')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    portable_metadata = None
    if bundle_path is not None:
        from scripts.portable_parc_online_soma import load_terrain_bundle
        infer, portable = load_terrain_bundle(bundle_path, device)
        root, pose = infer._root_model, infer._pose_model
        portable_metadata = portable['metadata']
        root_result = {'final_checkpoint_sha256': portable_metadata['root_checkpoint_sha256']}
        pose_result = {'final_checkpoint_sha256': portable_metadata['pose_checkpoint_sha256']}
        pose_contract = {'dataset_manifest_sha256': portable_metadata['dataset_manifest_sha256'],
                         'online_contract': {'root_dependency': portable_metadata['pose_root_dependency']}}
    else:
        root, root_contract, root_result = load_control_checkpoint(root_run, device)
        pose, pose_contract, pose_result = load_control_checkpoint(pose_run, device)
    if root.args.get('online_conditioning') != ONLINE_PROTOCOL['version'] \
            or pose.args.get('online_conditioning') != ONLINE_PROTOCOL['version']:
        raise ValueError('Evaluation requires both online-conditioned checkpoints')
    dependency = pose_contract['online_contract']['root_dependency']
    if dependency['checkpoint_sha256'] != root_result['final_checkpoint_sha256']:
        raise ValueError('Pose was not trained with this Root prediction dependency')
    if bundle_path is None:
        infer = motion_inference({'root': root, 'pose': pose}, pose.args, device=device)
    directory, windows, _, excluded, identity, _, _, _ = _dataset_windows(Path(dataset), max_windows)
    if identity['dataset_manifest_sha256'] != pose_contract['dataset_manifest_sha256']:
        raise ValueError('Held-out dataset must match checkpoint normalization and provenance')
    dataset_loader = NativeMBMotionDataset(directory, 'validation', max_frames=10**9, min_frames=65,
                require_terrain=True, require_online_terrain=True, cache_max_bytes=256 * 1024**2)
    indices = {r['shard']: i for i, r in enumerate(dataset_loader.records)}
    rep = infer.global_motion_rep
    rows = []
    with torch.inference_mode():
        for number, window in enumerate(windows):
            item = dataset_loader[indices[window['shard']]]
            batch = collate_native_batch([item])
            start = window['start']
            raw = rep.unnormalize(item['motion'][start:start + 65].to(device))[None]
            local = infer.motion_rep.dual_rep.global_to_local(raw, False, False,
                                                lengths=torch.tensor([65], device=device))[:, :64]
            roots = raw[..., :5]
            goal = planar_goal(roots[:, 63])
            conditions = sparse_planar_conditions(roots[:, :4], local[:, :4, :4],
                extract_feature_from_motion_rep(local[:, :4], infer.local_motion_rep,
                                               infer.EXTERNAL_POSE_FEATURE_MODE), goal)
            scene = gather_online_terrain_observation(batch,
                {'chosen_ids': torch.tensor([0]), 'frame_indices': torch.arange(start, start + 65)[None]},
                goal_direction_local(roots[:, :4], goal).cpu())
            scene = {k: v.to(device) for k, v in scene.items()}
            observed_points = int(scene['valid'].sum())
            if terrain_mode == 'invalid':
                scene['valid'] = torch.zeros_like(scene['valid'])
            traced = {}

            def capture_root(_module, _inputs, result):
                traced['root'] = result['pred_global_root_values'].detach().clone()

            hook = root.backbone_net.register_forward_hook(capture_root)
            try:
                from contextlib import nullcontext
                from scripts.review_parc_decoder_height_response import capture_decoder_call
                capture = capture_decoder_call(infer._vqvae_pose_model) if decoder_review else nullcontext()
                with capture as decoder_calls:
                    prediction, tokens = infer.predict(
                        conditions['global_root_values'], conditions['has_global_root_values'],
                        conditions['local_root_values'], conditions['has_local_root_values'],
                        conditions['local_poses'], conditions['has_local_poses'], 16,
                        global_root_component_mask=conditions['global_root_component_mask'], terrain_observation=scene,
                        config={'num_inference_step': inference_steps, 'pose_token_sampling_use_argmax': True,
                                'allow_pred_out_of_reach_num_tokens': False}, info={})
            finally:
                hook.remove()
            if int(tokens.item()) != 16 or not torch.isfinite(prediction).all():
                raise ValueError('Invalid online native generation')
            offline_decoder = None
            if decoder_review:
                from motionbricks.data.native_mb_decoder_review import decoder_factorization
                if len(decoder_calls) != 1:
                    raise ValueError('Expected one actual native decoder call')
                offline_decoder = decoder_factorization(infer, raw[:, :64], local,
                                                        decoder_calls[0], prediction)
            metrics, _ = _metric_arrays(rep, rep, raw[:, :64], prediction, None)
            predicted_root = rep.unnormalize(traced['root'][:, :64])
            if pose.args.get('vertical_frame'):
                from motionbricks.data.native_mb_vertical_frame import center_motion_window
                _, origin_y = center_motion_window(raw, rep, normalized=False)
                predicted_root[..., 1] += origin_y[:, None]
            # Native Root retains absolute heading/Y and only recenters XZ.
            predicted_root[..., [0, 2]] += roots[:, :1, [0, 2]]
            diagnostics = stage_metrics(rep, raw[:, :64], prediction, predicted_root)
            world_info, _ = _world_transform(directory, window, device)
            if world_info is None:
                raise ValueError('Online review requires measured world transform')
            transform = _window_world_transform(rep, raw, raw[:, :64], world_info)
            source = rep.inverse(raw[:, :64], False, return_all=True)
            pred = rep.inverse(prediction, False, return_all=True)
            from scripts.review_parc_tokenizer_terrain import transform_motion_to_world
            source_roots, source_local, _ = transform_motion_to_world(
                source['posed_joints'][0].cpu().numpy(), source['local_joint_rots'][0].cpu().numpy(),
                transform, int(rep.skeleton.root_idx))
            pred_roots, pred_local, _ = transform_motion_to_world(
                pred['posed_joints'][0].cpu().numpy(), pred['local_joint_rots'][0].cpu().numpy(),
                transform, int(rep.skeleton.root_idx))
            geometry = item['terrain_scene']
            example = f'online-{number:03d}.npz'
            np.savez_compressed(output / example, roots_source=source_roots, local_source=source_local,
                roots_prediction=pred_roots, local_prediction=pred_local,
                root_stage_canonical=predicted_root[0].cpu().numpy(),
                goal_xz_heading_canonical=goal[0].cpu().numpy(),
                hf=geometry['heightfield_y_m'].numpy(),
                min_point=geometry['min_point_world_xyz_m'].numpy(), dx=geometry['dx_m'].numpy(), fps=30)
            rows.append({'window': window, 'example': example, 'valid_points': int(scene['valid'].sum()),
                         'observed_points': observed_points, 'stage_metrics': diagnostics,
                         'metrics': metrics, 'future_y_provided': False, 'future_pose_provided': False,
                         'offline_decoder_factorization': offline_decoder})
    if not rows:
        raise ValueError('No held-out online contract windows')
    result = {'schema': 'parc.online_planar_review.v2', 'protocol': ONLINE_PROTOCOL,
        'task': 'held-out hindsight planar goals; not a WASD closed-loop traversal acceptance',
        'quality_approved': False, 'postprocessing': 'none', 'root_overwrite': False,
        'dataset': identity, 'root_checkpoint_sha256': root_result['final_checkpoint_sha256'],
        'pose_checkpoint_sha256': pose_result['final_checkpoint_sha256'],
        'inference_steps': inference_steps, 'terrain_mode': terrain_mode,
        'portable_metadata': portable_metadata,
        'summary': {key: float(np.mean([row['metrics'][key] for row in rows])) for key in rows[0]['metrics']},
        'stage_summary': {key: float(np.mean([row['stage_metrics'][key] for row in rows]))
                          for key in rows[0]['stage_metrics']},
        'frame_scoring': {'start': SCORE_START, 'stop_exclusive': SCORE_STOP},
        'interval_scoring': {'start': SCORE_START, 'stop_exclusive': SCORE_STOP - 1},
        'windows': rows, 'excluded': excluded}
    if decoder_review:
        cases = rows[0]['offline_decoder_factorization']['cases']
        numerical = ('root_y_mae_m', 'root_y_signed_bias_m', 'current_frame_root_y_error_m',
                     'first_future_root_y_step_m', 'source_first_future_root_y_step_m',
                     'first_future_root_y_step_error_m', 'fk_mpjpe_m', 'root_relative_fk_mpjpe_m')
        result['offline_decoder_summary'] = {
            case: {key: float(np.mean([r['offline_decoder_factorization']['cases'][case][key]
                                      for r in rows])) for key in numerical} for case in cases}
    (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--root-run', type=Path)
    parser.add_argument('--pose-run', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-windows', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inference-steps', type=int, default=1)
    parser.add_argument('--terrain-mode', choices=('observed', 'invalid'), default='observed')
    parser.add_argument('--decoder-review', action='store_true',
                        help='Offline source-token/root oracle replays, never online conditions')
    args = parser.parse_args()
    if args.bundle is None and (args.root_run is None or args.pose_run is None):
        parser.error('Provide a verified bundle or both trained stage runs')
    result = evaluate(args.dataset, args.root_run, args.pose_run, args.output, args.device, args.max_windows,
                      args.inference_steps, args.terrain_mode, args.bundle, args.decoder_review)
    print(json.dumps({'motion': result['summary'], 'stages': result['stage_summary'],
                      'offline_decoder': result.get('offline_decoder_summary')}, indent=2))


if __name__ == '__main__':
    main()
