"""Read-only checks of online condition parity, world translation and prefetch joins."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import NativeMBMotionDataset, collate_native_batch
from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from motionbricks.data.native_mb_frame import translate_global_height
from motionbricks.data.native_mb_online_conditions import dense_history_pose, planar_goal
from motionbricks.data.native_mb_online_live import (
    NativeTerrainOnlineBackend, planar_command_goal, predict_planar_terrain,
)
from motionbricks.data.native_mb_online_training import predict_training_root
from motionbricks.data.native_mb_terrain_batch import root_height_condition
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.evaluate_parc_native_terrain import _dataset_windows
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import select_gpu, sha256


def snapshot(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(item) for item in value)
    return value


@contextmanager
def capture(module):
    calls = []
    def record(_module, args, kwargs, result):
        calls.append((snapshot(args), snapshot(kwargs), snapshot(result)))
    hook = module.register_forward_hook(record, with_kwargs=True)
    try:
        yield calls
    finally:
        hook.remove()


def difference(first, second):
    if first.shape != second.shape:
        raise ValueError(f'Comparison shape mismatch: {first.shape} vs {second.shape}')
    first, second = first.detach().cpu(), second.detach().cpu()
    if first.dtype == torch.bool:
        return {'different_elements': int((first != second).sum())}
    delta = (first - second).abs().double()
    return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean())}


@torch.inference_mode()
def condition_parity(inference, dataset, max_windows=2):
    rep, local_rep = inference.global_motion_rep, inference.local_motion_rep
    _, windows, _, _, identity, _, _, _ = _dataset_windows(dataset, max_windows)
    loader = NativeMBMotionDataset(dataset, 'validation', max_frames=10**9, min_frames=65,
                                  require_terrain=True, require_online_terrain=True)
    index = {row['shard']: i for i, row in enumerate(loader.records)}
    root, pose = inference._root_model, inference._pose_model
    root.requires_grad_(False)
    object.__setattr__(pose, 'online_root_model', root)
    rows = []
    for window in windows:
        item = loader[index[window['shard']]]
        original_batch = collate_native_batch([item])
        start = window['start']
        for tokens in (8, 16):
            frames = tokens * 4
            raw = rep.unnormalize(item['motion'][start:start + frames + 1].to(inference.device))[None]
            world_roots = item['terrain_root_world'][start:start + frames + 1].to(inference.device)[None]
            heading = torch.atan2(world_roots[:, 0, 4], world_roots[:, 0, 3])
            canonical = rep.change_first_heading(raw, heading, False, False)
            world = canonical.clone()
            world[..., rep.indices['global_root_pos_2d']] += world_roots[:, :1, [0, 2]]
            if pose.args.get('vertical_frame'):
                from motionbricks.data.native_mb_vertical_frame import center_motion_window
                canonical, _ = center_motion_window(canonical, rep, normalized=False)
            local = inference.motion_rep.dual_rep.global_to_local(
                canonical, False, True, lengths=torch.tensor([frames + 1], device=inference.device))[:, :frames]
            batch = dict(original_batch, global_motions=rep.normalize(canonical[:, :frames]),
                         local_motions=local,
                         num_tokens=torch.tensor([[tokens]], device=inference.device))
            info = {'chosen_ids': torch.tensor([0]),
                    'frame_indices': torch.arange(start, start + frames + 1)[None]}
            with capture(root.backbone_net) as train_calls:
                train_root, train_local, terrain = predict_training_root(pose, batch, info)
            with capture(root.backbone_net) as live_root_calls, capture(pose.backbone_net) as live_pose_calls:
                predict_planar_terrain(inference, world[:, :4], planar_goal(world[:, frames - 1, :5]),
                                       item['terrain_scene'], num_tokens=tokens,
                                       config={'num_inference_step': 1, 'pose_token_sampling_use_argmax': True})
            train_args, train_kwargs, _ = train_calls[0]
            live_args, live_kwargs, live_result = live_root_calls[0]
            fields = ('global_root', 'global_mask', 'local_root', 'local_mask',
                      'poses', 'pose_mask', 'num_tokens')
            comparisons = {name: difference(train_args[i], live_args[i]) for i, name in enumerate(fields)}
            comparisons['terrain_points'] = difference(terrain['points'], live_kwargs['terrain_observation']['points'])
            comparisons['terrain_valid'] = difference(terrain['valid'], live_kwargs['terrain_observation']['valid'])
            comparisons['root_prediction'] = difference(train_root, live_result['pred_global_root_values'][:, :frames])
            pose_args, pose_kwargs, pose_result = live_pose_calls[0]
            pose_cond, has_pose = dense_history_pose(extract_feature_from_motion_rep(
                local, local_rep, pose.args['local_pose_feature']))
            root_is_local = pose.args['cond_root_feature_is_from_motion_rep'] == 'local'
            root_cond = extract_feature_from_motion_rep(
                train_local if root_is_local else train_root,
                local_rep if root_is_local else rep, pose.args['cond_root_feature'])
            train_height = root_height_condition(train_root, rep)
            comparisons['pose_root_input'] = difference(root_cond, pose_args[1][:, :frames])
            comparisons['pose_context'] = difference(pose_cond, pose_args[2][:, :frames])
            comparisons['pose_context_mask'] = difference(has_pose, pose_args[3][:, :frames])
            comparisons['pose_height_input'] = difference(train_height, pose_kwargs['root_height_trajectory'][:, :frames])
            train_pose_result = pose.backbone_net(pose_args[0][:, :tokens], root_cond, pose_cond,
                has_pose, batch['num_tokens'], None, None, terrain_observation=terrain,
                root_height_trajectory=train_height)
            comparisons['pose_logits'] = difference(train_pose_result['pose_logits'], pose_result['pose_logits'][:, :tokens])
            rows.append({'window': window, 'tokens': tokens, 'comparisons': comparisons})
    return {'dataset': identity, 'rows': rows,
            'scope': 'same expert hindsight XZ/heading goal; no source future Y/pose supplied to live prediction'}


@torch.inference_mode()
def vertical_translation(inference, bundle):
    history = bundle['initial_history'].to(inference.device)
    goal = planar_command_goal(history[..., :5], [0., 1.6], 0., 8)
    with capture(inference._root_model.backbone_net) as base_calls:
        base, _ = predict_planar_terrain(inference, history, goal, bundle['scene'], num_tokens=8,
            config={'num_inference_step': 4, 'pose_token_sampling_use_argmax': True})
    rep = inference.global_motion_rep
    base_root = rep.unnormalize(base_calls[0][2]['pred_global_root_values'])
    if inference._pose_model.args.get('vertical_frame'):
        from motionbricks.data.native_mb_vertical_frame import center_motion_window
        _, base_origin = center_motion_window(history, rep, normalized=False)
        base_root[..., 1] += base_origin[:, None]
    base_observation = base_calls[0][1]['terrain_observation']
    base_fk = rep.inverse(base[:, 4:28], False, return_all=True)['posed_joints']
    rows = []
    for offset in (-1., 1., 3.):
        shifted_history = translate_global_height(history, rep, offset)
        scene = {key: torch.as_tensor(value).clone() for key, value in bundle['scene'].items()}
        scene['heightfield_y_m'] += offset
        scene['min_point_world_xyz_m'][1] += offset
        with capture(inference._root_model.backbone_net) as calls:
            prediction, _ = predict_planar_terrain(inference, shifted_history, goal, scene, num_tokens=8,
                config={'num_inference_step': 4, 'pose_token_sampling_use_argmax': True})
        restored = translate_global_height(prediction, rep, -offset)
        predicted_root = rep.unnormalize(calls[0][2]['pred_global_root_values'])
        if inference._pose_model.args.get('vertical_frame'):
            _, shifted_origin = center_motion_window(shifted_history, rep, normalized=False)
            predicted_root[..., 1] += shifted_origin[:, None]
        observation = calls[0][1]['terrain_observation']
        fk = rep.inverse(restored[:, 4:28], False, return_all=True)['posed_joints']
        rows.append({'world_translation_y_m': offset,
            'observation_points_difference': difference(base_observation['points'], observation['points']),
            'observation_valid_difference': difference(base_observation['valid'], observation['valid']),
            'root_stage_y_equivariance_mae_m': float((predicted_root[:, 4:28, 1] - offset - base_root[:, 4:28, 1]).abs().mean()),
            'decoded_root_y_equivariance_mae_m': float((restored[:, 4:28, 1] - base[:, 4:28, 1]).abs().mean()),
            'decoded_root_xyz_equivariance_mae_m': float((restored[:, 4:28, :3] - base[:, 4:28, :3]).norm(dim=-1).mean()),
            'fk_equivariance_mae_m': float((fk - base_fk).norm(dim=-1).mean())})
    return {'scope': 'same physical history, terrain and intent under a common world translation', 'rows': rows}


@contextmanager
def capture_prediction(inference):
    original = inference.predict
    outputs = []
    had_override = 'predict' in vars(inference)
    def run(*args, **kwargs):
        result = original(*args, **kwargs)
        outputs.append(snapshot(result))
        return result
    inference.predict = run
    try:
        yield outputs
    finally:
        if had_override:
            inference.predict = original
        else:
            delattr(inference, 'predict')


@torch.inference_mode()
def prefetch_joins(inference, bundle):
    backend = NativeTerrainOnlineBackend(inference, bundle['initial_history'], bundle['scene'])
    prior = backend.history.clone()
    block = backend.predict_plan([[0., 1.6]], [0.])['planned_motion']
    rep = inference.global_motion_rep
    rows = []
    for index in range(8):
        observed = backend.observe_history(torch.cat((prior, block[:, :2]), 1)[:, -4:])
        expected = backend.observe_history(block[:, -4:])
        backend.history = observed.clone()
        with capture_prediction(inference) as outputs:
            candidate = backend.prepare_next_plan([0., 1.6], 0., observed, expected, overlap_frames=6)
        deferred = candidate.get('deferred') is True
        if deferred:
            if outputs:
                raise RuntimeError('A deferred prefetch unexpectedly ran inference')
            backend.history = expected.clone()
            with capture_prediction(inference) as outputs:
                next_plan = backend.predict_plan([[0., 1.6]], [0.])
        if outputs:
            prediction = outputs[0][0]
            predicted_prior = prediction[:, :4] if deferred else prediction[:, 6:10]
            fk_prior = rep.inverse(predicted_prior, False, return_all=True)['posed_joints']
            fk_actual = rep.inverse(expected, False, return_all=True)['posed_joints']
            mismatch = fk_prior[:, -1] - fk_actual[:, -1]
            rows.append({'join': index + 1, 'neural_generation': True,
                         'inference_from_full_played_boundary': deferred,
                         'discarded_overlap_frames': 0 if deferred else 6,
                         'forecast_predecessor_root_gap_m': float(mismatch[:, rep.skeleton.root_idx].norm(dim=-1).mean()),
                         'forecast_predecessor_fk_mean_gap_m': float(mismatch.norm(dim=-1).mean()),
                         'forecast_predecessor_fk_max_gap_m': float(mismatch.norm(dim=-1).max())})
        else:
            rows.append({'join': index + 1, 'neural_generation': False,
                         'forecast_predecessor_root_gap_m': 0., 'forecast_predecessor_fk_mean_gap_m': 0.,
                         'forecast_predecessor_fk_max_gap_m': 0.})
        if not deferred:
            backend.history = expected.clone()
            backend.commit_prepared_plan(candidate)
        prior, block = expected, next_plan['planned_motion'] if deferred else candidate['planned_motion']
    return {'scope': 'prefix-two prefetch path; deferred generation uses full played boundary; no pose correction', 'rows': rows}


@full_precision_evaluation()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new audit output')
    gpu = select_gpu(args.gpu)
    torch.set_num_threads(4)
    inference, bundle = load_terrain_bundle(args.bundle, 'cuda:0')
    result = {'schema': 'parc.self_review_online_contract.v1', 'bundle_sha256': sha256(args.bundle),
              'source_sha256': sha256(Path(__file__)), 'gpu_at_launch': gpu, 'quality_approved': False}
    if args.dataset:
        result['condition_parity'] = condition_parity(inference, args.dataset)
    result['vertical_translation'] = vertical_translation(inference, bundle)
    result['prefetch_joins'] = prefetch_joins(inference, bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
