"""Paired masked-token completion vs the actual all-unknown online first pass."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from motionbricks.data.native_mb_data import NativeMBMotionDataset, collate_native_batch
from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from motionbricks.data.native_mb_vertical_frame import training_vertical_frame
from motionbricks.data.native_mb_online_training import predict_training_root
from motionbricks.data.native_mb_online_conditions import dense_history_pose
from motionbricks.data.native_mb_terrain_batch import root_height_condition
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.evaluate_parc_native_terrain import _dataset_windows
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import select_gpu, sha256


@torch.inference_mode()
def evaluate(infer, bundle, dataset):
    _, windows, _, _, identity, _, _, _ = _dataset_windows(dataset, 8)
    if identity['dataset_manifest_sha256'] != bundle['metadata']['dataset_manifest_sha256']:
        raise ValueError('Dataset binding differs')
    loader = NativeMBMotionDataset(dataset, 'validation', min_frames=65, max_frames=10**9,
                                  require_terrain=True, require_online_terrain=True)
    lookup = {r['shard']: i for i, r in enumerate(loader.records)}
    rep, local_rep, pose = infer.global_motion_rep, infer.local_motion_rep, infer._pose_model
    root = infer._root_model.eval().requires_grad_(False)
    object.__setattr__(pose, 'online_root_model', root)
    rows = []
    torch.manual_seed(2333)
    for window in windows:
        item = loader[lookup[window['shard']]]
        batch = collate_native_batch([item])
        start = window['start']
        raw = item['motion'][start:start + 65].to(infer.device)[None]
        raw = rep.change_first_heading(raw, 0., True, True)
        raw = training_vertical_frame(raw, rep, pose.args)
        local = infer.motion_rep.dual_rep.global_to_local(raw, True, True,
                    lengths=torch.tensor([65], device=infer.device))[:, :64]
        batch.update(global_motions=raw[:, :64], local_motions=local,
                     num_tokens=torch.tensor([[16]], device=infer.device))
        predicted_root, predicted_local, terrain = predict_training_root(pose, batch,
            {'chosen_ids': torch.tensor([0]), 'frame_indices': torch.arange(start, start + 65)[None]})
        is_local = pose.args['cond_root_feature_is_from_motion_rep'] == 'local'
        root_condition = extract_feature_from_motion_rep(predicted_local if is_local else predicted_root,
            local_rep if is_local else rep, pose.args['cond_root_feature'])
        context, has_context = dense_history_pose(extract_feature_from_motion_rep(local, local_rep, pose.args['local_pose_feature']))
        height = root_height_condition(predicted_root, rep)
        tokens = infer._vqvae_pose_model.encode_into_idx(local, fetch_overall_indices=False)
        batch['groundtruth_pose_tokens'] = tokens
        (batch['focused_token_mask'], batch['masked_token_mask'], batch['incorrect_token_mask'],
         batch['correct_token_mask'], _, _) = pose._get_token_masks(batch)
        ordinary = pose._generate_tokens_for_training(batch)
        mask = ordinary == pose.backbone_net.POSE_MASK_ID
        mask[:, 0] = False
        cases = {}
        for name, inputs in (('native_partial_mask', ordinary),
                             ('online_all_unknown', torch.full_like(tokens, pose.backbone_net.POSE_MASK_ID))):
            logits = pose.backbone_net(inputs, root_condition, context, has_context,
                batch['num_tokens'], None, None, terrain_observation=terrain,
                root_height_trajectory=height)['pose_logits']
            loss = F.cross_entropy(logits.flatten(0, 2), tokens.flatten().long(), reduction='none').reshape_as(tokens)
            correct = logits.argmax(-1) == tokens
            cases[name] = {'common_masked_tokens': int(mask.sum()),
                'common_masked_ce_sum': float(loss[mask].sum()),
                'common_masked_correct': int(correct[mask].sum()),
                'future_ce': float(loss[:, 1:].mean()),
                'future_accuracy': float(correct[:, 1:].float().mean())}
        rows.append({'window': window, 'retained_source_future_tokens': int(((ordinary == tokens)[:, 1:]).sum()),
                     'cases': cases})
    summary = {}
    for name in ('native_partial_mask', 'online_all_unknown'):
        count = sum(r['cases'][name]['common_masked_tokens'] for r in rows)
        summary[name] = {'common_masked_tokens': count,
            'cross_entropy': sum(r['cases'][name]['common_masked_ce_sum'] for r in rows) / count if count else None,
            'accuracy': sum(r['cases'][name]['common_masked_correct'] for r in rows) / count if count else None}
    return {'schema': 'parc.pose_masking_diagnostic.v1', 'summary': summary, 'rows': rows,
            'interpretation': 'same scored token positions and all other conditions; partial-mask arm has privileged future token context',
            'partial_mask_arm_is_runtime_candidate': False,
            'all_unknown_is_supported_by_training': False,
            'native_max_mask_fraction': float(pose.args['masked_token_ratio']),
            'quality_approved': False}


@full_precision_evaluation()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'dataset', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output')
    gpu = None if args.cpu else select_gpu(args.gpu)
    torch.set_num_threads(4)
    infer, bundle = load_terrain_bundle(args.bundle, 'cpu' if args.cpu else 'cuda:0')
    result = evaluate(infer, bundle, args.dataset)
    result.update(bundle_sha256=sha256(args.bundle), gpu_at_launch=gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
