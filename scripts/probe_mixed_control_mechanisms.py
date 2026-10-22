"""Read-only native stage and feedback diagnosis on held-out mixed-corpus seeds."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import NativeMBMotionDataset
from motionbricks.data.native_mb_online_live import predict_planar_terrain, online_command_goal
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from scripts.portable_parc_online_soma import load_terrain_bundle, world_history
from scripts.train_native_mb import select_gpu, sha256


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'dataset', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--gpu', type=int, choices=(0, 1), default=0)
    p.add_argument('--blocks', type=int, default=12)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh probe output')
    gpu = select_gpu(args.gpu)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    infer, bundle = load_terrain_bundle(args.bundle, 'cuda:0')
    data = NativeMBMotionDataset(args.dataset, 'validation', min_frames=65, max_frames=10**9,
        require_terrain=True, require_online_terrain=True, online_only=True)
    if sha256(args.dataset / 'native_manifest.json') != bundle['metadata']['dataset_manifest_sha256']:
        raise ValueError('Probe dataset does not match the bundle')
    rep, codec = infer.global_motion_rep, infer._vqvae_pose_model
    args.output.mkdir(parents=True)
    report = {'gpu': gpu, 'bundle_sha256': sha256(args.bundle), 'cases': [],
              'scope': 'offline intervention only; no output modification in the live controller'}
    seeds = []
    for index, row in enumerate(data.records):
        if row.get('origin') != 'bones_seed' or row.get('control_category') != 'idle':
            continue
        item = data[index]
        roots = item['terrain_root_world'][:, :3]
        speed = torch.diff(roots, dim=0).norm(dim=-1) * 30
        if len(speed) < 120:
            continue
        start = int(speed.unfold(0, 120, 1).amax(-1).argmin())
        seeds.append((row, item, start))
        if len(seeds) == 2:
            break
    if len(seeds) != 2:
        raise ValueError('Two held-out quiet seeds are required')
    capture = {}
    original_decoder = codec.forward_decoder
    inference_config = {'num_inference_step': 4, 'pose_token_sampling_use_argmax': True}

    def decoder(*values, **options):
        result = original_decoder(*values, **options)
        capture.update(values=values, options=options, reconstruction=result['recon_state'])
        return result

    codec.forward_decoder = decoder
    try:
        for seed_id, (row, item, start) in enumerate(seeds):
            initial = world_history(rep, item, start).to(infer.device)
            heading = float(torch.atan2(initial[0, -1, 4], initial[0, -1, 3]))
            scene = item['terrain_scene']
            for dy in (0., -.1, .1):
                history = shift_global_motion(initial, rep, dy)
                goal = online_command_goal(history[..., :5], [0., 0.], heading, 8, bundle['protocol']['version'])
                debug = {}
                generated, _ = predict_planar_terrain(infer, history, goal, scene, diagnostics=debug, config=inference_config)
                raw_local = infer.local_motion_rep.unnormalize(capture['reconstruction'])
                options = dict(capture['options'])
                for key in ('target_cond', 'has_target_cond', 'external_cond'):
                    if options.get(key) is not None:
                        options[key] = options[key][:, :32]
                options['token_mask'] = options['token_mask'][:, :8]
                cropped = original_decoder(capture['values'][0][:, :8], **options)['recon_state']
                raw_cropped = infer.local_motion_rep.unnormalize(cropped)
                root_trace = np.asarray(debug['root_stage_world_xyz_m'])
                report['cases'].append({'seed': seed_id, 'shard': row['shard'], 'start': start,
                    'mode': 'height_intervention', 'history_shift_y_m': dy,
                    'history_y_m': float(history[0, -1, 1]),
                    'root_future_y_mean_m': float(root_trace[:, 1].mean()),
                    'decoded_future_y_mean_m': float(generated[0, 4:28, 1].mean()),
                    'padded_vs_cropped_feature_max': float((cropped-capture['reconstruction'][:, :32]).abs().max()),
                    'padded_vs_cropped_height_max_m': float((raw_local[:, :32, 3]-raw_cropped[..., 3]).abs().max())})
            for mode in ('teacher_history', 'played_history'):
                history = initial
                blocks = []
                for block in range(args.blocks):
                    if mode == 'teacher_history':
                        history = world_history(rep, item, start + block * 8).to(infer.device)
                    goal = online_command_goal(history[..., :5], [0., 0.], heading, 8, bundle['protocol']['version'])
                    debug = {}
                    generated, _ = predict_planar_terrain(infer, history, goal, scene, diagnostics=debug, config=inference_config)
                    future = generated[:, 4:12]
                    observed = reencode_played_history(future[:, -4:], rep)
                    root_trace = np.asarray(debug['root_stage_world_xyz_m'])[:8]
                    blocks.append({'block': block, 'observed_y_m': float(history[0, -1, 1]),
                        'root_future_y_mean_m': float(root_trace[:, 1].mean()),
                        'decoded_future_y_mean_m': float(future[0, :, 1].mean()),
                        'new_observed_y_m': float(observed[0, -1, 1]),
                        'root_planar_step_m': (root_trace[-1, [0, 2]]-history[0, -1, [0, 2]].cpu().numpy()).tolist(),
                        'decoded_planar_step_m': (future[0, -1, [0, 2]]-history[0, -1, [0, 2]]).cpu().tolist(),
                        'observer_root_error_m': float((observed[0, :, :3]-future[0, -4:, :3]).abs().max())})
                    history = observed
                report['cases'].append({'seed': seed_id, 'mode': mode, 'blocks': blocks})
        (args.output / 'mechanisms.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        codec.forward_decoder = original_decoder


if __name__ == '__main__':
    main()
