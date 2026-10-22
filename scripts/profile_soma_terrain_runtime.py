"""Measure the exact live terrain inference, history and official skin paths."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.soma_lab_server import load_terrain_live_bundle
from motionbricks.data.native_mb_online_live import planar_command_goal, predict_planar_terrain
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--compiled', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, scene, history = load_terrain_live_bundle(args.bundle, 'cuda:0', args.scene)
    history = history.to('cuda:0')
    scene = {key: torch.as_tensor(value, device='cuda:0') for key, value in scene.items()}
    skinner = move_skinner(load_official_skinner(inference.motion_rep.skeleton), torch.device('cuda:0'))
    skinner.skeleton_skin.float().cuda()
    if args.compiled:
        from motionbricks.data.soma_terrain_execution import TerrainExecution
        execution = TerrainExecution(inference, scene, skinner, history)
        object.__setattr__(inference, '_terrain_execution', execution)
    settings = {'num_inference_step': 4, 'pose_token_sampling_use_argmax': True,
                'cache_terrain_encoder': True}
    timings = {}

    def instrument(owner, name, label):
        original = getattr(owner, name)

        def measured(*inputs, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.profiler.record_function(label):
                result = original(*inputs, **kwargs)
            torch.cuda.synchronize()
            timings.setdefault(label, []).append((time.perf_counter() - started) * 1000)
            return result

        setattr(owner, name, measured)

    for name in ('_predict_root_trajectories', '_predict_pose_tokens',
                 '_decode_motions_from_predicted_root_and_pose_tokens'):
        instrument(inference, name, name)
    for network, name in ((inference._root_model.backbone_net, 'root_backbone'),
                          (inference._pose_model.backbone_net, 'pose_backbone')):
        instrument(network, 'forward', name)

    @torch.inference_mode()
    def run():
        goal = planar_command_goal(history[..., :5], [0.0, 1.6], 0.0, 8)
        torch.cuda.synchronize()
        started = time.perf_counter()
        if args.compiled:
            output, _ = execution.predict(history, goal, {})
        else:
            output, _ = predict_planar_terrain(inference, history, goal, scene,
                                              config=settings, diagnostics={})
        torch.cuda.synchronize()
        predicted = time.perf_counter()
        if args.compiled:
            execution.observe(output[:, 8:12])
        else:
            reencode_played_history(output[:, 8:12], inference.global_motion_rep)
        torch.cuda.synchronize()
        observed = time.perf_counter()
        if args.compiled:
            execution.geometry(output[:, 4:12])['rig'].cpu().numpy()
        else:
            pose = inference.motion_rep.inverse(output[:, 4:12], False, return_all=False)
            vertices = skinner.skin(pose['global_joint_rots'][0], pose['posed_joints'][0], rot_is_global=True)
            vertices.cpu().numpy()
        finished = time.perf_counter()
        for label, ms in (('prediction', (predicted-started)*1000),
                          ('history', (observed-predicted)*1000),
                          ('skin_transfer', (finished-observed)*1000)):
            timings.setdefault(label, []).append(ms)

    run()
    timings.clear()
    for _ in range(12):
        run()
    report = {key: {'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95)),
                    'calls': len(values)} for key, values in timings.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'profile.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            run()
        profile.export_chrome_trace(str(args.output / 'trace.json'))
        print(profile.key_averages().table(sort_by='self_cpu_time_total', row_limit=16))


if __name__ == '__main__':
    main()
