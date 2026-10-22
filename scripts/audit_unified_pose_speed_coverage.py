"""Read-only source terminal-speed coverage and same-history command probes."""
import argparse
from pathlib import Path
import torch
import numpy as np

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.unified_training_goals import source_terminal_goal
from scripts.audit_unified_stair_gait import summarize
from scripts.audit_unified_stair_anticipation import fresh_trace
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    controller = UnifiedTerrainController(inference, bundle).eval().requires_grad_(False)
    saved = torch.load(args.initial, map_location='cpu', weights_only=False)
    if saved['contract']['bundle_sha256'] != sha256(args.bundle):
        raise ValueError('Source baseline mismatch')
    controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
    initial_hash = state_digest(controller, list(controller.state_dict()))
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    coverage = []
    for case in cases:
        windows = torch.stack([case['world'][start:start + 33, :5] for start in range(len(case['world']) - 32)])
        speed = source_terminal_goal(windows)[:, :2].norm(dim=-1)
        coverage.append(dict(label=case['label'], entry=case.get('entry'), windows=len(windows),
            speed_quantiles_mps=torch.quantile(speed, speed.new_tensor([0., .1, .5, .9, 1.])).tolist(),
            fraction_at_08pm02=float(((speed >= .6) & (speed <= 1.)).float().mean()),
            fraction_at_16pm02=float(((speed >= 1.4) & (speed <= 1.8)).float().mean())))
    with np.load(args.scene, allow_pickle=False) as data:
        scene = {k: torch.from_numpy(data[k].copy()) for k in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
    with np.load(args.recording, allow_pickle=False) as data:
        world = torch.from_numpy(data['global_features'].copy()).to(args.device)
    rows = []
    for frame in (72, 80, 88):
        history = reencode_played_history(world[None, frame:frame + 4], inference.global_motion_rep)
        reference = None
        for speed in (.8, 1.6, 2.4, 3.2):
            command = history.new_tensor([[0., speed, 1., 0.]])
            result, root, trace = fresh_trace(controller, history, command, scene)
            rows.append(dict(frame=frame, speed=speed, trace=trace,
                gait=summarize(inference.global_motion_rep, result['world'][:, 4:28], scene),
                not_published=True))
            if speed == 1.6:
                reference = result['world'].clone()
        replay, _, _ = fresh_trace(controller, history, history.new_tensor([[0., 1.6, 1., 0.]]), scene)
        torch.testing.assert_close(replay['world'], reference, atol=0, rtol=0)
    if state_digest(controller, list(controller.state_dict())) != initial_hash:
        raise ValueError('Read-only probe changed weights')
    write_json(args.output / 'complete.json', dict(coverage=coverage, rows=rows, optimizer_updates=0,
        weights_unchanged=True, replay_error=0., runtime_changed=False,
        note='Changing desired speed also changes causal lookahead and the requested trajectory. '
             'This is not isolated SDF causality or a traversal pass, and does not authorize changing user controls.'))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('bundle', 'initial', 'cases', 'scene', 'recording', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
