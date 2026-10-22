"""Offline effect of the final decoder velocity assignment; no actor deployment."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

from motionbricks.data import native_mb_online_conditions as conditions
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.audit_unified_stair_gait import generate, difference, token_difference
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


@contextmanager
def model_terminal_velocity():
    original = conditions.apply_terminal_velocity
    conditions.apply_terminal_velocity = lambda trajectory, _sparse, _mask, _lengths: trajectory
    try:
        yield
    finally:
        conditions.apply_terminal_velocity = original


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle).eval().requires_grad_(False)
    saved = torch.load(a.initial, map_location='cpu', weights_only=False)
    if saved['contract']['bundle_sha256'] != sha256(a.bundle):
        raise ValueError('Baseline mismatch')
    controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
    before = state_digest(controller, list(controller.state_dict()))
    with np.load(a.scene, allow_pickle=False) as data:
        scene = {k: torch.from_numpy(data[k].copy()) for k in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
    with np.load(a.recording, allow_pickle=False) as data:
        world = torch.from_numpy(data['global_features'].copy()).to(a.device)
    rows = []
    for frame in (48, 72, 96, 120, 144, 168):
        history = reencode_played_history(world[None, frame:frame + 4], inference.global_motion_rep)
        command = history.new_tensor([[0., 1.6, 1., 0.]])
        actual = generate(controller, history, command, scene)
        with model_terminal_velocity():
            relaxed = generate(controller, history, command, scene)
        replay = generate(controller, history, command, scene)
        torch.testing.assert_close(actual['world'], replay['world'], atol=0, rtol=0)
        row = dict(frame=frame, tokens=token_difference(actual, relaxed),
            root_plan_max_difference=float((actual['root_output']['pred_global_root_values'] - relaxed['root_output']['pred_global_root_values']).abs().max()),
            published_difference=difference(inference.global_motion_rep, actual['world'][:, 4:28], relaxed['world'][:, 4:28]),
            replay_error=0., not_published=True)
        rows.append(row)
    if state_digest(controller, list(controller.state_dict())) != before:
        raise ValueError('Diagnostic changed model weights')
    write_json(a.output / 'complete.json', dict(rows=rows, optimizer_updates=0, weights_unchanged=True,
        runtime_changed=False, root_condition_mode=inference._pose_model.args['cond_root_feature_is_from_motion_rep'],
        scope='Only the final local decoder-velocity assignment is removed offline. '
              'The original desired-speed condition to the Root network remains. '
              'This is not a fully learned soft-speed protocol or a deployment repair.'))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('bundle', 'initial', 'scene', 'recording', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
