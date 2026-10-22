"""Compare real online publication with the default training rollout protocol."""

import argparse
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.joint_terrain_live import JointTerrainOnlineBackend
from motionbricks.data.locomotion_command_adapter import install_checkpoint_command
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_runtime_controls import JOINT_RUNTIME_DEFAULTS
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    inference, bundle = load_terrain_bundle(Path(saved['contract']['source']), args.device)
    model = JointSDFController(inference, bundle)
    installed = install_checkpoint_command(model, saved)
    model.load_state_dict(saved['model_state'], strict=True)
    model.requires_grad_(False); model.eval()
    with np.load(args.seed_scene, allow_pickle=False) as archive:
        initial = torch.from_numpy(archive['initial_history'].copy()).to(args.device)
    rep = inference.global_motion_rep
    initial = rep.change_first_heading(initial, 0., False, False)
    initial[..., [0, 2]] -= initial[:, -1:, [0, 2]].clone()
    scene = dict(heightfield_y_m=torch.zeros(201, 201), min_point_world_xyz_m=torch.tensor([-50., 0., -50.]), dx_m=torch.tensor(.5))
    backend = JointTerrainOnlineBackend(model, initial, scene)
    backend.configure_runtime(dict(JOINT_RUNTIME_DEFAULTS))
    driver = SoftCommandState()
    commands = [('W', 0., 1.6)]*2 + [('release', 0., 0.)]*3 + [('A', np.pi/2, 1.6)]*2
    commands += [('release_A', np.pi/2, 0.)]*2 + [('S', np.pi, 1.6)]*2 + [('W_restart', 0., 1.6)]*2
    rows = []
    for index, (key, yaw, speed) in enumerate(commands):
        # Match the scalar conversion used by the online backend, not a hidden
        # reference trajectory or any future height/body pose.
        velocity = [np.sin(yaw)*speed, np.cos(yaw)*speed]
        task = driver.intent(backend.history, velocity, yaw, 8)
        direct = model(backend.history, task, [scene])['world']
        expected = direct[:, 4:12].clone()
        observed = reencode_played_history(direct[:, 8:12], rep)
        online = backend.predict_plan(velocity, yaw)
        error = float((expected-online['planned_motion']).abs().max())
        history_error = float((observed-online['published_history']).abs().max())
        row = dict(block=index, key=key, output_max_abs=error, published_history_max_abs=history_error)
        rows.append(row)
        torch.testing.assert_close(expected, online['planned_motion'], atol=3e-5, rtol=1e-5)
        torch.testing.assert_close(observed, online['published_history'], atol=3e-5, rtol=1e-5)
        backend.history = online['published_history'].clone()
    write_json(args.output, dict(checkpoint_sha256=sha256(args.checkpoint), command_modules_installed=installed,
        runtime_defaults=JOINT_RUNTIME_DEFAULTS, rows=rows, source_future_used=False, postprocessing=None,
        optimizer_updates=0, actual_browser_keyboard_test=False, numerical_interface_test_not_quality_approval=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'seed-scene', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    main(parser.parse_args())
