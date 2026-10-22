"""Replay factual STOP histories and distinguish Root planning from final decoding."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.stop_command_adapter import StopCommandAdapter
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from scripts.pilot_joint_sdf_contact_preserve import load_model


def speed(points):
    return torch.diff(points[..., [0, 2]], dim=1).norm(dim=-1)*30


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = torch.load(args.parent, map_location='cpu', mmap=True, weights_only=False)
    model, _ = load_model(parent)
    model.requires_grad_(False)
    original_control = model.control
    receipt = json.loads((args.artifact/'export.json').read_text())
    rows = []
    for arm, folder in [('before', 'baseline'), ('candidate', 'final')]:
        if arm == 'candidate':
            model.control = StopCommandAdapter(original_control)
            candidate = torch.load(args.run/'weights.pt', map_location='cpu', mmap=True, weights_only=False)
            model.load_state_dict(candidate['model_state'], strict=True)
            model.requires_grad_(False)
        model.eval()
        for entry in receipt['entries']:
            with np.load(args.artifact/(entry['entry']+'-scene.npz'), allow_pickle=False) as data:
                initial = torch.from_numpy(data['initial_history'].copy()).cuda()
                scene = {k:torch.from_numpy(data[k].copy()).cuda()
                         for k in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
            with np.load(args.run/folder/(entry['entry']+'-release_resume.npz'), allow_pickle=False) as data:
                played = torch.from_numpy(data['world'].copy())[None].cuda()
            driver = SoftCommandState()
            for block in range(13):
                history = initial if block == 0 else reencode_played_history(
                    played[:, block*8-4:block*8], model.inference.global_motion_rep)
                stopped = block >= 3
                velocity = [0., 0.] if stopped else entry['command_velocity_mps']
                intent = driver.intent(history, velocity, entry['command_heading'], 8)
                if not stopped:
                    continue
                with torch.no_grad():
                    result = model(history, intent, [scene])
                expected = played[:, block*8:block*8+8]
                published = result['world'][:, 4:12]
                torch.testing.assert_close(published, expected, atol=3e-5, rtol=1e-5)
                planned = model.inference.global_motion_rep.unnormalize(
                    result['root_output']['pred_global_root_values'])[:, :32]
                decoded = result['world']
                row = dict(arm=arm, entry=entry['entry'], stop_block=block-3,
                    factual_input_speed_mps=float(speed(history).mean()),
                    planned_published_mean_speed_mps=float(speed(planned[:, 4:12]).mean()),
                    decoded_published_mean_speed_mps=float(speed(decoded[:, 4:12]).mean()),
                    planned_horizon_tail_mean_speed_mps=float(speed(planned[:, -8:]).mean()),
                    decoded_horizon_tail_mean_speed_mps=float(speed(decoded[:, -8:]).mean()),
                    recording_replay_max_error=float((published-expected).abs().max()),
                    scope='Speed norms in the same30Hz predictions. Root and decoder are distinct neural outputs; no Root trajectory overwrite.')
                rows.append(row)
                print(json.dumps(row), flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(rows=rows, updates=0,
        motion_modified=False, future_truth_in_actor=False), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'run', 'artifact', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
