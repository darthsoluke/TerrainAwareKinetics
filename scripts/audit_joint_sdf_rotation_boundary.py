"""Decompose saved SO(3) boundary jumps without changing any predicted motion."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from scripts.train_joint_terrain_v3 import write_json


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('rotation_helpers', args.helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    rep = build_native_representation(args.stats)
    reports = []
    names = rep.skeleton.bone_order_names
    for path in sorted(args.input.glob('block-*/candidate-*-full.npz')):
        with np.load(path, allow_pickle=False) as saved:
            prediction = torch.from_numpy(saved['world'].copy())[None]
            history = torch.from_numpy(saved['history'].copy())[None]
        observed_pose = rep.inverse(history, False, return_all=False, return_quat=False,
                                    joint_positions_from='global_rot_data')
        predicted_pose = rep.inverse(prediction, False, return_all=False, return_quat=False,
                                     joint_positions_from='global_rot_data')
        observed = observed_pose['global_joint_rots'][0].double()
        decoded = predicted_pose['global_joint_rots'][0].double()
        actual_boundary = observed[3].transpose(-1, -2) @ decoded[4]
        past_error = observed[3].transpose(-1, -2) @ decoded[3]
        predicted_next_step = decoded[3].transpose(-1, -2) @ decoded[4]
        identity_error = (actual_boundary - past_error @ predicted_next_step).abs().max()
        if identity_error > 3e-6:
            raise ValueError('SO(3) product identity failed')
        angles = dict(past=helper.rotation_error_degrees(observed[3], decoded[3]),
            model_step=helper.rotation_error_degrees(decoded[3], decoded[4]),
            published_step=helper.rotation_error_degrees(observed[3], decoded[4]),
            previous_played_step=helper.rotation_error_degrees(observed[2], observed[3]))
        pj, hj = predicted_pose['posed_joints'][0], observed_pose['posed_joints'][0]
        prefix_offset = pj[3] - hj[3]
        model_step = pj[4] - pj[3]
        actual_step = pj[4] - hj[3]
        torch.testing.assert_close(actual_step, prefix_offset + model_step, atol=1e-6, rtol=0)
        joints = []
        for index, name in enumerate(names):
            joints.append(dict(name=name, **{key + '_deg': float(value[index]) for key, value in angles.items()},
                past_position_error_m=float(prefix_offset[index].norm()),
                model_position_step_m=float(model_step[index].norm()),
                published_position_step_m=float(actual_step[index].norm())))
        reports.append(dict(block=int(path.parent.name.split('-')[1]), candidate=int(path.stem.split('-')[1]),
            matrix_identity_max_error=float(identity_error),
            mean={key + '_deg': float(value.mean()) for key, value in angles.items()},
            maximum={key + '_deg': float(value.max()) for key, value in angles.items()},
            joints=joints))
    write_json(args.output / 'complete.json', dict(optimizer_updates=0, quality_approved=False, rows=reports,
        scope='Rotation product and position vector identities on saved unmodified predictions. '
              'Rotation angles cannot be added; their axes and noncommuting composition matter. '
              'A large reconstructed-past error is not corrected at runtime and does not by itself validate a new loss.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('stats', 'helper', 'input', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    main(parser.parse_args())
