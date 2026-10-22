"""Offline paired factual-source/generated-history response; zero optimization."""

import argparse
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.source_episode_control import source_episode_task
from motionbricks.data.terrain_control import SoftCommandState
from scripts.pilot_bones_restart_repair import load_command_model
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_command_model(torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False))
    rep = model.inference.global_motion_rep
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    all_rows = []
    for case in cases:
        source = case['world'][None].cuda()
        controls = case['controls'][None].cuda()
        with np.load(args.recordings/(case['label']+'.npz'), allow_pickle=False) as archive:
            recorded = torch.from_numpy(archive['world'].copy())[None].cuda()
        driver = SoftCommandState()
        rows = []
        for block in range(case['training_segments']):
            offset = block*8
            factual = source[:, offset:offset+4]
            generated = source[:, :4] if block==0 else reencode_played_history(recorded[:, offset-4:offset], rep)
            task = driver.from_controls(generated, controls[:, block], 8)
            if block < case['restart_block']:
                continue
            output = model(generated, task, [case['scene']])['world'][:, 4:12].clone()
            replay_error = float((output-recorded[:, offset:offset+8]).abs().max())
            torch.testing.assert_close(output, recorded[:, offset:offset+8], atol=3e-5, rtol=1e-5)
            source_task = source_episode_task(source, controls, block)
            teacher = model(factual, source_task, [case['scene']])['world'][:, 4:12].clone()
            direction = controls[0, block, :2]/controls[0, block, :2].norm()
            expected = source[0, offset+11, [0, 2]]-source[0, offset+3, [0, 2]]
            predicted = output[0, -1, [0, 2]]-generated[0, 3, [0, 2]]
            supported = teacher[0, -1, [0, 2]]-factual[0, 3, [0, 2]]
            gj = rep.inverse(generated, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
            sj = rep.inverse(factual, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
            relative_error = ((gj-generated[:, :, None, :3])-(sj-factual[:, :, None, :3])).norm(dim=-1).mean()
            rows.append(dict(block=block, source_frame=case['source_start_frame']+offset,
                generated_replay_max_abs=replay_error, request_mps=float(controls[0, block, :2].norm()),
                source_forward_mps=float(expected@direction*30/8),
                factual_history_forward_mps=float(supported@direction*30/8),
                generated_history_forward_mps=float(predicted@direction*30/8),
                history_root_gap_m=float((generated[0, 3, :3]-factual[0, 3, :3]).norm()),
                history_relative_joint_error_m=float(relative_error)))
        tail = rows[-5:]
        all_rows.append(dict(label=case['label'], split=case['pilot_split'], rows=rows,
            last5_mean={key:sum(row[key] for row in tail)/len(tail) for key in
                ('source_forward_mps', 'factual_history_forward_mps', 'generated_history_forward_mps',
                 'history_root_gap_m', 'history_relative_joint_error_m')}))
        write_json(args.output, dict(checkpoint_sha256=sha256(args.checkpoint), rows=all_rows,
            optimizer_updates=0, factual_history_branch='Offline source past at each source time; not a continuous controllable actor',
            future_source_in_actor=False, source_pose_stitching=False, postprocessing=None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'cases', 'recordings', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
