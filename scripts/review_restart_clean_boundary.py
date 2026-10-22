"""Test continuous restart from one real stopped prefix, without teacher resets."""

import argparse
from pathlib import Path

import torch

from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_bones_restart_repair import load_command_model, review_restarts
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_command_model(torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False))
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    selected = []
    for case in cases:
        block = case['restart_block']
        selected.append(dict(case, world=case['world'][block*8:].clone(), controls=case['controls'][block:].clone(),
            training_segments=case['training_segments']-block, restart_block=0,
            source_start_frame=case['source_start_frame']+block*8))
    geometry = JointGeometryObjective(model.inference.global_motion_rep, 'cuda:0')
    rows = review_restarts(model, selected, geometry, args.output)
    write_json(args.output/'complete.json', dict(rows=rows, checkpoint_sha256=sha256(args.checkpoint),
        initialization='One real stopped history immediately before the existing MOVE event; thereafter fully generated, never reset',
        controls_unchanged=True, future_source_in_actor=False, postprocessing=None, optimizer_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'cases', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
