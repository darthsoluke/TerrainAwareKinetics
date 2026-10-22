"""Zero-update ablation of stopping vs returning to the key-release position."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from motionbricks.data.stop_command_adapter import StopCommandAdapter
from scripts.pilot_joint_sdf_contact_preserve import load_model


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('stop_goal_review', args.review_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = torch.load(args.parent, map_location='cpu', mmap=True, weights_only=False)
    model, _ = load_model(parent)
    model.requires_grad_(False)
    model.eval()
    original = helper.review(model, args, args.output/'original-current-goal',
        reference=args.run/'baseline', stop_goal_policy='current')
    model.control = StopCommandAdapter(model.control)
    candidate = torch.load(args.run/'weights.pt', map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(candidate['model_state'], strict=True)
    model.requires_grad_(False)
    model.eval()
    adapted = helper.review(model, args, args.output/'adapter-current-goal',
        reference=args.run/'baseline', stop_goal_policy='current')
    (args.output/'complete.json').write_text(json.dumps(dict(original=original, adapted=adapted,
        updates=0, output_modified=False, future_truth_in_actor=False, quality_approved=False,
        intervention='For STOP only, requested planar goal is the current observed Root, not the old key-release point. Requested speed remains0. No generated position is replaced.'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'run', 'artifact', 'output', 'review-helper'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
