"""Read-only objective-gradient attribution for the frozen-body Root adapter fit."""

import argparse
from pathlib import Path

import torch

from motionbricks.data.terrain_decoder_pilot import training_targets
from motionbricks.data.unified_terrain import UnifiedTerrainController, SCHEMA
from motionbricks.data.unified_training_goals import source_terminal_goal
from scripts.pilot_unified_goal_alignment import freeze_for_root_comparison
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def cosine(a, b):
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-20))


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    controller = UnifiedTerrainController(inference, bundle)
    freeze_for_root_comparison(controller)
    readout = controller.root._conv_output.model[-1]
    native_readout = {key: value.detach().clone() for key, value in readout.state_dict().items()}
    parameters = [p for p in controller.parameters() if p.requires_grad]
    keys = list(controller.state_dict())
    cases = [c for c in torch.load(args.cases, map_location='cpu', weights_only=False) if c['native_split'] == 'train']
    rows = []
    for checkpoint in args.candidates:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved.get('schema') not in (SCHEMA, 'motionbricks.unified_terrain.root_height_readout.v1'):
            raise ValueError('Unsupported research checkpoint schema')
        if saved['contract']['bundle_sha256'] != sha256(args.bundle):
            raise ValueError('Research checkpoint belongs to a different baseline')
        readout.load_state_dict(native_readout, strict=True)
        if saved['schema'] != SCHEMA:
            state = saved['root_height_readout']
            for key in ('weight', 'bias'):
                torch.testing.assert_close(state[key].to(native_readout[key])[[0, 2, 3, 4]],
                    native_readout[key][[0, 2, 3, 4]], atol=0, rtol=0)
            readout.load_state_dict(state, strict=True)
        controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
        before = state_digest(controller, keys)
        for case in cases:
            for start in (0, 12, 24, 36, 48):
                if start + 33 > len(case['world']):
                    continue
                torch.manual_seed(14294 + start)
                source = case['world'][None, start:start + 33].to(args.device)
                goal = source_terminal_goal(inference.global_motion_rep.extract_root(source))
                result = controller(source[:, :4], goal, case['scene'], training=True, straight_through=True)
                parts = controller.native_losses(result, source, source[:, :4])
                gradients = {}
                connections = {}
                for name in ('native_root', 'native_motion', 'token_ce'):
                    grad = (torch.autograd.grad(parts[name], parameters, retain_graph=True, allow_unused=True)
                            if parts[name].requires_grad else [None] * len(parameters))
                    connections[name] = any(g is not None for g in grad)
                    gradients[name] = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                                                 for p, g in zip(parameters, grad)])
                root = gradients['native_root']
                other = gradients['native_motion'] + gradients['token_ce']
                target = training_targets(inference, source, source[:, :4], result['observation'])
                predicted_root = inference.global_motion_rep.unnormalize(result['root_output']['pred_global_root_values'][:, :32])
                true_root = inference.global_motion_rep.unnormalize(target['global_motions'])
                row = dict(checkpoint=str(checkpoint.parent.name), checkpoint_sha256=sha256(checkpoint),
                    label=case['label'], start=start,
                    losses={name: float(parts[name].detach()) for name in gradients},
                    gradient_norms={name: float(value.norm()) for name, value in gradients.items()},
                    gradient_paths=connections,
                    root_vs_motion_cosine=cosine(root, gradients['native_motion']),
                    root_vs_token_cosine=cosine(root, gradients['token_ce']),
                    root_vs_other_cosine=cosine(root, other),
                    root_vs_total_cosine=cosine(root, root + other),
                    root_height_rms_m=float((predicted_root[:, 4:28, 1] - true_root[:, 4:28, 1]).square().mean().sqrt()))
                rows.append(row)
                write_json(args.output / 'progress.json', dict(rows=rows))
                del result, parts, gradients, root, other
        if state_digest(controller, keys) != before:
            raise ValueError('Read-only gradient audit changed weights')
    write_json(args.output / 'complete.json', dict(rows=rows, optimizer_updates=0, weights_unchanged=True,
        runtime_changed=False, note='Parameter-gradient directions are local evidence, not a motion-quality or causal-fix certificate.'))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'cases', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--candidates', type=Path, nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        if arguments.output.exists():
            write_json(arguments.output / 'failure.json', dict(error=repr(error)))
        raise
