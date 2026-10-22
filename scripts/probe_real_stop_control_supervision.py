"""Zero-update feasibility/gradient audit of real braking targets for the controller."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from scripts.pilot_joint_sdf_contact_preserve import load_model
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    expected = 'ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf'
    if sha256(args.checkpoint) != expected:
        raise ValueError('Approved orange checkpoint changed')
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model, _ = load_model(saved)
    model.requires_grad_(False)
    model.control.requires_grad_(True)
    initial_digest = state_digest(model)
    rep = build_native_representation(args.dataset / 'stats')
    data = TerrainLearningDataset(args.dataset, rep, 'train')
    audit = json.loads(args.audit.read_text())
    rows = []
    for example in audit['examples']:
        if example['split'] != 'train' or example['domain'] != 'bones' or not example['restart']:
            continue
        # A real moving prefix, braking trajectory, then a real quiet tail.
        begin = example['rest_start'] - 24
        if begin < 0 or begin+33 > example['frames']:
            continue
        item = data.full_reference_item(example['source_index'])
        if item['row']['source_id'] != example['source_id']:
            raise ValueError('Source cursor differs from the read-only audit')
        source = item['world'][None, begin:begin+33].cuda()
        input_speed = float((source[0, 3, [0, 2]]-source[0, 2, [0, 2]]).norm()*30)
        terminal_speed = float(torch.diff(source[0, -6:, [0, 2]], dim=0).norm(dim=-1).max()*30)
        if input_speed < .3 or terminal_speed > .12:
            continue
        history = source[:, :4]
        yaw = torch.atan2(history[:, -1, 4], history[:, -1, 3])
        intent = SoftCommandState().intent(history, history.new_zeros(1, 2), yaw, 8)
        with torch.no_grad():
            plain = model(history, intent, [item['scene']])['world'].clone()
            altered = source.clone()
            altered[:, 4:] += 3.
            mutated = model(history, intent, [item['scene']], source=altered)['world']
            torch.testing.assert_close(plain, mutated, atol=0, rtol=0)
        result = model(history, intent, [item['scene']], source=source, straight_through=True)
        torch.testing.assert_close(plain, result['world'], atol=3e-5, rtol=1e-5)
        losses = model.native_losses(result, source)
        model.zero_grad(set_to_none=True)
        losses['native_total'].backward()
        gradients = [p.grad for p in model.control.parameters() if p.grad is not None]
        norm = sum(g.double().square().sum() for g in gradients).sqrt()
        if not gradients or not torch.isfinite(norm) or norm <= 0:
            raise ValueError('The existing command adapter has no finite nonzero training signal')
        noncontrol = [name for name, p in model.named_parameters()
                      if not name.startswith('control.') and p.grad is not None]
        if noncontrol:
            raise ValueError('Frozen modules received gradients')
        if state_digest(model) != initial_digest:
            raise ValueError('Zero-update audit changed weights')
        row = dict(source_id=example['source_id'], source_start=begin, source_stop=begin+33,
            real_rest_interval=[example['rest_start'], example['rest_end']],
            input_speed_mps=input_speed, source_terminal_max_speed_mps=terminal_speed,
            generated_terminal_mean_speed_mps=float(torch.diff(plain[0, -6:, [0, 2]], dim=0).norm(dim=-1).mean()*30),
            command_gradient_norm=float(norm), future_target_actor_error=float((plain-mutated).abs().max()),
            native_losses={k:float(v.detach()) for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1},
            native_training_forward_error=float((plain-result['world']).abs().max()),
            frozen_parameter_gradients=noncontrol,
            weak_command='zero requested speed inferred from real upcoming stop; not recorded user keys',
            scope='Unmodified real flat-ground braking target; not stair-stop or full-skin certification')
        rows.append(row)
        print(json.dumps(row), flush=True)
        (args.output/'progress.json').write_text(json.dumps(dict(rows=rows), indent=2))
        if len(rows) >= args.cases:
            break
    if len(rows) != args.cases:
        raise ValueError('Insufficient genuine moving-to-rest targets for the requested audit')
    report = dict(rows=rows, updates=0, weights_sha256=expected, unchanged_model_state=initial_digest,
        postprocessing=None, future_truth_in_actor=False, quality_approved=False,
        proposal='Existing command adapter can be trained with real braking labels without changing approved decoder weights. Autoregressive stair-stop quality remains untested.')
    (args.output/'complete.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'dataset', 'audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cases', type=int, choices=range(1, 9), default=4)
    main(parser.parse_args())
