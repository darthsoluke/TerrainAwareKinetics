"""Deterministic full-bank convergence check with unchanged native codec loss."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_unified_native_codec import source_decode, evaluate
from scripts.probe_unified_native_stages import prepare
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def stack_examples(prepared):
    examples = [item['example'] for item in prepared]
    return dict(target={key: torch.cat([e['target'][key] for e in examples])
                       for key in ('local_motions', 'global_motions', 'executed_history_positions')},
        tokens=torch.cat([e['tokens'] for e in examples]), memory=torch.cat([e['memory'] for e in examples]),
        valid=torch.cat([e['valid'] for e in examples]))


def group_weights(prepared):
    groups = defaultdict(list)
    for index, item in enumerate(prepared):
        groups[item['window']['label']].append(index)
    flats = [key for key, ids in groups.items() if prepared[ids[0]]['case'].get('entry') not in ('up', 'down')]
    result = []
    for key, ids in groups.items():
        entry = prepared[ids[0]]['case'].get('entry')
        weight = .25 if entry in ('up', 'down') else .5 / len(flats)
        result.append((ids, weight))
    if abs(sum(weight for _, weight in result) - 1.) > 1e-10:
        raise ValueError('Expected one up/down and balanced flat sources')
    return result


def weighted_native_loss(controller, example, prediction, groups):
    loss = prediction.sum() * 0
    for indices, weight in groups:
        ids = torch.tensor(indices, device=prediction.device)
        target = {k: v.index_select(0, ids) for k, v in example['target'].items()}
        local = prediction.index_select(0, ids)
        loss = loss + weight * controller.loss_model.loss(target,
            dict(recon_state=local, l_commit=local.sum() * 0, perplexity=0.), None)['loss']
    return loss


def main(a):
    if not 50 <= a.steps <= 300 or a.steps % 50:
        raise ValueError('Full-bank diagnostic is bounded to50..300optimizer updates')
    a.output.mkdir(parents=True, exist_ok=False); (a.output / 'checks').mkdir()
    torch.set_num_threads(4); torch.manual_seed(17119)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.parent, map_location='cpu', weights_only=False)
    if (saved['schema'] != 'motionbricks.unified.native_route_curriculum.v1'
            or saved['contract']['stage'] != 'codec' or saved['contract']['source_fidelity_objective'] is not None
            or saved['contract']['cases_sha256'] != sha256(a.cases)
            or saved['contract']['bundle_sha256'] != sha256(a.bundle)):
        raise ValueError('Use the native-loss control with matched real sources')
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True); controller.eval().requires_grad_(False)
    prepared = prepare(controller, torch.load(a.cases, map_location='cpu', weights_only=False),
                       saved['contract']['source_windows'], a.device)
    for item in prepared:
        item['case'] = dict(item['case'], label=item['window']['label'] + f'-f{item["window"]["start"]}')
        item['example']['target'] = {k: v for k, v in item['example']['target'].items()
                                    if k not in ('motion_frame_mask', 'motion_interval_mask')}
    example, groups = stack_examples(prepared), group_weights(prepared)
    with torch.no_grad():
        batch = source_decode(controller, example)['recon_state']
        single = torch.cat([source_decode(controller, item['example'])['recon_state'] for item in prepared])
        torch.testing.assert_close(batch, single, atol=5e-5, rtol=1e-5)
        batch_loss = weighted_native_loss(controller, example, batch, groups)
        single_loss = weighted_native_loss(controller, example, single, groups)
        torch.testing.assert_close(batch_loss, single_loss, atol=1e-6, rtol=1e-5)
    write_json(a.output / 'batch_parity.json', dict(examples=len(prepared),
        local_forward_max=float((batch - single).abs().max()),
        objective_difference=float((batch_loss - single_loss).abs()), native_loss_unchanged=True))
    controller.codec.decoder.requires_grad_(True)
    names = [n for n, p in controller.named_parameters() if p.requires_grad]
    protected_keys = [n for n in controller.state_dict() if n not in names]
    protected = state_digest(controller, protected_keys)
    parameters = list(controller.codec.decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.steps, eta_min=5e-7)
    contract = dict(saved['contract'], parent_sha256=sha256(a.parent),
        max_updates=a.steps, batch=len(prepared), max_exposures=a.steps * len(prepared),
        source_fidelity_objective=None, full_bank=True,
        optimizer='fresh Adam5e-5, cosine to5e-7, full source-balanced bank gradient, clip0.5',
        source_group_weights=[dict(label=prepared[ids[0]]['window']['label'], windows=len(ids), weight=w) for ids, w in groups],
        note='Changed optimization, not model/loss/data; actual exposures differ from minibatch trials')
    write_json(a.output / 'contract.json', contract)
    geometry = JointGeometryObjective(inference.global_motion_rep, a.device)
    reviews = [(dict(i['case'], probe_start=i['window']['start']), i['source'], i['example']) for i in prepared]
    losses, curve, final = [], [], []
    passed = False
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = source_decode(controller, example)['recon_state']
        loss = weighted_native_loss(controller, example, prediction, groups)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite full-bank native loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid full-bank gradient')
        optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
        if (step + 1) % 10 == 0:
            write_json(a.output / 'progress.json', dict(step=step + 1, max_steps=a.steps,
                exposures=(step + 1) * len(prepared), native_loss=float(loss.detach()),
                lr=optimizer.param_groups[0]['lr'], gates=curve))
        if (step + 1) % 50 == 0:
            final = evaluate(controller, reviews, geometry, a.output / 'checks' / f'step{step + 1:04d}')
            passed = all(r['capacity_gate_pass'] for r in final)
            curve.append(dict(step=step + 1, passed=sum(r['capacity_gate_pass'] for r in final), all_passed=passed))
            write_json(a.output / 'checks' / f'step{step + 1:04d}-report.json', dict(rows=final, passed=passed))
            if passed:
                break
    if state_digest(controller, protected_keys) != protected:
        raise ValueError('Protected module changed')
    checkpoint = a.output / 'weights.pt'
    torch.save(dict(schema=saved['schema'], contract=contract, model_state=controller.state_dict(),
        optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
        torch_rng=torch.random.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), actual_updates=step + 1), checkpoint)
    write_json(a.output / 'complete.json', dict(contract=contract, actual_updates=step + 1,
        primitive_exposures=(step + 1) * len(prepared), losses=losses, gates=curve, final=final,
        training_bank_gate_pass=passed, protected_hash_unchanged=True, checkpoint_sha256=sha256(checkpoint),
        quality_approved=False, final_recordings=str(a.output / 'checks' / f'step{step + 1:04d}')))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'parent', 'cases', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
