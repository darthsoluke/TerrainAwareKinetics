"""Bounded deterministic L-BFGS convergence check for the unchanged Root loss."""
import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_pose_training import pose_memory
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.fit_native_codec_fullbank import group_weights
from scripts.fit_native_root_fullbank import stack_calls, weighted_root_loss, review_root_chain
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_unified_native_stages import fresh_kwargs, prepare
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


class EvaluationBudgetReached(RuntimeError):
    pass


def main(a):
    a.output.mkdir(parents=True, exist_ok=False); (a.output / 'checks').mkdir()
    torch.set_num_threads(4); torch.manual_seed(17123)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.parent, map_location='cpu', weights_only=False)
    receipt = json.loads(a.residual_gate.read_text())
    if (saved['schema'] != 'motionbricks.unified.native_route_curriculum.v1'
            or saved['contract']['stage'] != 'root'
            or saved['contract']['bundle_sha256'] != sha256(a.bundle)
            or saved['contract']['cases_sha256'] != sha256(a.cases)
            or receipt['parent_sha256'] != sha256(a.parent) or not receipt['selected_final']['gate_pass']):
        raise ValueError('Matched residual capacity evidence is required')
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True); controller.eval().requires_grad_(False)
    prepared = prepare(controller, torch.load(a.cases, map_location='cpu', weights_only=False),
                       saved['contract']['source_windows'], a.device)
    args = stack_calls([item['root_args'] for item in prepared])
    kwargs = stack_calls([fresh_kwargs(item['root_kwargs']) for item in prepared])
    targets = stack_calls([item['root_target'] for item in prepared])
    memory = stack_calls([{k: item['example'][k] for k in ('memory', 'valid')} for item in prepared])
    groups = group_weights(prepared)
    controller.root.requires_grad_(True)
    for key in ('root_stage1', 'root_stage2'):
        controller.adaptation[key].requires_grad_(True)
    names = [n for n, p in controller.named_parameters() if p.requires_grad]
    if names != saved['contract']['trainable']:
        raise ValueError('Native Root optimization scope changed')
    protected_keys = [n for n in controller.state_dict() if n not in names]
    protected = state_digest(controller, protected_keys)
    parameters = [p for p in controller.parameters() if p.requires_grad]
    optimizer = torch.optim.LBFGS(parameters, lr=1., max_iter=1, max_eval=30, history_size=10,
        line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
    contract = dict(saved['contract'], parent_sha256=sha256(a.parent),
        residual_gate_sha256=sha256(a.residual_gate), max_updates=100, max_loss_evaluations=400,
        max_exposures=400 * len(prepared), previous_root_updates=saved['actual_updates'] + saved['contract']['previous_root_updates'],
        optimizer='L-BFGS history10, strong Wolfe, exact unclipped native gradient; max100steps/400loss evaluations',
        stop_rule='Unchanged complete Root/codec gates every10accepted steps; stop at evaluation budget',
        note='Optimizer-only convergence study; all objectives, sources, weights and runtime unchanged')
    write_json(a.output / 'contract.json', contract)
    geometry = JointGeometryObjective(inference.global_motion_rep, a.device)
    baseline, _ = review_root_chain(controller, prepared, geometry)
    losses, curve = [], []
    evaluations, accepted = 0, 0
    budget_exhausted, passed, unchanged_step = False, False, False

    def closure():
        nonlocal evaluations
        if evaluations >= 400:
            raise EvaluationBudgetReached()
        optimizer.zero_grad(set_to_none=True)
        with pose_memory(controller, memory):
            prediction = controller.root(*args, **fresh_kwargs(kwargs))
        loss = weighted_root_loss(inference, targets, prediction, groups)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite native L-BFGS objective')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('Nonfinite native L-BFGS gradient')
        evaluations += 1; losses.append(float(loss.detach()))
        return loss

    for step in range(100):
        previous = [p.detach().clone() for p in parameters]
        try:
            optimizer.step(closure)
        except EvaluationBudgetReached:
            # A line-search trial is not an accepted optimizer state.
            with torch.no_grad():
                for parameter, value in zip(parameters, previous):
                    parameter.copy_(value)
            budget_exhausted = True
            break
        if all(torch.equal(p.detach(), old) for p, old in zip(parameters, previous)):
            unchanged_step = True
            break
        accepted += 1; del previous
        write_json(a.output / 'progress.json', dict(accepted_steps=accepted, loss_evaluations=evaluations,
            exposures=evaluations * len(prepared), last_evaluated_loss=losses[-1], gates=curve))
        if accepted % 10 == 0:
            final, passed = review_root_chain(controller, prepared, geometry)
            curve.append(dict(step=accepted, passed=sum(row['gate_pass'] for row in final), all_passed=passed))
            write_json(a.output / 'checks' / f'step{accepted:04d}-report.json', dict(rows=final, passed=passed))
            if passed:
                break
    final, passed = review_root_chain(controller, prepared, geometry)
    if state_digest(controller, protected_keys) != protected:
        raise ValueError('Protected module changed')
    checkpoint = a.output / 'weights.pt'
    torch.save(dict(schema=saved['schema'], contract=contract, model_state=controller.state_dict(),
        actual_updates=accepted), checkpoint)
    write_json(a.output / 'complete.json', dict(contract=contract, baseline=baseline, final=final,
        actual_updates=accepted, loss_evaluations=evaluations, primitive_exposures=evaluations * len(prepared),
        losses=losses, gates=curve, evaluation_budget_exhausted=budget_exhausted, unchanged_step=unchanged_step,
        training_bank_gate_pass=passed, protected_hash_unchanged=True, checkpoint_sha256=sha256(checkpoint), quality_approved=False))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'parent', 'residual-gate', 'cases', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
