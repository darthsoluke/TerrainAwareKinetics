"""Fit causal native Root on the complete verified codec bank before Pose."""
import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.played_motion_review import PUBLICATION_GEOMETRY, played_window
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_pose_training import pose_memory
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.audit_unified_stair_gait import difference, generate, source_token_intervention
from scripts.fit_native_codec_fullbank import group_weights
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_unified_native_stages import fresh_kwargs, prepare, root_review
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def stack_calls(values):
    """Batch captured single-character native calls, rejecting unequal metadata."""
    first = values[0]
    if torch.is_tensor(first):
        if first.ndim and first.shape[0] == 1:
            return torch.cat(values, dim=0)
        if not all(torch.equal(first, other) for other in values[1:]):
            raise ValueError('Unbatched tensor differs between requests')
        return first
    if isinstance(first, dict):
        if not all(first.keys() == other.keys() for other in values):
            raise ValueError('Native request keys differ')
        return {key: stack_calls([value[key] for value in values]) for key in first}
    if isinstance(first, (tuple, list)):
        if not all(len(first) == len(other) for other in values):
            raise ValueError('Native request lengths differ')
        return type(first)(stack_calls([value[i] for value in values]) for i in range(len(first)))
    if not all(first == other for other in values[1:]):
        raise ValueError('Native request metadata differs')
    return first


def weighted_root_loss(inference, targets, output, groups):
    result = output['pred_global_root_values'].sum() * 0
    count = targets['global_motions'].shape[0]
    for ids, weight in groups:
        target = {k: v[ids] for k, v in targets.items()}
        prediction = {k: v[ids] if torch.is_tensor(v) and v.ndim and v.shape[0] == count else v
                      for k, v in output.items()}
        result = result + weight * inference._root_model.loss(target, prediction)['loss']
    return result


def checked_forward_difference(actual, expected):
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-5)
    # Native duration masking uses matching -inf logits; subtracting them makes NaN.
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    return float((actual[finite] - expected[finite]).abs().max()) if finite.any() else 0.


@torch.no_grad()
def review_root_chain(controller, prepared, geometry):
    rows = root_review(controller, prepared)
    if all(row['gate_pass'] for row in rows):
        for row, item in zip(rows, prepared):
            with source_token_intervention(controller.inference, item['example']['tokens']):
                output = generate(controller, item['source'][:, :4], item['example']['command'],
                                  item['case']['scene'], item['example']['tokens'])
            future = output['world'][:, 4:28]
            errors = difference(controller.inference.global_motion_rep, future, item['source'][:, 4:28])
            rms = float((future[..., :3] - item['source'][:, 4:28, :3]).square().mean().sqrt())
            skin = exact_skin_metrics(geometry, played_window(item['source'][:, :4], output['world']), item['case']['scene'])
            passed = (errors['leg_relative_mpjpe_m'] <= .015 and rms <= .02
                      and skin['body_prism_max_m'] <= .02 and skin['unknown_body_fraction'] == 0
                      and skin['uncertain_inside_samples'] == 0)
            row['offline_source_token_chain'] = dict(errors=errors, root_xyz_rms_m=rms, skin=skin, gate_pass=passed)
            row['gate_pass'] = row['gate_pass'] and passed
    return rows, all(row['gate_pass'] for row in rows)


def main(a):
    if not 50 <= a.steps <= 300 or a.steps % 50:
        raise ValueError('Bounded full-bank Root fit requires50..300updates')
    a.output.mkdir(parents=True, exist_ok=False); (a.output / 'checks').mkdir()
    torch.set_num_threads(4); torch.manual_seed(17120)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.parent, map_location='cpu', weights_only=False)
    gate = json.loads(a.codec_gate.read_text())
    codec_hash = saved['contract'].get('codec_checkpoint_sha256', saved['contract']['parent_sha256']) \
        if a.continue_root else sha256(a.parent)
    if (saved['schema'] != 'motionbricks.unified.native_route_curriculum.v1'
            or saved['contract']['stage'] != ('root' if a.continue_root else 'codec')
            or saved['contract']['bundle_sha256'] != sha256(a.bundle)
            or saved['contract']['cases_sha256'] != sha256(a.cases)
            or gate['schema'] != 'motionbricks.codec_publication_gate.v1'
            or gate['geometry_contract'] != PUBLICATION_GEOMETRY
            or gate['source_checkpoint_sha256'] != codec_hash
            or gate['cases_sha256'] != sha256(a.cases)
            or gate['source_windows'] != saved['contract']['source_windows']
            or len(gate['rows']) != len(gate['source_windows'])
            or not gate['training_bank_gate_pass']):
        raise ValueError('Complete codec publication gate and provenance must match')
    if not all(r['errors']['leg_relative_mpjpe_m'] <= .015 and r['root_xyz_rms_m'] <= .02
               and r['skin']['body_prism_max_m'] <= .02 and r['skin']['unknown_body_fraction'] == 0
               and r['skin']['uncertain_inside_samples'] == 0 for r in gate['rows']):
        raise ValueError('Upstream geometry thresholds changed or failed')
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
    with torch.no_grad():
        with pose_memory(controller, memory):
            batched = controller.root(*args, **fresh_kwargs(kwargs))
        singles = []
        for item in prepared:
            with pose_memory(controller, item['example']):
                singles.append(controller.root(*item['root_args'], **fresh_kwargs(item['root_kwargs'])))
        parity = {}
        for key in ('pred_global_root_values', 'num_token_logits'):
            single = torch.cat([value[key] for value in singles])
            parity[key] = checked_forward_difference(batched[key], single)
        reference = sum(weight * sum(inference._root_model.loss(prepared[i]['root_target'], singles[i])['loss']
                        for i in ids) / len(ids) for ids, weight in groups)
        objective = weighted_root_loss(inference, targets, batched, groups)
        torch.testing.assert_close(objective, reference, atol=1e-6, rtol=1e-5)
        parity['objective_difference'] = float((objective - reference).abs())
    del batched, singles
    controller.root.requires_grad_(True)
    for key in ('root_stage1', 'root_stage2'):
        controller.adaptation[key].requires_grad_(True)
    names = [n for n, p in controller.named_parameters() if p.requires_grad]
    protected_keys = [n for n in controller.state_dict() if n not in names]
    protected = state_digest(controller, protected_keys)
    parameters = [p for p in controller.parameters() if p.requires_grad]
    # Compare full native parameter gradients on four distinct source windows.
    chosen = [ids[0] for ids, _ in groups[:4]]
    small_memory = {k: v[chosen] for k, v in memory.items()}
    small_args = stack_calls([prepared[i]['root_args'] for i in chosen])
    small_kwargs = stack_calls([fresh_kwargs(prepared[i]['root_kwargs']) for i in chosen])
    small_targets = stack_calls([prepared[i]['root_target'] for i in chosen])
    with pose_memory(controller, small_memory):
        output = controller.root(*small_args, **fresh_kwargs(small_kwargs))
    inference._root_model.loss(small_targets, output)['loss'].backward()
    gradients = [p.grad.detach().clone() if p.grad is not None else None for p in parameters]
    controller.zero_grad(set_to_none=True)
    for index in chosen:
        item = prepared[index]
        with pose_memory(controller, item['example']):
            output = controller.root(*item['root_args'], **fresh_kwargs(item['root_kwargs']))
        (inference._root_model.loss(item['root_target'], output)['loss'] / len(chosen)).backward()
    maximum = 0.
    for p, grad in zip(parameters, gradients):
        if grad is None or p.grad is None:
            if grad is not None or p.grad is not None:
                raise ValueError('Batched gradient presence differs')
        else:
            torch.testing.assert_close(p.grad, grad, atol=5e-6, rtol=1e-3)
            maximum = max(maximum, float((p.grad - grad).abs().max()))
    parity.update(gradient_max=maximum, future_root_pose_conditions_absent=True, optimizer_updates=0)
    write_json(a.output / 'batch_parity.json', parity)
    controller.zero_grad(set_to_none=True); del gradients, output
    previous_updates = saved['actual_updates'] + saved['contract'].get('previous_root_updates', 0) if a.continue_root else 0
    contract = dict(saved['contract'], stage='root', parent_sha256=sha256(a.parent),
        codec_checkpoint_sha256=codec_hash, previous_root_updates=previous_updates,
        upstream_publication_receipt_sha256=sha256(a.codec_gate), max_updates=a.steps,
        batch=len(prepared), max_exposures=a.steps * len(prepared), full_bank=True,
        trainable=names, trainable_count=sum(p.numel() for p in parameters),
        optimizer=('retained Adam moments1e-5' if a.continue_root else 'fresh Adam5e-5')
            + ', cosine to5e-7, full source-balanced bank gradient, clip0.5',
        source_root_pose_teacher=False, root_gate_includes_source_token_codec_chain=True,
        actual_actor_future_truth=False, geometry_contract=PUBLICATION_GEOMETRY,
        stop_rule='Complete training-bank Root and source-token codec-chain gates every50updates',
        note='Native Root causal-source convergence only; not generated-history or WASD approval')
    write_json(a.output / 'contract.json', contract)
    geometry = JointGeometryObjective(inference.global_motion_rep, a.device)
    baseline, _ = review_root_chain(controller, prepared, geometry)
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    if a.continue_root:
        if saved['contract']['trainable'] != names:
            raise ValueError('Continuation parameter order differs')
        optimizer.load_state_dict(saved['optimizer_state'])
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] = 1e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.steps, eta_min=5e-7)
    losses, curve, final, passed = [], [], [], False
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        with pose_memory(controller, memory):
            output = controller.root(*args, **fresh_kwargs(kwargs))
        loss = weighted_root_loss(inference, targets, output, groups)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite native Root objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid Root gradient')
        optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
        if (step + 1) % 10 == 0:
            write_json(a.output / 'progress.json', dict(step=step + 1, max_steps=a.steps,
                exposures=(step + 1) * len(prepared), native_loss=float(loss.detach()),
                lr=optimizer.param_groups[0]['lr'], gates=curve))
        if (step + 1) % 50 == 0:
            final, passed = review_root_chain(controller, prepared, geometry)
            curve.append(dict(step=step + 1, passed=sum(r['gate_pass'] for r in final), all_passed=passed))
            write_json(a.output / 'checks' / f'step{step + 1:04d}-report.json', dict(rows=final, passed=passed))
            if passed:
                break
    if state_digest(controller, protected_keys) != protected:
        raise ValueError('Protected module changed')
    checkpoint = a.output / 'weights.pt'
    torch.save(dict(schema=saved['schema'], contract=contract, model_state=controller.state_dict(),
        optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
        torch_rng=torch.random.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), actual_updates=step + 1), checkpoint)
    write_json(a.output / 'complete.json', dict(contract=contract, baseline=baseline, final=final,
        actual_updates=step + 1, primitive_exposures=(step + 1) * len(prepared), losses=losses, gates=curve,
        training_bank_gate_pass=passed, protected_hash_unchanged=True, checkpoint_sha256=sha256(checkpoint),
        quality_approved=False))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'parent', 'codec-gate', 'cases', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--continue-root', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
