"""Native source-Root Pose fitting after the complete causal Root/codec gate."""
import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_pose_training import pose_memory, pose_teacher
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.fit_native_codec_fullbank import group_weights
from scripts.fit_native_root_fullbank import stack_calls
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_unified_native_stages import fresh_kwargs, prepare, pose_review
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def sliced_batch(batch, ids):
    count = batch['groundtruth_pose_tokens'].shape[0]
    return {k: v[ids] if torch.is_tensor(v) and v.ndim and v.shape[0] == count else v for k, v in batch.items()}


def source_balanced_pose_loss(controller, batch, logits, groups):
    # Match the previous single-window native CE expectation, including zero-focus rows.
    rows = []
    for index in range(logits.shape[0]):
        row = sliced_batch(batch, [index])
        value = controller.inference._pose_model.loss(row, dict(pose_logits=logits[index:index + 1]))['loss'] \
            if row['focused_token_mask'].any() else logits[index:index + 1].sum() * 0
        rows.append(value)
    return sum(weight * sum(rows[i] for i in ids) / len(ids) for ids, weight in groups)


def main(a):
    if not 50 <= a.steps <= 300 or a.steps % 50:
        raise ValueError('Bounded full-bank Pose fit requires50..300updates')
    a.output.mkdir(parents=True, exist_ok=False); (a.output / 'checks').mkdir()
    torch.set_num_threads(4); torch.manual_seed(17121)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.parent, map_location='cpu', weights_only=False)
    gate = json.loads(a.parent.with_name('complete.json').read_text())
    upstream_pass = gate['training_bank_gate_pass'] and all(
        row['gate_pass'] and row.get('offline_source_token_chain', {}).get('gate_pass', False) for row in gate['final'])
    composition = None
    if a.root_chain_gate:
        composition = json.loads(a.root_chain_gate.read_text())
        if (composition['schema'] != 'motionbricks.native_root_bank_chain_audit.v1'
                or composition['parent_sha256'] != sha256(a.parent)
                or composition['cases_sha256'] != sha256(a.cases)
                or [dict(label=row['label'], start=row['start']) for row in composition['rows']]
                    != saved['contract']['source_windows']):
            raise ValueError('Full predicted-Root codec composition receipt must match')
        upstream_pass = all(row['offline_source_token_chain']['errors']['leg_relative_mpjpe_m'] <= .015
            and row['offline_source_token_chain']['root_xyz_rms_m'] <= .02
            and row['offline_source_token_chain']['skin']['body_prism_max_m'] <= .02
            and row['offline_source_token_chain']['skin']['unknown_body_fraction'] == 0
            and row['offline_source_token_chain']['skin']['uncertain_inside_samples'] == 0
            for row in composition['rows'])
    if (saved['schema'] != 'motionbricks.unified.native_route_curriculum.v1'
            or saved['contract']['stage'] != 'root'
            or saved['contract']['bundle_sha256'] != sha256(a.bundle)
            or saved['contract']['cases_sha256'] != sha256(a.cases)
            or gate['checkpoint_sha256'] != sha256(a.parent)
            or not upstream_pass
            or len(gate['final']) != len(saved['contract']['source_windows'])
            ):
        raise ValueError('Complete causal Root and codec composition gate required')
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True); controller.eval().requires_grad_(False)
    prepared = prepare(controller, torch.load(a.cases, map_location='cpu', weights_only=False),
                       saved['contract']['source_windows'], a.device)
    for item in prepared:
        item['case'] = dict(item['case'], label=item['window']['label'] + f'-f{item["window"]["start"]}')
    teachers = [item['source_teacher'] for item in prepared]
    example = dict(args=stack_calls([e['args'] for e in teachers]),
        kwargs=stack_calls([fresh_kwargs(e['kwargs']) for e in teachers]),
        target=dict(local_motions=torch.cat([e['target']['local_motions'] for e in teachers])),
        tokens=torch.cat([e['tokens'] for e in teachers]),
        memory=torch.cat([e['memory'] for e in teachers]), valid=torch.cat([e['valid'] for e in teachers]))
    groups = group_weights(prepared)
    with torch.no_grad():
        _, batched, batch, corrupted = pose_teacher(controller, example, 17121)
        singles = []
        for index, teacher in enumerate(teachers):
            with pose_memory(controller, teacher):
                singles.append(controller.pose(corrupted[index:index + 1], *teacher['args'][1:],
                    **fresh_kwargs(teacher['kwargs']))['pose_logits'][:, :8])
        single = torch.cat(singles)
        torch.testing.assert_close(batched, single, atol=1e-4, rtol=1e-5)
        token_differences = int((batched.argmax(-1) != single.argmax(-1)).sum())
        probability_max = float((batched.softmax(-1) - single.softmax(-1)).abs().max())
        if token_differences or probability_max > 1e-5:
            raise ValueError('Batching changed categorical predictions or probabilities')
        first = source_balanced_pose_loss(controller, batch, batched, groups)
        second = source_balanced_pose_loss(controller, batch, single, groups)
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
        parity = dict(logit_max=float((batched - single).abs().max()), objective_difference=float((first - second).abs()),
            token_differences=token_differences, probability_max=probability_max, logit_atol=1e-4, logit_rtol=1e-5)
    del batched, single, singles
    controller.pose.requires_grad_(True); controller.adaptation['pose'].requires_grad_(True)
    names = [n for n, p in controller.named_parameters() if p.requires_grad]
    protected_keys = [n for n in controller.state_dict() if n not in names]
    protected = state_digest(controller, protected_keys)
    parameters = [p for p in controller.parameters() if p.requires_grad]
    chosen = [ids[0] for ids, _ in groups[:4]]
    small = dict(memory=example['memory'][chosen], valid=example['valid'][chosen])
    small_args = stack_calls([teachers[i]['args'] for i in chosen])
    small_kwargs = stack_calls([fresh_kwargs(teachers[i]['kwargs']) for i in chosen])
    with pose_memory(controller, small):
        logits = controller.pose(corrupted[chosen], *small_args[1:], **fresh_kwargs(small_kwargs))['pose_logits'][:, :8]
    source_balanced_pose_loss(controller, sliced_batch(batch, chosen), logits, [(list(range(len(chosen))), 1.)]).backward()
    gradients = [p.grad.detach().cpu().clone() if p.grad is not None else None for p in parameters]
    controller.zero_grad(set_to_none=True)
    for index in chosen:
        teacher = teachers[index]
        with pose_memory(controller, teacher):
            logits = controller.pose(corrupted[index:index + 1], *teacher['args'][1:],
                **fresh_kwargs(teacher['kwargs']))['pose_logits'][:, :8]
        (source_balanced_pose_loss(controller, sliced_batch(batch, [index]), logits, [([0], 1.)]) / len(chosen)).backward()
    maximum = 0.
    for parameter, gradient in zip(parameters, gradients):
        if parameter.grad is None or gradient is None:
            if parameter.grad is not None or gradient is not None:
                raise ValueError('Batched Pose gradient presence differs')
        else:
            actual = parameter.grad.detach().cpu()
            torch.testing.assert_close(actual, gradient, atol=5e-6, rtol=1e-3)
            maximum = max(maximum, float((actual - gradient).abs().max()))
    parity.update(gradient_max=maximum, optimizer_updates=0, native_masked_token_objective=True)
    write_json(a.output / 'batch_parity.json', parity)
    controller.zero_grad(set_to_none=True); del gradients, logits, batch, corrupted
    contract = dict(saved['contract'], stage='pose', parent_sha256=sha256(a.parent),
        upstream_root_chain_receipt_sha256=sha256(a.root_chain_gate) if a.root_chain_gate else None,
        upstream_raw_root_gate_pass=gate['training_bank_gate_pass'],
        upstream_admission='actual predicted-Root/source-token codec composition' if composition else 'trajectory and composition',
        max_updates=a.steps, batch=len(prepared), max_exposures=a.steps * len(prepared), full_bank=True,
        trainable=names, trainable_count=sum(p.numel() for p in parameters),
        optimizer='fresh Adam5e-5, cosine to5e-7, full source-balanced bank gradient, clip0.5',
        source_root_pose_teacher=True, actual_actor_future_truth=False,
        stop_rule='Complete normal predicted-Root actor bank every50updates',
        note='Native cosine token corruption and source Root teacher; actor gets own predicted Root/tokens')
    write_json(a.output / 'contract.json', contract)
    geometry = JointGeometryObjective(inference.global_motion_rep, a.device)
    baseline = pose_review(controller, prepared, geometry, a.output / 'baseline')
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.steps, eta_min=5e-7)
    losses, curve, final, passed = [], [], [], False
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        teacher = dict(example, kwargs=fresh_kwargs(example['kwargs']))
        _, logits, batch, _ = pose_teacher(controller, teacher, 17121 + step)
        loss = source_balanced_pose_loss(controller, batch, logits, groups)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite native Pose objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid Pose gradient')
        optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
        if (step + 1) % 10 == 0:
            write_json(a.output / 'progress.json', dict(step=step + 1, max_steps=a.steps,
                exposures=(step + 1) * len(prepared), native_loss=float(loss.detach()),
                lr=optimizer.param_groups[0]['lr'], gates=curve))
        if (step + 1) % 50 == 0:
            final = pose_review(controller, prepared, geometry, a.output / 'checks' / f'step{step + 1:04d}')
            passed = all(row['values']['actor']['gate_pass'] for row in final)
            curve.append(dict(step=step + 1, passed=sum(row['values']['actor']['gate_pass'] for row in final), all_passed=passed))
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
        quality_approved=False, final_recordings=str(a.output / 'checks' / f'step{step + 1:04d}')))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'parent', 'cases', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--root-chain-gate', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
