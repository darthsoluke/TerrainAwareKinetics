"""Bounded prefix training with genuine source or generated-history conditions."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from scripts.run_pfnn_bias_workflow import digest, save
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(args):
    snapshot, idle = gpu_snapshot()
    if args.gpu not in idle:
        raise ValueError('Selected GPU is occupied')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from motionbricks.data.full_corpus_inference import inference_receipt, load_candidate, tensor_digest
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.data.native_mb_observed_history import reencode_played_history
    from motionbricks.data.reactive_prefix_training import (
        configure_scope, prefix_command, prefix_loss, TrainingRootPathObservation)
    from motionbricks.data.reactive_training_data import select_records, selection_receipt, ReactiveSourceBank
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle

    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    expected = dict(updates_per_arm=64, global_batch=8, micro_batch=1, rollout_steps=7,
        executed_frames=4, source_windows_per_arm=512, supervised_prefixes_per_arm=3584,
        train_quotas=dict(pfnn=48, parc=24, bones=24), validation_quotas=dict(pfnn=12, parc=6, bones=6),
        runtime=dict(tokens=8, pose_passes=4, history_frames=4, fps=30, replan_frames=4),
        eligible_for_automatic_export=False, automatic_resume=False,
        conditioning_protocol='native-mb-velocity-online-v2', pose_decoder_terminal_velocity_override_index=31,
        new_immediate_intent_encoder=False)
    if args.arm not in plan['arms'] or any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('Invalid arm or exposure budget')
    source_files = [path for root in ('motionbricks', 'scripts', 'out/vendor/kimodo/kimodo')
        for path in Path(root).rglob('*.py')]
    sources = {str(path): digest(path) for path in sorted(source_files)}
    plan_hash = digest(args.plan)
    if args.mode == 'execute':
        proof = json.loads(args.preflight.read_text())
        if (not proof['passed'] or proof['optimizer_updates'] != 0 or
                proof['plan_sha256'] != plan_hash or proof['sources'] != sources):
            raise ValueError('Preflight source/plan binding differs')
    torch.set_num_threads(4)
    torch.manual_seed(plan['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(plan['cuda_memory_fraction'])
    started = time.monotonic()
    parent_hash, bundle_hash = digest(args.parent), digest(args.bundle)
    if parent_hash != plan['parent_sha256']:
        raise ValueError('Final parent checkpoint changed')
    parent = torch.load(args.parent, map_location='cpu', mmap=True, weights_only=False)
    receipts = parent['contract']['cache_receipts']
    if len(receipts) != len(args.cache):
        raise ValueError('Cache list differs from audited parent')
    for path, receipt in zip(args.cache, receipts):
        if (str(path.resolve()) != receipt['path'] or digest(path/'complete.json') != receipt['complete_sha256']
                or digest(path/'data/records.jsonl') != receipt['records_sha256']):
            raise ValueError('Audited cache identity changed')
    inference, bundle = load_terrain_bundle(args.bundle, 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    load_candidate(controller, parent, inference_receipt(parent, parent_hash, parent_hash, bundle_hash), bundle_hash)
    observer = TrainingRootPathObservation(controller)
    names = configure_scope(controller)
    parameters = [parameter for parameter in controller.parameters() if parameter.requires_grad]
    protected = [name for name in controller.state_dict() if name not in names]
    frozen_hash = tensor_digest({key: controller.state_dict()[key] for key in protected})
    initial_hash = tensor_digest(controller.state_dict())
    rep = inference.global_motion_rep
    cpu_rep = build_native_representation(args.mixed/'stats')
    bank = ReactiveSourceBank(args.cache, args.pfnn, args.mixed, cpu_rep)
    train = select_records(args.cache, 'train', plan['train_quotas'], seed=plan['seed'])
    validation = select_records(args.cache, 'validation', plan['validation_quotas'], seed=plan['seed'])
    save(args.output/'selection.json', dict(train=train, validation=validation,
        train_receipt=selection_receipt(train), validation_receipt=selection_receipt(validation)))
    selection_hash = digest(args.output/'selection.json')
    if args.mode == 'execute' and (proof['selection_sha256'] != selection_hash or
            proof['trainable_parameters'] != names or proof['parent_sha256'] != parent_hash
            or proof['bundle_sha256'] != bundle_hash):
        raise ValueError('Preflight data/model/scope differs')
    loaded = {}

    def item(selection):
        key = (selection['cache_index'], selection['row_index'])
        if key not in loaded:
            value = bank.load(selection)
            loaded[key] = dict(value, world=value['world'].to('cuda:0'), scene={
                key: tensor.to('cuda:0') if torch.is_tensor(tensor) else tensor
                for key, tensor in value['scene'].items()})
            save(args.output/'source_receipts.json', [value['receipt'] for value in loaded.values()])
        value = loaded[key]
        return value['world'], value['scene']

    contract = dict(schema=plan['schema'], plan=plan, plan_sha256=plan_hash, sources=sources,
        parent_sha256=parent_hash, bundle_sha256=bundle_hash, arm=args.arm,
        trainable_parameters=names, trainable_count=sum(p.numel() for p in parameters),
        frozen_sha256=frozen_hash, cache_receipts=receipts,
        command_semantics=plan['command_semantics'], eligible_for_automatic_export=False,
        quality_approved=False, delivery_approved=False)
    save(args.output/'contract.json', contract)
    save(args.output/'state.json', dict(pid=os.getpid(), mode=args.mode, gpu=args.gpu,
        optimizer_updates=0, gpu_snapshot=snapshot, stage='preparing'))

    def rollout(selection, history_kind, backward=False, scale=1.):
        source, scene = item(selection)
        history = reencode_played_history(source[:, :4], rep)
        rows, generated = [], []
        for step in range(plan['rollout_steps']):
            offset = step*plan['executed_frames']
            if history_kind == 'source_history':
                history = reencode_played_history(source[:, offset:offset+4], rep)
            command = prefix_command(source, offset)
            output = controller(history, command, scene, training=backward, straight_through=backward)
            losses = prefix_loss(rep, output['world'], history, source, offset)
            if not torch.isfinite(losses['total']):
                raise ValueError('Nonfinite prefix loss')
            emitted = output['world'][:, 4:8].detach().clone()
            if backward:
                (losses['total']*scale/plan['rollout_steps']).backward()
            rows.append({key: float(value.detach()) for key, value in losses.items()})
            generated.append(emitted.cpu())
            history = reencode_played_history(emitted, rep)
        return rows, torch.cat(generated, 1)

    def evaluate(tag):
        rows = []
        with torch.no_grad():
            for selection in validation:
                values, motion = rollout(selection, 'generated_history')
                source, _ = item(selection)
                target = source[:, 4:32].cpu()
                rows.append(dict(sample=selection, loss=float(np.mean([v['total'] for v in values])),
                    root_rms_m=float((motion[..., :3]-target[..., :3]).square().mean().sqrt()),
                    root_endpoint_error_m=float((motion[:, -1, :3]-target[:, -1, :3]).norm()),
                    source_distance_m=float((target[:, -1, [0,2]]-source[:, 3, [0,2]].cpu()).norm()),
                    generated_distance_m=float((motion[:, -1, [0,2]]-source[:, 3, [0,2]].cpu()).norm())))
        result = dict(tag=tag, rows=rows, mean_loss=float(np.mean([r['loss'] for r in rows])),
            mean_root_rms_m=float(np.mean([r['root_rms_m'] for r in rows])))
        save(args.output/f'{tag}.json', result)
        return result

    def frozen_check():
        if tensor_digest({key: controller.state_dict()[key] for key in protected}) != frozen_hash:
            raise ValueError('Frozen encoder/codebook/shared scene tensors changed')

    if args.mode == 'preflight':
        selected = train[0]
        source, scene = item(selected)
        history = reencode_played_history(source[:, :4], rep)
        command = prefix_command(source, 0)
        with torch.no_grad():
            native = controller(history, command, scene)['world'].clone()
        result = controller(history, command, scene, training=True, straight_through=True)['world']
        parity = float((native-result.detach()).abs().max())
        torch.testing.assert_close(native, result.detach(), atol=3e-5, rtol=0)
        del result
        for arm in plan['arms']:
            controller.zero_grad(set_to_none=True)
            for domain in ('pfnn', 'parc', 'bones'):
                rollout(next(row for row in train if row['record']['domain'] == domain),
                    arm, backward=True, scale=1/3)
            groups = {}
            for name, parameter in controller.named_parameters():
                if parameter.grad is not None:
                    if not parameter.requires_grad or not torch.isfinite(parameter.grad).all():
                        raise ValueError('Invalid gradient')
                    key = name.split('.')[0]
                    groups[key] = groups.get(key, 0.) + float(parameter.grad.double().square().sum())
            if any(groups.get(key, 0.) <= 0 for key in ('root', 'pose', 'codec')):
                raise ValueError(f'Missing required model gradient: {groups}')
            save(args.output/f'{arm}_gradients.json', groups)
        controller.zero_grad(set_to_none=True)
        frozen_check()
        if tensor_digest(controller.state_dict()) != initial_hash:
            raise ValueError('Preflight changed model tensors')
        save(args.output/'complete.json', dict(passed=True, optimizer_updates=0, sources=sources,
            plan_sha256=plan_hash, forward_parity_max=parity, weights_unchanged=True,
            selection_sha256=selection_hash, trainable_parameters=names,
            parent_sha256=parent_hash, bundle_sha256=bundle_hash,
            elapsed_seconds=time.monotonic()-started, allocated_bytes=torch.cuda.max_memory_allocated()))
        controller.close()
        return

    baseline = evaluate('baseline')
    optimizer = torch.optim.Adam(parameters, lr=plan['learning_rate'])
    rng = np.random.default_rng(plan['seed'])
    groups = {domain: [s for s in train if s['record']['domain'] == domain] for domain in ('pfnn','parc','bones')}
    updates, exposures = 0, 0
    sample_hash = hashlib.sha256()
    try:
        for step in range(plan['updates_per_arm']):
            if time.monotonic()-started > plan['maximum_seconds_per_arm']:
                raise TimeoutError('Pilot wall-time budget exhausted')
            controller.zero_grad(set_to_none=True)
            batch = [pool[int(rng.integers(len(pool)))] for domain, count in [('pfnn',4),('parc',2),('bones',2)]
                for pool in [groups[domain]] for _ in range(count)]
            losses = []
            for selection in batch:
                sample_hash.update(json.dumps([selection['cache_index'], selection['row_index']]).encode())
                values, _ = rollout(selection, args.arm, backward=True, scale=1/plan['global_batch'])
                losses.append(float(np.mean([v['total'] for v in values])))
            norm = torch.nn.utils.clip_grad_norm_(parameters, plan['gradient_clip'])
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError('Nonfinite optimizer gradient')
            optimizer.step()
            updates, exposures = step+1, exposures+len(batch)
            if updates == 1:
                frozen_check()
                if any(not torch.isfinite(value).all() for value in controller.state_dict().values()):
                    raise ValueError('First Adam update produced nonfinite model tensors')
            lr = plan['minimum_learning_rate']+(plan['learning_rate']-plan['minimum_learning_rate'])*(
                1+math.cos(math.pi*updates/plan['updates_per_arm']))/2
            for group in optimizer.param_groups:
                group['lr'] = lr
            state = dict(pid=os.getpid(), arm=args.arm, optimizer_updates=updates,
                source_window_exposures=exposures, supervised_prefixes=exposures*plan['rollout_steps'],
                loss=float(np.mean(losses)), gradient_norm=float(norm), elapsed_seconds=time.monotonic()-started,
                sample_sequence_sha256=sample_hash.hexdigest())
            save(args.output/'state.json', state)
            print(json.dumps(state), flush=True)
        frozen_check()
        torch.save(dict(schema=plan['schema'], contract=contract, model_state=controller.state_dict(),
            optimizer=optimizer.state_dict(), updates=updates, exposures=exposures), args.output/'weights.pt')
        final = evaluate('final')
        save(args.output/'complete.json', dict(completed=True, optimizer_updates=updates,
            source_window_exposures=exposures, supervised_prefixes=exposures*plan['rollout_steps'],
            baseline_mean_loss=baseline['mean_loss'], final_mean_loss=final['mean_loss'],
            sample_sequence_sha256=sample_hash.hexdigest(),
            protected_unchanged=True, checkpoint_sha256=digest(args.output/'weights.pt'),
            eligible_for_automatic_export=False, quality_approved=False,
            elapsed_seconds=time.monotonic()-started))
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error), optimizer_updates=updates,
            source_window_exposures=exposures, elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','parent','plan','output','pfnn','mixed'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cache', type=Path, nargs=2, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--mode', choices=('preflight','execute'), default='preflight')
    parser.add_argument('--arm', choices=('source_history','generated_history'), default='generated_history')
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir() and not (args.output/'failure.json').exists():
            save(args.output/'failure.json', dict(error=repr(error), mode=args.mode))
        raise
