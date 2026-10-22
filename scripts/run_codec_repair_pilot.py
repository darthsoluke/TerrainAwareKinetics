"""Preflight or explicitly execute the separately budgeted codec repair pilot."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.run_pfnn_bias_workflow import digest, read, save
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(args):
    snapshot, idle = gpu_snapshot()
    processes = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv'], text=True)
    if args.gpu not in idle:
        raise ValueError('Selected GPU is no longer idle')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from motionbricks.data.codec_repair_pilot import (
        PilotObjective, accounting, assert_frozen_scope, configure_scope, learning_rate,
        validate_plan, window_metrics)
    from motionbricks.data.full_corpus_inference import inference_receipt, load_candidate, tensor_digest
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches, collate_primitives
    from motionbricks.data.reference_exposure_budget import ReferenceExposureBatches
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch, rng_state, split_micro
    from scripts.diagnose_final_domain_generalization import select_windows

    plan = read(args.plan)
    plan_hash = validate_plan(plan)
    source = {str(path.name): digest(path) for path in (Path(__file__),
        Path(__file__).parents[1]/'motionbricks/data/codec_repair_pilot.py')}
    if args.mode == 'execute':
        if not args.preflight:
            raise ValueError('Execution requires the matching completed zero-update preflight')
        proof = read(args.preflight)
        if (not proof.get('passed') or proof.get('actual_updates') != 0
                or proof.get('arm') != args.arm or proof.get('plan_sha256') != plan_hash
                or proof.get('source_sha256') != source or not proof.get('weights_unchanged')):
            raise ValueError('Preflight does not bind this exact arm, plan and source')
    torch.set_num_threads(4)
    torch.manual_seed(plan['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.set_per_process_memory_fraction(plan['cuda_memory_fraction'])
    parent_hash, bundle_hash = digest(args.parent), digest(args.bundle)
    if parent_hash != plan['parent_sha256']:
        raise ValueError('Pilot must start from the fixed final candidate')
    parent = torch.load(args.parent, map_location='cpu', mmap=True, weights_only=False)
    receipts = parent['contract']['cache_receipts']
    if [str(path.resolve()) for path in args.cache] != [entry['path'] for entry in receipts]:
        raise ValueError('Pilot caches differ from the audited parent')
    for path, entry in zip(args.cache, receipts):
        if (digest(path/'complete.json') != entry['complete_sha256'] or
                digest(path/'data/records.jsonl') != entry['records_sha256']):
            raise ValueError('Cache receipt or sample index changed')
    inference, bundle = load_terrain_bundle(args.bundle, 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    load_candidate(controller, parent,
        inference_receipt(parent, parent_hash, parent_hash, bundle_hash), bundle_hash)
    stage = NativeTrainingStage(controller, 'codec', coalesce_dilated_bias=True)
    names = configure_scope(stage)
    protected = [name for name in controller.state_dict() if name not in names]
    frozen_hash = tensor_digest({key: controller.state_dict()[key] for key in protected})
    initial_hash = tensor_digest(controller.state_dict())
    objective = PilotObjective(controller, plan, args.arm)
    controller.loss_model.loss = objective
    parameters = [p for p in controller.parameters() if p.requires_grad]
    train = PrimitiveDataset([path/'data' for path in args.cache], 'train')
    first = plan['first_reference_step']
    end = first+plan['exposures_per_arm']//256
    reference = BalancedPrimitiveBatches(train, 256, end, seed=plan['seed'], start_step=first)
    batches = ReferenceExposureBatches(reference, train, plan['global_batch'], first, end)
    contract = dict(schema=plan['schema'], plan=plan, plan_sha256=plan_hash, source_sha256=source,
        parent_sha256=parent_hash, arm=args.arm, trainable_parameters=names,
        protected_sha256=frozen_hash, corpus_audit_sha256=parent['contract']['corpus_audit_sha256'],
        cache_receipts=parent['contract']['cache_receipts'], optimizer_is_fresh=True,
        deterministic_algorithms=True, deterministic_warn_only=True,
        source_teacher_only=True, actor_changed=False, quality_approved=False, delivery_approved=False)
    save(args.output/'contract.json', contract)
    save(args.output/'state.json', dict(pid=os.getpid(), mode=args.mode, arm=args.arm, gpu=args.gpu,
        gpu_snapshot=snapshot, compute_processes=processes, **accounting(plan, 0)))
    sample_hash = hashlib.sha256()
    updates = 0
    start = time.monotonic()
    optimizer = (torch.optim.Adam(parameters, lr=plan['learning_rate']) if args.mode == 'execute' else None)

    def check_protected():
        assert_frozen_scope(stage)
        current = tensor_digest({key: controller.state_dict()[key] for key in protected})
        if current != frozen_hash:
            raise ValueError('Protected model tensor changed')

    def validation():
        stage.eval()
        heldout = PrimitiveDataset([path/'data' for path in args.cache], 'validation')
        result = {}
        try:
            with torch.no_grad():
                for domain in ('pfnn', 'parc', 'bones'):
                    ids = select_windows(heldout, domain, 64)
                    batch = device_batch(collate_primitives([heldout[i] for i in ids]), 'cuda:0')
                    stage(batch)
                    # These are codec reconstruction diagnostics, not actor or skin acceptance.
                    from motionbricks.data.terrain_decoder_pilot import training_targets
                    target = training_targets(inference, batch['world'], batch['world'][:, :4], batch['observation'])
                    from motionbricks.data.codec_repair_pilot import fk_reconstruction_loss
                    fk = fk_reconstruction_loss(controller, objective.last['reconstruction'],
                        target['local_motions'], plan)
                    result[domain] = dict(examples=64, native_loss=float(objective.last['native_loss']),
                        scaled_fk_loss=float(fk), sample_ids=batch['sample_ids'].cpu().tolist(),
                        physical=window_metrics(controller, objective.last['reconstruction'],
                            batch['world'], batch['observation']))
        finally:
            stage.train()
        return result

    try:
        for indices in batches:
            if args.mode == 'preflight' and updates:
                break
            batch = device_batch(collate_primitives([train[index] for index in indices]), 'cuda:0')
            for domain, count in (('pfnn', 512), ('parc', 256), ('bones', 256)):
                if batch['domains'].count(domain) != count:
                    raise ValueError('Training domain quota changed')
            stage.zero_grad(set_to_none=True)
            value = 0.
            for offset in range(0, plan['global_batch'], plan['micro_batch']):
                assert_frozen_scope(stage)
                loss = stage(split_micro(batch, offset, offset+plan['micro_batch']))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite pilot loss')
                (loss*plan['micro_batch']/plan['global_batch']).backward()
                value += float(loss.detach())*plan['micro_batch']/plan['global_batch']
            norm = torch.nn.utils.clip_grad_norm_(parameters, plan['gradient_clip'])
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError('Invalid trainable gradient')
            assert_frozen_scope(stage)
            sample_hash.update(batch['sample_ids'].cpu().numpy().astype(np.int64).tobytes())
            if args.mode == 'preflight':
                stage.zero_grad(set_to_none=True)
                metrics = validation()
                baseline_hashes = {}
                for domain in ('pfnn', 'parc', 'bones'):
                    path = args.baseline/domain/'complete.json'
                    baseline = read(path)
                    if baseline['checkpoint_sha256'] != parent_hash or not baseline['weights_unchanged']:
                        raise ValueError('Validation baseline is not the fixed final model')
                    row = next(row for row in baseline['rows'] if row['split'] == 'validation')
                    actual = metrics[domain]
                    if actual['sample_ids'] != [window['sample_id'] for window in row['windows']]:
                        raise ValueError('Pilot validation samples differ from the baseline')
                    if abs(actual['native_loss']-row['native_losses']['codec']) > 5e-6:
                        raise ValueError('Pilot native validation loss differs before any update')
                    torch.testing.assert_close(torch.tensor(actual['physical']['per_window_leg_mpjpe_m']),
                        torch.tensor([window['source_codec_errors']['leg_relative_mpjpe_m']
                            for window in row['windows']]), atol=5e-6, rtol=1e-3)
                    torch.testing.assert_close(torch.tensor(actual['physical']['per_window_root_xyz_rms_m']),
                        torch.tensor([window['source_codec_root_xyz_rms_m'] for window in row['windows']]),
                        atol=5e-6, rtol=1e-3)
                    baseline_hashes[domain] = digest(path)
                check_protected()
                if tensor_digest(controller.state_dict()) != initial_hash:
                    raise ValueError('Zero-update preflight changed model state')
                save(args.output/'complete.json', dict(passed=True, arm=args.arm,
                    plan_sha256=plan_hash, source_sha256=source, weights_unchanged=True,
                    trainable_parameters=names, trainable_parameter_count=sum(p.numel() for p in parameters),
                    loss=value, gradient_norm_before_clip=float(norm),
                    sample_digest=sample_hash.hexdigest(), numerical_equivalence_claim=False,
                    validation=metrics, baseline_report_sha256=baseline_hashes,
                    diagnostic_forward_examples=plan['global_batch'], validation_examples=192,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    elapsed_seconds=time.monotonic()-start, **accounting(plan, 0)))
                return
            optimizer.step()
            updates += 1
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(plan, updates)
            progress = dict(arm=args.arm, loss=value, gradient_norm=float(norm),
                sample_digest=sample_hash.hexdigest(), elapsed_seconds=time.monotonic()-start,
                quality_approved=False, delivery_approved=False, **accounting(plan, updates))
            save(args.output/'progress.json', progress)
            if updates % plan['checkpoint_every_updates'] == 0:
                metrics = validation()
                check_protected()
                state = dict(contract=contract, model_state=controller.state_dict(),
                    optimizer=optimizer.state_dict(), rng=rng_state(), validation=metrics, **progress)
                checkpoint = args.output/f'checkpoint_{updates:04}.pt'
                temporary = checkpoint.with_suffix('.tmp.pt')
                torch.save(state, temporary)
                temporary.replace(checkpoint)
                save(args.output/f'validation_{updates:04}.json', dict(**progress, domains=metrics))
        if updates != plan['updates_per_arm']:
            raise ValueError('Pilot ended before its declared budget')
        check_protected()
        save(args.output/'complete.json', dict(**progress, completed=True, protected_hash_passed=True,
            final_checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
            geometry_evaluated=False, eligible_for_automatic_export=False))
    finally:
        controller.loss_model.loss = objective.native
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'bundle', 'parent', 'baseline', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cache', type=Path, nargs='+', required=True)
    parser.add_argument('--arm', choices=('native', 'native_fk'), required=True)
    parser.add_argument('--mode', choices=('preflight', 'execute'), default='preflight')
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        main(args)
    except Exception as error:
        progress_path = args.output/'progress.json'
        counts = ({key: read(progress_path)[key] for key in ('actual_updates', 'actual_exposures')}
                  if progress_path.exists() else dict(actual_updates=0, actual_exposures=0))
        save(args.output/'failure.json', dict(error=repr(error), **counts))
        raise
