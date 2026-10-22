"""Exercise the actual trainer, including preload, pinned workers, Adam and saves."""

import argparse
import json
import os
from pathlib import Path

from scripts.run_pfnn_bias_workflow import digest, save, read
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(a):
    snapshot, available = gpu_snapshot()
    if len(set(a.gpus)) != 2 or any(gpu not in available for gpu in a.gpus):
        raise ValueError('Both selected GPUs must still be idle')
    run = a.checkpoint.parents[2]
    if a.output.parent != run or not a.output.name.startswith('pose_loader_optimizer_'):
        raise ValueError('Diagnostic producer outputs must be separate from formal training')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, a.gpus))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    import scripts.train_unified_full_corpus as trainer
    from motionbricks.data.parallel_pose_training import ParallelPoseMicrobatches

    a.output.mkdir(exist_ok=False)
    source = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if source['contract']['stage'] != 'pose' or source['actual_updates'] != 2584:
        raise ValueError('Use the reviewed fixed formal Pose2584 checkpoint')
    input_path = a.resume_state or a.checkpoint
    saved = torch.load(input_path, map_location='cpu', mmap=True, weights_only=False)
    if a.resume_state:
        identity = a.resume_state.parent/'diagnostic-identity.json'
        if saved['actual_updates'] != source['actual_updates']+2:
            raise ValueError('Cold recovery must begin at the real two-update midpoint')
    else:
        identity = a.output/'diagnostic-identity.json'
        save(identity, dict(diagnostic_only=True, checkpoint_sha256=digest(input_path),
            production_budget_updates=0, mode=a.mode,
            scope='Permit only test-source/execution identity changes before generating admission evidence.'))
    options = argparse.Namespace(stage='pose', bundle=run.parent/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt',
        parent=run/'full_training_v12/codec/weights.pt', preflight=run/'preflight_world_v1/complete.json',
        output=a.output/'pose', cache=[run/'cache_mixed_v1', run/'cache_pfnn_world_v2'],
        resume=input_path, resume_migration=identity, corpus_audit=run/'full_training_v3/corpus_audit.json',
        steps=50000, batch=6144, micro_batch=3072, lr=5e-5, seed=2333, workers=4,
        checkpoint_every=24, validation_every=24, stop_after=source['actual_updates']+4,
        device='cuda:0', mechanics_only=False, cache_pose_tokens=True, coalesce_dilated_bias=True,
        graph_dilated_convolutions=False, cuda_memory_fraction=.9, root_parallel_devices=1,
        codec_exposure_pipeline=False, codec_pipeline_devices=2, pose_exposure_parallel=True,
        graph_frozen_pose_labels=a.mode == 'graph', await_resume=True)
    captures, executors = [], []
    native_validate = trainer.validate_implementation_migration
    native_init = ParallelPoseMicrobatches.__init__
    native_clip = torch.nn.utils.clip_grad_norm_
    native_step = torch.optim.Adam.step
    native_save = torch.save
    native_write = trainer.write_json

    def admission_for_test(value, contract, receipt, checkpoint_hash):
        # Only this isolated harness admits an unapproved execution candidate.
        # The real optimizer, loader, math and checkpoint functions remain live.
        if checkpoint_hash != digest(input_path) or receipt.get('diagnostic_only') is not True:
            raise ValueError('Diagnostic checkpoint identity differs')
        allowed = {'source_code_sha256', 'implementation_migration_sha256',
                   'frozen_pose_label_graphs', 'frozen_pose_capture_error_mode'}
        if {k: v for k, v in value['contract'].items() if k not in allowed} != {
                k: v for k, v in contract.items() if k not in allowed}:
            raise ValueError('Diagnostic candidate changed a learning, data or budget contract')

    def instrument_init(self, stages, *, capture_outputs=False):
        native_init(self, stages, capture_outputs=True)
        executors.append(self)

    def instrument_clip(parameters, *args, **kwargs):
        executor = executors[-1]
        captures.append(dict(raw_gradients={name: None if p.grad is None else p.grad.detach().cpu().clone()
            for name, p in executor.parameters[0].items()}, logits=executor.last_outputs))
        return native_clip(parameters, *args, **kwargs)

    def instrument_step(self, *args, **kwargs):
        value = native_step(self, *args, **kwargs)
        torch.cuda.synchronize(0)
        a.completed_updates += 1
        return value

    def instrument_save(value, path, *args, **kwargs):
        result = native_save(value, path, *args, **kwargs)
        if (isinstance(value, dict) and value.get('schema') == trainer.SCHEMA
                and value.get('actual_updates') == source['actual_updates']+2 and not a.resume_state):
            native_save(value, a.output/'midpoint.pt')
        return result

    def instrument_progress(path, value):
        native_write(path, value)
        if Path(path).name == 'progress.json':
            captures[-1].update({key: value[key] for key in ('actual_updates', 'actual_exposures',
                'equivalent_reference_updates', 'loss', 'gradient_norm', 'sample_digest')})
            native_save(captures[-1], a.output/f"observed_{value['actual_updates']}.pt")
            captures[-1] = {key: item for key, item in captures[-1].items() if key not in ('raw_gradients', 'logits')}

    trainer.validate_implementation_migration = admission_for_test
    ParallelPoseMicrobatches.__init__ = instrument_init
    torch.nn.utils.clip_grad_norm_ = instrument_clip
    torch.optim.Adam.step = instrument_step
    torch.save = instrument_save
    trainer.write_json = instrument_progress
    try:
        trainer.main(options)
    finally:
        trainer.validate_implementation_migration = native_validate
        ParallelPoseMicrobatches.__init__ = native_init
        torch.nn.utils.clip_grad_norm_ = native_clip
        torch.optim.Adam.step = native_step
        torch.save = native_save
        trainer.write_json = native_write
    endpoint = torch.load(a.output/'pose/latest.pt', map_location='cpu', mmap=True, weights_only=False)
    expected_count = 2 if a.resume_state else 4
    if a.completed_updates != expected_count or endpoint['actual_updates'] != source['actual_updates']+4:
        raise ValueError('The actual producer did not complete the bounded optimizer sequence')

    def differences(left, right, path=''):
        if torch.is_tensor(left):
            return [] if torch.equal(left, right) else [dict(path=path,
                max_abs=float((left.to(torch.float64)-right.to(torch.float64)).abs().max()))]
        if isinstance(left, np.ndarray):
            return [] if np.array_equal(left, right) else [dict(path=path)]
        if isinstance(left, dict):
            if left.keys() != right.keys():
                return [dict(path=path, kind='inventory')]
            return [item for key in left for item in differences(left[key], right[key], path+'.'+str(key))]
        if isinstance(left, (list, tuple)):
            if len(left) != len(right):
                return [dict(path=path, kind='length')]
            return [item for index, (x, y) in enumerate(zip(left, right))
                    for item in differences(x, y, path+'.'+str(index))]
        return [] if left == right else [dict(path=path)]

    computation_changes, endpoint_changes, validation_changes = [], [], []
    validation = [json.loads(line) for line in (a.output/'pose/validation.jsonl').read_text().splitlines()]
    if a.reference:
        reference = torch.load(a.reference/'pose/latest.pt', map_location='cpu', mmap=True, weights_only=False)
        for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'actual_updates',
                    'actual_exposures', 'sample_digest'):
            endpoint_changes.extend(differences(reference[key], endpoint[key], key))
        for row in captures:
            name = f"observed_{row['actual_updates']}.pt"
            computation_changes.extend(differences(torch.load(a.reference/name, map_location='cpu', weights_only=False),
                                                  torch.load(a.output/name, map_location='cpu', weights_only=False), name))
        reference_validation = [json.loads(line) for line in (a.reference/'pose/validation.jsonl').read_text().splitlines()]
        validation_changes = differences(reference_validation[-expected_count:], validation)
    code = Path(__file__).resolve().parents[1]
    save(a.output/'complete.json', dict(passed=not computation_changes and not endpoint_changes and not validation_changes,
        actual_trainer_entrypoint=True, diagnostic_admission_identity_override=True, production_budget_updates=0,
        actual_optimizer_updates=a.completed_updates, source_updates=source['actual_updates'],
        checkpoint_sha256=digest(a.checkpoint), mode=a.mode, capture_error_mode='thread_local' if a.mode == 'graph' else None,
        cold_process_resume=a.resume_state is not None, loader=dict(workers=4, pin_memory=True, persistent_workers=True),
        preload_workspaces_gib_per_device=50, global_batch=6144, compute_micro=3072, memory_fraction=.9,
        preload_receipt=read(a.output/'pose-await-resume.json'),
        endpoint_actual_updates=endpoint['actual_updates'], endpoint_actual_exposures=endpoint['actual_exposures'],
        computation_changes=computation_changes, endpoint_changes=endpoint_changes, validation_changes=validation_changes,
        validation=validation, progress=captures, reference_report_sha256=digest(a.reference/'complete.json') if a.reference else None,
        resume_state_sha256=digest(a.resume_state) if a.resume_state else None, gpus=a.gpus, gpu_snapshot_before=snapshot,
        source_sha256={name: digest(code/name) for name in ('scripts/train_unified_full_corpus.py',
            'motionbricks/data/frozen_graph_pose_training.py', 'motionbricks/data/dilated_graph_convolution.py',
            'scripts/check_pose_actual_loader_optimizer.py')}, production_approved=False, quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs=2, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('native', 'graph'), required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--resume-state', type=Path)
    args = parser.parse_args()
    args.completed_updates = 0
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=args.completed_updates,
                                                production_budget_updates=0))
        raise
