"""Serial, concurrent and fresh-process Pose Adam proofs at one exposure policy."""

import argparse
import os
from pathlib import Path
import subprocess
import time

from scripts.run_pfnn_bias_workflow import digest, read, save
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(a):
    snapshot, available = gpu_snapshot()
    devices = 1 if a.phase == 'reference' else 2
    if len(set(a.gpus)) != devices or any(gpu not in available for gpu in a.gpus):
        raise ValueError('Selected diagnostic GPUs must be distinct and freshly idle')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                   '--format=csv,noheader'], text=True)
    code = Path(__file__).resolve().parents[1]
    run = a.checkpoint.parent.parent
    prerequisite = read(run/'pose_teacher_rng_v1/complete.json')
    if not prerequisite['passed'] or prerequisite['checkpoint_sha256'] != digest(a.checkpoint):
        raise ValueError('Required device-local teacher proof is absent or mismatched')
    for name in ('motionbricks/data/pose_teacher_rng.py', 'motionbricks/data/unified_pose_training.py',
                 'motionbricks/data/unified_full_training.py'):
        if prerequisite['source_sha256'][name] != digest(code/name):
            raise ValueError('Pose teacher implementation differs from its prerequisite proof')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, a.gpus))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import random
    import numpy as np
    import torch
    from motionbricks.data.exposure_codec_training import original_order_indices
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches, collate_primitives
    from motionbricks.data.parallel_pose_training import ParallelPoseMicrobatches, pose_micro_weights
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch, rng_state, restore_rng, split_micro, validation

    torch.set_num_threads(4)
    torch.manual_seed(2333); np.random.seed(2333); random.seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    controllers, stages = [], []
    for worker in range(devices):
        torch.cuda.set_per_process_memory_fraction(.9, worker)
        torch.cuda.set_device(worker)
        inference, bundle = load_terrain_bundle(
            run.parent/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt', f'cuda:{worker}')
        controller = UnifiedTerrainController(inference, bundle)
        controller.load_state_dict(saved['model_state'], strict=True)
        controller.codebook.copy_(controller.codec.get_codebook())
        controller.pose.init_embedding_from_codebooks(controller.codec.get_codebook(), None)
        controllers.append(controller)
        stages.append(NativeTrainingStage(controller, 'pose', cache_pose_tokens=True,
            coalesce_dilated_bias=True, isolate_pose_teacher_rng=True).train(True))
    torch.cuda.set_device(0)
    controller = controllers[0]
    parameters = [p for p in controller.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000, eta_min=5e-7)
    dataset = PrimitiveDataset([run/'cache_mixed_v1/data', run/'cache_pfnn_world_v2/data'], 'train')
    heldout = PrimitiveDataset([run/'cache_mixed_v1/data', run/'cache_pfnn_world_v2/data'], 'validation')
    schedule = BalancedPrimitiveBatches(dataset, 256, 50000, seed=2333)
    counts = [6144, 6144, 6144, 1536]
    reference_steps = [0, 24, 48, 72]
    batches = [collate_primitives([dataset[index] for index in original_order_indices(schedule, dataset, start, count)])
               for start, count in zip(reference_steps, counts)]

    def cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu(item) for item in value)
        return value

    def logical_rng():
        state = cpu(rng_state())
        if any(not torch.equal(state['cuda'][0], other) for other in state['cuda'][1:]):
            raise ValueError('Physical replicas disagree with the completed logical CUDA stream')
        # The serial reference has one physical device and two logical entries.
        state['cuda'] = [state['cuda'][0].clone() for _ in range(2)]
        return state

    def state(step):
        exposures = sum(counts[:step])
        return dict(model=cpu(controller.state_dict()), optimizer=cpu(optimizer.state_dict()),
            scheduler=cpu(scheduler.state_dict()), rng=logical_rng(), actual_updates=step,
            diagnostic_exposures=exposures, equivalent_reference_updates=exposures//256)

    def restore(value):
        for replica in controllers:
            replica.load_state_dict(value['model'], strict=True)
        optimizer.load_state_dict(value['optimizer'])
        scheduler.load_state_dict(value['scheduler'])
        rng = dict(value['rng'], cuda=value['rng']['cuda'][:devices])
        restore_rng(rng)

    def differences(left, right, path=''):
        if torch.is_tensor(left):
            if not torch.is_tensor(right) or left.shape != right.shape:
                return [dict(path=path, kind='shape_or_type')]
            return [] if torch.equal(left, right) else [dict(path=path,
                max_abs=float((left.to(torch.float64)-right.to(torch.float64)).abs().max()))]
        if isinstance(left, np.ndarray):
            return [] if np.array_equal(left, right) else [dict(path=path)]
        if isinstance(left, dict):
            if left.keys() != right.keys():
                return [dict(path=path, kind='keys')]
            return [item for key in left for item in differences(left[key], right[key], path+'.'+str(key))]
        if isinstance(left, (list, tuple)):
            if len(left) != len(right):
                return [dict(path=path, kind='length')]
            return [item for i, (x, y) in enumerate(zip(left, right)) for item in differences(x, y, path+'.'+str(i))]
        return [] if left == right else [dict(path=path)]

    reference = run/'pose_parallel_exposure_reference_v1'
    if a.phase == 'reference':
        initial = state(0)
        torch.save(initial, a.output/'initial.pt')
    else:
        initial = torch.load(reference/'initial.pt', map_location='cpu', weights_only=False)
        restore(initial)
    frozen = {name: value for name, value in initial['model'].items()
              if name not in {name for name, p in controller.named_parameters() if p.requires_grad}}
    first = 0
    if a.phase == 'resume':
        midpoint = torch.load(run/'pose_parallel_exposure_parallel_v1/midpoint.pt',
                              map_location='cpu', weights_only=False)
        restore(midpoint)
        first = midpoint['actual_updates']
        if first != 2 or any(stage._pose_tokens for stage in stages):
            raise ValueError('Resume requires the exact two-update checkpoint and empty label caches')
    executor = None if a.phase == 'reference' else ParallelPoseMicrobatches(stages, capture_outputs=True)
    rows, errors = [], []
    validation_result = None
    try:
        for index in range(first, 4):
            batch = batches[index]
            offsets = list(range(0, len(batch['world']), 3072))
            micros = [split_micro(batch, offset, offset+3072) for offset in offsets]
            weights = pose_micro_weights([len(micro['world']) for micro in micros])
            seeds = [2333+reference_steps[index]*103+offset for offset in offsets]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            if executor is None:
                losses, outputs = [], []
                for micro, seed, weight in zip(micros, seeds, weights):
                    loss = stages[0](device_batch(micro, 'cuda:0'), seed)
                    (loss * weight).backward()
                    losses.append(float(loss.detach()))
                    outputs.append(cpu(controller.last_logits))
            else:
                losses = executor.backward(micros, seeds)
                outputs = executor.last_outputs
            observed = dict(losses=losses, outputs=outputs, gradients={name: cpu(p.grad)
                for name, p in controller.named_parameters() if p.requires_grad}, rng=logical_rng(),
                sample_ids=batch['sample_ids'].tolist())
            if a.phase == 'reference':
                torch.save(observed, a.output/f'observed_{index}.pt')
            else:
                expected = torch.load(reference/f'observed_{index}.pt', map_location='cpu', weights_only=False)
                errors.extend(differences(expected, observed, f'update{index}'))
                del expected
            del observed, outputs
            norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
            if not torch.isfinite(norm) or norm <= 0 or not all(np.isfinite(losses)):
                raise ValueError('Nonfinite loss or zero/nonfinite Pose gradient')
            optimizer.step()
            a.completed_updates += 1
            scheduler.step(reference_steps[index]+counts[index]//256)
            torch.cuda.synchronize(0)
            rows.append(dict(index=index, examples=counts[index], losses=losses, gradient_norm=float(norm),
                seconds=time.perf_counter()-started, diagnostic_updates=a.completed_updates))
            save(a.output/'state.json', dict(status='running', actual_optimizer_updates=a.completed_updates,
                production_budget_updates=0, rows=rows, computation_changes=errors))
            if index == 1:
                torch.save(state(2), a.output/'midpoint.pt')
            if index == 2:
                validation_result = validation(stages[0], heldout, 64)
        endpoint = state(4)
        frozen_errors = differences(frozen, {name: endpoint['model'][name] for name in frozen})
        endpoint_errors, validation_errors = [], []
        if a.phase == 'reference':
            torch.save(endpoint, a.output/'endpoint.pt')
        else:
            endpoint_errors = differences(torch.load(reference/'endpoint.pt', map_location='cpu', weights_only=False), endpoint)
            validation_errors = differences(read(reference/'complete.json')['validation'], validation_result)
        save(a.output/'complete.json', dict(passed=not errors and not frozen_errors and not endpoint_errors and not validation_errors,
            phase=a.phase, rows=rows, computation_changes=errors, endpoint_changes=endpoint_errors,
            validation_changes=validation_errors, frozen_changes=frozen_errors, validation=validation_result,
            checkpoint_sha256=digest(a.checkpoint), prerequisite_sha256=digest(run/'pose_teacher_rng_v1/complete.json'),
            implementation_sha256=digest(code/'motionbricks/data/parallel_pose_training.py'),
            script_sha256=digest(Path(__file__)), actual_optimizer_updates=a.completed_updates,
            diagnostic_endpoint_exposures=sum(counts), diagnostic_endpoint_reference_steps=sum(counts)//256,
            global_batch=6144, compute_micro=3072, final_partial_batch=1536, memory_fraction=.9,
            gpus=a.gpus, gpu_snapshot_before=snapshot, compute_processes_before=apps,
            calibration_microbatches=executor.calibration_microbatches if executor else 0,
            rng_increments=executor.rng_increments if executor else {}, rng_from_reference_oracle=False,
            production_budget_updates=0, provisional_codec=True, cache_persisted=False,
            production_approved=False, quality_approved=False,
            peak_reserved=[torch.cuda.max_memory_reserved(worker) for worker in range(devices)]))
    finally:
        if executor:
            executor.close()
        for replica in controllers:
            replica.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--phase', choices=('reference', 'parallel', 'resume'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.completed_updates = 0
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=args.completed_updates,
                                            production_budget_updates=0))
        raise
