"""Compare native Pose teachers with isolated concurrent CUDA teachers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess

from scripts.run_pfnn_bias_workflow import digest, save
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(a):
    snapshot, available = gpu_snapshot()
    if len(set(a.gpus)) != 2 or any(gpu not in available for gpu in a.gpus):
        raise ValueError('Two distinct freshly idle GPUs are required')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                   '--format=csv,noheader'], text=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, a.gpus))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import random
    import numpy as np
    import torch
    from motionbricks.data.exposure_codec_training import original_order_indices
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches, collate_primitives
    from motionbricks.data.pose_teacher_rng import isolated_pose_seed
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch, rng_state, restore_rng

    torch.set_num_threads(4)
    torch.manual_seed(2333); np.random.seed(2333); random.seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    run = a.checkpoint.parent.parent
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    controllers, stages = [], []
    for worker in range(2):
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
            coalesce_dilated_bias=True).train(True))
    dataset = PrimitiveDataset([run/'cache_mixed_v1/data', run/'cache_pfnn_world_v2/data'], 'train')
    schedule = BalancedPrimitiveBatches(dataset, 256, 50000, seed=2333)
    initial = [{name: value.detach().cpu().clone() for name, value in controller.state_dict().items()}
               for controller in controllers]

    def observe(worker, batch, seed):
        with torch.cuda.device(worker):
            controller = controllers[worker]
            controller.zero_grad(set_to_none=True)
            loss = stages[worker](device_batch(batch, f'cuda:{worker}'), seed)
            loss.backward()
            torch.cuda.synchronize(worker)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite diagnostic loss')
            result = dict(loss=loss.detach().cpu(), logits=controller.last_logits.detach().cpu().clone(),
                gradients={name: None if p.grad is None else p.grad.detach().cpu().clone()
                           for name, p in controller.named_parameters() if p.requires_grad},
                rng=torch.cuda.get_rng_state(worker))
            controller.zero_grad(set_to_none=True)
            return result

    def changes(left, right, path=''):
        if isinstance(left, dict):
            if left.keys() != right.keys():
                raise ValueError('Computation inventory changed')
            return [item for key in left for item in changes(left[key], right[key], path+'.'+key)]
        if left is None or right is None:
            return [] if left is right else [dict(path=path)]
        return [] if torch.equal(left, right) else [dict(path=path,
            max_abs=float((left.to(torch.float64)-right.to(torch.float64)).abs().max()))]

    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for size in (64, 3072, 1536):
            # The 64-example case covers validation; larger cases cover compute
            # batches and an unpadded partial batch, without any optimizer step.
            indices = [original_order_indices(schedule, dataset, 48+worker*12, 3072)[:size]
                       for worker in range(2)]
            batches = [collate_primitives([dataset[index] for index in selected]) for selected in indices]
            seeds = [2333+worker*3072 for worker in range(2)]
            before = rng_state()
            reference = []
            for worker in range(2):
                restore_rng(before)
                stages[worker].isolate_pose_teacher_rng = False
                reference.append(observe(worker, batches[worker], seeds[worker]))
            restore_rng(before)
            for stage in stages:
                stage.isolate_pose_teacher_rng = True
            futures = [pool.submit(observe, worker, batches[worker], seeds[worker]) for worker in range(2)]
            candidate = [future.result() for future in futures]
            differences = [changes(reference[worker], candidate[worker]) for worker in range(2)]
            cpu_unchanged = torch.equal(before['torch'], torch.get_rng_state())
            rows.append(dict(batch=size, differences=differences, cpu_rng_unchanged=cpu_unchanged,
                native_target_rng_equal=all(torch.equal(reference[w]['rng'], candidate[w]['rng']) for w in range(2))))
            save(a.output/'state.json', dict(status='running', rows=rows, actual_optimizer_updates=0))
            if any(differences) or not cpu_unchanged:
                raise ValueError('Concurrent isolated teacher differs from the native single-device computation')
            del reference, candidate, batches

    before = rng_state()
    try:
        with isolated_pose_seed(torch.empty(1, device='cuda:0'), 97):
            torch.rand(17, device='cuda:0')
            raise RuntimeError('injected scope failure')
    except RuntimeError as error:
        if str(error) != 'injected scope failure':
            raise
    exception_restored = torch.equal(before['torch'], torch.get_rng_state()) and all(
        torch.equal(x, y) for x, y in zip(before['cuda'], torch.cuda.get_rng_state_all()))
    frozen_changes = [changes(initial[worker], {name: value.detach().cpu()
        for name, value in controller.state_dict().items()}) for worker, controller in enumerate(controllers)]
    code = Path(__file__).resolve().parents[1]
    report = dict(passed=exception_restored and not any(frozen_changes), rows=rows,
        exception_restored=exception_restored, model_changes=frozen_changes,
        checkpoint_sha256=digest(a.checkpoint), source_sha256={name: digest(code/name) for name in (
            'motionbricks/data/pose_teacher_rng.py', 'motionbricks/data/unified_pose_training.py',
            'motionbricks/data/unified_full_training.py', 'scripts/check_pose_teacher_device_rng.py')},
        gpus=a.gpus, gpu_snapshot_before=snapshot, compute_processes_before=apps,
        actual_optimizer_updates=0, production_budget_updates=0, provisional_codec=True,
        parallel_optimizer_proven=False, production_approved=False, quality_approved=False,
        peak_reserved=[torch.cuda.max_memory_reserved(worker) for worker in range(2)])
    save(a.output/'complete.json', report)
    for controller in controllers:
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs=2, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=0,
            production_budget_updates=0, production_approved=False))
        raise
