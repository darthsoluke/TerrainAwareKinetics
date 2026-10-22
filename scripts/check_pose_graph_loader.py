"""Isolate capture mode and the real DataLoader pinning thread without Adam updates."""

import argparse
import os
from pathlib import Path
import time

from scripts.run_pfnn_bias_workflow import digest, save
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(a):
    snapshot, available = gpu_snapshot()
    if len(set(a.gpus)) != 2 or any(gpu not in available for gpu in a.gpus):
        raise ValueError('Both selected GPUs must still be idle')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, a.gpus))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from torch.utils.data import DataLoader
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches, collate_primitives
    from motionbricks.data.reference_exposure_budget import ReferenceExposureBatches
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from motionbricks.data.parallel_pose_training import ParallelPoseMicrobatches
    from motionbricks.data.frozen_graph_pose_training import FrozenGraphPoseMicrobatches
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import restore_rng, rng_state, split_micro
    from scripts.train_native_mb_control import state_digest

    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    for device in range(2):
        torch.cuda.set_per_process_memory_fraction(.9, device)
    workspaces = [torch.empty(50*1024**3, dtype=torch.uint8, device=f'cuda:{index}') for index in range(2)]
    run = a.checkpoint.parents[2]
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    parent = torch.load(run/'full_training_v12/codec/weights.pt', map_location='cpu', mmap=True, weights_only=False)
    controllers, stages = [], []
    for index in range(2):
        inference, bundle = load_terrain_bundle(run.parent/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt', f'cuda:{index}')
        controller = UnifiedTerrainController(inference, bundle)
        controller.load_state_dict(parent['model_state'], strict=True)
        controller.codebook.copy_(controller.codec.get_codebook())
        controllers.append(controller)
        stages.append(NativeTrainingStage(controller, 'pose', cache_pose_tokens=True,
            coalesce_dilated_bias=True, isolate_pose_teacher_rng=True).train(True))
    torch.cuda.set_device(0)
    workspaces.clear()
    torch.cuda.empty_cache()
    for controller in controllers:
        controller.load_state_dict(saved['model_state'], strict=True)
    parameters = [parameter for parameter in controllers[0].parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    optimizer.load_state_dict(saved['optimizer_state'])
    restore_rng(saved['rng_state'])
    before = [state_digest(controller, list(controller.state_dict())) for controller in controllers]
    native_graph = torch.cuda.graph
    if a.mode == 'thread_local':
        def thread_local_graph(*args, **kwargs):
            return native_graph(*args, capture_error_mode='thread_local', **kwargs)
        torch.cuda.graph = thread_local_graph
    executor = (ParallelPoseMicrobatches if a.mode == 'native' else FrozenGraphPoseMicrobatches)(stages, capture_outputs=True)
    dataset = PrimitiveDataset([run/'cache_mixed_v1/data', run/'cache_pfnn_world_v2/data'], 'train')
    start = saved['actual_exposures']//256
    schedule = BalancedPrimitiveBatches(dataset, 256, 50000, seed=2333, start_step=start)
    sampler = ReferenceExposureBatches(schedule, dataset, 6144, start, 50000)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, collate_fn=collate_primitives,
        pin_memory=a.pin_memory == 'on', generator=torch.Generator().manual_seed(2333), persistent_workers=True)
    iterator = iter(loader)
    save(a.output/'state.json', dict(status='running', mode=a.mode, pin_memory=a.pin_memory,
        actual_optimizer_updates=0, source_updates=saved['actual_updates'], source_reference_steps=start))
    try:
        batch = next(iterator)
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        started = time.perf_counter()
        losses = executor.backward([split_micro(batch, offset, offset+3072) for offset in (0, 3072)],
            [2333+start*103+offset for offset in (0, 3072)])
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        elapsed = time.perf_counter()-started
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        observed = dict(losses=losses, logits=executor.last_outputs,
            gradients={name: None if p.grad is None else p.grad.detach().cpu().clone()
                       for name, p in controllers[0].named_parameters() if p.requires_grad},
            rng=rng_state(), sample_ids=batch['sample_ids'], gradient_norm=float(norm))
        torch.save(observed, a.output/'observed.pt')
        after = [state_digest(controller, list(controller.state_dict())) for controller in controllers]
        save(a.output/'complete.json', dict(passed=before == after, mode=a.mode, pin_memory=a.pin_memory,
            computation_completed=True, seconds=elapsed, losses=losses, model_unchanged=before == after,
            actual_optimizer_updates=0, production_budget_updates=0, workers=4, persistent_workers=True,
            preload_workspaces_gib_per_device=50, source_updates=saved['actual_updates'], source_reference_steps=start,
            checkpoint_sha256=digest(a.checkpoint), script_sha256=digest(Path(__file__)), gpus=a.gpus,
            gpu_snapshot_before=snapshot, peak_reserved=[torch.cuda.max_memory_reserved(index) for index in range(2)],
            production_approved=False, quality_approved=False))
    except Exception as error:
        queued_errors = []
        queue = getattr(iterator, '_data_queue', None)
        if queue is not None:
            while not queue.empty():
                try:
                    entry = queue.get_nowait()
                except Exception:
                    break
                value = entry[1] if isinstance(entry, tuple) and len(entry) == 2 else entry
                if hasattr(value, 'exc_msg'):
                    queued_errors.append(str(value.exc_msg))
        save(a.output/'failure.json', dict(error=repr(error), mode=a.mode, pin_memory=a.pin_memory,
            queued_loader_errors=queued_errors, actual_optimizer_updates=0, production_budget_updates=0,
            source_updates=saved['actual_updates'], checkpoint_sha256=digest(a.checkpoint),
            gpus=a.gpus, gpu_snapshot_before=snapshot))
        raise
    finally:
        executor.close()
        torch.cuda.graph = native_graph
        iterator._shutdown_workers()
        for controller in controllers:
            controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs=2, type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('native', 'global', 'thread_local'), required=True)
    parser.add_argument('--pin-memory', choices=('on', 'off'), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except Exception as error:
        if not (args.output/'failure.json').exists():
            save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=0, production_budget_updates=0))
        raise
