"""Measure a new exposure-equivalent codec configuration without Adam updates."""

import argparse
import os
from pathlib import Path
import time

from scripts.run_pfnn_bias_workflow import save, digest
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(a):
    snapshot, available = gpu_snapshot()
    if a.gpu not in available:
        raise ValueError('Selected GPU is no longer idle')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from motionbricks.data.exposure_codec_training import ordered_ema_chunks, original_order_indices
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches, collate_primitives
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch, restore_rng
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction, 0)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    run = a.checkpoint.parent.parent
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if saved['contract']['stage'] != 'codec':
        raise ValueError('A codec checkpoint is required')
    inference, bundle = load_terrain_bundle(run.parent / 'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt', 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True)
    controller.codebook.copy_(controller.codec.get_codebook())
    stage = NativeTrainingStage(controller, 'codec', coalesce_dilated_bias=True,
                                graph_dilated_convolutions=True).train(True)
    dataset = PrimitiveDataset([run/'cache_mixed_v1/data', run/'cache_pfnn_world_v2/data'], 'train')
    schedule = BalancedPrimitiveBatches(dataset, 256, 50000, seed=saved['contract']['seed'])
    ids = original_order_indices(schedule, dataset, saved['actual_updates'], a.micro_batch)
    batch = device_batch(collate_primitives([dataset[i] for i in ids]), 'cuda:0')
    rows = []
    try:
        for repeat in range(5):
            controller.load_state_dict(saved['model_state'], strict=True)
            controller.zero_grad(set_to_none=True)
            restore_rng(saved['rng_state'])
            torch.cuda.synchronize()
            started = time.perf_counter()
            with ordered_ema_chunks(controller):
                loss = stage(batch, saved['contract']['seed'])
                loss.backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            parameters = [p for p in controller.parameters() if p.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError('Nonfinite large-batch loss or gradient')
            rows.append(dict(seconds=elapsed, samples_per_second=a.micro_batch/elapsed,
                             loss=float(loss), gradient_norm=float(norm)))
            save(a.output/'state.json', dict(status='running', repeat=repeat+1,
                                            actual_optimizer_updates=0, production_budget_updates=0))
        controller.load_state_dict(saved['model_state'], strict=True)
        unchanged = all(torch.equal(value.cpu(), saved['model_state'][name])
                        for name, value in controller.state_dict().items())
        if not unchanged:
            raise ValueError('Diagnostic did not restore the initial model')
        save(a.output/'complete.json', dict(passed=True, rows=rows, micro_batch=a.micro_batch,
            memory_fraction=a.memory_fraction, gpu=a.gpu, gpu_snapshot_before=snapshot,
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            graph_signatures=len(stage._dilated_graph_execution.calls), ema_chunk_size=64,
            ema_calls_per_batch=a.micro_batch//64, initial_model_restored=unchanged,
            checkpoint_sha256=digest(a.checkpoint), actual_optimizer_updates=0,
            production_budget_updates=0, production_approved=False, quality_approved=False))
        save(a.output/'state.json', dict(status='complete', actual_optimizer_updates=0))
    finally:
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--micro-batch', type=int, choices=(64, 128, 256, 512, 1024), required=True)
    parser.add_argument('--memory-fraction', type=float, default=.9)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=0,
                                             production_budget_updates=0))
        raise
