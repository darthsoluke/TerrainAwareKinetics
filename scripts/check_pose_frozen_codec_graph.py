"""Zero-update comparison of graph replay only inside frozen Pose label encoding."""

import argparse
import os
from pathlib import Path
import time

from scripts.run_pfnn_bias_workflow import digest, save
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(a):
    snapshot, available = gpu_snapshot()
    if a.gpu not in available:
        raise ValueError('Selected GPU is no longer idle')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    from motionbricks.data.dilated_graph_convolution import GraphConvolutionExecution
    from motionbricks.data.dilated_graph_context import graphed_dilated_convolutions
    from motionbricks.data.exposure_codec_training import original_order_indices
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches, collate_primitives
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch, restore_rng, rng_state

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.9, 0)
    torch.cuda.set_device(0)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    run = a.checkpoint.parents[2]
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    inference, bundle = load_terrain_bundle(run.parent/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt', 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True)
    controller.codebook.copy_(controller.codec.get_codebook())
    stage = NativeTrainingStage(controller, 'pose', cache_pose_tokens=True,
        coalesce_dilated_bias=True, isolate_pose_teacher_rng=True).train(True)
    dataset = PrimitiveDataset([run/'cache_mixed_v1/data', run/'cache_pfnn_world_v2/data'], 'train')
    schedule = BalancedPrimitiveBatches(dataset, 256, 50000, seed=2333)
    batches = [collate_primitives([dataset[index] for index in original_order_indices(schedule, dataset,
        saved['actual_exposures']//256+round_index*12, size)]) for round_index, size in enumerate((3072, 3072, 3072, 64))]
    initial = {name: value.detach().cpu().clone() for name, value in controller.state_dict().items()}

    def cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(cpu(item) for item in value)
        return value

    def changes(left, right, path=''):
        if torch.is_tensor(left):
            return [] if torch.equal(left, right) else [dict(path=path,
                max_abs=float((left.to(torch.float64)-right.to(torch.float64)).abs().max()))]
        if isinstance(left, np.ndarray):
            return [] if np.array_equal(left, right) else [dict(path=path)]
        if isinstance(left, dict):
            return [item for key in left for item in changes(left[key], right[key], path+'.'+str(key))]
        if isinstance(left, (list, tuple)):
            return [item for index, (x, y) in enumerate(zip(left, right)) for item in changes(x, y, path+'.'+str(index))]
        return [] if left == right else [dict(path=path)]

    def compute(batch, seed, training):
        stage._pose_tokens.clear()
        stage._pose_token_layouts.clear()
        stage.train(training)
        controller.zero_grad(set_to_none=True)
        value = device_batch(batch, 'cuda:0')
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = stage(value, seed)
        if training:
            loss.backward()
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        observed = dict(loss=float(loss), logits=cpu(controller.last_logits), rng=cpu(rng_state()),
            gradients={name: cpu(p.grad) for name, p in controller.named_parameters() if p.requires_grad},
            labels=cpu(stage._pose_tokens), layouts=stage._pose_token_layouts.copy())
        controller.zero_grad(set_to_none=True)
        return seconds, observed

    reference, starts, rows = [], [], []
    for index, batch in enumerate(batches):
        starts.append(rng_state())
        seconds, observed = compute(batch, 2333+index*3072, index < 3)
        reference.append(observed)
        rows.append(dict(batch=len(batch['world']), training=index < 3, native_seconds=seconds))
    execution = GraphConvolutionExecution()
    native_tokens = stage.pose_tokens

    def graph_tokens(batch, local):
        if controller.codec.training or any(p.requires_grad for p in controller.codec.parameters()):
            raise ValueError('Graph labels require the frozen eval codec')
        with graphed_dilated_convolutions(controller.codec, execution):
            return native_tokens(batch, local)

    stage.pose_tokens = graph_tokens
    errors = []
    try:
        for index, batch in enumerate(batches):
            restore_rng(starts[index])
            seconds, observed = compute(batch, 2333+index*3072, index < 3)
            errors.append(changes(reference[index], observed))
            rows[index].update(graph_seconds=seconds, differences=errors[-1], graph_signatures=len(execution.calls))
            save(a.output/'state.json', dict(status='running', rows=rows, actual_optimizer_updates=0))
        model_changes = changes(initial, {name: value.detach().cpu() for name, value in controller.state_dict().items()})
        code = Path(__file__).resolve().parents[1]
        save(a.output/'complete.json', dict(passed=not any(errors) and not model_changes,
            rows=rows, model_changes=model_changes, actual_optimizer_updates=0, production_budget_updates=0,
            checkpoint_sha256=digest(a.checkpoint), final_codec_parent_sha256=saved['contract']['parent_sha256'],
            code_sha256={name: digest(code/name) for name in ('motionbricks/data/dilated_graph_convolution.py',
                'motionbricks/data/dilated_graph_context.py', 'scripts/check_pose_frozen_codec_graph.py')},
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            gpu=a.gpu, gpu_snapshot_before=snapshot, production_approved=False, quality_approved=False,
            cache_persisted=False, parallel_capture_and_optimizer_proven=False))
    finally:
        del stage.pose_tokens
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=0, production_budget_updates=0))
        raise
