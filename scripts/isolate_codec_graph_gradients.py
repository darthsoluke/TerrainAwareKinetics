"""Isolate raw-gradient drift before clipping or Adam; never changes production."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
from unittest.mock import patch


def main(a):
    from scripts.probe_full_training_execution import save
    from scripts.run_pfnn_parallel_workflow import gpu_snapshot
    snapshot, available = gpu_snapshot()
    if a.gpu not in available: raise ValueError('Selected GPU is occupied')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from motionbricks.data.dilated_graph_convolution import GraphConvolutionExecution
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches, collate_primitives
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch, split_micro, rng_state, restore_rng
    from scripts.train_native_mb_control import state_digest
    from scripts.train_native_mb import sha256
    _, available = gpu_snapshot()
    if a.gpu not in available: raise ValueError('Selected GPU became occupied')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.set_per_process_memory_fraction(.1, 0)
    run = a.checkpoint.parent.parent
    base = run.parent
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    inference, bundle = load_terrain_bundle(base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt', 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True)
    controller.codebook.copy_(controller.codec.get_codebook())
    stage = NativeTrainingStage(controller, 'codec', coalesce_dilated_bias=True).train(True)
    original_hash = state_digest(controller)
    dataset = PrimitiveDataset([run/'cache_mixed_v1/data', run/'cache_pfnn_world_v2/data'], 'train')
    schedule = BalancedPrimitiveBatches(dataset, 256, saved['actual_updates']+1,
        seed=saved['contract']['seed'], start_step=saved['actual_updates'])
    batch = collate_primitives([dataset[i] for i in next(iter(schedule))])
    indices = []
    for domain, count in (('pfnn', 32), ('parc', 16), ('bones', 16)):
        indices += [i for i, value in enumerate(batch['domains']) if value == domain][:count]
    def select(value):
        if torch.is_tensor(value): return value[indices]
        if isinstance(value, dict): return {key: select(item) for key, item in value.items()}
        return [value[i] for i in indices]
    micro = device_batch(select(batch), 'cuda:0')
    target_name = 'codec.decoder.model.6'
    target = dict(controller.named_modules())[target_name]
    captures = {}
    def capture(_module, inputs, output):
        captures['input'] = inputs[0].detach().cpu().clone()
        captures['input_stride'] = list(inputs[0].stride())
        def upstream(value):
            captures['upstream'] = value.detach().cpu().clone()
            captures['upstream_stride'] = list(value.stride())
        output.register_hook(upstream)

    def cpu(value):
        return value.detach().cpu().clone() if value is not None else None

    rows = []
    def execute(label, execution):
        controller.load_state_dict(saved['model_state'], strict=True)
        controller.zero_grad(set_to_none=True)
        restore_rng(saved['rng_state'])
        captures.clear()
        hook = target.register_forward_hook(capture)
        try:
            torch.cuda.synchronize(); started = time.perf_counter()
            with (patch('motionbricks.data.dilated_bias_convolution.BatchedBiasNativeBackward', execution)
                    if execution else nullcontext()):
                loss = stage(micro, saved['contract']['seed']+saved['actual_updates']*103)
                (loss*.25).backward()
            torch.cuda.synchronize(); elapsed = time.perf_counter()-started
        finally:
            hook.remove()
        raw = {name: cpu(value.grad) for name, value in controller.named_parameters() if value.requires_grad}
        observed = dict(loss=float(loss.detach()), output=cpu(controller.last_local), gradients=raw,
            quantizer={name: cpu(value) for name, value in controller.named_buffers() if name.startswith('codec.quantizer.')},
            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(), target=dict(captures),
            sample_ids=micro['sample_ids'].tolist())
        torch.save(observed, a.output/(label+'.pt'))
        row = dict(label=label, seconds=elapsed, allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(), graph_signatures=len(execution.calls) if execution else 0)
        rows.append(row); save(a.output/'state.json', dict(status='running', rows=rows, actual_optimizer_updates=0))
        return observed

    def compare(left, right):
        gradient_differences = []
        for name, expected in left['gradients'].items():
            actual = right['gradients'][name]
            if expected is None or actual is None:
                if (expected is None) != (actual is None): raise ValueError('Gradient scope changed')
            elif not torch.equal(expected, actual):
                gradient_differences.append(dict(name=name, max_abs=float((expected-actual).abs().max())))
        return dict(loss_equal=left['loss']==right['loss'], output_equal=torch.equal(left['output'],right['output']),
            raw_gradient_differences=gradient_differences,
            quantizer_equal=all(torch.equal(value,right['quantizer'][name]) for name,value in left['quantizer'].items()),
            rng_equal=torch.equal(left['rng_cpu'],right['rng_cpu']) and torch.equal(left['rng_cuda'],right['rng_cuda']),
            target_input_equal=torch.equal(left['target']['input'],right['target']['input']),
            target_upstream_equal=torch.equal(left['target']['upstream'],right['target']['upstream']),
            target_input_stride_equal=left['target']['input_stride']==right['target']['input_stride'],
            target_upstream_stride_equal=left['target']['upstream_stride']==right['target']['upstream_stride'])

    try:
        execute('native_warmup', None)
        reference = execute('native_reference', None)
        replay = execute('native_replay', None)
        forward = GraphConvolutionExecution(capture_backward=False)
        forward_result = execute('forward_graph_cold', forward)
        forward_warm = execute('forward_graph_warm', forward)
        native_with_forward = execute('native_with_forward_cache', None)
        both = GraphConvolutionExecution(capture_backward=True)
        both_result = execute('both_graph_cold', both)
        both_warm = execute('both_graph_warm', both)
        native_with_both = execute('native_with_both_caches', None)
        comparisons = {name: compare(reference,result) for name,result in [
            ('native_replay',replay),('forward_graph_cold',forward_result),('forward_graph_warm',forward_warm),
            ('native_with_forward_cache',native_with_forward),('both_graph_cold',both_result),
            ('both_graph_warm',both_warm),('native_with_both_caches',native_with_both)]}
        controller.load_state_dict(saved['model_state'],strict=True);controller.zero_grad(set_to_none=True)
        if state_digest(controller)!=original_hash:raise ValueError('Original model tensors not restored')
        save(a.output/'complete.json',dict(completed=True,comparisons=comparisons,rows=rows,
            target_layer=target_name,checkpoint_sha256=sha256(a.checkpoint),source_updates=saved['actual_updates'],
            actual_optimizer_updates=0,model_restored=True,allocation=snapshot,
            implementation_sha256=sha256(Path(__file__).parents[1]/'motionbricks/data/dilated_graph_convolution.py'),
            script_sha256=sha256(Path(__file__)),production_approved=False,
            scope='Same real microbatch, raw pre-clip gradients and target-layer inputs/upstream; no optimizer.'))
        save(a.output/'state.json',dict(status='complete',actual_optimizer_updates=0))
    finally:
        controller.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:main(args)
    except Exception as error:
        (args.output/'failure.json').write_text(json.dumps(dict(error=repr(error),actual_optimizer_updates=0)))
        raise
