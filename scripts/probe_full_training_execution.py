"""Zero-update memory-policy and two-device microbatch execution diagnostics."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main(a):
    a.completed_optimizer_updates = 0
    snapshot = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader,nounits'], text=True)
    rows = {int(x.split(',')[0]): [v.strip() for v in x.split(',')[1:]] for x in snapshot.splitlines()}
    occupied = {x.split(',')[1].strip() for x in apps.splitlines()}
    for gpu in a.gpus:
        uuid, memory, busy = rows[gpu]
        if int(memory) > 500 or int(busy) > 5 or uuid in occupied:
            raise RuntimeError('Selected GPU is no longer idle: '+snapshot)
    save(a.output/'state.json', dict(status='running', pid=os.getpid(), gpus=a.gpus,
        gpu_snapshot=snapshot, actual_optimizer_updates=0))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, a.gpus))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches, collate_primitives
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch, split_micro
    from scripts.train_native_mb_control import state_digest
    from contextlib import nullcontext
    from unittest.mock import patch

    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=a.stage == 'codec')
    if a.mode == 'parallel' and len(a.gpus) != 2:
        raise ValueError('Parallel diagnostic requires exactly two idle GPUs')
    if a.mode == 'parallel' and a.stage != 'root':
        raise ValueError('Codec EMA updates are sequential; this parallel probe is Root-only')
    if a.graph_dilated and (a.mode!='memory' or a.stage!='codec' or len(a.gpus)!=1):
        raise ValueError('Convolution graph probe requires one-device codec execution')
    graph_execution = None
    if a.graph_dilated:
        from motionbricks.data.dilated_graph_convolution import GraphConvolutionExecution
        graph_execution = GraphConvolutionExecution()
    for index in range(len(a.gpus)):
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction, index)
    base = Path('${PRIVATE_STORAGE}/motionbricks-20261010')
    run = base/'pfnn_mixed_20261015'
    checkpoint_hash = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    controllers, stages, protected = [], [], []
    try:
        for index in range(len(a.gpus)):
            device = f'cuda:{index}'
            inference, bundle = load_terrain_bundle(base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt', device)
            controller = UnifiedTerrainController(inference, bundle)
            controller.load_state_dict(saved['model_state'], strict=True)
            controller.codebook.copy_(controller.codec.get_codebook())
            stages.append(NativeTrainingStage(controller, a.stage, coalesce_dilated_bias=True).train(True))
            controllers.append(controller)
            protected.append(state_digest(controller))
        dataset = PrimitiveDataset([run/'cache_mixed_v1/data', run/'cache_pfnn_world_v2/data'], 'train')
        if a.apply_optimizer and a.mode != 'memory':
            raise ValueError('Optimizer fit probe is single-device only')
        optimizer = (torch.optim.Adam([p for p in controllers[0].parameters() if p.requires_grad], lr=5e-5)
            if a.apply_optimizer else None)
        schedule = BalancedPrimitiveBatches(dataset, 256, saved['actual_updates']+a.batches,
            seed=saved['contract']['seed'], start_step=saved['actual_updates'])
        reports = []
        for batch_index, ids in enumerate(schedule):
            if optimizer is None:
                for controller in controllers:
                    controller.load_state_dict(saved['model_state'], strict=True)
            started = time.perf_counter()
            batch = collate_primitives([dataset[i] for i in ids])
            order = []
            for offset in range(4):
                for domain, count in (('pfnn',32), ('parc',16), ('bones',16)):
                    selected = [i for i, name in enumerate(batch['domains']) if name == domain]
                    order.extend(selected[offset*count:(offset+1)*count])
            def reorder(value):
                if torch.is_tensor(value): return value[order]
                if isinstance(value, dict): return {k:reorder(v) for k,v in value.items()}
                return [value[i] for i in order]
            batch = reorder(batch)
            data_seconds = time.perf_counter()-started
            sample_ids = batch['sample_ids'].tolist()
            batch0 = device_batch(batch, 'cuda:0')
            torch.cuda.synchronize(0)
            cuda_initial = saved['rng_state']['cuda'][0].cpu()
            cpu_initial = saved['rng_state']['torch'].cpu()
            torch.set_rng_state(cpu_initial)
            torch.cuda.set_rng_state(cuda_initial, 0)
            controllers[0].zero_grad(set_to_none=True)
            starts, losses, outputs, times = [], [], [], []
            torch.cuda.reset_peak_memory_stats(0)
            beginning = time.perf_counter()
            for micro in range(4):
                starts.append(torch.cuda.get_rng_state(0))
                tick = time.perf_counter()
                with (patch('motionbricks.data.dilated_bias_convolution.BatchedBiasNativeBackward',graph_execution)
                        if graph_execution else nullcontext()):
                    loss = stages[0](split_micro(batch0, micro*64, (micro+1)*64),
                        saved['contract']['seed']+(saved['actual_updates']+batch_index)*103+micro*64)
                    (loss*.25).backward()
                torch.cuda.synchronize(0)
                times.append(time.perf_counter()-tick)
                losses.append(float(loss.detach()))
                prediction = (controllers[0].last_root['pred_global_root_values'] if a.stage == 'root'
                    else controllers[0].last_local)
                outputs.append(prediction.detach().cpu().clone())
                del loss
            norm = torch.nn.utils.clip_grad_norm_([p for p in controllers[0].parameters() if p.requires_grad], .5)
            if optimizer is not None:
                optimizer.step()
                a.completed_optimizer_updates += 1
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter()-beginning
            expected = dict(losses=losses, outputs=outputs,
                gradients={name:p.grad.detach().cpu().clone() if p.grad is not None else None
                    for name,p in controllers[0].named_parameters() if p.requires_grad},
                quantizer_buffers={name:value.detach().cpu().clone() for name,value in controllers[0].named_buffers()
                    if name.startswith('codec.quantizer.')},
                rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(0), sample_ids=sample_ids)
            report = dict(batch_index=batch_index, data_seconds=data_seconds, forward_backward_clip_seconds=elapsed,
                microbatch_seconds=times, loss=sum(losses)/4, gradient_norm=float(norm),
                max_memory_allocated=torch.cuda.max_memory_allocated(0),
                max_memory_reserved=torch.cuda.max_memory_reserved(0))
            if optimizer is not None:
                report['optimizer_steps'] = sorted({int(x['step']) for x in optimizer.state_dict()['state'].values()})
            if a.mode == 'memory':
                torch.save(expected, a.output/f'batch_{batch_index}.pt')
            else:
                # This first probe copies RNG starts from the serial reference.
                # It cannot yet schedule a production parallel update independently.
                microbatches = [device_batch(split_micro(batch, i*64, (i+1)*64), f'cuda:{i%2}') for i in range(4)]
                torch.set_rng_state(cpu_initial)
                accumulated, observed_losses, observed_outputs = {}, [], []
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
                beginning = time.perf_counter()
                def execute(micro):
                    index = micro%2
                    with torch.cuda.device(index):
                        controllers[index].zero_grad(set_to_none=True)
                        torch.cuda.set_rng_state(starts[micro], index)
                        loss = stages[index](microbatches[micro],
                            saved['contract']['seed']+(saved['actual_updates']+batch_index)*103+micro*64)
                        (loss*.25).backward()
                        torch.cuda.synchronize(index)
                        return dict(loss=float(loss.detach()),
                            output=controllers[index].last_root['pred_global_root_values'].detach().cpu().clone(),
                            gradients={name:p.grad.detach() if p.grad is not None else None
                                for name,p in controllers[index].named_parameters() if p.requires_grad},
                            rng=torch.cuda.get_rng_state(index))
                with ThreadPoolExecutor(max_workers=2) as pool:
                    for first in (0, 2):
                        futures = [pool.submit(execute, first+i) for i in range(2)]
                        for future in futures:
                            result = future.result()
                            observed_losses.append(result['loss'])
                            observed_outputs.append(result['output'])
                            for name, gradient in result['gradients'].items():
                                if gradient is None: continue
                                if name not in accumulated:
                                    accumulated[name] = gradient.to('cuda:0').clone()
                                else:
                                    accumulated[name].add_(gradient.to('cuda:0'))
                            final_rng = result['rng']
                for name, parameter in controllers[0].named_parameters():
                    if parameter.requires_grad: parameter.grad = accumulated.get(name)
                torch.nn.utils.clip_grad_norm_([p for p in controllers[0].parameters() if p.requires_grad], .5)
                torch.cuda.synchronize(0)
                parallel_seconds = time.perf_counter()-beginning
                errors, maximum = [], 0.
                for name, parameter in controllers[0].named_parameters():
                    if not parameter.requires_grad: continue
                    reference = expected['gradients'][name]
                    actual = parameter.grad.detach().cpu() if parameter.grad is not None else None
                    if actual is None or reference is None:
                        if (actual is None) != (reference is None): errors.append('gradient presence:'+name)
                        continue
                    maximum = max(maximum, float((actual-reference).abs().max()))
                    if not torch.equal(actual, reference): errors.append('gradient:'+name)
                if observed_losses != expected['losses']: errors.append('losses')
                if not all(torch.equal(x,y) for x,y in zip(observed_outputs, expected['outputs'])): errors.append('outputs')
                if not torch.equal(torch.get_rng_state(), expected['rng_cpu']): errors.append('CPU RNG')
                if not torch.equal(final_rng, expected['rng_cuda']): errors.append('logical CUDA RNG')
                report.update(parallel_seconds=parallel_seconds, bitwise_equal=not errors, errors=errors,
                    gradient_max_abs=maximum, serial_rng_starts=[x.tolist()[:24] for x in starts],
                    rng_schedule_from_reference=True)
                del accumulated, microbatches
            reports.append(report)
            save(a.output/'progress.json', dict(rows=reports))
            print(json.dumps(report), flush=True)
            del batch0, expected, batch
            for controller in controllers: controller.zero_grad(set_to_none=True)
        for controller in controllers:
            controller.load_state_dict(saved['model_state'], strict=True)
        if any(state_digest(controller) != initial for controller, initial in zip(controllers, protected)):
            raise ValueError('Diagnostic changed model tensors')
        save(a.output/'complete.json', dict(mode=a.mode, stage=a.stage, rows=reports, memory_fraction=a.memory_fraction,
            gpus=a.gpus, checkpoint_sha256=checkpoint_hash, source_updates=saved['actual_updates'],
            actual_optimizer_updates=a.batches if optimizer is not None else 0,
            graph_dilated=a.graph_dilated,graph_signatures=len(graph_execution.calls) if graph_execution else 0,
            graph_implementation_sha256=hashlib.sha256((Path(__file__).parents[1]/'motionbricks/data/dilated_graph_convolution.py').read_bytes()).hexdigest() if graph_execution else None,
            mechanics_only=optimizer is not None, production_budget_updates=0,
            initial_weights_restored=True, quality_approved=False,
            production_approved=False, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        save(a.output/'state.json', dict(status='complete', actual_optimizer_updates=a.completed_optimizer_updates,
            production_budget_updates=0))
    finally:
        for controller in controllers: controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('memory', 'parallel'), required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-fraction', type=float, default=1.)
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--stage', choices=('root', 'codec'), default='root')
    parser.add_argument('--apply-optimizer', action='store_true')
    parser.add_argument('--graph-dilated', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error),
            completed_optimizer_updates=getattr(args, 'completed_optimizer_updates', 0),
            production_budget_updates=0, partial_update_possible=args.apply_optimizer))
        save(args.output/'state.json', dict(status='failed', error=repr(error)))
        raise
