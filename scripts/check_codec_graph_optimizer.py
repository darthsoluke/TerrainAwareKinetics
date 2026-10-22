"""Actual codec Adam trajectories and cold graph-cache restoration diagnostics."""

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
    def check_idle(gpu):
        snapshot, available = gpu_snapshot()
        if gpu not in available: raise ValueError('Selected GPU is no longer idle')
        return snapshot
    allocation = check_idle(a.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    from motionbricks.data.dilated_graph_convolution import GraphConvolutionExecution
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches, collate_primitives
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch, split_micro, rng_state, restore_rng
    from scripts.train_native_mb import sha256
    check_idle(a.gpu)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.set_per_process_memory_fraction(a.graph_memory_fraction if a.cold_reference_directory else .1, 0)
    base = Path('${PRIVATE_STORAGE}/motionbricks-20261010')
    run = base/'pfnn_mixed_20261015'
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    contract = saved['contract']
    if contract['stage'] != 'codec' or contract['batch'] != 256 or contract['micro_batch'] != 64:
        raise ValueError('Actual codec256/64 checkpoint required')
    inference, bundle = load_terrain_bundle(base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt', 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True)
    controller.codebook.copy_(controller.codec.get_codebook())
    stage = NativeTrainingStage(controller, 'codec', coalesce_dilated_bias=True).train(True)
    parameters = [p for p in controller.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=contract['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000, eta_min=5e-7)
    optimizer.load_state_dict(saved['optimizer_state'])
    scheduler.load_state_dict(saved['scheduler_state'])
    dataset = PrimitiveDataset([run/'cache_mixed_v1/data', run/'cache_pfnn_world_v2/data'], 'train')
    schedule = BalancedPrimitiveBatches(dataset, 256, saved['actual_updates']+4,
        seed=contract['seed'], start_step=saved['actual_updates'])
    batches = []
    for ids in schedule:
        batch = collate_primitives([dataset[i] for i in ids])
        order = []
        for micro in range(4):
            for domain, count in (('pfnn', 32), ('parc', 16), ('bones', 16)):
                selected = [i for i, name in enumerate(batch['domains']) if name == domain]
                order.extend(selected[micro*count:(micro+1)*count])
        def reorder(value):
            if torch.is_tensor(value): return value[order]
            if isinstance(value, dict): return {k: reorder(v) for k, v in value.items()}
            return [value[i] for i in order]
        batches.append(reorder(batch))
    restore_rng(saved['rng_state'])

    def cpu(value):
        if torch.is_tensor(value): return value.detach().cpu().clone()
        if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return type(value)(cpu(v) for v in value)
        return value

    def snapshot():
        return dict(model=cpu(controller.state_dict()), optimizer=cpu(optimizer.state_dict()),
            scheduler=cpu(scheduler.state_dict()), rng=cpu(rng_state()))

    def differences(left, right, prefix=''):
        result = []
        if torch.is_tensor(left):
            if not torch.equal(left, right):
                result.append(dict(path=prefix, max_abs=float((left-right).abs().max())))
        elif isinstance(left, np.ndarray):
            if not np.array_equal(left, right): result.append(dict(path=prefix))
        elif isinstance(left, dict):
            if left.keys() != right.keys(): raise ValueError('Snapshot inventory differs')
            for key in left: result += differences(left[key], right[key], prefix+'.'+str(key))
        elif isinstance(left, (tuple, list)):
            if len(left) != len(right): raise ValueError('Snapshot length differs')
            for index, (x, y) in enumerate(zip(left, right)): result += differences(x, y, prefix+'.'+str(index))
        elif left != right:
            result.append(dict(path=prefix, reference=left, candidate=right))
        return result

    initial = snapshot()
    restoration_checks = []

    def new_execution():
        torch.cuda.set_per_process_memory_fraction(a.graph_memory_fraction, 0)
        return GraphConvolutionExecution(capture_backward=a.graph_mode == 'both')

    def restore(value):
        controller.zero_grad(set_to_none=True)
        controller.load_state_dict(value['model'], strict=True)
        optimizer.load_state_dict(value['optimizer'])
        scheduler.load_state_dict(value['scheduler'])
        restore_rng(value['rng'])
        errors = differences(value, snapshot())
        restoration_checks.append(not errors)
        if errors: raise ValueError('Model/Adam/scheduler/RNG restoration differs')

    def update(index, execution, phase):
        batch = device_batch(batches[index], 'cuda:0')
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        losses, outputs = [], []
        if a.stage_integration:
            stage.graph_dilated_convolutions = execution is not None
            stage._dilated_graph_execution = execution
        with (patch('motionbricks.data.dilated_bias_convolution.BatchedBiasNativeBackward', execution)
                if execution is not None and not a.stage_integration else nullcontext()):
            for offset in range(0, 256, 64):
                loss = stage(split_micro(batch, offset, offset+64),
                    contract['seed']+(saved['actual_updates']+index)*103+offset)
                (loss*.25).backward()
                losses.append(float(loss.detach()))
                outputs.append(controller.last_local.detach().cpu().clone())
        torch.nn.utils.clip_grad_norm_(parameters, .5)
        torch.cuda.synchronize()
        compute_seconds = time.perf_counter()-started
        observed = dict(losses=losses, outputs=outputs,
            gradients={name: cpu(p.grad) for name, p in controller.named_parameters() if p.requires_grad},
            quantizer={name: cpu(value) for name, value in controller.named_buffers()
                if name.startswith('codec.quantizer.')}, rng=cpu(rng_state()), sample_ids=batch['sample_ids'].cpu().tolist())
        optimizer.step()
        a.completed_updates += 1
        scheduler.step()
        torch.cuda.synchronize()
        save(a.output/'state.json', dict(status='running', phase=phase, source_step=saved['actual_updates']+index+1,
            actual_optimizer_updates=a.completed_updates, production_budget_updates=0))
        return observed, dict(losses=losses, forward_backward_clip_seconds=compute_seconds,
            reserved_bytes=torch.cuda.max_memory_reserved(), graph_signatures=len(execution.calls) if execution else 0)

    def computation_gate(left, right):
        errors, changes = [], differences(left, right)
        for change in changes:
            path = change['path']
            if path.startswith('.gradients.'):
                name = path[len('.gradients.'):]
                passed = torch.allclose(left['gradients'][name], right['gradients'][name], atol=5e-6, rtol=1e-3)
            elif path.startswith('.outputs.'):
                index = int(path.split('.')[-1])
                passed = torch.allclose(left['outputs'][index], right['outputs'][index], atol=1e-6, rtol=1e-5)
            elif path.startswith('.losses.'):
                index = int(path.split('.')[-1]); expected = left['losses'][index]
                passed = abs(expected-right['losses'][index]) <= 1e-6+1e-5*abs(expected)
            else:
                passed = False
            if not passed: errors.append(path)
        return dict(passed=not errors, errors=errors, bitwise_equal=not changes, differences=changes)

    try:
        if a.cold_reference_directory:
            previous = a.cold_reference_directory
            metadata = json.loads((previous/'complete.json').read_text())
            if not metadata['passed'] or metadata['checkpoint_sha256'] != sha256(a.checkpoint):
                raise ValueError('Cold-process reference identity or proof differs')
            restore(torch.load(previous/'native_midpoint.pt', weights_only=False))
            execution = new_execution()
            rows, gates = [], []
            for index in range(2, 4):
                observed, row = update(index, execution, 'fresh_process_native_to_graph')
                rows.append(row)
                gates.append(computation_gate(torch.load(previous/f'native_{index}.pt',weights_only=False), observed))
            changes = differences(torch.load(previous/'native_final.pt',weights_only=False), snapshot())
            save(a.output/'complete.json',dict(passed=not changes and all(row['bitwise_equal'] for row in gates),
                cold_process=True,endpoint_changes=changes,computation_gates=gates,rows=rows,
                actual_optimizer_updates=a.completed_updates,production_budget_updates=0,
                checkpoint_sha256=sha256(a.checkpoint),reference_proof_sha256=sha256(previous/'complete.json'),
                graph_mode=a.graph_mode,graph_memory_fraction=a.graph_memory_fraction,stage_integration=a.stage_integration,
                implementation_sha256=sha256(Path(__file__).parents[1]/'motionbricks/data/dilated_graph_convolution.py'),
                quality_approved=False,production_approved=False))
            save(a.output/'state.json',dict(status='complete',actual_optimizer_updates=a.completed_updates))
            return
        native_rows = []
        for index in range(4):
            observed, row = update(index, None, 'native')
            torch.save(observed, a.output/f'native_{index}.pt'); native_rows.append(row)
            if index == 1: torch.save(snapshot(), a.output/'native_midpoint.pt')
        torch.save(snapshot(), a.output/'native_final.pt')
        restore(initial)
        execution = new_execution()
        candidate_rows, gates = [], []
        for index in range(4):
            observed, row = update(index, execution, 'graph_continuous')
            torch.save(observed, a.output/f'graph_{index}.pt'); candidate_rows.append(row)
            gates.append(computation_gate(torch.load(a.output/f'native_{index}.pt', weights_only=False), observed))
            if index == 1: torch.save(snapshot(), a.output/'graph_midpoint.pt')
        graph_final = snapshot()
        torch.save(graph_final, a.output/'graph_final.pt')
        native_changes = differences(torch.load(a.output/'native_final.pt', weights_only=False), graph_final)
        del execution, observed
        controller.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        restore(torch.load(a.output/'graph_midpoint.pt', weights_only=False))
        execution = new_execution()
        resume_rows = []
        for index in range(2, 4):
            observed, row = update(index, execution, 'graph_cold_resume'); resume_rows.append(row)
            gates.append(computation_gate(torch.load(a.output/f'graph_{index}.pt', weights_only=False), observed))
        resume_changes = differences(graph_final, snapshot())
        del execution, observed
        controller.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        restore(torch.load(a.output/'native_midpoint.pt', weights_only=False))
        execution = new_execution()
        transition_rows = []
        for index in range(2, 4):
            observed, row = update(index, execution, 'native_to_graph_resume'); transition_rows.append(row)
            gates.append(computation_gate(torch.load(a.output/f'native_{index}.pt', weights_only=False), observed))
        transition_changes = differences(torch.load(a.output/'native_final.pt', weights_only=False), snapshot())
        save(a.output/'complete.json', dict(passed=all(row['passed'] for row in gates)
            and all(restoration_checks) and not resume_changes,
            computation_gates=gates, restoration_bitwise_checks=restoration_checks,
            native_graph_endpoint_changes=native_changes, cold_resume_endpoint_changes=resume_changes,
            native_to_graph_endpoint_changes=transition_changes, native=native_rows, graph=candidate_rows,
            resumed=resume_rows, transition=transition_rows, actual_optimizer_updates=a.completed_updates,
            production_budget_updates=0, checkpoint_sha256=sha256(a.checkpoint), source_updates=saved['actual_updates'],
            graph_mode=a.graph_mode,
            native_memory_fraction=.1,graph_memory_fraction=a.graph_memory_fraction,
            stage_integration=a.stage_integration,
            integration_source_sha256={name:sha256(Path(__file__).parents[1]/name) for name in
                ['motionbricks/data/unified_full_training.py','motionbricks/data/dilated_graph_context.py']} if a.stage_integration else {},
            implementation_sha256=sha256(Path(__file__).parents[1]/'motionbricks/data/dilated_graph_convolution.py'),
            tolerances_unchanged=True, production_approved=False, quality_approved=False))
        save(a.output/'state.json', dict(status='complete', actual_optimizer_updates=a.completed_updates,
            production_budget_updates=0))
    finally:
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--graph-mode',choices=('both','forward'),default='both')
    parser.add_argument('--graph-memory-fraction',type=float,default=.1)
    parser.add_argument('--stage-integration',action='store_true')
    parser.add_argument('--cold-reference-directory',type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.completed_updates = 0
    if not .1 <= args.graph_memory_fraction <= .2:
        parser.error('Diagnostic graph memory fraction must remain within[.1,.2]')
    args.output.mkdir(parents=True, exist_ok=False)
    try: main(args)
    except Exception as error:
        (args.output/'failure.json').write_text(json.dumps(dict(error=repr(error),
            completed_optimizer_updates=args.completed_updates, production_budget_updates=0,
            partial_update_possible=True), indent=2))
        raise
