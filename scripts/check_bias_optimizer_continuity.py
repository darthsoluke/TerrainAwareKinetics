"""Real production checkpoint: native two steps versus saved step plus fast step."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time


def main(a):
    owner = Path('/proc')/str(a.paused_owner_pid)
    assert 'full_training_v3/root' in (owner/'cmdline').read_bytes().decode()
    assert next(x.split()[1] for x in (owner/'status').read_text().splitlines() if x.startswith('State:')) == 'T'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches, collate_primitives
    from motionbricks.data.dilated_bias_convolution import coalesced_dilated_bias
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch, split_micro, rng_state, restore_rng

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    base = Path('${PRIVATE_STORAGE}/motionbricks-20261010')
    run = base/'pfnn_mixed_20261015'
    saved = torch.load(run/'full_training_v3/root/latest.pt', map_location='cpu', mmap=True, weights_only=False)
    contract = saved['contract']
    inference, bundle = load_terrain_bundle(base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt', 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True)
    stage = NativeTrainingStage(controller, 'root').train(True)
    parameters = [p for p in controller.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=contract['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=contract['steps'], eta_min=contract['eta_min'])
    optimizer.load_state_dict(saved['optimizer_state'])
    scheduler.load_state_dict(saved['scheduler_state'])
    dataset = PrimitiveDataset([run/'cache_mixed_v1/data', run/'cache_pfnn_world_v2/data'], 'train')
    schedule = BalancedPrimitiveBatches(dataset, 256, saved['actual_updates']+2,
        seed=contract['seed'], start_step=saved['actual_updates'])
    batches = []
    for indices in schedule:
        batch = collate_primitives([dataset[i] for i in indices])
        order = []
        for offset in range(4):
            for domain, quota in (('pfnn',32), ('parc',16), ('bones',16)):
                selected = [i for i, name in enumerate(batch['domains']) if name == domain]
                order.extend(selected[offset*quota:(offset+1)*quota])
        def reorder(value):
            if torch.is_tensor(value):
                return value[order]
            if isinstance(value, dict):
                return {key: reorder(item) for key, item in value.items()}
            return [value[index] for index in order]
        batches.append(device_batch(reorder(batch), 'cuda:0'))
    restore_rng(saved['rng_state'])

    def cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(cpu(v) for v in value)
        return value

    def snapshot():
        return dict(model=cpu(controller.state_dict()), optimizer=cpu(optimizer.state_dict()),
            scheduler=cpu(scheduler.state_dict()), rng=cpu(rng_state()))

    def update(index, accelerated):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        values = []
        with coalesced_dilated_bias(controller) if accelerated else nullcontext():
            for offset in range(0, 256, 64):
                loss = stage(split_micro(batches[index], offset, offset+64),
                    contract['seed']+(saved['actual_updates']+index)*103+offset)
                (loss*.25).backward()
                values.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(parameters, .5)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        return dict(losses=values, seconds=time.perf_counter()-started)

    try:
        update(0, False)
        midpoint = snapshot()
        torch.save(midpoint, a.output/'resume_fixture.pt')
        native = update(1, False)
        reference = snapshot()
        del midpoint
        midpoint = torch.load(a.output/'resume_fixture.pt', map_location='cpu', weights_only=False)
        controller.load_state_dict(midpoint['model'], strict=True)
        optimizer.load_state_dict(midpoint['optimizer'])
        scheduler.load_state_dict(midpoint['scheduler'])
        restore_rng(midpoint['rng'])
        accelerated = update(1, True)
        actual = snapshot()
        errors = []
        def compare(left, right, path=''):
            import numpy as np
            if torch.is_tensor(left):
                if not torch.equal(left, right): errors.append(path)
            elif isinstance(left, np.ndarray):
                if not np.array_equal(left, right): errors.append(path)
            elif isinstance(left, dict):
                if left.keys() != right.keys(): errors.append(path+'.keys')
                for key in left: compare(left[key], right[key], path+'.'+str(key))
            elif isinstance(left, (tuple, list)):
                if len(left) != len(right): errors.append(path+'.length')
                for i, (x,y) in enumerate(zip(left,right)): compare(x,y,path+'.'+str(i))
            elif left != right:
                errors.append(path)
        compare(reference, actual)
        if native['losses'] != accelerated['losses']:
            errors.append('native_microbatch_losses')
        report = dict(passed=not errors, errors=errors, source_updates=saved['actual_updates'],
            baseline=native, accelerated=accelerated, compared=dict(model=True, optimizer=True,
                scheduler=True, rng=True, samples_bitwise_equal=True), batch=256, micro_batch=64,
            actual_optimizer_updates=3, mechanics_only=True, production_budget_updates=0,
            quality_approved=False)
        (a.output/'complete.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--paused-owner-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except Exception as error:
        (args.output/'failure.json').write_text(json.dumps(dict(error=repr(error)), indent=2))
        raise
