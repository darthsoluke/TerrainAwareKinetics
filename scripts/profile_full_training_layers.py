"""Bounded layer-event timing, launched only after an approved GPU is idle."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def select_idle(a):
    while True:
        snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        measured = {int(row.split(',')[0]): [value.strip() for value in row.split(',')[1:]]
            for row in snapshot.splitlines()}
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
            '--format=csv,noheader'],text=True)
        occupied = {line.split(',')[0].strip() for line in processes.splitlines()}
        for gpu in a.gpu_candidates:
            uuid, memory, busy = measured[gpu]
            if int(memory) <= 500 and int(busy) <= 5 and uuid not in occupied:
                save(a.output/'state.json', dict(status='starting', gpu=gpu,
                    gpu_snapshot_before=snapshot, compute_processes=processes, actual_optimizer_updates=0))
                return gpu
        save(a.output/'state.json', dict(status='waiting_for_idle_gpu', candidates=a.gpu_candidates,
            snapshot=snapshot, pid=os.getpid(), actual_optimizer_updates=0))
        time.sleep(30)


def profile(a, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, collate_primitives
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch
    from scripts.train_native_mb_control import state_digest
    from scripts.train_native_mb import sha256

    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not 0 < a.cuda_memory_fraction <= 1:
        raise ValueError('CUDA memory fraction must be within(0,1]')
    torch.cuda.set_per_process_memory_fraction(a.cuda_memory_fraction,0)
    inference, bundle = load_terrain_bundle(a.bundle, 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    saved = torch.load(a.initial, map_location='cpu', mmap=True, weights_only=False)
    controller.load_state_dict(saved['model_state'], strict=True)
    controller.codebook.copy_(controller.codec.get_codebook())
    original_hash = state_digest(controller)
    dataset = PrimitiveDataset([path/'data' for path in a.cache], 'train')
    rng, rows = np.random.default_rng(2333), []
    for domain, count in (('pfnn', 32), ('parc', 16), ('bones', 16)):
        ids = [i for i, actual in enumerate(dataset.ids) if dataset.records[actual]['domain'] == domain]
        rows.extend(dataset[int(i)] for i in rng.choice(ids, count, replace=False))
    batch = device_batch(collate_primitives(rows), 'cuda:0')
    summaries = []
    try:
        for stage_name in a.stages:
            torch.use_deterministic_algorithms(True, warn_only=stage_name == 'codec')
            controller.load_state_dict(saved['model_state'], strict=True)
            controller.zero_grad(set_to_none=True)
            stage = NativeTrainingStage(controller, stage_name,
                coalesce_dilated_bias=a.coalesce_dilated_bias).train(True)
            stage(batch, 2333).backward()
            torch.cuda.synchronize()
            controller.zero_grad(set_to_none=True)
            controller.load_state_dict(saved['model_state'], strict=True)
            pending, events, handles = {}, [], []
            selected = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear,
                        torch.nn.MultiheadAttention, torch.nn.TransformerEncoderLayer)

            def install(name, module):
                def before(current, args):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    shape = list(args[0].shape) if args and torch.is_tensor(args[0]) else None
                    pending[id(current)] = (event, time.perf_counter(), shape)

                def after(current, args, output):
                    started, wall, shape = pending.pop(id(current))
                    ended = torch.cuda.Event(enable_timing=True)
                    ended.record()
                    meta = dict(name=name, kind=type(current).__name__, input_shape=shape,
                        cpu_wall_ms=(time.perf_counter()-wall)*1000)
                    if isinstance(current, (torch.nn.Conv1d, torch.nn.Conv2d)):
                        meta.update(weight_shape=list(current.weight.shape),
                            dilation=list(current.dilation), padding=list(current.padding))
                    events.append((meta, started, ended))

                handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))

            for name, module in controller.named_modules():
                if isinstance(module, selected):
                    install(name, module)
            try:
                torch.manual_seed(2333)
                torch.cuda.synchronize()
                start = time.perf_counter()
                loss = stage(batch, 2333)
                torch.cuda.synchronize()
                middle = time.perf_counter()
                # Events intentionally do not retain inputs or outputs.
                loss.backward()
                torch.cuda.synchronize()
                finish = time.perf_counter()
            finally:
                for handle in handles:
                    handle.remove()
            values = [dict(meta, device_ms=first.elapsed_time(last)) for meta, first, last in events]
            summaries.append(dict(stage=stage_name, loss=float(loss.detach()),
                forward_ms=(middle-start)*1000, backward_ms=(finish-middle)*1000,
                layers=sorted(values, key=lambda row: row['device_ms'], reverse=True),
                note='Parent attention/layer events overlap; do not sum unlike scopes.'))
            save(a.output/'progress.json', dict(stages=summaries))
        controller.load_state_dict(saved['model_state'], strict=True)
        controller.zero_grad(set_to_none=True)
        if state_digest(controller) != original_hash:
            raise ValueError('Layer profiling did not restore original model tensors')
        save(a.output/'complete.json', dict(passed=True, stages=summaries, gpu=gpu,
            actual_optimizer_updates=0, original_weights_restored=True,
            coalesced_dilated_bias=a.coalesce_dilated_bias,cuda_memory_fraction=a.cuda_memory_fraction,
            initial_sha256=sha256(a.initial), script_sha256=sha256(Path(__file__)),
            quality_approved=False))
        save(a.output/'state.json', dict(status='complete', gpu=gpu, actual_optimizer_updates=0))
    finally:
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'initial', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cache', nargs='+', type=Path, required=True)
    parser.add_argument('--gpu-candidates', nargs='+', type=int, default=[2, 6])
    parser.add_argument('--stages',nargs='+',choices=('root','codec'),default=['root','codec'])
    parser.add_argument('--coalesce-dilated-bias',action='store_true')
    parser.add_argument('--cuda-memory-fraction',type=float,default=1.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        profile(args, select_idle(args))
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=0))
        save(args.output/'state.json', dict(status='failed', error=repr(error)))
        raise
