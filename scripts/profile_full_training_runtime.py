"""Measure unchanged native stage costs on real cached mixed-domain inputs."""

import argparse
import json
import os
from pathlib import Path
import time
from contextlib import nullcontext

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from motionbricks.data.full_terrain_cache import PrimitiveDataset, collate_primitives
import motionbricks.data.unified_full_training as training
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_unified_full_corpus import device_batch
from scripts.train_joint_terrain_v3 import write_json


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    inference, bundle = load_terrain_bundle(a.bundle, 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    saved = torch.load(a.initial, map_location='cpu', weights_only=False)
    mixed = PrimitiveDataset(a.mixed / 'data', 'train')
    fixture = PrimitiveDataset(a.fixture / 'data', 'train')
    pf = [i for i, idx in enumerate(fixture.ids) if fixture.records[idx]['domain'] == 'pfnn']
    rng = np.random.default_rng(2333)
    rows = [fixture[pf[i % len(pf)]] for i in range(32)]
    for domain in ('parc', 'bones'):
        ids = [i for i, idx in enumerate(mixed.ids) if mixed.records[idx]['domain'] == domain]
        rows.extend(mixed[int(i)] for i in rng.choice(ids, 16, replace=False))
    batch = device_batch(collate_primitives(rows), 'cuda:0')
    original_prepare = training.prepare_native_batch
    prepare_times = []

    def traced_prepare(*args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_prepare(*args, **kwargs)
        torch.cuda.synchronize()
        prepare_times.append(time.perf_counter() - started)
        return result

    training.prepare_native_batch = traced_prepare
    reports = []
    try:
        for name in ('root', 'codec', 'pose'):
            controller.load_state_dict(saved['model_state'], strict=True)
            stage = training.NativeTrainingStage(controller, name).train(True)
            torch.use_deterministic_algorithms(True, warn_only=name == 'codec')
            timings = []
            forwards = []
            backwards = []
            prepare_times.clear()
            for _ in range(2):
                controller.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                loss = stage(batch, 2333)
                torch.cuda.synchronize()
                middle = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
                forwards.append(middle - started)
                backwards.append(time.perf_counter() - middle)
                print(json.dumps(dict(stage=name, repeat=len(timings), total=timings[-1],
                    forward=forwards[-1], backward=backwards[-1], prepare=prepare_times[-1])), flush=True)
            controller.zero_grad(set_to_none=True)
            events = []
            if a.trace:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) as profile:
                    loss = stage(batch, 2333)
                    loss.backward()
                    torch.cuda.synchronize()
                events = profile.key_averages()
            summaries = [dict(name=e.key, calls=e.count, cpu_self_us=e.self_cpu_time_total,
                cpu_total_us=e.cpu_time_total, device_self_us=e.self_device_time_total,
                device_total_us=e.device_time_total) for e in events]
            reports.append(dict(stage=name, batch=64, forward_backward_seconds=timings,
                forward_seconds=forwards, backward_seconds=backwards,
                preparation_seconds=list(prepare_times),
                top_device=sorted(summaries, key=lambda e: e['device_self_us'], reverse=True)[:20],
                top_cpu=sorted(summaries, key=lambda e: e['cpu_self_us'], reverse=True)[:20],
                regions=[e for e in summaries if e['name'].startswith(('stage_', 'native_input_'))]))
            write_json(a.output / 'progress.json', dict(stages=reports))
        write_json(a.output / 'complete.json', dict(passed=True, stages=reports,
            actual_optimizer_updates=0, micro_batch=64, quality_approved=False,
            note='Corrected PFNN fixture repeated; real PARC/BONES windows. Profiling only.'))
    finally:
        training.prepare_native_batch = original_prepare
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'initial', 'mixed', 'fixture', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
