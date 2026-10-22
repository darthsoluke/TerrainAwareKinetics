"""Prove frozen-token reuse across shuffled real batches preserves Pose training."""

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from motionbricks.data.full_terrain_cache import PrimitiveDataset, collate_primitives
from motionbricks.data.unified_full_training import NativeTrainingStage
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_unified_full_corpus import device_batch, rng_state, restore_rng
from scripts.train_native_mb_control import state_digest
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    inference, bundle = load_terrain_bundle(a.bundle, 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if not saved['contract']['mechanics_only']:
        raise ValueError('Engineering probe requires explicit mechanics checkpoint')
    controller.load_state_dict(saved['model_state'], strict=True)
    original_hash = state_digest(controller)
    dataset = PrimitiveDataset([p/'data' for p in a.cache], 'train')
    rng = np.random.default_rng(2333)
    pools = {}
    for domain, count in (('pfnn', 128), ('parc', 64), ('bones', 64)):
        ids = [i for i, idx in enumerate(dataset.ids) if dataset.records[idx]['domain'] == domain]
        pools[domain] = rng.choice(ids, count, replace=False)
    stage = NativeTrainingStage(controller, 'pose', cache_pose_tokens=True).train(True)
    # Populate by one grouping, then compare against fresh encoding after grouping
    # and row order change. No gradients or optimizer updates warm the cache.
    with torch.no_grad():
        for offset in range(4):
            ids = []
            for domain, count in (('pfnn', 32), ('parc', 16), ('bones', 16)):
                ids.extend(pools[domain][offset*count:(offset+1)*count])
            stage(device_batch(collate_primitives([dataset[int(i)] for i in ids]), 'cuda:0'), 2333)
    reports = []
    token_method = stage.pose_tokens
    captured_tokens = []

    def capture_tokens(batch, local):
        value = token_method(batch, local)
        captured_tokens.append(value.detach().clone())
        return value

    stage.pose_tokens = capture_tokens
    for repeat in range(4):
        ids = []
        for domain, count in (('pfnn', 32), ('parc', 16), ('bones', 16)):
            ids.extend(rng.choice(pools[domain], count, replace=False))
        rng.shuffle(ids)
        batch = device_batch(collate_primitives([dataset[int(i)] for i in ids]), 'cuda:0')
        values = []
        captured_tokens.clear()
        torch.manual_seed(2345 + repeat)
        initial_rng = rng_state()
        for cached in (False, True):
            stage.cache_pose_tokens = cached
            controller.zero_grad(set_to_none=True)
            restore_rng(initial_rng)
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss = stage(batch, 2345 + repeat)
            loss.backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            gradient = {k: p.grad.detach().clone() if p.grad is not None else None
                for k, p in controller.named_parameters() if p.requires_grad}
            after_rng = rng_state()
            values.append((loss.detach().clone(), gradient, elapsed, after_rng))
        ref, actual = values
        write_json(a.output/('comparison-'+str(repeat)+'.json'), dict(
            token_changes=int((captured_tokens[0]!=captured_tokens[1]).sum()),
            reference_stride=list(captured_tokens[0].stride()), cached_stride=list(captured_tokens[1].stride()),
            reference_loss=float(ref[0]), cached_loss=float(actual[0]),
            original_seconds=ref[2], cached_seconds=actual[2],
            cpu_rng_equal=torch.equal(actual[3]['torch'], ref[3]['torch']),
            cuda_rng_equal=all(torch.equal(x,y) for x,y in zip(actual[3]['cuda'],ref[3]['cuda']))))
        torch.testing.assert_close(captured_tokens[0], captured_tokens[1], atol=0, rtol=0)
        torch.testing.assert_close(actual[0], ref[0], atol=0, rtol=0)
        maximum = 0.
        for key, expected in ref[1].items():
            changed = actual[1][key]
            if expected is None or changed is None:
                if (expected is None) != (changed is None):
                    raise ValueError('Gradient presence differs: ' + key)
                continue
            torch.testing.assert_close(changed, expected, atol=0, rtol=0)
            maximum = max(maximum, float((changed-expected).abs().max()))
        torch.testing.assert_close(actual[3]['torch'], ref[3]['torch'], atol=0, rtol=0)
        for x, y in zip(actual[3]['cuda'], ref[3]['cuda']):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        reports.append(dict(repeat=repeat, samples=64, original_seconds=ref[2], cached_seconds=actual[2],
            speedup=ref[2]/actual[2], loss=float(ref[0]), loss_difference=0., gradient_max_abs=maximum,
            rng_equal=True))
        write_json(a.output/'progress.json', dict(rows=reports))
    if state_digest(controller) != original_hash:
        raise ValueError('Frozen-label test changed model tensors')
    controller.zero_grad(set_to_none=True)
    write_json(a.output/'complete.json', dict(passed=True, rows=reports, unique_cached_samples=len(stage._pose_tokens),
        cache_hits=stage.token_cache_hits, cache_misses=stage.token_cache_misses,
        checkpoint_sha256=sha256(a.checkpoint), script_sha256=sha256(Path(__file__)),
        all_model_tensors_unchanged=True, actual_optimizer_updates=0, quality_approved=False,
        scope='All losses/parameter gradients/RNG match exactly on regrouped real immutable samples; warm-cache timing.'))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'checkpoint', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cache', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output/'failure.json', dict(error=repr(error)))
        raise
