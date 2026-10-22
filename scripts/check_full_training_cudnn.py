"""Check native CUDA convolution execution choices without changing model math."""

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
from scripts.train_unified_full_corpus import device_batch
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    inference, bundle = load_terrain_bundle(a.bundle, 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    saved = torch.load(a.initial, map_location='cpu', mmap=True, weights_only=False)
    dataset = PrimitiveDataset([p / 'data' for p in a.cache], 'train')
    rng = np.random.default_rng(2333)
    rows = []
    for domain, count in (('pfnn', a.batch//2), ('parc', a.batch//4), ('bones', a.batch//4)):
        ids = [i for i, idx in enumerate(dataset.ids) if dataset.records[idx]['domain'] == domain]
        rows.extend(dataset[int(i)] for i in rng.choice(ids, count, replace=False))
    batch = device_batch(collate_primitives(rows), 'cuda:0')
    reports = []
    for name in ('root', 'codec', 'pose'):
        torch.use_deterministic_algorithms(True, warn_only=name == 'codec')
        reference, reference_loss, reference_buffers = None, None, None
        for variant in ('baseline', 'benchmark', 'native'):
            controller.load_state_dict(saved['model_state'], strict=True)
            controller.zero_grad(set_to_none=True)
            stage = NativeTrainingStage(controller, name).train(True)
            torch.manual_seed(2333)
            torch.backends.cudnn.enabled = variant != 'native'
            torch.backends.cudnn.benchmark = variant == 'benchmark'
            times, loss_value, gradients, buffers = [], None, None, None
            for repeat in range(2):
                controller.load_state_dict(saved['model_state'], strict=True)
                controller.zero_grad(set_to_none=True)
                torch.manual_seed(2333)
                torch.cuda.synchronize()
                started = time.perf_counter()
                loss = stage(batch, 2333)
                loss.backward()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
                print(json.dumps(dict(stage=name, variant=variant, repeat=repeat, seconds=times[-1])), flush=True)
            gradients = {k: p.grad.detach().clone() if p.grad is not None else None
                for k, p in controller.named_parameters() if p.requires_grad}
            buffers = {k: v.detach().clone() for k, v in controller.state_dict().items()
                if k.startswith('codec.quantizer.')}
            loss_value = loss.detach().clone()
            errors, maximum, squared_error, squared_reference = [], 0., 0., 0.
            if variant == 'baseline':
                reference, reference_loss, reference_buffers = gradients, loss_value, buffers
            else:
                for k, expected in reference.items():
                    actual = gradients[k]
                    if expected is None or actual is None:
                        if (expected is None) != (actual is None):
                            errors.append('Gradient presence: ' + k)
                        continue
                    difference = actual - expected
                    maximum = max(maximum, float(difference.abs().max()))
                    squared_error += float(difference.double().square().sum())
                    squared_reference += float(expected.double().square().sum())
                    if not torch.allclose(actual, expected, atol=5e-6, rtol=1e-3):
                        errors.append('Gradient tolerance: ' + k)
                if not torch.allclose(loss_value, reference_loss, atol=1e-6, rtol=1e-5):
                    errors.append('Loss tolerance')
            reports.append(dict(stage=name, variant=variant, seconds=times, loss=float(loss_value),
                loss_difference=abs(float(loss_value-reference_loss)), gradient_max_abs=maximum,
                gradient_relative_l2=(squared_error/max(squared_reference, 1e-30))**.5,
                quantizer_buffers_bitwise_equal=all(torch.equal(v, reference_buffers[k]) for k,v in buffers.items()),
                passed=not errors, errors=errors, batch=a.batch))
            write_json(a.output / 'progress.json', dict(rows=reports))
            if variant != 'baseline':
                del gradients, buffers
        del reference, reference_buffers, stage, loss
        controller.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    controller.close()
    write_json(a.output / 'complete.json', dict(completed=True, rows=reports,
        bundle_sha256=sha256(a.bundle), initial_sha256=sha256(a.initial),
        actual_optimizer_updates=0, quality_approved=False,
        note='Execution alternatives on one real balanced batch. Not production approval or resume proof.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'initial', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cache', type=Path, nargs='+', required=True)
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
