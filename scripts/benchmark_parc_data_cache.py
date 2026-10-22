"""Check real paired-window equality and file reads across cold/hot cache passes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_data import NativeMBMotionDataset
from scripts.train_native_mb import sha256, verify_dataset, verify_training_source_policy, write_json


def benchmark(dataset_path, output):
    if output.exists():
        raise FileExistsError('Use a new cache preflight output')
    manifest = json.loads((dataset_path / 'native_manifest.json').read_text())
    verify_dataset(dataset_path, manifest)
    sources = verify_training_source_policy(dataset_path, manifest, parc_only=True)
    torch.set_num_threads(1)
    dataset = NativeMBMotionDataset(dataset_path, 'train', min_frames=65, max_frames=200,
                                   require_terrain=True, cache_max_bytes=256 * 1024 * 1024)
    original_load = np.load
    reads = 0

    def counted_load(*args, **kwargs):
        nonlocal reads
        if Path(args[0]).suffix == '.npz':
            reads += 1
        return original_load(*args, **kwargs)

    results = {}
    np.load = counted_load
    try:
        for phase in ('cold', 'hot'):
            torch.manual_seed(7)
            before = reads
            digest = hashlib.sha256()
            started = time.perf_counter()
            for index in range(len(dataset)):
                item = dataset[index]
                for key in ('motion', 'terrain_points', 'terrain_valid'):
                    value = item[key].numpy()
                    digest.update(key.encode())
                    digest.update(str((value.shape, value.dtype)).encode())
                    digest.update(value.tobytes())
            results[phase] = {'seconds': time.perf_counter() - started,
                              'npz_reads': reads - before, 'sampled_tensors_sha256': digest.hexdigest()}
    finally:
        np.load = original_load
    if results['cold']['sampled_tensors_sha256'] != results['hot']['sampled_tensors_sha256']:
        raise RuntimeError('Cache changed seeded paired training windows')
    all_cached = len(dataset._motion_cache) == len(dataset)
    if all_cached and results['hot']['npz_reads'] != 0:
        raise RuntimeError('Fully cached terrain windows still read files')
    result = {'status': 'paired_cache_verified', 'dataset': str(dataset_path),
              'dataset_manifest_sha256': sha256(dataset_path / 'native_manifest.json'),
              'train_source_counts': sources['train_origin_counts'], 'records': len(dataset),
              'cache_budget_bytes_per_worker': dataset.cache_max_bytes,
              'actual_cached_tensor_bytes_per_worker': dataset._cached_bytes,
              'all_train_records_cached': all_cached, 'seeded_windows_unchanged': True,
              'passes': results,
              'meaning': 'Real dataset loader microbenchmark, includes tensor hashing; not GPU training throughput'}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.dataset.resolve(), args.output.resolve()), indent=2))


if __name__ == '__main__':
    main()
