"""Export a paired terrain and exactly four observed frames for a live trial."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.portable_parc_online_soma import load_terrain_bundle, world_history
from scripts.evaluate_parc_native_terrain import _dataset_windows
from scripts.train_native_mb import sha256
from motionbricks.data.native_mb_data import NativeMBMotionDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'dataset', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--window', type=int, default=2)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new scene artifact')
    torch.set_num_threads(4)
    infer, bundle = load_terrain_bundle(args.bundle, 'cpu')
    _, windows, _, _, identity, _, _, _ = _dataset_windows(args.dataset, 8)
    if identity['dataset_manifest_sha256'] != bundle['metadata']['dataset_manifest_sha256']:
        raise ValueError('Scene and model dataset differ')
    window = windows[args.window]
    data = NativeMBMotionDataset(args.dataset, 'validation', max_frames=10**9, min_frames=65,
                                require_terrain=True, require_online_terrain=True)
    idx = next(i for i, row in enumerate(data.records) if row['shard'] == window['shard'])
    item = data[idx]
    history = world_history(infer.global_motion_rep, item, window['start'])
    metadata = {'schema': 'parc.live_scene.v1', 'window': window,
                'dataset_manifest_sha256': identity['dataset_manifest_sha256'],
                'source_bundle_sha256': sha256(args.bundle),
                'history_frames': 4, 'future_motion_included': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, initial_history=history.cpu().numpy(),
        **{key: value.cpu().numpy() for key, value in item['terrain_scene'].items()},
        metadata_json=np.asarray(json.dumps(metadata)))
    metadata['scene_sha256'] = sha256(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
