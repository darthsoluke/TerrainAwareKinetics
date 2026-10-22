"""Select a small split-preserving, manifest-bound decoder experiment corpus."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import NativeMBMotionDataset, build_native_representation
from scripts.train_native_mb import sha256, verify_dataset


def choose(records, split):
    candidates = [r for r in records if r['split'] == split and r['frames_30hz'] >= 65]
    selected = []
    quotas = ({'bones_seed': 8, 'parc_official': 8, 'parc_generated': 8} if split == 'train'
              else {'bones_seed': 4, 'parc_official': 5, 'parc_generated': 3})
    for origin, quota in quotas.items():
        groups = defaultdict(list)
        for row in candidates:
            if row['origin'] == origin:
                category = row.get('control_category', row.get('terrain_family', 'unspecified'))
                groups[category].append(row)
        for values in groups.values():
            values.sort(key=lambda row: hashlib.sha256(row['source_id'].encode()).hexdigest())
        names = sorted(groups)
        for _ in range(quota):
            if not names:
                break
            name = names.pop(0)
            selected.append(groups[name].pop(0))
            if groups[name]:
                names.append(name)
    return selected


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'source', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh corpus directory')
    torch.set_num_threads(2)
    manifest = json.loads((args.dataset/'native_manifest.json').read_text())
    verify_dataset(args.dataset, manifest)
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    dataset_hash = sha256(args.dataset/'native_manifest.json')
    if source['metadata']['dataset_manifest_sha256'] != dataset_hash:
        raise ValueError('Source model and corpus identity differ')
    rep = build_native_representation(args.dataset/'stats')
    records = json.loads((args.dataset/'accepted.json').read_text())
    scenes, windows, selection = {}, [], []
    for split in ('train', 'validation'):
        chosen = choose(records, split)
        loader = NativeMBMotionDataset(args.dataset, split, min_frames=65, max_frames=10**9,
            require_terrain=True, require_online_terrain=True, online_only=True)
        lookup = {r['shard']: i for i, r in enumerate(loader.records)}
        for row in chosen:
            item = loader[lookup[row['shard']]]
            raw = rep.unnormalize(item['motion'])[None]
            root = item['terrain_root_world'][None]
            world = rep.change_first_heading(raw, torch.atan2(root[:, 0, 4], root[:, 0, 3]), False, False)
            world[..., [0, 2]] += root[:, :1, [0, 2]]
            torch.testing.assert_close(world[..., :5], root, atol=5e-5, rtol=1e-5)
            key = row['terrain_geometry_sha256']
            scenes[key] = {k: torch.as_tensor(v).cpu().clone() for k, v in item['terrain_scene'].items()}
            starts = sorted(set(np.linspace(0, world.shape[1]-65, 4 if split == 'train' else 3).astype(int).tolist()))
            selection.append(dict(row, pilot_starts=starts))
            for start in starts:
                windows.append({'world': world[:, start:start+65].cpu().clone(),
                    'scene': key, 'split': split, 'source_id': row['source_id'],
                    'origin': row['origin'], 'category': row.get('control_category', row['terrain_family']),
                    'source_group': row['source_group'], 'start': start})
    owners = {}
    for row in selection:
        key = row['source_group']
        if key in owners and owners[key] != row['split']:
            raise ValueError('Pilot source group leaks across splits')
        owners[key] = row['split']
    args.output.mkdir(parents=True)
    torch.save({'scenes': scenes, 'windows': windows}, args.output/'windows.pt')
    receipt = {'schema': 'motionbricks.terrain_decoder_pilot_data.v1',
        'source_bundle_sha256': sha256(args.source), 'source_dataset_manifest_sha256': dataset_hash,
        'windows_sha256': sha256(args.output/'windows.pt'), 'selection': selection,
        'clips': {split: sum(r['split'] == split for r in selection) for split in ('train', 'validation')},
        'windows': {split: sum(r['split'] == split for r in windows) for split in ('train', 'validation')},
        'future_motion_role': 'training targets or explicitly marked offline oracle only',
        'quality_approved': False}
    (args.output/'manifest.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k != 'selection'}), flush=True)


if __name__ == '__main__':
    main()
