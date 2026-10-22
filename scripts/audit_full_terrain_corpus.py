"""Bind complete cache tensors and report physical command/recording coverage."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from motionbricks.data.full_terrain_cache import FIELDS
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_unified_full_corpus import dataset_contract


def main(a):
    receipts = dataset_contract(a.cache, sha256(a.bundle))
    rows, files = [], []
    for folder in a.cache:
        records = [json.loads(line) for line in (folder/'data/records.jsonl').read_text().splitlines()]
        world = np.load(folder/'data/world.npy', mmap_mode='r')
        if len(world) != len(records) or world.shape[1] != 33:
            raise ValueError('Complete real33-frame cache shape required')
        for key in FIELDS:
            path = folder/'data'/(key+'.npy')
            value = np.load(path, mmap_mode='r')
            if len(value) != len(records):
                raise ValueError('Unequal cache field rows')
            files.append(dict(path=str(path.resolve()), sha256=sha256(path), bytes=path.stat().st_size))
        for domain in ('pfnn','parc','bones'):
            for split in ('train','validation','test'):
                ids = np.asarray([i for i,row in enumerate(records) if row['domain'] == domain and row['split'] == split])
                if not len(ids):
                    continue
                terminal, prefix, rise, side = [], [], [], []
                for start in range(0,len(ids),4096):
                    part = np.array(world[ids[start:start+4096]], copy=True)
                    if not np.isfinite(part).all():
                        raise ValueError('Nonfinite source motion cache')
                    velocity = (part[:,32][:,[0,2]]-part[:,31][:,[0,2]])*30
                    terminal.extend(np.linalg.norm(velocity,axis=-1))
                    prefix.extend(np.linalg.norm((part[:,3][:,[0,2]]-part[:,2][:,[0,2]])*30,axis=-1))
                    rise.extend(part[:,:,1].max(1)-part[:,:,1].min(1))
                    heading = part[:,32,3:5]
                    forward = np.stack((heading[:,1],heading[:,0]),-1)
                    side.extend((velocity*forward).sum(-1)/np.maximum(np.linalg.norm(velocity,axis=-1),1e-8))
                terminal, prefix, rise, side = map(np.asarray,(terminal,prefix,rise,side))
                rows.append(dict(cache=str(folder),domain=domain,split=split,augmented_primitives=len(ids),
                    recording_families=len({records[i]['source_group'] for i in ids}),
                    source_ids=len({records[i]['source_id'] for i in ids}),
                    terminal_speed_mps_quantiles=np.quantile(terminal,[0,.05,.5,.95,1]).tolist(),
                    speed_bin_edges_mps=[0,.05,.25,.5,1,2,3,4,6,100],
                    speed_bin_counts=np.histogram(terminal,[0,.05,.25,.5,1,2,3,4,6,100])[0].tolist(),
                    source_idle_goal_windows=int((terminal<.05).sum()),
                    source_start_windows=int(((prefix<.1)&(terminal>.3)).sum()),
                    source_stop_windows=int(((prefix>.3)&(terminal<.1)).sum()),
                    root_height_excursion_over10cm=int((rise>.1).sum()),
                    sideways_or_backward_terminal_windows=int(((side<.5)&(terminal>.3)).sum())))
    counts = Counter()
    for row in rows:
        counts[(row['domain'],row['split'])] += row['augmented_primitives']
    if any(counts[(domain,split)] == 0 for domain in ('pfnn','parc','bones') for split in ('train','validation','test')):
        raise ValueError('All three domains require admitted train/validation/test data')
    write_json(a.output, dict(passed=True,receipts=receipts,fields=files,coverage=rows,
        labels_are_recorded_wasd=False, phase_is_actor_input=False,
        retained_normalization_bundle_sha256=sha256(a.bundle), full_training_started=False,
        note='Quarter views and overlapping windows are correlated, not independent recordings.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', nargs='+', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
