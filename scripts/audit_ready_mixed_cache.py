"""Audit completed PARC/BONES coverage while PFNN preparation continues."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.train_unified_full_corpus import dataset_contract
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = dataset_contract([a.cache], sha256(a.bundle))
    records = [json.loads(line) for line in (a.cache/'data/records.jsonl').read_text().splitlines()]
    rejected = json.loads((a.cache/'rejected.json').read_text())
    world = np.load(a.cache/'data/world.npy', mmap_mode='r')
    rows = []
    for domain in ('parc', 'bones'):
        for split in ('train', 'validation', 'test'):
            ids = np.asarray([i for i, row in enumerate(records)
                if row['domain'] == domain and row['split'] == split and row['quarter'] == 0])
            if not len(ids):
                raise ValueError('No admitted original windows for '+domain+'/'+split)
            before, after, rise = [], [], []
            for start in range(0, len(ids), 4096):
                part = np.array(world[ids[start:start+4096]], copy=True)
                if not np.isfinite(part).all():
                    raise ValueError('Nonfinite motion in completed cache')
                before.extend(np.linalg.norm(np.diff(part[:, 2:4, :][:, :, [0,2]],axis=1)[:,0]*30,axis=-1))
                after.extend(np.linalg.norm(np.diff(part[:,31:33,:][:,:,[0,2]],axis=1)[:,0]*30,axis=-1))
                rise.extend(np.ptp(part[:,:,1],axis=1))
            before, after, rise = map(np.asarray, (before, after, rise))
            rows.append(dict(domain=domain,split=split,base_windows=len(ids),
                recording_families=len({records[i]['source_group'] for i in ids}),
                source_ids=len({records[i]['source_id'] for i in ids}),
                terminal_speed_mps_quantiles=np.quantile(after,[0,.05,.5,.95,1]).tolist(),
                idle_goals=int((after<.05).sum()), starts=int(((before<.1)&(after>.3)).sum()),
                stops=int(((before>.3)&(after<.1)).sum()),height_excursion_over10cm=int((rise>.1).sum())))
    write_json(a.output/'complete.json',dict(passed=True,cache_receipts=receipt,
        augmented_primitives=len(records),base_windows=sum(row['base_windows'] for row in rows),
        rejected_base_windows=len(rejected),
        rejection_domains=dict(Counter(row['domain'] for row in rejected)),rows=rows,
        family_split_passed=True,full_corpus_audit=False,full_training_started=False,
        note='Source velocity is a hindsight command, not recorded WASD. Quarter views are correlated.'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','bundle','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output/'failure.json',dict(error=repr(error)))
        raise
