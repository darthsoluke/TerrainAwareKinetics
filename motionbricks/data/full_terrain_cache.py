"""Disk-backed primitive data and deterministic source-balanced batch schedules."""

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


FIELDS = ('world', 'volume_features', 'volume_distance', 'volume_valid', 'observation_points',
          'observation_valid', 'observation_envelope')


class PrimitiveWriter:
    def __init__(self, folder, chunk_size=256):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.chunk_size = chunk_size
        self.buffer, self.chunks, self.records = [], [], []

    def append(self, tensors, record):
        if set(tensors) != set(FIELDS):
            raise ValueError('Primitive cache tensor schema differs')
        self.buffer.append({key: value.detach().cpu().contiguous() for key, value in tensors.items()})
        self.records.append(record)
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        values = {key: torch.stack([row[key] for row in self.buffer]) for key in FIELDS}
        path = self.folder/('chunk-'+str(len(self.chunks))+'.pt')
        torch.save(values, path)
        self.chunks.append(path)
        self.buffer.clear()

    def finish(self):
        self.flush()
        if not self.records:
            raise ValueError('No valid primitives admitted')
        first = torch.load(self.chunks[0], map_location='cpu', mmap=True, weights_only=True)
        maps = {key: np.lib.format.open_memmap(self.folder/(key+'.npy'), mode='w+',
            dtype=first[key].numpy().dtype, shape=(len(self.records), *first[key].shape[1:])) for key in FIELDS}
        cursor = 0
        for path in self.chunks:
            values = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
            count = len(values['world'])
            for key in FIELDS:
                maps[key][cursor:cursor+count] = values[key].numpy()
            cursor += count
        for value in maps.values():
            value.flush()
        with (self.folder/'records.jsonl').open('w', encoding='utf-8') as stream:
            for record in self.records:
                stream.write(json.dumps(record)+'\n')
        return dict(examples=len(self.records), fields={key: dict(shape=list(value.shape), dtype=str(value.dtype))
            for key, value in maps.items()})


class PrimitiveDataset(Dataset):
    def __init__(self, folder, split):
        self.folders = [Path(path) for path in folder] if isinstance(folder, (tuple, list)) else [Path(folder)]
        self.records, self.offsets = [], [0]
        for path in self.folders:
            self.records.extend(json.loads(line) for line in (path/'records.jsonl').read_text().splitlines())
            self.offsets.append(len(self.records))
        self.ids = [index for index, row in enumerate(self.records) if row['split'] == split]
        self.maps = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        actual = self.ids[index]
        part = bisect_right(self.offsets, actual)-1
        if part not in self.maps:
            self.maps[part] = {key: np.load(self.folders[part]/(key+'.npy'), mmap_mode='r') for key in FIELDS}
        row = {key: torch.from_numpy(np.array(value[actual-self.offsets[part]], copy=True))
               for key, value in self.maps[part].items()}
        row['sample_id'] = actual
        row['domain'] = self.records[actual]['domain']
        return row


def collate_primitives(rows):
    arrays = {key: torch.stack([row[key] for row in rows]) for key in FIELDS}
    return dict(world=arrays['world'],
        volume=dict(features=arrays['volume_features'], distance=arrays['volume_distance'], valid=arrays['volume_valid']),
        observation=dict(points=arrays['observation_points'], valid=arrays['observation_valid'],
                         query_envelope_m=arrays['observation_envelope']),
        sample_ids=torch.tensor([row['sample_id'] for row in rows]), domains=[row['domain'] for row in rows])


class BalancedPrimitiveBatches(Sampler):
    """Permutation cycles keyed by optimizer step, unaffected by worker prefetch."""
    def __init__(self, dataset, batch_size, steps, seed=2333, rank=0, world_size=1, start_step=0):
        if batch_size % 4 or batch_size < 4:
            raise ValueError('Per-rank batch must be divisible by4')
        self.batch_size, self.steps, self.start_step = batch_size, steps, start_step
        self.seed, self.rank = seed, rank
        self.groups = {}
        for domain in ('pfnn', 'parc', 'bones'):
            selected = [i for i, actual in enumerate(dataset.ids) if dataset.records[actual]['domain'] == domain]
            self.groups[domain] = np.asarray(selected[rank::world_size], dtype=np.int64)
            if not len(self.groups[domain]):
                raise ValueError('A rank has no admitted '+domain+' training examples')
        self.quota = dict(pfnn=batch_size//2, parc=batch_size//4, bones=batch_size//4)
        self.permutations = {}

    def permutation(self, domain, cycle):
        key = (domain, cycle)
        if key not in self.permutations:
            salt = ('pfnn', 'parc', 'bones').index(domain)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.rank, salt, cycle]))
            self.permutations[key] = rng.permutation(self.groups[domain])
            for previous in list(self.permutations):
                if previous[0] == domain and previous[1] < cycle-1:
                    del self.permutations[previous]
        return self.permutations[key]

    def indices(self, step):
        values = []
        for domain, quota in self.quota.items():
            position, remaining = step*quota, quota
            while remaining:
                cycle, offset = divmod(position, len(self.groups[domain]))
                take = min(remaining, len(self.groups[domain])-offset)
                values.extend(self.permutation(domain, cycle)[offset:offset+take].tolist())
                position += take
                remaining -= take
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.rank, step, 999]))
        rng.shuffle(values)
        return values

    def __iter__(self):
        for step in range(self.start_step, self.steps):
            yield self.indices(step)

    def __len__(self):
        return self.steps-self.start_step
