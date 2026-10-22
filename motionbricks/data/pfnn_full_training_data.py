"""Full PFNN cache coverage, source-aligned scenes and complete epoch sampling.

All intervals are half-open source-frame intervals at 30 Hz. Scene/contact
intervals are never joined across pair records. No data is regenerated here.
"""

from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .pfnn_world_contract import VERSION as WORLD_CONTRACT
from .reactive_training_data import CATEGORIES, WORLD_ATOL, _category


FPS = 30
WINDOW = 33
SPLITS = ('train', 'validation', 'test')
SPAN_FRAMES = (32, 60, 120, 240, 480)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return type(value) is int and value >= 0


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _checked_path(root, relative, expected):
    root = Path(root).resolve()
    _require(isinstance(relative, str) and bool(relative), 'Missing source artifact path')
    path = (root / relative).resolve()
    _require(root in path.parents, 'Source artifact escapes dataset: ' + relative)
    _require(_digest(expected) and sha256(path) == expected, 'Source SHA256 differs: ' + relative)
    return path


def merge_intervals(intervals):
    """Union observed timestamps, without filling missing frames."""
    result = []
    for start, stop in sorted(intervals):
        _require(_integer(start) and _integer(stop) and stop > start, 'Invalid source interval')
        if result and start <= result[-1][1]:
            result[-1][1] = max(stop, result[-1][1])
        else:
            result.append([start, stop])
    return result


def _continuous_spans(starts):
    # Abutting windows share no observed interval across their boundary. Only
    # overlapping windows establish continuity of both frames and transitions.
    spans = []
    for start in sorted(starts):
        stop = start + WINDOW
        if spans and start < spans[-1][1]:
            spans[-1][1] = max(stop, spans[-1][1])
        else:
            spans.append([start, stop])
    return spans


def _span_counts(spans, lengths):
    return {str(length): dict(
        qualifying_spans=sum(stop - start >= length for start, stop in spans),
        valid_starts=sum(max(0, stop - start - length + 1) for start, stop in spans),
    ) for length in lengths}


def _source_key(row):
    return row['source_id'].split(':terrain:', 1)[0]


class PFNNCacheIndex:
    """Validate all metadata before exposing any split or cached sample."""

    def __init__(self, cache, pfnn):
        self.cache, self.pfnn = Path(cache), Path(pfnn)
        manifest_path = self.pfnn / 'manifest.json'
        self.manifest, self.complete = _json(manifest_path), _json(self.cache / 'complete.json')
        self.manifest_sha256 = sha256(manifest_path)
        _require(self.manifest.get('schema') == 'motionbricks.pfnn_static_pairs.v2'
                 and self.manifest.get('world_contract') == WORLD_CONTRACT,
                 'PFNN requires dataset_world_v2 world/contact schema')
        _require(self.manifest.get('pilot') is False and self.complete.get('pilot') is False,
                 'Full-data loader requires non-pilot manifest and cache')
        _require(self.complete.get('schema') == 'motionbricks.full_terrain_primitives.v1'
                 and self.complete.get('source_manifests', {}).get('pfnn') == self.manifest_sha256,
                 'Cache source manifest SHA256 binding differs')
        dataset_complete = self.pfnn / 'complete.json'
        if dataset_complete.exists():
            _require(_json(dataset_complete).get('manifest_sha256') == self.manifest_sha256,
                     'Dataset complete manifest SHA256 differs')
        self.sources, groups, clips, bvh_groups, motion_identities = {}, {}, {}, {}, {}
        expected = set()
        for row in self.manifest['records']:
            source_id, group, split = row.get('source_id'), row.get('source_group'), row.get('split')
            _require(isinstance(source_id, str) and bool(source_id)
                     and isinstance(group, str) and bool(group) and split in SPLITS,
                     'Missing source identity/group/split')
            _require(source_id not in self.sources, 'Duplicate source_id in manifest')
            _require(group not in groups or groups[group] == split, 'Source group crosses splits: ' + group)
            groups[group] = split
            _require(row.get('world_contract') == WORLD_CONTRACT, 'Source world contract differs')
            _require(all(_digest(row.get(key + '_sha256')) for key in ('motion', 'geometry', 'contact'))
                     and _digest(row.get('source_bvh_sha256')), 'Missing source SHA256 identity')
            offset, count = row.get('contact_source_offset'), row.get('contact_source_frames')
            _require(_integer(offset) and _integer(count) and count > 0, 'Invalid source contact clock')
            starts = row['eligible_window_starts']
            _require(all(_integer(start) for start in starts) and len(set(starts)) == len(starts),
                     'Duplicate or invalid eligible start')
            _require(all(offset <= start and start + WINDOW <= offset + count for start in starts),
                     'Eligible window escapes source contact clock')
            identity = (group, split, row['source_bvh_sha256'], row['motion_shard'], row['motion_sha256'])
            clip = _source_key(row)
            _require(clip not in clips or clips[clip] == identity, 'Original source identity/split differs')
            bvh = row['source_bvh_sha256']
            _require(bvh not in bvh_groups or bvh_groups[bvh] == (group, split),
                     'Original BVH identity/group crosses splits')
            bvh_groups[bvh] = (group, split)
            motion_identity = (clip, *identity[:3], row['motion_sha256'])
            _require(row['motion_shard'] not in motion_identities
                     or motion_identities[row['motion_shard']] == motion_identity,
                     'Motion shard identity/group/split differs')
            motion_identities[row['motion_shard']] = motion_identity
            clips[clip] = identity
            self.sources[source_id] = row
            expected.update((source_id, start, quarter) for start in starts for quarter in range(4))
        self.records = [json.loads(line) for line in
                        (self.cache / 'data/records.jsonl').read_text(encoding='utf-8').splitlines()]
        seen = set()
        for record in self.records:
            source = self.sources.get(record.get('source_id'))
            _require(record.get('domain') == 'pfnn' and source is not None, 'Unknown cache domain/source')
            _require(_integer(record.get('start')) and type(record.get('quarter')) is int
                     and 0 <= record['quarter'] < 4, 'Invalid cache start/quarter')
            _require(all(record.get(key) == source[key] for key in ('source_id', 'source_group', 'split'))
                     and record.get('source_sha256') == source['source_bvh_sha256'],
                     'Cache source identity/group/split/SHA256 differs')
            key = (record['source_id'], record['start'], record['quarter'])
            _require(key not in seen, 'Duplicate augmented cache identity')
            seen.add(key)
        _require(seen == expected,
                 f'Cache four-quarter coverage differs: missing={len(expected - seen)}, extra={len(seen - expected)}')
        summary = self.complete.get('data', {})
        _require(summary.get('examples') == len(self.records), 'Cache complete example count differs')
        _require(self.complete.get('considered_base_windows') == len(expected) // 4
                 and self.complete.get('rejected_base_windows') == 0, 'Cache complete admission count differs')
        if 'admitted_windows' in self.manifest:
            _require(self.manifest['admitted_windows'] == len(expected) // 4, 'Manifest admission count differs')
        self.array_headers = {}
        fields = summary.get('fields', {})
        _require('world' in fields, 'Cache complete lacks raw world field')
        for key, spec in fields.items():
            _require(Path(key).name == key and '/' not in key and '\\' not in key, 'Invalid cache field path')
            array = np.load(self.cache / 'data' / (key + '.npy'), mmap_mode='r', allow_pickle=False)
            _require(list(array.shape) == spec['shape'] and str(array.dtype) == spec['dtype']
                     and array.shape[0] == len(self.records), 'Cache array header differs: ' + key)
            self.array_headers[key] = dict(shape=list(array.shape), dtype=str(array.dtype))
        self.world = np.load(self.cache / 'data/world.npy', mmap_mode='r', allow_pickle=False)
        _require(self.world.shape == (len(self.records), WINDOW, 366) and self.world.dtype == np.float32,
                 'Cached world must be float32 [records,33,366]')


class _ByteLRU:
    def __init__(self, limit):
        _require(type(limit) is int and limit >= 0, 'LRU byte limit must be nonnegative')
        self.limit, self.bytes, self.values = limit, 0, OrderedDict()

    def get(self, key, load):
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key][0]
        value, size = load()
        while self.values and self.bytes + size > self.limit:
            _, (_, previous) = self.values.popitem(last=False)
            self.bytes -= previous
        if size <= self.limit:
            self.values[key] = value, size
            self.bytes += size
        return value


def _load_motion(root, row):
    path = _checked_path(root, row['motion_shard'], row['motion_sha256'])
    with np.load(path, allow_pickle=False) as archive:
        _require(float(archive['fps']) == FPS and str(archive['world_contract']) == WORLD_CONTRACT,
                 'Source motion FPS/world contract differs')
        world = archive['global_features'].copy()
    _require(world.ndim == 2 and world.shape[1] == 366 and world.dtype == np.float32
             and np.isfinite(world).all(), 'Source world must be finite float32 [frames,366]')
    return world


def _load_pair(root, row, frames):
    path = _checked_path(root, row['contact_shard'], row['contact_sha256'])
    with np.load(path, allow_pickle=False) as archive:
        contacts = archive['contacts'].copy()
        offset = int(archive['source_frame_offset'])
        _require(offset == row['contact_source_offset'] and len(contacts) == row['contact_source_frames']
                 and float(archive['fps']) == FPS and np.array_equal(archive['source_frame_indices'],
                     np.arange(offset, offset + len(contacts))), 'Source contact clock differs')
    _require(0 <= offset and offset + len(contacts) <= frames
             and contacts.shape == (row['contact_source_frames'], 4)
             and np.isfinite(contacts).all() and ((contacts == 0) | (contacts == 1)).all(),
             'Scene contacts escape real source frames or contain invalid labels')
    path = _checked_path(root, row['geometry_shard'], row['geometry_sha256'])
    with np.load(path, allow_pickle=False) as archive:
        scene = {key: torch.from_numpy(np.array(archive[key], copy=True)) for key in archive.files}
    _require({'heightfield_y_m', 'min_point_world_xyz_m', 'dx_m'} <= set(scene)
             and all(torch.isfinite(value).all() for value in scene.values()), 'Invalid source terrain arrays')
    _require(scene['heightfield_y_m'].ndim == 2 and min(scene['heightfield_y_m'].shape) >= 2
             and scene['min_point_world_xyz_m'].shape == (3,) and scene['dx_m'].numel() == 1
             and float(scene['dx_m']) > 0, 'Invalid source terrain dimensions/spacing')
    if 'heightfield_valid' in scene:
        _require(scene['heightfield_valid'].shape == scene['heightfield_y_m'].shape
                 and scene['heightfield_valid'].dtype == torch.bool, 'Invalid terrain validity mask')
    return contacts, scene


def _scene_chart(scene, original_first, cached_first, quarter):
    """Recover the cache's rotation AND translation from its actual first sample."""
    old_yaw = torch.atan2(original_first[4], original_first[3])
    new_yaw = torch.atan2(cached_first[4], cached_first[3])
    angle_error = torch.atan2(torch.sin(new_yaw - old_yaw - quarter * np.pi / 2),
                              torch.cos(new_yaw - old_yaw - quarter * np.pi / 2))
    _require(abs(float(angle_error)) < WORLD_ATOL, 'Cached quarter heading differs from source')
    rotations = (((1., 0.), (0., 1.)), ((0., 1.), (-1., 0.)),
                 ((-1., 0.), (0., -1.)), ((0., -1.), (1., 0.)))
    rotation = original_first.new_tensor(rotations[quarter])
    shift = cached_first[[0, 2]] - rotation @ original_first[[0, 2]]
    origin = scene['min_point_world_xyz_m'].clone().float()
    height, spacing = scene['heightfield_y_m'], float(scene['dx_m'])
    width, depth = (height.shape[1] - 1) * spacing, (height.shape[0] - 1) * spacing
    corners = torch.stack([origin[[0, 2]] + origin.new_tensor([x, z])
                           for x, z in ((0, 0), (width, 0), (0, depth), (width, depth))])
    origin[[0, 2]] = (corners @ rotation.T).amin(0) + shift
    changed = dict(heightfield_y_m=torch.rot90(height, quarter, (0, 1)).contiguous(),
                   min_point_world_xyz_m=origin, dx_m=torch.tensor(spacing))
    if 'heightfield_valid' in scene:
        changed['heightfield_valid'] = torch.rot90(scene['heightfield_valid'], quarter, (0, 1)).contiguous()
    return changed, rotation, shift


def input_intent_labels(world, source_frame_offset=0):
    """Source-derived immediate commands, without padded or synthetic frames.

    At current source frame t (3..T-5), observed speed is the mean of the three
    intervals t-3..t-1. Future speed is the mean of the four interval norms
    t..t+3. A stopping interval requires observed speed >0.12 m/s, future speed
    <=0.12 m/s, and a speed reduction >0.05 m/s. Only these intervals receive a
    zero current velocity command while the genuine source continues braking.
    Other commands use measured mean next-four velocity. Facing is the source
    heading at t+4. Measured velocity/speed remain separate audit targets.
    """
    values = np.asarray(world.detach().cpu() if isinstance(world, torch.Tensor) else world)
    _require(values.ndim == 2 and values.shape[1] == 366 and len(values) >= 8
             and np.isfinite(values).all() and _integer(source_frame_offset), 'Intent labels require real source frames')
    _require(np.allclose(np.linalg.norm(values[:, 3:5], axis=-1), 1., atol=1e-3, rtol=0),
             'Intent headings must be unit cos/sin pairs')
    current = np.arange(3, len(values) - 4)
    velocity = np.diff(values[:, [0, 2]].astype(np.float64), axis=0) * FPS
    speed = np.linalg.norm(velocity, axis=-1)
    history_speed = np.array([speed[t - 3:t].mean() for t in current])
    future_speed = np.array([speed[t:t + 4].mean() for t in current])
    measured_velocity = ((values[current + 4][:, [0, 2]] - values[current][:, [0, 2]]) * (FPS / 4)).astype(np.float32)
    command = np.concatenate((measured_velocity,
                              values[current + 4, 3:5]), axis=-1).astype(np.float32)
    stop = (history_speed > .12) & (future_speed <= .12) & (history_speed - future_speed > .05)
    command[stop, :2] = 0
    return dict(source_frames=torch.from_numpy(current + source_frame_offset),
                command=torch.from_numpy(command), stop_event=torch.from_numpy(stop),
                measured_next4_velocity_mps=torch.from_numpy(measured_velocity.copy()),
                history_speed_mps=torch.from_numpy(history_speed.astype(np.float32)),
                next4_speed_mps=torch.from_numpy(future_speed.astype(np.float32)))


class PFNNFrameDataset(Dataset):
    """All selected cache rows, with CPU source LRU and canonical paired terrain.

    ``records`` and sampler indices are split-local; ``cache_row_index`` retains
    the immutable original cache row. Item world has shape [33,366].
    """

    def __init__(self, cache, pfnn, rep_cpu, split, *, source_cache_bytes=256 * 1024**2,
                 scene_cache_bytes=64 * 1024**2):
        _require(split in SPLITS, 'Unknown split')
        if isinstance(rep_cpu, torch.nn.Module):
            _require(all(value.device.type == 'cpu' for value in
                         (*rep_cpu.parameters(), *rep_cpu.buffers())), 'Representation must remain on CPU')
        self.index, self.rep, self.split = PFNNCacheIndex(cache, pfnn), rep_cpu, split
        self.ids = [i for i, row in enumerate(self.index.records) if row['split'] == split]
        self.records = []
        for actual in self.ids:
            category, summary = _category(np.array(self.index.world[actual, :, :5], copy=True))
            self.records.append(dict(self.index.records[actual], cache_row_index=actual,
                                     category=category, root_summary=summary))
        self.source_cache, self.scene_cache = _ByteLRU(source_cache_bytes), _ByteLRU(scene_cache_bytes)

    def __len__(self):
        return len(self.ids)

    def _motion(self, row):
        def load():
            world = _load_motion(self.index.pfnn, row)
            return world, world.nbytes
        return self.source_cache.get((row['motion_shard'], row['motion_sha256']), load)

    @torch.no_grad()
    def __getitem__(self, index):
        record = self.records[index]
        row = self.index.sources[record['source_id']]
        source = self._motion(row)

        def load_pair():
            contacts, scene = _load_pair(self.index.pfnn, row, len(source))
            size = contacts.nbytes + sum(value.numel() * value.element_size() for value in scene.values())
            return (contacts, scene), size

        contacts, scene = self.scene_cache.get(row['source_id'], load_pair)
        start, quarter = record['start'], record['quarter']
        _require(start + WINDOW <= len(source), 'Window escapes actual source frames')
        original = torch.from_numpy(source[start:start + WINDOW].copy())
        contact_start = start - row['contact_source_offset']
        original[:, self.rep.indices['foot_contacts']] = torch.from_numpy(contacts[contact_start:contact_start + WINDOW]).to(original)
        world = torch.from_numpy(np.array(self.index.world[self.ids[index]], copy=True))
        _require(torch.isfinite(world).all(), 'Cached world must be finite')
        transformed, _, _ = _scene_chart(scene, original[0], world[0], quarter)
        heading = torch.atan2(original[:1, 4], original[:1, 3]) + quarter * np.pi / 2
        recovered = self.rep.change_first_heading(original[None], heading, False, False)[0]
        _require(float((recovered[0] - world[0]).abs().max()) <= WORLD_ATOL,
                 'Cached first sample differs from source chart')
        _require(torch.equal(original[:, self.rep.indices['foot_contacts']], world[:, self.rep.indices['foot_contacts']]),
                 'Cached contacts differ from source clock')
        _require(float((recovered - world).abs().max()) <= WORLD_ATOL,
                 'Cached window differs from source world/Root chart')
        return dict(world=world, scene=transformed, record=dict(record),
                    command_sequence=input_intent_labels(world, start))


class FullEpochSampler(Sampler):
    """Complete primary permutation, then separately counted category extras.

    Rank shards are strided slices of one global order. They may have unequal
    lengths; use DDP join or explicit uneven-input handling, never padded IDs.
    """

    def __init__(self, datasetrecords, split='train', seed=2333, *, rank=0, world_size=1, extras=None):
        records = datasetrecords.records if isinstance(datasetrecords, PFNNFrameDataset) else datasetrecords
        _require(split == 'train', 'FullEpochSampler only samples train')
        _require(type(seed) is int and seed >= 0 and type(world_size) is int and world_size > 0
                 and type(rank) is int and 0 <= rank < world_size, 'Invalid sampler seed/rank/world_size')
        self.seed, self.rank, self.world_size, self.epoch = seed, rank, world_size, 0
        self.ids = [i for i, row in enumerate(records) if row['split'] == split]
        _require(bool(self.ids), 'No training rows')
        self.pools = {category: [] for category in CATEGORIES}
        for i in self.ids:
            _require(records[i].get('domain') == 'pfnn' and records[i].get('category') in CATEGORIES,
                     'Sampler requires actual PFNN category labels on every training record')
            self.pools[records[i]['category']].append(i)
        self.extras = len(self.ids) // 4 if extras is None else extras
        _require(_integer(self.extras), 'Extra exposure count must be a nonnegative integer')
        self.set_epoch(0)

    def set_epoch(self, epoch):
        _require(_integer(epoch), 'Epoch must be a nonnegative integer')
        self.epoch = epoch
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        self.global_primary_indices = rng.permutation(self.ids).tolist()
        available = [key for key in CATEGORIES if self.pools[key]]
        categories = []
        while len(categories) < self.extras:
            categories.extend(rng.permutation(available).tolist())
        self.global_extra_indices, cycles, cursors = [], {}, Counter()
        for category in categories[:self.extras]:
            if category not in cycles or cursors[category] == len(cycles[category]):
                cycles[category] = rng.permutation(self.pools[category]).tolist()
                cursors[category] = 0
            self.global_extra_indices.append(cycles[category][cursors[category]])
            cursors[category] += 1
        self.primary_indices = self.global_primary_indices[self.rank::self.world_size]
        self.extra_indices = self.global_extra_indices[self.rank::self.world_size]
        self.extra_category_counts = dict(Counter(categories[:self.extras]))

    def phases(self):
        return dict(primary=list(self.primary_indices), extra=list(self.extra_indices))

    def coverage_receipt(self):
        return dict(epoch=self.epoch, seed=self.seed, rank=self.rank, world_size=self.world_size,
                    expected_primary=len(self.ids), visited_primary=len(self.global_primary_indices),
                    missing_primary_ids=sorted(set(self.ids) - set(self.global_primary_indices)),
                    duplicate_primary_count=len(self.global_primary_indices) - len(set(self.global_primary_indices)),
                    extra_exposures=len(self.global_extra_indices),
                    extra_categories={key: self.extra_category_counts.get(key, 0) for key in CATEGORIES},
                    missing_categories=[key for key in CATEGORIES if not self.pools[key]],
                    rank_primary=len(self.primary_indices), rank_extra=len(self.extra_indices), padding=0, dropped=0)

    def __iter__(self):
        return iter(self.primary_indices + self.extra_indices)

    def __len__(self):
        return len(self.primary_indices) + len(self.extra_indices)


def audit_full_coverage(cache, pfnn, inventory=None, *, span_frames=SPAN_FRAMES):
    """Return JSON-ready P0 audit and per-file coverage, including zero files.

    Every referenced source/geometry/contact artifact is hashed. Original BVH
    coverage unions pairings; continuous spans retain pair IDs and clocks.
    Inventory is an explicit optional path to the original source inventory.
    """
    lengths = tuple(sorted(set(span_frames)))
    _require(lengths and all(type(value) is int and value > 0 for value in lengths), 'Invalid span lengths')
    index = PFNNCacheIndex(cache, pfnn)
    files, original_inventory = {}, _json(inventory) if inventory is not None else None
    if original_inventory is not None:
        _require(index.manifest.get('source_inventory_sha256') == sha256(inventory), 'Original inventory SHA256 differs')
        for clip in original_inventory['clips']:
            if clip['role'] != 'motion':
                continue
            key = 'pfnn:' + Path(clip['filename']).stem
            _require(key not in files, 'Duplicate BVH in original inventory')
            frames = int(np.floor((clip['frames'] - 1) / clip['fps'] * FPS + 1e-8)) + 1
            files[key] = dict(source_id=key, filename=clip['filename'], source_bvh_sha256=clip['sha256'],
                              source_group='pfnn:' + clip['source_recording_group'], mirrored=clip['mirrored'],
                              frames=frames, intervals=[], pairs=[], admitted_windows=0)
    for source in index.manifest.get('source_audit', []):
        key = source['source_id']
        if key not in files:
            _require(original_inventory is None, 'Manifest source absent from original inventory')
            files[key] = dict(source_id=key, filename=None, frames=source['world_encoding']['actual_frames'],
                              intervals=[], pairs=[], admitted_windows=0)
        _require(files[key]['frames'] == source['world_encoding']['actual_frames'], 'Original source frame count differs')
    motion_cache = _ByteLRU(256 * 1024**2)
    pair_reports, by_split = [], {split: dict(cache_rows=0, pair_windows=0, source_ids=set(), groups=set(),
                                             spans=[], categories=Counter()) for split in SPLITS}
    for row in index.sources.values():
        key = _source_key(row)
        def load_motion():
            value = _load_motion(index.pfnn, row)
            return value, value.nbytes
        world = motion_cache.get((row['motion_shard'], row['motion_sha256']), load_motion)
        contacts, _ = _load_pair(index.pfnn, row, len(world))
        if key not in files:
            _require(original_inventory is None, 'Manifest source absent from original inventory')
            files[key] = dict(source_id=key, filename=row['motion_shard'], frames=len(world),
                              intervals=[], pairs=[], admitted_windows=0)
        file = files[key]
        _require(file['frames'] == len(world), 'Original source frame count differs')
        for field, value in (('source_bvh_sha256', row['source_bvh_sha256']),
                             ('source_group', row['source_group']), ('split', row['split'])):
            _require(field not in file or file[field] == value, 'Original BVH identity/group/split differs')
            file[field] = value
        spans = _continuous_spans(row['eligible_window_starts'])
        file['intervals'].extend((start, start + WINDOW) for start in row['eligible_window_starts'])
        file['pairs'].append(row['source_id'])
        file['admitted_windows'] += len(row['eligible_window_starts'])
        pair_reports.append(dict(source_id=row['source_id'], original_source_id=key, split=row['split'],
            geometry_sha256=row['geometry_sha256'], contact_sha256=row['contact_sha256'],
            contact_clock=[row['contact_source_offset'], row['contact_source_offset'] + len(contacts)],
            admitted_windows=len(row['eligible_window_starts']), spans=spans,
            span_counts=_span_counts(spans, lengths)))
        bucket = by_split[row['split']]
        bucket['pair_windows'] += len(row['eligible_window_starts'])
        bucket['spans'].extend(spans)
        bucket['source_ids'].add(row['source_id'])
        bucket['groups'].add(row['source_group'])
    if original_inventory is None:
        _require(len(files) == index.manifest.get('full_motion_clips_considered'),
                 'Cannot enumerate zero-admission files; supply original inventory or full source_audit')
    else:
        _require(len(files) == index.manifest.get('full_motion_clips_considered'), 'Original inventory file count differs')
    for actual, row in enumerate(index.records):
        category, _ = _category(np.array(index.world[actual, :, :5], copy=True))
        bucket = by_split[row['split']]
        bucket['cache_rows'] += 1
        bucket['categories'][category] += 1
    per_file = []
    pair_by_file = defaultdict(list)
    for pair in pair_reports:
        pair_by_file[pair['original_source_id']].extend(pair['spans'])
    for key, file in sorted(files.items()):
        intervals = merge_intervals(file.pop('intervals'))
        covered = sum(stop - start for start, stop in intervals)
        file.update(covered_intervals=intervals, covered_frames=covered,
                    covered_seconds=covered / FPS, time_extent_seconds=sum(max(0, b - a - 1) / FPS for a, b in intervals),
                    coverage_fraction=covered / file['frames'],
                    span_counts=_span_counts(pair_by_file[key], lengths),
                    max_same_scene_span_frames=max((b - a for a, b in pair_by_file[key]), default=0))
        per_file.append(file)
    split_reports = {}
    for split, bucket in by_split.items():
        split_reports[split] = dict(cache_rows=bucket['cache_rows'], pair_windows=bucket['pair_windows'],
            source_ids=len(bucket['source_ids']), source_groups=len(bucket['groups']),
            categories={key: bucket['categories'][key] for key in CATEGORIES},
            span_counts=_span_counts(bucket['spans'], lengths))
    return dict(schema='motionbricks.pfnn_full_coverage.v1', passed=True, phase='P0',
        training_exposures=0, optimizer_updates=0, fps=FPS, window_frames=WINDOW,
        interval_convention='half-open source frames; spans join overlapping admitted windows within one pair only',
        span_lengths_frames=list(lengths), manifest_sha256=index.manifest_sha256,
        cache_complete_sha256=sha256(index.cache / 'complete.json'),
        cache_records_sha256=sha256(index.cache / 'data/records.jsonl'), array_headers=index.array_headers,
        inventory_sha256=sha256(inventory) if inventory is not None else None,
        all_quarters_verified=True, source_artifacts_verified=True, original_files=len(per_file),
        zero_admission_files=[file['source_id'] for file in per_file if not file['admitted_windows']],
        augmented_rows=len(index.records), admitted_pair_windows=len(index.records) // 4,
        source_groups=len({row['source_group'] for row in index.sources.values()}),
        splits=split_reports, per_file=per_file, pairs=pair_reports)
