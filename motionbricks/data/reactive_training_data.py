"""Bounded, source-stratified selection and factual scene recovery for a pilot.

Cache paths name existing cache roots (each containing ``data/records.jsonl``).
Parent complete/index hashes are checked by the caller. Selection reads at most
eight candidates per requested record, and only memory-maps ``world.npy``.
Categories describe source motion, not measured joystick commands.
"""

from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .full_terrain_cache import PrimitiveDataset
from .pfnn_world_contract import VERSION as WORLD_CONTRACT, apply_scene_contacts
from .terrain_learning_data import TerrainLearningDataset


TRAIN_QUOTAS = dict(pfnn=48, parc=24, bones=24)
VALIDATION_QUOTAS = dict(pfnn=12, parc=6, bones=6)
CATEGORIES = ('idle', 'start', 'stop', 'strafe', 'reverse', 'forward')
CANDIDATES_PER_QUOTA = 8
WORLD_ATOL = 3e-5


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _identity(row):
    for key in ('source_id', 'source_group'):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError('Missing source identity: ' + key)
    if row.get('split') not in ('train', 'validation', 'test'):
        raise ValueError('Unknown source split')


def _cache_index(cache_paths, split):
    if split not in ('train', 'validation'):
        raise ValueError('Reactive selection uses train or validation only')
    roots = [Path(path) for path in cache_paths]
    data = PrimitiveDataset([root / 'data' for root in roots], split)
    groups = {}
    for row in data.records:
        _identity(row)
        group = row['source_group']
        if group in groups and groups[group] != row['split']:
            raise ValueError('Source group crosses splits: ' + group)
        groups[group] = row['split']
    return roots, data


def _world_map(data, cache_index, maps):
    if cache_index not in maps:
        world = np.load(data.folders[cache_index] / 'world.npy', mmap_mode='r', allow_pickle=False)
        count = data.offsets[cache_index + 1] - data.offsets[cache_index]
        if world.shape != (count, 33, 366) or world.dtype != np.float32:
            raise ValueError('Cached world must be float32 [records, 33, 366]')
        maps[cache_index] = world
    return maps[cache_index]


def _category(root):
    if root.shape != (33, 5) or not np.isfinite(root).all():
        raise ValueError('Source classification requires finite 33-frame roots')
    if not np.allclose(np.linalg.norm(root[:, 3:5], axis=-1), 1., atol=1e-3, rtol=0):
        raise ValueError('Source headings must be unit cos/sin pairs')
    velocity = np.diff(root[:, [0, 2]].astype(np.float64), axis=0) * 30.
    # Four observed frames give three intervals; each future block gives four.
    speeds = [float(np.linalg.norm(part, axis=-1).mean())
              for part in (velocity[:3], velocity[3:7], velocity[-4:])]
    history, following, terminal = speeds
    if max(speeds) <= .12:
        category = 'idle'
    elif history <= .12 and max(following, terminal) > .12:
        category = 'start'
    elif max(history, following) > .12 and terminal <= .12:
        category = 'stop'
    else:
        part, headings = (velocity[3:7], root[4:8]) if following > .12 else (velocity[-4:], root[-4:])
        # SOMA yaw=0 faces +Z, hence the planar forward vector is [sin, cos].
        forward = headings[:, [4, 3]]
        along = float((part * forward).sum(-1).mean())
        lateral = float(np.abs(part[:, 0] * forward[:, 1] - part[:, 1] * forward[:, 0]).mean())
        category = 'strafe' if lateral > abs(along) else 'reverse' if along < 0 else 'forward'
    return category, dict(zip(('history_speed_mps', 'next4_speed_mps', 'last4_speed_mps'), speeds))


def select_records(cache_paths, split, quotas, seed=20261021):
    """Return ordinary selection dictionaries, with cache-local original row IDs.

    Only q0 participates. A seeded round-robin over shuffled source groups bounds
    candidate reads to ``8 * quota`` per domain. Among those candidates, unused
    groups take precedence, followed by the least represented actual category.
    Missing categories remain missing; insufficient records raise an error.
    Use :func:`selection_receipt` to count the returned rows for a run receipt.
    """
    if (set(quotas) - set(TRAIN_QUOTAS)
            or any(type(count) is not int or count < 0 for count in quotas.values())):
        raise ValueError('Quotas must be nonnegative integer pfnn/parc/bones counts')
    _, data = _cache_index(cache_paths, split)
    pools = {domain: defaultdict(list) for domain in quotas}
    identities = set()
    for actual in data.ids:
        record = data.records[actual]
        if record.get('domain') not in TRAIN_QUOTAS:
            raise ValueError('Unknown cache domain')
        if type(record.get('quarter')) is not int or not 0 <= record['quarter'] <= 3:
            raise ValueError('Invalid cached quarter rotation')
        if type(record.get('start')) is not int or record['start'] < 0:
            raise ValueError('Invalid real source start')
        domain = record['domain']
        if record['quarter'] or not quotas.get(domain, 0):
            continue
        identity = (domain, record['source_id'], record['start'])
        if identity in identities:
            raise ValueError('Duplicate q0 source window across caches')
        identities.add(identity)
        pools[domain][record['source_group']].append(actual)

    selections, maps = [], {}
    for salt, domain in enumerate(TRAIN_QUOTAS):
        quota = quotas.get(domain, 0)
        if not quota:
            continue
        pool = pools[domain]
        available = sum(map(len, pool.values()))
        if available < quota:
            raise ValueError(f'Insufficient admitted q0 {split}/{domain}: {available} < {quota}')
        rng = np.random.default_rng(np.random.SeedSequence([seed, salt]))
        groups = sorted(pool)
        rng.shuffle(groups)
        queues = [rng.permutation(pool[group]).tolist() for group in groups]
        candidates = []
        depth = 0
        limit = min(available, CANDIDATES_PER_QUOTA * quota)
        while len(candidates) < limit:
            for queue in queues:
                if depth >= len(queue):
                    continue
                actual = queue[depth]
                cache_index = bisect_right(data.offsets, actual) - 1
                row_index = actual - data.offsets[cache_index]
                world = _world_map(data, cache_index, maps)
                category, summary = _category(np.array(world[row_index, :, :5], copy=True))
                candidates.append(dict(cache_index=cache_index, row_index=row_index,
                    sample_id=actual, record=dict(data.records[actual]), category=category,
                    root_summary=summary))
                if len(candidates) == limit:
                    break
            depth += 1
        used, counts = set(), Counter()
        for _ in range(quota):
            chosen = min(range(len(candidates)), key=lambda i: (
                candidates[i]['record']['source_group'] in used,
                counts[candidates[i]['category']], i))
            row = candidates.pop(chosen)
            used.add(row['record']['source_group'])
            counts[row['category']] += 1
            selections.append(row)
    return selections


def selection_receipt(selection):
    """JSON-ready counts of actual rows, including zero category coverage."""
    domains = {}
    for domain in TRAIN_QUOTAS:
        rows = [row for row in selection if row['record']['domain'] == domain]
        counts = Counter(row['category'] for row in rows)
        domains[domain] = dict(count=len(rows), categories={key: counts[key] for key in CATEGORIES},
            source_groups=len({row['record']['source_group'] for row in rows}))
    return dict(count=len(selection), splits=dict(Counter(row['record']['split'] for row in selection)),
        domains=domains, quarter=0)


def _checked_file(root, relative, expected):
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or _sha256(path) != expected:
        raise ValueError('Source path or SHA256 differs: ' + relative)
    return path


def _source_index(records):
    result, groups = {}, {}
    for row in records:
        _identity(row)
        if row['source_id'] in result:
            raise ValueError('Duplicate source_id in source manifest')
        group = row['source_group']
        if group in groups and groups[group] != row['split']:
            raise ValueError('Source group crosses source splits: ' + group)
        groups[group] = row['split']
        result[row['source_id']] = row
    return result


class ReactiveSourceBank:
    """Recover one unchanged source window and its jointly transformed CPU scene.

    Paths are explicit caller inputs; no remote path or dataset version is
    inferred. Cache source-manifest bindings and selected source files are
    validated here. Original admission is retained without geometry rescreening.
    """
    def __init__(self, cache_paths, pfnn_path, mixed_path, rep_cpu):
        self.roots, self.data = _cache_index(cache_paths, 'train')
        self.pfnn_path, self.mixed_path = Path(pfnn_path), Path(mixed_path)
        if isinstance(rep_cpu, torch.nn.Module) and any(
                value.device.type != 'cpu' for value in (*rep_cpu.parameters(), *rep_cpu.buffers())):
            raise ValueError('ReactiveSourceBank requires a CPU representation')
        self.rep = rep_cpu
        self.maps, self.datasets, self.sources = {}, {}, {}

    def _sources(self, cache_index, kind):
        key = (cache_index, kind)
        if key in self.sources:
            return self.sources[key]
        root = self.pfnn_path if kind == 'pfnn' else self.mixed_path
        name = 'manifest.json' if kind == 'pfnn' else 'native_manifest.json'
        manifest = _json(root / name)
        complete = _json(self.roots[cache_index] / 'complete.json')
        if (complete.get('schema') != 'motionbricks.full_terrain_primitives.v1'
                or complete.get('source_manifests', {}).get(kind) != _sha256(root / name)):
            raise ValueError('Cache source manifest SHA256 binding differs: ' + kind)
        if kind == 'pfnn':
            if (manifest.get('schema') != 'motionbricks.pfnn_static_pairs.v2'
                    or manifest.get('world_contract') != WORLD_CONTRACT):
                raise ValueError('PFNN requires the factual world/contact contract')
            records = manifest['records']
        else:
            _checked_file(root, 'accepted.json', manifest['artifact_sha256']['accepted.json'])
            records = _json(root / 'accepted.json')
        self.sources[key] = root, manifest, _source_index(records)
        return self.sources[key]

    def _pfnn(self, root, row, record):
        if row.get('world_contract') != WORLD_CONTRACT:
            raise ValueError('PFNN source row world contract differs')
        if record['start'] not in row['eligible_window_starts']:
            raise ValueError('Window is not in the original PFNN admission')
        paths = {key: _checked_file(root, row[key + '_shard'], row[key + '_sha256'])
                 for key in ('motion', 'geometry', 'contact')}
        with np.load(paths['motion'], allow_pickle=False) as archive:
            if str(archive['world_contract']) != WORLD_CONTRACT or float(archive['fps']) != 30.:
                raise ValueError('PFNN motion world contract or FPS differs')
            world = torch.from_numpy(archive['global_features'].copy())
        with np.load(paths['contact'], allow_pickle=False) as archive:
            contacts = torch.from_numpy(archive['contacts'].copy())
            offset = int(archive['source_frame_offset'])
            if (offset != row['contact_source_offset'] or len(contacts) != row['contact_source_frames']
                    or float(archive['fps']) != 30.
                    or not np.array_equal(archive['source_frame_indices'], np.arange(offset, offset + len(contacts)))):
                raise ValueError('PFNN contact sidecar source clock differs')
        if record['start'] < offset or record['start'] + 33 > offset + len(contacts):
            raise ValueError('Selected window escapes its scene-contact interval')
        world = apply_scene_contacts(world, contacts, offset, self.rep.indices['foot_contacts'])
        with np.load(paths['geometry'], allow_pickle=False) as archive:
            scene = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
        return world, scene

    @torch.no_grad()
    def load(self, selection):
        from scripts.cache_full_terrain_primitives import rotate_scene_world

        cache_index, row_index = selection['cache_index'], selection['row_index']
        if (type(cache_index) is not int or not 0 <= cache_index < len(self.roots)
                or type(row_index) is not int
                or not 0 <= row_index < self.data.offsets[cache_index + 1] - self.data.offsets[cache_index]):
            raise ValueError('Selection cache/row index is outside the original cache')
        record = self.data.records[self.data.offsets[cache_index] + row_index]
        if record != selection['record'] or record['split'] not in ('train', 'validation'):
            raise ValueError('Selection record or split differs from the original cache')
        if record['quarter'] != 0:
            raise ValueError('The reactive pilot selects q0 only')
        kind = 'pfnn' if record['domain'] == 'pfnn' else 'mixed'
        root, manifest, sources = self._sources(cache_index, kind)
        row = sources.get(record['source_id'])
        if row is None or any(row[key] != record[key] for key in ('source_id', 'source_group', 'split')):
            raise ValueError('Source identity/group/split differs from cache')
        if kind == 'pfnn':
            expected = row['source_bvh_sha256']
        else:
            domain = 'bones' if row['origin'] == 'bones_seed' else 'parc' if row['origin'].startswith('parc') else None
            if record['domain'] != domain:
                raise ValueError('Mixed source domain differs from cache')
            expected = row['converted_shard_sha256']
        if record['source_sha256'] != expected:
            raise ValueError('Source SHA256 identity differs from cache')
        if kind == 'pfnn':
            world, scene = self._pfnn(root, row, record)
        else:
            artifacts = manifest['artifact_sha256']
            files = [(row['shard'], expected), (row['terrain_geometry_snapshot'], row['terrain_geometry_sha256'])]
            if 'source_contact_sidecar' in row:
                files.append((row['source_contact_sidecar'], row['source_contact_sidecar_sha256']))
            for relative, digest in files:
                if artifacts.get(relative) != digest:
                    raise ValueError('Mixed source artifact is not bound by its manifest')
                _checked_file(root, relative, digest)
            split = record['split']
            if split not in self.datasets:
                data = TerrainLearningDataset(root, self.rep, split, cache_bytes=0)
                indices = {entry['source_id']: i for i, entry in enumerate(data.records)}
                self.datasets[split] = data, indices
            data, indices = self.datasets[split]
            item = data.full_reference_item(indices[record['source_id']])
            if item['row'] != row:
                raise ValueError('Mixed dataset source record differs from its manifest')
            world, scene = item['world'], item['scene']
        start = record['start']
        if type(start) is not int or start < 0 or start + 33 > len(world):
            raise ValueError('Window escapes real source frames')
        world = world[None, start:start + 33].detach().clone()
        if world.shape != (1, 33, 366) or world.device.type != 'cpu' or not torch.isfinite(world).all():
            raise ValueError('Recovered world must be finite CPU [1, 33, 366]')
        scene = {key: torch.as_tensor(value).detach().clone() for key, value in scene.items()}
        if any(value.device.type != 'cpu' or not torch.isfinite(value).all() for value in scene.values()):
            raise ValueError('Source scene must contain finite CPU arrays')
        world, scene = rotate_scene_world(self.rep, world, scene, record['quarter'])
        cached = torch.from_numpy(np.array(_world_map(self.data, cache_index, self.maps)[row_index], copy=True))[None]
        if not torch.isfinite(cached).all():
            raise ValueError('Cached world must be finite')
        difference = (world - cached).abs()
        world_error, root_error = float(difference.max()), float(difference[..., :5].max())
        if not torch.isfinite(world).all() or world_error > WORLD_ATOL or root_error > WORLD_ATOL:
            raise ValueError(f'Recovered source differs from cached world/root: {world_error}/{root_error}')
        category, summary = _category(cached[0, :, :5].numpy())
        if category != selection['category']:
            raise ValueError('Selection source category differs from cached motion')
        return dict(world=world, scene=scene, record=dict(record), receipt=dict(
            cache_index=cache_index, row_index=row_index, category=category, root_summary=summary,
            world_max_abs_error=world_error, root_max_abs_error=root_error,
            source_manifest_sha256=_sha256(root / ('manifest.json' if kind == 'pfnn' else 'native_manifest.json')),
            source_sha256=expected))
