import hashlib
import json

import numpy as np
import pytest
import torch

from motionbricks.data import reactive_training_data as data


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def motion(category='forward', frames=33, yaw=0.):
    world = np.zeros((frames, 366), dtype=np.float32)
    velocity = np.tile([np.sin(yaw), np.cos(yaw)], (frames - 1, 1))
    if category == 'idle':
        velocity[:] = 0
    elif category == 'start':
        velocity[:3] = 0
    elif category == 'stop':
        velocity[-4:] = 0
    elif category == 'reverse':
        velocity *= -1
    elif category == 'strafe':
        velocity[:] = [np.cos(yaw), -np.sin(yaw)]
    world[0, [0, 2]] = [17., -11.]
    world[1:, [0, 2]] = world[0, [0, 2]] + np.cumsum(velocity / 30, axis=0)
    world[:, 1] = 2.
    world[:, 3:5] = [np.cos(yaw), np.sin(yaw)]
    return world


def record(domain='pfnn', split='train', source_id='source', source_group='group', **changes):
    return dict(domain=domain, split=split, source_id=source_id, source_group=source_group,
                start=changes.pop('start', 0), quarter=changes.pop('quarter', 0),
                source_sha256=changes.pop('source_sha256', 'b' * 64), **changes)


def cache(root, records, worlds):
    folder = root / 'data'
    folder.mkdir(parents=True)
    (folder / 'records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
    np.save(folder / 'world.npy', np.stack(worlds))
    return root


def test_domain_quotas_original_rows_split_and_q0(tmp_path):
    paths = []
    for index, domain in enumerate(data.TRAIN_QUOTAS):
        rows, worlds = [], []
        for split in ('test', 'train', 'validation'):
            for number in range(50):
                for quarter in (0, 1):
                    rows.append(record(domain, split, f'{domain}:{split}:{number}',
                                       f'{domain}:{split}:group:{number}', quarter=quarter))
                    worlds.append(motion(data.CATEGORIES[number % 6]))
        paths.append(cache(tmp_path / str(index), rows, worlds))
    for split, quotas in (('train', data.TRAIN_QUOTAS), ('validation', data.VALIDATION_QUOTAS)):
        selected = data.select_records(paths, split, quotas)
        receipt = data.selection_receipt(selected)
        assert type(selected) is list
        assert receipt['count'] == sum(quotas.values())
        assert receipt['splits'] == {split: sum(quotas.values())}
        for index, domain in enumerate(quotas):
            assert receipt['domains'][domain]['count'] == quotas[domain]
            assert receipt['domains'][domain]['source_groups'] == quotas[domain]
            for row in [entry for entry in selected if entry['record']['domain'] == domain]:
                assert row['cache_index'] == index
                assert row['row_index'] % 2 == 0
                assert row['record'] == rows_from(paths[index])[row['row_index']]
                assert row['record']['quarter'] == 0


def rows_from(path):
    return [json.loads(line) for line in (path / 'data/records.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('yaw', [0., np.pi / 2, -1.3])
def test_all_motion_strata_and_soma_forward_axis(tmp_path, yaw):
    categories = data.CATEGORIES
    path = cache(tmp_path, [record(source_id=name, source_group=name) for name in categories],
                 [motion(name, yaw=yaw) for name in categories])
    selected = data.select_records([path], 'train', {'pfnn': 6})
    assert {row['record']['source_id']: row['category'] for row in selected} == {name: name for name in categories}
    assert all(row['root_summary']['next4_speed_mps'] > .9 for row in selected if row['category'] == 'forward')


def test_determinism_seed_and_global_rng_independence(tmp_path):
    path = cache(tmp_path, [record(source_id=str(i), source_group=str(i)) for i in range(60)],
                 [motion() for _ in range(60)])
    first = data.select_records([path], 'train', {'pfnn': 8})
    np.random.seed(4)
    np.random.random(100)
    assert first == data.select_records([path], 'train', {'pfnn': 8})
    assert first != data.select_records([path], 'train', {'pfnn': 8}, seed=5)


def test_distinct_groups_before_repeats_and_no_fabricated_categories(tmp_path):
    rows = [record(source_id=str(i), source_group='large') for i in range(40)]
    rows += [record(source_id='small', source_group='small')]
    path = cache(tmp_path, rows, [motion() for _ in rows])
    selected = data.select_records([path], 'train', {'pfnn': 4})
    assert len({row['record']['source_group'] for row in selected[:2]}) == 2
    assert len({row['row_index'] for row in selected}) == 4
    receipt = data.selection_receipt(selected)['domains']['pfnn']
    assert receipt['categories'] == dict.fromkeys(data.CATEGORIES[:-1], 0) | {'forward': 4}


def test_candidate_reads_are_bounded_world_only_and_never_test(tmp_path, monkeypatch):
    rows = [record(source_id=str(i), source_group=str(i)) for i in range(100)]
    rows += [record(split='test', source_id='test', source_group='test')]
    worlds = [motion() for _ in rows]
    worlds[-1][:] = np.nan
    path = cache(tmp_path, rows, worlds)
    original = np.load
    opened, accessed = [], []

    class ReadCounter:
        def __init__(self, value):
            self.value, self.shape, self.dtype = value, value.shape, value.dtype

        def __getitem__(self, key):
            accessed.append(key[0])
            return self.value[key]

    def counted(path, **kwargs):
        opened.append(path.name)
        assert kwargs == dict(mmap_mode='r', allow_pickle=False)
        return ReadCounter(original(path, **kwargs))

    monkeypatch.setattr(data.np, 'load', counted)
    selected = data.select_records([path], 'train', {'pfnn': 2})
    assert len(selected) == 2
    assert opened == ['world.npy']
    assert len(accessed) == 2 * data.CANDIDATES_PER_QUOTA
    assert 100 not in accessed


@pytest.mark.parametrize('field,value,message', [
    ('source_group', None, 'source_group'), ('source_id', '', 'source_id'),
    ('quarter', -1, 'quarter'), ('start', -1, 'source start'), ('domain', 'unknown', 'domain'),
])
def test_invalid_metadata_fails_without_substituting_groups(tmp_path, field, value, message):
    row = record()
    row[field] = value
    path = cache(tmp_path, [row], [motion()])
    with pytest.raises(ValueError, match=message):
        data.select_records([path], 'train', {'pfnn': 1})


def test_group_leakage_is_rejected_and_test_split_cannot_be_requested(tmp_path):
    path = cache(tmp_path, [record(), record(split='test', source_id='other')], [motion(), motion()])
    with pytest.raises(ValueError, match='crosses splits'):
        data.select_records([path], 'train', {'pfnn': 1})
    with pytest.raises(ValueError, match='train or validation'):
        data.select_records([path], 'test', {'pfnn': 1})


@pytest.mark.parametrize('quotas', [{'pfnn': -1}, {'pfnn': True}, {'pfnn': 1.5}, {'other': 1}])
def test_invalid_quotas(tmp_path, quotas):
    with pytest.raises(ValueError, match='Quotas'):
        data.select_records([tmp_path], 'train', quotas)


def test_insufficient_records_do_not_repeat_or_substitute_rotations(tmp_path):
    path = cache(tmp_path, [record(), record(quarter=1)], [motion(), motion()])
    with pytest.raises(ValueError, match='Insufficient'):
        data.select_records([path], 'train', {'pfnn': 2})


class CpuRep(torch.nn.Module):
    indices = {'foot_contacts': [362, 363, 364, 365]}

    def change_first_heading(self, world, heading, is_normalized, to_normalize):
        assert not is_normalized and not to_normalize
        torch.testing.assert_close(heading, torch.atan2(world[:, 0, 4], world[:, 0, 3]))
        changed = world.clone()
        changed[..., [0, 2]] -= world[:, :1, [0, 2]]
        return changed


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    def make(domain='pfnn', split='train'):
        source = tmp_path / domain
        source.mkdir()
        world = motion(frames=65)
        world[:, 12] = np.arange(65)
        scene = dict(heightfield_y_m=np.arange(12, dtype=np.float32).reshape(3, 4),
            min_point_world_xyz_m=np.array([90., 7., -70.], dtype=np.float32), dx_m=np.float32(.25))
        start = 16
        np.savez(source / 'motion.npz', global_features=world, fps=np.float32(30), world_contract=data.WORLD_CONTRACT)
        np.savez(source / 'geometry.npz', **scene)
        cached = world[start:start + 33].copy()
        cached[:, [0, 2]] -= cached[0, [0, 2]]
        base = dict(source_id='id', source_group='real-group', split=split)
        calls = []
        if domain == 'pfnn':
            offset, count = 5, 60
            contacts = (np.arange(count * 4).reshape(count, 4) % 3 == 0).astype(np.float32)
            cached[:, 362:366] = contacts[start - offset:start - offset + 33]
            np.savez(source / 'contacts.npz', contacts=contacts, fps=np.float32(30),
                source_frame_offset=offset, source_frame_indices=np.arange(offset, offset + count))
            row = dict(base, motion_shard='motion.npz', motion_sha256=digest(source / 'motion.npz'),
                geometry_shard='geometry.npz', geometry_sha256=digest(source / 'geometry.npz'),
                contact_shard='contacts.npz', contact_sha256=digest(source / 'contacts.npz'),
                contact_source_offset=offset, contact_source_frames=count, world_contract=data.WORLD_CONTRACT,
                eligible_window_starts=[start], source_bvh_sha256='b' * 64)
            manifest = dict(schema='motionbricks.pfnn_static_pairs.v2', world_contract=data.WORLD_CONTRACT, records=[row])
            manifest_name = 'manifest.json'
            source_hash = row['source_bvh_sha256']
        else:
            row = dict(base, shard='motion.npz', converted_shard_sha256=digest(source / 'motion.npz'),
                terrain_geometry_snapshot='geometry.npz', terrain_geometry_sha256=digest(source / 'geometry.npz'),
                origin='bones_seed' if domain == 'bones' else 'parc_official', frames_30hz=65)
            write_json(source / 'accepted.json', [row])
            manifest = dict(artifact_sha256={name: digest(source / name)
                for name in ('motion.npz', 'geometry.npz', 'accepted.json')})
            manifest_name = 'native_manifest.json'
            source_hash = row['converted_shard_sha256']

            class MixedDataset:
                def __init__(self, directory, rep, requested_split, *, cache_bytes):
                    assert directory == source and requested_split == split and cache_bytes == 0
                    # Cache row index is zero, whereas the source_id maps to one.
                    self.records = [dict(row, source_id='other'), row]

                def full_reference_item(self, index):
                    calls.append(index)
                    assert index == 1
                    return dict(world=torch.from_numpy(world.copy()),
                        scene={key: torch.as_tensor(value).clone() for key, value in scene.items()}, row=row)

            monkeypatch.setattr(data, 'TerrainLearningDataset', MixedDataset)
        write_json(source / manifest_name, manifest)
        entry = record(domain, split, 'id', 'real-group', start=start, source_sha256=source_hash)
        path = cache(tmp_path / ('cache-' + domain), [entry], [cached])

        def rebind():
            write_json(path / 'complete.json', dict(schema='motionbricks.full_terrain_primitives.v1',
                source_manifests={'pfnn' if domain == 'pfnn' else 'mixed': digest(source / manifest_name)}))

        rebind()
        selected = data.select_records([path], split, {domain: 1})[0]

        def bank():
            return data.ReactiveSourceBank([path], source, source, CpuRep())

        return dict(source=source, path=path, row=row, manifest=manifest, manifest_name=manifest_name,
            selection=selected, bank=bank, cached=cached, world=world, scene=scene,
            rebind=rebind, calls=calls, start=start)
    return make


@pytest.mark.parametrize('domain', ['pfnn', 'parc', 'bones'])
@pytest.mark.parametrize('split', ['train', 'validation'])
def test_exact_source_recovery_and_original_chart(source_fixture, domain, split):
    f = source_fixture(domain, split)
    bank = f['bank']()
    item = bank.load(f['selection'])
    assert item['world'].shape == (1, 33, 366)
    torch.testing.assert_close(item['world'][0], torch.from_numpy(f['cached']), rtol=0, atol=0)
    expected_origin = f['scene']['min_point_world_xyz_m'].copy()
    expected_origin[[0, 2]] -= f['world'][f['start'], [0, 2]]
    torch.testing.assert_close(item['scene']['min_point_world_xyz_m'], torch.from_numpy(expected_origin), rtol=0, atol=0)
    torch.testing.assert_close(item['scene']['heightfield_y_m'], torch.from_numpy(f['scene']['heightfield_y_m']), rtol=0, atol=0)
    assert item['scene']['dx_m'].item() == .25
    assert item['record'] == f['selection']['record']
    assert all(value.device.type == 'cpu' for value in [item['world'], *item['scene'].values()])
    assert item['receipt']['world_max_abs_error'] == item['receipt']['root_max_abs_error'] == 0
    assert len(bank.sources) == 1
    if domain != 'pfnn':
        assert f['calls'] == [1]


@pytest.mark.parametrize('which', ['motion.npz', 'geometry.npz', 'contacts.npz'])
def test_pfnn_source_hash_changes_are_rejected(source_fixture, which):
    f = source_fixture()
    with (f['source'] / which).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        f['bank']().load(f['selection'])


@pytest.mark.parametrize('field,value', [('source_group', 'other'), ('split', 'validation'),
    ('source_bvh_sha256', 'c' * 64), ('world_contract', 'old'), ('eligible_window_starts', [])])
def test_pfnn_original_identity_contract_and_admission(source_fixture, field, value):
    f = source_fixture()
    f['row'][field] = value
    write_json(f['source'] / f['manifest_name'], f['manifest'])
    f['rebind']()
    with pytest.raises(ValueError):
        f['bank']().load(f['selection'])


def test_cache_manifest_binding_rejects_wrong_source_dataset(source_fixture):
    f = source_fixture()
    f['manifest']['note'] = 'changed'
    write_json(f['source'] / f['manifest_name'], f['manifest'])
    with pytest.raises(ValueError, match='manifest SHA256'):
        f['bank']().load(f['selection'])


def test_pfnn_sidecar_clock_checked_even_with_correct_file_hash(source_fixture):
    f = source_fixture()
    path = f['source'] / 'contacts.npz'
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays['source_frame_indices'][5] += 1
    np.savez(path, **arrays)
    f['row']['contact_sha256'] = digest(path)
    write_json(f['source'] / f['manifest_name'], f['manifest'])
    f['rebind']()
    with pytest.raises(ValueError, match='source clock'):
        f['bank']().load(f['selection'])


@pytest.mark.parametrize('column,delta,passes', [(0, .00004, False), (40, .00004, False), (40, .00002, True), (40, np.nan, False)])
def test_exact_cache_comparison_includes_root_and_body(source_fixture, column, delta, passes):
    f = source_fixture()
    cached = f['cached'].copy()
    cached[10, column] += delta
    np.save(f['path'] / 'data/world.npy', cached[None])
    if passes:
        assert f['bank']().load(f['selection'])['receipt']['world_max_abs_error'] <= data.WORLD_ATOL
    else:
        with pytest.raises(ValueError, match='cached|Cached'):
            f['bank']().load(f['selection'])


def test_source_hash_remains_checked_after_source_index_is_cached(source_fixture):
    f = source_fixture('parc')
    bank = f['bank']()
    bank.load(f['selection'])
    with (f['source'] / 'motion.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        bank.load(f['selection'])


@pytest.mark.parametrize('change', ['row_index', 'record', 'category'])
def test_selection_cannot_change_cached_identity(source_fixture, change):
    f = source_fixture()
    selection = dict(f['selection'])
    if change == 'row_index':
        selection[change] = -1
    elif change == 'record':
        selection[change] = dict(selection[change], source_group='forged')
    else:
        selection[change] = 'idle'
    with pytest.raises(ValueError):
        f['bank']().load(selection)
