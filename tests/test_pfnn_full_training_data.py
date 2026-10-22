import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motionbricks.data import pfnn_full_training_data as data
from scripts.audit_pfnn_full_coverage import main as audit_main
from scripts.cache_full_terrain_primitives import rotate_scene_world


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


class CpuRep(torch.nn.Module):
    indices = {'foot_contacts': [362, 363, 364, 365]}

    def change_first_heading(self, world, heading, is_normalized, to_normalize):
        assert not is_normalized and not to_normalize
        result = world.clone()
        angle = heading - torch.atan2(world[:, 0, 4], world[:, 0, 3])
        c, s = torch.cos(angle), torch.sin(angle)
        rotation = torch.stack((c, s, -s, c), -1).reshape(-1, 2, 2)
        centered = world[..., [0, 2]] - world[:, :1, [0, 2]]
        result[..., [0, 2]] = torch.einsum('bij,btj->bti', rotation, centered)
        yaw = torch.atan2(world[..., 4], world[..., 3]) + angle[:, None]
        result[..., 3:5] = torch.stack((torch.cos(yaw), torch.sin(yaw)), -1)
        return result


def world_motion(frames=200, kind='forward', yaw=.63):
    world = np.zeros((frames, 366), dtype=np.float32)
    velocity = np.tile([np.sin(yaw), np.cos(yaw)], (frames - 1, 1))
    if kind == 'idle':
        velocity[:] = 0
    elif kind == 'start':
        velocity[:3] = 0
    elif kind == 'stop':
        velocity[3:] = 0
    elif kind == 'reverse':
        velocity *= -1
    elif kind == 'strafe':
        velocity[:] = [np.cos(yaw), -np.sin(yaw)]
    world[0, [0, 2]] = [17., -11.]
    world[1:, [0, 2]] = world[0, [0, 2]] + np.cumsum(velocity / 30, axis=0)
    world[:, 1] = 1.3
    world[:, 3:5] = [np.cos(yaw), np.sin(yaw)]
    world[:, 12] = np.arange(frames) / 100
    return world


@pytest.fixture
def full_cache(tmp_path):
    root, cache = tmp_path / 'pfnn', tmp_path / 'cache'
    root.mkdir()
    (cache / 'data').mkdir(parents=True)
    rep, originals, records, arrays, pairs = CpuRep(), {}, [], [], []
    scene = dict(heightfield_y_m=torch.arange(12, dtype=torch.float32).reshape(3, 4),
                 heightfield_valid=torch.tensor([[True, True, False, True]] * 3),
                 min_point_world_xyz_m=torch.tensor([90., 7., -70.]), dx_m=torch.tensor(.25))
    for number, (name, split) in enumerate((('train', 'train'), ('valid', 'validation'))):
        world = world_motion()
        np.savez(root / (name + '.npz'), global_features=world, fps=np.float32(30), world_contract=data.WORLD_CONTRACT)
        originals[name] = world
        starts_list = ([8, 16, 24, 32, 40, 48], [81, 89, 97, 105, 113, 121]) if split == 'train' else ([8],)
        for pair_index, starts in enumerate(starts_list):
            pair_name = f'{name}-{pair_index}'
            offset, count = 5, 180
            contacts = ((np.arange(count * 4).reshape(count, 4) + pair_index) % 3 == 0).astype(np.float32)
            np.savez(root / (pair_name + '-contact.npz'), contacts=contacts, source_frame_offset=np.int64(offset),
                     source_frame_indices=np.arange(offset, offset + count), fps=np.float32(30))
            np.savez(root / (pair_name + '-scene.npz'), **{key: value.numpy() for key, value in scene.items()})
            row = dict(source_id=f'pfnn:{name}:terrain:{pair_index}', source_group='pfnn:' + name, split=split,
                source_bvh_sha256=str(number + 1) * 64, motion_shard=name + '.npz',
                motion_sha256=data.sha256(root / (name + '.npz')), geometry_shard=pair_name + '-scene.npz',
                geometry_sha256=data.sha256(root / (pair_name + '-scene.npz')), contact_shard=pair_name + '-contact.npz',
                contact_sha256=data.sha256(root / (pair_name + '-contact.npz')), contact_source_offset=offset,
                contact_source_frames=count, world_contract=data.WORLD_CONTRACT, eligible_window_starts=starts)
            pairs.append(row)
            for start in starts:
                original = torch.from_numpy(world[start:start + 33].copy())
                original[:, rep.indices['foot_contacts']] = torch.from_numpy(contacts[start - offset:start - offset + 33])
                for quarter in range(4):
                    rotated, _ = rotate_scene_world(rep, original[None], scene, quarter)
                    arrays.append(rotated[0].numpy())
                    records.append(dict(domain='pfnn', source_id=row['source_id'], source_group=row['source_group'],
                        source_sha256=row['source_bvh_sha256'], split=split, start=start, quarter=quarter))
    manifest = dict(schema='motionbricks.pfnn_static_pairs.v2', world_contract=data.WORLD_CONTRACT, pilot=False,
        records=pairs, full_motion_clips_considered=3, admitted_windows=len(records) // 4,
        source_audit=[dict(source_id='pfnn:' + name, world_encoding=dict(actual_frames=200))
                      for name in ('train', 'valid', 'zero')])
    np.save(cache / 'data/world.npy', np.stack(arrays))
    complete = dict(schema='motionbricks.full_terrain_primitives.v1', pilot=False,
        considered_base_windows=len(records) // 4, rejected_base_windows=0,
        data=dict(examples=len(records), fields=dict(world=dict(shape=list(np.stack(arrays).shape), dtype='float32'))))

    def bind():
        write_json(root / 'manifest.json', manifest)
        write_json(root / 'complete.json', dict(manifest_sha256=data.sha256(root / 'manifest.json')))
        complete['source_manifests'] = dict(pfnn=data.sha256(root / 'manifest.json'))
        write_json(cache / 'complete.json', complete)
        (cache / 'data/records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
    bind()
    return SimpleNamespace(root=root, cache=cache, rep=rep, originals=originals, scene=scene, pairs=pairs,
                           manifest=manifest, complete=complete, records=records, bind=bind)


@pytest.mark.parametrize('quarter', range(4))
def test_all_quarters_use_source_first_sample_and_identical_scene_chart(full_cache, quarter):
    f = full_cache
    dataset = data.PFNNFrameDataset(f.cache, f.root, f.rep, 'train')
    item = dataset[quarter]
    row = f.pairs[0]
    source = torch.from_numpy(f.originals['train'][8:41].copy())
    with np.load(f.root / row['contact_shard']) as archive:
        source[:, f.rep.indices['foot_contacts']] = torch.from_numpy(archive['contacts'][3:36])
    expected_world, expected_scene = rotate_scene_world(f.rep, source[None], f.scene, quarter)
    torch.testing.assert_close(item['world'], expected_world[0], atol=0, rtol=0)
    assert item['world'].shape == (33, 366)
    assert item['record']['quarter'] == quarter
    for key in expected_scene:
        torch.testing.assert_close(item['scene'][key], expected_scene[key], atol=2e-5, rtol=0)
    # A scene rotated about zero alone would have an entirely different origin.
    assert not torch.equal(item['scene']['min_point_world_xyz_m'], f.scene['min_point_world_xyz_m'])
    commands = item['command_sequence']
    assert commands['command'].shape == (26, 4)
    assert commands['source_frames'].tolist() == list(range(11, 37))
    assert dataset.records[quarter]['cache_row_index'] == quarter
    assert dataset.records[quarter]['category'] == 'forward'


def test_split_local_indices_and_raw_world_read_only_mapping(full_cache):
    f = full_cache
    dataset = data.PFNNFrameDataset(f.cache, f.root, f.rep, 'validation')
    assert len(dataset) == 4 and dataset.ids == [48, 49, 50, 51]
    assert isinstance(dataset.index.world, np.memmap) and not dataset.index.world.flags.writeable
    item = dataset[0]
    item['world'][0, 12] = -999
    assert float(dataset[0]['world'][0, 12]) != -999
    item['scene']['heightfield_y_m'].zero_()
    assert dataset[0]['scene']['heightfield_y_m'].sum() > 0


@pytest.mark.parametrize('change,match', [
    ('missing', 'four-quarter'), ('duplicate', 'Duplicate augmented'),
    ('extra', 'four-quarter'), ('split', 'identity/group/split'),
    ('sha', 'identity/group/split'), ('domain', 'domain/source'), ('quarter', 'start/quarter'),
])
def test_cache_identity_and_complete_quarters_fail(full_cache, change, match):
    f = full_cache
    if change == 'missing':
        f.records.pop()
    elif change == 'duplicate':
        f.records.append(dict(f.records[0]))
    elif change == 'extra':
        f.records[0]['start'] = 9
    elif change == 'split':
        f.records[0]['split'] = 'test'
    elif change == 'sha':
        f.records[0]['source_sha256'] = 'a' * 64
    elif change == 'domain':
        f.records[0]['domain'] = 'bones'
    else:
        f.records[0]['quarter'] = True
    f.bind()
    with pytest.raises(ValueError, match=match):
        data.PFNNCacheIndex(f.cache, f.root)


@pytest.mark.parametrize('change,match', [
    ('group', 'crosses splits'), ('manifest_hash', 'manifest SHA256'), ('pilot', 'non-pilot'),
    ('count', 'example count'), ('shape', 'array header'), ('admission', 'admission count'),
    ('clock', 'contact clock'), ('eligible_duplicate', 'eligible start'),
])
def test_manifest_and_cache_receipt_validation(full_cache, change, match):
    f = full_cache
    if change == 'group':
        f.pairs[-1]['source_group'] = 'pfnn:train'
    elif change == 'pilot':
        f.manifest['pilot'] = True
    elif change == 'count':
        f.complete['data']['examples'] += 1
    elif change == 'shape':
        f.complete['data']['fields']['world']['shape'][1] = 32
    elif change == 'admission':
        f.complete['considered_base_windows'] += 1
    elif change == 'clock':
        f.pairs[0]['contact_source_offset'] = 10
    elif change == 'eligible_duplicate':
        f.pairs[0]['eligible_window_starts'].append(8)
    f.bind()
    if change == 'manifest_hash':
        f.complete['source_manifests']['pfnn'] = 'e' * 64
        write_json(f.cache / 'complete.json', f.complete)
    with pytest.raises(ValueError, match=match):
        data.PFNNCacheIndex(f.cache, f.root)


@pytest.mark.parametrize('artifact', ['motion', 'geometry', 'contact'])
def test_source_hash_checked_before_use(full_cache, artifact):
    f = full_cache
    path = f.root / f.pairs[0][artifact + '_shard']
    with path.open('ab') as stream:
        stream.write(b'changed')
    dataset = data.PFNNFrameDataset(f.cache, f.root, f.rep, 'train')
    with pytest.raises(ValueError, match='Source SHA256'):
        dataset[0]


def test_contact_clock_mismatch_with_rebound_hash(full_cache):
    f = full_cache
    row = f.pairs[0]
    path = f.root / row['contact_shard']
    with np.load(path) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    values['source_frame_indices'][0] += 1
    np.savez(path, **values)
    row['contact_sha256'] = data.sha256(path)
    f.bind()
    with pytest.raises(ValueError, match='contact clock'):
        data.PFNNFrameDataset(f.cache, f.root, f.rep, 'train')[0]


@pytest.mark.parametrize('column,match', [(0, 'first sample'), (3, 'quarter heading'), (363, 'first sample')])
def test_cached_chart_or_contact_corruption_rejected(full_cache, column, match):
    f = full_cache
    array = np.load(f.cache / 'data/world.npy')
    array[0, 0, column] += .25
    if column == 3:
        array[0, 0, 3:5] = [np.cos(1.), np.sin(1.)]
    np.save(f.cache / 'data/world.npy', array)
    with pytest.raises(ValueError, match=match):
        data.PFNNFrameDataset(f.cache, f.root, f.rep, 'train')[0]


def test_midwindow_contact_and_root_corruption_rejected(full_cache):
    f = full_cache
    array = np.load(f.cache / 'data/world.npy')
    array[0, 12, 363] = 1 - array[0, 12, 363]
    np.save(f.cache / 'data/world.npy', array)
    with pytest.raises(ValueError, match='contacts differ'):
        data.PFNNFrameDataset(f.cache, f.root, f.rep, 'train')[0]


@pytest.mark.parametrize('column', [0, 12, 100])
def test_entire_cached_world_is_bound_to_source_not_only_first_frame(full_cache, column):
    f = full_cache
    array = np.load(f.cache / 'data/world.npy')
    array[0, 12, column] += .25
    np.save(f.cache / 'data/world.npy', array)
    with pytest.raises(ValueError, match='Cached window differs'):
        data.PFNNFrameDataset(f.cache, f.root, f.rep, 'train')[0]


def test_complete_motion_source_is_reused_but_lru_is_capped(full_cache, monkeypatch):
    f = full_cache
    original = data._load_motion
    calls = []

    def counted(root, row):
        calls.append(row['motion_shard'])
        return original(root, row)
    monkeypatch.setattr(data, '_load_motion', counted)
    dataset = data.PFNNFrameDataset(f.cache, f.root, f.rep, 'train', scene_cache_bytes=0)
    dataset[0]
    dataset[4]
    dataset[24]
    assert calls == ['train.npz']
    assert dataset.source_cache.bytes == f.originals['train'].nbytes
    assert dataset.scene_cache.bytes == 0
    no_cache = data.PFNNFrameDataset(f.cache, f.root, f.rep, 'train', source_cache_bytes=1)
    no_cache[0]
    no_cache[1]
    assert len(calls) == 3 and no_cache.source_cache.bytes == 0


def test_audit_unions_source_time_but_keeps_pair_scene_spans_separate(full_cache):
    f = full_cache
    report = data.audit_full_coverage(f.cache, f.root)
    assert report['passed'] and report['phase'] == 'P0'
    assert report['optimizer_updates'] == report['training_exposures'] == 0
    assert report['original_files'] == 3 and report['zero_admission_files'] == ['pfnn:zero']
    assert report['augmented_rows'] == 52 and report['admitted_pair_windows'] == 13
    assert report['splits']['train']['cache_rows'] == 48
    file = next(row for row in report['per_file'] if row['source_id'] == 'pfnn:train')
    assert file['covered_intervals'] == [[8, 154]]
    assert file['covered_frames'] == 146 and file['max_same_scene_span_frames'] == 73
    assert file['span_counts']['60'] == dict(qualifying_spans=2, valid_starts=28)
    assert file['span_counts']['120'] == dict(qualifying_spans=0, valid_starts=0)
    assert report['pairs'][0]['spans'] == [[8, 81]]
    assert report['pairs'][1]['spans'] == [[81, 154]]
    assert report['pairs'][0]['contact_clock'] == [5, 185]
    zero = next(row for row in report['per_file'] if row['source_id'] == 'pfnn:zero')
    assert zero['covered_frames'] == zero['admitted_windows'] == 0
    assert zero['covered_intervals'] == []
    json.dumps(report)


def test_inventory_explicit_binding_and_zero_files(full_cache):
    f = full_cache
    inventory = f.root / 'inventory.json'
    clips = [dict(filename=f'data/animations/{name}.bvh', sha256=str(i + 1) * 64, role='motion',
                  frames=200, fps=30., source_recording_group=name, mirrored=False)
             for i, name in enumerate(('train', 'valid', 'zero'))]
    write_json(inventory, dict(clips=clips + [dict(role='rest')]))
    f.manifest['source_inventory_sha256'] = data.sha256(inventory)
    f.bind()
    report = data.audit_full_coverage(f.cache, f.root, inventory)
    assert len(report['per_file']) == 3
    assert report['inventory_sha256'] == data.sha256(inventory)
    with inventory.open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='inventory SHA256'):
        data.audit_full_coverage(f.cache, f.root, inventory)


def test_audit_requires_zero_file_inventory_information(full_cache):
    f = full_cache
    f.manifest.pop('source_audit')
    f.bind()
    with pytest.raises(ValueError, match='zero-admission files'):
        data.audit_full_coverage(f.cache, f.root)


def test_cli_outputs_all_files_and_full_pair_details(full_cache, tmp_path):
    output = tmp_path / 'audit'
    audit_main(SimpleNamespace(cache=full_cache.cache, pfnn=full_cache.root, inventory=None,
                               output=output, span_frames=[32, 60, 120, 240, 480]))
    report = json.loads((output / 'audit.json').read_text())
    files = json.loads((output / report['per_file_coverage']).read_text())
    assert len(files) == 3 and len(report['pairs']) == 3
    assert report['span_lengths_frames'] == [32, 60, 120, 240, 480]
    assert not (output / 'failure.json').exists()
    with pytest.raises(FileExistsError):
        audit_main(SimpleNamespace(output=output))


def test_unobserved_transition_between_abutting_windows_is_not_a_long_span():
    assert data.merge_intervals([(0, 33), (33, 66)]) == [[0, 66]]
    assert data._continuous_spans([0, 33]) == [[0, 33], [33, 66]]
    assert data._continuous_spans([0, 32]) == [[0, 65]]


def category_records():
    return [dict(domain='pfnn', source_id=category, source_group=category, split='train',
                 quarter=quarter, start=0, category=category)
            for category in data.CATEGORIES for quarter in range(4)]


def test_sampler_complete_primary_then_balanced_actual_extras():
    records = category_records()
    records.append(dict(records[0], split='test'))
    sampler = data.FullEpochSampler(records, seed=123)
    assert len(sampler) == 30
    assert sorted(sampler.primary_indices) == list(range(24))
    assert len(sampler.extra_indices) == 6 and 24 not in list(sampler)
    assert {records[i]['category'] for i in sampler.extra_indices} == set(data.CATEGORIES)
    assert list(sampler) == sampler.phases()['primary'] + sampler.phases()['extra']
    receipt = sampler.coverage_receipt()
    assert receipt['duplicate_primary_count'] == 0 and receipt['missing_primary_ids'] == []
    assert receipt['padding'] == receipt['dropped'] == 0
    assert data.FullEpochSampler(records, seed=123).phases() == sampler.phases()
    old = sampler.phases()
    sampler.set_epoch(1)
    assert sampler.phases() != old and sorted(sampler.primary_indices) == list(range(24))


def test_sampler_no_tiny_bank_at_real_full_epoch_size():
    records = [dict(domain='pfnn', split='train', category=data.CATEGORIES[i % 6]) for i in range(48232)]
    sampler = data.FullEpochSampler(records, seed=7)
    assert len(sampler.global_primary_indices) == 48232
    assert len(set(sampler.global_primary_indices)) == 48232
    assert len(sampler.global_extra_indices) == 12058
    assert max(sampler.extra_category_counts.values()) - min(sampler.extra_category_counts.values()) == 1


@pytest.mark.parametrize('world_size', [2, 5, 31])
def test_rank_shards_are_uneven_without_padding_or_dropping(world_size):
    records = category_records()
    ranks = [data.FullEpochSampler(records, seed=41, rank=rank, world_size=world_size, extras=13)
             for rank in range(world_size)]
    primary = [i for rank in ranks for i in rank.primary_indices]
    assert sorted(primary) == list(range(24)) and len(set(primary)) == len(primary)
    assert sum(len(rank.extra_indices) for rank in ranks) == 13
    assert max(map(len, ranks)) - min(map(len, ranks)) <= 2
    for rank, sampler in enumerate(ranks):
        assert sampler.primary_indices == ranks[0].global_primary_indices[rank::world_size]
        assert sampler.extra_indices == ranks[0].global_extra_indices[rank::world_size]


def test_missing_categories_stay_explicit_without_fabricated_samples():
    records = [row for row in category_records() if row['category'] == 'forward']
    sampler = data.FullEpochSampler(records, extras=20)
    assert sampler.coverage_receipt()['missing_categories'] == list(data.CATEGORIES[:-1])
    assert sampler.coverage_receipt()['extra_categories']['forward'] == 20
    assert set(sampler.extra_indices) == {0, 1, 2, 3}


@pytest.mark.parametrize('kwargs', [dict(split='test'), dict(rank=2, world_size=2), dict(world_size=0),
                                    dict(extras=-1), dict(seed=True)])
def test_bad_sampler_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        data.FullEpochSampler(category_records(), **kwargs)


@pytest.mark.parametrize('kind', ['forward', 'reverse', 'strafe', 'idle', 'start', 'stop'])
def test_intent_targets_real_four_interval_average_and_original_clock(kind):
    world = world_motion(frames=33, kind=kind)
    labels = data.input_intent_labels(world, 123)
    velocity = (world[7, [0, 2]] - world[3, [0, 2]]) * (30 / 4)
    np.testing.assert_allclose(labels['command'][0, :2], velocity, atol=0, rtol=0)
    np.testing.assert_allclose(labels['command'][0, 2:], world[7, 3:5], atol=0, rtol=0)
    assert labels['source_frames'][0] == 126 and labels['source_frames'][-1] == 151
    assert bool(labels['stop_event'][0]) == (kind == 'stop')
    assert not labels['stop_event'].any() if kind == 'idle' else True


def test_intent_label_has_no_padded_tail_and_no_zero_velocity_from_returning_path():
    with pytest.raises(ValueError, match='real source frames'):
        data.input_intent_labels(world_motion(7))
    labels = data.input_intent_labels(world_motion(8))
    assert labels['command'].shape == (1, 4) and labels['source_frames'].tolist() == [3]
    world = world_motion(8)
    world[4:8, 2] = world[3, 2] + np.array([1, 0, 1, 0]) / 30
    assert not data.input_intent_labels(world)['stop_event'][0]


def test_stopping_command_is_zero_while_measured_source_still_brakes():
    world = world_motion(8)
    # Three genuinely moving intervals, followed by four still-moving but
    # decelerating intervals. All source positions remain unchanged by labeling.
    speeds = np.array([.6, .5, .4, .12, .10, .08, .04])
    world[:, 0] = 17.
    world[1:, 2] = world[0, 2] + np.cumsum(speeds / 30)
    before = world.copy()
    labels = data.input_intent_labels(world, 100)
    assert labels['stop_event'].tolist() == [True]
    assert torch.equal(labels['command'][0, :2], torch.zeros(2))
    assert labels['measured_next4_velocity_mps'][0, 1] > .08
    assert labels['next4_speed_mps'][0] > .08
    np.testing.assert_array_equal(world, before)
    np.testing.assert_array_equal(labels['command'][0, 2:], world[7, 3:5])


@pytest.mark.parametrize('speed', [.03, .08, .12])
def test_sustained_slow_movement_keeps_measured_command(speed):
    world = world_motion(12)
    world[:, 0] = 17.
    world[:, 2] = -11. + np.arange(12) * speed / 30
    labels = data.input_intent_labels(world)
    assert not labels['stop_event'].any()
    torch.testing.assert_close(labels['command'][:, :2], labels['measured_next4_velocity_mps'], atol=0, rtol=0)
    assert torch.all(labels['command'][:, 1] > 0)
