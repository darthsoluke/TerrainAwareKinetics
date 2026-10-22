from types import SimpleNamespace

import torch

from motionbricks.data.full_terrain_cache import BalancedPrimitiveBatches, PrimitiveDataset, PrimitiveWriter, FIELDS


def dataset():
    records = [{'domain': name, 'split': 'train'} for name in ('pfnn', 'parc', 'bones') for _ in range(40)]
    return SimpleNamespace(records=records, ids=list(range(len(records))))


def test_source_quota_and_resume_are_independent_of_future_prefetch():
    source = dataset()
    sampler = BalancedPrimitiveBatches(source, 16, 20, seed=123)
    expected = sampler.indices(5)
    for step in range(6, 15):
        sampler.indices(step)
    assert sampler.indices(5) == expected
    resumed = BalancedPrimitiveBatches(source, 16, 20, seed=123, start_step=5)
    assert next(iter(resumed)) == expected
    domains = [source.records[index]['domain'] for index in expected]
    assert domains.count('pfnn') == 8 and domains.count('parc') == 4 and domains.count('bones') == 4


def test_rank_partitions_do_not_duplicate_training_examples():
    source = dataset()
    first = BalancedPrimitiveBatches(source, 16, 20, rank=0, world_size=2)
    second = BalancedPrimitiveBatches(source, 16, 20, rank=1, world_size=2)
    assert not set(first.indices(0)) & set(second.indices(0))
    for domain in first.groups:
        assert set(first.groups[domain]) | set(second.groups[domain]) == {
            index for index, row in enumerate(source.records) if row['domain'] == domain}


def test_multi_part_memmaps_keep_global_sample_identity_and_split(tmp_path):
    for part in range(2):
        writer = PrimitiveWriter(tmp_path/str(part), chunk_size=1)
        for index in range(2):
            writer.append({key: torch.full((2,), part*10+index) for key in FIELDS},
                dict(domain='pfnn' if part == 0 else 'bones', split='train' if index else 'validation'))
        writer.finish()
    data = PrimitiveDataset([tmp_path/'0',tmp_path/'1'], 'train')
    assert len(data) == 2
    assert data[0]['sample_id'] == 1 and data[1]['sample_id'] == 3
    assert data[0]['world'].tolist() == [1,1]
    assert data[1]['world'].tolist() == [11,11]
