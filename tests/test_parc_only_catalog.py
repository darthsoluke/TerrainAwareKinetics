import copy

import numpy as np

import pytest

from scripts.prepare_parc_only_catalog import group_splits, bind_geometry_components,generated_recording_identity


def test_recording_identity_distinguishes_same_candidate_name_across_batches():
    first=generated_recording_identity('a'*64,'long_raw_c005')
    second=generated_recording_identity('b'*64,'long_raw_c005')
    assert first!=second
    assert first==generated_recording_identity('a'*64,'long_raw_c005')


def row(group, family='rerun_april272025_iter_1_boxes', origin='parc_official'):
    return {'source_group': group, 'origin': origin,
            'official_member': 'dec_release/' + family + '/sample.pkl', 'disposition': 'admitted'}


def test_split_uses_whole_groups_and_covers_each_family():
    rows = [row(f'{family}:{number}', family) for family in ('family-a', 'family-b')
            for number in range(32)]
    rows += [copy.deepcopy(item) for item in rows[:5]]
    rows += [row('generated:one', origin='parc_generated')]
    owners = group_splits(rows)
    assert group_splits(list(reversed(rows))) == owners
    for family in ('family-a', 'family-b'):
        splits = [owners[f'{family}:{number}'] for number in range(32)]
        assert splits.count('validation') == 3
        assert splits.count('test') == 3
        assert splits.count('train') == 26
    assert owners['generated:one'] == 'train'


def test_small_cohort_has_two_genuine_holdouts_after_source_failures():
    rows = [row('one', 'f1'), row('two', 'f2'), row('three', 'f3'),
            dict(row('failed', 'f4'), disposition='excluded')]
    owners = group_splits(rows)
    assert set(owners.values()) == {'train', 'validation', 'test'}
    assert 'failed' not in owners


def test_no_fabricated_holdout_when_only_two_official_groups():
    with pytest.raises(ValueError, match='three usable'):
        group_splits([row('one'), row('two'), row('generated', origin='parc_generated')])


def test_ambiguous_group_family_fails():
    with pytest.raises(ValueError, match='multiple release families'):
        group_splits([row('same', 'first'), row('same', 'second'), row('other'), row('third')])


def test_exact_geometry_and_source_variants_share_components(tmp_path):
    rows = [row('variant', 'first'), row('variant', 'first'), row('copy', 'second'),
            row('other1'), row('other2'), row('other3'), row('generated', origin='parc_generated')]
    for index, item in enumerate(rows):
        path = tmp_path / f'{index}.npz'
        hf = np.full((3, 4), index, dtype=np.float32)
        if index == 2:
            hf[:] = 0  # An exact scene copied to another release family.
        if index == 6:
            hf[:] = 4  # A generated scene also present in the official release.
        np.savez(path, hf=hf, dx=.1, min_point=np.array([index * 100., 0.]))
        item['retarget_path'] = str(path)
    summary = bind_geometry_components(rows)
    assert summary['source_groups_before'] == 6
    assert summary['split_components_after'] == 4
    assert rows[0]['source_group'] == rows[1]['source_group'] == rows[2]['source_group']
    assert rows[4]['source_group'] == rows[6]['source_group']
    owners = group_splits(rows)
    assert owners[rows[4]['source_group']] == 'train'
    assert set(owners.values()) == {'train', 'validation', 'test'}


def test_geometry_binding_excludes_rejected_rows(tmp_path):
    rows = [dict(row('failed'), disposition='excluded')]
    assert bind_geometry_components(rows)['admitted_rows'] == 0
