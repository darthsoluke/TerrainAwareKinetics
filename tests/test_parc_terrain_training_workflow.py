from argparse import Namespace
import copy
import hashlib
import json

import pytest

from scripts.run_parc_terrain_training import (
    preparation_commands, corpus_budget_readiness, prepared_catalog_contract, prepared_dataset_contract,
)
from motionbricks.data.native_mb_paper_protocol import profile_notes


def records():
    rows = [dict(split='train', origin='parc_official', source_group=f'group{i}', frames_30hz=512,
        source_manifest_identity=f'mxucg/PARC@revision:dec_release/family{i % 8}/{i}.pkl')
        for i in range(80)]
    rows += [dict(split='train', origin='parc_generated', source_group=f'generated{i}', frames_30hz=100)
             for i in range(3)]
    for split in ('validation', 'test'):
        rows.append(dict(split=split, origin='parc_official', source_group=split, frames_30hz=100,
            source_manifest_identity=f'mxucg/PARC@revision:dec_release/family0/{split}.pkl'))
    return rows


def test_resource_guard_counts_groups_and_train_only_frames():
    result = corpus_budget_readiness(records())
    assert result['train_frames'] == 80 * 512 + 300
    assert result['official_source_groups'] == 82
    assert result['train_origin_counts'] == {'parc_official': 80, 'parc_generated': 3}
    assert 'not a quality' in result['meaning']


@pytest.mark.parametrize('mode', ['no_generated', 'alien_origin', 'tiny', 'single_family',
                                'short_train', 'no_test', 'bad_identity', 'single_generated'])
def test_resource_guard_rejects_long_training_on_invalid_cohort(mode):
    rows = copy.deepcopy(records())
    if mode == 'no_generated':
        rows = [row for row in rows if row['origin'] != 'parc_generated']
    elif mode == 'single_generated':
        rows = [row for row in rows if row['origin'] != 'parc_generated' or row['source_group'] == 'generated0']
    elif mode == 'alien_origin':
        rows[0]['origin'] = 'lafan1'
    elif mode == 'tiny':
        rows = rows[:8] + rows[-5:]
    elif mode == 'single_family':
        for row in rows:
            row['source_manifest_identity'] = 'release:dec_release/family0/clip.pkl'
    elif mode == 'short_train':
        for row in rows:
            row['frames_30hz'] = 65
    elif mode == 'no_test':
        rows = [row for row in rows if row['split'] != 'test']
    elif mode == 'bad_identity':
        rows[0]['source_manifest_identity'] = 'release:clip.pkl'
    with pytest.raises(ValueError):
        corpus_budget_readiness(rows)


def test_preparation_commands_are_pinned_parc_only_and_versioned(tmp_path):
    args = Namespace(output=tmp_path / 'new', archive=tmp_path / 'dec_release.zip',
        archive_index=tmp_path / 'index.json', release_audit=tmp_path / 'audit.json',
        generated_export=[tmp_path / 'generated/export'], groups_per_family=32)
    commands = preparation_commands(args)
    assert [key for key, _ in commands] == ['extract', 'retarget', 'dataset']
    assert '--prepare-retargets' in commands[1][1]
    assert str((tmp_path / 'generated/export').resolve()) in commands[1][1]
    assert commands[2][1][-1] == str((tmp_path / 'new/dataset').resolve())
    assert all('--warmstart' not in command and '--resume' not in command for _, command in commands)


def test_parc_training_scope_does_not_claim_flat_subset():
    notes = profile_notes('root', parc_only=True)
    assert 'PARC paired motion-terrain' in notes['explicit_deviations'][1]
    assert 'flat locomotion' not in notes['explicit_deviations'][1]
    assert notes['paper_disclosed_optimizer_and_schedule'] == profile_notes('root')['paper_disclosed_optimizer_and_schedule']


def test_prepared_catalog_reuses_conversion_but_rebuilds_new_dataset(tmp_path):
    path = tmp_path / 'completed/catalog.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'schema': 'parc.only_teacher_catalog.v1', 'rows': []}))
    args = Namespace(output=tmp_path / 'v2', prepared_catalog=path,
        prepared_catalog_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    contract = prepared_catalog_contract(args)
    assert contract['sha256'] == args.prepared_catalog_sha256
    commands = preparation_commands(args)
    assert [phase for phase, _ in commands] == ['dataset']
    assert str(path.resolve()) in commands[0][1]
    assert commands[0][1][-1] == str((args.output / 'dataset').resolve())
    path.write_text(json.dumps({'schema': 'parc.only_teacher_catalog.v1', 'rows': [1]}))
    with pytest.raises(ValueError, match='pinned SHA256 changed'):
        prepared_catalog_contract(args)


def test_prepared_dataset_reuses_pinned_conversion_without_rewriting_it(tmp_path):
    dataset = tmp_path / 'completed/dataset'
    dataset.mkdir(parents=True)
    manifest = dataset / 'native_manifest.json'
    manifest.write_text(json.dumps({'schema': 'parc-only-native-dataset.v1'}))
    args = Namespace(output=tmp_path / 'v3', prepared_dataset=dataset,
        prepared_dataset_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert prepared_dataset_contract(args)['path'] == str(dataset.resolve())
    assert preparation_commands(args) == []
    manifest.write_text(json.dumps({'schema': 'parc-only-native-dataset.v1', 'changed': True}))
    with pytest.raises(ValueError, match='pinned SHA256 changed'):
        prepared_dataset_contract(args)
