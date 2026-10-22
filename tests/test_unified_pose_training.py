from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motionbricks.data.unified_pose_training import configure_pose_scope, load_pose_checkpoint, POSE_SCHEMA, teacher_history_args
from scripts.pilot_unified_pose_capacity import build_schedule, full_mask_schedule


def model():
    m = torch.nn.Module()
    m.root = torch.nn.Linear(2, 2)
    m.codec = torch.nn.Linear(2, 2)
    m.adaptation = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in ('pose', 'encoder', 'decoder', 'root_stage1', 'root_stage2')})
    m.pose = torch.nn.Module()
    m.pose._transformer_model = torch.nn.Module()
    m.pose._transformer_model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(16)])
    for name in ('_proj_pose_output_logit', '_proj_local_root_values', '_terrain_root_height_projection'):
        setattr(m.pose, name, torch.nn.Linear(2, 2))
    m.pose._terrain_root_height_gate = torch.nn.Parameter(torch.zeros(()))
    return m


def test_pose_scope_keeps_root_vocabulary_shared_encoder_and_first14_layers_frozen():
    m = model()
    names = configure_pose_scope(m, 'adapter')
    assert set(names) == {'adaptation.pose.weight', 'adaptation.pose.bias'}
    names = configure_pose_scope(m, 'capacity')
    assert 'pose._transformer_model.layers.14.weight' in names
    assert 'pose._transformer_model.layers.15.weight' in names
    assert 'pose._terrain_root_height_gate' in names
    assert all(not p.requires_grad for module in (m.root, m.codec, m.adaptation['encoder'],
               m.adaptation['decoder'], *m.pose._transformer_model.layers[:14]) for p in module.parameters())
    names = configure_pose_scope(m, 'root_pose')
    assert set(names) == {f'adaptation.{component}.{kind}' for component in
        ('pose', 'root_stage1', 'root_stage2') for kind in ('weight', 'bias')}
    assert all(not p.requires_grad for module in (m.root, m.pose, m.codec, m.adaptation['encoder'],
                                                  m.adaptation['decoder']) for p in module.parameters())


def test_loader_requires_every_changed_native_tensor_and_restores_it():
    source = model()
    names = configure_pose_scope(source, 'capacity')
    delta = {name.removeprefix('pose.'): value.detach().clone() + .3 for name, value in source.named_parameters()
             if name in names and name.startswith('pose.')}
    checkpoint = dict(schema=POSE_SCHEMA, contract=dict(arm='capacity', bundle_sha256='expected', trainable_parameters=names),
                      pose_delta=delta, adaptation=source.adaptation.state_dict())
    target = model()
    root = {k: v.clone() for k, v in target.root.state_dict().items()}
    load_pose_checkpoint(target, checkpoint, 'expected')
    for name, value in delta.items():
        torch.testing.assert_close(target.pose.state_dict()[name], value, atol=0, rtol=0)
    for name, value in root.items():
        torch.testing.assert_close(target.root.state_dict()[name], value, atol=0, rtol=0)
    incomplete = dict(checkpoint, pose_delta={})
    with pytest.raises(ValueError, match='every changed'):
        load_pose_checkpoint(model(), incomplete, 'expected')
    with pytest.raises(ValueError, match='schema'):
        load_pose_checkpoint(model(), checkpoint, 'wrong')
    with pytest.raises(ValueError, match='placement'):
        load_pose_checkpoint(model(), dict(checkpoint, contract=dict(checkpoint['contract'], pose_condition_stage='output')), 'expected')


def test_schedule_balances_real_sources_and_is_repeatable():
    cases = [dict(label=f'{entry}{i}', entry=entry, world=torch.zeros(105, 5), row=dict(origin='parc_generated'))
             for entry in ('up', 'down') for i in range(2)]
    cases += [dict(label=f'flat{i}', world=torch.zeros(129, 5), row=dict(origin='bones_seed')) for i in range(6)]
    schedule = build_schedule(cases, 300)
    assert schedule == build_schedule(cases, 300)
    assert sum(len(batch) for batch in schedule) == 1200
    seen = set()
    for batch in schedule:
        assert [cases[d['case']].get('entry') for d in batch] == ['up', 'down', None, None]
        for draw in batch:
            assert 0 <= draw['start'] <= len(cases[draw['case']]['world']) - 33
            seen.add(draw['case'])
    assert seen == set(range(10))
    flags = full_mask_schedule(schedule, .25)
    assert flags == full_mask_schedule(schedule, .25)
    assert sum(sum(row) for row in flags) == 300
    assert sum(sum(row[:2]) for row in flags) == 150
    assert sum(sum(row[2:]) for row in flags) == 150
    for case_id in range(10):
        for phase in range(3):
            entries = [flags[step][slot] for step, batch in enumerate(schedule)
                       for slot, draw in enumerate(batch) if draw['case'] == case_id and draw['phase'] == phase]
            assert entries
            assert abs(sum(entries) - len(entries) * .25) <= 1
    assert sum(sum(row) for row in full_mask_schedule(schedule, 0)) == 0


@pytest.mark.parametrize('up,down', [(2, 2), (6, 4), (6, 6)])
def test_every_source_visits_all_window_thirds_without_changing_identity_or_mask_seeds(up, down):
    cases = [dict(label=f'{entry}{i}', entry=entry, world=torch.zeros(81 + i * 24, 5), row=dict(origin='parc_generated'))
             for entry, count in (('up', up), ('down', down)) for i in range(count)]
    cases += [dict(label=f'flat{i}', world=torch.zeros(129, 5), row=dict(origin='bones_seed')) for i in range(6)]
    old = build_schedule(cases, 300, 'global-step-phase-v1')
    new = build_schedule(cases, 300)
    for left, right in zip(old, new):
        assert [(d['case'], d['mask_seed']) for d in left] == [(d['case'], d['mask_seed']) for d in right]
    for case_id, case in enumerate(cases):
        draws = [d for batch in new for d in batch if d['case'] == case_id]
        counts = [sum(d['phase'] == phase for d in draws) for phase in range(3)]
        assert min(counts) > 0 and max(counts) - min(counts) <= 1
        thirds = torch.tensor_split(torch.arange(len(case['world']) - 32), 3)
        assert all(d['start'] in thirds[d['phase']] for d in draws)
    if up == 6:
        assert len({step % 3 for step, batch in enumerate(old) for d in batch if d['case'] == 0}) == 1


def test_native_sparse_teacher_drops_only_prefix_hints_without_mutation_or_rng_effects():
    pose = torch.randn(128, 32, 9)
    known = torch.zeros(128, 32, dtype=torch.bool)
    known[:, :4] = True
    args = (torch.zeros(128, 8, 8), torch.randn(128, 32, 4), pose, known, torch.full((128, 1), 8))
    before_pose, before_mask = pose.clone(), known.clone()
    torch_rng, numpy_rng = torch.random.get_rng_state(), np.random.get_state()
    masked = teacher_history_args(args, 15114, 'native-sparse-start-v1')
    torch.testing.assert_close(torch.random.get_rng_state(), torch_rng, atol=0, rtol=0)
    after_numpy = np.random.get_state()
    assert numpy_rng[0] == after_numpy[0] and numpy_rng[2:] == after_numpy[2:]
    np.testing.assert_array_equal(numpy_rng[1], after_numpy[1])
    assert masked[1] is args[1] and masked[4] is args[4]
    assert not masked[3][:, 4:].any()
    assert set(masked[3].sum(1).tolist()) == {1, 2, 3, 4}
    torch.testing.assert_close(pose, before_pose, atol=0, rtol=0)
    torch.testing.assert_close(known, before_mask, atol=0, rtol=0)
    torch.testing.assert_close(masked[2][masked[3]], pose[masked[3]], atol=0, rtol=0)
    torch.testing.assert_close(masked[2][~masked[3]], torch.zeros_like(masked[2][~masked[3]]), atol=0, rtol=0)
    replay = teacher_history_args(args, 15114, 'native-sparse-start-v1')
    torch.testing.assert_close(masked[3], replay[3], atol=0, rtol=0)
    assert teacher_history_args(args, 15114) is args
    future_known = known.clone(); future_known[:, 8] = True
    with pytest.raises(ValueError, match='no future'):
        teacher_history_args((*args[:3], future_known, args[4]), 15114, 'native-sparse-start-v1')
