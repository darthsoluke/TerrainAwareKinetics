from copy import deepcopy
import json
from pathlib import Path

import pytest

from motionbricks.data.codec_exposure_migration import file_sha
from motionbricks.data.pose_graph_migration import validate_pose_graph_migration
from motionbricks.data.reference_exposure_budget import ExposureBudget
from scripts.continue_pfnn_pose_graph_workflow import successor_command


@pytest.fixture
def transition(tmp_path):
    names = ('unified_full_training.py', 'unified_pose_training.py', 'pose_teacher_rng.py',
        'parallel_pose_training.py', 'dilated_bias_convolution.py', 'exposure_codec_training.py',
        'reference_exposure_budget.py', 'dilated_graph_convolution.py', 'dilated_graph_context.py',
        'frozen_graph_pose_training.py')
    sources = {'motionbricks/data/'+name: 'a'*64 for name in names}
    sources['scripts/check_pose_graph_optimizer.py'] = 'a'*64
    old = dict(stage='pose', batch=6144, micro_batch=3072, cuda_memory_fraction=.9, pose_parallel_devices=2,
        isolated_pose_teacher_rng=True, pose_token_memoization=True, coalesced_dilated_bias=True,
        graph_dilated_convolutions=False, parent_sha256='parent', source_code_sha256=deepcopy(sources),
        exposure_budget=ExposureBudget(2000, 512000, 6144).contract(), learning_rate=5e-5)
    saved = dict(contract=old, actual_updates=2100, actual_exposures=1126400)
    contract = dict(old, frozen_pose_label_graphs=True, source_code_sha256=deepcopy(sources))
    receipt = dict(actual_updates=2100, actual_exposures=1126400,
        from_source_code_sha256=old['source_code_sha256'], to_source_code_sha256=contract['source_code_sha256'],
        no_optimizer_updates_lost=True, execution_trajectory_bitwise=True, proofs=[])

    def add(role, value):
        folder = tmp_path/role
        folder.mkdir()
        path = folder/'complete.json'
        path.write_text(json.dumps(dict(passed=True, production_budget_updates=0, **value)))
        item = dict(role=role, path=str(path), sha256=file_sha(path))
        receipt['proofs'].append(item)
        return item

    add('zero_update', dict(actual_optimizer_updates=0, model_changes=[], final_codec_parent_sha256='parent',
        rows=[dict(batch=size, differences=[]) for size in (3072, 3072, 3072, 64)], code_sha256=sources))
    validation = {domain: dict(examples=128, native_loss=1.) for domain in ('pfnn', 'parc', 'bones')}
    native = add('native_reference', dict(phase='reference', actual_optimizer_updates=4, global_batch=6144,
        compute_micro=3072, implementation_sha256='a'*64, checkpoint_sha256='fixture', validation=validation))
    continuous = None
    for role, phase, count, samples in [('graph_continuous', 'parallel', 4, [6144]*3+[1536]),
                                      ('graph_cold_resume', 'resume', 2, [6144, 1536])]:
        value = dict(phase=phase, actual_optimizer_updates=count, global_batch=6144, compute_micro=3072,
            memory_fraction=.9, final_partial_batch=1536, implementation_sha256='a'*64, script_sha256='a'*64,
            native_reference_report_sha256=native['sha256'], checkpoint_sha256='fixture', frozen_graph_labels=True,
            cache_persisted=False, rng_from_reference_oracle=False, gpus=[0, 6], label_graph_warmups=3,
            label_graph_signatures=[18, 6], rows=[dict(examples=size) for size in samples],
            computation_changes=[], endpoint_changes=[], validation_changes=[], frozen_changes=[],
            validation=validation, graph_source_sha256=sources)
        if continuous:
            value['resume_state_sha256'] = file_sha(Path(continuous['path']).parent/'midpoint.pt')
        item = add(role, value)
        if phase == 'parallel':
            continuous = item
            (Path(item['path']).parent/'midpoint.pt').write_bytes(b'graph midpoint')
    return saved, contract, receipt


def test_exact_frozen_label_execution_transition_is_admitted(transition):
    validate_pose_graph_migration(*transition)


@pytest.mark.parametrize('field,value', [('batch', 12288), ('micro_batch', 1536),
    ('cuda_memory_fraction', .95), ('pose_parallel_devices', 4), ('learning_rate', .001),
    ('parent_sha256', 'other'), ('frozen_pose_label_graphs', False)])
def test_graph_migration_cannot_change_learning_or_resource_policy(transition, field, value):
    saved, contract, receipt = transition
    contract[field] = value
    with pytest.raises(ValueError):
        validate_pose_graph_migration(saved, contract, receipt)


@pytest.mark.parametrize('field,value', [('endpoint_changes', ['Adam']), ('validation_changes', ['pfnn']),
    ('rng_from_reference_oracle', True), ('cache_persisted', True), ('label_graph_warmups', 1),
    ('resume_state_sha256', 'other')])
def test_failed_cold_proof_is_not_admitted_by_merely_rehashing_its_report(transition, field, value):
    saved, contract, receipt = transition
    item = next(p for p in receipt['proofs'] if p['role'] == 'graph_cold_resume')
    path = Path(item['path'])
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    item['sha256'] = file_sha(path)
    with pytest.raises(ValueError):
        validate_pose_graph_migration(saved, contract, receipt)


def test_graph_does_not_allow_different_existing_sampler_code(transition):
    saved, contract, receipt = transition
    contract['source_code_sha256']['motionbricks/data/reference_exposure_budget.py'] = 'different'
    with pytest.raises(ValueError, match='another validated computation'):
        validate_pose_graph_migration(saved, contract, receipt)


def test_successor_preserves_batch_cache_and_parent_and_replaces_resume_paths():
    original = ['python', '-m', 'scripts.train_unified_full_corpus', '--stage', 'pose', '--output', 'old/pose',
        '--batch', '6144', '--micro-batch', '3072', '--parent', 'final-codec', '--cache-pose-tokens',
        '--pose-exposure-parallel', '--cuda-memory-fraction', '0.9', '--resume', 'old-checkpoint',
        '--resume-migration', 'old-receipt', '--await-resume']
    actual = successor_command(original, Path('new'), Path('handoff'), Path('migration'))
    for flag, expected in [('--batch', '6144'), ('--micro-batch', '3072'), ('--parent', 'final-codec'),
                           ('--resume', 'handoff'), ('--resume-migration', 'migration')]:
        assert actual.count(flag) == 1 and actual[actual.index(flag)+1] == expected
    assert actual.count('--await-resume') == 1 and '--graph-frozen-pose-labels' in actual
    assert '--cache-pose-tokens' in actual and '--pose-exposure-parallel' in actual
    assert original[-3:] == ['--resume-migration', 'old-receipt', '--await-resume']
