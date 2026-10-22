from copy import deepcopy
import json

import pytest

from motionbricks.data.codec_exposure_migration import file_sha
from motionbricks.data.pose_exposure_migration import validate_pose_exposure_migration
from motionbricks.data.reference_exposure_budget import ExposureBudget


@pytest.fixture
def migration(tmp_path):
    sources = {name: 'a'*64 for name in ('motionbricks/data/pose_teacher_rng.py',
        'motionbricks/data/unified_pose_training.py', 'motionbricks/data/unified_full_training.py',
        'motionbricks/data/parallel_pose_training.py', 'scripts/check_parallel_pose_exposure.py')}
    old = dict(stage='pose', batch=256, micro_batch=64, cuda_memory_fraction=1.,
        pose_token_memoization=True, coalesced_dilated_bias=True, graph_dilated_convolutions=False,
        root_parallel_devices=1, parent_sha256='p'*64, steps=50000, learning_rate=5e-5,
        native_losses_unchanged=True, source_code_sha256={'old.py': 'b'*64})
    saved = dict(contract=old, actual_updates=2000, actual_exposures=512000)
    contract = dict(old, batch=6144, micro_batch=3072, cuda_memory_fraction=.9,
        pose_parallel_devices=2, isolated_pose_teacher_rng=True, source_code_sha256=sources,
        exposure_budget=ExposureBudget(2000, 512000, 6144).contract())
    receipt = dict(actual_updates=2000, actual_exposures=512000,
        from_source_code_sha256=old['source_code_sha256'], to_source_code_sha256=sources,
        authorization='user_requested_multigpu_large_memory_reference_50000',
        optimizer_trajectory_equivalent=False, no_optimizer_updates_lost=True, proofs=[])

    def proof(role, result):
        folder = tmp_path/role
        folder.mkdir()
        path = folder/'complete.json'
        path.write_text(json.dumps(result))
        receipt['proofs'].append(dict(role=role, path=str(path), sha256=file_sha(path)))
        return path

    teacher = proof('teacher_rng', dict(passed=True, production_budget_updates=0,
        actual_optimizer_updates=0, exception_restored=True, model_changes=[[], []],
        rows=[dict(batch=size, differences=[[], []], cpu_rng_unchanged=True, native_target_rng_equal=True)
              for size in (64, 3072, 1536)], source_sha256=sources, checkpoint_sha256='c'*64))
    for role, phase, count, samples in [('reference', 'reference', 4, [6144]*3+[1536]),
            ('continuous', 'parallel', 4, [6144]*3+[1536]), ('cold_resume', 'resume', 2, [6144, 1536])]:
        proof(role, dict(passed=True, production_budget_updates=0, phase=phase,
            actual_optimizer_updates=count, global_batch=6144, compute_micro=3072,
            final_partial_batch=1536, memory_fraction=.9, diagnostic_endpoint_exposures=19968,
            diagnostic_endpoint_reference_steps=78, cache_persisted=False, rng_from_reference_oracle=False,
            prerequisite_sha256=file_sha(teacher), checkpoint_sha256='c'*64,
            rows=[dict(examples=size) for size in samples], computation_changes=[], endpoint_changes=[],
            validation_changes=[], frozen_changes=[], validation={domain: dict(examples=128, native_loss=1.)
                for domain in ('pfnn', 'parc', 'bones')}, implementation_sha256='a'*64, script_sha256='a'*64))
    midpoint = tmp_path/'continuous/midpoint.pt'
    midpoint.write_bytes(b'diagnostic checkpoint identity')
    receipt['diagnostic_midpoint_sha256'] = file_sha(midpoint)
    return saved, contract, receipt


def test_bound_pose_exposure_transition_is_admitted(migration):
    validate_pose_exposure_migration(*migration)


@pytest.mark.parametrize('field,value', [('batch', 12288), ('micro_batch', 1024),
    ('cuda_memory_fraction', .8), ('pose_parallel_devices', 4), ('isolated_pose_teacher_rng', False),
    ('parent_sha256', 'different'), ('learning_rate', .001), ('native_losses_unchanged', False)])
def test_learning_parent_or_unproven_topology_changes_are_rejected(migration, field, value):
    saved, contract, receipt = deepcopy(migration)
    contract[field] = value
    with pytest.raises(ValueError):
        validate_pose_exposure_migration(saved, contract, receipt)


def test_unproven_final_microbatch_size_is_rejected(migration):
    saved, contract, receipt = deepcopy(migration)
    saved.update(actual_updates=1000, actual_exposures=256000)
    receipt.update(actual_updates=1000, actual_exposures=256000)
    contract['exposure_budget'] = ExposureBudget(1000, 256000, 6144).contract()
    with pytest.raises(ValueError, match='proven tail'):
        validate_pose_exposure_migration(saved, contract, receipt)


@pytest.mark.parametrize('role,field,value', [('cold_resume', 'endpoint_changes', ['Adam']),
    ('continuous', 'rng_from_reference_oracle', True), ('reference', 'global_batch', 3072),
    ('cold_resume', 'cache_persisted', True), ('continuous', 'implementation_sha256', 'different'),
    ('teacher_rng', 'exception_restored', False)])
def test_failed_or_different_proofs_cannot_be_readmitted_with_a_new_file_hash(migration, role, field, value):
    saved, contract, receipt = migration
    item = next(p for p in receipt['proofs'] if p['role'] == role)
    from pathlib import Path
    path = Path(item['path'])
    result = json.loads(path.read_text())
    result[field] = value
    path.write_text(json.dumps(result))
    item['sha256'] = file_sha(path)
    with pytest.raises(ValueError):
        validate_pose_exposure_migration(saved, contract, receipt)


def test_diagnostic_midpoint_cannot_be_substituted(migration):
    saved, contract, receipt = migration
    receipt['diagnostic_midpoint_sha256'] = 'd'*64
    with pytest.raises(ValueError, match='checkpoint changed'):
        validate_pose_exposure_migration(saved, contract, receipt)


@pytest.mark.parametrize('current,target', [(2, 2000), (1000, 2000), (1990, 2000),
    (2000, 5000), (4990, 5000), (5000, 8000)])
def test_checkpoint_handoff_leaves_only_proven_full_compute_batches(current, target):
    from scripts.continue_pfnn_pose_exposure_workflow import next_handoff_step
    assert next_handoff_step(current) == target
    assert target > current and target % 1000 == 0
    assert (12800000-target*256) % 6144 == 0
