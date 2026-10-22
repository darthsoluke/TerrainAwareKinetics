"""Admission for a proven two-device, sample-equivalent Pose continuation."""

import json
from pathlib import Path

from .codec_exposure_migration import file_sha
from .reference_exposure_budget import ExposureBudget, TARGET_EXPOSURES


def validate_pose_exposure_migration(saved, contract, receipt):
    old = saved['contract']
    if (old['stage'] != 'pose' or contract['stage'] != 'pose' or old['batch'] != 256
            or old['micro_batch'] != 64 or old['cuda_memory_fraction'] != 1.
            or contract['batch'] != 6144 or contract['micro_batch'] != 3072
            or contract['cuda_memory_fraction'] != .9 or contract.get('pose_parallel_devices') != 2
            or contract.get('isolated_pose_teacher_rng') is not True
            or not old.get('pose_token_memoization') or not old.get('coalesced_dilated_bias')
            or old.get('exposure_budget') or old.get('graph_dilated_convolutions')
            or old['root_parallel_devices'] != 1):
        raise ValueError('Unexpected sample-equivalent Pose transition')
    changed = {'batch', 'micro_batch', 'cuda_memory_fraction', 'pose_parallel_devices',
        'isolated_pose_teacher_rng', 'exposure_budget', 'source_code_sha256', 'implementation_migration_sha256'}
    if {k: v for k, v in old.items() if k not in changed} != {
            k: v for k, v in contract.items() if k not in changed}:
        raise ValueError('Pose transition changed another data, model, loss or optimizer contract')
    expected = ExposureBudget(saved['actual_updates'], saved['actual_exposures'], 6144).contract()
    if (saved['actual_exposures'] != saved['actual_updates'] * 256
            or (TARGET_EXPOSURES-saved['actual_exposures']) % 6144 not in (0, 1536)
            or contract['exposure_budget'] != expected
            or receipt.get('actual_updates') != saved['actual_updates']
            or receipt.get('actual_exposures') != saved['actual_exposures']
            or receipt.get('from_source_code_sha256') != old['source_code_sha256']
            or receipt.get('to_source_code_sha256') != contract['source_code_sha256']
            or receipt.get('authorization') != 'user_requested_multigpu_large_memory_reference_50000'
            or receipt.get('optimizer_trajectory_equivalent') is not False
            or receipt.get('no_optimizer_updates_lost') is not True):
        raise ValueError('Pose exposure budget, proven tail, authorization or source differs')
    proofs = {}
    for item in receipt.get('proofs', []):
        role = item['role']
        if role in proofs or file_sha(item['path']) != item['sha256']:
            raise ValueError('Pose proof identity differs')
        result = json.loads(Path(item['path']).read_text())
        if not result.get('passed') or result.get('production_budget_updates') != 0:
            raise ValueError('Pose execution proof did not pass')
        proofs[role] = (item, result)
    if set(proofs) != {'teacher_rng', 'reference', 'continuous', 'cold_resume'}:
        raise ValueError('Pose teacher, serial, parallel and cold-process proofs are all required')
    teacher_item, teacher = proofs['teacher_rng']
    if (teacher.get('actual_optimizer_updates') != 0 or not teacher.get('exception_restored')
            or teacher.get('model_changes') != [[], []]
            or [row['batch'] for row in teacher.get('rows', [])] != [64, 3072, 1536]
            or any(row['differences'] != [[], []] or not row['cpu_rng_unchanged']
                   or not row['native_target_rng_equal'] for row in teacher['rows'])):
        raise ValueError('Device-local Pose teacher computation is unproven')
    for name in ('motionbricks/data/pose_teacher_rng.py', 'motionbricks/data/unified_pose_training.py',
                 'motionbricks/data/unified_full_training.py'):
        if teacher['source_sha256'].get(name) != contract['source_code_sha256'].get(name):
            raise ValueError('Pose teacher source differs from its proof')
    expected_phases = {'reference': ('reference', 4, [6144, 6144, 6144, 1536]),
                       'continuous': ('parallel', 4, [6144, 6144, 6144, 1536]),
                       'cold_resume': ('resume', 2, [6144, 1536])}
    for role, (phase, count, samples) in expected_phases.items():
        _, result = proofs[role]
        if (result.get('phase') != phase or result.get('actual_optimizer_updates') != count
                or result.get('global_batch') != 6144 or result.get('compute_micro') != 3072
                or result.get('final_partial_batch') != 1536 or result.get('memory_fraction') != .9
                or result.get('diagnostic_endpoint_exposures') != 19968
                or result.get('diagnostic_endpoint_reference_steps') != 78
                or result.get('cache_persisted') is not False or result.get('rng_from_reference_oracle') is not False
                or result.get('prerequisite_sha256') != teacher_item['sha256']
                or result.get('checkpoint_sha256') != teacher['checkpoint_sha256']
                or [row['examples'] for row in result.get('rows', [])] != samples
                or any(result.get(key) != [] for key in ('computation_changes', 'endpoint_changes',
                                                         'validation_changes', 'frozen_changes'))
                or set(result.get('validation', {})) != {'pfnn', 'parc', 'bones'}
                or any(row['examples'] != 128 for row in result['validation'].values())
                or result.get('implementation_sha256') != contract['source_code_sha256'].get(
                    'motionbricks/data/parallel_pose_training.py')
                or result.get('script_sha256') != contract['source_code_sha256'].get(
                    'scripts/check_parallel_pose_exposure.py')):
            raise ValueError('Pose optimizer, validation, tail or source proof differs')
    midpoint = Path(proofs['continuous'][0]['path']).parent/'midpoint.pt'
    if receipt.get('diagnostic_midpoint_sha256') != file_sha(midpoint):
        raise ValueError('Pose diagnostic continuation checkpoint changed')
