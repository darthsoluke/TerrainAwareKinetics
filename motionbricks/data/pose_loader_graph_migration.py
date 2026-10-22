"""Admit label graphs only after the actual pinned-loader producer is proven."""

import json
from pathlib import Path

from .codec_exposure_migration import file_sha
from .reference_exposure_budget import ExposureBudget


def validate_pose_loader_graph_migration(saved, contract, receipt):
    old = saved['contract']
    if (old['stage'] != 'pose' or old['batch'] != 6144 or old['micro_batch'] != 3072
            or old['cuda_memory_fraction'] != .9 or old.get('pose_parallel_devices') != 2
            or not old.get('isolated_pose_teacher_rng') or not old.get('pose_token_memoization')
            or not old.get('coalesced_dilated_bias') or old.get('frozen_pose_label_graphs')
            or contract.get('frozen_pose_label_graphs') is not True
            or contract.get('frozen_pose_capture_error_mode') != 'thread_local'):
        raise ValueError('Unexpected actual-loader Pose graph transition')
    changed = {'source_code_sha256', 'implementation_migration_sha256',
               'frozen_pose_label_graphs', 'frozen_pose_capture_error_mode'}
    if {k: v for k, v in old.items() if k not in changed} != {
            k: v for k, v in contract.items() if k not in changed}:
        raise ValueError('Loader graph execution cannot change another training contract')
    prior = old['exposure_budget']
    ExposureBudget(prior['initial_updates'], prior['initial_exposures'], 6144,
                   saved['actual_updates'], saved['actual_exposures'])
    if (receipt.get('actual_updates') != saved['actual_updates']
            or receipt.get('actual_exposures') != saved['actual_exposures']
            or receipt.get('from_source_code_sha256') != old['source_code_sha256']
            or receipt.get('to_source_code_sha256') != contract['source_code_sha256']
            or receipt.get('no_optimizer_updates_lost') is not True
            or receipt.get('execution_trajectory_bitwise') is not True):
        raise ValueError('Loader graph source or checkpoint accounting differs')
    sources = contract['source_code_sha256']
    allowed_source_changes = {'scripts/train_unified_full_corpus.py',
                              'motionbricks/data/dilated_graph_convolution.py'}
    for name, sha in old['source_code_sha256'].items():
        if name not in allowed_source_changes and sources.get(name) != sha:
            raise ValueError('Loader graph migration changed another existing source')
    proofs = {}
    for item in receipt.get('proofs', []):
        if item['role'] in proofs or file_sha(item['path']) != item['sha256']:
            raise ValueError('Actual-loader proof identity differs')
        result = json.loads(Path(item['path']).read_text())
        if not result.get('passed') or result.get('production_budget_updates') != 0:
            raise ValueError('Actual-loader execution proof did not pass')
        proofs[item['role']] = (item, result)
    if set(proofs) != {'actual_native', 'actual_graph', 'actual_cold_resume'}:
        raise ValueError('Actual producer native, graph and cold-resume optimizer proofs are required')
    reference_item, reference = proofs['actual_native']
    for role, mode, count, cold in [('actual_native', 'native', 4, False),
                                  ('actual_graph', 'graph', 4, False), ('actual_cold_resume', 'graph', 2, True)]:
        _, result = proofs[role]
        if (result.get('mode') != mode or result.get('actual_optimizer_updates') != count
                or result.get('actual_trainer_entrypoint') is not True
                or result.get('diagnostic_admission_identity_override') is not True
                or result.get('cold_process_resume') is not cold
                or result.get('loader') != dict(workers=4, pin_memory=True, persistent_workers=True)
                or result.get('preload_workspaces_gib_per_device') != 50
                or result.get('preload_receipt', {}).get('preload_reserved_bytes_per_device') != 50*1024**3
                or result.get('global_batch') != 6144 or result.get('compute_micro') != 3072
                or result.get('memory_fraction') != .9 or result.get('source_updates') != 2584
                or result.get('endpoint_actual_updates') != 2588 or result.get('endpoint_actual_exposures') != 4124672
                or result.get('checkpoint_sha256') != reference.get('checkpoint_sha256')
                or any(result.get(key) != [] for key in ('computation_changes', 'endpoint_changes', 'validation_changes'))
                or len(result.get('validation', [])) != count or len(result.get('progress', [])) != count):
            raise ValueError('Actual producer loader, preload, optimizer, validation or recovery differs')
        if mode == 'graph' and (result.get('capture_error_mode') != 'thread_local'
                or result.get('reference_report_sha256') != reference_item['sha256']):
            raise ValueError('Thread-local graph execution lacks the actual native reference')
        for row in result['validation']:
            if (set(row['domains']) != {'pfnn', 'parc', 'bones'}
                    or any(value['examples'] != 128 for value in row['domains'].values())):
                raise ValueError('Actual producer validation coverage differs')
        for name in ('scripts/train_unified_full_corpus.py', 'motionbricks/data/frozen_graph_pose_training.py',
                     'motionbricks/data/dilated_graph_convolution.py', 'scripts/check_pose_actual_loader_optimizer.py'):
            if name not in sources or result['source_sha256'].get(name) != sources[name]:
                raise ValueError('Actual producer execution source differs from its proof')
    graph_item, _ = proofs['actual_graph']
    _, cold = proofs['actual_cold_resume']
    if cold.get('resume_state_sha256') != file_sha(Path(graph_item['path']).parent/'midpoint.pt'):
        raise ValueError('Actual-loader cold-resume checkpoint identity differs')
