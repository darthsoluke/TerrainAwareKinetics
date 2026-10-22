"""Require exact frozen-label graph proofs without changing the Pose budget."""

import json
from pathlib import Path

from .codec_exposure_migration import file_sha
from .reference_exposure_budget import ExposureBudget


def validate_pose_graph_migration(saved, contract, receipt):
    old = saved['contract']
    if (old['stage'] != 'pose' or old['batch'] != 6144 or old['micro_batch'] != 3072
            or old['cuda_memory_fraction'] != .9 or old.get('pose_parallel_devices') != 2
            or not old.get('isolated_pose_teacher_rng') or not old.get('pose_token_memoization')
            or not old.get('coalesced_dilated_bias') or old.get('graph_dilated_convolutions')
            or old.get('frozen_pose_label_graphs') or contract.get('frozen_pose_label_graphs') is not True):
        raise ValueError('Unexpected frozen-label Pose graph transition')
    changed = {'source_code_sha256', 'implementation_migration_sha256', 'frozen_pose_label_graphs'}
    if {k: v for k, v in old.items() if k not in changed} != {
            k: v for k, v in contract.items() if k not in changed}:
        raise ValueError('Pose label graphs cannot change data, scope, batch, optimizer or exposure policy')
    policy = old['exposure_budget']
    ExposureBudget(policy['initial_updates'], policy['initial_exposures'], 6144,
                   saved['actual_updates'], saved['actual_exposures'])
    if (receipt.get('actual_updates') != saved['actual_updates']
            or receipt.get('actual_exposures') != saved['actual_exposures']
            or receipt.get('from_source_code_sha256') != old['source_code_sha256']
            or receipt.get('to_source_code_sha256') != contract['source_code_sha256']
            or receipt.get('no_optimizer_updates_lost') is not True
            or receipt.get('execution_trajectory_bitwise') is not True):
        raise ValueError('Pose graph source, accounting or continuity binding differs')
    sources = contract['source_code_sha256']
    for name in ('motionbricks/data/unified_full_training.py', 'motionbricks/data/unified_pose_training.py',
                 'motionbricks/data/pose_teacher_rng.py', 'motionbricks/data/parallel_pose_training.py',
                 'motionbricks/data/dilated_bias_convolution.py', 'motionbricks/data/exposure_codec_training.py',
                 'motionbricks/data/reference_exposure_budget.py'):
        if name not in sources or sources[name] != old['source_code_sha256'].get(name):
            raise ValueError('Pose graph migration changed another validated computation')
    proofs = {}
    for item in receipt.get('proofs', []):
        if item['role'] in proofs or file_sha(item['path']) != item['sha256']:
            raise ValueError('Pose graph proof identity differs')
        result = json.loads(Path(item['path']).read_text())
        if not result.get('passed') or result.get('production_budget_updates') != 0:
            raise ValueError('Pose graph execution proof failed')
        proofs[item['role']] = (item, result)
    if set(proofs) != {'zero_update', 'native_reference', 'graph_continuous', 'graph_cold_resume'}:
        raise ValueError('Frozen-label, native optimizer, graph optimizer and cold proofs are required')
    _, zero = proofs['zero_update']
    if (zero.get('actual_optimizer_updates') != 0 or zero.get('model_changes') != []
            or zero.get('final_codec_parent_sha256') != old['parent_sha256']
            or [row['batch'] for row in zero.get('rows', [])] != [3072, 3072, 3072, 64]
            or any(row['differences'] != [] for row in zero['rows'])):
        raise ValueError('Frozen final-codec label graph equality is unproven')
    native_item, native = proofs['native_reference']
    base_hash = sources['motionbricks/data/parallel_pose_training.py']
    if (native.get('phase') != 'reference' or native.get('actual_optimizer_updates') != 4
            or native.get('global_batch') != 6144 or native.get('compute_micro') != 3072
            or native.get('implementation_sha256') != base_hash
            or set(native.get('validation', {})) != {'pfnn', 'parc', 'bones'}
            or any(value['examples'] != 128 for value in native['validation'].values())):
        raise ValueError('Native optimizer reference differs from the deployed Pose executor')
    for role, phase, count, samples in (
            ('graph_continuous', 'parallel', 4, [6144, 6144, 6144, 1536]),
            ('graph_cold_resume', 'resume', 2, [6144, 1536])):
        _, result = proofs[role]
        if (result.get('phase') != phase or result.get('actual_optimizer_updates') != count
                or result.get('global_batch') != 6144 or result.get('compute_micro') != 3072
                or result.get('memory_fraction') != .9 or result.get('final_partial_batch') != 1536
                or result.get('implementation_sha256') != base_hash
                or result.get('script_sha256') != sources.get('scripts/check_pose_graph_optimizer.py')
                or result.get('native_reference_report_sha256') != native_item['sha256']
                or result.get('checkpoint_sha256') != native['checkpoint_sha256']
                or result.get('frozen_graph_labels') is not True or result.get('cache_persisted') is not False
                or result.get('rng_from_reference_oracle') is not False or len(result.get('gpus', [])) != 2
                or result.get('label_graph_warmups') != 3 or result.get('label_graph_signatures') != [18, 6]
                or [row['examples'] for row in result.get('rows', [])] != samples
                or any(result.get(key) != [] for key in ('computation_changes', 'endpoint_changes',
                                                         'validation_changes', 'frozen_changes'))
                or result.get('validation') != native.get('validation')):
            raise ValueError('Pose graph Adam, validation, tail or cold continuation differs')
        for name in ('motionbricks/data/frozen_graph_pose_training.py',
                     'motionbricks/data/dilated_graph_convolution.py', 'motionbricks/data/dilated_graph_context.py'):
            if name not in sources or result['graph_source_sha256'].get(name) != sources[name]:
                raise ValueError('Frozen-label graph source differs from its optimizer proof')
            if 'frozen_graph_pose_training' not in name and zero['code_sha256'].get(name) != sources[name]:
                raise ValueError('Graph numerical proof identifies another implementation')
    continuous, _ = proofs['graph_continuous']
    _, resume = proofs['graph_cold_resume']
    if resume.get('resume_state_sha256') != file_sha(Path(continuous['path']).parent/'midpoint.pt'):
        raise ValueError('Graph cold-resume midpoint identity differs')
