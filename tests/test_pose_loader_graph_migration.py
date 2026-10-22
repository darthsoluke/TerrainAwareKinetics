from copy import deepcopy
import json
from pathlib import Path

import pytest

from motionbricks.data.codec_exposure_migration import file_sha
from motionbricks.data.dilated_graph_convolution import GraphConvolutionExecution
from motionbricks.data.pose_loader_graph_migration import validate_pose_loader_graph_migration
from motionbricks.data.reference_exposure_budget import ExposureBudget


@pytest.fixture
def actual_proof(tmp_path):
    names = ('scripts/train_unified_full_corpus.py', 'motionbricks/data/dilated_graph_convolution.py',
        'motionbricks/data/frozen_graph_pose_training.py', 'scripts/check_pose_actual_loader_optimizer.py',
        'motionbricks/data/parallel_pose_training.py')
    sources = {name: 'a'*64 for name in names}
    old = dict(stage='pose', batch=6144, micro_batch=3072, cuda_memory_fraction=.9, pose_parallel_devices=2,
        isolated_pose_teacher_rng=True, pose_token_memoization=True, coalesced_dilated_bias=True,
        source_code_sha256=deepcopy(sources), parent_sha256='final-codec', learning_rate=5e-5,
        exposure_budget=ExposureBudget(2000,512000,6144).contract())
    saved = dict(contract=old, actual_updates=2600, actual_exposures=4198400)
    contract = dict(old, frozen_pose_label_graphs=True, frozen_pose_capture_error_mode='thread_local',
                    source_code_sha256=deepcopy(sources))
    receipt = dict(actual_updates=2600, actual_exposures=4198400,
        from_source_code_sha256=old['source_code_sha256'], to_source_code_sha256=contract['source_code_sha256'],
        no_optimizer_updates_lost=True, execution_trajectory_bitwise=True, proofs=[])
    reference = None
    for role, mode, count, cold in [('actual_native','native',4,False),('actual_graph','graph',4,False),
                                    ('actual_cold_resume','graph',2,True)]:
        folder = tmp_path/role
        folder.mkdir()
        row = dict(passed=True, production_budget_updates=0, mode=mode, actual_optimizer_updates=count,
            actual_trainer_entrypoint=True, diagnostic_admission_identity_override=True, cold_process_resume=cold,
            loader=dict(workers=4,pin_memory=True,persistent_workers=True), preload_workspaces_gib_per_device=50,
            preload_receipt=dict(preload_reserved_bytes_per_device=50*1024**3), global_batch=6144,
            compute_micro=3072, memory_fraction=.9, source_updates=2584, endpoint_actual_updates=2588,
            endpoint_actual_exposures=4124672, checkpoint_sha256='fixture', computation_changes=[],
            endpoint_changes=[], validation_changes=[], source_sha256=sources, progress=[{} for _ in range(count)],
            validation=[dict(domains={domain:dict(examples=128,native_loss=1.) for domain in ('pfnn','parc','bones')})
                        for _ in range(count)])
        if mode == 'graph':
            row.update(capture_error_mode='thread_local',reference_report_sha256=reference['sha256'])
        if cold:
            row['resume_state_sha256'] = file_sha(tmp_path/'actual_graph/midpoint.pt')
        path = folder/'complete.json'
        path.write_text(json.dumps(row))
        item = dict(role=role,path=str(path),sha256=file_sha(path))
        receipt['proofs'].append(item)
        if role == 'actual_native':reference = item
        if role == 'actual_graph':(folder/'midpoint.pt').write_bytes(b'actual producer midpoint')
    return saved, contract, receipt


def test_actual_loader_native_graph_and_cold_proofs_are_admitted(actual_proof):
    validate_pose_loader_graph_migration(*actual_proof)


@pytest.mark.parametrize('field,value', [('actual_trainer_entrypoint',False),
    ('loader',dict(workers=0,pin_memory=False,persistent_workers=False)),
    ('preload_workspaces_gib_per_device',0), ('capture_error_mode','global'),
    ('actual_optimizer_updates',0), ('endpoint_changes',['Adam']), ('resume_state_sha256','other')])
def test_prepared_batches_or_failed_recovery_cannot_replace_actual_loader_proof(actual_proof, field, value):
    saved,contract,receipt=actual_proof
    item=receipt['proofs'][2]
    path=Path(item['path']);report=json.loads(path.read_text());report[field]=value
    path.write_text(json.dumps(report));item['sha256']=file_sha(path)
    with pytest.raises(ValueError):validate_pose_loader_graph_migration(saved,contract,receipt)


def test_loader_graph_cannot_change_the_existing_native_executor(actual_proof):
    saved,contract,receipt=actual_proof
    contract['source_code_sha256']['motionbricks/data/parallel_pose_training.py']='changed'
    with pytest.raises(ValueError,match='another existing source'):
        validate_pose_loader_graph_migration(saved,contract,receipt)


def test_capture_configuration_preserves_checks_and_rejects_relaxed_mode():
    assert GraphConvolutionExecution().capture_error_mode=='global'
    assert GraphConvolutionExecution(capture_error_mode='thread_local').capture_error_mode=='thread_local'
    with pytest.raises(ValueError):GraphConvolutionExecution(capture_error_mode='relaxed')
