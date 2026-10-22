import json
from pathlib import Path

import pytest
import torch

from motionbricks.data.dilated_graph_context import graphed_dilated_convolutions
from motionbricks.data.unified_full_training import NativeTrainingStage
from scripts.train_unified_full_corpus import validate_implementation_migration, sha256


def fixture(tmp_path):
    old=dict(stage='codec',steps=50000,batch=256,micro_batch=64,cuda_memory_fraction=.1,
        coalesced_dilated_bias=True,quantizer_ema_updates_per_optimizer_step=4,
        source_code_sha256={'native.py':'original'},parent_sha256='root-final')
    sources={'motionbricks/data/dilated_graph_convolution.py':'graph',
        'motionbricks/data/unified_full_training.py':'stage','motionbricks/data/dilated_graph_context.py':'context'}
    new=dict(old,graph_dilated_convolutions=True,cuda_memory_fraction=.125,source_code_sha256=sources)
    common=dict(passed=True,implementation_sha256='graph',graph_mode='both',graph_memory_fraction=.125,
        native_memory_fraction=.1,tolerances_unchanged=True,actual_optimizer_updates=12,
        native_graph_endpoint_changes=[],cold_resume_endpoint_changes=[],native_to_graph_endpoint_changes=[],
        restoration_bitwise_checks=[True]*3,computation_gates=[dict(passed=True,bitwise_equal=True)]*8)
    proofs=[]
    for role in ('optimizer_resume','stage_integration'):
        result=dict(common,stage_integration=role=='stage_integration',
            integration_source_sha256={k:v for k,v in sources.items() if k.endswith(('unified_full_training.py','dilated_graph_context.py'))})
        path=tmp_path/(role+'.json');path.write_text(json.dumps(result))
        proofs.append(dict(role=role,path=str(path),sha256=sha256(path)))
    cold=dict(passed=True,cold_process=True,actual_optimizer_updates=2,endpoint_changes=[],
        stage_integration=True,graph_memory_fraction=.125,graph_mode='both',implementation_sha256='graph',
        computation_gates=common['computation_gates'][:2],reference_proof_sha256=proofs[1]['sha256'])
    path=tmp_path/'cold.json';path.write_text(json.dumps(cold))
    proofs.append(dict(role='cold_process',path=str(path),sha256=sha256(path)))
    saved=dict(contract=old,actual_updates=4000,actual_exposures=1024000)
    receipt=dict(transition='codec-dilated-graph-v1',from_checkpoint_sha256='checkpoint',
        from_source_code_sha256=old['source_code_sha256'],to_source_code_sha256=sources,
        actual_updates=4000,graph_implementation_sha256='graph',proofs=proofs)
    return saved,new,receipt


def test_proven_codec_graph_migration(tmp_path):
    saved,new,receipt=fixture(tmp_path)
    validate_implementation_migration(saved,new,receipt,'checkpoint')


@pytest.mark.parametrize('change',['stage','memory','budget','batch','ema','parent','missing_cold',
    'changed_source','accounting','endpoint','nonexact_gate','cold_reference','cold_failed'])
def test_rejects_unproven_codec_execution_changes(tmp_path,change):
    saved,new,receipt=fixture(tmp_path)
    if change=='stage':new['stage']='pose'
    if change=='memory':new['cuda_memory_fraction']=.1
    if change=='budget':new['steps']=49999
    if change=='batch':new['batch']=512
    if change=='ema':new['quantizer_ema_updates_per_optimizer_step']=1
    if change=='parent':new['parent_sha256']='other-root'
    if change=='missing_cold':receipt['proofs']=receipt['proofs'][:2]
    if change=='changed_source':new['source_code_sha256']['motionbricks/data/dilated_graph_context.py']='changed'
    if change=='accounting':saved['actual_exposures']-=256
    if change in ('endpoint','nonexact_gate','cold_reference','cold_failed'):
        proof=receipt['proofs'][2 if change.startswith('cold') else 0]
        path=Path(proof['path']);data=json.loads(path.read_text())
        if change=='endpoint':data['native_graph_endpoint_changes']=['model.weight']
        if change=='nonexact_gate':data['computation_gates'][0]['bitwise_equal']=False
        if change=='cold_reference':data['reference_proof_sha256']='other'
        if change=='cold_failed':data['passed']=False
        path.write_text(json.dumps(data));proof['sha256']=sha256(path)
    with pytest.raises(ValueError):validate_implementation_migration(saved,new,receipt,'checkpoint')


@pytest.mark.parametrize('stage,coalesced',[('root',True),('pose',True),('codec',False)])
def test_graph_stage_guard(stage,coalesced):
    with pytest.raises(ValueError,match='coalesced codec'):
        NativeTrainingStage(torch.nn.Module(),stage,coalesce_dilated_bias=coalesced,graph_dilated_convolutions=True)


def test_graph_context_restores_forward_after_exception():
    module=torch.nn.Conv1d(2,2,3,padding=3,dilation=3)
    data=torch.randn(1,2,8);expected=module(data)
    class UnexpectedReplay:
        def apply(self,*args):raise AssertionError('CPU must retain native forward')
    with pytest.raises(RuntimeError,match='fixture'):
        with graphed_dilated_convolutions(module,UnexpectedReplay()):
            torch.testing.assert_close(module(data),expected,atol=0,rtol=0)
            raise RuntimeError('fixture')
    assert 'forward' not in module.__dict__
    torch.testing.assert_close(module(data),expected,atol=0,rtol=0)
