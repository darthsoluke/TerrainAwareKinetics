from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.train_unified_full_corpus import validate_implementation_migration, sha256


def fixture(tmp_path):
    old = dict(stage='root', steps=50000, batch=256, micro_batch=64,
        source_quota={'pfnn': .5, 'parc': .25, 'bones': .25}, source_code_sha256={'native.py': 'old'})
    new = dict(old, source_code_sha256={'motionbricks/data/dilated_bias_convolution.py': 'tested'},
        coalesced_dilated_bias=True, pose_token_memoization=False, implementation_migration_sha256='receipt')
    proofs = []
    for role in ('full_model', 'optimizer_resume'):
        path = tmp_path/(role+'.json')
        path.write_text(json.dumps({'passed': True}))
        proofs.append(dict(role=role, path=str(path), sha256=sha256(path)))
    saved = dict(contract=old, actual_updates=2000, actual_exposures=512000)
    receipt = dict(from_checkpoint_sha256='checkpoint', from_source_code_sha256=old['source_code_sha256'],
        to_source_code_sha256=new['source_code_sha256'], actual_updates=2000, proofs=proofs,
        validated_implementation_sha256='tested')
    return saved, new, receipt


def test_exact_verified_implementation_transition(tmp_path):
    saved, new, receipt = fixture(tmp_path)
    validate_implementation_migration(saved, new, receipt, 'checkpoint')


@pytest.mark.parametrize('change', ['budget', 'quota', 'checkpoint', 'source', 'implementation', 'accounting', 'proof'])
def test_migration_rejects_changed_contract_or_evidence(tmp_path, change):
    saved, new, receipt = fixture(tmp_path)
    checkpoint = 'checkpoint'
    if change == 'budget': new['steps'] = 50001
    if change == 'quota': new['source_quota'] = {'pfnn': .6, 'parc': .2, 'bones': .2}
    if change == 'checkpoint': checkpoint = 'other'
    if change == 'source': receipt = dict(receipt, from_source_code_sha256={'native.py': 'other'})
    if change == 'implementation': receipt['validated_implementation_sha256'] = 'other'
    if change == 'accounting': saved['actual_exposures'] -= 256
    if change == 'proof': receipt['proofs'] = receipt['proofs'][:1]
    with pytest.raises(ValueError):
        validate_implementation_migration(saved, new, receipt, checkpoint)


def parallel_fixture(tmp_path):
    old = dict(stage='root',steps=50000,batch=256,micro_batch=64,
        source_quota={'pfnn':.5,'parc':.25,'bones':.25},coalesced_dilated_bias=True,
        pose_token_memoization=False,source_code_sha256={'native.py':'old'},
        implementation_migration_sha256='previous')
    new = dict(old,source_code_sha256={'motionbricks/data/parallel_root_training.py':'tested'},
        cuda_memory_fraction=.1,root_parallel_devices=2,implementation_migration_sha256='new')
    proofs=[]
    for role,result in [('memory_equivalence',dict(passed=True,tolerances_unchanged=True)),
            ('parallel_optimizer_resume',dict(passed=True,rng_schedule_from_reference=False,implementation_sha256='tested'))]:
        path=tmp_path/(role+'.json');path.write_text(json.dumps(result))
        proofs.append(dict(role=role,path=str(path),sha256=sha256(path)))
    saved=dict(contract=old,actual_updates=6000,actual_exposures=1536000)
    receipt=dict(transition='root-parallel-memory-v1',from_checkpoint_sha256='checkpoint',
        from_source_code_sha256=old['source_code_sha256'],to_source_code_sha256=new['source_code_sha256'],
        actual_updates=6000,parallel_implementation_sha256='tested',proofs=proofs)
    return saved,new,receipt


def test_parallel_execution_migration_keeps_original_budget(tmp_path):
    saved,new,receipt=parallel_fixture(tmp_path)
    validate_implementation_migration(saved,new,receipt,'checkpoint')


@pytest.mark.parametrize('change',['budget','batch','quota','memory','devices','implementation','missing_proof','reference_rng'])
def test_parallel_transition_rejects_unvalidated_changes(tmp_path,change):
    saved,new,receipt=parallel_fixture(tmp_path)
    if change=='budget':new['steps']=50001
    if change=='batch':new['batch']=512
    if change=='quota':new['source_quota']={'pfnn':1.}
    if change=='memory':new['cuda_memory_fraction']=.2
    if change=='devices':new['root_parallel_devices']=3
    if change=='implementation':receipt['parallel_implementation_sha256']='other'
    if change=='missing_proof':receipt['proofs']=receipt['proofs'][:1]
    if change=='reference_rng':
        proof=receipt['proofs'][1];path=Path(proof['path'])
        path.write_text(json.dumps(dict(passed=True,rng_schedule_from_reference=True,implementation_sha256='tested')))
        proof['sha256']=sha256(path)
    with pytest.raises(ValueError):
        validate_implementation_migration(saved,new,receipt,'checkpoint')


def four_fixture(tmp_path):
    saved,new,receipt=parallel_fixture(tmp_path)
    old=saved['contract']
    old.update(cuda_memory_fraction=.1,root_parallel_devices=2,
        source_code_sha256={'motionbricks/data/parallel_root_training.py':'deployed'})
    new.update(root_parallel_devices=4)
    receipt.update(transition='root-parallel-four-v1',from_source_code_sha256=old['source_code_sha256'])
    result=dict(passed=True,rng_schedule_from_reference=False,implementation_sha256='tested',
        reference_devices=2,target_devices=4,resume_from_reference_devices=2,
        reference_implementation_sha256='deployed',logical_cuda_rng_canonicalized=True,
        serial_errors=[],resume_errors=[])
    path=tmp_path/'four.json';path.write_text(json.dumps(result))
    receipt['proofs']=[dict(role='parallel_optimizer_resume',path=str(path),sha256=sha256(path))]
    return saved,new,receipt,result


def test_four_device_transition_binds_deployed_reference(tmp_path):
    saved,new,receipt,_=four_fixture(tmp_path)
    validate_implementation_migration(saved,new,receipt,'checkpoint')


@pytest.mark.parametrize('change',['old_memory','memory','budget','batch','old_devices','new_devices',
    'reference','resume_topology','rng','model_difference','resume_difference'])
def test_four_device_migration_rejects_scope_or_continuity_drift(tmp_path,change):
    saved,new,receipt,proof=four_fixture(tmp_path)
    if change=='old_memory':saved['contract']['cuda_memory_fraction']=1.
    if change=='memory':new['cuda_memory_fraction']=.2
    if change=='budget':new['steps']=49999
    if change=='batch':new['batch']=512
    if change=='old_devices':saved['contract']['root_parallel_devices']=1
    if change=='new_devices':new['root_parallel_devices']=3
    if change=='reference':proof['reference_implementation_sha256']='other'
    if change=='resume_topology':proof['resume_from_reference_devices']=4
    if change=='rng':proof['logical_cuda_rng_canonicalized']=False
    if change=='model_difference':proof['serial_errors']=['model.weight']
    if change=='resume_difference':proof['resume_errors']=['optimizer.state']
    path=Path(receipt['proofs'][0]['path']);path.write_text(json.dumps(proof))
    receipt['proofs'][0]['sha256']=sha256(path)
    with pytest.raises(ValueError):validate_implementation_migration(saved,new,receipt,'checkpoint')
