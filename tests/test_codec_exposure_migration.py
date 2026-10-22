import json

import pytest

from motionbricks.data.codec_exposure_migration import file_sha,validate_codec_exposure_migration
from motionbricks.data.reference_exposure_budget import ExposureBudget


def fixture(tmp_path,four=False):
    source=tmp_path/'source/motionbricks/data/exposure_codec_training.py'
    source.parent.mkdir(parents=True);source.write_text('verified EMA chunks')
    sources={'motionbricks/data/exposure_codec_training.py':file_sha(source),
        'motionbricks/data/ordered_codec_pipeline.py':'pipeline',
        'motionbricks/data/dilated_graph_convolution.py':'graph'}
    old=dict(stage='codec',steps=50000,batch=256,micro_batch=64,cuda_memory_fraction=.125,
        graph_dilated_convolutions=True,quantizer_ema_updates_per_optimizer_step=4,
        source_code_sha256={'native':'old'},parent_sha256='root',max_exposures=12800000)
    if four:old.update(batch=1024,micro_batch=512,cuda_memory_fraction=.9,codec_pipeline_devices=2,
        quantizer_ema_updates_per_optimizer_step=16,exposure_budget=ExposureBudget(10000,10000*256,1024).contract())
    exposures=(18000 if four else 12000)*256;batch=2048 if four else 1024;devices=4 if four else 2
    saved=dict(contract=old,actual_updates=12000,actual_exposures=exposures)
    new=dict(old,batch=batch,micro_batch=512,cuda_memory_fraction=.9,codec_pipeline_devices=devices,
        quantizer_ema_updates_per_optimizer_step=batch//64,source_code_sha256=sources,
        exposure_budget=ExposureBudget(12000,exposures,batch).contract())
    common=dict(passed=True,exposure_mode=True,global_batch=batch,compute_micro=512,execution_devices=devices,
        ema_chunk_size=64,reference_units_per_update=batch//256,production_budget_updates=0)
    proofs=[]
    def proof(role,value):
        path=tmp_path/role/'complete.json';path.parent.mkdir();path.write_text(json.dumps(value))
        row=dict(role=role,path=str(path),sha256=file_sha(path),source_directory=str(tmp_path/'source'))
        proofs.append(row)
        return row
    reference=proof('reference',dict(common,serial_reference=True,actual_optimizer_updates=4,
        validation_between=True,reference_memory_fraction=.9))
    continuous=proof('continuous',dict(common,cold_process=True,cold_start_step=0,pipeline_memory_fraction=.9,
        source_validation_between=True,endpoint_changes=[],micro_changes=[[],[],[],[]],actual_optimizer_updates=4,
        implementation_sha256='pipeline',candidate_graph_sha256='graph',reference_report_sha256=reference['sha256']))
    midpoint=tmp_path/'continuous/pipeline_midpoint.pt';midpoint.write_bytes(b'complete midpoint')
    proof('cold_resume',dict(common,cold_process=True,cold_start_step=2,pipeline_memory_fraction=.9,
        source_validation_between=True,endpoint_changes=[],micro_changes=[[],[]],actual_optimizer_updates=2,
        implementation_sha256='pipeline',candidate_graph_sha256='graph',reference_report_sha256=reference['sha256'],
        resume_state_sha256=file_sha(midpoint)))
    receipt=dict(actual_updates=12000,actual_exposures=exposures,
        transition='codec-reference-exposure-four-v1' if four else 'codec-reference-exposure-v1',
        from_source_code_sha256=old['source_code_sha256'],to_source_code_sha256=sources,
        authorization='user_requested_multigpu_large_memory_reference_50000',optimizer_trajectory_equivalent=False,proofs=proofs)
    return saved,new,receipt


@pytest.mark.parametrize('four',[False,True])
def test_complete_authorized_exposure_transition(tmp_path,four):
    validate_codec_exposure_migration(*fixture(tmp_path,four))


@pytest.mark.parametrize('change',['batch','micro','memory','ema','parent','quota','count','exposures',
    'authorization','trajectory','missing_cold','failed','validation','endpoint','gradient','resume_hash','source'])
@pytest.mark.parametrize('four',[False,True])
def test_rejects_unsupported_migration(tmp_path,change,four):
    saved,new,receipt=fixture(tmp_path,four)
    if change in ('batch','micro','memory','ema'):
        key={'batch':'batch','micro':'micro_batch','memory':'cuda_memory_fraction','ema':'quantizer_ema_updates_per_optimizer_step'}[change]
        new[key]=0
    if change=='parent':new['parent_sha256']='another'
    if change=='quota':new['source_quota']={'pfnn':1}
    if change=='count':receipt['actual_updates']-=1
    if change=='exposures':receipt['actual_exposures']-=256
    if change=='authorization':receipt['authorization']='not authorized'
    if change=='trajectory':receipt['optimizer_trajectory_equivalent']=True
    if change=='missing_cold':receipt['proofs'].pop()
    if change in ('failed','validation','endpoint','gradient','resume_hash','source'):
        row=receipt['proofs'][-1]
        from pathlib import Path
        path=Path(row['path']);data=json.loads(path.read_text())
        if change=='failed':data['passed']=False
        if change=='validation':data['source_validation_between']=False
        if change=='endpoint':data['endpoint_changes']=['changed']
        if change=='gradient':data['micro_changes'][0]=['changed']
        if change=='resume_hash':data['resume_state_sha256']='wrong'
        if change=='source':data['implementation_sha256']='changed'
        path.write_text(json.dumps(data));row['sha256']=file_sha(path)
    with pytest.raises(ValueError):validate_codec_exposure_migration(saved,new,receipt)
