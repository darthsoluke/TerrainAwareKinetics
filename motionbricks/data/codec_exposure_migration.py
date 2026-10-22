"""Admission for the user-authorized sample-equivalent codec continuation."""

import hashlib
import json
from pathlib import Path

from .reference_exposure_budget import ExposureBudget


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def validate_codec_exposure_migration(saved,contract,receipt):
    old=saved['contract']
    four=receipt.get('transition')=='codec-reference-exposure-four-v1'
    devices=4 if four else 2
    batch=512*devices
    if (old['stage']!='codec' or contract['stage']!='codec' or old['batch']!=(1024 if four else 256)
            or old['micro_batch']!=(512 if four else 64) or old['cuda_memory_fraction']!=(.9 if four else .125)
            or contract['batch']!=batch or contract['micro_batch']!=512
            or contract['cuda_memory_fraction']!=.9 or contract.get('codec_pipeline_devices')!=devices
            or not old.get('graph_dilated_convolutions') or not contract.get('graph_dilated_convolutions')
            or old.get('quantizer_ema_updates_per_optimizer_step')!=(16 if four else 4)
            or contract['quantizer_ema_updates_per_optimizer_step']!=batch//64):
        raise ValueError('Unexpected exposure-equivalent codec transition')
    changed={'batch','micro_batch','cuda_memory_fraction','codec_pipeline_devices','exposure_budget',
             'quantizer_ema_updates_per_optimizer_step','source_code_sha256','implementation_migration_sha256'}
    if {k:v for k,v in old.items() if k not in changed}!={k:v for k,v in contract.items() if k not in changed}:
        raise ValueError('Exposure transition changed another learning or data contract')
    if four:
        if old.get('codec_pipeline_devices')!=2 or not old.get('exposure_budget'):
            raise ValueError('The four-GPU transition requires the proven two-GPU producer')
        prior=old['exposure_budget']
        ExposureBudget(prior['initial_updates'],prior['initial_exposures'],1024,saved['actual_updates'],saved['actual_exposures'])
    elif saved['actual_exposures']!=saved['actual_updates']*256:
        raise ValueError('Legacy source accounting differs')
    expected=ExposureBudget(saved['actual_updates'],saved['actual_exposures'],batch).contract()
    if (contract['exposure_budget']!=expected or saved['actual_exposures']%batch
            or receipt.get('actual_updates')!=saved['actual_updates']
            or receipt.get('actual_exposures')!=saved['actual_exposures']
            or receipt.get('from_source_code_sha256')!=old['source_code_sha256']
            or receipt.get('to_source_code_sha256')!=contract['source_code_sha256']
            or receipt.get('authorization')!='user_requested_multigpu_large_memory_reference_50000'
            or receipt.get('optimizer_trajectory_equivalent') is not False):
        raise ValueError('Exposure authorization, source or accounting differs')
    proofs={}
    for item in receipt.get('proofs',[]):
        if item['role'] in proofs or file_sha(item['path'])!=item['sha256']:
            raise ValueError('Execution proof identity differs')
        result=json.loads(Path(item['path']).read_text())
        if (not result.get('passed') or not result.get('exposure_mode') or result.get('global_batch')!=batch
                or result.get('compute_micro')!=512 or result.get('ema_chunk_size')!=64
                or result.get('reference_units_per_update')!=batch//256 or result.get('production_budget_updates')!=0
                or (four and result.get('execution_devices')!=4)):
            raise ValueError('Large-batch execution is unproven')
        if item['role']=='reference':
            if (not result.get('serial_reference') or result.get('actual_optimizer_updates')!=4
                    or not result.get('validation_between') or result.get('reference_memory_fraction')!=.9):
                raise ValueError('Serial exposure reference differs')
        else:
            count,start=(4,0) if item['role']=='continuous' else (2,2)
            if (not result.get('cold_process') or result.get('actual_optimizer_updates')!=count
                    or result.get('cold_start_step')!=start or result.get('pipeline_memory_fraction')!=.9
                    or not result.get('source_validation_between') or result.get('endpoint_changes')!=[]
                    or result.get('micro_changes')!=[[] for _ in range(count)]):
                raise ValueError('Cold optimizer, validation and exact computation proof required')
            for field,name in [('implementation_sha256','ordered_codec_pipeline.py'),
                               ('candidate_graph_sha256','dilated_graph_convolution.py')]:
                if result.get(field)!=contract['source_code_sha256'].get('motionbricks/data/'+name):
                    raise ValueError('Execution source differs from its proof')
        source=Path(item['source_directory'])/'motionbricks/data/exposure_codec_training.py'
        if file_sha(source)!=contract['source_code_sha256'].get('motionbricks/data/exposure_codec_training.py'):
            raise ValueError('EMA chunk implementation differs from its proof')
        proofs[item['role']]=(item,result)
    if set(proofs)!={'reference','continuous','cold_resume'}:
        raise ValueError('All exposure execution and continuation proofs are required')
    reference,continuous,resume=(proofs[k] for k in ('reference','continuous','cold_resume'))
    if (continuous[1]['reference_report_sha256']!=reference[0]['sha256']
            or resume[1]['reference_report_sha256']!=reference[0]['sha256']
            or resume[1]['resume_state_sha256']!=file_sha(Path(continuous[0]['path']).parent/'pipeline_midpoint.pt')):
        raise ValueError('Proof reference or midpoint chain differs')
