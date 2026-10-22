from copy import deepcopy

import pytest
import torch

from motionbricks.data.full_corpus_acceptance import SCHEMA,SCENE_SCHEMA,artifact_name,validate_acceptance,validate_scene,validate_fixed_profile
from motionbricks.data.full_corpus_inference import TRAINING_SCHEMA,RUNTIME_CONTRACT,tensor_digest


def fixture():
    saved=dict(schema=TRAINING_SCHEMA,contract=dict(stage='pose',bundle_sha256='bundle',
        runtime=dict(RUNTIME_CONTRACT),mechanics_only=False,steps=50000,batch=256,micro_batch=64,
        source_quota={'pfnn':.5,'parc':.25,'bones':.25},corpus_audit_sha256='audit',cache_receipts=[{'identity':'data'}]),
        actual_updates=50000,actual_exposures=12800000,
        model_state={'root.weight':torch.ones(2),'pose._pose_token_emb.weight':torch.ones(88,32),
            'codec.quantizer.vq._codebook.embed':torch.ones(8,10,32),'adaptation.weight':torch.ones(3)})
    rows=[dict(stage=stage,checkpoint_sha256=stage+'-weights',parent_sha256=parent,
        actual_updates=50000,actual_exposures=12800000,protected_hash_passed=True,corpus_audit_sha256='audit')
        for stage,parent in [('root','prototype'),('codec','root-weights'),('pose','codec-weights')]]
    receipt=dict(schema=SCHEMA,purpose='local_acceptance_test',interactive_allowed=True,loopback_only=True,
        quality_approved=False,delivery_approved=False,bundle_sha256='bundle',checkpoint_sha256='compact',
        source_checkpoint_sha256='pose-weights',runtime=dict(RUNTIME_CONTRACT),mechanics_only=False,
        actual_updates=50000,actual_exposures=12800000,checkpoint_file='weights.pt',scene_file='scene.npz',
        scene_sha256='scene',training_stages=rows,model_state_sha256=tensor_digest(saved['model_state']),
        codec_state_sha256=tensor_digest({k:v for k,v in saved['model_state'].items() if k.startswith('codec.')}))
    return saved,receipt


def test_complete_candidate_is_acceptance_only():
    saved,receipt=fixture()
    validate_acceptance(saved,receipt,'bundle','compact')
    assert receipt['delivery_approved'] is False


def test_codec_can_bind_authorized_exposure_budget_with_fewer_adam_updates():
    from motionbricks.data.reference_exposure_budget import ExposureBudget
    saved,receipt=fixture()
    row=receipt['training_stages'][1]
    row.update(actual_updates=21500,budget_contract=dict(steps=50000,batch=1024,micro_batch=512,
        exposure_budget=ExposureBudget(12000,12000*256,1024).contract()))
    validate_acceptance(saved,receipt,'bundle','compact')
    row['actual_updates']-=1
    with pytest.raises(ValueError):validate_acceptance(saved,receipt,'bundle','compact')


def test_final_pose_can_bind_6144_exposures_without_claiming_50000_adam_calls():
    from motionbricks.data.reference_exposure_budget import ExposureBudget
    saved,receipt=fixture()
    budget=dict(steps=50000,batch=6144,micro_batch=3072,
        exposure_budget=ExposureBudget(2000,2000*256,6144).contract())
    saved['contract'].update(budget)
    saved['actual_updates']=4000
    receipt['actual_updates']=4000
    receipt['training_stages'][2].update(actual_updates=4000,budget_contract=budget)
    validate_acceptance(saved,receipt,'bundle','compact')
    saved['actual_exposures']-=6144
    receipt['actual_exposures']-=6144
    with pytest.raises(ValueError):validate_acceptance(saved,receipt,'bundle','compact')


@pytest.mark.parametrize('mutation',[
    lambda s,r:r.update(quality_approved=True),
    lambda s,r:r.update(delivery_approved=True),
    lambda s,r:r.update(loopback_only=False),
    lambda s,r:r.update(schema='motionbricks.full_corpus_inference.v1'),
    lambda s,r:r.update(checkpoint_sha256='other'),
    lambda s,r:r['runtime'].update(refinement=1),
    lambda s,r:r['training_stages'][0].update(actual_updates=49999),
    lambda s,r:r['training_stages'][1].update(parent_sha256='other'),
    lambda s,r:r['training_stages'][2].update(corpus_audit_sha256='other'),
    lambda s,r:r.update(scene_file='../scene.npz'),
    lambda s,r:s['model_state']['codec.quantizer.vq._codebook.embed'].add_(.1),
])
def test_rejects_scope_identity_stage_and_tensor_mismatch(mutation):
    saved,receipt=fixture();mutation(saved,receipt)
    with pytest.raises(ValueError):validate_acceptance(saved,receipt,'bundle','compact')


def test_mechanics_requires_separate_explicit_opt_in():
    saved,receipt=fixture()
    saved['contract'].update(mechanics_only=True,steps=2,batch=16,micro_batch=4)
    saved.update(actual_updates=2,actual_exposures=32)
    receipt.update(mechanics_only=True,actual_updates=2,actual_exposures=32,training_stages=[])
    with pytest.raises(ValueError):validate_acceptance(saved,receipt,'bundle','compact')
    validate_acceptance(saved,receipt,'bundle','compact',allow_mechanics=True)


@pytest.mark.parametrize('name',['../weights.pt','C:weights.pt','dir\\weights.pt','/weights.pt',''])
def test_artifact_files_cannot_escape_package(name):
    with pytest.raises(ValueError):artifact_name(name)


def test_scene_has_four_factual_frames_and_no_future_array():
    scene=dict(heightfield_y_m=torch.zeros(3,3),min_point_world_xyz_m=torch.zeros(3),dx_m=torch.tensor(.1))
    metadata=dict(schema=SCENE_SCHEMA,bundle_sha256='bundle',history_frames=4,
        future_motion_included=False,coordinate_frame='soma_world_y_up')
    history=torch.zeros(1,4,12)
    validate_scene(scene,history,metadata,'bundle',12)
    with pytest.raises(ValueError):validate_scene(dict(scene,future_motion=history),history,metadata,'bundle')
    with pytest.raises(ValueError):validate_scene(scene,torch.zeros(1,5,12),metadata,'bundle')
    with pytest.raises(ValueError):validate_scene(scene,history,dict(metadata,future_motion_included=True),'bundle')
    with pytest.raises(ValueError):validate_scene(scene,history,metadata,'bundle',11)


@pytest.mark.parametrize('name,value',[('num_tokens',16),('inference_steps',8),('refresh_frames',8),('path_refinements',1)])
def test_runtime_profile_cannot_drift(name,value):
    profile=dict(num_tokens=8,inference_steps=4,lookahead_m=-1.,lateral_halfwidth_m=.6,
        refresh_frames=24,lookahead_gain=1.,lookahead_base_m=.4,path_refinements=0)
    assert validate_fixed_profile(profile,profile)==profile
    with pytest.raises(ValueError):validate_fixed_profile(dict(profile,**{name:value}),profile)
