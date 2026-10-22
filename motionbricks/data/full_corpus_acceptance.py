"""Independent model/scene binding for loopback-only acceptance testing."""

import torch

from .full_corpus_inference import RUNTIME_CONTRACT, tensor_digest, validate_candidate
from .reference_exposure_budget import validate_completion_counts


SCHEMA = 'motionbricks.full_corpus_acceptance.v1'
SCENE_SCHEMA = 'motionbricks.full_corpus_scene.v1'


def artifact_name(value):
    if (not isinstance(value,str) or value in ('','.','..')
            or any(char in value for char in ('/','\\',':','\x00'))):
        raise ValueError('Artifact files must be plain relative basenames')
    return value


def validate_acceptance(saved,receipt,bundle_hash,checkpoint_hash,*,allow_mechanics=False):
    contract=validate_candidate(saved,bundle_hash,allow_mechanics=allow_mechanics)
    if (receipt.get('schema')!=SCHEMA or receipt.get('purpose')!='local_acceptance_test'
            or receipt.get('interactive_allowed') is not True or receipt.get('loopback_only') is not True
            or receipt.get('delivery_approved') is not False or receipt.get('quality_approved') is not False
            or receipt.get('bundle_sha256')!=bundle_hash or receipt.get('checkpoint_sha256')!=checkpoint_hash
            or receipt.get('runtime')!=RUNTIME_CONTRACT
            or receipt.get('mechanics_only')!=contract['mechanics_only']
            or receipt.get('actual_updates')!=saved['actual_updates']
            or receipt.get('actual_exposures')!=saved['actual_exposures']):
        raise ValueError('Acceptance identity, scope or runtime contract differs')
    artifact_name(receipt.get('checkpoint_file'))
    artifact_name(receipt.get('scene_file'))
    if not receipt.get('scene_sha256'):
        raise ValueError('Acceptance artifact must bind its static scene and factual seed')
    if not contract['mechanics_only']:
        if (contract['steps']!=50000 or (not contract.get('exposure_budget') and
                (contract['batch']!=256 or contract['micro_batch']!=64))
                or contract['source_quota']!={'pfnn':.5,'parc':.25,'bones':.25}):
            raise ValueError('The authorized full training budget is required')
        stages=receipt.get('training_stages',[])
        if (not isinstance(stages,list) or not all(isinstance(row,dict) for row in stages)
                or [row.get('stage') for row in stages]!=['root','codec','pose']):
            raise ValueError('All three completed training stages must be bound')
        for row in stages:
            budget=row.get('budget_contract',dict(steps=50000,batch=256,micro_batch=64))
            if not budget.get('exposure_budget') and (budget.get('steps')!=50000
                    or budget.get('batch')!=256 or budget.get('micro_batch')!=64):
                raise ValueError('Unapproved legacy training budget')
            validate_completion_counts(budget,row.get('actual_updates'),row.get('actual_exposures'))
            if (row.get('actual_exposures')!=12800000
                    or row.get('protected_hash_passed') is not True
                    or row.get('corpus_audit_sha256')!=contract['corpus_audit_sha256']
                    or not row.get('checkpoint_sha256')):
                raise ValueError('Completed stage budget or audited corpus identity differs')
        if (stages[1].get('parent_sha256')!=stages[0]['checkpoint_sha256']
                or stages[2].get('parent_sha256')!=stages[1]['checkpoint_sha256']
                or stages[2]['checkpoint_sha256']!=receipt.get('source_checkpoint_sha256')):
            raise ValueError('Root/codec/Pose dependency chain differs')
    elif receipt.get('training_stages'):
        raise ValueError('A mechanics fixture cannot claim completed production stages')
    if tensor_digest(saved['model_state'])!=receipt.get('model_state_sha256'):
        raise ValueError('Complete model tensor digest differs')
    codec={k:v for k,v in saved['model_state'].items() if k.startswith('codec.')}
    if tensor_digest(codec)!=receipt.get('codec_state_sha256'):
        raise ValueError('Final codec tensor digest differs')
    return contract


def load_acceptance_candidate(controller,saved,receipt,bundle_hash,checkpoint_hash,*,allow_mechanics=False):
    validate_acceptance(saved,receipt,bundle_hash,checkpoint_hash,allow_mechanics=allow_mechanics)
    controller.load_state_dict(saved['model_state'],strict=True)
    controller.codebook.copy_(controller.codec.get_codebook())
    controller.eval().requires_grad_(False)
    if tensor_digest(controller.state_dict())!=receipt['model_state_sha256']:
        raise ValueError('Loaded model differs from the acceptance artifact')
    inference=controller.inference
    if (controller.codec is not inference._vqvae_pose_model
            or controller.codec is not inference._pose_model.supporting_nets['pose_net']
            or controller.root is not inference._root_model.backbone_net
            or controller.pose is not inference._pose_model.backbone_net):
        raise ValueError('Runtime does not reference every loaded native component')
    torch.testing.assert_close(controller.codebook,controller.codec.get_codebook(),atol=0,rtol=0)


def validate_scene(scene,history,metadata,bundle_hash,feature_width=None):
    required={'heightfield_y_m','min_point_world_xyz_m','dx_m'}
    if not isinstance(scene,dict) or set(scene)!=required:
        raise ValueError('Only the declared static scene arrays are allowed')
    if (not isinstance(metadata,dict) or metadata.get('schema')!=SCENE_SCHEMA or metadata.get('bundle_sha256')!=bundle_hash
            or metadata.get('history_frames')!=4 or metadata.get('future_motion_included') is not False
            or metadata.get('coordinate_frame')!='soma_world_y_up'):
        raise ValueError('Scene must bind the same normalization and four factual world frames')
    history=torch.as_tensor(history)
    if (history.ndim!=3 or tuple(history.shape[:2])!=(1,4) or history.shape[2]<5
            or (feature_width is not None and history.shape[2]!=feature_width)
            or not torch.isfinite(history).all()):
        raise ValueError('Exactly four finite matching history frames are required')
    height=torch.as_tensor(scene['heightfield_y_m']);origin=torch.as_tensor(scene['min_point_world_xyz_m'])
    dx=torch.as_tensor(scene['dx_m'])
    if (height.ndim!=2 or min(height.shape)<1 or not torch.isfinite(height).any()
            or origin.shape!=(3,) or not torch.isfinite(origin).all()
            or dx.ndim!=0 or not torch.isfinite(dx) or float(dx)<=0):
        raise ValueError('Static terrain arrays are invalid')


def validate_fixed_profile(values,expected):
    from .terrain_runtime_controls import validate_runtime_controls
    actual=validate_runtime_controls(values)
    if actual!=expected:
        raise ValueError('Acceptance testing uses the fixed trained runtime profile')
    return actual
