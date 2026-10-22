"""Strict offline loading of full-corpus candidates without release approval."""

import hashlib

import torch

from .reference_exposure_budget import validate_completion_counts


TRAINING_SCHEMA = 'motionbricks.unified.full_corpus.v1'
INFERENCE_SCHEMA = 'motionbricks.full_corpus_inference.v1'
RUNTIME_CONTRACT = dict(tokens=8, pose_passes=4, history_frames=4, played_frames=24,
    packet_frames=8, fps=30, refinement=0, postprocessing=None)


def tensor_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        if not isinstance(name,str) or not torch.is_tensor(value):
            raise ValueError('Expected named model tensors')
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_candidate(saved,bundle_sha256,*,allow_mechanics=False):
    contract=saved.get('contract',{})
    if (saved.get('schema')!=TRAINING_SCHEMA or contract.get('stage')!='pose'
            or contract.get('bundle_sha256')!=bundle_sha256):
        raise ValueError('Completed full-corpus Pose checkpoint and matching bundle required')
    if contract.get('runtime')!=RUNTIME_CONTRACT:
        raise ValueError('Training/runtime profile differs')
    mechanics=contract.get('mechanics_only')
    if not isinstance(mechanics,bool) or (mechanics and not allow_mechanics):
        raise ValueError('Mechanics fixture is not a full-corpus candidate')
    updates,batch=contract.get('steps'),contract.get('batch')
    if type(updates) is not int or type(batch) is not int or updates<1 or batch<1:
        raise ValueError('Only completed optimizer/exposure accounting can be loaded')
    validate_completion_counts(contract,saved.get('actual_updates'),saved.get('actual_exposures'))
    if not mechanics and (not contract.get('corpus_audit_sha256') or not contract.get('cache_receipts')):
        raise ValueError('Full-corpus checkpoint must bind its audited data')
    state=saved.get('model_state',{})
    if not isinstance(state,dict) or not all(any(key.startswith(prefix) for key in state)
            for prefix in ('root.','pose.','codec.','adaptation.')):
        raise ValueError('All four native model components are required')
    for value in state.values():
        if not torch.is_tensor(value) or (value.is_floating_point() and not torch.isfinite(value).all()):
            raise ValueError('Model state contains nonfinite or nontensor values')
    codebook=state.get('codec.quantizer.vq._codebook.embed')
    embedding=state.get('pose._pose_token_emb.weight')
    if (codebook is None or embedding is None or codebook.ndim!=3
            or embedding.shape!=(codebook.shape[0]*(codebook.shape[1]+1),codebook.shape[2])):
        raise ValueError('Pose token vocabulary is incompatible with the current codec')
    return contract


def compact_candidate(saved,bundle_sha256,*,allow_mechanics=False):
    validate_candidate(saved,bundle_sha256,allow_mechanics=allow_mechanics)
    keys=('schema','contract','model_state','actual_updates','actual_exposures','sample_digest')
    result={key:saved[key] for key in keys}
    result['quality_approved']=False
    return result


def inference_receipt(saved,source_sha256,compact_sha256,bundle_sha256,*,allow_mechanics=False):
    contract=validate_candidate(saved,bundle_sha256,allow_mechanics=allow_mechanics)
    return dict(schema=INFERENCE_SCHEMA,source_checkpoint_sha256=source_sha256,
        checkpoint_sha256=compact_sha256,bundle_sha256=bundle_sha256,
        model_state_sha256=tensor_digest(saved['model_state']),
        codec_state_sha256=tensor_digest({key:value for key,value in saved['model_state'].items() if key.startswith('codec.')}),
        runtime=contract['runtime'],mechanics_only=contract['mechanics_only'],
        actual_updates=saved['actual_updates'],actual_exposures=saved['actual_exposures'],
        purpose='offline_evaluation',interactive_allowed=False,quality_approved=False)


def load_candidate(controller,saved,receipt,bundle_sha256,*,allow_mechanics=False):
    contract=validate_candidate(saved,bundle_sha256,allow_mechanics=allow_mechanics)
    if (receipt.get('schema')!=INFERENCE_SCHEMA or receipt.get('purpose')!='offline_evaluation'
            or receipt.get('interactive_allowed') is not False or receipt.get('quality_approved') is not False
            or receipt.get('bundle_sha256')!=bundle_sha256 or receipt.get('runtime')!=RUNTIME_CONTRACT
            or receipt.get('mechanics_only')!=contract['mechanics_only']
            or receipt.get('actual_updates')!=saved['actual_updates']
            or receipt.get('actual_exposures')!=saved['actual_exposures']):
        raise ValueError('Offline inference receipt differs from the checkpoint contract')
    state=saved['model_state']
    if tensor_digest(state)!=receipt.get('model_state_sha256'):
        raise ValueError('Complete model tensor digest differs')
    codec={key:value for key,value in state.items() if key.startswith('codec.')}
    if tensor_digest(codec)!=receipt.get('codec_state_sha256'):
        raise ValueError('Codec tensor digest differs')
    controller.load_state_dict(state,strict=True)
    controller.codebook.copy_(controller.codec.get_codebook())
    controller.eval().requires_grad_(False)
    if tensor_digest(controller.state_dict())!=receipt['model_state_sha256']:
        raise ValueError('Loaded full model changed during inference construction')
    inference=controller.inference
    if (controller.codec is not inference._vqvae_pose_model
            or controller.codec is not inference._pose_model.supporting_nets['pose_net']
            or controller.root is not inference._root_model.backbone_net
            or controller.pose is not inference._pose_model.backbone_net):
        raise ValueError('Runtime models do not reference the loaded components')
    torch.testing.assert_close(controller.codebook,controller.codec.get_codebook(),atol=0,rtol=0)
