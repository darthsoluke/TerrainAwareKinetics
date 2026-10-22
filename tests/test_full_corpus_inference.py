from copy import deepcopy

import pytest
import torch

from motionbricks.data.full_corpus_inference import (
    TRAINING_SCHEMA,RUNTIME_CONTRACT,compact_candidate,inference_receipt,validate_candidate,tensor_digest,
)


def candidate():
    return dict(schema=TRAINING_SCHEMA,contract=dict(stage='pose',bundle_sha256='bundle',
        runtime=dict(RUNTIME_CONTRACT),mechanics_only=False,steps=50000,batch=256,
        corpus_audit_sha256='audit',cache_receipts=[{'identity':'data'}]),
        actual_updates=50000,actual_exposures=12800000,sample_digest='samples',
        model_state={'root.weight':torch.ones(2),'pose._pose_token_emb.weight':torch.ones(88,32),
            'codec.quantizer.vq._codebook.embed':torch.ones(8,10,32),'adaptation.weight':torch.ones(3)},
        optimizer_state={'private':1},quality_approved=False)


@pytest.mark.parametrize('change',[
    lambda value:value['contract'].update(stage='root'),
    lambda value:value['contract']['runtime'].update(refinement=1),
    lambda value:value.update(actual_updates=49999),
    lambda value:value.update(actual_exposures=12799744),
    lambda value:value['model_state'].pop('root.weight'),
    lambda value:value['contract'].update(mechanics_only=True),
    lambda value:value['model_state'].update({'codec.quantizer.vq._codebook.embed':torch.ones(8,11,32)}),
    lambda value:value['model_state'].update({'root.weight':torch.tensor([float('nan')])}),
])
def test_rejects_incomplete_or_incompatible_candidate(change):
    saved=candidate()
    change(saved)
    with pytest.raises(ValueError):
        validate_candidate(saved,'bundle')


def test_compaction_preserves_tensors_but_cannot_grant_quality_approval():
    saved=candidate()
    compact=compact_candidate(saved,'bundle')
    assert 'optimizer_state' not in compact
    assert tensor_digest(compact['model_state'])==tensor_digest(saved['model_state'])
    receipt=inference_receipt(compact,'source','compact','bundle')
    assert receipt['interactive_allowed'] is False
    assert receipt['quality_approved'] is False


def test_mechanics_requires_explicit_diagnostic_opt_in():
    saved=candidate()
    saved['contract']['mechanics_only']=True
    with pytest.raises(ValueError):
        compact_candidate(saved,'bundle')
    compact=compact_candidate(saved,'bundle',allow_mechanics=True)
    assert inference_receipt(compact,'source','compact','bundle',allow_mechanics=True)['mechanics_only']
