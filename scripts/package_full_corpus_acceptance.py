"""Create a separate acceptance-only artifact; never grants release approval."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.full_corpus_acceptance import SCHEMA,SCENE_SCHEMA,validate_acceptance,validate_scene
from motionbricks.data.full_corpus_inference import RUNTIME_CONTRACT,compact_candidate,tensor_digest
from motionbricks.data.reference_exposure_budget import validate_completion_counts
from scripts.train_native_mb import sha256


def production_stages(a,saved):
    if a.allow_mechanics:
        if not saved['contract']['mechanics_only']:
            raise ValueError('Diagnostic opt-in requires an actual mechanics fixture')
        return []
    if a.root_stage is None or a.codec_stage is None:
        raise ValueError('Full candidate packaging requires Root and codec stage directories')
    records=[]
    for stage,folder in [('root',a.root_stage),('codec',a.codec_stage),('pose',a.checkpoint.parent)]:
        complete=json.loads((folder/'complete.json').read_text())
        weights=folder/'weights.pt'
        checkpoint_hash=sha256(weights)
        if (not complete['protected_hash_passed'] or complete['checkpoint_sha256']!=checkpoint_hash):
            raise ValueError('A completed stage identity or budget differs')
        checkpoint=torch.load(weights,map_location='cpu',mmap=True,weights_only=False)
        contract=checkpoint['contract']
        validate_completion_counts(contract,checkpoint['actual_updates'],checkpoint['actual_exposures'])
        if (complete['actual_updates']!=checkpoint['actual_updates']
                or complete['actual_exposures']!=checkpoint['actual_exposures']):
            raise ValueError('Stage completion and weights disagree on actual accounting')
        if contract['stage']!=stage or contract['mechanics_only']:
            raise ValueError('Unexpected stage schema')
        if stage=='pose' and weights.resolve()!=a.checkpoint.resolve():
            raise ValueError('The final Pose completion must identify the selected checkpoint')
        prefixes={'root':('root.','adaptation.encoder.','adaptation.root_stage1.','adaptation.root_stage2.'),
            'codec':('codec.','adaptation.decoder.','adaptation.height_encoder.')}.get(stage)
        if prefixes:
            first={k:v for k,v in checkpoint['model_state'].items() if k.startswith(prefixes)}
            final={k:v for k,v in saved['model_state'].items() if k.startswith(prefixes)}
            if tensor_digest(first)!=tensor_digest(final):
                raise ValueError('A frozen completed component differs in final Pose weights')
        records.append(dict(stage=stage,checkpoint_sha256=checkpoint_hash,parent_sha256=contract['parent_sha256'],
            actual_updates=complete['actual_updates'],actual_exposures=complete['actual_exposures'],
            budget_contract={key:contract[key] for key in ('steps','batch','micro_batch','exposure_budget') if key in contract},
            corpus_audit_sha256=contract['corpus_audit_sha256'],protected_hash_passed=True))
    return records


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    bundle_hash=sha256(a.bundle)
    compact=compact_candidate(saved,bundle_hash,allow_mechanics=a.allow_mechanics)
    stages=production_stages(a,saved)
    bundle=torch.load(a.bundle,map_location='cpu',mmap=True,weights_only=False)
    if a.scene is None:
        scene={k:torch.as_tensor(bundle['scene'][k]).cpu().numpy() for k in
            ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
        history=torch.as_tensor(bundle['initial_history']).cpu().numpy()
        origin=dict(source='bound_bundle_seed',source_sha256=bundle_hash)
    else:
        with np.load(a.scene,allow_pickle=False) as archive:
            expected={'heightfield_y_m','min_point_world_xyz_m','dx_m','initial_history','metadata_json'}
            if set(archive.files)!=expected:raise ValueError('Scene source must contain only static terrain and four-frame history')
            scene={k:archive[k].copy() for k in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
            history=archive['initial_history'].copy()
        origin=dict(source='explicit_static_scene',source_sha256=sha256(a.scene))
    metadata=dict(schema=SCENE_SCHEMA,bundle_sha256=bundle_hash,history_frames=4,
        future_motion_included=False,coordinate_frame='soma_world_y_up',provenance=origin)
    validate_scene(scene,history,metadata,bundle_hash)
    scene_path=a.output/'scene.npz'
    np.savez_compressed(scene_path,**scene,initial_history=history,metadata_json=json.dumps(metadata))
    weights=a.output/'weights.pt';torch.save(compact,weights)
    receipt=dict(schema=SCHEMA,purpose='local_acceptance_test',interactive_allowed=True,loopback_only=True,
        quality_approved=False,delivery_approved=False,checkpoint_file=weights.name,scene_file=scene_path.name,
        source_checkpoint_sha256=sha256(a.checkpoint),checkpoint_sha256=sha256(weights),
        bundle_sha256=bundle_hash,scene_sha256=sha256(scene_path),runtime=dict(RUNTIME_CONTRACT),
        model_state_sha256=tensor_digest(compact['model_state']),
        codec_state_sha256=tensor_digest({k:v for k,v in compact['model_state'].items() if k.startswith('codec.')}),
        mechanics_only=compact['contract']['mechanics_only'],actual_updates=compact['actual_updates'],
        actual_exposures=compact['actual_exposures'],training_stages=stages)
    validate_acceptance(compact,receipt,bundle_hash,receipt['checkpoint_sha256'],allow_mechanics=a.allow_mechanics)
    (a.output/'acceptance.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(packaged=True,artifact=str(a.output),mechanics_only=receipt['mechanics_only'],
        quality_approved=False,delivery_approved=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','checkpoint','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('root-stage','codec-stage','scene'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--allow-mechanics',action='store_true')
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.is_dir():(args.output/'failure.json').write_text(json.dumps(dict(error=repr(error))))
        raise
