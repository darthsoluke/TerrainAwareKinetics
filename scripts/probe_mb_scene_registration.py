"""Check reference-chart translation invariance with the frozen model on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import motionbricks.mb_sonic.mb_reference as module
from motionbricks.mb_sonic.reference import NativeSomaHistory
from motionbricks.mb_sonic.scene import load_scene
from scripts.isolate_mb_continuous_reference import save_json
from scripts.probe_mb_intent_stages import legacy_scene_reframe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Evidence already exists')
    config=json.loads(args.config.read_text())
    scene=load_scene(Path(config['scene_root'])/'scene/scene_manifest.json')
    with np.load(scene.root/scene.manifest.source['scene']['file'],allow_pickle=False) as archive:
        seed=archive['initial_history'][0].copy()
    model=module.TrainedMBReferenceAdapter.load(config['mb_artifact'],config['mb_bundle'],canonical_scene=scene,device='cpu')
    original_frame,original_audit=module.reframe_scene_inputs,module.audit_legacy_observation
    legacy_frame=legacy_scene_reframe()
    origin=seed[-1,:3].copy()
    outputs=[]
    with torch.inference_mode():
        for name,shift,registered in (('baseline',np.zeros(3),True),
                ('shifted_legacy',np.array([8.,0.,-6.]),False),('shifted_registered',np.array([8.,0.,-6.]),True)):
            features=seed.copy();features[:,:3]+=shift
            history=NativeSomaHistory((np.arange(4)-3)/30,features,scene.manifest.scene_id,
                dict(kind='legacy_reference_history',counterfactual='reference_chart_translation_only'))
            module.reframe_scene_inputs=original_frame if registered else legacy_frame
            def legacy_audit(scene,h,local,volume,**kwargs):
                return original_audit(scene,h,local,volume)
            module.audit_legacy_observation=original_audit if registered else legacy_audit
            result=model.generate(history,velocity_world_xz_mps=[0.,.8],heading_world_rad=0.,
                command_id=name,blocks=1,conditioning_mode='B1',query_origin_soma_xyz=origin,
                output_frame='reference_chart')
            engine=model._native_prepared
            h=torch.tensor(features)[None];controls=h.new_tensor([[0.,.8,1.,0.]])
            values=engine.scene_inputs(h,engine._goal(h,controls),origin)
            outputs.append(dict(name=name,translation=shift,root=result.source.root_positions_m-shift,
                joints=result.source.world_positions_m-shift,tokens=engine._captured_tokens.cpu().numpy().copy(),
                patch=values[0].cpu().numpy(),volume=values[12].cpu().numpy(),
                query=values[2].cpu().numpy()-(shift[[0,2]] if registered else 0),
                audit=result.raw_blocks[0].provenance['geometry_audit']))
    base=outputs[0];rows=[]
    for result in outputs:
        rows.append(dict(name=result['name'],translation=result['translation'].tolist(),
            tokens_equal=bool(np.array_equal(result['tokens'],base['tokens'])),
            root_error_m=float(np.linalg.norm(result['root']-base['root'],axis=-1).max()),
            joint_error_m=float(np.linalg.norm(result['joints']-base['joints'],axis=-1).max()),
            patch_input_error_m=float(np.abs(result['patch']-base['patch']).max()),
            volume_feature_error=float(np.abs(result['volume']-base['volume']).max()),
            physical_query_error_m=float(np.abs(result['query']-base['query']).max()),audit=result['audit']))
    save_json(args.output,dict(schema='motionbricks.scene_registration_probe.v1',device='cpu',
        optimizer_updates=0,new_physics_steps=0,rows=rows))
    print(json.dumps([{k:v for k,v in row.items() if k!='audit'} for row in rows],indent=2))


if __name__=='__main__':main()
