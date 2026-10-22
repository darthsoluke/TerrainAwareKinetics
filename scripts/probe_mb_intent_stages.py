"""Frozen CPU Root/codec comparison on saved continuous-gamepad histories."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.mb_sonic.mb_reference import TrainedMBReferenceAdapter,world_xy_command_to_soma
from motionbricks.mb_sonic.reference import NativeSomaHistory,file_sha256
from motionbricks.mb_sonic.scene import load_scene
from scripts.audit_mb_tracker_binary import read_index
from scripts.isolate_mb_continuous_reference import load_record,save_json


def legacy_scene_reframe():
    path=Path(__file__).resolve().parents[1]/'out/mb_sonic_e0_e3_20261018/interactive_clock_v48/source/mb_reference.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reframe_scene_inputs')
    scope={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['reframe_scene_inputs']


def trajectory(values):
    values=np.asarray(values)
    velocity=np.diff(values[:,[0,2]],axis=0)*30
    return dict(mean_velocity_xz_mps=((values[27,[0,2]]-values[3,[0,2]])/.8).tolist(),
        mean_speed_mps=float(np.linalg.norm(velocity[3:27],axis=-1).mean()),
        max_speed_mps=float(np.linalg.norm(velocity[3:27],axis=-1).max()),
        root=values[:32,:5].tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bounded-root',action='store_true')
    parser.add_argument('--legacy-scene',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Probe evidence already exists')
    config=json.loads((args.trial/'config.json').read_text())
    recording=next((args.trial/'recording').glob('episode_0001_*.jsonl'))
    _,_,artifacts,_,_=read_index(recording)
    plans=[e for e in map(json.loads,recording.read_text().splitlines()) if e['kind']=='reference_packet_adopted']
    chosen=[]
    for time in (6.,20.,25.,72.,100.):
        for event in plans:
            if event['sim_time_s']<time:continue
            item=artifacts[event['plan_id']]
            if file_sha256(item['path'])!=item['sha256']:raise ValueError('Source changed')
            record=load_record(item['path'])
            if record['raw_blocks']:
                chosen.append((event,record,item));break
    scene=load_scene(Path(config['scene_root'])/'scene/scene_manifest.json')
    model=TrainedMBReferenceAdapter.load(config['mb_artifact'],config['mb_bundle'],canonical_scene=scene,device='cpu')
    registration={}
    if args.legacy_scene:
        import motionbricks.mb_sonic.mb_reference as reference_module
        original_audit=reference_module.audit_legacy_observation
        reference_module.reframe_scene_inputs=legacy_scene_reframe()
        def legacy_audit(scene,history,local,volume,**kwargs):
            return original_audit(scene,history,local,volume)
        reference_module.audit_legacy_observation=legacy_audit
    inference=model.controller.inference
    planned_capture={}
    if args.bounded_root:
        from motionbricks.mb_sonic.root_plan import plan_root_conditions
        native_root=inference._predict_root_trajectories
        def planned_root(batch,config):
            count,root,local=native_root(batch,config)
            root,local=plan_root_conditions(inference,dict(batch,pred_num_tokens=count),root)
            planned_capture['root']=root
            return count,root,local
        inference._predict_root_trajectories=planned_root
    rows=[]
    with torch.inference_mode():
        for event,record,item in chosen:
            history=NativeSomaHistory(**record['conditioning_history'])
            heading=float(np.arctan2(history.features[-1,4],history.features[-1,3]))
            velocity,_=world_xy_command_to_soma(record['command']['velocity_xy_mps'],0.)
            origin=history.provenance['scene_query_origin_soma_xyz_m']
            registration['translation']=np.asarray(origin)-history.features[-1,:3]
            generated=model.generate(history,velocity_world_xz_mps=velocity,heading_world_rad=heading,
                command_id='frozen_stage_probe',blocks=1,conditioning_mode='B1',query_origin_soma_xyz=origin,
                output_frame='reference_chart')
            root=model.controller.inference.global_motion_rep.unnormalize(
                model.controller.last_root['pred_global_root_values'])[0].cpu().numpy()
            decoded=generated.raw_blocks[0].raw_features
            row=dict(source_plan=item,source_time_s=event['sim_time_s'],
                requested_velocity_xz_mps=velocity.tolist(),heading_condition='hold_saved_history_heading',
                query_translation_soma_xyz_m=registration['translation'].tolist(),
                history_mean_velocity_xz_mps=((history.features[-1,[0,2]]-history.features[0,[0,2]])*10).tolist(),
                root_network=trajectory(root),decoded=trajectory(decoded),
                decoder_past_error_xyz_m=(decoded[3,:3]-history.features[3,:3]).tolist())
            rows.append(row)
            if args.bounded_root:
                row['planned_root']=trajectory(inference.global_motion_rep.unnormalize(planned_capture['root'])[0].cpu().numpy())
            print(json.dumps(dict(time_s=row['source_time_s'],intent=velocity.tolist(),
                root_mean_mps=row['root_network']['mean_speed_mps'],
                decoded_mean_mps=row['decoded']['mean_speed_mps'],
                decoded_velocity=row['decoded']['mean_velocity_xz_mps'])),flush=True)
    save_json(args.output,dict(schema='motionbricks.intent_stage_probe.v1',optimizer_updates=0,
        physical_steps=0,device='cpu',cases=rows,model_identity=model.identity,
        bounded_root=args.bounded_root,registered_scene=not args.legacy_scene,
        acceleration_mps2=2.5 if args.bounded_root else None))


if __name__=='__main__':main()
