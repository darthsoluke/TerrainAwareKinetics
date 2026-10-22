"""Read-only causal terrain-response probe on an existing fixed checkpoint.

Compare the old weak-pair goal and actual online WASD goal with identical
observed history, desired velocity and heading. Remove only the authored wall.
This is diagnostic evidence, never a training target or deployment approval.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_model_v3 import JointTerrainController,SCHEMA
from motionbricks.data.terrain_joint_objective import SoftCommandState,JointGeometryObjective
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','checkpoint','dataset','pairs','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seconds',type=int,default=6)
    args=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True,exist_ok=False)
    device='cuda:0'
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    if checkpoint['schema']!=SCHEMA or sha256(args.source)!=checkpoint['contract']['source_bundle_sha256']:
        raise ValueError('The requested checkpoint and source model do not match')
    inference,bundle=load_terrain_bundle(args.source,device);model=JointTerrainController(inference,bundle)
    model.load_state_dict(checkpoint['model_state'],strict=True);model.eval()
    data=TerrainLearningDataset(args.dataset,build_native_representation(args.dataset/'stats'),'validation',args.pairs)
    geometry=JointGeometryObjective(inference.global_motion_rep,device)
    groups={}
    for index,row in enumerate(data.pairs):groups.setdefault(row['source_group'],index)
    report={'schema':'motionbricks.braking_counterfactual.v1','checkpoint_step':checkpoint['step'],
        'checkpoint_sha256':sha256(args.checkpoint),'postprocessing':None,'quality_approved':False,
        'seconds':args.seconds,'cases':[],'paired':[],
        'scope':'one held-out source per existing validation group; full official surface; no unseen-shape claim'}
    for index in groups.values():
        item=data[(len(data.base)+index,1801+index)]
        source=item['world'][None].to(device);intent=item['override'].to(device)
        rows={}
        for protocol in ('weak_fixed_goal','online_wasd_goal'):
            for terrain in ('wall','open'):
                scene=dict(item['scene'])
                if terrain=='open':scene['heightfield_y_m']=torch.zeros_like(scene['heightfield_y_m'])
                history=source[:,:4].clone();parts=[];state=SoftCommandState()
                heading=float(torch.atan2(intent[5],intent[4]))
                for _ in range(int(np.ceil(args.seconds*30/8))):
                    task=intent[None] if protocol=='weak_fixed_goal' else state.intent(history,intent[2:4],heading,8)
                    predicted=model(history,task,[scene],num_tokens=8)['world']
                    parts.append(predicted[:,4:12])
                    history=reencode_played_history(predicted[:,8:12],inference.global_motion_rep)
                motion=torch.cat(parts,1)[:,:args.seconds*30]
                roots=motion[0,:,:3]
                velocity=(roots[1:,[0,2]]-roots[:-1,[0,2]])*30
                direction=intent[2:4]/intent[2:4].norm()
                along=velocity@direction
                displacement=roots[:,[0,2]]-source[0,3,[0,2]]
                progress=displacement@direction
                name=f'{index}-{protocol}-{terrain}'
                surface=geometry.full_surface_metrics(motion,scene,source,item['row']['source_id']+terrain)
                row={'source_id':item['row']['source_id'],'source_group':item['row']['source_group'],
                    'protocol':protocol,'terrain':terrain,'desired_speed_mps':float(intent[2:4].norm()),
                    'initial_speed_mps':float((source[0,3,[0,2]]-source[0,0,[0,2]]).norm()*10),
                    'initial_history_sha256':hashlib.sha256(source[:,:4].cpu().numpy().tobytes()).hexdigest(),
                    'net_forward_progress_m':float(progress[-1]),'first_second_forward_speed_mps':float(along[:29].mean()),
                    'final_forward_speed_mps':float(along[-30:].mean()),
                    'final_speed_mps':float(velocity[-30:].norm(dim=-1).mean()),**surface}
                np.savez_compressed(args.output/(name+'.npz'),roots=roots.cpu().numpy(),
                    forward_speed_mps=along.cpu().numpy(),forward_progress_m=progress.cpu().numpy(),
                    metadata_json=np.asarray(json.dumps(row)))
                rows[(protocol,terrain)]=row;report['cases'].append(row);save(args.output/'report.json',report)
            wall,clear=rows[(protocol,'wall')],rows[(protocol,'open')]
            report['paired'].append({'source_id':item['row']['source_id'],'protocol':protocol,
                'open_minus_wall_progress_m':clear['net_forward_progress_m']-wall['net_forward_progress_m'],
                'open_minus_wall_final_speed_mps':clear['final_speed_mps']-wall['final_speed_mps'],
                'open_forward_progress_ratio':clear['net_forward_progress_m']/(args.seconds*wall['desired_speed_mps']),
                'wall_final_speed_mps':wall['final_speed_mps'],
                'wall_max_penetration_m':wall['body_penetration_max_m']})
            save(args.output/'report.json',report)
        print(json.dumps({'completed_source_group':item['row']['source_group']}),flush=True)
    report['complete']=True;save(args.output/'report.json',report)
    print(json.dumps(report['paired'],indent=2),flush=True)


if __name__=='__main__':main()
