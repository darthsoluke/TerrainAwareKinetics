"""Read-only world-yaw equivariance and command-following check on flat ground."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_model_v3 import JointTerrainController
from motionbricks.data.terrain_scene_frame import HorizontalSceneFrame
from motionbricks.data.terrain_control import SoftCommandState
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_terrain_overfit import select_cases
from scripts.train_joint_terrain_v3 import write_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoints',nargs='+',required=True,help='label=path')
    args=p.parse_args();torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    infer,bundle=load_terrain_bundle(args.source,'cuda:0');model=JointTerrainController(infer,bundle)
    data=TerrainLearningDataset(args.dataset,build_native_representation(args.dataset/'stats'),'train')
    case=next(c for c in select_cases(data) if c['name']=='walk')
    history=case['world'][None,:4].to('cuda:0');rep=infer.global_motion_rep
    yaw=torch.atan2(history[:,3,4],history[:,3,3])
    velocity=torch.stack((yaw.sin(),yaw.cos()),-1)*1.6
    task=SoftCommandState().intent(history,velocity,yaw,8)
    center=history[0,3,:3].cpu()
    scene={'heightfield_y_m':torch.zeros(401,401),'min_point_world_xyz_m':torch.tensor([center[0]-100,0.,center[2]-100]),'dx_m':torch.tensor(.5)}
    report={'source_id':case['row']['source_id'],'physical_scene':'flat','intervention':'rigid rotation of history and command together','models':{}}
    for entry in args.checkpoints:
        label,path=entry.split('=',1)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model_state'],strict=True);del saved
        rows=[];baseline=None
        for degrees in (0.,5.,-5.,45.,90.,180.,270.):
            frame=HorizontalSceneFrame(history.new_tensor([degrees*torch.pi/180]),history[:,3,[0,2]])
            rotated=frame.motion(rep,history);command=frame.task(task)
            result=model(rotated,command,[scene],num_tokens=8)
            world=frame.motion(rep,result['world'],inverse=True)
            if baseline is None:baseline=world.clone()
            step=result['world'][:,11,[0,2]]-rotated[:,3,[0,2]]
            direction=command[:,2:4]/1.6
            rows.append({'yaw_degrees':degrees,
                'root_covariance_error_m':float((world[:,4:12,:3]-baseline[:,4:12,:3]).square().sum(-1).mean().sqrt()),
                'command_progress_m':float((step*direction).sum()),
                'cross_step_m':float((step*torch.stack((direction[:,1],-direction[:,0]),-1)).sum()),
                'requested_step_m':1.6*8/30})
        report['models'][label]=rows
    write_json(args.output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
