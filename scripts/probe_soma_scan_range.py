"""Paired flat-ground scan-range interventions using the actual live checkpoint."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.native_mb_online_live import online_command_goal,predict_planar_terrain
from motionbricks.data.native_mb_observed_history import reencode_played_history
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--seed-scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--rollout-blocks',type=int,default=30)
    args=p.parse_args();torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest');torch.manual_seed(2333)
    inference,bundle=load_terrain_bundle(args.bundle,args.device)
    with np.load(args.seed_scene,allow_pickle=False) as source:
        seed=torch.as_tensor(source['initial_history'],device=args.device)
    scene={'heightfield_y_m':seed.new_zeros(601,601),
           'min_point_world_xyz_m':seed.new_tensor([float(seed[0,3,0])-30,0,float(seed[0,3,2])-30]),'dx_m':.1}
    settings={'num_inference_step':4,'pose_token_sampling_use_argmax':True,'cache_terrain_encoder':True}
    version=bundle['protocol']['version']
    report={'source_sha256':sha256(args.bundle),'scene':'exact constant world Y=0',
            'runtime_changed':False,'paired':[],'rollouts':[]}

    def predict(history,distance,velocity,refine):
        goal=online_command_goal(history[...,:5],velocity,0.,12,version)
        query=history.new_tensor([[distance,.6,1.,.4]])
        diagnostics={}
        prediction,_=predict_planar_terrain(inference,history,goal,scene,num_tokens=12,
            config=settings,query_controls=query,diagnostics=diagnostics)
        if refine:
            path=prediction[:,4:44,[0,2]].detach()
            prediction,_=predict_planar_terrain(inference,history,goal,scene,num_tokens=12,
                config=settings,query_controls=query,query_path_world_xz=path,
                query_path_valid=torch.ones(path.shape[:-1],dtype=torch.bool,device=path.device),diagnostics=diagnostics)
        return prediction,diagnostics

    for velocity in ([0.,0.],[0.,1.6]):
        for refine in (0,1):
            reference=None
            trained_reference=None
            for distance in (.8,1.9,3.4,6.,12.,20.):
                world,debug=predict(seed,distance,velocity,refine)
                root=torch.tensor(debug['root_stage_world_xyz_m'])
                decoded=world[0,4:44,:3].cpu()
                if reference is None:reference=decoded.clone()
                if distance==3.4:trained_reference=decoded.clone()
                row={'velocity_xz':velocity,'path_refinements':refine,'lookahead_m':distance,
                     'root_prefix_dy_m':float((root[:8,1]-seed[0,3,1].cpu()).mean()),
                     'root_final_dy_m':float(root[-1,1]-seed[0,3,1].cpu()),
                     'decoded_prefix_dy_m':float((decoded[:8,1]-seed[0,3,1].cpu()).mean()),
                     'decoded_final_dy_m':float(decoded[-1,1]-seed[0,3,1].cpu()),
                     'decoded_final_xz_m':(decoded[-1,[0,2]]-seed[0,3,[0,2]].cpu()).tolist(),
                     'decoded_difference_from_08m_m':float((decoded-reference).abs().mean())}
                if trained_reference is not None:
                    row['decoded_difference_from_34m_max_m']=float((decoded-trained_reference).abs().max())
                row['model_extent_m']=debug.get('motion_observation_extent_m')
                report['paired'].append(row);print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    trained_trajectory=None
    for distance in (1.9,3.4,20.):
        history=seed.clone();roots=[];started=time.perf_counter()
        for block in range(args.rollout_blocks):
            world,_=predict(history,distance,[0.,0.],1)
            roots.append(world[0,4:12,:3].clone())
            history=reencode_played_history(world[:,8:12],inference.global_motion_rep)
        trajectory=torch.cat(roots).cpu()
        row={'lookahead_m':distance,'seconds':len(trajectory)/30,
             'final_delta_xyz_m':(trajectory[-1]-seed[0,3,:3].cpu()).tolist(),
             'root_y_min_m':float(trajectory[:,1].min()),'root_y_max_m':float(trajectory[:,1].max()),
             'elapsed_seconds':time.perf_counter()-started}
        if distance==3.4:trained_trajectory=trajectory.clone()
        if distance==20 and trained_trajectory is not None:
            row['trajectory_difference_from_34m_max_m']=float((trajectory-trained_trajectory).abs().max())
        report['rollouts'].append(row);print(json.dumps({'rollout':row}),flush=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
