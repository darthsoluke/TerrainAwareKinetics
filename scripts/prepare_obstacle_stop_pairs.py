"""Create declared weak scene/intent pairings from unchanged genuine stopping motion."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import NativeMBMotionDataset,build_native_representation
from motionbricks.data.terrain_learning_data import world_from_item
from motionbricks.data.soma_skin_reference import load_official_skinner,move_skinner
from scripts.train_native_mb import sha256
from motionbricks.data.terrain_control import CONTROL_VERSION


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-per-split',type=int,default=256)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new weak-pair artifact')
    torch.set_num_threads(4);torch.manual_seed(2333)
    rep=build_native_representation(args.dataset/'stats')
    skin=move_skinner(load_official_skinner(rep.skeleton),torch.device(args.device))
    skin.skeleton_skin.float().to(args.device)
    args.output.mkdir(parents=True)
    manifest={'schema':'motionbricks.weak_obstacle_stop_pairs.v2','control_version':CONTROL_VERSION,
        'dataset_manifest_sha256':sha256(args.dataset/'native_manifest.json'),
        'motion_source':'unchanged BONES-SEED stop clips','intent_source':'authored weak continuation labels, not recorded WASD',
        'geometry':'varied high barriers beyond full source skinned trajectory; held-out dimensions and angles',
        'control_fields':['velocity_x_mps','velocity_z_mps','body_heading_cos','body_heading_sin'],
        'goal_policy':'constructed from actual observed history and horizon by shared SoftCommandState',
        'minimum_contiguous_final_rest_seconds':1.,
        'split_policy':'source family split unchanged','postprocessing':None,'items':[],'rejected':[]}
    owners={}
    for split in ('train','validation'):
        data=NativeMBMotionDataset(args.dataset,split,min_frames=65,max_frames=10**9,
            require_terrain=True,require_online_terrain=True,online_only=True)
        eligible=[i for i,r in enumerate(data.records) if r.get('origin')=='bones_seed' and r.get('control_category')=='stop']
        accepted=0
        for index in eligible:
            row=data.records[index]
            if accepted>=args.max_per_split:break
            item=data[index];world=world_from_item(rep,item)
            roots=world[:,:3]
            velocity=(roots[1:,[0,2]]-roots[:-1,[0,2]])*30
            candidates=[]
            for start in range(0,len(world)-64,8):
                current=velocity[start:start+3].mean(0);speed=float(current.norm())
                jitter=float((velocity[start:start+3]-current).norm(dim=-1).max())
                if speed<.3 or jitter>.15:continue
                for stop in range(start+65,min(len(world),start+257)+1):
                    if float(velocity[stop-31:stop-1].norm(dim=-1).max())<=.12:
                        candidates.append((start,stop-start,current,speed));break
                if candidates:break
            if not candidates:
                manifest['rejected'].append({'source_id':row['source_id'],'split':split,'reason':'no real moving-to-rest window with one second of sustained rest'})
                continue
            start,frames,current,speed=candidates[0]
            selected=world[start:start+frames].to(args.device)[None]
            pose=rep.to(args.device).inverse(selected,False,return_all=True,joint_positions_from='global_rot_data')
            vertices=skin.skin(pose['global_joint_rots'][0],pose['posed_joints'][0],rot_is_global=True)
            travel_direction=current.to(args.device)/speed
            angle=np.deg2rad((0.,20.,-20.)[accepted%3] if split=='train' else (15.,-35.,40.)[accepted%3])
            rotation=selected.new_tensor([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            direction=rotation@travel_direction
            projection=(vertices[...,[0,2]]*direction).sum(-1)
            front=float(projection.max())
            vertex_step=float((vertices[1:]-vertices[:-1]).norm(dim=-1).max())
            margin=max(.08,vertex_step/2+.02)
            # Raster boxes extend half a cell beyond their centers. Account for
            # their full projected footprint when placing the closest wall face.
            dx=.1;count=801
            center=selected[0,3,:3].cpu();origin=torch.tensor([center[0]-40,0,center[2]-40])
            x=origin[0]+torch.arange(count)*dx;z=origin[2]+torch.arange(count)*dx
            xx,zz=torch.meshgrid(x,z,indexing='xy')
            forward=xx*float(direction[0])+zz*float(direction[1])
            lateral=xx*float(direction[1])-zz*float(direction[0])
            side_center=float(center[0]*direction[1].cpu()-center[2]*direction[0].cpu())
            footprint=dx/2*(float(direction[0].abs())+float(direction[1].abs()))
            face=front+margin
            depth=(.3,.6,1.)[accepted%3] if split=='train' else (1.8,.45,.85)[accepted%3]
            width=(8.,10.,12.)[accepted%3] if split=='train' else (4.5,7.,11.)[accepted%3]
            wall=(forward-footprint>=face)&(forward-footprint<=face+depth)&((lateral-side_center).abs()<=width/2)
            if not wall.any():raise ValueError('Authored wall is outside its declared scene')
            clearance=float((forward[wall]-footprint).min())-front
            if clearance<margin-1e-5:raise ValueError('Source body intersects the authored wall')
            heights=torch.zeros_like(xx);heights[wall]=(2.7,3.7,4.7)[accepted%3] if split=='train' else (3.2,4.2,5.2)[accepted%3]
            # A backward/sideways stop keeps the demonstrated body-facing goal.
            # Velocity is authored continuation intent, not a body-yaw label.
            facing=selected[0,-1,3:5].cpu();facing=facing/facing.norm()
            control=torch.cat((current,facing))
            identity=hashlib.sha256((row['source_id']+':'+str(start)).encode()).hexdigest()[:20]
            filename=f'{split}-{identity}.npz'
            np.savez_compressed(args.output/filename,world=selected[0].cpu().numpy(),
                heightfield_y_m=heights.numpy(),min_point_world_xyz_m=origin.numpy(),dx_m=np.float32(dx),control=control.numpy())
            group=row['source_group']
            if group in owners and owners[group]!=split:raise ValueError('Weak pair source family leaks between splits')
            owners[group]=split
            manifest['items'].append({'file':filename,'sha256':sha256(args.output/filename),'split':split,
                'source_id':row['source_id'],'source_group':group,'origin':'bones_seed','control_category':'obstacle_stop',
                'source_shard':row['shard'],'source_start':start,'frames_30hz':frames,
                'source_speed_mps':speed,'wall_height_m':float(heights.max()),
                'wall_width_m':width,'wall_depth_m':depth,'wall_normal_world_xz':direction.cpu().tolist(),
                'wall_face_projection_m':face,'wall_yaw_offset_degrees':float(np.rad2deg(angle)),
                'body_heading_label_error_degrees':0.,'final_rest_intervals':30,
                'minimum_wall_clearance_m':clearance,'swept_margin_m':margin,
                'unchanged_source_world_sha256':hashlib.sha256(selected[0].cpu().numpy().tobytes()).hexdigest()})
            accepted+=1
            if accepted%20==0:print(json.dumps({'split':split,'pairs':accepted}),flush=True)
            rep.cpu()
        print(json.dumps({'split':split,'pairs':accepted,'eligible_source_clips':len(eligible)}),flush=True)
    counts={split:sum(r['split']==split for r in manifest['items']) for split in ('train','validation')}
    manifest['counts']=counts
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if counts['train']<16 or counts['validation']<2:raise ValueError('Insufficient independent obstacle-stop supervision')


if __name__=='__main__':main()
