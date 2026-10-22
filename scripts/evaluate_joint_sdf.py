"""Uncorrected continuous terrain rollouts, with causal controls and SDF ablation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState,demonstration_controls
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.joint_sdf_controller import JointSDFController,SDF_VERSION
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


@torch.no_grad()
def save_motion(path,world,scene,rep,metadata):
    pose=rep.inverse(world,False,return_all=True,joint_positions_from='global_rot_data')
    np.savez_compressed(path,fps=30,world=world[0].cpu().numpy(),roots=world[0,:,:3].cpu().numpy(),
        local_rot_mats=pose['local_joint_rots'][0].cpu().numpy(),
        heightfield_y_m=torch.as_tensor(scene['heightfield_y_m']).cpu().numpy(),
        min_point_world_xyz_m=torch.as_tensor(scene['min_point_world_xyz_m']).cpu().numpy(),
        dx_m=float(scene['dx_m']),metadata_json=np.asarray(json.dumps(metadata)))


def cases(saved):
    path=Path(saved['contract']['dataset']);rep=build_native_representation(path/'stats');result=[]
    for split in ('train','validation'):
        data=TerrainLearningDataset(path,rep,split)
        for family in ('stairs','climbing','mixed','boxes','siggraph_terrain','idle','walk'):
            eligible=[i for i,r in enumerate(data.records) if (r.get('control_category')==family if family in ('idle','walk')
                else r.get('terrain_family')==family)]
            if not eligible:continue
            index=sorted(eligible,key=lambda i:(data.records[i]['source_id']!='official_df3f1541e287fcb3',data.records[i]['source_id']))[0]
            item=data.full_reference_item(index);item=dict(item,name=split+'-'+family,split=split,family=family)
            item['full_source_frames']=len(item['world']);item['world']=item['world'][:257]
            if family in ('idle','walk'):
                center=item['world'][3,:3]
                item['scene']={'heightfield_y_m':torch.zeros(201,201),
                    'min_point_world_xyz_m':torch.tensor([center[0]-50,0,center[2]-50]),'dx_m':torch.tensor(.5)}
            result.append(item)
    return result


@torch.no_grad()
def evaluate(model,items,geometry,output):
    output.mkdir(parents=True,exist_ok=False);rep=model.inference.global_motion_rep;rows=[]
    device=next(model.parameters()).device
    for item in items:
        source=item['world'][None].to(device);scene=item['scene'];field=HeightfieldPrismSDF(scene)
        terrain=item['family'] not in ('idle','walk')
        modes=('source_velocity_schedule','constant_approach') if terrain else ('source_velocity_schedule',)
        for mode in modes:
            state=SoftCommandState();history=source[:,:4].clone();parts=[];times=[];failure=None
            blocks=(source.shape[1]-4)//8
            # Constant request is estimated once from the actual approach XZ.
            initial_control=demonstration_controls(source[:,:33],32,category=item['family'],reference=source)
            for block in range(blocks):
                offset=min(block*8,source.shape[1]-33)
                control=initial_control if mode=='constant_approach' else demonstration_controls(
                    source[:,offset:offset+33],32,category=item['family'],reference=source)
                task=state.from_controls(history,control,8);started=time.perf_counter()
                try:result=model(history,task,[scene])
                except ValueError as error:failure={'frame':block*8,'reason':str(error)};break
                torch.cuda.synchronize();times.append(time.perf_counter()-started)
                parts.append(result['world'][:,4:12]);history=reencode_played_history(result['world'][:,8:12],rep)
            row={'name':item['name'],'mode':mode,'source_id':item['row']['source_id'],'split':item['split'],
                'frames':sum(p.shape[1] for p in parts),'requested_frames':blocks*8,'full_source_frames':item['full_source_frames'],
                'failure':failure,'postprocessing':None,'reference_is_recorded_control':False,'passed':False}
            if parts:
                world=torch.cat(parts,1);truth=source[:,4:4+world.shape[1]]
                joints,points=geometry.geometry(world);source_joints,source_points=geometry.geometry(truth)
                gap,known=heightfield_gap(points[0,:,:geometry.sole_count],scene)
                distance,_,valid=field.query(points[0]);sd,_,sv=field.query(source_points[0])
                body_gap,body_known=heightfield_gap(points[0],scene)
                source_gap,source_known=heightfield_gap(source_points[0],scene)
                penetration=(-torch.where(valid,distance,body_gap)[body_known]).clamp_min(0)
                source_pen=(-torch.where(sv,sd,source_gap)[source_known]).clamp_min(0)
                delta=world[0,-1,:3]-source[0,3,:3];desired=source[0,blocks*8+3,:3]-source[0,3,:3]
                axis=desired[[0,2]]/desired[[0,2]].norm().clamp_min(1e-6)
                row.update(root_endpoint_error_m=float((world[0,-1,:3]-source[0,blocks*8+3,:3]).norm()),
                    root_rise_m=float(delta[1]),source_rise_m=float(desired[1]),
                    progress_m=float(delta[[0,2]].dot(axis)),source_progress_m=float(desired[[0,2]].norm()),
                    root_drift_m=float(delta[[0,2]].norm()),
                    cross_track_m=float((delta[[0,2]]*torch.stack((axis[1],-axis[0]))).sum().abs()),
                    root_rms_m=float((world[...,:3]-truth[...,:3]).square().sum(-1).mean().sqrt()),
                    relative_mpjpe_m=float(((joints-world[:,:,:3,None].transpose(-1,-2))-(source_joints-truth[:,:,:3,None].transpose(-1,-2))).norm(dim=-1).mean()),
                    sole_vertical_p99_m=float(torch.quantile((-gap[known]).clamp_min(0),.99)) if known.any() else None,
                    sampled_body_sdf_p99_m=float(torch.quantile(penetration,.99)) if len(penetration) else None,
                    sampled_body_sdf_max_m=float(penetration.max()) if len(penetration) else None,
                    source_body_sdf_p99_m=float(torch.quantile(source_pen,.99)) if len(source_pen) else None,
                    sampled_vertices_per_frame=len(geometry.indices),unknown_body_fraction=float((~body_known).float().mean()),
                    uncertified_nearest_surface_fraction=float((~valid).float().mean()),
                    inference_median_ms=float(np.median(times)*1000))
                complete=row['frames']==row['requested_frames'] and failure is None
                geometric=(known.all() and row['unknown_body_fraction']==0 and row['sole_vertical_p99_m']<=.04 and row['sampled_body_sdf_p99_m']<=.02)
                traversed=(row['progress_m']>=.8*row['source_progress_m'] and abs(row['root_rise_m']-row['source_rise_m'])<=.2 and row['cross_track_m']<=.35)
                if item['family']=='idle':traversed=row['root_drift_m']<=.1
                row['passed']=bool(complete and geometric and traversed)
                name=item['name']+'-'+mode
                save_motion(output/(name+'.npz'),world,scene,rep,row)
                save_motion(output/(name+'-source.npz'),source[:,4:4+blocks*8],scene,rep,
                    dict(row,role='unchanged_reference',frames=blocks*8,failure=None))
            rows.append(row);write_json(output/'metrics.json',{'cases':rows,'quality_approved':False})
            print(json.dumps(row),flush=True)
    return rows


def main(args):
    torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    path=Path(saved['contract']['source'])
    if sha256(path)!=saved['contract']['source_bundle_sha256']:raise ValueError('Source/checkpoint hash mismatch')
    infer,bundle=load_terrain_bundle(path,'cuda:0');model=JointSDFController(infer,bundle)
    model.load_state_dict(saved['model_state'],strict=True);model.requires_grad_(False);model.eval()
    selected=cases(saved)
    if args.limit:selected=selected[:args.limit]
    geometry=JointGeometryObjective(infer.global_motion_rep,'cuda:0')
    args.output.mkdir(parents=True,exist_ok=False)
    final=evaluate(model,selected,geometry,args.output/'final')
    model.sdf_enabled=False
    disabled=evaluate(model,selected,geometry,args.output/'disabled_sdf')
    write_json(args.output/'complete.json',{'checkpoint_sha256':sha256(args.checkpoint),
        'final':final,'disabled_sdf':disabled,'numerical_pass':all(r['passed'] for r in final),
        'quality_approved':False,'pending':['official skin visual inspection','independent WASD','local latency'],
        'metric_limit':'SDF is exact for the raster solid, but this audit samples body vertices and is not a swept all-surface certificate'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);main(p.parse_args())
