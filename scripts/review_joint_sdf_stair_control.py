"""Fixed-command staircase trials with varied starting positions; no output repair."""
import argparse
import json
import math
from pathlib import Path

import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.native_mb_observed_history import reencode_played_history
from scripts.evaluate_joint_sdf import save_motion
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    case=torch.load(args.cases,map_location='cpu',weights_only=False)[0]
    variants=[('moving-default',False,1.,0.,0.),('idle-default',True,1.,0.,0.),
        ('moving-slow',False,.8,0.,0.),('moving-fast',False,1.2,0.,0.),
        ('moving-left20cm',False,1.,.2,0.),('moving-right20cm',False,1.,-.2,0.),
        ('moving-back30cm',False,1.,0.,-.3),('idle-back30cm',True,1.,0.,-.3)]
    all_rows=[];identities={}
    for entry in args.runs:
        label,path=entry.split('=',1);path=Path(path)
        saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        source_path=Path(saved['contract']['source'])
        if sha256(source_path)!=saved['contract']['source_bundle_sha256']:raise ValueError('Source bundle mismatch')
        identities[label]=sha256(path)
        inference,bundle=load_terrain_bundle(source_path,'cuda:0');rep=inference.global_motion_rep
        model=JointSDFController(inference,bundle);model.load_state_dict(saved['model_state'],strict=True)
        model.requires_grad_(False);model.eval();geometry=JointGeometryObjective(rep,'cuda:0')
        scene=case['scene'];field=HeightfieldPrismSDF(scene);source=case['world'][None].to('cuda:0')
        direction=source.new_tensor(case['command_velocity']);speed=direction.norm();direction=direction/speed
        side=torch.stack((direction[1],-direction[0]));goal=source[0,-5,[0,2]]
        target_root_y=float(source[0,-5,1]);directory=args.output/label;directory.mkdir()
        for name,idle,scale,lateral,along in variants:
            history=source[:,:4].clone()
            if idle:
                history=rep.change_first_heading(bundle['initial_history'].to(source),case['command_heading'],False,False)
                history[...,[0,2]]+=source[:,3:4,[0,2]]-history[:,3:4,[0,2]]
            shift=side*lateral+direction*along;history[...,[0,2]]+=shift
            # All perturbations begin on the same measured flat approach.
            gap,known=heightfield_gap(history[0,-1:,:3],scene)
            if not known.all() or abs(float(history[0,-1,1]-gap[0]))>.001:
                raise ValueError('Perturbed spawn is not on the factual flat approach')
            initial=history.clone();state=SoftCommandState();parts=[];failure=None;reached=False
            needed=float((goal-history[0,3,[0,2]]).dot(direction))
            maximum=math.ceil((needed/float(speed*scale)+2.)*30/8)
            for block in range(maximum):
                intent=state.intent(history,direction*speed*scale,case['command_heading'],8)
                try:result=model(history,intent,[scene])
                except ValueError as error:failure=str(error);break
                parts.append(result['world'][:,4:12]);history=reencode_played_history(result['world'][:,8:12],rep)
                reached=float((history[0,-1,[0,2]]-goal).dot(direction))>=0
                if reached:break
            row={'name':name,'arm':label,'constant_command':True,'velocity_mps':float(speed*scale),
                'lateral_offset_m':lateral,'along_offset_m':along,'idle_spawn':idle,'goal_reached':reached,
                'frames':sum(p.shape[1] for p in parts),'frame_limit':maximum*8,'failure':failure,'passed':False,
                'evaluation_end':'first published-block endpoint crossing the goal plane or bounded timeout; does not alter generated motion',
                'postprocessing':None,'training_staircase':True}
            if parts:
                world=torch.cat(parts,1);_,points=geometry.geometry(world)
                sole_gap,sole_known=heightfield_gap(points[0,:,:geometry.sole_count],scene)
                distance,_,valid=field.query(points[0]);vertical,occupied=heightfield_gap(points[0],scene)
                penetration=(-torch.where(valid,distance,vertical)[occupied]).clamp_min(0)
                delta=world[0,-1,:3]-initial[0,-1,:3]
                row.update(progress_m=float(delta[[0,2]].dot(direction)),required_progress_m=needed,
                    cross_track_m=float(delta[[0,2]].dot(side).abs()),
                    root_height_error_m=abs(float(world[0,-1,1])-target_root_y),
                    sole_vertical_p99_m=float(torch.quantile((-sole_gap[sole_known]).clamp_min(0),.99)) if sole_known.any() else None,
                    sole_vertical_max_m=float((-sole_gap[sole_known]).clamp_min(0).max()) if sole_known.any() else None,
                    frames_sole_inside_2cm=int(((sole_gap<-.02)&sole_known).any(-1).sum()),
                    sampled_body_penetration_p99_m=float(torch.quantile(penetration,.99)) if len(penetration) else None,
                    sampled_body_penetration_max_m=float(penetration.max()) if len(penetration) else None,
                    unknown_body_fraction=float((~occupied).float().mean()))
                row['passed']=bool(reached and failure is None and occupied.all() and sole_known.all()
                    and row['sole_vertical_p99_m']<=.04 and row['sampled_body_penetration_p99_m']<=.02
                    and row['cross_track_m']<=.35 and row['root_height_error_m']<=.2)
                # Each clip retains its actual duration; arms with different clocks
                # are separate recordings, never padded into synchronized evidence.
                save_motion(directory/(label+'-'+name+'.npz'),world,scene,rep,row)
            all_rows.append(row);write_json(args.output/'progress.json',{'rows':all_rows,'quality_approved':False})
            print(json.dumps(row),flush=True)
        del model,geometry,inference,bundle,saved
        torch.cuda.empty_cache()
    write_json(args.output/'complete.json',{'rows':all_rows,'checkpoint_hashes':identities,
        'source_id':case['row']['source_id'],'quality_approved':False,
        'scope':'same training staircase; start/speed sensitivity, not unseen-terrain generalization'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
