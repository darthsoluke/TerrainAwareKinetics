"""Position-triggered planar input tests; no teacher height or pose in the actor."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_contact_preserve import load_model
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.pilot_rugged_transfer import route_coverage
from scripts.prepare_joint_sdf_context_pilot import recorded_direction
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def planar_waypoints(world,spacing=.75):
    points=world[:,[0,2]];indices=[3];distance=0.
    for i in range(4,len(points)):
        distance+=float((points[i]-points[i-1]).norm())
        if distance>=spacing:indices.append(i);distance=0.
    if indices[-1]!=len(points)-1:indices.append(len(points)-1)
    return points[indices].clone(),indices


def joystick(history,waypoints,index,speed,facing_offset,radius=.45):
    position=history[0,-1,[0,2]]
    while index<len(waypoints)-1 and (waypoints[index]-position).norm()<=radius:index+=1
    delta=waypoints[index]-position
    reached=index==len(waypoints)-1 and delta.norm()<=radius
    direction=delta/delta.norm().clamp_min(1e-6)
    heading=torch.atan2(direction[0],direction[1])+facing_offset
    controls=torch.cat((direction*speed,heading.cos()[None],heading.sin()[None]))[None]
    return controls,index,bool(reached)


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=json.loads(args.audit.read_text());selected={r['source_id']:r for r in report['rows']}
    summaries={}
    for arm in args.arms:
        label,file=arm.split('=',1);file=Path(file)
        model,_=load_model(torch.load(file,map_location='cpu',mmap=True,weights_only=False))
        rep=model.inference.global_motion_rep;geometry=JointGeometryObjective(rep,'cuda:0');rows=[]
        folder=args.output/label;folder.mkdir()
        for split in ('train','validation','test'):
            data=TerrainLearningDataset(args.dataset,rep,split)
            for i,row in enumerate(data.records):
                if row['source_id'] not in selected:continue
                item=data.full_reference_item(i);source=item['world'].cuda();scene=item['scene']
                directions=recorded_direction(data,row,len(source)).cuda()
                gap,known=heightfield_gap(source[:,:3],scene);ground=source[:,1]-gap
                edge=torch.nonzero((ground[4:]-ground[3]).abs()>.03).flatten()
                end=min(30,int(edge[0])+4 if len(edge) else 30)
                if end<9:raise ValueError('Missing real flat approach')
                speed=torch.diff(source[3:end,[0,2]],dim=0).norm(dim=-1).median()*30
                offset=torch.atan2(source[3,4],source[3,3])-torch.atan2(directions[3,0],directions[3,1])
                waypoints,indices=planar_waypoints(source)
                history=source[None,:4].clone();state=SoftCommandState();parts=[];commands=[];index=1;failure=None;reached=False
                limit=min(100,2*((len(source)-4)//8))
                for block in range(limit):
                    controls,index,reached=joystick(history,waypoints,index,speed,offset)
                    if reached:break
                    commands.append(dict(frame=block*8,waypoint=index,source_planar_index=indices[index],
                        target_xz=waypoints[index].cpu().tolist(),controls=controls[0].cpu().tolist()))
                    try:
                        world=model(history,state.from_controls(history,controls,8),[scene])['world']
                        parts.append(world[:,4:12]);history=reencode_played_history(world[:,8:12],rep)
                    except ValueError as error:failure=str(error);break
                if parts and not reached:_,index,reached=joystick(history,waypoints,index,speed,offset)
                value=dict(source_id=row['source_id'],split=split,family=selected[row['source_id']]['family'],
                    source_exact_skin_pass=selected[row['source_id']]['source_exact_skin_pass'],
                    frames=len(parts)*8,max_frames=limit*8,goal_reached=reached,failure=failure,commands=commands,
                    control='position-triggered reference XZ waypoint joystick; automated input test, not actual human WASD',
                    teacher_future_height_or_pose=False,postprocessing=None)
                if parts:
                    generated=torch.cat(parts,1);value.update(exact_skin_metrics(geometry,generated,scene))
                    value.update(route_coverage(generated[0,:,:3],source[4:,:3]))
                    _,points=geometry.geometry(generated);g,k=heightfield_gap(points[0,:,:geometry.sole_count],scene)
                    value['both_soles_above5cm_fraction']=float((g.amin(-1)>.05).float().mean())
                    save_motion(folder/(row['source_id']+'-generated.npz'),generated,scene,rep,value)
                save_motion(folder/(row['source_id']+'-source.npz'),source[None],scene,rep,dict(role='unmodified_teacher'))
                rows.append(value);write_json(folder/'metrics.json',dict(rows=rows,quality_approved=False))
        summaries[label]=dict(rows=rows,checkpoint_sha256=sha256(file))
        del model,geometry
        torch.cuda.empty_cache()
    write_json(args.output/'complete.json',dict(arms=summaries,optimizer_updates=0,
        only_planar_input_goals=True,quality_approved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','audit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arms',nargs='+',required=True)
    main(p.parse_args())
