"""Fixed held-out FK rollouts; no runtime collision response or pose corrections."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_model_v3 import JointTerrainController,SCHEMA
from motionbricks.data.terrain_joint_objective import JointGeometryObjective,SoftCommandState
from motionbricks.data.terrain_learning_data import TerrainLearningDataset,source_task_intent
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.terrain_distance_volume import swept_surface_points
from motionbricks.data.terrain_braking_metrics import braking_kinematics,paired_braking_checks
from motionbricks.data.terrain_control import CONTROL_VERSION
from motionbricks.data.terrain_traversal_metrics import TRAVERSAL_PROTOCOL,traversal_window,traversal_metrics,traversal_checks
from motionbricks.data.terrain_scene_frame import HorizontalSceneFrame
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256


def write_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def evaluate_controller(model,data,geometry,output,*,seconds=10,max_short=16,max_walls=0,training_data=None):
    original=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        return _evaluate_controller(model,data,geometry,output,seconds=seconds,max_short=max_short,max_walls=max_walls,training_data=training_data)
    finally:torch.backends.cuda.matmul.allow_tf32=original


@torch.no_grad()
def _evaluate_controller(model,data,geometry,output,*,seconds=10,max_short=16,max_walls=0,training_data=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    device=next(model.parameters()).device
    report={'schema':SCHEMA,'control_version':CONTROL_VERSION,'quality_approved':False,'postprocessing':None,'short':[],'flat':[],'walls':[],
        'criteria':{'flat_drift_max_m':.1,'flat_speed_p95_mps':.1,'foot_penetration_p99_m':.02,
                    'foot_gap_median_m':.05,'wall_body_penetration_p99_m':.03,
                    'wall_final_speed_mps':.15,'start_progress_ratio_min':.5},
        'scope':'model outputs from current local scene and soft intent; no future pose/height input'}
    report.update(evaluation_goal=TRAVERSAL_PROTOCOL,traversal=[],traversal_coverage={},directional=[],
        auxiliary_wall_diagnostics_only=True)
    rep=model.inference.global_motion_rep
    def observe(motion):
        checks={}
        try:return reencode_played_history(motion,rep,tensor_checks=checks)
        except ValueError:
            write_json(output/'failure.json',{'history_error':{key:float(v) for key,v in checks.items()}})
            raise
    groups={}
    for index,row in enumerate(data.base.records):
        category=row.get('control_category') if row['origin']=='bones_seed' else row.get('terrain_family','parc')
        groups.setdefault((row['origin'],category),[]).append(index)
    ranked=[]
    for key,indices in sorted(groups.items()):
        indices.sort(key=lambda i:hashlib.sha256(data.records[i]['source_id'].encode()).hexdigest())
        ranked.append(indices)
    selection=[indices[round_] for round_ in range(2) for indices in ranked if len(indices)>round_][:max_short]

    def store(name,world,scene,extra=None):
        pose=rep.inverse(world,False,return_all=True,joint_positions_from='global_rot_data')
        np.savez_compressed(output/(name+'.npz'),fps=30,world=world[0].cpu().numpy(),
            roots=world[0,:,:3].cpu().numpy(),local_rot_mats=pose['local_joint_rots'][0].cpu().numpy(),
            heightfield_y_m=torch.as_tensor(scene['heightfield_y_m']).cpu().numpy(),
            min_point_world_xyz_m=torch.as_tensor(scene['min_point_world_xyz_m']).cpu().numpy(),
            dx_m=float(scene['dx_m']),metadata_json=np.asarray(json.dumps(extra or {})))

    def physical_motion(world,scene,source,identity,full_surface=False):
        joints,points=geometry.geometry(world)
        gaps,known=heightfield_gap(points[0,:,:geometry.sole_count],scene)
        half=geometry.sole_count//2
        masked=gaps.masked_fill(~known,torch.inf)
        feet=torch.stack((masked[:,:half].amin(-1),masked[:,half:].amin(-1)),-1)
        foot_known=torch.stack((known[:,:half].any(-1),known[:,half:].any(-1)),-1)
        volume=geometry.volume(scene,source,identity)
        swept=swept_surface_points(points[0,:,::3],8)
        distance,valid=volume.query(swept)
        fallback,full_known=heightfield_gap(swept,scene)
        distance=torch.where(valid,distance,fallback)
        assessed=full_known
        metric={'body_penetration_p99_m':float(torch.quantile((-distance[assessed]).clamp_min(0),.99)) if assessed.any() else None,
            'body_penetration_max_m':float((-distance[assessed]).clamp_min(0).max()) if assessed.any() else None,
            'unknown_body_fraction':float((~assessed).float().mean()),
            'foot_penetration_p99_m':float(torch.quantile((-feet[foot_known]).clamp_min(0),.99)) if foot_known.any() else None,
            'unknown_foot_fraction':float((~foot_known).float().mean())}
        if full_surface:metric.update(geometry.full_surface_metrics(world,scene,source,identity))
        return joints,points,feet,metric

    for index in selection:
        item=data[(index,1801+index)];world=item['world'][None].to(device)
        history=world[:,:4];generated=[];command_state=SoftCommandState()
        for block in range(4):
            offset=block*8;window=world[:,offset:offset+33]
            task=source_task_intent(window,32,category=item['row'].get('control_category'),
                                    history=history,state=command_state,reference=world)
            result=model(history,task,[item['scene']],num_tokens=8)
            generated.append(result['world'][:,4:12])
            history=observe(result['world'][:,8:12])
        prediction=torch.cat(generated,1)
        pj,_,_,metrics=physical_motion(prediction,item['scene'],world,item['row']['source_id'])
        sj=rep.inverse(world[:,4:36],False,return_all=False,joint_positions_from='global_rot_data')['posed_joints']
        row={'source_id':item['row']['source_id'],'source_group':item['row']['source_group'],
             'origin':item['row']['origin'],'category':item['row'].get('control_category',item['row'].get('terrain_family')),
             'fk_mpjpe_m':float((pj-sj).norm(dim=-1).mean()),**metrics}
        report['short'].append(row)
        store(f'short-{index}',prediction,item['scene'],row)
        write_json(output/'metrics.json',report)

    # Full real reference clips, not random one-second prefixes or added walls.
    # Training cases show fit; validation retains the existing scene-group split.
    for split,corpus in (('train',training_data),(data.split,data)):
        if corpus is None:continue
        families={}
        for index,row in enumerate(corpus.base.records):
            if row['origin'].startswith('parc'):
                families.setdefault(row.get('terrain_family','unspecified'),[]).append(index)
        completed=[]
        for family,indices in sorted(families.items()):
            ranked=sorted(indices,key=lambda i:hashlib.sha256(corpus.records[i]['source_id'].encode()).hexdigest())
            candidates=[]
            for index in ranked[:4]:
                item=corpus.full_reference_item(index)
                start,blocks,activity=traversal_window(item['world'][:,:3].numpy(),seconds)
                reference=item['world'][start+4:start+4+blocks*8,:3].numpy()
                if np.linalg.norm(np.diff(reference,axis=0),axis=-1).sum()>=.5:
                    candidates.append((activity,index,start,blocks,item))
            if not candidates:continue
            _,index,start,blocks,item=max(candidates,key=lambda r:(r[0],-r[1]))
            world=item['world'][start:start+blocks*8+25][None].to(device)
            history=world[:,:4];state=SoftCommandState();parts=[];controls=[];goals=[];rollout_failure=None
            for block in range(blocks):
                offset=block*8;window=world[:,offset:offset+33]
                task=source_task_intent(window,32,history=history,state=state,reference=world)
                controls.append(task[0,2:].cpu().numpy())
                goals.append(task[0,:2].cpu().numpy())
                try:
                    result=model(history,task,[item['scene']],num_tokens=8)
                except ValueError as error:
                    if str(error)!='Current terrain origin is unobserved; do not invent a ground height' or not parts:
                        raise
                    rollout_failure={'reason':str(error),'block':block,'generated_frames':sum(p.shape[1] for p in parts)}
                    break
                parts.append(result['world'][:,4:12]);history=observe(result['world'][:,8:12])
            prediction=torch.cat(parts,1);reference=world[:,4:4+blocks*8]
            pj,_,_,metric=physical_motion(prediction,item['scene'],world,item['row']['source_id'],full_surface=True)
            sj=rep.inverse(reference[:,:prediction.shape[1]],False,return_all=False,joint_positions_from='global_rot_data')['posed_joints']
            metric.update(traversal_metrics(prediction[0,:,:3].cpu().numpy(),reference[0,:,:3].cpu().numpy()))
            metric['fk_mpjpe_m']=float((pj-sj).norm(dim=-1).mean())
            source_metrics=geometry.full_surface_metrics(reference,item['scene'],world,item['row']['source_id'])
            checks=traversal_checks(metric)
            row={'source_id':item['row']['source_id'],'source_group':item['row']['source_group'],
                'split':split,'family':family,'start_frame':start,'emitted_frames':prediction.shape[1],
                'requested_frames':blocks*8,'rollout_failure':rollout_failure,
                'seconds':prediction.shape[1]/30,'metrics':metric,'source_physical_metrics':source_metrics,
                'checks':checks,'passed':all(checks.values()),
                'control_provenance':'requested reference planar waypoint plus inferred velocity/facing; not recorded WASD; no future Y/pose',
                'evaluation_mode':'requested_path_replay; free joystick response is tested separately in flat cases',
                'requested_planar_goals_world_xz_m':np.asarray(goals).tolist(),
                'controls':np.asarray(controls).tolist()}
            report['traversal'].append(row);completed.append(family)
            name=f'traversal-{split}-{index}'
            store(name,prediction,item['scene'],row)
            store(name+'-source',reference,item['scene'],dict(row,role='unchanged_reference'))
            write_json(output/'metrics.json',report)
        report['traversal_coverage'][split]={'available_families':sorted(families),'assessed_families':completed,
            'missing_families':sorted(set(families)-set(completed)),
            'claim':'seen-source fit diagnostic' if split=='train' else 'held-out source/scene-group traversal'}
        write_json(output/'metrics.json',report)

    quiet=[]
    for index,row in enumerate(data.base.records):
        if row['origin']!='bones_seed' or row.get('control_category')!='idle':continue
        item=data[(index,1801+index)];world=item['world'][None].to(device)
        if float((world[:,1:,:3]-world[:,:-1,:3]).norm(dim=-1).max()*30)>.08:continue
        quiet.append(item)
        if len(quiet)==2:break
    if len(quiet)!=2:raise ValueError('Two independent held-out quiet seeds are required')

    # Same physical flat task in four world headings, with causal joystick
    # requests only. Reference-path waypoints are not supplied to this test.
    for degrees in (0.,90.,180.,270.):
        item=quiet[0];source=item['world'][None].to(device)
        frame=HorizontalSceneFrame(source.new_tensor([degrees*np.pi/180]),source[:,3,[0,2]])
        history=frame.motion(rep,source[:,:4]);origin=history[0,3,:3].clone()
        heading=torch.atan2(history[:,3,4],history[:,3,3]);direction=torch.stack((heading.sin(),heading.cos()),-1)
        state=SoftCommandState();parts=[]
        blocks=int(np.ceil(seconds*30/8))
        for _ in range(blocks):
            task=state.intent(history,direction*1.6,heading,8)
            result=model(history,task,[item['scene']],num_tokens=8)
            parts.append(result['world'][:,4:12]);history=observe(result['world'][:,8:12])
        motion=torch.cat(parts,1)
        _,_,_,metric=physical_motion(motion,item['scene'],source,item['row']['source_id'],full_surface=True)
        delta=motion[0,:,:3]-origin
        perpendicular=torch.stack((direction[0,1],-direction[0,0]))
        metric.update(progress_ratio=float((delta[-1,[0,2]]*direction[0]).sum())/(blocks*8/30*1.6),
            cross_track_max_m=float((delta[:,[0,2]]*perpendicular).sum(-1).abs().max()))
        checks={'follows_requested_speed':.85<=metric['progress_ratio']<=1.15,
            'follows_requested_direction':metric['cross_track_max_m']<=.25,
            'known_scene':metric['unknown_body_fraction']==0,
            'no_large_penetration':metric['body_penetration_p99_m'] is not None and metric['body_penetration_p99_m']<=.03}
        row={'yaw_degrees':degrees,'input':'constant 1.6 m/s causal joystick; no reference future',
            'metrics':metric,'checks':checks,'passed':all(checks.values())}
        report['directional'].append(row);store(f'joystick-yaw-{int(degrees)}',motion,item['scene'],row)
        write_json(output/'metrics.json',report)
    for seed_index,item in enumerate(quiet):
        for behavior in ('rest','start_stop'):
            world=item['world'][None].to(device);history=world[:,:4]
            state=SoftCommandState();parts=[]
            heading=float(torch.atan2(history[0,3,4],history[0,3,3]))
            warm=15 if behavior=='start_stop' else 0
            blocks=int(np.ceil(seconds*30/8));start=history[0,-1,:3].clone()
            for block in range(warm+blocks):
                velocity=[np.sin(heading)*1.6,np.cos(heading)*1.6] if block<warm else [0.,0.]
                task=state.intent(history,velocity,heading,8)
                result=model(history,task,[item['scene']],num_tokens=8)
                parts.append(result['world'][:,4:12]);history=observe(result['world'][:,8:12])
            motion=torch.cat(parts,1)
            _,_,feet,metric=physical_motion(motion,item['scene'],world,item['row']['source_id'],full_surface=True)
            settled=warm*8+60
            roots=motion[0,settled:,:3]
            metric.update(drift_max_m=float((roots[:,[0,2]]-roots[:1,[0,2]]).norm(dim=-1).max()),
                settled_speed_p95_mps=float(torch.quantile((roots[1:]-roots[:-1]).norm(dim=-1)*30,.95)),
                root_y_range_m=float(roots[:,1].max()-roots[:,1].min()),
                median_higher_foot_gap_m=float(feet[settled:].max(-1).values.median()))
            if warm:
                metric['start_progress_ratio']=float((motion[0,warm*8-1,[0,2]]-start[[0,2]]).norm())/(warm*8/30*1.6)
            passed=(metric['unknown_foot_fraction']==0 and metric['drift_max_m']<=.1 and metric['settled_speed_p95_mps']<=.1
                and metric['foot_penetration_p99_m']<=.02 and metric['median_higher_foot_gap_m']<=.05
                and metric.get('start_progress_ratio',1)>=.5)
            row={'seed':seed_index,'behavior':behavior,'source_id':item['row']['source_id'],
                 'metrics':metric,'passed':passed}
            report['flat'].append(row);store(f'flat-{seed_index}-{behavior}',motion,item['scene'],row)
            write_json(output/'metrics.json',report)

    groups={}
    for index,row in enumerate(data.pairs):groups.setdefault(row['source_group'],[]).append(index)
    chosen=[indices[round_] for round_ in range(2) for indices in groups.values() if len(indices)>round_][:max_walls]
    for index in chosen:
        row=data.pairs[index]
        item=data[(len(data.base)+index,1801+index)]
        world=item['world'][None].to(device);control=item['override'][None].to(device)
        reference=braking_kinematics(world[0,4:,:3].cpu().numpy(),world[0,:4,:3].cpu().numpy(),control[0].cpu().numpy())
        paired={}
        for terrain in ('wall','open'):
            scene=dict(item['scene'])
            if terrain=='open':scene['heightfield_y_m']=torch.zeros_like(scene['heightfield_y_m'])
            history=world[:,:4].clone();state=SoftCommandState();parts=[]
            for block in range(int(np.ceil(max(6,seconds)*30/8))):
                task=state.from_controls(history,control,8)
                result=model(history,task,[scene],num_tokens=8)
                parts.append(result['world'][:,4:12]);history=observe(result['world'][:,8:12])
            motion=torch.cat(parts,1)
            _,_,_,metric=physical_motion(motion,scene,world,row['source_id']+terrain,full_surface=True)
            clearance=geometry.barrier_clearance(motion,row) if terrain=='wall' else None
            metric.update(braking_kinematics(motion[0,:,:3].cpu().numpy(),world[0,:4,:3].cpu().numpy(),
                control[0].cpu().numpy(),clearance))
            paired[terrain]=metric
            store(f'{terrain}-{index}',motion,scene,metric)
        checks=paired_braking_checks(paired['wall'],paired['open'],reference)
        entry={'source_id':row['source_id'],'source_group':row['source_group'],'metrics':paired['wall'],
            'open_metrics':paired['open'],'source_kinematics':reference,'checks':checks,'passed':all(checks.values())}
        report['walls'].append(entry)
        write_json(output/'metrics.json',report)
    report['flat_gate_passed']=all(r['passed'] for r in report['flat'])
    report['wall_gate_passed']=bool(report['walls']) and all(r['passed'] for r in report['walls'])
    report['short_mean_mpjpe_m']=float(np.mean([r['fk_mpjpe_m'] for r in report['short']]))
    report['short_body_penetration_p99_mean_m']=float(np.mean([r['body_penetration_p99_m'] for r in report['short'] if r['body_penetration_p99_m'] is not None]))
    covered=len(report['traversal_coverage'])==2 and all(not row['missing_families'] for row in report['traversal_coverage'].values())
    report['traversal_gate_passed']=covered and bool(report['traversal']) and all(r['passed'] for r in report['traversal'])
    report['directional_gate_passed']=len(report['directional'])==4 and all(r['passed'] for r in report['directional'])
    report['absolute_gates_passed']=report['flat_gate_passed'] and report['traversal_gate_passed'] and report['directional_gate_passed']
    write_json(output/'metrics.json',report)
    print(json.dumps({'evaluation':str(output),'flat_gate':report['flat_gate_passed'],
        'traversal_gate':report['traversal_gate_passed'],'short_mpjpe_m':report['short_mean_mpjpe_m']}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','dataset','pairs','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=int,default=60)
    p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--auxiliary-walls',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    checkpoint=torch.load(args.checkpoint,map_location='cuda:0',weights_only=False)
    if checkpoint['schema']!=SCHEMA or sha256(args.source)!=checkpoint['contract']['source_bundle_sha256']:
        raise ValueError('Joint checkpoint/source binding differs')
    inference,bundle=load_terrain_bundle(args.source,'cuda:0');model=JointTerrainController(inference,bundle)
    model.load_state_dict(checkpoint['model_state'],strict=True);model.eval()
    data=TerrainLearningDataset(args.dataset,build_native_representation(args.dataset/'stats'),args.split,args.pairs)
    training=TerrainLearningDataset(args.dataset,build_native_representation(args.dataset/'stats'),'train',args.pairs)
    geometry=JointGeometryObjective(inference.global_motion_rep,'cuda:0')
    evaluate_controller(model,data,geometry,args.output,seconds=args.seconds,max_walls=args.auxiliary_walls,training_data=training)


if __name__=='__main__':main()
