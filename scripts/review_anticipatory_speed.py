"""Bounded final-model speed-policy comparison and packet-state proofs."""

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.anticipatory_speed import DEFAULTS
from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.full_corpus_inference import tensor_digest
from motionbricks.data.native_mb_online_live import online_command_goal
from motionbricks.data.root_path_terrain import RootPathTerrainExecution
from motionbricks.data.sole_region_contact import sole_region_indices
from motionbricks.data.soma_lab import SomaLabSession
from motionbricks.data.soma_static_execution import restore_native_transformers,use_dense_transformers
from motionbricks.data.terrain_distance_volume import swept_surface_points
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.soma_pfnn_maps_server import PFNNMapRuntime


def save(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def percentiles(values):
    return dict(n=len(values),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95))) if values else None


@torch.inference_mode()
def proofs(runtime,reference_source):
    spec=importlib.util.spec_from_file_location('motionbricks.data._pre_speed_reference',reference_source)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    report=dict(disabled=[],enabled=[])
    base=runtime.runtime_contract()['defaults']
    for refresh,count in ((4,2),(24,3)):
        arms=[]
        for old in (True,False):
            session=SomaLabSession(runtime,'terrain')
            if old:session.backend=reference.TunablePFNNBackend(runtime.unified_controller,
                runtime.terrain_initial_history.clone(),runtime.terrain_scene)
            else:session.backend.configure_governor(dict(DEFAULTS,enabled=False))
            packets=[]
            for index in range(count):
                metadata,payload=session.plan(dict(type='plan',mode='terrain',request_id=index+1,
                    velocity_xz=[0.,3.],heading=0.,runtime_controls=dict(base,refresh_frames=refresh)))
                pending=session.pending_playback
                packets.append((pending['motion'].clone(),pending['history'].clone(),payload))
                if not old:
                    assert session.backend.governor_state.frame==index*8
                    assert pending['governor_state'].frame==(index+1)*8
                session.acknowledge_played(dict(type='played',request_id=index+1,frame_count=8))
                if not old:assert session.backend.governor_state.frame==(index+1)*8
            arms.append(packets)
        for a,b in zip(*arms):
            assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]
        report['disabled'].append(dict(refresh_frames=refresh,packets=count,motion_history_payload_bitwise=True))
    for strength in (.5,1.):
        session=SomaLabSession(runtime,'terrain')
        session.backend.configure_governor(dict(DEFAULTS,enabled=True,strength=strength))
        before=session.backend.governor_state
        first,_=session.plan(dict(type='plan',mode='terrain',request_id=1,velocity_xz=[0.,3.],heading=0.,runtime_controls=base))
        assert first['debug']['speed_governor']['enabled']
        assert first['debug']['speed_governor']['target_speed_mps']<3.
        assert session.backend.governor_state is before
        pending=session.pending_playback
        serial=copy.copy(session.backend)
        serial.history=pending['history'].clone()
        serial.governor_state=pending['governor_state']
        expected=serial.predict_plan([[0.,3.]],[0.])
        next_meta,_=session.prefetch(dict(type='prefetch',mode='terrain',request_id=2,source_request_id=1,
            played_frame_count=2,velocity_xz=[0.,3.],heading=0.))
        prepared=pending['prepared']
        assert session.backend.governor_state is before
        assert torch.equal(prepared['motion'],expected['planned_motion'])
        assert torch.equal(prepared['history'],expected['published_history'])
        assert prepared['governor_state']==expected['governor_state_after']
        session.acknowledge_played(dict(type='played',request_id=1,frame_count=8))
        assert session.backend.governor_state==pending['governor_state']
        session.acknowledge_played(dict(type='played',request_id=2,frame_count=8))
        assert session.backend.governor_state==prepared['governor_state']
        assert session.backend.governor_state.frame==16
        report['enabled'].append(dict(strength=strength,serial_prefetch_bitwise=True,
            state_unchanged_until_ack=True,ack_frames=16,split_graph_eligible=session.backend.execution.supports(base),
            first_target=first['debug']['speed_governor']['target_speed_mps']))
    # The same governed intent is fed through the original and split network paths.
    runner=runtime.inference._terrain_execution
    native=copy.copy(runner);native.__class__=RootPathTerrainExecution
    decoder=runtime.inference._vqvae_pose_model.forward_decoder
    captured={}
    def decode(ids,*args,**kwargs):
        captured['tokens']=ids.clone();return decoder(ids,*args,**kwargs)
    runtime.inference._vqvae_pose_model.forward_decoder=decode
    report['input_execution']=[]
    try:
        for strength in (.5,1.):
            session=SomaLabSession(runtime,'terrain')
            roots=session.backend.history[0,:,:5].cpu().numpy()
            target,_,info=runtime.speed_map().govern(roots,[0.,3.],session.backend.governor_state,
                dict(DEFAULTS,enabled=True,strength=strength),4)
            assert info['enabled'] and info['target_speed_mps']<info['requested_speed_mps']
            goal=online_command_goal(session.backend.history[...,:5],target,0.,8,runner.protocol_version)
            restore_native_transformers(runtime.inference)
            with runtime.unified_controller.context(session.backend.history,runtime.terrain_scene):
                expected=native._prediction(session.backend.history,goal)
            tokens=captured['tokens'].clone()
            actual=runner._prediction(session.backend.history,goal)
            poses=[runtime.inference.motion_rep.inverse(row['motion'][:,:32],False,return_all=False)['posed_joints']
                   for row in (expected,actual)]
            error=float((poses[0]-poses[1]).norm(dim=-1).max())
            assert torch.equal(tokens[:,:8],actual['tokens'][:,:8]) and error<=3e-5
            report['input_execution'].append(dict(strength=strength,target=target,goal_velocity=goal[0,:2].cpu().tolist(),
                tokens_equal=True,fk_max_error_m=error,root_passes=1))
    finally:
        runtime.inference._vqvae_pose_model.forward_decoder=decoder
        use_dense_transformers(runtime.inference)
    return report


@torch.inference_mode()
def geometry_metrics(runtime,world,origin,direction,target):
    field=runtime.unified_controller.scene_field(runtime.terrain_scene)
    surfaces,*_=official_sole_surfaces(runtime.inference.global_motion_rep.skeleton)
    soles=np.concatenate([sole_region_indices(surfaces,side) for side in ('Left','Right')])
    peak=0.;unknown=uncertain=checked=over=0;foot_parts=[];foot_gaps=[];foot_valid=[]
    sole_peak=0.;sole_total=sole_known=sole_uncertain=sole_over=0
    frame_peaks=[]
    origin=world.new_tensor(origin);direction=world.new_tensor(direction)
    for start in range(3,world.shape[1]-1,8):
        part=world[:,start:min(start+9,world.shape[1])]
        pose=runtime.inference.motion_rep.inverse(part,False,return_all=False)
        body=runtime.skinner.skin(pose['global_joint_rots'][0],pose['posed_joints'][0],rot_is_global=True)
        points=swept_surface_points(body,8)
        root_swept=swept_surface_points(part[0,:,:3,None].transpose(1,2),8)[:,0]
        progress=((root_swept[:,[0,2]]-origin)*direction).sum(-1)
        keep=progress<=target+1e-6
        points=points[keep]
        if points.numel():
            gap,known=heightfield_gap(points,runtime.terrain_scene)
            unknown+=int((~known).sum());checked+=int(known.sum())
            inside=known&(gap<0)
            depth=torch.zeros_like(gap)
            unassessed=torch.zeros_like(known)
            if inside.any():
                distances,_,valid=field.query(points[inside])
                uncertain+=int((~valid).sum())
                unassessed[inside]=~valid
                depth[inside]=torch.where(valid,(-distances).clamp_min(0),0.)
            sole_depth=depth[:,soles]
            sole_peak=max(sole_peak,float(sole_depth.max()))
            sole_total+=int(known[:,soles].numel());sole_known+=int(known[:,soles].sum())
            sole_uncertain+=int(unassessed[:,soles].sum());sole_over+=int((sole_depth>.02).sum())
            peak=max(peak,float(depth.max()));over+=int((depth>.02).sum())
            frame_peaks.extend(depth.amax(-1).cpu().tolist())
        feet=body[:,soles]
        gap,known=heightfield_gap(feet,runtime.terrain_scene)
        if start>3:feet,gap,known=feet[1:],gap[1:],known[1:]
        foot_parts.append(feet);foot_gaps.append(gap);foot_valid.append(known)
    feet=torch.cat(foot_parts);gap=torch.cat(foot_gaps);known=torch.cat(foot_valid)
    velocity=torch.diff(feet,dim=0)*30
    near=(gap.abs()<=.035)&known
    support=near[1:]&near[:-1]&(velocity[...,1].abs()<=.3)
    slide=velocity[...,[0,2]].norm(dim=-1)[support]
    return dict(full_vertices=runtime.vertex_count,swept_substeps=8,peak_penetration_m=peak,
        sole_peak_penetration_m=sole_peak,sole_total_samples=sole_total,sole_unknown_samples=sole_total-sole_known,
        sole_uncertain_inside_samples=sole_uncertain,sole_assessed_samples=sole_known-sole_uncertain,
        sole_over_2cm_samples=sole_over,sole_fraction_over_2cm=sole_over/max(1,sole_known-sole_uncertain),
        fraction_over_2cm=over/max(checked,1),unknown_samples=unknown,uncertain_inside_samples=uncertain,
        swept_frame_peaks_m=frame_peaks,near_surface_sole_slide_mps=dict(samples=int(slide.numel()),
            mean=float(slide.mean()) if slide.numel() else None,max=float(slide.max()) if slide.numel() else None),
        slide_definition='Official sole vertices; both endpoint gaps abs<=.035m and vertical speed abs<=.3m/s. Horizontal speed is reported; no contact labels are invented.')


@torch.inference_mode()
def supplement(a):
    """Query only official sole vertices from the saved generated worlds."""
    a.output.mkdir(parents=True,exist_ok=False)
    runtime=PFNNMapRuntime(a.bundle,a.catalog,execution='eager',initial_map='007')
    try:
        surfaces,*_=official_sole_surfaces(runtime.inference.global_motion_rep.skeleton)
        indices=np.concatenate([sole_region_indices(surfaces,side) for side in ('Left','Right')])
        ids=torch.as_tensor(indices,device=runtime.inference.device,dtype=torch.long)
        assert len(indices)==196 and len(np.unique(indices))==196
        field=runtime.unified_controller.scene_field(runtime.terrain_scene)
        rows=[]
        for folder in a.supplement:
            previous=json.loads((folder/'complete.json').read_text())
            for row in previous['rows']:
                record=torch.load(folder/f'{row["case"]}-{row["arm"]}.pt',map_location='cpu',weights_only=False)
                cut=round(row['observed_duration_s']*30)
                world=record['world'][:,:4+cut].to(runtime.inference.device)
                origin=world.new_tensor(record['case']['origin_xz_m'])
                direction=world.new_tensor(record['case']['direction'])
                total=known_count=uncertain=over=0;peak=0.
                for start in range(3,world.shape[1]-1,8):
                    part=world[:,start:min(start+9,world.shape[1])]
                    pose=runtime.inference.motion_rep.inverse(part,False,return_all=False)
                    full_body=runtime.skinner.skin(pose['global_joint_rots'][0],pose['posed_joints'][0],rot_is_global=True)
                    assert full_body.shape[1]==18056
                    body=full_body.index_select(1,ids)
                    assert body.shape[1:]==(196,3)
                    points=swept_surface_points(body,8)
                    roots=swept_surface_points(part[0,:,:3,None].transpose(1,2),8)[:,0]
                    progress=((roots[:,[0,2]]-origin)*direction).sum(-1)
                    points=points[progress<=row['target_progress_m']+1e-6]
                    assert points.shape[1:]==(196,3)
                    if not points.numel():continue
                    gap,known=heightfield_gap(points,runtime.terrain_scene)
                    total+=int(known.numel());known_count+=int(known.sum())
                    inside=known&(gap<0)
                    if inside.any():
                        distance,_,valid=field.query(points[inside]);uncertain+=int((~valid).sum())
                        depths=(-distance[valid]).clamp_min(0)
                        if depths.numel():peak=max(peak,float(depths.max()));over+=int((depths>.02).sum())
                assert total%196==0 and known_count<=total and uncertain<=known_count
                rows.append(dict(source_review=str(folder),case=row['case'],arm=row['arm'],status=row['status'],
                    progress_m=row['progress_m'],official_sole_vertices=len(indices),swept_substeps=8,
                    swept_time_samples=total//196,denominator_verified=total==(total//196)*196,
                    sole_peak_penetration_m=peak,sole_total_samples=total,sole_unknown_samples=total-known_count,
                    sole_uncertain_inside_samples=uncertain,sole_assessed_samples=known_count-uncertain,
                    sole_over_2cm_samples=over,sole_fraction_over_2cm=over/max(1,known_count-uncertain)))
                save(a.output/'progress.json',dict(rows=rows))
        save(a.output/'complete.json',dict(completed=True,rows=rows,model_generation_calls=0,
            full_body_distance_queries=0,actual_optimizer_updates=0))
    finally:runtime.close()


@torch.inference_mode()
def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    declaration=json.loads(a.declaration.read_text())
    assert all(case['contains_transition'] for case in declaration['cases'])
    save(a.output/'declaration.json',declaration)
    runtime=PFNNMapRuntime(a.bundle,a.catalog,execution='root-path-live',initial_map='007')
    before=tensor_digest(runtime.unified_controller.state_dict())
    try:
        checks=proofs(runtime,a.reference)
        save(a.output/'proofs.json',checks)
        rows=[]
        for case in declaration['cases']:
            direction=np.asarray(case['direction']);origin=np.asarray(case['origin_xz_m'])
            for label,enabled,strength in [('off',False,.5),('balanced',True,.5),('cautious',True,1.)]:
                session=SomaLabSession(runtime,'terrain')
                session.backend.configure_governor(dict(DEFAULTS,enabled=enabled,strength=strength))
                parts=[session.backend.history.clone()];packets=[];failure=None;status='incomplete';crossing=None
                for index in range(case['max_frames']//8):
                    started=time.perf_counter()
                    try:
                        metadata,payload=session.plan(dict(type='plan',mode='terrain',request_id=index+1,
                            velocity_xz=(direction*3).tolist(),heading=math.atan2(direction[0],direction[1]),
                            runtime_controls=runtime.runtime_contract()['defaults']))
                        motion=session.pending_playback['motion'].clone()
                        encoded=json.dumps(metadata)
                        elapsed=(time.perf_counter()-started)*1000
                        parts.append(motion)
                        steps=metadata['debug'].get('internal_observations',[metadata['debug']])
                        packet=dict(index=index,packet_ms=elapsed,governor=[step['speed_governor'] for step in steps],
                            neural_execution=metadata['generation']['neural_execution'])
                        packets.append(packet)
                        session.acknowledge_played(dict(type='played',request_id=index+1,frame_count=8))
                        roots=motion[0,:,[0,2]].cpu().numpy()-origin
                        progress=roots@direction
                        lateral=np.abs(roots@np.array([direction[1],-direction[0]]))
                        crossed=np.flatnonzero(progress>=case['target_progress_m'])
                        departed=np.flatnonzero(lateral>.75)
                        if len(departed) and (not len(crossed) or departed[0]<=crossed[0]):status='corridor_departure';break
                        if len(crossed):crossing=index*8+int(crossed[0])+1;status='completed';break
                    except Exception as error:failure=repr(error);status='runtime_failure';break
                world=torch.cat(parts,1)
                torch.save(dict(world=world.cpu(),packets=packets,case=case,label=label),a.output/f'{case["name"]}-{label}.pt')
                roots=world[0,3:,[0,2]].cpu().numpy()-origin
                progress=roots@direction;lateral=np.abs(roots@np.array([direction[1],-direction[0]]))
                cut=crossing if crossing is not None else len(roots)-1
                duration=cut/30
                if crossing is not None:
                    fraction=(case['target_progress_m']-progress[cut-1])/max(progress[cut]-progress[cut-1],1e-9)
                    duration=(cut-1+fraction)/30
                actual=np.linalg.norm(np.diff(roots[:cut+1],axis=0),axis=-1)*30
                infos=[info for packet in packets for info in packet['governor']]
                first_brake=next((info for info in infos if info['target_speed_mps']<info['requested_speed_mps']-1e-4),None)
                speed_errors=[]
                for packet in packets:
                    for sub,info in enumerate(packet['governor']):
                        begin=packet['index']*8+sub*4
                        values=actual[begin:min(begin+4,len(actual))]
                        if len(values):speed_errors.extend((values-info['target_speed_mps']).tolist())
                if status=='incomplete' and (float(progress[-1])<.2 or (len(progress)>60 and float(progress[-1]-progress[-61])<.05)):
                    status='stalled'
                score_world=world[:,:4+cut]
                metric=geometry_metrics(runtime,score_world,case['origin_xz_m'],case['direction'],case['target_progress_m']) if cut else None
                row=dict(case=case['name'],arm=label,settings=dict(DEFAULTS,enabled=enabled,strength=strength),
                    target_progress_m=case['target_progress_m'],progress_m=float(progress[cut]),
                    lateral_max_m=float(lateral[:cut+1].max()),travelled_m=float(np.linalg.norm(np.diff(roots[:cut+1],axis=0),axis=-1).sum()),
                    elapsed_to_endpoint_s=duration if crossing is not None else None,observed_duration_s=cut/30,
                    actual_speed_mean_mps=float(actual.mean()) if len(actual) else 0.,
                    actual_target_speed_error_rmse_mps=float(np.sqrt(np.mean(np.square(speed_errors)))) if speed_errors else None,
                    status=status,failure=failure,first_braking=first_brake,geometry=metric,
                    governor_step_overhead_ms=percentiles([info['overhead_ms'] for info in infos]),
                    packet_ms=percentiles([packet['packet_ms'] for packet in packets]),packets=packets)
                rows.append(row);save(a.output/f'{case["name"]}-{label}.json',row)
                save(a.output/'progress.json',dict(completed=len(rows),rows=rows))
                print(json.dumps({key:row[key] for key in ('case','arm','status','progress_m','elapsed_to_endpoint_s','actual_speed_mean_mps','failure')}),flush=True)
        after=tensor_digest(runtime.unified_controller.state_dict())
        assert before==after
        save(a.output/'complete.json',dict(passed=True,proofs=checks,rows=rows,weights_unchanged=True,
            model_tensor_hash=before,actual_optimizer_updates=0,quality_approved=False))
    finally:runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bundle','catalog','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('declaration','reference'):p.add_argument('--'+name,type=Path)
    p.add_argument('--supplement',type=Path,nargs='+')
    a=p.parse_args()
    try:supplement(a) if a.supplement else main(a)
    except Exception as error:
        if a.output.is_dir():save(a.output/'failure.json',dict(error=repr(error)))
        raise
