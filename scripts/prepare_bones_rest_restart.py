"""Select genuine uninterrupted rest->move sequences with explicit weak commands."""

import argparse
import json
import math
from pathlib import Path

import torch

from motionbricks.data.bones_seed_rehearsal import BonesSeedPool,group
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=False)
    pool=BonesSeedPool(args.dataset,build_native_representation(args.stats),args.stats)
    geometry=JointGeometryObjective(pool.datasets['train'].rep,'cpu')
    events=json.loads(args.audit.read_text())['examples']
    indexed={row['source_id']:i for i,row in enumerate(pool.datasets['train'].records)}
    events=[e for e in events if e['domain']=='bones' and e['split']=='train' and e['restart']]
    events.sort(key=lambda e:(-(e['rest_end']-e['rest_start']),e['source_id'],e['rest_start']))
    audit=[];selected=[];families=set()
    for event in events:
        label=event['source_id'];begin=event['rest_start'];end_rest=event['rest_end']
        if label not in indexed:continue
        item=pool._item('train',indexed[label]);family=group(item['row'])
        if family in pool.contract['reserved_recording_groups'] or family in families:continue
        record=dict(source_id=label,source_group=family,begin=begin,rest_end=end_rest)
        if event.get('category') in ('backward','lateral'):continue
        if end_rest-begin<40:continue
        # Request motion at the last boundary before the demonstrated rest ends.
        # Never require uncommanded acceleration before the MOVE request.
        restart_block=math.floor((end_rest-begin-4)/8)
        segments=restart_block+9
        length=(segments-1)*8+33
        if begin+length>len(item['world']):
            audit.append(dict(record,rejected='insufficient genuine moving tail'));continue
        world=item['world'][begin:begin+length].clone()
        rest=world[:end_rest-begin]
        yaw=torch.atan2(world[3,4],world[3,3])
        angles=torch.atan2(rest[:,4],rest[:,3])-yaw
        angles=torch.atan2(angles.sin(),angles.cos())
        velocity=(world[end_rest-begin+36,[0,2]]-world[end_rest-begin+6,[0,2]])
        speed=float(velocity.norm())
        if (float(torch.diff(rest[:,:3],dim=0)[:3,[0,2]].norm(dim=-1).mean()*30)>.12
                or float(angles.abs().max())>math.radians(20) or not .3<=speed<=3.):
            audit.append(dict(record,rejected='ambiguous rest/facing/cruise semantics',speed=speed));continue
        travel=torch.atan2(velocity[0],velocity[1])
        facing_error=abs(float(torch.atan2((travel-yaw).sin(),(travel-yaw).cos())))
        final_heading=torch.atan2(world[-32:,4].mean(),world[-32:,3].mean())
        final_error=abs(float(torch.atan2((travel-final_heading).sin(),(travel-final_heading).cos())))
        if final_error>math.radians(25):
            audit.append(dict(record,rejected='moving tail does not face requested travel',facing_error_degrees=math.degrees(final_error)));continue
        metrics=exact_skin_metrics(geometry,world[None],item['scene'])
        if metrics['body_prism_max_m']>.02 or metrics['unknown_body_fraction'] or metrics['uncertain_inside_samples']:
            audit.append(dict(record,rejected='source exact-skin screen',metrics=metrics));continue
        controls=world.new_zeros(segments,4);controls[:,2]=yaw.cos();controls[:,3]=yaw.sin()
        controls[restart_block:,:2]=velocity
        controls[restart_block:,2]=travel.cos();controls[restart_block:,3]=travel.sin()
        flags=controls[:,:2].norm(dim=-1)>1e-5
        if int((~flags[:-1]&flags[1:]).sum())!=1:raise ValueError('Expected one real rest->MOVE command event')
        selected.append(dict(world=world,scene=item['scene'],row=item['row'],
            controls=controls,category='rest_restart',label=label,source_start_frame=begin,
            training_segments=segments,restart_block=restart_block,native_supervision_blocks=[restart_block],
            pilot_split='train' if len(selected)<4 else 'held_out_this_probe'))
        families.add(family)
        audit.append(dict(record,selected=True,metrics=metrics,rest_seconds=(end_rest-begin)/30,
            command_restart_frame=restart_block*8+3,source_restart_frame=end_rest-begin,
            training_segments=segments,source_frames=length,weak_velocity_xz_mps=velocity.tolist(),
            requested_turn_degrees=math.degrees(facing_error)))
        write_json(args.output/'selection.json',dict(rows=audit))
        if len(selected)==6:break
    write_json(args.output/'selection.json',dict(rows=audit))
    if len(selected)<6:raise ValueError('Need4train+2distinct-family probe holdouts; do not relax source gates')
    torch.save(selected,args.output/'cases.pt')
    write_json(args.output/'complete.json',dict(cases_sha256=sha256(args.output/'cases.pt'),
        source_manifest_sha256=pool.contract['manifest_sha256'],rows=audit,
        control='zero/rest facing through real rest;MOVE velocity and desired facing from post-rest planar trajectory, as camera-relative WASD; future body only audited,not conditioned',
        source_frames_modified=False,source_future_body_in_actor=False,
        holdout_scope='Original train sources split by recording family for this new probe; not unseen pretraining',
        command_transition_count=6,synthetic_history_or_padding=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','stats','audit','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
