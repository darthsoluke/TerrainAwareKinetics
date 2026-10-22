"""Retained route regression and factual generated-history command evaluation."""

import argparse
import json
import math
from pathlib import Path
import time

import torch

from motionbricks.data.full_corpus_inference import inference_receipt,load_candidate,tensor_digest
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_terrain import UnifiedTerrainController,UnifiedTerrainBackend,PROFILE
from scripts.audit_native_terrain_gaps import route_audit,continuity,save_world
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


def flat_commands(heading):
    schedule=[('idle',60,0.,heading),('walk',15,1.6,heading),('stop',15,0.,heading),
        ('run',15,2.4,heading),('turn90',15,1.6,heading+math.pi/2),('final_stop',15,0.,heading+math.pi/2)]
    for label,blocks,speed,yaw in schedule:
        for _ in range(blocks):
            yield label,[speed*math.sin(yaw),speed*math.cos(yaw)],yaw


@torch.no_grad()
def flat_review(controller,idle,geometry,output,smoke):
    seed=idle['world'][None,:4].to(controller.inference.device)
    heading=float(torch.atan2(seed[0,-1,4],seed[0,-1,3]))
    backend=UnifiedTerrainBackend(controller,seed.clone(),idle['scene'])
    backend.configure_runtime(dict(PROFILE))
    parts,boundaries,commands=[seed.clone()],[],[]
    failure=None
    for block,(label,velocity,yaw) in enumerate(flat_commands(heading)):
        if smoke and block>=2:
            break
        started=time.perf_counter()
        try:
            result=backend.predict_plan(velocity,yaw)
            packet=result['planned_motion']
            if packet.shape!=(1,8,seed.shape[-1]) or not torch.isfinite(packet).all():
                raise ValueError('Runtime packet shape or finiteness changed')
            parts.append(packet.clone())
            backend.history=result['published_history'].clone()
            commands.append(dict(block=block,first_generated_frame=4+8*block,label=label,
                velocity_xz_mps=velocity,heading_rad=yaw,offline_call_ms=(time.perf_counter()-started)*1000))
            boundaries.append(dict(frame=4+8*block,neural_generation=backend.last_generation['neural_generation']))
        except (ValueError,RuntimeError) as error:
            failure=repr(error)
            break
    world=torch.cat(parts,1)
    metrics=exact_skin_metrics(geometry,world,idle['scene'])
    source_root=world[0,:,:3]
    intervals=[]
    for command in commands:
        first=command['first_generated_frame']
        velocity=(source_root[first:first+8][:,[0,2]]-source_root[first-1:first+7][:,[0,2]])*30
        direction=velocity.new_tensor([math.sin(command['heading_rad']),math.cos(command['heading_rad'])])
        intervals.append(dict(block=command['block'],label=command['label'],
            mean_speed_mps=float(velocity.norm(dim=-1).mean()),
            mean_forward_speed_mps=float((velocity@direction).mean())))
    save_world(output/'recordings','flat-idle16s-walk-stop-run-turn-stop',world,idle['scene'],'actual_generated_feedback')
    return dict(actual_frames=world.shape[1],smoke_only=smoke,failure=failure,commands=commands,
        motion_by_packet=intervals,skin=metrics,continuity=continuity(controller.inference.global_motion_rep,world,boundaries),
        completed_script=len(commands)==135,quality_approved=False,
        scope='Serial scripted commands and actual generated feedback; not browser WASD, prefetch or30fps evidence.')


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)
    preparation=json.loads((a.evaluation/'complete.json').read_text())
    if (not preparation['preparation_completed'] or sha256(a.cases)!=preparation['regression_cases_sha256']
            or sha256(a.evaluation/'heldout-cases.pt')!=preparation['heldout_cases_sha256']):
        raise ValueError('Evaluation source identities differ')
    saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if bool(saved['contract']['mechanics_only'])!=a.smoke:
        raise ValueError('Mechanics checkpoint requires smoke mode; full candidates require full evaluation')
    bundle_hash=sha256(a.bundle)
    checkpoint_hash=sha256(a.checkpoint)
    receipt=inference_receipt(saved,checkpoint_hash,checkpoint_hash,bundle_hash,allow_mechanics=a.smoke)
    inference,bundle=load_terrain_bundle(a.bundle,'cuda:0')
    controller=UnifiedTerrainController(inference,bundle)
    load_candidate(controller,saved,receipt,bundle_hash,allow_mechanics=a.smoke)
    before=tensor_digest(controller.state_dict())
    original=torch.load(a.cases,map_location='cpu',weights_only=False)
    idle=next(case for case in original if case['name']=='idle')
    regression=[case for case in original if case.get('entry') in ('up','down')]
    heldout=torch.load(a.evaluation/'heldout-cases.pt',map_location='cpu',weights_only=False)
    if a.smoke:
        regression=[next(case for case in regression if case['entry']==entry) for entry in ('up','down')]
        heldout=[]
    cases=regression+heldout+[idle]
    geometry=JointGeometryObjective(inference.global_motion_rep,'cuda:0')
    routes=route_audit(controller,cases,[],geometry,a.output,2 if a.smoke else 60)
    exposure={row['label']:row for row in preparation['regression_sources']}
    for row in routes['rows']:
        row.pop('latest_bank_trained_source',None)
        row.pop('scene_present_in_latest_bank',None)
        row['evaluation_role']='legacy_regression' if row['label'] in exposure else 'current_corpus_held_out'
        row['current_corpus_group_split']=exposure[row['label']]['current_corpus_group_split'] \
            if row['label'] in exposure else next(case['native_split'] for case in heldout if case['label']==row['label'])
        row['lifecycle_unseen']=False if row['label'] in exposure else 'not_exhaustively_audited'
    routes['route_scope']='Legacy regression sources and separately identified current-corpus held-out sources.'
    write_json(a.output/'routes.json',routes)
    flat=flat_review(controller,idle,geometry,a.output,a.smoke)
    if tensor_digest(controller.state_dict())!=before:
        raise ValueError('Offline evaluation changed model tensors')
    write_json(a.output/'complete.json',dict(evaluation_completed=True,smoke_only=a.smoke,
        checkpoint_sha256=checkpoint_hash,model_state_sha256=before,weights_unchanged=True,
        source_preparation_sha256=sha256(a.evaluation/'complete.json'),
        heldout_target_counts_met=preparation['target_counts_met'],routes=routes,flat=flat,
        runtime_profile=dict(PROFILE),actual_optimizer_updates=0,
        quality_approved=False,interactive_server_started=False))
    controller.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','checkpoint','cases','evaluation','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output/'failure.json',dict(error=repr(error)))
        raise
