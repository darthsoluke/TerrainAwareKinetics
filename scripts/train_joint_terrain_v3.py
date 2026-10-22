"""Resumable native MB joint model learning, with real FK-history feedback."""

import argparse
from collections import Counter
from contextlib import contextmanager,nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,Sampler

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_learning_data import TerrainLearningDataset,learning_sampling_weights,collate_learning_items,source_task_intent,sample_aligned_world_windows,sampling_manifest
from motionbricks.data.terrain_model_v3 import JointTerrainController,SCHEMA
from motionbricks.data.terrain_learning_episode import TerrainLearningEpisode
from motionbricks.data.terrain_training_execution import AsyncCheckpointWriter,validate_execution_migration
from motionbricks.data.terrain_sample_budget import TerrainSampleBudget
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_quality_selection import summarize,quality_score
from motionbricks.data.terrain_control import SoftCommandState,demonstration_controls,CONTROL_VERSION
from motionbricks.data.terrain_traversal_metrics import TRAVERSAL_PROTOCOL
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


@contextmanager
def geometry_precision():
    old=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:yield
    finally:torch.backends.cuda.matmul.allow_tf32=old


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


class ReplayableSampler(Sampler):
    def __init__(self,weights,steps,global_batch,rank,world,start_step,draw_offset=0):
        self.weights=weights;self.limit=draw_offset+steps*global_batch
        self.start=draw_offset+start_step*global_batch+rank;self.world=world

    def __iter__(self):
        generator=torch.Generator().manual_seed(2333)
        for offset in range(0,self.limit,65536):
            # Replacement multinomial has the same seeded stream across chunks.
            # Avoid allocating the entire 50k large-batch index array per rank.
            draws=torch.multinomial(self.weights,65536,replacement=True,generator=generator)
            first=max(self.start,offset+(self.start-offset)%self.world)
            for i in range(first,min(offset+len(draws),self.limit),self.world):yield int(draws[i-offset]),i

    def __len__(self):return max(0,(self.limit-self.start+self.world-1)//self.world)


def checkpoint(path,model,optimizer,step,contract,coefficients,counts,rank,world,selection,*,data_step,episodes,writer=None):
    rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None,
         'numpy':np.random.get_state(),'python':random.getstate(),'data_step':data_step,
         'episodes':[e.state_dict() for e in episodes],'counts':dict(counts)}
    gathered=[None]*world
    if world>1:torch.distributed.all_gather_object(gathered,rng)
    else:gathered=[rng]
    if rank==0:
        value={'schema':SCHEMA,'model_state':model.state_dict(),'optimizer':optimizer.state_dict(),
            'step':step,'contract':contract,'coefficients':coefficients,'counts':dict(counts),
            'rank_rng':gathered,'selection':selection}
        if writer is None:AsyncCheckpointWriter.write(path,value)
        else:writer.submit(path,value)
    if world>1:torch.distributed.barrier()


def grad_norm(value,parameters):
    gradients=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
    return sum(float(g.detach().square().sum()) for g in gradients if g is not None)**.5


def intents(items,windows,frames,device,history=None,state=None):
    controls=torch.cat([demonstration_controls(window[None],frames,
        category=item['row'].get('control_category'),
        override=item['override'][None].to(device) if item['override'] is not None else None,
        reference=item['world'][None].to(device) if item['row'].get('control_category')=='start' else None)
        for item,window in zip(items,windows)],0)
    if any(item['override'] is not None for item in items):
        raise ValueError('Weak wall velocity labels do not define the requested path for fixed-world feedback supervision')
    return (state or SoftCommandState()).from_controls(windows[:,:4] if history is None else history,controls,frames//4,
        goal_xz=windows[:,frames-1,[0,2]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dataset','pairs','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=50000)
    parser.add_argument('--stop-at',type=int,help='Checkpoint and exit for review, keeping the full schedule contract')
    parser.add_argument('--global-batch',type=int,default=32)
    parser.add_argument('--micro-batch',type=int,default=8)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--adapter-warmup',type=int,default=2000)
    parser.add_argument('--lr-warmup',type=int,default=2000)
    parser.add_argument('--feedback-start',type=int,default=2000)
    parser.add_argument('--feedback-ramp',type=int,default=8000)
    parser.add_argument('--feedback-depth',type=int,default=13)
    parser.add_argument('--sequence-frames',type=int,default=257)
    parser.add_argument('--yaw-augmentation',action='store_true')
    parser.add_argument('--reuse-loss-calibration',action='store_true')
    parser.add_argument('--optimized-geometry',action='store_true')
    parser.add_argument('--fused-adam',action='store_true')
    parser.add_argument('--async-checkpoints',action='store_true')
    parser.add_argument('--final-evaluation-only',action='store_true')
    parser.add_argument('--skip-quality-evaluation',action='store_true',help='Explicit user trial: save final weights without offline quality evaluation')
    parser.add_argument('--cache-mib',type=int,default=2048)
    parser.add_argument('--validate-every',type=int,default=5000)
    parser.add_argument('--checkpoint-every',type=int,default=250)
    parser.add_argument('--evaluation-seconds',type=int,default=10)
    parser.add_argument('--native-only',action='store_true')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--diagnostic-no-evaluation',action='store_true',help='Code-only smoke, limited to at most eight updates; never a quality approval')
    parser.add_argument('--weak-fraction',type=float,default=0.,help='Auxiliary synthetic wall pair mass; real-source main run uses zero')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--repartition-resume',action='store_true',help='Explicitly resume optimizer/sample clock with a changed DDP layout')
    parser.add_argument('--amend-execution',action='store_true',help='Resume all saved state with an explicitly amended execution policy')
    parser.add_argument('--migrate-execution',type=Path,help='Preserve optimizer and schedule, explicitly repartition the data stream')
    parser.add_argument('--sample-budget',type=Path,help='Explicit authorized sample-exposure budget; never label its clock as actual updates')
    parser.add_argument('--initialize',type=Path,help='Explicit model-only transfer into a new optimizer/budget contract')
    parser.add_argument('--evaluate-initial',action='store_true')
    parser.add_argument('--adaptive-unfreeze',action='store_true')
    parser.add_argument('--max-adapter-updates',type=int,default=8000)
    parser.add_argument('--max-stale-reviews',type=int,default=4)
    args=parser.parse_args()
    if args.repartition_resume and args.resume is None:parser.error('Repartition requires a complete resume checkpoint')
    if args.amend_execution and (args.resume is None or args.repartition_resume):parser.error('Execution amendment requires an exact-state resume')
    if args.skip_quality_evaluation and (not args.final_evaluation_only or args.evaluate_initial or args.adaptive_unfreeze):
        parser.error('Skipping quality reviews requires explicit final-only execution without adaptive selection')
    sample_plan=json.loads(args.sample_budget.read_text()) if args.sample_budget else None
    sample_budget=None
    if sample_plan is not None:
        sample_budget=TerrainSampleBudget(**{key:sample_plan[key] for key in
            ('reference_batch','reference_updates','retained_examples','global_batch')})
        if args.global_batch!=sample_budget.global_batch:parser.error('Global batch differs from sample budget')
        expected=sample_plan['anchor_update']+sample_budget.remaining_updates
        if args.steps!=expected:parser.error(f'Sample budget requires --steps {expected}')
        if not args.final_evaluation_only:parser.error('Sample-budget campaign performs only final quality evaluation')
    if args.final_evaluation_only and (args.evaluate_initial or args.adaptive_unfreeze):parser.error('Final-only execution cannot request initial/adaptive reviews')
    if sum(x is not None for x in (args.resume,args.initialize,args.migrate_execution))>1:parser.error('Choose exactly one resume, initialization or execution migration')
    if args.fused_adam and args.device!='cuda':parser.error('Fused Adam requires CUDA')
    if args.weak_fraction!=0:parser.error('This goal-supervised run requires real source trajectories, not weak wall pairs')
    if args.feedback_depth<1 or args.sequence_frames<65:parser.error('Require positive feedback depth and at least 65 real frames')
    if args.diagnostic_no_evaluation and (args.stop_at is None or args.stop_at>8 or args.adaptive_unfreeze or args.evaluate_initial):
        parser.error('Skipping quality evaluation is allowed only in an explicit <=8-update code diagnostic')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local_rank=int(os.environ.get('LOCAL_RANK',0))
    if args.global_batch%(world*args.micro_batch) or min(args.steps,args.validate_every,args.checkpoint_every,args.micro_batch)<1:
        parser.error('Positive budgets and exact global/micro/DDP batch divisibility are required')
    if args.output.exists() and args.resume is None:parser.error('Use a new run or explicitly resume')
    if args.resume and args.initialize:parser.error('Resume and model-only initialization are distinct operations')
    torch.set_num_threads(2 if args.device=='cpu' else 4);torch.set_float32_matmul_precision('high')
    if args.device=='cuda':torch.cuda.set_device(local_rank)
    if world>1:torch.distributed.init_process_group('nccl' if args.device=='cuda' else 'gloo',timeout=timedelta(minutes=30))
    torch.manual_seed(2333);np.random.seed(2333);random.seed(2333)
    device=torch.device('cuda',local_rank) if args.device=='cuda' else torch.device('cpu')
    inference,bundle=load_terrain_bundle(args.source,device)
    if sha256(args.dataset/'native_manifest.json')!=bundle['metadata']['dataset_manifest_sha256']:
        raise ValueError('Joint corpus differs from source model identity')
    model=JointTerrainController(inference,bundle)
    model.set_learning_phase(False)
    frozen={'encoder':state_digest(model.codec.encoder),'quantizer':state_digest(model.codec.quantizer)}
    source_rep=build_native_representation(args.dataset/'stats')
    data=TerrainLearningDataset(args.dataset,source_rep,'train',args.pairs,max_frames=args.sequence_frames,cache_bytes=args.cache_mib*1024**2)
    sample_weights=learning_sampling_weights(data,args.weak_fraction)
    sample_contract=sampling_manifest(data,sample_weights)
    validation=TerrainLearningDataset(args.dataset,source_rep,'validation',args.pairs,cache_bytes=512*1024**2)
    geometry=JointGeometryObjective(inference.global_motion_rep,device)
    initialization=None;initial_coefficients=None
    if args.initialize:
        initial=torch.load(args.initialize,map_location=device,weights_only=False)
        expected=initial['contract']
        if (initial['schema']!=SCHEMA or expected['source_bundle_sha256']!=sha256(args.source)
            or expected['dataset_manifest_sha256']!=sha256(args.dataset/'native_manifest.json')
            or expected['pair_manifest_sha256']!=sha256(args.pairs/'manifest.json')):
            raise ValueError('Initialization source/corpus/weak-label identity differs')
        model.load_state_dict(initial['model_state'],strict=True)
        if frozen!={'encoder':state_digest(model.codec.encoder),'quantizer':state_digest(model.codec.quantizer)}:
            raise ValueError('Initialization changed the frozen motion vocabulary')
        initialization={'path':str(args.initialize),'sha256':sha256(args.initialize),'step':initial['step'],
            'optimizer':'fresh Adam; this is explicit transfer, not exact resume'}
        if args.reuse_loss_calibration:initial_coefficients=initial['coefficients']
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=5e-5,fused=args.fused_adam)
    start_step=0;coefficients=initial_coefficients;counts=Counter()
    selection={'interface_ready':not args.adaptive_unfreeze,'initial':None,'best_score':None,
               'best_step':None,'stale_reviews':0,'evaluated_step':None,'stop_reason':None}
    resume=None
    if args.resume or args.migrate_execution:
        resume=torch.load(args.resume or args.migrate_execution,map_location='cpu',mmap=True,weights_only=False)
        if resume['schema']!=SCHEMA:raise ValueError('Incompatible resume model schema')
        model.load_state_dict(resume['model_state'],strict=True);optimizer.load_state_dict(resume['optimizer'])
        start_step=resume['step'];coefficients=resume['coefficients'];counts.update(resume['counts'])
        selection.update(resume.get('selection',{}))
        initialization=resume['contract'].get('initialization')
    if sample_plan is not None:
        if resume is None:raise ValueError('Sample-budget execution requires a retained optimizer checkpoint')
        if args.migrate_execution and (start_step!=sample_plan['anchor_update'] or
                sha256(args.migrate_execution)!=sample_plan['checkpoint_sha256']):
            raise ValueError('Sample budget is bound to another retained checkpoint')
    if args.fused_adam:
        for group in optimizer.param_groups:group.update(fused=True,foreach=False)
        for state in optimizer.state.values():
            if torch.is_tensor(state.get('step')):state['step']=state['step'].to(device)
    migration=resume['contract'].get('execution_migration') if resume else None
    draw_offset=resume['contract'].get('sampler_draw_offset',0) if resume else 0
    if args.migrate_execution or args.repartition_resume:
        draw_offset+=max(r['data_step'] for r in resume['rank_rng'])*resume['contract']['global_batch']
        migration_checkpoint=args.migrate_execution or args.resume
        migration={'checkpoint':str(migration_checkpoint.resolve()),'sha256':sha256(migration_checkpoint),
            'preserved_updates':start_step,'optimizer_preserved':True,'old_world_size':resume['contract']['world_size'],
            'old_global_batch':resume['contract']['global_batch'],'old_rank_counts':[r['counts'] for r in resume['rank_rng']],
            'history_policy':'explicit world/batch repartition starts new factual episodes; no claim of bitwise replay',
            'authorization':'user requested all free A100 resources and evaluation only at cumulative 50000 updates'}
        counts=Counter();selection.update(stop_reason=None,stale_reviews=0)
    contract={'schema':SCHEMA,'source_bundle_sha256':sha256(args.source),'source':str(args.source),
        'dataset':str(args.dataset),'dataset_manifest_sha256':sha256(args.dataset/'native_manifest.json'),
        'pairs':str(args.pairs),'pair_manifest_sha256':sha256(args.pairs/'manifest.json'),
        'steps':args.steps,'global_batch':args.global_batch,'micro_batch':args.micro_batch,'world_size':world,
        'workers':args.workers,'seed':2333,'native_only':args.native_only,'initialization':initialization,
        'device_type':args.device,'diagnostic_no_evaluation':args.diagnostic_no_evaluation,
        'execution_options':{'optimized_geometry':args.optimized_geometry,'fused_adam':args.fused_adam,
            'async_checkpoints':args.async_checkpoints,'final_evaluation_only':args.final_evaluation_only,'cache_mib':args.cache_mib},
        'execution_migration':migration,'sampler_draw_offset':draw_offset,
        'sample_budget':sample_plan,
        'code_sha256':{name:sha256(Path(__file__).parents[1]/name) for name in (
            'scripts/train_joint_terrain_v3.py','scripts/evaluate_joint_terrain_v3.py',
            'motionbricks/data/terrain_model_v3.py','motionbricks/data/terrain_joint_objective.py',
            'motionbricks/data/terrain_learning_data.py','motionbricks/data/terrain_distance_volume.py',
            'motionbricks/data/terrain_quality_selection.py',
            'motionbricks/data/terrain_control.py',
            'motionbricks/data/terrain_traversal_metrics.py',
            'motionbricks/data/terrain_scene_frame.py',
            'motionbricks/data/terrain_learning_episode.py',
            'motionbricks/data/terrain_batched_geometry.py','motionbricks/data/terrain_training_execution.py',
            'motionbricks/data/terrain_sample_budget.py',
            'motionbricks/motion_backbone/inference/motion_inference.py')},
        'far_observation':'intent plus predicted-route union; dense 201x121 surveys; 9x9 max/min relief per view; normalized far XYZ',
        'adapter_warmup':args.adapter_warmup,'lr_warmup':args.lr_warmup,
        'adaptive_unfreeze':args.adaptive_unfreeze,'max_adapter_updates':args.max_adapter_updates,
        'max_stale_reviews':args.max_stale_reviews,
        'feedback_start':args.feedback_start,'feedback_ramp':args.feedback_ramp,
        'feedback_depth':args.feedback_depth,'sequence_frames':args.sequence_frames,
        'yaw_augmentation':args.yaw_augmentation,'reuse_loss_calibration':args.reuse_loss_calibration,
        'scene_coordinates':'rigid native heading augmentation; queries/losses return to the unchanged physical scene',
        'distance_volume_reference':'one static scene crop covering the entire sampled rollout',
        'optimizer_update_unit':'one primitive per global batch; generated history persists across updates as in DART',
        'data_accounting':'50/50 weights apply to episode draws; actual per-domain primitive exposure is logged separately',
        'unknown_history_policy':'retain previous loss, restart only affected training rows from factual history at the same source time; no evaluation resets',
        'validate_every':args.validate_every,'checkpoint_every':args.checkpoint_every,
        'evaluation_seconds':args.evaluation_seconds,
        'loader_transport':'batch-owned scene copies; bounded prefetch; 120s worker timeout',
        'optimizer':'Adam5e-5, clip0.5; declared warmstart cosine schedule',
        'precision':'FP32 neural high; frozen labels, FK and skin highest',
        'frozen_vocabulary':frozen,'source_motion_unchanged':True,'postprocessing':None,
        'sampling':sample_contract,'weak_fraction':args.weak_fraction,'evaluation_goal':TRAVERSAL_PROTOCOL,
        'discrete_gradient':'hard codewords forward; soft-code straight-through geometric gradient',
        'feedback':'detached official played-FK history; unchanged source world/clock; explicit requested planar waypoint',
        'request_provenance':'source-derived planar path supervision; not captured WASD; no future height/pose',
        'source_window_sampling':'uniform complete windows; one third of weak pairs use the final complete window',
        'quality_approved':False}
    contract['control_version']=CONTROL_VERSION
    if args.skip_quality_evaluation:contract['execution_options']['skip_quality_evaluation']=True
    if args.migrate_execution or args.repartition_resume or args.amend_execution:validate_execution_migration(resume['contract'],contract,sample_budget_authorized=sample_plan is not None)
    elif resume and resume['contract']!=contract:raise ValueError('Resume training contract changed')
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/'contract.json',contract)
    if world>1:torch.distributed.barrier()
    exact_resume=resume is not None and not (args.migrate_execution or args.repartition_resume)
    data_step=resume['rank_rng'][rank]['data_step'] if exact_resume else 0
    sampler=ReplayableSampler(sample_weights,args.steps,args.global_batch,rank,world,data_step,draw_offset)
    loader=DataLoader(data,batch_size=args.micro_batch,sampler=sampler,num_workers=args.workers,
        collate_fn=collate_learning_items,pin_memory=True,persistent_workers=args.workers>0,drop_last=True,
        multiprocessing_context='spawn' if args.workers else None,
        prefetch_factor=1 if args.workers else None,timeout=120 if args.workers else 0)
    iterator=iter(loader);accumulation=args.global_batch//(world*args.micro_batch)
    sentinel=[model.root._conv_output.model[-1].weight,model.pose._proj_pose_output_logit.weight,
              model.codec.decoder.model[-1].weight]
    if coefficients is None:
        calibration=[]
        model.set_learning_phase(False)
        # Calibrate on actual training scenes. Artificial walls cannot set the
        # main objective's scale when their training probability is zero.
        parc=[i for i,r in enumerate(data.base.records) if r['origin'].startswith('parc')]
        indices=[next(i for i,r in enumerate(data.base.records) if r['origin']=='bones_seed' and r.get('control_category')=='idle'),
                 parc[0],parc[-1]]
        for index in indices:
            item=data[(index,100000+index)]
            source=item['world'][:33][None].to(device);history=source[:,:4]
            task=intents([item],source,32,device)
            result=model(history,task,[item['scene']],source=source,num_tokens=8,straight_through=True)
            with geometry_precision():
                native=model.native_losses(result,source)
                physical=geometry.losses(result['world'],source,history,[item['scene']],[item['row']['source_id']])
            record={'native':grad_norm(native['native_total'],sentinel)}
            record.update({key:grad_norm(physical[key],sentinel) for key in
                ('fk_m','boundary_mps','collision_m2','support_gap_m','source_support_velocity_mps')})
            calibration.append(record)
        native_scale=np.mean([r['native'] for r in calibration])
        if any(np.mean([r[key] for r in calibration])<1e-8 for key in calibration[0] if key!='native'):
            raise ValueError('Physical-loss calibration lacks an active training gradient; do not scale a zero signal')
        coefficients={key:float(native_scale/(5*max(np.mean([r[key] for r in calibration]),1e-8))) for key in calibration[0] if key!='native'}
        if world>1:
            shared=[coefficients if rank==0 else None];torch.distributed.broadcast_object_list(shared,src=0);coefficients=shared[0]
        if rank==0:write_json(args.output/'loss_calibration.json',{'samples':calibration,'coefficients':coefficients})
    with torch.no_grad(),geometry_precision():
        episodes=[TerrainLearningEpisode.restore(state,data,inference.global_motion_rep,device)
            for state in resume['rank_rng'][rank]['episodes']] if exact_resume else []
        if args.optimized_geometry:
            for episode in episodes:episode.prepared_geometry=geometry.prepare_episode(episode)
    # Rebuilding deterministic data can seed CUDA as well as CPU generators.
    # Restore the actual training RNG after reconstructing saved episodes.
    if exact_resume:
        rng=resume['rank_rng'][rank]
        torch.set_rng_state(rng['torch'].cpu())
        if args.device=='cuda':torch.cuda.set_rng_state(rng['cuda'].cpu())
        np.random.set_state(rng['numpy']);random.setstate(rng['python'])
        counts=Counter(rng['counts'])
    elif args.migrate_execution or args.repartition_resume:
        torch.manual_seed(2333+start_step*world+rank);np.random.seed(2333+start_step*world+rank);random.seed(2333+start_step*world+rank)
    del resume
    if episodes and len(episodes)!=accumulation:raise ValueError('Incomplete saved global batch of episodes')
    optimizer.zero_grad(set_to_none=True)
    model.set_learning_phase(False)
    wrapped=DistributedDataParallel(model,device_ids=[local_rank] if args.device=='cuda' else None,find_unused_parameters=args.adapter_warmup>0 or args.adaptive_unfreeze,
        broadcast_buffers=False,gradient_as_bucket_view=True) if world>1 else model
    writer=AsyncCheckpointWriter() if rank==0 and args.async_checkpoints else None
    def evaluate(step):
        if world>1:torch.distributed.barrier()
        if rank==0:
            from scripts.evaluate_joint_terrain_v3 import evaluate_controller
            review=args.output/f'review-{step:06d}'
            existing=review/'metrics.json'
            if existing.exists() and 'absolute_gates_passed' in json.loads(existing.read_text()):
                report=json.loads(existing.read_text())
            else:
                if review.exists():review.rename(review.with_name(review.name+f'-incomplete-{time.time_ns()}'))
                report=evaluate_controller(model,validation,geometry,review,
                    seconds=args.evaluation_seconds,max_short=16,max_walls=0,training_data=data)
            values=summarize(report)
            if selection['initial'] is None:selection['initial']=values
            initial=selection['initial']
            retained=(values['full_surface_audit'] and values['unknown_body_fraction']==0
                and values['fk_mpjpe_m']<=1.1*initial['fk_mpjpe_m']+.01
                and values['foot_gap_m']<=.05)
            interface=(retained and values['start_progress_ratio']>=.5
                and values['idle_drift_m']<=max(.1,1.1*initial['idle_drift_m']+.01))
            if args.adaptive_unfreeze and step>=args.adapter_warmup and interface:
                selection['interface_ready']=True
            score=quality_score(values)
            if retained and values['start_progress_ratio']>=.5 and (
                    selection['best_score'] is None or score<selection['best_score']):
                selection.update(best_score=score,best_step=step,stale_reviews=0)
                write_json(args.output/'best_candidate.json',{'step':step,'score':score,
                    'weights':f'weights-{step:06d}.pt','metrics':str(existing),'quality_approved':False})
            elif selection['interface_ready'] and step>args.adapter_warmup:
                selection['stale_reviews']+=1
            selection.update(evaluated_step=step,current=values)
            if args.adaptive_unfreeze:
                if step>=args.max_adapter_updates and not selection['interface_ready']:
                    selection['stop_reason']='soft task interface failed to learn while the base was protected'
                elif selection['stale_reviews']>=args.max_stale_reviews:
                    selection['stop_reason']='held-out model quality stopped improving; retain earlier candidates'
            write_json(args.output/'selection.json',selection)
        if world>1:
            shared=[selection if rank==0 else None]
            torch.distributed.broadcast_object_list(shared,src=0)
            selection.update(shared[0]);torch.distributed.barrier()
    if args.evaluate_initial or args.adaptive_unfreeze and selection['initial'] is None:
        if rank==0 and not start_step:
            torch.save({'schema':SCHEMA,'model_state':model.state_dict(),'step':0,
                'contract':contract,'coefficients':coefficients},args.output/'weights-000000.pt')
        evaluate(start_step)
    started=time.perf_counter();step_times=[];step=start_step-1
    for step in range(start_step,args.steps):
        began=time.perf_counter()
        completed=step-sample_plan['anchor_update'] if sample_plan is not None else None
        clock=sample_budget.learning_clock_after(completed,sample_plan['inherited_reference_updates']) if sample_budget else step
        model.set_learning_phase(clock<args.adapter_warmup or not selection['interface_ready'])
        if sample_budget:
            learning_rate=sample_budget.learning_rate_after(completed,sample_plan['inherited_reference_updates'],args.lr_warmup)
        else:
            learning_rate=5e-5*min(1.,(step+1)/max(1,args.lr_warmup))
            if step>=args.lr_warmup:
                progress=(step-args.lr_warmup)/max(1,args.steps-args.lr_warmup)
                learning_rate=2e-6+(5e-5-2e-6)*(.5+.5*np.cos(np.pi*progress))
        for group in optimizer.param_groups:group['lr']=learning_rate
        if not episodes:
            batches=[next(iterator) for _ in range(accumulation)]
            minimum=min(len(item['world']) for items in batches for item in items)
            if world>1:
                value=torch.tensor(minimum,device=device)
                torch.distributed.all_reduce(value,op=torch.distributed.ReduceOp.MIN);minimum=int(value)
            rng=np.random.default_rng(2333+data_step)
            tokens=8 if rng.random()<.6 else int(rng.integers(6,17))
            tokens=min(tokens,(minimum-1)//4);frames=tokens*4
            depth=1 if clock<args.feedback_start else args.feedback_depth
            segments=min(depth,1+(minimum-frames-1)//8)
            far=float(rng.uniform(3.4,20.));width=float(rng.uniform(.6,6.))
            for micro,items in enumerate(batches):
                crop_rng=np.random.default_rng(7301+data_step*world*accumulation+rank*accumulation+micro)
                _,starts,late=sample_aligned_world_windows(items,frames+1+(segments-1)*8,crop_rng)
                yaw_rng=np.random.default_rng(9917+data_step*world*accumulation+rank*accumulation+micro)
                yaw=yaw_rng.uniform(0,2*np.pi,len(items)) if args.yaw_augmentation else None
                with torch.no_grad(),geometry_precision():
                    episodes.append(TerrainLearningEpisode(items,starts,frames,segments,inference.global_motion_rep,device,
                        yaw=yaw,far=far,width=width))
                    if args.optimized_geometry:episodes[-1].prepared_geometry=geometry.prepare_episode(episodes[-1])
                counts['late_weak_windows']+=late
                counts['nonprefix_world_windows']+=sum(start>0 for start in starts)
                counts['source_clips']+=len(items)
                for item in items:
                    row=item['row'];family=row.get('control_category','unspecified') if row.get('origin')=='bones_seed' else row.get('terrain_family','unspecified')
                    counts['sampled:'+row.get('origin','weak')+':'+str(family)]+=1
            data_step+=1
        optimizer.zero_grad(set_to_none=True);logging=Counter()
        probability=min(1.,max(0.,(clock-args.feedback_start)/max(1,args.feedback_ramp))) if not args.native_only else 0.
        forcing_rng=np.random.default_rng(17333+step)
        for micro,episode in enumerate(episodes):
            teacher=episode.position==0 or forcing_rng.random()>probability
            with torch.no_grad(),geometry_precision():
                window,physical_window,physical_history,resets=episode.window(teacher)
            counts['unknown_history_teacher_resets']+=resets
            counts['teacher_primitives']+=len(episode.items) if teacher else resets
            counts['generated_primitives']+=0 if teacher else len(episode.items)-resets
            task=intents(episode.model_items,window,episode.frames,device,episode.history,episode.command_state)
            context=wrapped.no_sync() if world>1 and micro<accumulation-1 else nullcontext()
            with context:
                result=wrapped(episode.history,task,episode.scenes,source=window,num_tokens=episode.frames//4,
                    far_distance=episode.far,far_halfwidth=episode.width,straight_through=not args.native_only,
                    scene_frame=episode.frame,packed_scene=episode.packed_scene)
                with geometry_precision():
                    native=model.native_losses(result,window)
                    physical_prediction=result['world'] if episode.frame is None else episode.frame.motion(
                        inference.global_motion_rep,result['world'],inverse=True)
                    if args.optimized_geometry:
                        physical=geometry.prepared_losses(physical_prediction,physical_history,episode.prepared_geometry,
                            offset=episode.position*8,diagnostics=step%25==0 or step+1==args.steps or step+1==args.stop_at)
                    else:
                        physical=geometry.losses(physical_prediction,physical_window,physical_history,
                            episode.scenes,episode.identities,volume_source=episode.physical_source)
                loss=native['native_total']
                if not args.native_only:loss=loss+sum(coefficients[key]*physical[key] for key in coefficients)
                if not torch.isfinite(loss):raise ValueError('Nonfinite joint native/geometry loss')
                (loss/accumulation).backward()
            logging['loss']+=float(loss.detach())/accumulation
            for key in ('native_motion','native_root','token_ce'):logging[key]+=float(native[key].detach())/accumulation
            for key in coefficients:logging[key]+=float(physical[key].detach())/accumulation
            logging['unknown_surface_points']+=physical['unknown_surface_points']
            logging['source_collision_p99_m']=max(logging['source_collision_p99_m'],physical['source_collision_p99_m'])
            with torch.no_grad(),geometry_precision():
                episode.advance(reencode_played_history(result['world'][:,8:12].detach(),inference.global_motion_rep))
            counts['primitives']+=len(episode.items)
            for item in episode.items:
                row=item['row'];family=row.get('control_category','unspecified') if row.get('origin')=='bones_seed' else row.get('terrain_family','unspecified')
                counts['primitive_exposure:'+row.get('origin','weak')+':'+str(family)]+=1
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5)
        if not torch.isfinite(norm):raise ValueError('Nonfinite joint gradient')
        optimizer.step()
        if all(e.position==e.segments for e in episodes):
            episodes=[]
            del episode  # Release the last scene/cache before allocating the next episode.
        elif any(e.position>=e.segments for e in episodes):raise ValueError('Global batch episode clocks diverged')
        step_times.append(time.perf_counter()-began)
        if rank==0 and (step%25==0 or step+1==args.steps or step+1==args.stop_at):
            row={'step':step+1,**dict(logging),'lr':learning_rate,'gradient_norm':float(norm),
                 'seconds_per_update_median':float(np.median(step_times[-100:])),
                 'elapsed_seconds':time.perf_counter()-started,'counts_per_rank':dict(counts),
                 'data_draw_step':data_step,'episode_position':episodes[0].position if episodes else None,
                 'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1024**2 if args.device=='cuda' else 0.}
            if sample_budget:
                row.update(actual_cumulative_optimizer_updates=step+1+sample_plan['inherited_reference_updates'],
                    sample_budget_new_updates=completed+1,
                    primitive_examples=sample_budget.examples_after(completed+1),
                    reference_batch_equivalent_updates=sample_budget.equivalent_updates_after(completed+1),
                    sample_target=sample_budget.target_examples,feedback_probability=probability,
                    examples_per_second=((step-start_step+1)*args.global_batch)/(time.perf_counter()-started))
            write_json(args.output/'progress.json',row);print(json.dumps(row),flush=True)
        review_due=step+1==args.steps or (not args.final_evaluation_only and ((step+1)%args.validate_every==0 or step+1==args.stop_at))
        save_due=(step+1)%args.checkpoint_every==0 or review_due or step+1==args.stop_at
        if save_due:
            if frozen!={'encoder':state_digest(model.codec.encoder),'quantizer':state_digest(model.codec.quantizer)}:
                raise ValueError('The frozen token vocabulary changed')
            checkpoint(args.output/'last.pt',model,optimizer,step+1,contract,coefficients,counts,rank,world,selection,
                data_step=data_step,episodes=episodes,writer=writer)
        if review_due:
            if rank==0:
                milestone=args.output/f'weights-{step+1:06d}.pt'
                torch.save({'schema':SCHEMA,'model_state':model.state_dict(),'step':step+1,
                    'contract':contract,'coefficients':coefficients},milestone)
            if not args.diagnostic_no_evaluation and not args.skip_quality_evaluation:evaluate(step+1)
            checkpoint(args.output/'last.pt',model,optimizer,step+1,contract,coefficients,counts,rank,world,selection,
                data_step=data_step,episodes=episodes,writer=writer)
            if selection['stop_reason']:break
        if step+1==args.stop_at:break
    if writer is not None:writer.close()
    if rank==0:
        complete={'steps':step+1,'requested_steps':args.steps,
            'stop_reason':selection['stop_reason'],'counts_per_rank':dict(counts),'quality_approved':False}
        if args.skip_quality_evaluation:complete['quality_evaluation']='skipped at explicit user request for direct WASD trial'
        if sample_budget:
            complete.update(sample_budget=sample_budget.report(),actual_cumulative_optimizer_updates=step+1+sample_plan['inherited_reference_updates'],
                primitive_examples=sample_budget.examples_after(step+1-sample_plan['anchor_update']),
                reference_batch_equivalent_updates=sample_budget.equivalent_updates_after(step+1-sample_plan['anchor_update']))
        write_json(args.output/'complete.json',complete)
    if world>1:torch.distributed.destroy_process_group()


if __name__=='__main__':main()
