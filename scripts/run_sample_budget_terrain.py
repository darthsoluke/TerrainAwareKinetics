"""Finish the original 1.6M primitive-example budget with a measured larger batch."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import torch

from motionbricks.data.terrain_sample_budget import TerrainSampleBudget
from scripts.run_final_terrain_50k import require_free,save
from scripts.train_native_mb import sha256


def make_plan(task,batch):
    checkpoint=task/'control-v7-input-004000.pt'
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if saved['step']!=4000 or saved['contract']['global_batch']!=32:raise ValueError('Unexpected budget anchor')
    inherited=saved['contract']['initialization']['step']
    if inherited!=5000:raise ValueError('Unexpected inherited-update count')
    budget=TerrainSampleBudget(32,50000,(inherited+saved['step'])*32,batch)
    return {'reference_batch':32,'reference_updates':50000,'retained_examples':budget.retained_examples,
        'global_batch':batch,'anchor_update':saved['step'],'inherited_reference_updates':inherited,
        'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
        'authorization':'User explicitly chose original sample total with large-batch acceleration',
        'counting':'repeated primitive examples; not equal optimizer count, unique clips or guaranteed equal convergence'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',type=int,required=True)
    p.add_argument('--batch-per-gpu',type=int,default=128)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--resume-checkpoint',type=Path)
    p.add_argument('--repartition-resume',action='store_true')
    p.add_argument('--unreviewed',action='store_true',help='User-requested direct WASD delivery, without offline quality review')
    args=p.parse_args();base=args.base.resolve();task=base/'model_terrain_v3_20261011'
    code=task/('code_control_v8_live' if args.unreviewed else 'code_control_v8')
    output=task/('control_sample_v8_live' if args.unreviewed else 'control_sample_v8')
    prefix='control-sample-v8-live' if args.unreviewed else 'control-sample-v8'
    state_file=task/(prefix+'-state.json')
    plan_path=task/'control-sample-v8-plan.json';batch=args.batch_per_gpu*len(args.gpus)
    if not plan_path.exists():save(plan_path,make_plan(task,batch))
    plan=json.loads(plan_path.read_text())
    if plan['global_batch']!=batch:raise ValueError('Prepared batch differs')
    budget=TerrainSampleBudget(**{k:plan[k] for k in ('reference_batch','reference_updates','retained_examples','global_batch')})
    target=plan['anchor_update']+budget.remaining_updates
    if args.prepare_only:print(json.dumps({'plan':str(plan_path),'target_optimizer_step':target,**budget.report()}));return
    if not args.worker:
        if state_file.exists() or output.exists():raise ValueError('Existing task; inspect before recovery')
        require_free(args.gpus)
        soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,max(soft,65536)),hard))
        with (task/(prefix+'-supervisor.log')).open('x') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--base',str(base),
                '--gpus',*map(str,args.gpus),'--batch-per-gpu',str(args.batch_per_gpu),'--worker',
                *(['--repartition-resume'] if args.repartition_resume else []),
                *(['--unreviewed'] if args.unreviewed else []),
                *(['--resume-checkpoint',str(args.resume_checkpoint.resolve())] if args.resume_checkpoint else [])],
                cwd=code,start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({'supervisor_pid':process.pid,'output':str(output),'budget':budget.report()}),flush=True);return
    state={'status':'starting','pid':os.getpid(),'gpus':args.gpus,'global_batch':batch,'batch_per_gpu':args.batch_per_gpu,
        'budget':budget.report(),'target_optimizer_step':target,'evaluation_policy':'once after original primitive-example budget',
        'actual_cumulative_target_updates':target+plan['inherited_reference_updates'],'quality_approved':False}
    if args.unreviewed:state['evaluation_policy']='none; user requested direct WASD without quality evaluation'
    save(state_file,state)
    try:
        state['gpu_snapshot']=require_free(args.gpus)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,args.gpus)),PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        dataset=base/'long_route_mixed_10k_ab_v1/mixed_dataset';source=base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'
        command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node='+str(len(args.gpus)),
            '-m','scripts.train_joint_terrain_v3','--source',str(source),'--dataset',str(dataset),'--pairs',str(task/'weak_pairs_v2'),
            '--output',str(output),'--steps',str(target),'--global-batch',str(batch),'--micro-batch',str(args.batch_per_gpu),
            '--workers','1','--cache-mib','2048','--adapter-warmup','0','--lr-warmup','1000','--feedback-start','1000',
            '--feedback-ramp','4000','--feedback-depth','13','--sequence-frames','257','--yaw-augmentation',
            '--reuse-loss-calibration','--optimized-geometry','--fused-adam','--async-checkpoints','--final-evaluation-only',
            '--checkpoint-every','50','--validate-every',str(target),'--evaluation-seconds','10','--sample-budget',str(plan_path),
            *(['--repartition-resume'] if args.repartition_resume else []),
            *(['--skip-quality-evaluation','--amend-execution'] if args.unreviewed else []),
            *(['--resume',str(args.resume_checkpoint.resolve())] if args.resume_checkpoint else ['--migrate-execution',plan['checkpoint']])]
        with (task/(prefix+'-training.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='training',training_pid=process.pid,command=command);save(state_file,state)
            while process.poll() is None:time.sleep(20)
            if process.returncode:raise RuntimeError(f'Training exited {process.returncode}; inspect retained log')
        complete=json.loads((output/'complete.json').read_text())
        if complete['steps']!=target or complete['primitive_examples']<budget.target_examples:raise ValueError('Original sample budget not completed')
        if args.unreviewed:
            state.update(status='exporting_unreviewed_trial',completion=complete);save(state_file,state)
            cloud='mb-transfer:runs/terrain-control-50k-20261011/v8-final-trial'
            subprocess.run([sys.executable,'-m','scripts.export_joint_terrain_trial','--base',str(base),
                '--checkpoint',str(output/'last.pt'),'--output',str(output),'--cloud',cloud],cwd=code,env=env,check=True)
            receipt=json.loads((output/'export.json').read_text())
            state.update(status='ready_for_unreviewed_trial',private_cloud=cloud,export=receipt);save(state_file,state)
            return
        review=output/f'review-{target:06d}';report=json.loads((review/'metrics.json').read_text())
        checks={key:bool(report.get(key)) for key in ('traversal_gate_passed','directional_gate_passed','flat_gate_passed')}
        state.update(status='exporting_final_review',completion=complete,final_checks=checks);save(state_file,state)
        visual=output/'visualizations/final'
        subprocess.run([sys.executable,'-m','scripts.export_joint_terrain_visuals','--stats',str(dataset/'stats'),
            '--reviews','baseline='+str(task/'control_50k_v6/review-000000'),'sample_budget='+str(review),
            '--output',str(visual)],cwd=code,env=env,check=True)
        weights=output/f'weights-{target:06d}.pt'
        receipt={'schema':'motionbricks.joint_terrain_export.v1','trained_updates':complete['actual_cumulative_optimizer_updates'],
            'primitive_examples':complete['primitive_examples'],'reference_batch':32,
            'reference_batch_equivalent_updates':complete['reference_batch_equivalent_updates'],
            'weights_file':weights.name,'weights_sha256':sha256(weights),'source_file':'source.pt','source_sha256':sha256(source),
            'control_version':report['control_version'],'model_checks_passed':all(checks.values()),'local_live_validation_pending':True,
            'quality_approved':False,'postprocessing':None}
        save(output/'export.json',receipt)
        proxy=dict(os.environ,HTTP_PROXY='http://127.0.0.1:27890',HTTPS_PROXY='http://127.0.0.1:27890')
        prefix=['rclone','--config',str(base/'drive-auth/rclone.conf')];cloud='mb-transfer:runs/terrain-control-50k-20261011/v8-sample-budget'
        subprocess.run(prefix+['copy',str(output),cloud,'--include','/'+weights.name,'--include','/export.json',
            '--include','/contract.json','--include','/complete.json','--include',f'/review-{target:06d}/metrics.json',
            '--include','/visualizations/**','--immutable'],env=proxy,check=True)
        subprocess.run(prefix+['copyto',str(source),cloud+'/source.pt','--immutable'],env=proxy,check=True)
        state.update(status='ready_for_local_trial' if all(checks.values()) else 'final_needs_review',private_cloud=cloud,export=receipt)
        save(state_file,state)
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error));save(state_file,state);raise


if __name__=='__main__':main()
