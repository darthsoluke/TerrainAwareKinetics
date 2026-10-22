"""User-authorized all-free-GPU continuation with a single final quality review."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from scripts.train_native_mb import sha256


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def require_free(indices):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    rows={int(parts[0]):parts for line in output.splitlines() if (parts:=line.split(', '))}
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines()
    if not indices or len(set(indices))!=len(indices):raise ValueError('Distinct available physical GPU IDs required')
    chosen=[rows[i] for i in indices]
    if any(float(r[2])>512 or float(r[3])>10 or r[1] in active for r in chosen):
        raise RuntimeError('A requested card is occupied; never terminate other users or share their GPU')
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',type=int,required=True)
    p.add_argument('--batch-per-gpu',type=int,required=True)
    p.add_argument('--resume-checkpoint',type=Path,help='Continue the verified same-contract seven-rank checkpoint')
    p.add_argument('--worker',action='store_true')
    args=p.parse_args();base=args.base.resolve();task=base/'model_terrain_v3_20261011'
    code=task/'code_control_v7';output=task/'control_50k_v7';state_file=task/'control-50k-v7-state.json'
    if not args.worker:
        if state_file.exists() or output.exists():raise ValueError('Run already exists; inspect actual state before recovery')
        snapshot=require_free(args.gpus)
        soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,max(soft,65536)),hard))
        with (task/'control-50k-v7-supervisor.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--base',str(base),
                '--gpus',*map(str,args.gpus),'--batch-per-gpu',str(args.batch_per_gpu),'--worker',
                *(['--resume-checkpoint',str(args.resume_checkpoint.resolve())] if args.resume_checkpoint else [])],
                cwd=code,start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({'supervisor_pid':process.pid,'gpu_snapshot':snapshot,'output':str(output)}),flush=True);return
    state={'status':'starting','pid':os.getpid(),'gpus':args.gpus,'warmstart_updates':5000,
        'run_target_updates':45000,'cumulative_target_updates':50000,'evaluation_policy':'only at cumulative 50000',
        'batch_per_gpu':args.batch_per_gpu,'global_batch':args.batch_per_gpu*len(args.gpus),'quality_approved':False}
    save(state_file,state)
    try:
        state['gpu_snapshot']=require_free(args.gpus)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,args.gpus)),PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        dataset=base/'long_route_mixed_10k_ab_v1/mixed_dataset';source=base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'
        migration=json.loads((task/'control-v7-migration-input.json').read_text())
        command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node='+str(len(args.gpus)),
            '-m','scripts.train_joint_terrain_v3','--source',str(source),'--dataset',str(dataset),'--pairs',str(task/'weak_pairs_v2'),
            '--output',str(output),'--steps','45000','--global-batch',str(state['global_batch']),
            '--micro-batch',str(args.batch_per_gpu),'--workers','1','--cache-mib','2048',
            '--adapter-warmup','0','--lr-warmup','1000','--feedback-start','1000','--feedback-ramp','4000',
            '--feedback-depth','13','--sequence-frames','257','--yaw-augmentation','--reuse-loss-calibration',
            '--optimized-geometry','--fused-adam','--async-checkpoints','--final-evaluation-only',
            '--checkpoint-every','1000','--validate-every','45000','--evaluation-seconds','10',
            *(['--resume',str(args.resume_checkpoint.resolve())] if args.resume_checkpoint else ['--migrate-execution',migration['input']])]
        with (task/'control-50k-v7-training.log').open('x') as log:
            process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='training',training_pid=process.pid,command=command,migration=migration);save(state_file,state)
            while process.poll() is None:time.sleep(30)
            if process.returncode:raise RuntimeError('Training exited '+str(process.returncode)+'; inspect retained log before recovery')
        complete=json.loads((output/'complete.json').read_text())
        if complete['steps']!=45000:raise ValueError('The cumulative 50000-update objective was not reached')
        review=output/'review-045000';report=json.loads((review/'metrics.json').read_text())
        checks={key:bool(report.get(key)) for key in ('traversal_gate_passed','directional_gate_passed','flat_gate_passed')}
        state.update(status='exporting_final_review',completed_updates=50000,final_checks=checks);save(state_file,state)
        visual=output/'visualizations/final'
        subprocess.run([sys.executable,'-m','scripts.export_joint_terrain_visuals','--stats',str(dataset/'stats'),
            '--reviews','baseline_5k='+str(task/'control_50k_v6/review-000000'),'learned_50k='+str(review),
            '--output',str(visual)],cwd=code,env=env,check=True)
        weights=output/'weights-045000.pt'
        receipt={'schema':'motionbricks.joint_terrain_export.v1','trained_updates':50000,'v7_start_cumulative_updates':migration['cumulative_step'],
            'weights_file':weights.name,'weights_sha256':sha256(weights),'source_file':'source.pt','source_sha256':sha256(source),
            'control_version':report['control_version'],'model_checks_passed':all(checks.values()),'local_live_validation_pending':True,
            'quality_approved':False,'postprocessing':None}
        save(output/'export.json',receipt)
        proxy=dict(os.environ,HTTP_PROXY='http://127.0.0.1:27890',HTTPS_PROXY='http://127.0.0.1:27890')
        prefix=['rclone','--config',str(base/'drive-auth/rclone.conf')];cloud='mb-transfer:runs/terrain-control-50k-20261011/v7'
        subprocess.run(prefix+['copy',str(output),cloud,'--include','/'+weights.name,'--include','/export.json',
            '--include','/contract.json','--include','/review-045000/metrics.json','--include','/visualizations/**','--immutable'],env=proxy,check=True)
        subprocess.run(prefix+['copyto',str(source),cloud+'/source.pt','--immutable'],env=proxy,check=True)
        state.update(status='ready_for_local_trial' if all(checks.values()) else 'final_needs_review',private_cloud=cloud,export=receipt)
        save(state_file,state)
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error));save(state_file,state);raise


if __name__=='__main__':main()
