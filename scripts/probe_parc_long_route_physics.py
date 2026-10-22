"""Track raw long-route candidates diagnostically, without claiming admission."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):value.update(data)
    return value.hexdigest()


def make_raw_bundle(assets,generation,output,completed_only=False,min_seconds=20.,require_full_route=False,require_stops=False):
    from parc_generation_profile import resolve_sampling_profile
    report=json.loads((generation/'generation.json').read_text())
    if report['status']!='kinematic_candidates_generated_physics_pending':
        raise ValueError('Generation must be complete')
    sampling_profile,profile_evidence=resolve_sampling_profile(report)
    output.mkdir(parents=True,exist_ok=False)
    for name in ('PARC','tracker'):
        shutil.copytree(assets/name,output/name,copy_function=shutil.copyfile,
                       ignore=shutil.ignore_patterns('__pycache__','user_config.yaml'))
    (output/'motions').mkdir()
    motions={}
    for row in report['candidates']:
        if completed_only and (not row['endpoint_reached'] or row['duration_s'] < min_seconds):
            continue
        if require_full_route and not row.get('geometric_route_covered',False):
            continue
        if require_stops and not row.get('stop_response',{}).get('kinematic_stops_pass',False):
            continue
        source=Path(row['motion'])
        if digest(source)!=row['sha256'] or digest(row['intent'])!=row['intent_sha256']:
            raise ValueError('Raw motion or input intent changed')
        name=f'long_raw_c{row["candidate_id"]:03d}'
        relative=f'motions/{name}.pkl'
        intent=f'motions/{name}_intent.npz'
        shutil.copyfile(source,output/relative)
        shutil.copyfile(row['intent'],output/intent)
        motions[relative]={'candidate_key':name,'sha256':row['sha256'],'source_frames':row['frames'],
            'reference_stage':'raw_diagnostic_no_kinematic_optimization',
            'original_candidate_sha256':row['sha256'],'source_group':'long_route_'+str(report['seed']),
            'intent_path':intent,'intent_sha256':row['intent_sha256']}
    if not motions:
        raise ValueError('No completed long candidates available for physical tracking')
    files=[{'path':p.relative_to(output).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)}
           for p in sorted(output.rglob('*')) if p.is_file()]
    manifest={'schema_version':1,'motions':motions,'representatives':[],
        'diagnostic_only':True,'training_approved':False,'kinematic_optimization_applied':False,
        'generation_report_path':str(generation/'generation.json'),
        'generation_report_sha256':digest(generation/'generation.json'),
        'generator_checkpoint_sha256':report['checkpoint_sha256'],
        'sampling_profile':sampling_profile,'sampling_profile_evidence':profile_evidence,
        'files':files,'source_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
    (output/'_bundle.json').write_text(json.dumps(manifest,indent=2))


def wait_for_idle_gpu(gpu_index, report_path=None, attempts=20, delay_seconds=3):
    if attempts < 1:
        raise ValueError('At least one readiness observation is required')
    observations=[]
    for attempt in range(attempts):
        line=subprocess.check_output(['nvidia-smi','-i',str(gpu_index),
            '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        gpu,memory,utilization=[value.strip() for value in line.split(',')]
        ready=int(memory)<=1024 and int(utilization)<=5
        observations.append(dict(attempt=attempt+1,gpu=gpu,memory_mib=int(memory),
            utilization_percent=int(utilization),ready=ready))
        if report_path is not None:
            report_path.write_text(json.dumps(dict(observations=observations,ready=ready,
                limits=dict(memory_mib=1024,utilization_percent=5)),indent=2))
        if ready:
            return gpu
        if attempt+1<attempts:
            time.sleep(delay_seconds)
    raise RuntimeError('Probe GPU is occupied after bounded readiness wait')


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--generation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--completed-only',action='store_true')
    p.add_argument('--min-seconds',type=float,default=20.)
    p.add_argument('--require-full-route',action='store_true')
    p.add_argument('--require-stops',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.runtime/f'gpu{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Generation has just exited; utilization telemetry can lag allocation release.
        # Retain the same occupancy limits and wait, never take an occupied card.
        gpu=wait_for_idle_gpu(args.gpu,args.output/'gpu-readiness.json')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            PATH=str(args.runtime/'env/bin')+':'+os.environ.get('PATH',''),
            LD_LIBRARY_PATH=str(args.runtime/'env/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
            PYTHONPATH=str(args.runtime/'sdk/isaacgym/python'),WANDB_MODE='disabled',
            TORCH_EXTENSIONS_DIR=str(args.runtime/'torch_extensions'),MAX_JOBS='4')
        state={'status':'preparing_raw_diagnostic','training_approved':False,'started_at':time.time()}
        status=args.output/'status.json'
        status.write_text(json.dumps(state,indent=2))
        try:
            if args.min_seconds<0:raise ValueError('Minimum duration must be nonnegative')
            make_raw_bundle(args.runtime/'inputs_v1',args.generation,args.output/'inputs',args.completed_only,args.min_seconds,args.require_full_route,args.require_stops)
            with (args.output/'physics.log').open('w') as log:
                subprocess.run([str(args.runtime/'env/bin/python'),'-u',str(args.runtime/'run_parc_physics_record.py'),
                    '--bundle',str(args.output/'inputs'),'--output',str(args.output/'physics'),'--selection','all'],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            report=json.loads((args.output/'physics/report.json').read_text())
            state.update(status=report['status'],saved=report.get('saved',[]),elapsed_seconds=time.time()-state['started_at'])
        except Exception as error:
            state.update(status='failed',error=type(error).__name__+': '+str(error))
            raise
        finally:status.write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
