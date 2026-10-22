"""Bounded continuous-route throughput experiment; not a training admission."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--sampling-profile',choices=('official_data','official_eval'),required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261010)
    parser.add_argument('--variant',type=int,choices=(0,1,2),default=0)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    work=args.runtime/'wide_v1'
    python=args.runtime/'env/bin/python'
    with (args.runtime/f'gpu{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        line=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
            '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        gpu,memory,utilization=[x.strip() for x in line.split(',')]
        if int(memory)>1024 or int(utilization)>5:
            raise RuntimeError('Probe card is occupied')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            PATH=str(args.runtime/'env/bin')+':'+os.environ.get('PATH',''),
            LD_LIBRARY_PATH=str(args.runtime/'env/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
            PYTHONPATH=str(args.runtime/'sdk/isaacgym/python'),WANDB_MODE='disabled')
        state={'status':'generating','gpu':args.gpu,'sampling_profile':args.sampling_profile,
               'batch_size':args.batch_size,'seed':args.seed,'variant':args.variant,
               'training_approved':False,'physics_verified':False,'motion_concatenation':False,
               'started_at':time.time()}
        status_path=args.output/'status.json'
        status_path.write_text(json.dumps(state,indent=2))
        try:
            with (args.output/'generation.log').open('w') as log:
                subprocess.run([str(python),'-u',str(work/'scripts/generate_parc_long_stairs.py'),
                    '--repo',str(work/'out/vendor/PARC'),
                    '--checkpoint',str(work/'out/parc_legacy_models_20261008/demo_iter3_verified.ckpt'),
                    '--output',str(args.output/'generation'),'--kind','long_mixed','--course-sections','3',
                    '--batch-size',str(args.batch_size),'--max-seconds','90','--seed',str(args.seed),
                    '--variant',str(args.variant),
                    '--sampling-profile',args.sampling_profile,'--score-frame-chunk','4'],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            report=json.loads((args.output/'generation/generation.json').read_text())
            state.update(status=report['status'],elapsed_seconds=report['elapsed_seconds'],
                path_length_m=report['course_planar_m'],candidates=[{
                    key:row[key] for key in ('candidate_id','duration_s','endpoint_reached',
                        'geometric_route_covered','weighted_contact_score','weighted_penetration_score')}
                    for row in report['candidates']])
        except Exception as error:
            state.update(status='failed',error=type(error).__name__+': '+str(error))
            raise
        finally:
            status_path.write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
