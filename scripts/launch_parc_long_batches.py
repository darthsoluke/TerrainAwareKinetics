"""Launch independent continuous-course batches on explicitly assigned cards."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='2,3,4,5,6')
    p.add_argument('--append',action='store_true')
    p.add_argument('--seed-start',type=int,default=20261012)
    p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args()
    cards=[int(value) for value in args.gpus.split(',')]
    if not cards or len(set(cards))!=len(cards) or any(card not in (1,2,3,4,5,6) for card in cards):
        raise ValueError('This run reserves physical GPU0/7 for other users')
    args.output.mkdir(parents=True,exist_ok=args.append)
    code=Path(__file__).resolve().parents[1];runtime=args.base/'parc_runtime_v1'
    environment=dict(os.environ,PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    prior=json.loads((args.output/'launch.json').read_text()) if args.append else {}
    jobs=prior.get('jobs',[]);started_at=prior.get('started_at',time.time())
    if set(cards)&{job['gpu'] for job in jobs}:
        raise ValueError('A GPU already belongs to this benchmark cohort')
    for index,card in enumerate(cards):
        seed=args.seed_start+index
        if seed in {job['seed'] for job in jobs}:raise ValueError('Duplicate generation seed')
        directory=args.output/f'seed_{seed}';directory.mkdir()
        generation=directory/'generation_run';finished=directory/'finished'
        commands=[('generation',[str(runtime/'env/bin/python'),'-u',str(runtime/'wide_v1/scripts/run_parc_long_route_probe.py'),
            '--runtime',str(runtime),'--output',str(generation),'--gpu',str(card),
            '--sampling-profile','official_eval','--batch-size',str(args.batch_size),'--seed',str(seed),'--variant',str(index%3)]),
            ('finish',[sys.executable,'-u','-m','scripts.finish_parc_long_route_batch',
             '--base',str(args.base),'--generation-run',str(generation),'--output',str(finished),
             '--gpu',str(card),'--split','train'])]
        job={'gpu':card,'seed':seed,'variant':index%3,'split':'train','batch_size':args.batch_size,
             'generation_run':str(generation),'finished':str(finished),'training_approved':False}
        for name,command in commands:
            with (directory/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=code,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            job[name+'_pid']=child.pid
        jobs.append(job)
        (args.output/'launch.json').write_text(json.dumps({'started_at':started_at,'jobs':jobs,'reserved_gpus':[0,7]},indent=2))
    print(json.dumps({'jobs':jobs,'reserved_gpus':[0,7]},indent=2))


if __name__=='__main__':main()
