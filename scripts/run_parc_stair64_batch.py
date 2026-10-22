"""Generate exactly eight groups of eight, without touching occupied GPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def write(path,value):
    pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value,indent=2));pending.replace(path)


def main(args):
    cards=list(map(int,args.gpus.split(',')))
    if len(set(cards))!=len(cards) or 1 in cards or not cards:
        raise ValueError('Distinct explicitly assigned cards required; GPU1 belongs to another user')
    args.output.mkdir(parents=True,exist_ok=False)
    runtime=args.base/'parc_runtime_v1';code=Path(__file__).resolve().parent
    write(args.output/'plan.json',dict(total_candidates=64,layouts=8,candidates_per_layout=8,
        phases=['approach','ascent','mid_ascent_stop','resume','platform','descent','mid_descent_stop','resume','exit'],
        history_reset_between_phases=False,training_started=False,approved_orange_unchanged=True,
        approved_checkpoint_sha256='ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf',
        gpus=cards,held_out_layouts={'validation':[6],'test':[7]},stop_requests_are_not_stop_success=True))
    def job(index,card):
        output=args.output/f'layout_{index:02d}';output.mkdir()
        state=dict(layout=index,gpu=card,status='waiting_for_idle_gpu',candidates_requested=8,training_approved=False)
        write(output/'status.json',state)
        with (runtime/f'gpu{card}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for _ in range(180):
                status=subprocess.check_output(['nvidia-smi','-i',str(card),
                    '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                gpu,memory,util=[v.strip() for v in status.split(',')]
                if int(memory)<=256 and int(util)<=5:break
                time.sleep(10)
            else:raise RuntimeError('GPU remained occupied; no process was stopped')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                PATH=str(runtime/'env/bin')+':'+os.environ.get('PATH',''),
                LD_LIBRARY_PATH=str(runtime/'env/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
                PYTHONPATH=str(runtime/'sdk/isaacgym/python'),WANDB_MODE='disabled')
            command=[str(runtime/'env/bin/python'),'-u',str(code/'generate_parc_stair64.py'),
                '--repo',str(runtime/'wide_v1/out/vendor/PARC'),
                '--checkpoint',str(runtime/'wide_v1/out/parc_legacy_models_20261008/demo_iter3_verified.ckpt'),
                '--output',str(output/'generation'),'--layout',str(index),'--seed',str(2026101300+index)]
            try:
                with (output/'generation.log').open('x') as log:
                    child=subprocess.Popen(command,env=env,cwd=runtime/'wide_v1',stdout=log,stderr=subprocess.STDOUT)
                    state.update(status='generating',pid=child.pid,command=command,occupancy_before_launch=status)
                    write(output/'status.json',state)
                    code_value=child.wait(timeout=1800)
                if code_value!=0:raise RuntimeError('Generation exited '+str(code_value))
                report=json.loads((output/'generation/generation.json').read_text())
                if report['prefix_contract']['checked_history_calls']==0:raise ValueError('No checked continuation')
                state.update(status='generated_prescreen_complete',candidates=len(report['candidates']),
                    complete_routes=report['geometric_coverage_candidates'],
                    kinematic_stop_pass=report['kinematic_stop_pass_candidates'],
                    prefix_error=report['prefix_contract']['max_prefix_error'])
            except Exception as error:
                state.update(status='failed',error=str(error));write(output/'status.json',state);raise
            write(output/'status.json',state)
            return state
    results=[job(0,cards[0])]
    write(args.output/'progress.json',dict(results=results,status='first_layout_verified'))
    def worker(slot):
        return [job(i,cards[slot]) for i in range(1+slot,8,len(cards))]
    with ThreadPoolExecutor(max_workers=len(cards)) as pool:
        futures=[pool.submit(worker,i) for i in range(len(cards))]
        for future in futures:
            results.extend(future.result())
            write(args.output/'progress.json',dict(results=results,status='generating_remaining_layouts'))
    total=sum(r['candidates'] for r in results)
    if total!=64:raise ValueError('Candidate inventory does not total64')
    write(args.output/'complete.json',dict(status='generation_and_native_prescreen_complete',results=results,
        total_candidates=total,kinematic_stop_pass=sum(r['kinematic_stop_pass'] for r in results),
        physical_tracking_and_soma_screen_pending=True,training_started=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',default='2,3,4,0')
    main(parser.parse_args())
