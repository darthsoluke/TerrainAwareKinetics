"""Bounded corrected-control pilot, exact 50k continuation, and private export."""

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import torch

from scripts.launch_joint_terrain_training import available_cards
from scripts.train_native_mb import sha256
from motionbricks.data.terrain_traversal_metrics import traversal_pilot_checks


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def pilot_checks(initial,report):
    return traversal_pilot_checks(initial,report)


def phase_command(command,arguments):
    result=list(command)
    if '--resume' in arguments and '--initialize' in result:
        index=result.index('--initialize')
        del result[index:index+2]
    return result+arguments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--revision',default='v6')
    p.add_argument('--initialize',type=Path,help='Warmstart updates count toward the cumulative 50k target')
    p.add_argument('--gpus',nargs=2,type=int,default=[2,3])
    args=p.parse_args();base=args.base.resolve();task=base/'model_terrain_v3_20261011'
    if not re.fullmatch('v[0-9]+',args.revision):p.error('Revision must have the form v1, v2, ...')
    code=task/('code_control_'+args.revision);output=task/('control_50k_'+args.revision)
    state_file=task/f'control-50k-{args.revision}-state.json'
    if not args.worker:
        if output.exists() or state_file.exists():raise ValueError('This campaign already exists; inspect it before recovery')
        available_cards(args.gpus)
        with (task/f'control-50k-{args.revision}-supervisor.log').open('x') as log:
            worker=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--base',str(base),
                '--gpus',*map(str,args.gpus),'--revision',args.revision,'--worker',
                *(['--initialize',str(args.initialize.resolve())] if args.initialize else [])],cwd=code,start_new_session=True,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({'pid':worker.pid,'output':str(output),'pilot_updates':5000,'total_budget':50000}),flush=True)
        return
    state={'status':'checking_free_cards','pid':os.getpid(),'total_budget':50000,'quality_approved':False}
    save(state_file,state)
    try:
        state['gpu_snapshot']=available_cards(args.gpus)
        warmstart=0
        if args.initialize:
            initial_weights=torch.load(args.initialize,map_location='cpu',mmap=True,weights_only=False)
            warmstart=int(initial_weights['step']);del initial_weights
            if not 0<=warmstart<50000:raise ValueError('Invalid cumulative warmstart count')
        budget=50000-warmstart;pilot_updates=min(5000,budget)
        state.update(warmstart_updates=warmstart,run_budget=budget,initialization=str(args.initialize) if args.initialize else None)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,args.gpus)),PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'))
        source=base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'
        dataset=base/'long_route_mixed_10k_ab_v1/mixed_dataset'
        command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            '-m','scripts.train_joint_terrain_v3','--source',str(source),'--dataset',str(dataset),
            '--pairs',str(task/'weak_pairs_v2'),'--output',str(output),'--steps',str(budget),
            '--global-batch','32','--micro-batch','16','--workers','2','--weak-fraction','0',
            '--adapter-warmup','0' if args.initialize else '1000','--lr-warmup','1000','--feedback-start','1000','--feedback-ramp','4000',
            '--feedback-depth','13','--sequence-frames','257','--yaw-augmentation',
            '--checkpoint-every','250','--validate-every','1000','--evaluation-seconds','10',
            *(['--initialize',str(args.initialize.resolve()),'--reuse-loss-calibration'] if args.initialize else [])]
        def train_phase(label,arguments):
            phase=phase_command(command,arguments)
            with (task/f'control-50k-{args.revision}-{label}.log').open('x') as log:
                process=subprocess.Popen(phase,cwd=code,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT)
                state.update(status=label,training_pid=process.pid,command=phase);save(state_file,state)
                while process.poll() is None:time.sleep(30)
                if process.returncode:raise RuntimeError(f'{label} exited {process.returncode}; see its retained log')
        train_phase('pilot',['--evaluate-initial','--stop-at',str(pilot_updates)])
        initial=json.loads((output/'review-000000/metrics.json').read_text())
        pilot=json.loads((output/f'review-{pilot_updates:06d}/metrics.json').read_text())
        checks=pilot_checks(initial,pilot);save(output/'pilot_decision.json',{'checks':checks,'eligible_for_continuation':all(checks.values())})
        state['pilot_checks']=checks
        if not all(checks.values()):
            state.update(status='pilot_needs_revision',completed_updates=warmstart+pilot_updates,run_completed_updates=pilot_updates);save(state_file,state)
            return
        train_phase('continuation',['--resume',str(output/'last.pt')])
        completed=json.loads((output/'complete.json').read_text())
        if completed['steps']!=budget:raise ValueError('Training did not reach cumulative 50k updates')
        final=json.loads((output/f'review-{budget:06d}/metrics.json').read_text())
        checks=pilot_checks(initial,final)
        checks.update(real_terrain_traversal=bool(final.get('traversal_gate_passed')),
            causal_directional_control=bool(final.get('directional_gate_passed')),
            flat_start_stop_and_rest=bool(final.get('flat_gate_passed')))
        state.update(completed_updates=50000,run_completed_updates=budget,final_checks=checks)
        if not all(checks.values()):
            state.update(status='final_needs_review');save(state_file,state);return
        state['status']='exporting';save(state_file,state)
        visual=output/'visualizations/final'
        subprocess.run([sys.executable,'-m','scripts.export_joint_terrain_visuals','--stats',str(dataset/'stats'),
            '--reviews','initial='+str(output/'review-000000'),'learned='+str(output/f'review-{budget:06d}'),
            '--output',str(visual)],cwd=code,env=env,check=True)
        weights=output/f'weights-{budget:06d}.pt'
        receipt={'schema':'motionbricks.joint_terrain_export.v1','trained_updates':50000,
            'this_run_updates':budget,'warmstart_updates':warmstart,
            'weights_file':weights.name,'weights_sha256':sha256(weights),'source_file':'source.pt',
            'source_sha256':sha256(source),'control_version':final['control_version'],
            'model_checks_passed':True,'local_live_validation_pending':True,'quality_approved':False,'postprocessing':None}
        save(output/'export.json',receipt)
        proxy=dict(os.environ,HTTP_PROXY='http://127.0.0.1:27890',HTTPS_PROXY='http://127.0.0.1:27890')
        prefix=['rclone','--config',str(base/'drive-auth/rclone.conf')]
        cloud='mb-transfer:runs/terrain-control-50k-20261011/'+args.revision
        subprocess.run(prefix+['copy',str(output),cloud,'--include','/'+weights.name,
            '--include','/export.json','--include','/contract.json','--include','/review-*/metrics.json',
            '--include','/visualizations/**','--immutable'],env=proxy,check=True)
        subprocess.run(prefix+['copyto',str(source),cloud+'/source.pt','--immutable'],env=proxy,check=True)
        state.update(status='ready_for_local_trial',private_cloud=cloud,export=receipt);save(state_file,state)
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error));save(state_file,state);raise


if __name__=='__main__':main()
