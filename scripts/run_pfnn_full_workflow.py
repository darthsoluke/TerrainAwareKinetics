"""Durable gated PFNN preparation -> reviewed full native MotionBricks training."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main(a):
    code = Path(__file__).resolve().parents[1]
    root = a.base/'pfnn_mixed_20261015'
    stop=json.loads((root/'full_training_v1/data_repair_stop.json').read_text())
    if stop['actual_optimizer_updates']!=0 or stop['status']!='stopped_before_training_for_data_repair':
        raise ValueError('Corrected workflow must supersede the old dispatcher before any optimizer updates')
    if not a.worker:
        receipt = a.output.with_suffix('.process.json')
        if a.output.exists() or receipt.exists():
            raise ValueError('Workflow already exists; inspect state instead of launching another')
        command = [sys.executable,'-u',str(Path(__file__).resolve()),'--base',str(a.base),
            '--output',str(a.output),'--gpu',str(a.gpu),'--steps',str(a.steps),'--worker']
        with a.output.with_suffix('.log').open('x') as stream:
            child = subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONPATH='.:out/vendor/kimodo',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        value = dict(pid=child.pid,command=command,code=str(code),gpu=a.gpu,
            budget_updates_per_stage=a.steps,batch=256,micro_batch=64,
            training_stages=['root','codec','pose'],full_training_started=False,
            supersedes_zero_update_workflow='full_training_v1',pfnn_world_contract='pfnn-world-and-scene-contacts-v2')
        save(receipt,value)
        print(json.dumps(value),flush=True)
        return
    a.output.mkdir(parents=True,exist_ok=False)
    state = dict(pid=os.getpid(),stage='preparation',quality_approved=False,full_training_started=False)
    state_path = a.output/'state.json'
    save(state_path,state)

    def wait_external(folder, receipt):
        while not (folder/'complete.json').exists():
            if (folder/'failure.json').exists():
                raise RuntimeError('Preparation failed: '+str(folder/'failure.json'))
            process = json.loads(receipt.read_text())
            cmdline = Path('/proc')/str(process['pid'])/'cmdline'
            if not cmdline.exists():
                raise RuntimeError('Preparation process ended without complete receipt: '+str(folder))
            actual = cmdline.read_bytes().replace(b'\0',b' ').decode()
            if str(folder) not in actual or process['command'][0] not in actual:
                raise RuntimeError('Preparation PID identity differs: '+str(folder))
            state.update(waiting_for=str(folder),status='waiting_for_existing_preparation')
            save(state_path,state)
            time.sleep(30)

    def wait_idle():
        while True:
            occupancy = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True)
            row = next(line for line in occupancy.splitlines() if int(line.split(',')[0]) == a.gpu)
            _, memory, busy = map(int,row.split(','))
            if memory <= 500 and busy <= 5:
                state['gpu_snapshot_before'] = occupancy
                return
            state.update(status='waiting_for_idle_gpu',gpu=a.gpu)
            save(state_path,state)
            time.sleep(30)

    def run(label, command, gpu=True):
        if gpu:
            wait_idle()
        env = dict(os.environ,PYTHONPATH='.:out/vendor/kimodo',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            CUDA_VISIBLE_DEVICES=str(a.gpu) if gpu else '',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (a.output/(label+'.log')).open('x') as stream:
            child = subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            state.update(status='running',job=label,child_pid=child.pid,command=command)
            save(state_path,state)
            if child.wait() != 0:
                raise RuntimeError(label+' failed; inspect '+str(a.output/(label+'.log')))

    try:
        wait_external(root/'mechanics_world_v1',root/'mechanics_world_v1.process.json')
        if not json.loads((root/'mechanics_world_v1/complete.json').read_text())['passed']:
            raise ValueError('Actual optimizer and resume preflight did not pass')
        wait_external(root/'dataset_world_v2',root/'dataset_world_v2.process.json')
        state['stage'] = 'cache_pfnn'
        cache = root/'cache_pfnn_world_v2'
        if cache.exists():
            raise ValueError('PFNN cache already exists; review identity before reusing it')
        bundle = a.base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'
        run('cache_pfnn',[sys.executable,'-u','-m','scripts.cache_full_terrain_primitives',
            '--bundle',str(bundle),'--pfnn',str(root/'dataset_world_v2'),'--stats',
            str(a.base/'long_route_mixed_10k_ab_v1/mixed_dataset/stats'),'--output',str(cache)])
        wait_external(root/'cache_mixed_v1',root/'cache_mixed_v1.process.json')
        state['stage'] = 'corpus_audit'
        caches = [root/'cache_mixed_v1',cache]
        run('corpus_audit',[sys.executable,'-u','-m','scripts.audit_full_terrain_corpus',
            '--cache',*map(str,caches),'--bundle',str(bundle),'--output',str(a.output/'corpus_audit.json')],gpu=False)
        if not json.loads((a.output/'corpus_audit.json').read_text())['passed']:
            raise ValueError('Full source geometry/cache/split audit failed')
        parent = a.base/'model_terrain_v3_20261011/parc_rugged32_20261013_v2/unified_native_pose_fullbank_20261015_v2/weights.pt'
        for stage in ('root','codec','pose'):
            state['stage'] = stage
            target = a.output/stage
            command = [sys.executable,'-u','-m','scripts.train_unified_full_corpus','--stage',stage,
                '--bundle',str(bundle),'--parent',str(parent),'--preflight',str(root/'preflight_world_v1/complete.json'),
                '--cache',*map(str,caches),'--output',str(target),'--steps',str(a.steps),
                '--batch','256','--micro-batch','64','--workers','4',
                '--corpus-audit',str(a.output/'corpus_audit.json')]
            # Two budgeted real updates verify the actual full-cache memory and path.
            run(stage+'-startup',command+['--stop-after','2'])
            stopped = json.loads((target/'stopped.json').read_text())
            if stopped['actual_updates'] != 2 or stopped['actual_exposures'] != 512:
                raise ValueError('Full-cache startup update accounting differs')
            state['full_training_started'] = True
            run(stage+'-resume',command+['--resume',str(target/'latest.pt')])
            complete = json.loads((target/'complete.json').read_text())
            if complete['actual_updates'] != a.steps or not complete['protected_hash_passed']:
                raise ValueError('Full stage did not complete the declared contract')
            parent = target/'weights.pt'
        state.update(status='training_complete_quality_review_required',actual_updates=3*a.steps,
            actual_exposures=3*a.steps*256,checkpoint=str(parent),quality_approved=False)
        save(a.output/'complete.json',state)
    except Exception as error:
        state.update(status='failed',error=repr(error))
        save(a.output/'failure.json',state)
        raise
    finally:
        save(state_path,state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--steps',type=int,default=50000)
    parser.add_argument('--worker',action='store_true')
    main(parser.parse_args())
