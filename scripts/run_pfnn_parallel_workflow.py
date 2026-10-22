"""Preload verified parallel Root execution, then continue the same stage budgets."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.run_pfnn_bias_workflow import read, save, digest, process_alive


def gpu_snapshot():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    occupied = {x.split(',')[1].strip() for x in apps.splitlines()}
    rows = {int(x.split(',')[0]): [v.strip() for v in x.split(',')[1:]] for x in text.splitlines()}
    available = [g for g,(uuid,memory,busy) in rows.items()
        if int(memory)<=500 and int(busy)<=5 and uuid not in occupied]
    return text, available


def main(a):
    run = a.base/'pfnn_mixed_20261015'
    code = Path(__file__).resolve().parents[1]
    four = a.root_parallel_devices == 4
    if len(a.gpus)!=a.root_parallel_devices or len(set(a.gpus))!=len(a.gpus):
        raise ValueError('Execution topology requires distinct selected GPUs')
    old = run/('full_training_v6/root' if four else 'full_training_v5/root')
    old_supervision=run/('full_training_v7' if four else 'full_training_v5')
    proofs = [dict(role=role,path=str(run/folder/'complete.json')) for role,folder in
        ([('parallel_optimizer_resume','parallel_four_optimizer_v1')] if four else
         [('memory_equivalence','execution_memory_compare_v1'),('parallel_optimizer_resume','parallel_optimizer_v1')])]
    for proof in proofs:
        if not read(Path(proof['path']))['passed']:
            raise ValueError('Required execution proof failed')
        proof['sha256'] = digest(Path(proof['path']))
    parallel_hash = digest(code/'motionbricks/data/parallel_root_training.py')
    parallel_proof=read(Path(proofs[-1]['path']))
    if parallel_hash != parallel_proof['implementation_sha256']:
        raise ValueError('Parallel source differs from tested implementation')
    future_proofs=[]
    if four:
        from types import SimpleNamespace
        from scripts.continue_pfnn_codec_pose_execution import validate
        future_proofs=validate(SimpleNamespace(base=a.base,previous=old.parent))
        if (parallel_proof['reference_devices']!=2 or parallel_proof['target_devices']!=4
                or parallel_proof['resume_from_reference_devices']!=2
                or parallel_proof['reference_implementation_sha256']!=read(old/'contract.json')[
                    'source_code_sha256']['motionbricks/data/parallel_root_training.py']):
            raise ValueError('Proof does not bind the current two-device implementation')
        if any((old_supervision/stage).exists() for stage in ('codec','pose')):
            raise ValueError('Previous workflow already started a dependent stage')
    if a.validate_only:
        print(json.dumps(dict(validated=True,proofs=proofs,parallel_implementation_sha256=parallel_hash)))
        return
    snapshot, available = gpu_snapshot()
    if any(g not in available for g in a.gpus):
        raise ValueError('Requested GPUs are no longer idle')
    if not process_alive(a.root_pid) or not process_alive(a.previous_supervisor):
        raise ValueError('Previous workflow identity must be live')
    root_command = (Path('/proc')/str(a.root_pid)/'cmdline').read_bytes().decode()
    supervisor_command = (Path('/proc')/str(a.previous_supervisor)/'cmdline').read_bytes().decode()
    supervisor_module='continue_pfnn_codec_pose_execution' if four else 'run_pfnn_bias_workflow'
    if (str(old) not in root_command or supervisor_module not in supervisor_command
            or str(old_supervision) not in supervisor_command
            or read(old_supervision/'state.json')['pid']!=a.previous_supervisor):
        raise ValueError('Previous process identity mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    handoff = old/'parallel_handoff.pt'
    migration_path = a.output/'migration.json'
    if handoff.exists() or (old/'latest_before_parallel_handoff.pt').exists():
        raise ValueError('Handoff already exists')
    state = dict(pid=os.getpid(),stage='root',status='preloading_parallel_root',gpus=a.gpus,
        gpu_snapshot_before=snapshot,previous_root_pid=a.root_pid,total_budget_updates=150000,quality_approved=False)
    save(a.output/'state.json',state)
    if future_proofs:save(a.output/'execution_proofs.json',dict(proofs=future_proofs,total_budget_updates=150000))

    def command(stage,parent):
        argv=[sys.executable,'-u','-m','scripts.train_unified_full_corpus','--stage',stage,
            '--bundle',str(a.base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'),
            '--parent',str(parent),'--preflight',str(run/'preflight_world_v1/complete.json'),
            '--cache',str(run/'cache_mixed_v1'),str(run/'cache_pfnn_world_v2'),
            '--output',str(a.output/stage),'--steps','50000','--batch','256','--micro-batch','64',
            '--workers','4','--corpus-audit',str(run/'full_training_v3/corpus_audit.json')]
        if four and stage in ('codec','pose'):
            argv+=['--coalesce-dilated-bias']
            argv+=['--cuda-memory-fraction','0.1'] if stage=='codec' else ['--cache-pose-tokens']
        else:
            argv += ['--cache-pose-tokens'] if stage=='pose' else ['--coalesce-dilated-bias']
        return argv

    def launch(label,argv,gpus):
        with (a.output/(label+'.log')).open('x') as stream:
            child=subprocess.Popen(argv,cwd=code,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),
                    TMPDIR='${PRIVATE_STORAGE}/mbtmp',PYTHONPATH='.:out/vendor/kimodo',
                    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        state.update(child_pid=child.pid,job=label,command=argv,job_gpus=gpus)
        save(a.output/'state.json',state)
        return child

    prototype=a.base/'model_terrain_v3_20261011/parc_rugged32_20261013_v2/unified_native_pose_fullbank_20261015_v2/weights.pt'
    argv=command('root',prototype)+['--resume',str(handoff),'--resume-migration',str(migration_path),
        '--cuda-memory-fraction','0.1','--root-parallel-devices',str(a.root_parallel_devices),'--await-resume']
    child=launch('root-resume',argv,a.gpus)
    while not (a.output/'root-await-resume.json').exists():
        if child.poll() is not None:
            raise RuntimeError('Parallel Root preload failed; previous training remains unchanged')
        time.sleep(5)
    if read(a.output/'root-await-resume.json')['pid']!=child.pid:
        raise ValueError('Unexpected preload acknowledgment')
    handoff_name='parallel_four_acceleration_handoff.json' if four else 'parallel_acceleration_handoff.json'
    save(old_supervision/handoff_name,dict(successor_pid=os.getpid(),
        successor=str(a.output),old_supervisor_pid=a.previous_supervisor,root_pid=a.root_pid,
        root_restarted=False,preloaded_child=child.pid,proofs=proofs))
    os.kill(a.previous_supervisor,signal.SIGTERM)
    owner_start=(Path('/proc')/str(a.root_pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    watchdog_code=(f'import os,signal,time,pathlib; time.sleep(55); p=pathlib.Path("/proc/{a.root_pid}/stat"); '
        f'os.kill({a.root_pid},signal.SIGCONT) if p.exists() and '
        f'p.read_text().rsplit(")",1)[1].split()[19]=={owner_start!r} else None')
    subprocess.Popen([sys.executable,'-c',watchdog_code],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,start_new_session=True)
    os.kill(a.root_pid,signal.SIGSTOP)
    try:
        for _ in range(150):
            status=next(x.split()[1] for x in (Path('/proc')/str(a.root_pid)/'status').read_text().splitlines()
                if x.startswith('State:'))
            if status=='T':break
            time.sleep(.02)
        if status!='T':raise RuntimeError('Could not arm checkpoint barrier atomically')
        progress=read(old/'progress.json')
        (old/'latest.pt').rename(old/'latest_before_parallel_handoff.pt')
        (old/'latest.pt').mkdir()
        save(old/'parallel_handoff_barrier.json',dict(owner_pid=a.root_pid,owner_starttime=owner_start,
            published_updates_before=progress['actual_updates'],successor_pid=os.getpid(),
            intentional_stop='Complete torch.save followed by IsADirectoryError before the next update'))
    finally:
        os.kill(a.root_pid,signal.SIGCONT)
    state['status']='waiting_for_complete_checkpoint'
    save(a.output/'state.json',state)
    while process_alive(a.root_pid):
        if child.poll() is not None:raise RuntimeError('Preloaded successor exited')
        time.sleep(5)
    if 'IsADirectoryError' not in read(old/'failure.json')['error']:
        raise ValueError('Old Root exited unexpectedly')
    import torch
    saved=torch.load(old/'latest.tmp.pt',map_location='cpu',mmap=True,weights_only=False)
    progress=read(old/'progress.json');step=saved['actual_updates']
    if (step!=progress['actual_updates'] or saved['actual_exposures']!=step*256
            or saved['sample_digest']!=progress['sample_digest']
            or {int(x['step']) for x in saved['optimizer_state']['state'].values()}!={step}
            or saved['scheduler_state']['last_epoch']!=step):
        raise ValueError('Checkpoint does not match the last complete optimizer update')
    os.link(old/'latest.tmp.pt',handoff)
    (old/'latest.pt').rmdir();os.link(handoff,old/'latest.pt')
    sources={p.relative_to(code).as_posix():digest(p) for folder in ('motionbricks','scripts')
        for p in sorted((code/folder).rglob('*.py'))}
    save(migration_path,dict(transition='root-parallel-four-v1' if four else 'root-parallel-memory-v1',from_checkpoint_sha256=digest(handoff),
        from_source_code_sha256=saved['contract']['source_code_sha256'],to_source_code_sha256=sources,
        actual_updates=step,parallel_implementation_sha256=parallel_hash,proofs=proofs,
        no_optimizer_updates_lost=True,old_failure_is_intentional_checkpoint_barrier=True))
    del saved
    state.update(status='running',handoff_updates=step)
    save(a.output/'state.json',state)
    if child.wait():raise RuntimeError('Parallel Root failed; inspect retained log')
    root_complete=read(a.output/'root/complete.json')
    if (root_complete['actual_updates']!=50000 or root_complete['actual_exposures']!=12800000
            or not root_complete['protected_hash_passed']
            or digest(a.output/'root/weights.pt')!=root_complete['checkpoint_sha256']):
        raise ValueError('Root completion identity or budget mismatch')
    for stage,parent in [('codec',a.output/'root/weights.pt'),('pose',a.output/'codec/weights.pt')]:
        state['stage']=stage
        argv=command(stage,parent)
        # Dependent stages retain the execution policy proved for this workflow.
        for label,extra in [(stage+'-startup',['--stop-after','2']),
                            (stage+'-resume',['--resume',str(a.output/stage/'latest.pt')])]:
            while True:
                snapshot,available=gpu_snapshot()
                if available:break
                state.update(status='waiting_for_idle_gpu',gpu_snapshot=snapshot)
                save(a.output/'state.json',state);time.sleep(10)
            selected=next((gpu for gpu in a.gpus if gpu in available),available[0])
            state['status']='running'
            process=launch(label,argv+extra,[selected])
            if process.wait():raise RuntimeError(label+' failed; inspect retained log')
            if label.endswith('startup') and read(a.output/stage/'stopped.json')['actual_updates']!=2:
                raise ValueError('Startup accounting mismatch')
            if label.endswith('resume'):
                complete=read(a.output/stage/'complete.json')
                if (complete['actual_updates']!=50000 or complete['actual_exposures']!=12800000
                        or not complete['protected_hash_passed']
                        or digest(a.output/stage/'weights.pt')!=complete['checkpoint_sha256']):
                    raise ValueError('Dependent stage completion identity or budget mismatch')
    if any(read(a.output/stage/'complete.json')['actual_updates']!=50000 for stage in ('root','codec','pose')):
        raise ValueError('Stage budget incomplete')
    save(a.output/'complete.json',dict(actual_updates=150000,actual_exposures=38400000,
        final_checkpoint=str(a.output/'pose/weights.pt'),quality_approved=False))
    state['status']='complete';save(a.output/'state.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root-pid',type=int,required=True)
    parser.add_argument('--previous-supervisor',type=int,required=True)
    parser.add_argument('--gpus',type=int,nargs='+',required=True)
    parser.add_argument('--root-parallel-devices',type=int,choices=(2,4),default=2)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.is_dir():save(args.output/'failure.json',dict(error=repr(error)))
        raise
