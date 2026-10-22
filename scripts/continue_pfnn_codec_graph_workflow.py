"""Preserve the completed Root and migrate the live codec at a full checkpoint."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.run_pfnn_bias_workflow import read, save, digest, process_alive
from scripts.run_pfnn_parallel_workflow import gpu_snapshot
from scripts.continue_pfnn_codec_pose_execution import stage_command


def main(a):
    code = Path(__file__).resolve().parents[1]
    run = a.base/'pfnn_mixed_20261015'
    old = a.previous/'codec'
    exposure=getattr(a,'exposure_pipeline',False)
    devices=getattr(a,'pipeline_devices',2)
    four=exposure and devices==4
    adopted=getattr(a,'adopt_preload',None)
    selected=a.gpus if exposure else [a.gpu]
    if exposure and (selected is None or len(selected)!=devices or len(set(selected))!=devices or selected[0]!=a.gpu):
        raise ValueError('Distinct GPUs matching the exposure pipeline are required')
    root = Path(read(a.previous/'root_completion.json')['directory']) if exposure else a.previous/'root'
    label='exposure_four' if four else 'exposure' if exposure else 'graph'
    transition='codec-reference-exposure-four-v1' if four else 'codec-reference-exposure-v1' if exposure else 'codec-dilated-graph-v1'
    root_complete = read(root/'complete.json')
    if (root_complete['actual_updates']!=50000 or root_complete['actual_exposures']!=12800000
            or not root_complete['protected_hash_passed']
            or digest(root/'weights.pt')!=root_complete['checkpoint_sha256']):
        raise ValueError('Completed Root identity differs')
    if (a.previous/'pose').exists(): raise ValueError('Previous workflow already started Pose')
    proof_names=([('reference','codec_exposure_reference_v1'),('continuous','codec_exposure_parallel_v1'),
        ('cold_resume','codec_exposure_resume_v1')] if exposure else [
        ('optimizer_resume','codec_graph_workspace_optimizer_v1'),
        ('stage_integration','codec_graph_integration_v1'),
        ('cold_process','codec_graph_cold_process_v1')])
    if four:proof_names=[(role,name.replace('codec_exposure_','codec_four_exposure_')) for role,name in proof_names]
    proofs = [dict(role=role,path=str(run/name/'complete.json')) for role,name in proof_names]
    for proof in proofs: proof['sha256']=digest(Path(proof['path']))
    if exposure:
        for proof in proofs:proof['source_directory']=str(run/('code_codec_four_exposure_optimizer_v1' if four else 'code_codec_exposure_optimizer_v1'))
    sources = {p.relative_to(code).as_posix():digest(p) for folder in ('motionbricks','scripts')
        for p in sorted((code/folder).rglob('*.py'))}
    graph_hash = sources['motionbricks/data/dilated_graph_convolution.py']
    import torch
    from scripts.train_unified_full_corpus import validate_implementation_migration
    fixture = run/('codec_four_exposure_inputs_v1/checkpoint.pt' if four else 'codec_pipeline_inputs_v1/checkpoint.pt' if exposure else 'codec_execution_inputs_v1/checkpoint.pt')
    saved = torch.load(fixture,map_location='cpu',mmap=True,weights_only=False)
    candidate = dict(saved['contract'],graph_dilated_convolutions=True,cuda_memory_fraction=.9 if exposure else .125,
        source_code_sha256=sources,implementation_migration_sha256='pending')
    if exposure:
        from motionbricks.data.reference_exposure_budget import ExposureBudget,validate_completion_counts
        candidate.update(batch=512*devices,micro_batch=512,codec_pipeline_devices=devices,
            quantizer_ema_updates_per_optimizer_step=8*devices,
            exposure_budget=ExposureBudget(saved['actual_updates'],saved['actual_exposures'],512*devices).contract())
    precheck = dict(transition=transition,from_checkpoint_sha256=digest(fixture),
        from_source_code_sha256=saved['contract']['source_code_sha256'],to_source_code_sha256=sources,
        actual_updates=saved['actual_updates'],graph_implementation_sha256=graph_hash,proofs=proofs)
    if exposure:precheck.update(actual_exposures=saved['actual_exposures'],
        authorization='user_requested_multigpu_large_memory_reference_50000',optimizer_trajectory_equivalent=False)
    validate_implementation_migration(saved,candidate,precheck,precheck['from_checkpoint_sha256'])
    del saved
    future_proofs = read(a.previous/'execution_proofs.json')
    for proof in future_proofs['proofs']:
        if digest(Path(proof['path']))!=proof['sha256']: raise ValueError('Pose execution proof changed')
    if a.validate_only:
        print(json.dumps(dict(validated=True,proofs=proofs,graph_implementation_sha256=graph_hash)))
        return
    if adopted:
        reassignment=read(adopted/'resource_reassignment.json')
        barrier=read(old/(label+'_handoff_barrier.json'))
        if (not exposure or reassignment.get('actual_optimizer_updates')!=0
                or reassignment.get('old_codec_pid')!=a.codec_pid
                or barrier.get('owner_pid')!=a.codec_pid
                or process_alive(reassignment['supervisor_pid']) or process_alive(reassignment['child_pid'])
                or (adopted/'codec/progress.json').exists() or (adopted/'migration.json').exists()):
            raise ValueError('Only a stopped zero-update preload with its original barrier may be reassigned')
    snapshot, available = gpu_snapshot()
    if any(gpu not in available for gpu in selected): raise ValueError('Selected GPU is no longer idle')
    if not adopted and (not process_alive(a.codec_pid) or not process_alive(a.previous_supervisor)):
        raise ValueError('Expected active codec and supervisor')
    codec_command = (Path('/proc')/str(a.codec_pid)/'cmdline').read_bytes().decode() if process_alive(a.codec_pid) else ''
    supervisor_command = (Path('/proc')/str(a.previous_supervisor)/'cmdline').read_bytes().decode() if process_alive(a.previous_supervisor) else ''
    if not adopted and (str(old) not in codec_command or 'train_unified_full_corpus' not in codec_command
            or str(a.previous) not in supervisor_command
            or ('continue_pfnn_codec_graph_workflow' if exposure else 'run_pfnn_parallel_workflow') not in supervisor_command
            or read(a.previous/'state.json')['child_pid']!=a.codec_pid):
        raise ValueError('Previous process identity mismatch')
    handoff = old/(label+'_handoff.pt')
    if not adopted and (handoff.exists() or (old/('latest_before_'+label+'_handoff.pt')).exists()):
        raise ValueError('Codec handoff already exists')
    a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'execution_proofs.json',future_proofs)
    save(a.output/'root_completion.json',dict(directory=str(root),**root_complete))
    state = dict(pid=os.getpid(),stage='codec',status='preloading_codec',root_directory=str(root),
        total_budget_updates=150000,quality_approved=False)
    if exposure:
        state.pop('total_budget_updates')
        state.update(total_budget_reference_updates=150000,total_budget_exposures=38400000,
            budget_unit='reference_sample_exposures',optimizer_trajectory_equivalent=False)
    save(a.output/'state.json',state)

    def launch(label, command, gpu, snapshot):
        devices=gpu if isinstance(gpu,list) else [gpu]
        with (a.output/(label+'.log')).open('x') as log:
            child = subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,devices)),TMPDIR='${PRIVATE_STORAGE}/mbtmp',
                    PYTHONPATH='.:out/vendor/kimodo',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        state.update(child_pid=child.pid,job=label,command=command,job_gpus=devices,gpu_snapshot_before=snapshot)
        save(a.output/'state.json',state)
        return child

    migration_path = a.output/'migration.json'
    command = stage_command(a,'codec',root/'weights.pt')
    command[command.index('--cuda-memory-fraction')+1]='0.9' if exposure else '0.125'
    if exposure:
        command[command.index('--batch')+1]=str(512*devices)
        command[command.index('--micro-batch')+1]='512'
        command+=['--codec-exposure-pipeline','--codec-pipeline-devices',str(devices)]
    command += ['--graph-dilated-convolutions','--resume',str(handoff),
        '--resume-migration',str(migration_path),'--await-resume']
    child = launch('codec-resume',command,selected,snapshot)
    while not (a.output/'codec-await-resume.json').exists():
        if child.poll() is not None: raise RuntimeError('Codec preload failed; original codec remains unchanged')
        time.sleep(5)
    if read(a.output/'codec-await-resume.json')['pid']!=child.pid:
        raise ValueError('Unexpected preload acknowledgment')
    save(a.previous/('codec_'+label+('_reassignment.json' if adopted else '_handoff.json')),dict(successor_pid=os.getpid(),successor=str(a.output),
        previous_supervisor_pid=a.previous_supervisor,codec_pid=a.codec_pid,preloaded_child=child.pid,proofs=proofs))
    if not adopted:
        os.kill(a.previous_supervisor,signal.SIGTERM)
        owner_start=(Path('/proc')/str(a.codec_pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
        watchdog=(f'import os,signal,time,pathlib; time.sleep(55); p=pathlib.Path("/proc/{a.codec_pid}/stat"); '
            f'os.kill({a.codec_pid},signal.SIGCONT) if p.exists() and '
            f'p.read_text().rsplit(")",1)[1].split()[19]=={owner_start!r} else None')
        subprocess.Popen([sys.executable,'-c',watchdog],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,start_new_session=True)
        os.kill(a.codec_pid,signal.SIGSTOP)
        try:
            for _ in range(150):
                status=next(x.split()[1] for x in (Path('/proc')/str(a.codec_pid)/'status').read_text().splitlines()
                    if x.startswith('State:'))
                if status=='T':break
                time.sleep(.02)
            if status!='T': raise RuntimeError('Could not arm checkpoint barrier atomically')
            progress=read(old/'progress.json')
            (old/'latest.pt').rename(old/('latest_before_'+label+'_handoff.pt'))
            (old/'latest.pt').mkdir()
            save(old/(label+'_handoff_barrier.json'),dict(owner_pid=a.codec_pid,owner_starttime=owner_start,
                published_updates_before=progress['actual_updates'],successor_pid=os.getpid(),
                intentional_stop='Complete torch.save then IsADirectoryError before the next optimizer update'))
        finally:
            os.kill(a.codec_pid,signal.SIGCONT)
    state['status']='waiting_for_complete_checkpoint';save(a.output/'state.json',state)
    while process_alive(a.codec_pid):
        if child.poll() is not None:raise RuntimeError('Preloaded codec exited')
        time.sleep(5)
    if 'IsADirectoryError' not in read(old/'failure.json')['error']:
        raise ValueError('Original codec exited unexpectedly')
    saved=torch.load(old/'latest.tmp.pt',map_location='cpu',mmap=True,weights_only=False)
    progress=read(old/'progress.json');step=saved['actual_updates']
    if four:
        prior=saved['contract']['exposure_budget']
        ExposureBudget(prior['initial_updates'],prior['initial_exposures'],1024,step,saved['actual_exposures'])
    reference_step=saved['actual_exposures']//256
    if (step!=progress['actual_updates'] or (not four and saved['actual_exposures']!=step*256)
            or saved['sample_digest']!=progress['sample_digest']
            or {int(x['step']) for x in saved['optimizer_state']['state'].values()}!={step}
            or saved['scheduler_state']['last_epoch']!=(reference_step if four else step)):
        raise ValueError('Codec checkpoint does not identify the last complete update')
    os.link(old/'latest.tmp.pt',handoff)
    (old/'latest.pt').rmdir();os.link(handoff,old/'latest.pt')
    migration=dict(transition=transition,from_checkpoint_sha256=digest(handoff),
        from_source_code_sha256=saved['contract']['source_code_sha256'],to_source_code_sha256=sources,
        actual_updates=step,graph_implementation_sha256=graph_hash,proofs=proofs,
        no_optimizer_updates_lost=True,old_failure_is_intentional_checkpoint_barrier=True)
    if exposure:migration.update(actual_exposures=saved['actual_exposures'],
        authorization='user_requested_multigpu_large_memory_reference_50000',optimizer_trajectory_equivalent=False)
    save(migration_path,migration);del saved
    state.update(status='running',handoff_updates=step,handoff_reference_updates=reference_step);save(a.output/'state.json',state)
    if child.wait():raise RuntimeError('Migrated codec failed; inspect retained evidence')
    completed=read(a.output/'codec/complete.json')
    if exposure:
        validate_completion_counts(read(a.output/'codec/contract.json'),completed['actual_updates'],completed['actual_exposures'])
    if ((not exposure and completed['actual_updates']!=50000) or completed['actual_exposures']!=12800000
            or not completed['protected_hash_passed']
            or digest(a.output/'codec/weights.pt')!=completed['checkpoint_sha256']):
        raise ValueError('Codec completion identity or budget differs')
    state['stage']='pose'
    command=stage_command(a,'pose',a.output/'codec/weights.pt')
    for label,extra in [('pose-startup',['--stop-after','2']),
            ('pose-resume',['--resume',str(a.output/'pose/latest.pt')])]:
        while True:
            snapshot,available=gpu_snapshot()
            if available:break
            state['status']='waiting_for_idle_gpu';save(a.output/'state.json',state);time.sleep(15)
        state['status']='running'
        process=launch(label,command+extra,available[0],snapshot)
        if process.wait():raise RuntimeError(label+' failed; inspect retained log')
        if label.endswith('startup'):
            startup=read(a.output/'pose/stopped.json')
            if startup['actual_updates']!=2 or startup['actual_exposures']!=512:raise ValueError('Pose startup accounting differs')
    pose=read(a.output/'pose/complete.json')
    if (pose['actual_updates']!=50000 or pose['actual_exposures']!=12800000 or not pose['protected_hash_passed']
            or digest(a.output/'pose/weights.pt')!=pose['checkpoint_sha256']):raise ValueError('Pose completion differs')
    state.update(status='training_complete_quality_review_required',
        actual_updates=root_complete['actual_updates']+completed['actual_updates']+pose['actual_updates'],actual_exposures=38400000,
        final_checkpoint=str(a.output/'pose/weights.pt'))
    if exposure:state['equivalent_reference_updates']=150000
    save(a.output/'complete.json',state);save(a.output/'state.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','previous','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--codec-pid',type=int,required=True)
    parser.add_argument('--previous-supervisor',type=int,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--gpus',type=int,nargs='+')
    parser.add_argument('--pipeline-devices',type=int,choices=(2,4),default=2)
    parser.add_argument('--exposure-pipeline',action='store_true')
    parser.add_argument('--adopt-preload',type=Path)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.is_dir():save(args.output/'failure.json',dict(error=repr(error)))
        raise
