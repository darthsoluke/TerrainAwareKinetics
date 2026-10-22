"""Adopt the live dual-GPU Root, then use verified codec/Pose execution policies."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_pfnn_bias_workflow import read,save,digest,process_alive
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def stage_command(a,stage,parent):
    run=a.base/'pfnn_mixed_20261015'
    command=[sys.executable,'-u','-m','scripts.train_unified_full_corpus','--stage',stage,
        '--bundle',str(a.base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'),
        '--parent',str(parent),'--preflight',str(run/'preflight_world_v1/complete.json'),
        '--cache',str(run/'cache_mixed_v1'),str(run/'cache_pfnn_world_v2'),
        '--output',str(a.output/stage),'--steps','50000','--batch','256','--micro-batch','64',
        '--workers','4','--corpus-audit',str(run/'full_training_v3/corpus_audit.json'),
        '--coalesce-dilated-bias']
    command += ['--cuda-memory-fraction','0.1'] if stage=='codec' else ['--cache-pose-tokens']
    return command


def validate(a):
    run=a.base/'pfnn_mixed_20261015';code=Path(__file__).resolve().parents[1]
    proof_names=('codec_memory_compare_v1','codec_memory_fit_v1','pose_cold_bias_v1',
        'pose_token_speed_v3','pose_token_resume_v1')
    proofs=[]
    for name in proof_names:
        path=run/name/'complete.json';data=read(path)
        if name=='codec_memory_fit_v1':
            if (data['actual_optimizer_updates']!=2 or data['production_budget_updates']!=0
                    or data['memory_fraction']!=.1 or data['rows'][-1]['optimizer_steps']!=[2]
                    or not data['initial_weights_restored']):
                raise ValueError('Codec full-optimizer memory fit is unproven')
        elif not data['passed']:
            raise ValueError('Execution proof failed: '+name)
        proofs.append(dict(name=name,path=str(path),sha256=digest(path)))
    if read(run/'pose_cold_bias_v1/complete.json')['implementation_sha256']!=digest(
            code/'motionbricks/data/dilated_bias_convolution.py'):
        raise ValueError('Cold-cache implementation differs from its proof')
    if not read(run/'codec_memory_compare_v1/complete.json')['tolerances_unchanged']:
        raise ValueError('Codec numerical gates changed')
    contract=read(a.previous/'root/contract.json')
    if (contract['stage']!='root' or contract['mechanics_only'] or contract['steps']!=50000
            or contract['batch']!=256 or contract['micro_batch']!=64 or contract['root_parallel_devices']!=2):
        raise ValueError('Unexpected Root budget or execution contract')
    for stage in ('codec','pose'):
        if (a.previous/stage).exists():raise ValueError('Previous workflow already advanced')
    return proofs


def main(a):
    code=Path(__file__).resolve().parents[1];proofs=validate(a)
    if a.validate_only:
        print(json.dumps(dict(validated=True,proofs=proofs,
            codec_command=stage_command(a,'codec',a.previous/'root/weights.pt'),
            pose_command=stage_command(a,'pose',a.output/'codec/weights.pt'))))
        return
    root=a.previous/'root'
    if not process_alive(a.root_pid) or not process_alive(a.previous_supervisor):
        raise ValueError('Expected live Root and supervisor')
    if str(root) not in (Path('/proc')/str(a.root_pid)/'cmdline').read_bytes().decode():
        raise ValueError('Live Root identity mismatch')
    if 'run_pfnn_parallel_workflow' not in (Path('/proc')/str(a.previous_supervisor)/'cmdline').read_bytes().decode():
        raise ValueError('Previous supervisor identity mismatch')
    progress=read(root/'progress.json')
    if progress['actual_updates']>=49900:raise ValueError('Adoption too close to stage transition')
    a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'execution_proofs.json',dict(proofs=proofs,total_budget_updates=150000))
    state=dict(pid=os.getpid(),stage='root',status='adopting_running_root',child_pid=a.root_pid,
        root_directory=str(root),root_restarted=False,total_budget_updates=150000,quality_approved=False)
    save(a.output/'state.json',state)
    save(a.previous/'codec_pose_execution_handoff.json',dict(previous_supervisor_pid=a.previous_supervisor,
        root_pid=a.root_pid,root_directory=str(root),successor_pid=os.getpid(),successor=str(a.output),
        root_restarted=False,proofs=proofs))
    os.kill(a.previous_supervisor,15)

    def execute(label,argv):
        while True:
            snapshot,available=gpu_snapshot()
            if available:break
            state.update(status='waiting_for_idle_gpu',gpu_snapshot=snapshot)
            save(a.output/'state.json',state);time.sleep(15)
        gpu=available[0]
        with (a.output/(label+'.log')).open('x') as stream:
            child=subprocess.Popen(argv,cwd=code,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TMPDIR='${PRIVATE_STORAGE}/mbtmp',
                    PYTHONPATH='.:out/vendor/kimodo',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            state.update(status='running',job=label,command=argv,child_pid=child.pid,gpu=gpu,gpu_snapshot_before=snapshot)
            save(a.output/'state.json',state)
            if child.wait():raise RuntimeError(label+' failed; inspect retained log')

    while not (root/'complete.json').exists():
        if (root/'failure.json').exists():raise RuntimeError('Adopted Root failed; inspect existing failure')
        if not process_alive(a.root_pid):raise RuntimeError('Adopted Root exited without completion')
        state['root_progress']=read(root/'progress.json');save(a.output/'state.json',state);time.sleep(30)
    completed=read(root/'complete.json')
    if (completed['actual_updates']!=50000 or completed['actual_exposures']!=12800000
            or not completed['protected_hash_passed'] or digest(root/'weights.pt')!=completed['checkpoint_sha256']):
        raise ValueError('Root completion identity or budget mismatch')
    parent=root/'weights.pt'
    for stage in ('codec','pose'):
        state['stage']=stage;argv=stage_command(a,stage,parent)
        execute(stage+'-startup',argv+['--stop-after','2'])
        startup=read(a.output/stage/'stopped.json')
        if startup['actual_updates']!=2 or startup['actual_exposures']!=512:raise ValueError('Startup accounting mismatch')
        execute(stage+'-resume',argv+['--resume',str(a.output/stage/'latest.pt')])
        complete=read(a.output/stage/'complete.json')
        if (complete['actual_updates']!=50000 or complete['actual_exposures']!=12800000
                or not complete['protected_hash_passed']):raise ValueError('Stage budget incomplete')
        parent=a.output/stage/'weights.pt'
    state.update(status='training_complete_quality_review_required',checkpoint=str(parent),
        actual_updates=150000,actual_exposures=38400000,quality_approved=False)
    save(a.output/'complete.json',state);save(a.output/'state.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','previous','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--root-pid',type=int,required=True)
    parser.add_argument('--previous-supervisor',type=int,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.is_dir():save(args.output/'failure.json',dict(error=repr(error)))
        raise
