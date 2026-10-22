"""Bounded32-candidate collection; no automatic training or model replacement."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def write(path, value):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def prepare(args):
    import build_parc_stair_course as base
    from build_parc_rugged_course import build_rugged_course
    base.PARC_REPO=args.base/'parc_runtime_v1/wide_v1/out/vendor/PARC'
    args.output.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for index in range(8):
        terrain,nodes,course=build_rugged_course(index)
        jobs.append(dict(layout=index,family=course['requested_family'],split=course['split'],
            seed=2026101350+index,terrain_sha256=course['terrain_sha256'],
            route_nodes=len(nodes),course=course,candidates=4))
    if len({row['terrain_sha256'] for row in jobs})!=8:raise ValueError('Duplicate terrain identity')
    parent=args.base/'model_terrain_v3_20261011/parc_stair64_20261013/pure_parc_fit_v1/episode-weights.pt'
    expected='ae558ee88542191b6cbe72659bd96f1482711ea2d8beec6c1641a0ae5b814934'
    if digest(parent)!=expected:raise ValueError('Approved baseline identity changed')
    write(args.output/'plan.json',dict(schema='motionbricks.parc_rugged32.v1',jobs=jobs,
        total_candidates=32,split_unit='whole authored geometry and all its candidates',
        stop_task=False,history_reset_between_sections=False,approved_model=str(parent),
        approved_model_sha256=expected,training_started=False,postprocessing=None,
        teacher_pipeline='native PARC, official physical tracking, existing SOMA retarget/screen',
        actor_pipeline='terrain-aware MotionBricks; no output correction',
        admission_is_not_controller_acceptance=True))
    print(json.dumps(dict(preflight_layouts=8,candidates=32,geometry_verified=True)),flush=True)


def worker(args):
    if args.gpu==1 or args.gpu not in range(8):raise ValueError('Explicit idle card required; GPU1 reserved')
    plan=json.loads((args.output/'plan.json').read_text())
    indices=[int(value) for value in args.layouts.split(',')]
    if not indices or len(set(indices))!=len(indices) or any(i not in range(8) for i in indices):
        raise ValueError('Unique declared layout indices required')
    runtime=args.base/'parc_runtime_v1';code=Path(__file__).resolve().parents[1]
    generator=code/'scripts'/args.generator_script
    if Path(args.generator_script).name!=args.generator_script or not generator.is_file():
        raise ValueError('Generation script must exist inside this code snapshot')
    physics_script=code/'scripts'/args.physics_script
    if Path(args.physics_script).name!=args.physics_script or not physics_script.is_file():
        raise ValueError('Physics script must exist inside this code snapshot')
    if Path(args.recording_subdir).name!=args.recording_subdir:
        raise ValueError('Recording subdirectory must be a simple name')
    progress=args.output/('worker_'+str(args.gpu)+'.json')
    state=dict(status='starting',pid=os.getpid(),gpu=args.gpu,layouts=indices,jobs=[],training_started=False,
               plan_sha256=digest(args.output/'plan.json'))

    def idle():
        for _ in range(36):
            status=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
                '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            uuid,memory,util=[v.strip() for v in status.split(',')]
            if int(memory)<=256 and int(util)<=5:return uuid,status
            state.update(status='waiting_for_idle_gpu',occupancy=status);write(progress,state)
            time.sleep(5)
        raise RuntimeError('GPU busy; no other process modified')

    @contextmanager
    def owned_card():
        with (runtime/('gpu'+str(args.gpu)+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield idle()

    with (args.output/('worker_'+str(args.gpu)+'.lock')).open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if progress.exists():
            if not args.resume:raise ValueError('Worker has a receipt; explicit recovery required')
            prior=json.loads(progress.read_text())
            if prior['plan_sha256']!=state['plan_sha256'] or prior['layouts']!=indices:
                raise ValueError('Cannot resume a changed plan or layout assignment')
            if prior['status']=='collection_complete':return
            write(args.output/('worker_'+str(args.gpu)+'.prior.'+str(time.time_ns())+'.json'),prior)
            state.update(jobs=prior['jobs'],resumed_from_pid=prior['pid'])
        write(progress,state)
        for index in indices:
            job=plan['jobs'][index];directory=args.output/('layout_%02d'%index)
            row=next((r for r in state['jobs'] if r['layout']==index),None)
            if row is not None and row['status'] in ('screened','no_complete_route'):continue
            directory.mkdir(exist_ok=args.resume)
            recording=directory/args.recording_subdir
            if row is None:
                row=dict(layout=index,family=job['family'],split=job['split'],status='starting')
                state['jobs'].append(row)
            def stage(name,command,env,cwd,timeout):
                row.pop('error',None)
                row.update(status=name,command=command,started_at=time.time());state['status']=name
                log_path=directory/(name+'.log');attempt=1
                while log_path.exists():
                    attempt+=1;log_path=directory/(name+'-attempt'+str(attempt)+'.log')
                row['log']=str(log_path)
                with log_path.open('x') as log:
                    child=subprocess.Popen(command,env=env,cwd=cwd,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT)
                    row['pid']=child.pid;write(progress,state)
                    try:result=child.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise RuntimeError('Own stage timed out: '+name)
                if result:raise RuntimeError(name+' exit '+str(result))
            env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled',
                PATH=str(runtime/'env/bin')+':'+os.environ.get('PATH',''),
                LD_LIBRARY_PATH=str(runtime/'env/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
                PYTHONPATH=str(runtime/'sdk/isaacgym/python'),
                TORCH_EXTENSIONS_DIR=str(runtime/'torch_extensions'),MAX_JOBS='4')
            try:
                generated_path=directory/'generation/generation.json'
                if generated_path.exists():
                    saved=json.loads(generated_path.read_text())
                    if saved['status']!='kinematic_candidates_generated_physics_pending' or not saved.get('prefix_contract'):
                        raise ValueError('Existing generation incomplete; preserve it for explicit diagnosis')
                    row['generation_reused']=True
                else:
                    with owned_card() as (uuid,occupancy):
                        env['CUDA_VISIBLE_DEVICES']=uuid;row['occupancy_before_generation']=occupancy
                        stage('generation',[str(runtime/'env/bin/python'),'-u',str(generator),
                            '--repo',str(runtime/'wide_v1/out/vendor/PARC'),
                            '--checkpoint',str(runtime/'wide_v1/out/parc_legacy_models_20261008/demo_iter3_verified.ckpt'),
                            '--output',str(directory/'generation'),'--layout',str(index),'--seed',str(job['seed']),
                            '--terrain-sha256',job['terrain_sha256']],env,runtime/'wide_v1',1800)
                generated=json.loads((directory/'generation/generation.json').read_text())
                actual=json.loads((directory/'generation/course/course_manifest.json').read_text())
                row.update(candidates=len(generated['candidates']),route_complete=generated['geometric_coverage_candidates'],
                    prefix_error=generated['prefix_contract']['max_prefix_error'])
                if actual['terrain_sha256']!=job['terrain_sha256']:
                    raise ValueError('Generated geometry differs from preflight')
                if not row['route_complete']:
                    row.update(status='no_complete_route',training_started=False);write(progress,state);continue
                physics_report=recording/'physics/report.json'
                if not physics_report.exists():
                    idle()
                    stage('physics',[str(runtime/'env/bin/python'),'-u',str(physics_script),
                        '--runtime',str(runtime),'--generation',str(directory/'generation'),
                        '--output',str(recording),'--gpu',str(args.gpu),
                        '--completed-only','--min-seconds','0','--require-full-route'],env,runtime/'wide_v1',1200)
                if json.loads(physics_report.read_text())['status']!='official_recording_finished':
                    raise ValueError('Existing physical recording incomplete; preserve it for diagnosis')
                if not (recording/'export/manifest.json').exists():
                    stage('export',[str(runtime/'env/bin/python'),str(runtime/'export_parc_physics_recordings.py'),
                        '--bundle',str(recording/'inputs'),'--run',str(recording/'physics'),
                        '--output',str(recording/'export')],env,runtime/'wide_v1',300)
                if not (directory/'admission/summary.json').exists():
                    with owned_card():
                        mb_env=dict(os.environ,PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),
                            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                        stage('soma',[str(args.base/'env/bin/python'),'-u','-m',args.admission_module,
                            '--recording',str(recording),'--output',str(directory/'admission'),
                            '--workers','2','--split',job['split'],'--terrain-family',job['family'],
                            '--sdf-gpu',str(args.gpu)],mb_env,code,2400)
                summary=json.loads((directory/'admission/summary.json').read_text())
                row.update(status='screened',summary=summary,training_started=False)
            except Exception as error:
                row.update(status='failed',error=type(error).__name__+': '+str(error));state['status']='failed'
                write(progress,state)
                raise
            write(progress,state)
        if digest(Path(plan['approved_model']))!=plan['approved_model_sha256']:
            raise ValueError('Approved model changed during collection')
        state.update(status='collection_complete',approved_model_unchanged=True,training_started=False)
        write(progress,state)
        print(json.dumps(state),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--gpu',type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--layouts')
    p.add_argument('--generator-script',default='generate_parc_rugged_candidates.py')
    p.add_argument('--admission-module',default='scripts.admit_parc_long_route_probe')
    p.add_argument('--physics-script',default='probe_parc_long_route_physics.py')
    p.add_argument('--recording-subdir',default='recording')
    args=p.parse_args()
    prepare(args) if args.prepare else worker(args)
