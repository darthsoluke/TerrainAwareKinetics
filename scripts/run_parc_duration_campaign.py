"""A100-local PARC waves with per-scene SOMA admission and a real-duration gate."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.parc_collection_budget import POLICY, FAMILIES, motion_identity, summarize
from scripts.train_native_mb import sha256


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def build_wave(output, next_indices, coverage):
    from scripts.build_parc_diverse_course import expanded_course_spec, build_diverse_course
    plan_dir = output/'plan'
    plan_dir.mkdir(parents=True, exist_ok=True)
    jobs, rejected = [], []
    family_order = sorted(FAMILIES, key=lambda name: coverage['by_split_family']['train'][name]['credited_seconds'])
    # Four independent layouts per family; later waves replenish coverage gaps.
    for _ in range(4):
        for family in family_order:
            for attempt in range(40):
                index = next_indices[family]
                next_indices[family] += 1
                spec = expanded_course_spec(family, index)
                split = spec['split']
                if split != 'train' and not any(row['family'] == family and row['split'] == split
                                               for row in coverage['coverage_deficits']):
                    continue
                try:
                    terrain, nodes, course = build_diverse_course(spec)
                except (ValueError, RuntimeError) as error:
                    rejected.append({'scene': spec['scene_id'], 'error': str(error)})
                    continue
                path = plan_dir/(spec['scene_id']+'.json')
                save(path, spec)
                length = course['course']['planar_length_m']
                # Explicit corpus budget extension; failed/incomplete routes never count.
                maximum = 90 if family == 'mixed' else 45
                seconds = min(maximum, max(15, math.ceil(length/.6 + 8)))
                jobs.append({'id': spec['scene_id'], 'spec': path.name, 'family': family,
                    'split': split, 'motion_seed': spec['motion_seed'], 'max_seconds': seconds,
                    'terrain_sha256': course['terrain_sha256'], 'route_length_m': length,
                    'difficulty': spec['difficulty'], 'reverse_route': spec['reverse_route'],
                    'terrain_parameters': course['terrain_parameters']})
                break
            else:
                raise RuntimeError('Geometry preflight exhausted for '+family)
    plan = {'version':'parc-duration-wave-v1', 'jobs':jobs, 'layouts':len(jobs),
            'candidate_batch':POLICY['candidate_batch'], 'retained_per_layout':POLICY['retained_per_layout'],
            'candidate_pose_min_rms_m':POLICY['candidate_pose_min_rms_m'],
            'max_attempts_per_layout':POLICY['max_attempts_per_layout'],
            'selection_policy':'pose_farthest_point_v1',
            'split_rule':'layout index modulo 10: 0 validation, 1 test, others train; assigned before scores',
            'max_seconds_rule':'ceil(route_m / 0.6 + 8), min 15, cap 45 single / 90 mixed; not an upstream default',
            'physics_required':True, 'training_approved':False, 'preflight_rejected':rejected,
            'next_indices':dict(next_indices)}
    save(plan_dir/'plan.json',plan)
    return plan


def wait_adopted_children(path, on_tick=None):
    if path is None:
        return
    for process in read(path)['preserved_children']:
        proc = Path('/proc')/str(process['pid'])
        while proc.exists():
            # A reaped PID can be reused. Never wait on an unrelated replacement.
            try:
                command = (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                if process['command'] != command or (proc/'stat').read_text().split()[2] == 'Z':
                    break
            except FileNotFoundError:
                break
            if on_tick:
                on_tick()
            time.sleep(15)


def main():
    import fcntl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='2,3')
    p.add_argument('--adopted-children',type=Path)
    p.add_argument('--legacy-collection',type=Path)
    p.add_argument('--legacy-scenes',nargs='*',default=[])
    p.add_argument('--max-waves',type=int,default=64)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--retry-scenes',nargs='*',default=[])
    p.add_argument('--elastic',action='store_true')
    p.add_argument('--adopt-workers',type=Path)
    args = p.parse_args()
    cards = [int(value) for value in args.gpus.split(',')]
    if not cards or len(set(cards)) != len(cards) or any(card not in range(8) for card in cards):
        p.error('GPU indices must be distinct physical cards in 0..7')
    if not args.elastic and len(cards) != 2:
        p.error('Use --elastic for a variable GPU pool')
    args.output.mkdir(parents=True,exist_ok=True)
    lock = (args.output/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.output/'dataset_ready.json').exists():
        from scripts.parc_collection_budget import verify_training_gate
        receipt = read(args.output/'dataset_ready.json')
        verify_training_gate(receipt)
        if sha256(Path(receipt['dataset'])/'native_manifest.json') != receipt['dataset_manifest_sha256']:
            raise ValueError('Completed dataset identity changed')
        return
    policy_file = args.output/'policy.json'
    if policy_file.exists() and read(policy_file) != POLICY:
        raise ValueError('Cannot mutate a running campaign policy')
    save(policy_file,POLICY)
    if args.prepare_only:
        path = args.output/'wave_000/plan/plan.json'
        plan = read(path) if path.exists() else build_wave(args.output/'wave_000',
            {family:0 for family in FAMILIES},summarize([]))
        print(json.dumps({'layouts':len(plan['jobs']),'policy':POLICY,'plan':str(path)}))
        return
    runtime = args.base/'parc_runtime_v1'
    work = runtime/'wide_v1'
    python = runtime/'env/bin/python'
    code = Path(__file__).resolve().parents[1]
    cpu_env = dict(os.environ,PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),
                   CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    worker_env = dict(os.environ,PATH=str(runtime/'env/bin')+':'+os.environ.get('PATH',''),
        LD_LIBRARY_PATH=str(runtime/'env/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
        PYTHONPATH=str(runtime/'sdk/isaacgym/python'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    worker_env.pop('CUDA_VISIBLE_DEVICES',None)
    state_path = args.output/'state.json'
    state = {'status':'waiting_for_preserved_stages','physical_gpus':cards,
             'scheduler':'elastic_scene_queue_v1' if args.elastic else 'fixed_pair_v1',
             'coordinator_pid':os.getpid(),
             'target_new_train_hours':2,'minimum_new_train_hours':1,'training_approved':False}
    rows, admitted_scenes = [], set()
    ledger = args.output/'catalog.json'
    if ledger.exists():
        old = read(ledger)
        rows, admitted_scenes = old['rows'], set(old['processed_scenes'])
    if args.retry_scenes:
        if any(row.get('scene_id') in args.retry_scenes and row.get('disposition')=='admitted' for row in rows):
            raise ValueError('Cannot discard already admitted motion during an infrastructure retry')
        if ledger.exists():
            shutil.copy2(ledger,args.output/('catalog.before_retry.'+str(time.time_ns())+'.json'))
        rows = [row for row in rows if row.get('scene_id') not in args.retry_scenes]
        admitted_scenes.difference_update(args.retry_scenes)
    pending = None
    children = []

    def record_progress(status):
        coverage = summarize(rows)
        state.update(status=status,updated_at=time.time(),campaign_coverage=coverage,
                     processed_scenes=len(admitted_scenes))
        save(state_path,state)
        save(ledger,{'schema':'parc.only_teacher_catalog.v1','rows':rows,
             'processed_scenes':sorted(admitted_scenes),'quality_approved':False})
        return coverage

    def poll_admission(collection,jobs):
        nonlocal pending
        if pending:
            proc,log,job,output = pending
            if proc.poll() is None:
                return
            log.close()
            if proc.returncode:
                raise RuntimeError('SOMA admission failed: '+str(output))
            fresh = read(output/'catalog.json')['rows']
            seen = {row.get('motion_content_sha256') for row in rows if row.get('disposition')=='admitted'}
            for row in fresh:
                if row.get('disposition') == 'admitted':
                    identity,seconds = motion_identity(row['retarget_path'])
                    row.update(motion_content_sha256=identity,admitted_seconds=seconds)
                    if identity in seen:
                        row.update(disposition='excluded',exclusion_reason='duplicate canonical SOMA motion')
                        row.pop('split',None)
                    else:
                        seen.add(identity)
                rows.append(row)
            admitted_scenes.add(job['id'])
            pending = None
            record_progress('generating_and_admitting')
        for job in jobs:
            if job['id'] in admitted_scenes or not (collection/job['id']/'receipt.json').exists():
                continue
            output = args.output/'admission'/job['id']
            if output.exists():
                output.rename(output.with_name(output.name+'.interrupted.'+str(time.time_ns())))
            log_path = output.with_suffix('.log'); log_path.parent.mkdir(parents=True,exist_ok=True)
            log = log_path.open('w')
            command = [sys.executable,'-u','-m','scripts.finish_parc_diverse_dataset',
                '--collection',str(collection),'--scene-id',job['id'],'--catalog-only',
                '--base-dataset',str(args.base/'parc_native'),'--output',str(output),
                '--drive-config',str(args.base/'drive-auth/rclone.conf'),'--local-only']
            proc = subprocess.Popen(command,cwd=code,env=cpu_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            pending = (proc,log,job,output)
            state['admission_scene'] = job['id']
            break

    def collect(collection,jobs,legacy=False):
        nonlocal children
        from scripts.run_mixed_control_workflow import available_devices
        while True:
            try:
                available_devices(cards)
                break
            except RuntimeError:
                record_progress('waiting_for_two_free_gpus');time.sleep(30)
        children = []
        for role,card in enumerate(cards):
            command = [str(python),'-u',str(work/'scripts/run_parc_diverse_collection.py'),
                '--work',str(work),'--output',str(collection),'--asset-bundle',str(runtime/'inputs_v1'),
                '--gpu',str(role),'--device-index',str(card)]
            if legacy:
                command += ['--job-ids',*[job['id'] for job in jobs]]
            log = (collection/f'campaign_worker_{role}.log').open('a')
            child = subprocess.Popen(command,cwd=work,env=worker_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((child,log))
        state.update(collection=str(collection),worker_pids=[child.pid for child,_ in children])
        while any(child.poll() is None for child,_ in children) or pending or any(job['id'] not in admitted_scenes for job in jobs):
            for child,_ in children:
                if child.poll() not in (None,0):
                    raise RuntimeError('Collection worker exited: '+str(child.returncode))
            poll_admission(collection,jobs)
            if all(child.poll() is not None for child,_ in children):
                missing = [job['id'] for job in jobs if not (collection/job['id']/'receipt.json').exists()]
                if missing:
                    raise RuntimeError('Collection completed without receipts: '+str(missing))
            record_progress('generating_and_admitting')
            time.sleep(15)
        for child,log in children:
            log.close()
        children = []

    def collect_elastic():
        from scripts.parc_collection_scheduler import ready_jobs, idle_cards, process_matches, worker_exit_action
        plans, next_indices = [], {family:0 for family in FAMILIES}
        for path in sorted(args.output.glob('wave_*/plan/plan.json')):
            plan = read(path)
            plans.append((path.parent.parent,plan))
            next_indices = dict(plan['next_indices'])
        active = {}
        if args.adopt_workers:
            for worker in read(args.adopt_workers)['workers']:
                card = worker['card']
                if card not in cards or card in active:
                    raise ValueError('Invalid adopted GPU allocation')
                collection = Path(worker['collection'])
                matches = [plan for directory,plan in plans if directory == collection]
                if len(matches) != 1 or sha256(collection/'plan/plan.json') != worker['plan_sha256']:
                    raise ValueError('Adopted worker plan changed')
                known = {job['id'] for job in matches[0]['jobs']}
                if not set(worker['job_ids']).issubset(known):
                    raise ValueError('Adopted worker references unknown scenes')
                active[card] = dict(worker, adopted=True)
        empty_waves = set()
        while True:
            for card,worker in list(active.items()):
                if worker['adopted']:
                    running = process_matches(worker['identity'])
                    code_value = None if running else 0
                else:
                    code_value = worker['process'].poll()
                if code_value is not None:
                    collection = Path(worker['collection'])
                    missing = [name for name in worker['job_ids'] if not (collection/name/'receipt.json').exists()]
                    action = worker_exit_action(code_value,not missing,worker['adopted'])
                    if action == 'failed':
                        raise RuntimeError('Worker stopped without complete receipts: '+str(missing))
                    if action == 'retry_when_idle':
                        state['gpu_claim_retries'] = state.get('gpu_claim_retries',0)+1
                    if not worker['adopted']:
                        worker['log'].close()
                        children.remove((worker['process'],worker['log']))
                    del active[card]
            for collection,plan in plans:
                poll_admission(collection,plan['jobs'])
            coverage = summarize(rows)
            reserved = {name for worker in active.values() for name in worker['job_ids']}
            queue = ready_jobs(plans,reserved,admitted_scenes)
            unprocessed = any((directory/job['id']/'receipt.json').exists() and job['id'] not in admitted_scenes
                              for directory,plan in plans for job in plan['jobs'])
            if coverage['target_met'] and not active and not pending and not unprocessed:
                return
            # Refill before GPUs drain, without waiting for the slowest whole wave.
            if not coverage['target_met'] and len(queue) < len(cards) and len(plans) < args.max_waves:
                if shutil.disk_usage(args.base).free < 100*1024**3:
                    raise RuntimeError('Less than 100 GiB free; stop dispatching new scenes')
                collection = args.output/f'wave_{len(plans):03d}'
                plan = build_wave(collection,next_indices,coverage)
                plans.append((collection,plan))
                queue = ready_jobs(plans,reserved,admitted_scenes)
            gpu_csv = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                               '--format=csv,noheader,nounits'],text=True)
            idle = idle_cards(gpu_csv,cards,active)
            if not coverage['target_met']:
                for card,(collection,job) in zip(idle,queue):
                    command = [str(python),'-u',str(work/'scripts/run_parc_diverse_collection.py'),
                        '--work',str(work),'--output',str(collection),'--asset-bundle',str(runtime/'inputs_v1'),
                        '--device-index',str(card),'--worker-name',f'gpu{card}',
                        '--explicit-jobs','--job-ids',job['id']]
                    log = (collection/f'campaign_gpu{card}.log').open('a')
                    process = subprocess.Popen(command,cwd=work,env=worker_env,stdout=log,
                                               stderr=subprocess.STDOUT,start_new_session=True)
                    children.append((process,log))
                    active[card] = {'adopted':False,'process':process,'log':log,'collection':str(collection),
                                    'job_ids':[job['id']],'state_file':f'worker_gpu{card}.json'}
            for directory,plan in plans:
                ids = {job['id'] for job in plan['jobs']}
                if ids.issubset(admitted_scenes) and not any(row.get('scene_id') in ids and
                        row.get('disposition')=='admitted' for row in rows):
                    empty_waves.add(directory.name)
            if any(all(f'wave_{i:03d}' in empty_waves for i in range(start,start+3))
                   for start in range(max(0,len(plans)-2))):
                raise RuntimeError('Three consecutive completed waves admitted no motion')
            if not queue and not active and not pending and not unprocessed and len(plans)>=args.max_waves:
                raise RuntimeError('Wave limit reached before duration/distribution target')
            state.update(active_workers=[{'card':card,'pid':worker['identity']['pid'] if worker['adopted']
                    else worker['process'].pid,'adopted':worker['adopted'],'collection':worker['collection'],
                    'job_ids':worker['job_ids'],'state_file':worker['state_file']} for card,worker in active.items()],
                worker_pids=[worker['identity']['pid'] if worker['adopted'] else worker['process'].pid for worker in active.values()],
                planned_waves=len(plans),queued_scenes=len(ready_jobs(plans,
                    {name for worker in active.values() for name in worker['job_ids']},admitted_scenes)),
                collection=str(plans[-1][0]) if plans else None,
                occupancy_snapshot=gpu_csv.strip().splitlines(),
                scheduling_policy='One scene worker per free GPU; running legacy queues reserved without interruption')
            record_progress('draining_active_scenes' if coverage['target_met'] else 'generating_and_admitting')
            time.sleep(15)

    try:
        from motionbricks.data.lafan_soma_retarget import load_gmr_core
        load_gmr_core(code/'out/vendor/GMR')
        record_progress('waiting_for_preserved_stages')
        legacy_jobs = ([job for job in read(args.legacy_collection/'plan/plan.json')['jobs']
                        if job['id'] in args.legacy_scenes] if args.legacy_collection else [])
        def admit_while_waiting():
            if legacy_jobs:
                poll_admission(args.legacy_collection,legacy_jobs)
            record_progress('finishing_preserved_stages_and_admitting')
        wait_adopted_children(args.adopted_children,admit_while_waiting)
        if args.legacy_collection and args.legacy_scenes:
            jobs = legacy_jobs
            if any(job['id'] not in admitted_scenes for job in jobs):
                collect(args.legacy_collection,jobs,True)
        if args.elastic:
            collect_elastic()
        else:
            next_indices = {family:0 for family in FAMILIES}
            stagnant = 0
            for wave in range(args.max_waves):
                coverage = record_progress('planning_next_wave')
                if coverage['target_met']:
                    break
                collection = args.output/f'wave_{wave:03d}'
                path = collection/'plan/plan.json'
                if path.exists():
                    plan = read(path)
                    next_indices = dict(plan['next_indices'])
                else:
                    if shutil.disk_usage(args.base).free < 100*1024**3:
                        raise RuntimeError('Less than 100 GiB free; preserve existing data and stop generation')
                    plan = build_wave(collection,next_indices,coverage)
                if all(job['id'] in admitted_scenes for job in plan['jobs']):
                    continue
                before = coverage['unique_admitted_clips']
                collect(collection,plan['jobs'])
                current = record_progress('wave_complete')
                stagnant = stagnant+1 if current['unique_admitted_clips']==before else 0
                if stagnant >= 3:
                    raise RuntimeError('Three waves admitted no motion; inspect failures before spending more GPU time')
        coverage = record_progress('checking_duration_target')
        if not coverage['target_met']:
            raise RuntimeError('Wave limit reached before duration/distribution target; training remains gated')
        from scripts.prepare_parc_only_catalog import bind_geometry_components
        from scripts.build_parc_only_dataset import build_dataset
        binding = bind_geometry_components(rows)
        save(ledger,{'schema':'parc.only_teacher_catalog.v1','rows':rows,'processed_scenes':sorted(admitted_scenes),
                     'quality_approved':False,'geometry_split_binding':binding})
        native = args.output/'parc_native'
        record_progress('building_native_dataset')
        build_dataset(ledger,native,incremental_base=args.base/'parc_native')
        # Reconcile the gate against the exact native records the trainer will receive.
        records = {row['source_id']:row for row in read(native/'accepted.json')}
        for row in rows:
            if row.get('disposition')=='admitted':
                actual = records[row['source_id']]
                if abs(actual['duration_seconds']-row['admitted_seconds'])>1e-6 or actual['split']!=row['split']:
                    raise ValueError('Native training records disagree with admission duration or split')
        receipt = {'dataset':str(native),'dataset_manifest_sha256':sha256(native/'native_manifest.json'),
                   'campaign_policy':POLICY,'campaign_coverage':summarize(rows),
                   'catalog_sha256':sha256(ledger),'status':'dataset_ready_local','training_approved':False}
        save(args.output/'dataset_ready.json',receipt)
        record_progress('dataset_ready_local')
    except Exception as error:
        for child,log in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
        if pending and pending[0].poll() is None:
            os.killpg(pending[0].pid,signal.SIGTERM)
        state.update(status='failed',error=type(error).__name__+': '+str(error),updated_at=time.time())
        save(state_path,state)
        raise


if __name__=='__main__':
    main()
