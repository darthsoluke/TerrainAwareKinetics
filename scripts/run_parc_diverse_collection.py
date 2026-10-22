"""Resumable official PARC generation, correction and physics per terrain group.

Workers own different GPUs and disjoint scene groups. Outputs remain
unapproved until full recording, paired geometry and SOMA retarget checks pass.
"""

from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_parc_curated_pipeline import digest, json_write, corrected_bundle, attach_export_intent
from build_parc_diverse_course import write_plan
from parc_collection_scheduler import select_jobs


def run(args):
    output, work = args.output.resolve(), args.work.resolve()
    host = work.parent
    output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        write_plan(output/'plan')
        return
    plan = json.loads((output/'plan/plan.json').read_text())
    jobs = select_jobs(plan,args.gpu,args.job_ids,args.explicit_jobs)
    if args.retry_failed:
        jobs.sort(key=lambda job: not (output/job['id']/'corrected/correction_report.json').exists())
    device_index=args.gpu if args.device_index is None else args.device_index
    with (host/f'gpu{device_index}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('GPU lock is occupied; retry when idle',file=sys.stderr)
            raise SystemExit(75)
        line = subprocess.check_output(['nvidia-smi', '-i', str(device_index),
            '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        gpu, memory, util = [part.strip() for part in line.split(',')]
        if int(memory) > 1024 or int(util) > 10:
            print('Requested GPU became busy; retry when idle',file=sys.stderr)
            raise SystemExit(75)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONNOUSERSITE='1',
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            WANDB_MODE='disabled', WANDB_SILENT='true', MPLBACKEND='Agg', MAX_JOBS='4',
            TORCH_EXTENSIONS_DIR=str(host/'torch_extensions'),
            PATH=str(host/'env/bin')+':'+os.environ.get('PATH',''),
            LD_LIBRARY_PATH=str(host/'env/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
        env.pop('PYTHONPATH', None)
        env['PYTHONPATH'] = str(host/'sdk/isaacgym/python')
        repo = work/'out/vendor/PARC'
        old_bundle = args.asset_bundle or Path('${PRIVATE_STORAGE}/motionbricks-parc-physics-20261009/inputs_v1')
        import yaml
        from correct_parc_candidates import selection
        from parc_collection_budget import diverse_selection
        config = yaml.safe_load((repo/'data/configs/parc_2_kin_gen_default.yaml').read_text())
        state = {'status':'running', 'gpu':gpu, 'jobs':[], 'training_approved':False,
                 'plan_sha256':digest(output/'plan/plan.json'),
                 'selection_code_sha256':digest(Path(__file__).with_name('parc_collection_budget.py'))}
        worker_name = args.worker_name or str(args.gpu)
        if not worker_name.replace('_','').isalnum():
            raise ValueError('Worker name must be alphanumeric or underscore')
        state_path = output/f'worker_{worker_name}.json'
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            if previous['plan_sha256'] != state['plan_sha256']:
                raise ValueError('Cannot resume a changed scene plan')
            state['jobs'] = previous['jobs']

        def stage(directory, name, command):
            state.update(active_scene=directory.name, active_stage=name, updated_at=time.time())
            json_write(state_path, state)
            with (directory/(name+'.log')).open('a') as log:
                subprocess.run(command, cwd=repo, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=14400)

        for job in jobs:
            directory = output/job['id']
            directory.mkdir(exist_ok=True)
            receipt_path = directory/'receipt.json'
            if receipt_path.exists():
                previous_status = json.loads(receipt_path.read_text()).get('status')
                if previous_status == 'failed' and args.retry_failed:
                    receipt_path.rename(directory/('receipt.failed.'+str(time.time_ns())+'.json'))
                elif previous_status in ('physics_exported', 'no_candidate_passed', 'failed'):
                    continue
            receipt = dict(job, status='generating', training_approved=False)
            try:
                selected = []
                for attempt in range(plan['max_attempts_per_layout']):
                    generation = directory/f'generation_{attempt}'
                    generation_report = generation/'generation.json'
                    if generation.exists() and (not generation_report.exists() or
                            json.loads(generation_report.read_text())['status']!='kinematic_candidates_generated_physics_pending'):
                        generation.resolve().relative_to(output)
                        generation.rename(directory/('failed_generation_'+str(attempt)+'_'+str(time.time_ns())))
                    if not generation_report.exists():
                        stage(directory, f'generation_{attempt}', [sys.executable,'-u',
                            str(work/'scripts/generate_parc_long_stairs.py'), '--repo',str(repo),
                            '--checkpoint',str(work/'out/parc_legacy_models_20261008/demo_iter3_verified.ckpt'),
                            '--output',str(generation), '--kind','diverse',
                            '--course-spec',str(output/'plan'/job['spec']),
                            '--seed',str(job['motion_seed']+10000*attempt),
                            '--batch-size',str(plan['candidate_batch']), '--max-seconds',str(job.get('max_seconds',60)),
                            '--score-frame-chunk','4',
                            *(['--expected-terrain-sha256',job['terrain_sha256']]
                              if job.get('terrain_sha256') else [])])
                    generated = json.loads(generation_report.read_text())
                    if generated['status'] != 'kinematic_candidates_generated_physics_pending':
                        continue
                    for row in generated['candidates']:
                        candidate = {'name':job['id']+f'_a{attempt}_c{row["candidate_id"]:03d}',
                            'motion':row['motion'], 'sha256':row['sha256'], 'completed':row['endpoint_reached'],
                            'contact_loss':row['weighted_contact_score'], 'penetration_loss':row['weighted_penetration_score'],
                            'total_loss':row['selection_score'], 'source_group':job['id'],
                            'intent':row['intent'], 'intent_sha256':row['intent_sha256']}
                        if selection(candidate, config) and row['geometric_route_covered']:
                            selected.append(candidate)
                    if plan.get('selection_policy') == 'pose_farthest_point_v1':
                        selected = diverse_selection(selected, plan['retained_per_layout'],
                            plan['candidate_pose_min_rms_m'])
                    if len(selected) >= plan['retained_per_layout']:
                        break
                selected.sort(key=lambda row:row['total_loss'])
                selected = selected[:plan['retained_per_layout']]
                if not selected:
                    receipt.update(status='no_candidate_passed', reason='official scores or route coverage')
                else:
                    jobs_file = directory/'jobs.json'
                    if not jobs_file.exists():
                        json_write(jobs_file, {'jobs':selected})
                    elif json.loads(jobs_file.read_text())['jobs'] != selected:
                        raise ValueError('Selected candidates changed during resume')
                    corrected = directory/'corrected'
                    report = corrected/'correction_report.json'
                    if not report.exists() or json.loads(report.read_text())['status'] != 'correction_finished_physics_pending':
                        stage(directory,'correction',[sys.executable,'-u',str(work/'scripts/correct_parc_candidates.py'),
                            '--repo',str(repo),'--jobs',str(jobs_file),'--output',str(corrected),
                            '--device','cuda:0',*(['--resume'] if corrected.exists() else [])])
                    inputs = directory/'inputs'
                    if not inputs.exists():
                        corrected_bundle(old_bundle, corrected, inputs)
                    physics = directory/'physics'
                    if physics.exists() and (not (physics/'report.json').exists() or
                            json.loads((physics/'report.json').read_text())['status']!='official_recording_finished'):
                        physics.resolve().relative_to(output)
                        physics.rename(directory/('failed_physics_'+str(time.time_ns())))
                    if not physics.exists():
                        stage(directory,'physics',[sys.executable,'-u',str(host/'run_parc_physics_record.py'),
                            '--bundle',str(inputs),'--output',str(physics),'--selection','all'])
                    if json.loads((physics/'report.json').read_text())['status'] != 'official_recording_finished':
                        raise ValueError('Physics stage incomplete')
                    export = directory/'export'
                    if export.exists() and not (export/'manifest.json').exists():
                        export.resolve().relative_to(output)
                        export.rename(directory/('failed_export_'+str(time.time_ns())))
                    if not export.exists():
                        stage(directory,'export',[sys.executable,'-u',str(host/'export_parc_physics_recordings.py'),
                            '--bundle',str(inputs),'--run',str(physics),'--output',str(export)])
                    attach_export_intent(inputs, export)
                    manifest_path = export/'manifest.json'
                    manifest = json.loads(manifest_path.read_text())
                    for row in manifest['recordings']:
                        row.update(terrain_family=job['family'], requested_split=job['split'], scene_id=job['id'])
                    json_write(manifest_path, manifest)
                    receipt.update(status='physics_exported', selected=len(selected),
                        full_recordings=sum(row['whole_motion'] for row in manifest['recordings']),
                        export=str(export), manifest_sha256=digest(manifest_path))
            except Exception as error:
                receipt.update(status='failed', error=type(error).__name__+': '+str(error))
            json_write(receipt_path, receipt)
            state['jobs'] = [row for row in state['jobs'] if row['id'] != receipt['id']] + [receipt]
            json_write(state_path,state)
        state.update(status='collection_finished_admission_pending', updated_at=time.time())
        json_write(state_path,state)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--gpu',type=int,choices=(0,1),default=0)
    parser.add_argument('--device-index',type=int,choices=range(8))
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--asset-bundle',type=Path)
    parser.add_argument('--job-ids',nargs='+')
    parser.add_argument('--explicit-jobs',action='store_true')
    parser.add_argument('--worker-name')
    run(parser.parse_args())
