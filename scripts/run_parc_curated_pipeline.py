"""Finite PARC candidate correction -> physical recording -> export workflow.

Runs only on a GPU observed idle twice and retains the advisory lock through
all child stages. No generated motion is admitted to MB training by this job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def json_write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(str(temporary), str(path))


def prepare_reference_jobs(catalog, summary, old_bundle, output):
    rows = {r['candidate_key']: r for r in json.loads(catalog.read_text())['candidates']}
    jobs = []
    for entry in json.loads(summary.read_text())['showcase']:
        if entry['name'] not in ('terraces', 'platform_2', 'three_stair_flights'):
            continue
        row = rows[entry['candidate_key']]
        score = row['report_entry']
        jobs.append({'name': entry['name'], 'candidate_key': row['candidate_key'],
            'sha256': row['reported_source_sha256'], 'source_group': row['candidate_key'].rsplit('/', 1)[0],
            'contact_loss': score['weighted_contact_score'],
            'penetration_loss': score['weighted_penetration_score'],
            'total_loss': score['selection_score'], 'completed': score['endpoint_reached']})
    if len(jobs) != 3:
        raise ValueError('Expected the three fixed regression cases')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    json_write(output, {'jobs': jobs, 'old_bundle': str(old_bundle)})


def corrected_bundle(old_bundle, corrections, output, reference_stage='final'):
    """New immutable input bundle; never replace the raw reference bundle."""
    report = json.loads((corrections / 'correction_report.json').read_text())
    if report['status'] != 'correction_finished_physics_pending' or not report['cases']:
        raise ValueError('No completed corrections')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('PARC', 'tracker'):
        shutil.copytree(str(old_bundle / name), str(output / name), copy_function=shutil.copyfile,
                        ignore=shutil.ignore_patterns('__pycache__', 'user_config.yaml'))
    (output / 'motions').mkdir()
    motions, representatives = {}, []
    for case in report['cases']:
        if case['status'] != 'kinematic_correction_complete_physics_pending':
            raise ValueError('Incomplete optimization case')
        source = Path(case['stages'][reference_stage]['motion_file'])
        if digest(source) != case['stages'][reference_stage]['sha256']:
            raise ValueError('Correction source changed')
        relative = 'motions/' + case['case'] + '.pkl'
        shutil.copyfile(str(source), str(output / relative))
        motions[relative] = {'candidate_key': case['case'], 'sha256': digest(source),
            'source_frames': case['stages'][reference_stage]['metrics']['frames'],
            'reference_stage': reference_stage,
            'original_candidate_sha256': case['source_sha256'], 'source_group': case['source_group']}
        if 'intent' in case and reference_stage == 'final':
            intent = Path(case['intent'])
            if digest(intent) != case['intent_sha256']:
                raise ValueError('Aligned intent changed')
            rel_intent = 'motions/' + case['case'] + '_intent.npz'
            shutil.copyfile(str(intent), str(output / rel_intent))
            motions[relative].update(intent_path=rel_intent, intent_sha256=digest(intent))
        representatives.append({'name': case['case'], 'path': relative, 'candidate_key': case['case']})
    files = [{'path': p.relative_to(output).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in sorted(output.rglob('*')) if p.is_file()]
    manifest = {'schema_version': 1, 'motions': motions, 'representatives': representatives,
                'files': files, 'source_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
    json_write(output / '_bundle.json', manifest)
    return manifest


def select_generated_jobs(generation_path, prefix, config):
    from correct_parc_candidates import selection

    report = json.loads(generation_path.read_text())
    if report['status'] != 'kinematic_candidates_generated_physics_pending':
        raise ValueError('Generation not complete')
    candidates = []
    for row in report['candidates']:
        job = {'name': prefix + '_c%03d' % row['candidate_id'], 'motion': row['motion'],
               'sha256': row['sha256'], 'completed': row['endpoint_reached'],
               'contact_loss': row['weighted_contact_score'],
               'penetration_loss': row['weighted_penetration_score'], 'total_loss': row['selection_score'],
               'source_group': prefix, 'generation_sha256': digest(generation_path),
               'raw_route_covered': row['geometric_route_covered'],
               'intent': row['intent'], 'intent_sha256': row['intent_sha256']}
        if selection(job, config) and job['raw_route_covered']:
            candidates.append(job)
    candidates.sort(key=lambda row: row['total_loss'])
    if len(candidates) < config['mdm_path']['top_k']:
        raise ValueError('Insufficient candidates passing released filter; do not relax thresholds')
    return candidates[:config['mdm_path']['top_k']]


def attach_export_intent(bundle, export):
    import numpy as np
    source = json.loads((bundle / '_bundle.json').read_text())
    sources = {row['candidate_key']: row for row in source['motions'].values()}
    manifest_path = export / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for row in manifest['recordings']:
        original = sources[row['candidate_key']]
        row['source_group'] = original['source_group']
        row['original_candidate_sha256'] = original['original_candidate_sha256']
        if 'intent_path' not in original or not row['whole_motion']:
            row['intent_status'] = 'unavailable_no_training_admission'
            continue
        source_path = bundle / original['intent_path']
        if digest(source_path) != original['intent_sha256']:
            raise ValueError('Physics input intent changed')
        with np.load(source_path, allow_pickle=False) as data:
            if data['command_world_xy'].shape != (row['frames'], 2) or float(data['fps']) != row['fps']:
                raise ValueError('Full physics recording has a different intent time axis')
        relative = 'recordings/' + row['candidate_key'] + '_intent.npz'
        shutil.copyfile(str(source_path), str(export / relative))
        row.update(intent_status='captured_before_generation_frame_aligned', intent_path=relative,
                   intent_sha256=digest(export / relative),
                   intent_coordinates='original PARC world XY; physics localization translation leaves directions unchanged')
    json_write(manifest_path, manifest)


def archive_failed_output(path, resume):
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        path.rename(path.with_name('failed_' + path.name + '_' + str(time.time_ns())))


def run(args):
    import fcntl

    host = Path.home() / 'projects/motionbricks-parc-physics-20261009'
    work = host / 'curated_v1'
    old = Path('${PRIVATE_STORAGE}/motionbricks-parc-physics-20261009/inputs_v1')
    output = Path('${PRIVATE_STORAGE}/motionbricks-parc-physics-20261009') / args.run_id
    if not args.run_id.replace('_', '').isalnum():
        raise ValueError('Noncanonical run id')
    lock_path = host / ('gpu%d.lock' % args.gpu)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid',
                                        '--format=csv,noheader'], text=True).strip()
        for _ in range(2):
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                  '--format=csv,noheader'], text=True)
            free, util = map(int, subprocess.check_output(['nvidia-smi', '-i', gpu,
                '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).split(','))
            if gpu in processes or free < 20000 or util > 10:
                raise RuntimeError('Requested GPU is not idle')
            time.sleep(2)
        output.mkdir(parents=True, exist_ok=args.resume)
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
            PATH=str(host / 'env/bin') + ':' + env.get('PATH', ''),
            LD_LIBRARY_PATH=str(host / 'env/lib') + ':' + env.get('LD_LIBRARY_PATH', ''),
            TORCH_EXTENSIONS_DIR=str(host / 'torch_extensions'), MAX_JOBS='4',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_MODE='disabled', WANDB_SILENT='true', MPLBACKEND='Agg')
        python = str(host / 'env/bin/python')
        status = {'status': 'running', 'gpu_uuid': gpu, 'stages': [], 'training_approved': False}
        started = time.time()

        def stage(name, command):
            status['active_stage'] = name
            json_write(output / 'pipeline.json', status)
            start = time.time()
            log_name = name + ('_resume_' + str(time.time_ns()) if args.resume else '') + '.log'
            with (output / log_name).open('x') as stream:
                result = subprocess.run(command, env=env, cwd=str(work / 'out/vendor/PARC'),
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=14400)
            status['stages'].append({'name': name, 'exit_code': result.returncode, 'elapsed_seconds': time.time() - start})
            if result.returncode:
                raise RuntimeError(name + ' failed; see its log')

        try:
            if args.generated_reports:
                import yaml
                config = yaml.safe_load((work / 'out/vendor/PARC/data/configs/parc_2_kin_gen_default.yaml').read_text())
                jobs = {'jobs': [], 'purpose': 'completed mixed generation; not training approved'}
                for generation in args.generated_reports:
                    jobs['jobs'].extend(select_generated_jobs(generation, generation.parent.name, config))
            elif args.mixed:
                import yaml
                config = yaml.safe_load((work / 'out/vendor/PARC/data/configs/parc_2_kin_gen_default.yaml').read_text())
                jobs = {'jobs': [], 'purpose': 'mixed terrain pilot; not training approved'}
                for variant in args.variants:
                    seed = 20261090 + variant
                    name = 'mixed_v%d_s%d' % (variant, seed)
                    generation = output / name
                    finished = generation / 'generation.json'
                    if args.resume and finished.exists() and json.loads(finished.read_text())['status'] == 'kinematic_candidates_generated_physics_pending':
                        recorded = json.loads(finished.read_text())
                        if recorded['seed'] != seed or recorded['batch_size'] != config['mdm_path']['mdm_batch_size']:
                            raise ValueError('Cannot resume a different generation contract')
                        for row in recorded['candidates']:
                            if digest(Path(row['motion'])) != row['sha256'] or digest(Path(row['intent'])) != row['intent_sha256']:
                                raise ValueError('Completed generation changed')
                    else:
                        archive_failed_output(generation, args.resume)
                        stage(name, [python, '-u', str(work / 'scripts/generate_parc_long_stairs.py'),
                        '--repo', str(work / 'out/vendor/PARC'), '--checkpoint',
                        str(work / 'out/parc_legacy_models_20261008/demo_iter3_verified.ckpt'),
                        '--output', str(generation), '--kind', 'mixed', '--variant', str(variant),
                        '--seed', str(seed), '--batch-size', str(config['mdm_path']['mdm_batch_size']),
                        '--max-seconds', '60', '--score-frame-chunk', '4'])
                    jobs['jobs'].extend(select_generated_jobs(generation / 'generation.json', name, config))
            else:
                jobs = json.loads(args.jobs.read_text())
                sources = {v['candidate_key']: (k, v) for k, v in json.loads((old / '_bundle.json').read_text())['motions'].items()}
                for row in jobs['jobs']:
                    path, metadata = sources[row['candidate_key']]
                    if row['sha256'] != metadata['sha256']:
                        raise ValueError('Fixed case identity mismatch')
                    row['motion'] = str(old / path)
            json_write(output / 'jobs.json', jobs)
            corrections = args.correction_source or (output / 'corrected')
            correction_command = [python, '-u', str(work / 'scripts/correct_parc_candidates.py'),
                '--repo', str(work / 'out/vendor/PARC'), '--jobs', str(output / 'jobs.json'),
                '--output', str(corrections), '--device', 'cuda:0']
            if args.resume:
                correction_command.append('--resume')
            if args.correction_source:
                existing = json.loads((corrections / 'correction_report.json').read_text())
                expected = {row['name']: row['sha256'] for row in jobs['jobs']}
                actual = {row['case']: row['source_sha256'] for row in existing['cases']}
                if existing['status'] != 'correction_finished_physics_pending' or expected != actual:
                    raise ValueError('Existing corrections do not match requested cases')
            else:
                stage('official_correction', correction_command)
            bundle = output / 'inputs'
            # Bundle is a derived snapshot; keep old attempts rather than overwrite them.
            archive_failed_output(bundle, args.resume)
            corrected_bundle(old, corrections, bundle, args.reference_stage)
            physics_report = output / 'physics/report.json'
            physics_complete = (args.resume and physics_report.exists() and
                json.loads(physics_report.read_text()).get('status') == 'official_recording_finished')
            if not physics_complete:
                archive_failed_output(output / 'physics', args.resume)
                stage('official_physics', [python, '-u', str(host / 'run_parc_physics_record.py'),
                    '--bundle', str(bundle), '--output', str(output / 'physics'), '--selection', 'all'])
            archive_failed_output(output / 'export', args.resume)
            stage('official_export', [python, '-u', str(host / 'export_parc_physics_recordings.py'),
                '--bundle', str(bundle), '--run', str(output / 'physics'), '--output', str(output / 'export')])
            attach_export_intent(bundle, output / 'export')
            physical = json.loads(physics_report.read_text())
            expected = {row['name'] for row in jobs['jobs']}
            complete = {row['candidate_key'] for row in physical['saved'] if row['whole_motion']}
            status['missing_full_recordings'] = sorted(expected - complete)
            status['status'] = ('completed_quality_review_pending' if expected == complete else
                                'partial_physics_quality_review_pending')
        except Exception as error:
            status.update(status='failed', error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            status['elapsed_seconds'] = time.time() - started
            json_write(output / 'pipeline.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--catalog', type=Path, required=True)
    prepare.add_argument('--summary', type=Path, required=True)
    prepare.add_argument('--old-bundle', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    remote = sub.add_parser('run')
    remote.add_argument('--jobs', type=Path)
    remote.add_argument('--mixed', action='store_true')
    remote.add_argument('--generated-reports', type=Path, nargs='+')
    remote.add_argument('--variants', type=int, nargs='+', choices=(0, 1, 2), default=[0, 1, 2])
    remote.add_argument('--resume', action='store_true')
    remote.add_argument('--correction-source', type=Path)
    remote.add_argument('--reference-stage', choices=('raw', 'optimized', 'final'), default='final')
    remote.add_argument('--run-id', required=True)
    remote.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_reference_jobs(args.catalog, args.summary, args.old_bundle, args.output)
    else:
        if sum((args.mixed, args.jobs is not None, bool(args.generated_reports))) != 1:
            parser.error('Choose exactly one of --jobs, --mixed, and --generated-reports')
        run(args)
