"""Prepare an expanded PARC-only corpus, train native MB, then evaluate it."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_parc_release_subset import OFFICIAL_ARCHIVE_SIZE
from scripts.run_native_mb_workflow import save_state
from scripts.train_native_mb import sha256, verify_dataset, verify_training_source_policy


ROOT = Path(__file__).resolve().parents[1]


def preparation_commands(args):
    output = args.output.resolve()
    if getattr(args, 'prepared_dataset', None) is not None:
        return []
    if getattr(args, 'prepared_catalog', None) is not None:
        return [('dataset', [sys.executable, '-u', 'scripts/build_parc_only_dataset.py',
            '--catalog', str(args.prepared_catalog.resolve()), '--output', str(output / 'dataset')])]
    return [
        ('extract', [sys.executable, '-u', 'scripts/prepare_parc_release_subset.py',
            '--archive', str(args.archive.resolve()), '--archive-index', str(args.archive_index.resolve()),
            '--release-audit', str(args.release_audit.resolve()),
            '--groups-per-family', str(args.groups_per_family), '--output', str(output / 'official')]),
        ('retarget', [sys.executable, '-u', 'scripts/prepare_parc_only_catalog.py',
            '--official-audit', str(output / 'official/inspection.json'),
            '--generated-export', *[str(path.resolve()) for path in args.generated_export],
            '--output', str(output / 'catalog'), '--prepare-retargets']),
        ('dataset', [sys.executable, '-u', 'scripts/build_parc_only_dataset.py',
            '--catalog', str(output / 'catalog/catalog.json'), '--output', str(output / 'dataset')]),
    ]


def prepared_catalog_contract(args):
    path = getattr(args, 'prepared_catalog', None)
    if path is None:
        return None
    path = path.resolve()
    expected = getattr(args, 'prepared_catalog_sha256', None)
    if not path.is_file() or expected != sha256(path):
        raise ValueError('Prepared catalog is missing or its pinned SHA256 changed')
    payload = json.loads(path.read_text())
    if payload.get('schema') != 'parc.only_teacher_catalog.v1':
        raise ValueError('Prepared catalog has an unsupported schema')
    return {'path': str(path), 'sha256': expected,
            'reuse': 'completed extraction/retargeting only; all source artifacts reverified by builder'}


def prepared_dataset_contract(args):
    directory = getattr(args, 'prepared_dataset', None)
    if directory is None:
        return None
    directory = directory.resolve()
    manifest = directory / 'native_manifest.json'
    expected = getattr(args, 'prepared_dataset_sha256', None)
    if not manifest.is_file() or expected != sha256(manifest):
        raise ValueError('Prepared dataset manifest is missing or its pinned SHA256 changed')
    payload = json.loads(manifest.read_text())
    if payload.get('schema') != 'parc-only-native-dataset.v1':
        raise ValueError('Prepared dataset has an unsupported schema')
    return {'path': str(directory), 'manifest_sha256': expected,
            'reuse': 'completed immutable PARC dataset; all artifact hashes and source policy reverified before training'}


def corpus_budget_readiness(records, *, min_official_groups=64, min_families=6,
                            min_train_frames=20000, min_generated_sources=3):
    """Resource guard against spending a long budget on a tiny engineering cohort."""
    train = [row for row in records if row['split'] == 'train']
    origins = Counter(row['origin'] for row in train)
    if set(origins) != {'parc_official', 'parc_generated'}:
        raise ValueError('Training requires both and only official/generated PARC origins')
    if origins['parc_generated'] < min_generated_sources:
        raise ValueError('Too few measured generated PARC sources for the declared training budget')
    official = [row for row in records if row['origin'] == 'parc_official']
    groups = {row['source_group'] for row in official}
    families = set()
    for row in official:
        member = row['source_manifest_identity'].rsplit(':', 1)[-1].split('/')
        if len(member) < 3:
            raise ValueError('Official identity is missing its release family')
        families.add(member[1])
    frames = sum(row['frames_30hz'] for row in train)
    if len(groups) < min_official_groups or len(families) < min_families or frames < min_train_frames:
        raise ValueError(f'Expanded corpus too small for declared long training budget: '
            f'{len(groups)} official groups, {len(families)} families, {frames} train frames')
    if not all(any(row['split'] == split and row['frames_30hz'] >= 65 for row in records)
               for split in ('validation', 'test')):
        raise ValueError('Real validation and test windows are required')
    return {'train_origin_counts': dict(origins), 'official_source_groups': len(groups),
        'official_release_families': sorted(families), 'train_frames': frames,
        'train_seconds': frames / 30,
        'minimums': {'official_groups': min_official_groups, 'release_families': min_families,
                     'train_frames': min_train_frames, 'generated_sources': min_generated_sources},
        'meaning': 'Declared resource guard, not a quality or generalization certificate'}


def execute(command, log, state, state_path, phase, deadline):
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'out/vendor/kimodo')])
    environment['MPLBACKEND'] = 'Agg'
    environment['OMP_NUM_THREADS'] = '1'
    environment['OPENBLAS_NUM_THREADS'] = '1'
    # The verified repository copy lives on a NAS mount with remapped ownership.
    gitconfig = state_path.parent / 'gitconfig'
    if not gitconfig.exists():
        subprocess.run(['git', 'config', '--file', str(gitconfig), '--add',
                        'safe.directory', str(ROOT / 'out/vendor/GMR')], check=True)
    environment['GIT_CONFIG_GLOBAL'] = str(gitconfig)
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(phase=phase, child_pid=child.pid, command=command, log=str(log))
        save_state(state_path, state)
        try:
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Expanded PARC workflow time limit exceeded')
                time.sleep(5)
            if child.returncode:
                raise RuntimeError(f'{phase} failed with exit {child.returncode}; see {log}')
        finally:
            if child.poll() is None:
                # Only this workflow's own child group is ever stopped.
                if os.name == 'posix':
                    os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                    child.wait()
        state.pop('child_pid', None)


def run(args):
    prepared = prepared_catalog_contract(args)
    prepared_dataset = prepared_dataset_contract(args)
    output = args.output.resolve()
    archive = args.archive.resolve() if prepared is None and prepared_dataset is None else None
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / 'state.json'
    state = {'status': 'running', 'phase': 'reusing_prepared_dataset' if prepared_dataset else 'reusing_prepared_catalog' if prepared else 'waiting_for_official_archive', 'pid': os.getpid(),
        'official_archive': str(archive) if archive else None, 'prepared_catalog': prepared,
        'prepared_dataset': prepared_dataset, 'steps_per_stage': args.steps,
        'source_policy': 'only parc_official and parc_generated; fresh weights and PARC-only stats',
        'quality_approved': False, 'skill_approved': False, 'started_at_unix': time.time()}
    deadline = time.monotonic() + args.wait_hours * 3600
    try:
        while prepared is None and prepared_dataset is None and not archive.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError('Official PARC archive did not arrive within the workflow time limit')
            part = Path(str(archive) + '.part')
            state.update(download_bytes=part.stat().st_size if part.is_file() else 0,
                         expected_archive_bytes=OFFICIAL_ARCHIVE_SIZE, updated_at_unix=time.time())
            save_state(state_path, state)
            time.sleep(args.poll_seconds)
        # Extraction verifies pinned full size, SHA256, ZIP index and member CRCs.
        for phase, command in preparation_commands(args):
            execute(command, output / (phase + '.log'), state, state_path, phase, deadline)
        if prepared is not None and sha256(Path(prepared['path'])) != prepared['sha256']:
            raise RuntimeError('Prepared catalog changed during dataset construction')
        dataset = Path(prepared_dataset['path']) if prepared_dataset else output / 'dataset'
        if prepared_dataset is not None:
            prepared_dataset_contract(args)
        manifest = json.loads((dataset / 'native_manifest.json').read_text())
        verify_dataset(dataset, manifest)
        policy = verify_training_source_policy(dataset, manifest, parc_only=True)
        records = json.loads((dataset / 'accepted.json').read_text())
        state.update(corpus=corpus_budget_readiness(records), source_accounting=policy,
                     dataset_manifest_sha256=sha256(dataset / 'native_manifest.json'))
        command = [sys.executable, '-u', 'scripts/run_parc_only_training.py',
            '--dataset', str(dataset), '--output', str(output / 'training'),
            '--steps', str(args.steps), '--gpus', *map(str, args.gpus),
            '--wait-hours', str(max(.01, (deadline - time.monotonic()) / 3600))]
        execute(command, output / 'training.log', state, state_path, 'training', deadline)
        command = [sys.executable, '-u', 'scripts/evaluate_parc_native_terrain.py',
            '--workflow', str(output / 'training/state.json'), '--output', str(output / 'review'), '--render']
        execute(command, output / 'review.log', state, state_path, 'offline_terrain_review', deadline)
        state.update(status='trained_offline_review_finished_not_live_approved',
                     finished_at_unix=time.time())
    except BaseException as error:
        state.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        save_state(state_path, state)
    return state


def dispatch(args):
    receipt = args.receipt.resolve()
    if receipt.exists() or args.output.exists():
        raise FileExistsError('Use a new receipt and output for the immutable workflow')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(Path(__file__).resolve()),
               *[item for item in sys.argv[1:] if item != '--dispatch']]
    log = receipt.with_suffix('.log')
    with log.open('xb') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    result = {'status': 'dispatched_not_completed', 'pid': process.pid, 'command': command,
              'workflow_state': str(args.output.resolve() / 'state.json'), 'log': str(log),
              'quality_approved': False, 'started_at_unix': time.time()}
    receipt.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--archive-index', type=Path)
    parser.add_argument('--release-audit', type=Path)
    parser.add_argument('--generated-export', type=Path, nargs='+')
    parser.add_argument('--prepared-catalog', type=Path)
    parser.add_argument('--prepared-catalog-sha256')
    parser.add_argument('--prepared-dataset', type=Path)
    parser.add_argument('--prepared-dataset-sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--groups-per-family', type=int, default=32)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--wait-hours', type=float, default=72)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--dispatch', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.prepared_dataset is not None:
        if args.prepared_dataset_sha256 is None or any(value is not None for value in (
                args.prepared_catalog, args.prepared_catalog_sha256, args.archive,
                args.archive_index, args.release_audit, args.generated_export)):
            parser.error('Prepared dataset requires its manifest SHA256 and cannot be mixed with preparation inputs')
    elif args.prepared_dataset_sha256 is not None:
        parser.error('--prepared-dataset-sha256 requires --prepared-dataset')
    elif args.prepared_catalog is not None:
        if args.prepared_catalog_sha256 is None or any(
                value is not None for value in (args.archive, args.archive_index, args.release_audit, args.generated_export)):
            parser.error('Prepared catalog requires its SHA256 and cannot be mixed with extraction inputs')
    elif args.prepared_catalog_sha256 is not None or any(
            value is None for value in (args.archive, args.archive_index, args.release_audit, args.generated_export)):
        parser.error('Provide a pinned prepared catalog, or all archive/audit/generated-export inputs')
    if args.dispatch and args.receipt is None:
        parser.error('--dispatch requires --receipt')
    if (not 1 <= args.steps <= 2000000 or args.groups_per_family < 1 or args.wait_hours <= 0
            or args.poll_seconds < 1 or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus)):
        parser.error('Invalid budget, GPU indices or polling interval')
    if args.dispatch and os.name != 'posix':
        parser.error('Durable server dispatch requires POSIX process groups')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Workflow stopped')))
    print(json.dumps(dispatch(args) if args.dispatch else run(args)), flush=True)


if __name__ == '__main__':
    main()
