"""Versioned bounded online-contract adaptation; preserve the running baseline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_native_mb import sha256
from motionbricks.data.native_mb_online_conditions import ONLINE_PROTOCOL
from motionbricks.data.native_mb_paper_protocol import PAPER_STEPS


def completed_checkpoint(run):
    run = Path(run)
    contract_path, result_path = run / 'run_contract.json', run / 'training_result.json'
    checkpoint = run / 'checkpoints/final.ckpt'
    if not all(path.is_file() for path in (contract_path, result_path, checkpoint)):
        return False
    contract = json.loads(contract_path.read_text())
    result = json.loads(result_path.read_text())
    checkpoint_sha256 = result.get('final_checkpoint_sha256')
    return (contract.get('status') in {'completed', 'trained_not_reviewed'}
            and isinstance(checkpoint_sha256, str) and len(checkpoint_sha256) == 64
            and sha256(checkpoint) == checkpoint_sha256)


def validate_step_budget(steps, *, continue_online):
    maximum = 10000 if continue_online else 1000
    if type(steps) is not int or not 1 <= steps <= maximum:
        if continue_online:
            raise ValueError('Online continuation accepts 1..10000 total target updates')
        raise ValueError('This preflight accepts 1..1000 updates; longer runs require --continue-online')
    return steps


def commands(args):
    common = [sys.executable, '-u', 'scripts/train_native_mb_control.py',
        '--dataset', str(args.dataset), '--steps', str(args.steps), '--batch-size', '256',
        '--schedule-steps', str(PAPER_STEPS), '--training-profile', 'paper', '--parc-only',
        '--terrain-conditioning', '--online-conditioning', '--skip-review',
        '--gpu', str(args.gpu), '--workers', str(args.workers), '--seed', '7']
    root_run, pose_run = args.output / 'root-online', args.output / 'pose-online'
    continuing = bool(getattr(args, 'continue_online', False))
    if continuing:
        root_init = ['--online-resume-run', str(args.online_root_run)]
        pose_init = ['--online-warmstart-run', str(args.online_pose_run)]
    else:
        root_init = ['--online-warmstart-run', str(args.root_baseline)]
        pose_init = ['--online-warmstart-run', str(args.pose_baseline)]
    result = {
        'root': common + ['--stage', 'root', '--output', str(root_run),
                          *root_init],
        'pose': common + ['--stage', 'pose', '--output', str(pose_run),
                          *pose_init,
                          '--online-root-run', str(root_run), '--tokenizer-run', str(args.tokenizer_run)],
        'review': [sys.executable, '-u', 'scripts/evaluate_parc_online_control.py',
                   '--dataset', str(args.dataset), '--root-run', str(root_run),
                   '--pose-run', str(pose_run), '--output', str(args.output / 'review'),
                   '--device', 'cuda', '--max-windows', '8'],
    }
    if continuing:
        result['review'] += ['--inference-steps', '4']
        result['review_invalid'] = result['review'].copy()
        output_index = result['review_invalid'].index('--output') + 1
        result['review_invalid'][output_index] = str(args.output / 'review-invalid')
        result['review_invalid'] += ['--terrain-mode', 'invalid']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'output', 'tokenizer-run'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--root-baseline', type=Path)
    parser.add_argument('--pose-baseline', type=Path)
    parser.add_argument('--continue-online', action='store_true',
        help='Resume the exact online Root run and weights-only warmstart Pose from its prior online run')
    parser.add_argument('--online-root-run', type=Path)
    parser.add_argument('--online-pose-run', type=Path)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    try:
        validate_step_budget(args.steps, continue_online=args.continue_online)
    except ValueError as exc:
        parser.error(str(exc))
    if args.workers < 0:
        parser.error('Workers cannot be negative')
    if args.continue_online:
        if args.root_baseline or args.pose_baseline:
            parser.error('--continue-online uses --online-root-run and --online-pose-run, not baseline runs')
        if not args.online_root_run or not args.online_pose_run:
            parser.error('--continue-online requires --online-root-run and --online-pose-run')
    elif args.online_root_run or args.online_pose_run:
        parser.error('--online-root-run and --online-pose-run require --continue-online')
    elif not args.root_baseline or not args.pose_baseline:
        parser.error('Initial preflight requires --root-baseline and --pose-baseline')
    path_names = ['dataset', 'output', 'tokenizer_run']
    path_names += (['online_root_run', 'online_pose_run'] if args.continue_online
                   else ['root_baseline', 'pose_baseline'])
    for name in path_names:
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        parser.error('Use a new versioned output directory')
    if not completed_checkpoint(args.tokenizer_run):
        parser.error('Codec source must have final verified run artifacts')
    if args.continue_online:
        if not completed_checkpoint(args.online_root_run) or not completed_checkpoint(args.online_pose_run):
            parser.error('Online continuation sources must have final verified run artifacts')
        root_result = json.loads((args.online_root_run / 'training_result.json').read_text())
        if args.steps <= root_result.get('optimizer_updates', 0) or args.steps > PAPER_STEPS:
            parser.error('--steps must exceed the saved Root updates and stay within the original 2M schedule')
    elif not completed_checkpoint(args.root_baseline):
        parser.error('Root baseline must have final verified run artifacts')
    args.output.mkdir(parents=True)
    state = {'schema': 'parc.online_adaptation.v1', 'status': 'running', 'phase': 'root',
             'pid': os.getpid(), 'protocol': ONLINE_PROTOCOL, 'steps_per_stage': args.steps,
             'continue_online': args.continue_online,
             'root_steps_are_total_target': args.continue_online,
             'dataset_manifest_sha256': sha256(args.dataset / 'native_manifest.json'),
             'code_root': str(Path(__file__).resolve().parents[1]), 'stages': {},
             'baseline_modified': False, 'live_backend_changed': False}
    path = args.output / 'state.json'

    def save():
        state['updated_at_unix'] = time.time()
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)

    def wait_idle_gpu():
        from scripts.train_native_mb import select_gpu
        while True:
            try:
                return select_gpu(args.gpu)
            except RuntimeError:
                state['waiting_for'] = f'idle GPU {args.gpu}; no other process will be stopped'
                save()
                time.sleep(60)

    try:
        save()
        for stage, command in commands(args).items():
            state['phase'] = stage
            if stage == 'pose' and not args.continue_online:
                while not completed_checkpoint(args.pose_baseline):
                    state['waiting_for'] = 'existing baseline Pose final checkpoint; no partial checkpoint imported'
                    save()
                    time.sleep(60)
            state.pop('waiting_for', None)
            gpu = wait_idle_gpu()
            state.pop('waiting_for', None)
            state['stages'][stage] = {'status': 'running', 'command': command, 'gpu_at_launch': gpu}
            save()
            with (args.output / f'{stage}.log').open('w', encoding='utf-8') as stream:
                child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                         stdout=stream, stderr=subprocess.STDOUT)
                state['stages'][stage]['pid'] = child.pid
                save()
                code = child.wait()
            state['stages'][stage].update(status='completed' if code == 0 else 'failed', exit_code=code)
            save()
            if code != 0:
                raise RuntimeError(f'{stage} failed; see {stage}.log')
        state.update(status='completed', phase='reviewed', quality_approved=False)
        save()
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    main()
