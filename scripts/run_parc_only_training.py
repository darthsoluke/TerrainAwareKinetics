"""Train native MB stages on an immutable PARC-only paired terrain dataset."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_paper_protocol import PAPER_BATCH_PER_GPU, PAPER_STEPS
from scripts.train_native_mb import (
    sha256, select_gpu, verify_dataset, verify_training_source_policy, source_lf_sha256,
)
from scripts.run_native_mb_workflow import launch, save_state, stop_owned


def stage_command(stage, dataset, output, gpu, steps, *, tokenizer_run=None):
    if stage not in ('tokenizer', 'root', 'pose') or (stage == 'pose') != (tokenizer_run is not None):
        raise ValueError('Only Pose loads a completed frozen tokenizer')
    script = 'scripts/train_native_mb.py' if stage == 'tokenizer' else 'scripts/train_native_mb_control.py'
    command = [sys.executable, '-u', script, '--dataset', str(dataset), '--output', str(output),
        '--parc-only', '--training-profile', 'paper', '--schedule-steps', str(PAPER_STEPS),
        '--steps', str(steps), '--batch-size', str(PAPER_BATCH_PER_GPU),
        '--seed', '7', '--workers', '2', '--gpu', str(gpu)]
    if stage != 'tokenizer':
        command += ['--stage', stage, '--terrain-conditioning', '--skip-review']
    if tokenizer_run is not None:
        command += ['--tokenizer-run', str(tokenizer_run)]
    return command


def validate_result(run, stage, dataset_hash, steps, tokenizer_hash=None):
    contract = json.loads((run / 'run_contract.json').read_text())
    result = json.loads((run / 'training_result.json').read_text())
    if (contract.get('stage') != 'native_' + stage or
            contract.get('dataset_manifest_sha256') != dataset_hash or
            contract.get('batch_size') != PAPER_BATCH_PER_GPU or
            contract.get('original_schedule_steps') != PAPER_STEPS or
            contract.get('stop_steps') != steps or contract.get('seed') != 7 or
            not contract.get('parc_only_sources') or
            contract.get('warmstart') is not None or
            contract.get('resume') is not None or
            sha256(run / 'native_config.yaml') != contract.get('config_sha256') or
            result.get('optimizer_updates') != steps or
            result.get('updates_this_run') != steps or
            result.get('skipped_batches') != 0 or
            result.get('observed_input_batch_sizes') != [PAPER_BATCH_PER_GPU] or
            result.get('optimizer_class') != 'torch.optim.adam.Adam' or
            sha256(run / 'checkpoints/final.ckpt') != result.get('final_checkpoint_sha256')):
        raise RuntimeError(f'{stage} did not execute the declared PARC-only native training')
    if stage != 'tokenizer' and not contract.get('terrain_conditioning'):
        raise RuntimeError('Control training did not enable local terrain observations')
    if stage != 'tokenizer' and steps >= 10:
        evidence = (result.get('terrain_training_evidence') or {}).get('nonzero_gradient_updates', {})
        if not all(evidence.get(key, 0) > 0 for key in ('encoder', 'attention')):
            raise RuntimeError('Local terrain conditions did not participate in optimization')
    if stage == 'pose' and (contract.get('tokenizer') or {}).get('checkpoint_sha256') != tokenizer_hash:
        raise RuntimeError('Pose loaded a different tokenizer')
    if stage == 'pose' and result.get('tokenizer_state_sha256') != contract.get('tokenizer_state_sha256'):
        raise RuntimeError('Pose modified its frozen tokenizer')
    if contract.get('status') not in ('completed', 'trained_not_reviewed'):
        raise RuntimeError('Stage did not finish')
    return result


def run(args):
    root = Path(__file__).resolve().parents[1]
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError('Training output must be separate from the immutable dataset')
    manifest = json.loads((dataset / 'native_manifest.json').read_text())
    verify_dataset(dataset, manifest)
    sources = verify_training_source_policy(dataset, manifest, parc_only=True)
    from motionbricks.data.native_mb_terrain_protocol import validate_terrain_dataset
    records = json.loads((dataset / 'accepted.json').read_text())
    terrain = validate_terrain_dataset(manifest, records)
    if not all(any(row['split'] == split and row['frames_30hz'] >= 65 for row in records)
               for split in ('validation', 'test')):
        raise ValueError('Both held-out splits need real native-frame windows')
    output.mkdir(parents=True, exist_ok=False)
    dataset_hash = sha256(dataset / 'native_manifest.json')
    files = list((root / 'motionbricks').rglob('*.py')) + [
        root / 'scripts' / name for name in (
            'train_native_mb.py', 'train_native_mb_control.py', 'run_parc_only_training.py',
            'evaluate_native_mb_checkpoint.py')]
    source_hashes = {path.relative_to(root).as_posix(): source_lf_sha256(path) for path in files}
    state = {'status': 'running', 'pid': os.getpid(), 'dataset': str(dataset),
        'dataset_manifest_sha256': dataset_hash, 'source_policy': sources, 'terrain_protocol': terrain,
        'steps_per_stage': args.steps, 'schedule_steps': PAPER_STEPS,
        'batch_per_gpu': PAPER_BATCH_PER_GPU, 'initialization': 'fresh PARC-only models and statistics',
        'stages': {stage: {'status': 'queued'} for stage in ('tokenizer', 'root', 'pose')},
        'quality_approved': False, 'source_lf_sha256': source_hashes}
    owned = {}
    deadline = time.monotonic() + args.wait_hours * 3600
    active = set()
    try:
        while not all(row['status'] == 'completed' for row in state['stages'].values()):
            if time.monotonic() > deadline:
                raise TimeoutError('Declared workflow time limit exceeded')
            if sha256(dataset / 'native_manifest.json') != dataset_hash:
                raise RuntimeError('Dataset changed during training')
            for relative, expected in source_hashes.items():
                if source_lf_sha256(root / relative) != expected:
                    raise RuntimeError('Frozen training source changed: ' + relative)
            for stage in tuple(owned):
                process, stream = owned[stage]
                if process.poll() is None:
                    continue
                stream.close()
                del owned[stage]
                info = state['stages'][stage]
                active.discard(info['gpu'])
                if process.returncode:
                    raise RuntimeError(f'{stage} failed with exit {process.returncode}; see {info["log"]}')
                tokenizer_hash = state['stages']['tokenizer'].get('result', {}).get('final_checkpoint_sha256')
                result = validate_result(Path(info['run']), stage, dataset_hash, args.steps, tokenizer_hash)
                info.update(status='completed', result=result)
            for stage, info in state['stages'].items():
                if info['status'] != 'queued':
                    continue
                if stage == 'pose' and state['stages']['tokenizer']['status'] != 'completed':
                    continue
                gpu = None
                for candidate in args.gpus:
                    if candidate in active:
                        continue
                    try:
                        select_gpu(candidate)
                    except RuntimeError:
                        continue
                    gpu = candidate
                    break
                if gpu is None:
                    continue
                run_dir = output / (stage + '-' + str(args.steps))
                command = stage_command(stage, dataset, run_dir, gpu, args.steps,
                    tokenizer_run=Path(state['stages']['tokenizer']['run']) if stage == 'pose' else None)
                log = output / (stage + '.log')
                # The trainer independently checks the GPU again before allocating.
                owned[stage] = launch(command, root, log)
                active.add(gpu)
                info.update(status='running', gpu=gpu, run=str(run_dir), command=command,
                    log=str(log), pid=owned[stage][0].pid)
            state.update(updated_at_unix=time.time())
            save_state(output / 'state.json', state)
            if owned or any(row['status'] == 'queued' for row in state['stages'].values()):
                time.sleep(args.poll_seconds)
        state.update(status='training_finished_behavior_review_pending', finished_at_unix=time.time())
    except BaseException as error:
        state.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        stop_owned(owned)
        save_state(output / 'state.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--wait-hours', type=float, default=72)
    args = parser.parse_args()
    if not 1 <= args.steps <= PAPER_STEPS or args.poll_seconds < 1 or args.wait_hours <= 0:
        parser.error('Invalid budget or polling interval')
    if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error('GPU indices must be distinct and nonnegative')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Workflow stopped')))
    print(json.dumps(run(args)), flush=True)


if __name__ == '__main__':
    main()
