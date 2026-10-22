"""Run independent native Tokenizer/Root stages, then their matching Pose stage."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.train_native_mb import sha256


def available_devices(indices):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    observed = {}
    for line in rows.splitlines():
        index, uuid, memory, utilization = [value.strip() for value in line.split(',')]
        observed[int(index)] = {'index': int(index), 'uuid': uuid, 'used_mib': int(memory),
                                'utilization_percent': int(utilization)}
    chosen = [observed[index] for index in indices]
    if any(row['used_mib'] > 1024 or row['utilization_percent'] > 5 for row in chosen):
        raise RuntimeError('A requested GPU is no longer free: ' + json.dumps(chosen))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--drive-config', type=Path)
    parser.add_argument('--skip-review', action='store_true', help='Only for a bounded mechanics smoke run')
    parser.add_argument('--batch-per-gpu', type=int)
    parser.add_argument('--pose-batch-per-gpu', type=int)
    parser.add_argument('--precision', choices=('32', 'bf16-mixed'), default='32')
    parser.add_argument('--pose-precision', choices=('32', 'bf16-mixed'))
    parser.add_argument('--validation-crops', type=int, default=4)
    parser.add_argument('--validate-every', type=int, default=5000)
    parser.add_argument('--sampling-policy', choices=('clips-v1', 'parc-windows-v1'), default='clips-v1')
    parser.add_argument('--vertical-frame', choices=('observed-pelvis-y-v1', 'observed-terrain-y-v1'))
    args = parser.parse_args()
    args.source, args.dataset, args.output = args.source.resolve(), args.dataset.resolve(), args.output.resolve()
    indices = [int(value) for value in args.gpus.split(',')]
    if (len(indices) != 2 or len(set(indices)) != len(indices) or args.output.exists()
            or (args.batch_per_gpu is not None and args.batch_per_gpu < 1)):
        parser.error('The current project allocation is two distinct available GPUs; use a fresh output directory')
    if args.batch_per_gpu is None and 256 % len(indices):
        parser.error('Declare --batch-per-gpu for a rank count that cannot divide global batch256')
    source_code = Path(__file__).resolve().parents[1]
    if source_code == args.output or source_code in args.output.parents:
        parser.error('Frozen workflow output must be outside its source code tree')
    args.output.mkdir(parents=True)
    code = args.output / 'frozen_code'
    shutil.copytree(source_code, code,
                    ignore=shutil.ignore_patterns('__pycache__', '.git', 'wandb', '*.pyc'))
    state = {'status': 'starting', 'pid': os.getpid(), 'source_sha256': sha256(args.source),
             'dataset_manifest_sha256': sha256(args.dataset / 'native_manifest.json'),
             'steps_per_stage': args.steps, 'batch_per_gpu': args.batch_per_gpu,
             'global_batch': None if args.batch_per_gpu else 256, 'precision': args.precision,
             'pose_precision': args.pose_precision or args.precision,
             'sampling_policy': args.sampling_policy,
             'stages': {}, 'code': str(code), 'runtime_deployment': 'local PC only',
             'quality_approved': False}

    def save():
        state['updated_at_unix'] = time.time()
        temporary = args.output / 'state.tmp'
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(args.output / 'state.json')

    def launch(stage, cards):
        devices = available_devices(cards)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(row['uuid'] for row in devices),
                   PYTHONPATH=str(code) + os.pathsep + str(code / 'out/vendor/kimodo'),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        cache = args.output.parent / 'cache'
        for key, relative in [('XDG_CACHE_HOME', ''), ('HF_HOME', 'huggingface'),
                              ('MPLCONFIGDIR', 'matplotlib'), ('TORCH_HOME', 'torch'),
                              ('CUDA_CACHE_PATH', 'cuda'), ('TMPDIR', 'tmp')]:
            # multiprocessing's AF_UNIX sockets have a roughly 108-byte path limit.
            path = Path(os.environ.get('TMPDIR', '/tmp')) if key == 'TMPDIR' else cache / relative
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
            '--nnodes=1', '--nproc_per_node=' + str(len(devices)), '-m', 'scripts.train_mixed_control_stage',
            '--source', str(args.source), '--dataset', str(args.dataset), '--output', str(args.output / stage),
            '--stage', stage, '--steps', str(args.steps), '--workers', str(args.workers),
            '--global-batch', str((args.pose_batch_per_gpu if stage=='pose' and args.pose_batch_per_gpu
                                  else args.batch_per_gpu)*len(devices)
                                 if args.batch_per_gpu or (stage=='pose' and args.pose_batch_per_gpu) else 256),
            '--precision', args.pose_precision if stage == 'pose' and args.pose_precision else args.precision,
            '--validation-crops', str(args.validation_crops),
            '--validate-every', str(args.validate_every), '--sampling-policy', args.sampling_policy]
        if stage == 'pose':
            command += ['--tokenizer-run', str(args.output / 'tokenizer'), '--root-run', str(args.output / 'root')]
        if args.vertical_frame:
            command += ['--vertical-frame', args.vertical_frame]
        log = (args.output / f'{stage}.log').open('w')
        process = subprocess.Popen(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
        state['stages'][stage] = {'status': 'running', 'pid': process.pid, 'gpus': devices,
                                  'command': command, 'started_at': time.time()}
        save()
        return process, log

    save()
    jobs = {}
    try:
        half = len(indices) // 2
        jobs['tokenizer'] = launch('tokenizer', indices[:half])
        jobs['root'] = launch('root', indices[half:])
        state['status'] = 'training_tokenizer_and_root'
        save()
        while jobs:
            for stage, (process, log) in list(jobs.items()):
                code_value = process.poll()
                if code_value is None:
                    continue
                log.close()
                state['stages'][stage].update(status='completed' if code_value == 0 else 'failed',
                                               exit_code=code_value, finished_at=time.time())
                del jobs[stage]
                save()
                if code_value:
                    raise RuntimeError(stage + ' training failed; see stage log')
            if jobs:
                time.sleep(10)
        state['status'] = 'training_pose'
        jobs['pose'] = launch('pose', indices)
        save()
        process, log = jobs['pose']
        code_value = process.wait()
        log.close()
        del jobs['pose']
        state['stages']['pose'].update(status='completed' if code_value == 0 else 'failed', exit_code=code_value)
        if code_value:
            raise RuntimeError('Pose training failed; see stage log')
        command = [sys.executable, '-u', '-m', 'scripts.export_mixed_control_bundle',
            '--source', str(args.source), '--dataset', str(args.dataset), '--runs', str(args.output),
            '--output', str(args.output / 'pipeline.pt')]
        with (args.output / 'export.log').open('w') as log:
            subprocess.run(command, cwd=code, stdout=log, stderr=subprocess.STDOUT, check=True,
                env=dict(os.environ, PYTHONPATH=str(code) + os.pathsep + str(code / 'out/vendor/kimodo')))
        state['status'] = 'trained_exported_pending_closed_loop_review'
        state['bundle_sha256'] = sha256(args.output / 'pipeline.pt')
        save()
        if not args.skip_review:
            review_device = available_devices([indices[0]])[0]
            state['status'] = 'reviewing_flat_stop'
            save()
            command = [sys.executable, '-u', '-m', 'scripts.evaluate_mixed_control_rollout',
                '--bundle', str(args.output / 'pipeline.pt'), '--dataset', str(args.dataset),
                '--output', str(args.output / 'closed_loop_review')]
            with (args.output / 'closed_loop_review.log').open('w') as log:
                subprocess.run(command, cwd=code, stdout=log, stderr=subprocess.STDOUT, check=True,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=review_device['uuid'],
                             PYTHONPATH=str(code) + os.pathsep + str(code / 'out/vendor/kimodo')))
            review = json.loads((args.output / 'closed_loop_review/metrics.json').read_text())
            state['flat_stop_gate_passed'] = review['flat_stop_gate_passed']
            state['status'] = 'reviewed_pending_local_gameplay_validation'
            save()
        if args.drive_config:
            remote = 'mb-transfer:runs/' + args.output.name
            command = ['rclone', '--config', str(args.drive_config), 'copy', str(args.output), remote,
                       '--include', '/pipeline.pt', '--include', '/pipeline.json', '--include', '/state.json',
                       '--include', '/closed_loop_review/**', '--immutable', '--drive-chunk-size', '64M']
            with (args.output / 'cloud_export.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            state['private_drive_output'] = remote
            save()
    except Exception as error:
        state.update(status='failed', error=str(error))
        save()
        raise
    finally:
        # Never stop unrelated users' processes. These are only our own torchrun children.
        for process, log in jobs.values():
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()


if __name__ == '__main__':
    main()
