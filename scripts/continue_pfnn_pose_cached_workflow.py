"""Adopt an unchanged running Root stage, then run codec and memoized Pose."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def stage_command(a, stage, parent):
    root = a.base / 'pfnn_mixed_20261015'
    command = [sys.executable, '-u', '-m', 'scripts.train_unified_full_corpus', '--stage', stage,
        '--bundle', str(a.base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'),
        '--parent', str(parent), '--preflight', str(root/'preflight_world_v1/complete.json'),
        '--cache', str(root/'cache_mixed_v1'), str(root/'cache_pfnn_world_v2'),
        '--output', str(a.output/stage), '--steps', '50000', '--batch', '256', '--micro-batch', '64',
        '--workers', '4', '--corpus-audit', str(a.previous/'corpus_audit.json')]
    if stage == 'pose':
        command.append('--cache-pose-tokens')
    return command


def validate(a):
    root = a.base/'pfnn_mixed_20261015'
    for name in ('pose_token_speed_v3', 'pose_token_resume_v1'):
        if read(root/name/'complete.json').get('passed') is not True:
            raise ValueError('Token equivalence and exact resume evidence are required')
    contract = read(a.previous/'root/contract.json')
    if (contract['stage'] != 'root' or contract['mechanics_only']
            or contract['steps'] != 50000 or contract['batch'] != 256 or contract['micro_batch'] != 64):
        raise ValueError('Only the same authorized production Root budget can be adopted')
    if not read(a.previous/'corpus_audit.json')['passed']:
        raise ValueError('Complete corpus audit required')
    for stage in ('codec', 'pose'):
        if (a.previous/stage).exists():
            raise ValueError('Old workflow has already advanced beyond Root')
    return contract


def main(a):
    code = Path(__file__).resolve().parents[1]
    contract = validate(a)
    if a.validate_only:
        print(json.dumps(dict(validated=True, root_directory=str(a.previous/'root'),
            codec_command=stage_command(a, 'codec', a.previous/'root/weights.pt'),
            pose_command=stage_command(a, 'pose', a.output/'codec/weights.pt'),
            total_budget_updates=150000, cache_changes_optimizer_math=False)))
        return
    handoff = read(a.previous/'acceleration_handoff.json')
    old = Path('/proc')/str(handoff['previous_supervisor_pid'])
    if old.exists() and '\nState:\tZ' not in (old/'status').read_text():
        raise ValueError('Previous supervisor must exit before continuation starts')
    if (handoff['successor_output'] != str(a.output) or handoff['previous_job'] != 'root-resume'
            or handoff['root_output'] != str(a.previous/'root')):
        raise ValueError('Unexpected handoff identity')
    if not a.worker:
        if a.output.exists() or a.output.with_suffix('.process.json').exists():
            raise ValueError('Continuation exists; inspect its state')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--base', str(a.base),
            '--previous', str(a.previous), '--output', str(a.output), '--gpu', str(a.gpu), '--worker']
        with a.output.with_suffix('.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=code, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ,
                    TMPDIR='${PRIVATE_STORAGE}/mbtmp', PYTHONPATH='.:out/vendor/kimodo',
                    OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
        save(a.output.with_suffix('.process.json'), dict(pid=child.pid, command=command,
            code=str(code), gpu=a.gpu, adopted_root_pid=handoff['root_pid'],
            adopted_root_output=handoff['root_output'], total_budget_updates=150000))
        print(json.dumps(dict(pid=child.pid, adopted_root_pid=handoff['root_pid'])))
        return
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), stage='root', status='adopting_running_root',
        child_pid=handoff['root_pid'], root_directory=str(a.previous/'root'),
        code=str(code), total_budget_updates=150000, batch=256, micro_batch=64,
        root_restarted=False, quality_approved=False)

    def run(label, command):
        while True:
            snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True)
            _, memory, busy = map(int, next(row for row in snapshot.splitlines()
                if int(row.split(',')[0]) == a.gpu).split(','))
            if memory <= 500 and busy <= 5:
                break
            state.update(status='waiting_for_idle_gpu', gpu=a.gpu)
            save(a.output/'state.json', state)
            time.sleep(30)
        with (a.output/(label+'.log')).open('x') as stream:
            child = subprocess.Popen(command, cwd=code, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu),
                    TMPDIR='${PRIVATE_STORAGE}/mbtmp', PYTHONPATH='.:out/vendor/kimodo',
                    OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8'))
            state.update(job=label, command=command, child_pid=child.pid, status='running',
                gpu_snapshot_before=snapshot)
            save(a.output/'state.json', state)
            if child.wait() != 0:
                raise RuntimeError(label+' failed; inspect preserved log')

    try:
        root_folder = a.previous/'root'
        while not (root_folder/'complete.json').exists():
            if (root_folder/'failure.json').exists():
                raise RuntimeError('Adopted Root failed; inspect previous Root failure')
            process = Path('/proc')/str(handoff['root_pid'])/'cmdline'
            if not process.exists() or str(root_folder) not in process.read_bytes().replace(b'\0', b' ').decode():
                raise RuntimeError('Adopted Root ended without completion; no automatic duplicate')
            state['root_progress'] = read(root_folder/'progress.json') if (root_folder/'progress.json').exists() else None
            save(a.output/'state.json', state)
            time.sleep(30)
        complete = read(root_folder/'complete.json')
        if complete['actual_updates'] != 50000 or complete['actual_exposures'] != 12800000 or not complete['protected_hash_passed']:
            raise ValueError('Adopted Root completion differs from the fixed budget')
        parent = root_folder/'weights.pt'
        for stage in ('codec', 'pose'):
            state['stage'] = stage
            command = stage_command(a, stage, parent)
            run(stage+'-startup', command+['--stop-after', '2'])
            stopped = read(a.output/stage/'stopped.json')
            if stopped['actual_updates'] != 2 or stopped['actual_exposures'] != 512:
                raise ValueError('Budgeted startup accounting differs')
            run(stage+'-resume', command+['--resume', str(a.output/stage/'latest.pt')])
            complete = read(a.output/stage/'complete.json')
            if complete['actual_updates'] != 50000 or complete['actual_exposures'] != 12800000 or not complete['protected_hash_passed']:
                raise ValueError('Completed stage differs from the original budget')
            parent = a.output/stage/'weights.pt'
        state.update(status='training_complete_quality_review_required', checkpoint=str(parent),
            actual_updates=150000, actual_exposures=38400000, quality_approved=False)
        save(a.output/'complete.json', state)
    except Exception as error:
        state.update(status='failed', error=repr(error))
        save(a.output/'failure.json', state)
        raise
    finally:
        save(a.output/'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'previous', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    main(parser.parse_args())
