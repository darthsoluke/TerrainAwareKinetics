"""Recover the exact native Pose checkpoint after a first-capture failure."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main(a):
    run = a.base/'pfnn_mixed_20261015'
    code = run/'code_full_workflow_v13'
    sys.path.insert(0, str(code))
    from scripts.run_pfnn_bias_workflow import read, save, digest, process_alive
    from scripts.run_pfnn_parallel_workflow import gpu_snapshot
    output = run/'full_training_v15'
    failed = run/'full_training_v14'
    source = run/'full_training_v13'
    handoff = source/'pose/graph_handoff.pt'
    if any(process_alive(pid) for pid in (2815677, 2816402, 3042913, 3070676)):
        raise ValueError('A prior producer is still alive; do not duplicate training')
    log = (failed/'pose-resume.log').read_text()
    if ('self._prepare_graph(worker, batch)' not in log or 'error during capture' not in log
            or (failed/'pose/progress.json').exists() or (failed/'pose/latest.pt').exists()):
        raise ValueError('Zero-update first-capture failure must be reviewed before recovery')
    migration = read(failed/'migration.json')
    if (migration['actual_updates'] != 2584 or migration['actual_exposures'] != 4100096
            or digest(handoff) != migration['from_checkpoint_sha256']):
        raise ValueError('Complete native handoff checkpoint identity changed')
    output.mkdir(exist_ok=False)
    for name in ('root_completion.json', 'codec_completion.json'):
        save(output/name, read(source/name))
    command = list(read(source/'state.json')['command'])
    command[command.index('--resume')+1] = str(handoff)
    command[command.index('--output')+1] = str(output/'pose')
    if '--graph-frozen-pose-labels' in command or command[command.index('--resume-migration')+1] != str(source/'migration.json'):
        raise ValueError('Recovery must retain the exact verified native source contract')
    state = dict(pid=os.getpid(), stage='pose', status='waiting_for_two_idle_gpus',
        total_budget_reference_updates=150000, total_budget_exposures=38400000,
        budget_unit='reference_sample_exposures', optimizer_trajectory_equivalent=False, quality_approved=False)
    stable = set()
    while True:
        snapshot, available = gpu_snapshot()
        choices = sorted(stable.intersection(available))
        if len(choices) >= 2:
            break
        stable = set(available)
        state.update(gpu_snapshot=snapshot, available=available)
        save(output/'state.json', state)
        time.sleep(3 if len(available) >= 2 else 15)
    gpus = choices[:2]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                   '--format=csv,noheader'], text=True)
    save(output/'recovery.json', dict(failed_directory=str(failed), source_directory=str(source),
        checkpoint=str(handoff), checkpoint_sha256=migration['from_checkpoint_sha256'],
        resumed_actual_updates=2584, resumed_actual_exposures=4100096, resumed_reference_updates=16016,
        failed_graph_optimizer_updates=0, no_optimizer_updates_lost=True,
        zero_update_basis='Failure in first shape preparation before base backward; all remaining batches are6144/3072.',
        original_source_reused=str(code), source_and_contract_unchanged=True, failure_log_sha256=digest(failed/'pose-resume.log')))
    with (output/'pose-resume.log').open('x') as stream:
        child = subprocess.Popen(command, cwd=code, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), TMPDIR='${PRIVATE_STORAGE}/mbtmp',
                     PYTHONPATH='.:out/vendor/kimodo', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    state.update(status='running', child_pid=child.pid, job_gpus=gpus, command=command,
                 gpu_snapshot_before=snapshot, compute_processes_before=apps)
    save(output/'state.json', state)
    if child.wait():
        raise RuntimeError('Native Pose recovery failed; inspect retained evidence')
    from motionbricks.data.reference_exposure_budget import validate_completion_counts
    complete = read(output/'pose/complete.json')
    validate_completion_counts(read(output/'pose/contract.json'), complete['actual_updates'], complete['actual_exposures'])
    if not complete['protected_hash_passed'] or digest(output/'pose/weights.pt') != complete['checkpoint_sha256']:
        raise ValueError('Final native Pose identity or protected state differs')
    root, codec = (read(output/name) for name in ('root_completion.json', 'codec_completion.json'))
    state.update(status='training_complete_quality_review_required', actual_exposures=38400000,
        equivalent_reference_updates=150000, actual_updates=root['actual_updates']+codec['actual_updates']+complete['actual_updates'],
        final_checkpoint=str(output/'pose/weights.pt'))
    save(output/'complete.json', state)
    save(output/'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        sys.path.insert(0, str(args.base/'pfnn_mixed_20261015/code_full_workflow_v13'))
        from scripts.run_pfnn_bias_workflow import save
        destination = args.base/'pfnn_mixed_20261015/full_training_v15'
        if destination.exists():
            save(destination/'failure.json', dict(error=repr(error)))
        raise
