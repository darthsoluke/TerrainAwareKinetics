"""Resume a zero-update resource rejection without repeating the Pose reference."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_pfnn_bias_workflow import read, save, digest
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(a):
    run = a.base/'pfnn_mixed_20261015'
    output = run/'pose_execution_recovery_v2'
    output.mkdir(exist_ok=False)
    failure = read(run/'pose_exposure_resume_v1/failure.json')
    reference = run/'pose_exposure_reference_v1/complete.json'
    if (failure['error'] != "ValueError('Selected GPU is no longer idle')"
            or failure['actual_optimizer_updates'] != 0 or not read(reference)['passed']):
        raise ValueError('Recovery is restricted to the reviewed zero-update resource rejection')
    for pid in (2509671, 2606024, 2656925):
        if Path(f'/proc/{pid}').exists():
            raise ValueError('An original proof supervisor is still alive')
    state = dict(pid=os.getpid(), production_budget_updates=0, production_modified=False,
        preserved_failure=str(run/'pose_exposure_resume_v1/failure.json'),
        reference_reused=str(reference), reference_sha256=digest(reference))
    phases = [
        ('resume', 1, 'code_pose_exposure_resume_v1', 'scripts.check_pose_exposure_resume',
         'pose_exposure_resume_v2', ['--batch', '3072', '--reference', str(reference.parent)]),
        ('teacher_rng', 2, 'code_pose_teacher_rng_v1', 'scripts.check_pose_teacher_device_rng',
         'pose_teacher_rng_v1', []),
        ('parallel_reference', 1, 'code_pose_parallel_exposure_v1', 'scripts.check_parallel_pose_exposure',
         'pose_parallel_exposure_reference_v1', ['--phase', 'reference']),
        ('parallel', 2, 'code_pose_parallel_exposure_v1', 'scripts.check_parallel_pose_exposure',
         'pose_parallel_exposure_parallel_v1', ['--phase', 'parallel']),
        ('parallel_resume', 2, 'code_pose_parallel_exposure_v1', 'scripts.check_parallel_pose_exposure',
         'pose_parallel_exposure_resume_v1', ['--phase', 'resume'])]
    for phase, count, folder, module, name, extra in phases:
        destination = run/name
        if destination.exists():
            raise ValueError('Refuse to overwrite a diagnostic result')
        stable = set()
        while True:
            snapshot, available = gpu_snapshot()
            candidates = sorted(stable.intersection(available))
            if len(candidates) >= count:
                break
            stable = set(available)
            state.update(status='waiting_for_stable_idle_gpus', phase=phase,
                         gpu_snapshot=snapshot, available=available, required_devices=count)
            save(output/'state.json', state)
            time.sleep(3 if len(available) >= count else 15)
        gpus = candidates[:count]
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                       '--format=csv,noheader'], text=True)
        gpu_flag = '--gpu' if phase == 'resume' else '--gpus'
        command = ['/usr/bin/timeout', '--kill-after=10s', '900s', sys.executable, '-u', '-m', module,
            gpu_flag, *map(str, gpus), '--checkpoint', str(run/'codec_pipeline_inputs_v1/checkpoint.pt'),
            '--output', str(destination), *extra]
        with (output/(phase+'.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=run/folder, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, TMPDIR='${PRIVATE_STORAGE}/mbtmp', PYTHONPATH='.:out/vendor/kimodo',
                         OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
            state.update(status='running', phase=phase, child_pid=child.pid, command=command,
                         gpus=gpus, gpu_snapshot_before=snapshot, compute_processes_before=apps)
            save(output/'state.json', state)
            result = child.wait()
        if result or not (destination/'complete.json').exists():
            raise RuntimeError(phase+' failed; inspect retained evidence before any new attempt')
        if not read(destination/'complete.json')['passed']:
            raise ValueError(phase+' numerical or optimizer continuation gate failed')
        save(output/(phase+'-verified.json'), dict(path=str(destination/'complete.json'),
                                                 sha256=digest(destination/'complete.json')))
    state.update(status='proofs_complete_production_integration_required', actual_diagnostic_updates=12,
                 reused_reference_updates=4, production_approved=False, quality_approved=False)
    save(output/'state.json', state)
    save(output/'complete.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        save(args.base/'pfnn_mixed_20261015/pose_execution_recovery_v2/failure.json', dict(error=repr(error)))
        raise
