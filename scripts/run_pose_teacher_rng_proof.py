"""Wait for the existing Pose resume proof, then verify two-device teacher RNG."""

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
    code = Path(__file__).resolve().parents[1]
    output = run/'pose_teacher_rng_workflow_v1'
    output.mkdir(exist_ok=False)
    dependency = run/'pose_exposure_resume_workflow_v1'
    destination = run/'pose_teacher_rng_v1'
    if destination.exists():
        raise ValueError('Refuse to repeat an existing diagnostic')
    while True:
        snapshot, available = gpu_snapshot()
        if (dependency/'failure.json').exists():
            raise ValueError('Existing Pose resume proof failed; review before more GPU work')
        dependency_complete = (dependency/'complete.json').exists()
        if dependency_complete and len(available) >= 2:
            break
        save(output/'state.json', dict(pid=os.getpid(), status='waiting_for_dependency' if not dependency_complete
            else 'waiting_for_two_idle_gpus', gpu_snapshot=snapshot, available=available,
            actual_optimizer_updates=0, production_budget_updates=0))
        time.sleep(15)
    for phase in ('reference', 'resume'):
        if not read(run/('pose_exposure_'+phase+'_v1')/'complete.json')['passed']:
            raise ValueError('Required warm/cold Pose proof did not pass')
    gpus = available[:2]
    command = ['/usr/bin/timeout', '--kill-after=10s', '900s', sys.executable, '-u', '-m',
        'scripts.check_pose_teacher_device_rng', '--gpus', *map(str, gpus),
        '--checkpoint', str(run/'codec_pipeline_inputs_v1/checkpoint.pt'), '--output', str(destination)]
    with (output/'diagnostic.log').open('x') as log:
        child = subprocess.Popen(command, cwd=code, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, TMPDIR='${PRIVATE_STORAGE}/mbtmp', PYTHONPATH='.:out/vendor/kimodo',
                OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
        save(output/'state.json', dict(pid=os.getpid(), status='running', child_pid=child.pid,
            command=command, gpus=gpus, gpu_snapshot_before=snapshot, actual_optimizer_updates=0))
        result = child.wait()
    if result or not (destination/'complete.json').exists():
        raise RuntimeError('Two-device RNG diagnostic failed; inspect retained evidence')
    if not read(destination/'complete.json')['passed']:
        raise ValueError('Two-device RNG numerical gate failed')
    save(output/'complete.json', dict(status='complete', report=str(destination/'complete.json'),
        report_sha256=digest(destination/'complete.json'), actual_optimizer_updates=0,
        production_budget_updates=0, production_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        save(args.base/'pfnn_mixed_20261015/pose_teacher_rng_workflow_v1/failure.json', dict(error=repr(error)))
        raise
