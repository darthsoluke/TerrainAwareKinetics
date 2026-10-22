"""Run bounded Pose optimizer proofs after the device-isolation prerequisite."""

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
    output = run/'pose_parallel_exposure_workflow_v1'
    output.mkdir(exist_ok=False)
    dependency = run/'pose_teacher_rng_workflow_v1'
    state = dict(pid=os.getpid(), production_budget_updates=0, production_modified=False)
    while not (dependency/'complete.json').exists():
        if (dependency/'failure.json').exists():
            raise ValueError('Teacher isolation prerequisite failed; inspect before more GPU work')
        snapshot, available = gpu_snapshot()
        state.update(status='waiting_for_teacher_rng_proof', gpu_snapshot=snapshot, available=available)
        save(output/'state.json', state)
        time.sleep(15)
    prerequisite = read(run/'pose_teacher_rng_v1/complete.json')
    if not prerequisite['passed']:
        raise ValueError('Teacher isolation gate did not pass')
    for phase in ('reference', 'parallel', 'resume'):
        destination = run/('pose_parallel_exposure_'+phase+'_v1')
        if destination.exists():
            raise ValueError('Refuse to repeat an existing optimizer diagnostic')
        count = 1 if phase == 'reference' else 2
        while True:
            snapshot, available = gpu_snapshot()
            if len(available) >= count:
                break
            state.update(status='waiting_for_idle_gpus', phase=phase, required_devices=count,
                         gpu_snapshot=snapshot, available=available)
            save(output/'state.json', state)
            time.sleep(15)
        gpus = available[:count]
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                       '--format=csv,noheader'], text=True)
        command = ['/usr/bin/timeout', '--kill-after=10s', '900s', sys.executable, '-u', '-m',
            'scripts.check_parallel_pose_exposure', '--gpus', *map(str, gpus), '--phase', phase,
            '--checkpoint', str(run/'codec_pipeline_inputs_v1/checkpoint.pt'), '--output', str(destination)]
        with (output/(phase+'.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=code, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, TMPDIR='${PRIVATE_STORAGE}/mbtmp', PYTHONPATH='.:out/vendor/kimodo',
                         OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
            state.update(status='running', phase=phase, child_pid=child.pid, command=command, gpus=gpus,
                         gpu_snapshot_before=snapshot, compute_processes_before=apps)
            save(output/'state.json', state)
            result = child.wait()
        if result or not (destination/'complete.json').exists():
            raise RuntimeError(phase+' failed; evidence retained without retry')
        report = read(destination/'complete.json')
        if not report['passed']:
            raise ValueError(phase+' optimizer or continuation gate failed')
        save(output/(phase+'-verified.json'), dict(path=str(destination/'complete.json'),
                                                 sha256=digest(destination/'complete.json')))
    state.update(status='proofs_complete_production_integration_required', actual_diagnostic_updates=10,
                 production_approved=False, quality_approved=False)
    save(output/'complete.json', state)
    save(output/'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        save(args.base/'pfnn_mixed_20261015/pose_parallel_exposure_workflow_v1/failure.json', dict(error=repr(error)))
        raise
