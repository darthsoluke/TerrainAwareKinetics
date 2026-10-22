"""Run only new graph execution arms against the retained native Pose reference."""

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
    output = run/'pose_graph_optimizer_workflow_v1'
    output.mkdir(exist_ok=False)
    native = run/'pose_parallel_exposure_reference_v1/complete.json'
    zero = run/'pose_frozen_graph_probe_v1/complete.json'
    if not read(native)['passed'] or not read(zero)['passed']:
        raise ValueError('Required native optimizer reference or frozen-label numerical proof failed')
    if read(native)['implementation_sha256'] != digest(code/'motionbricks/data/parallel_pose_training.py'):
        raise ValueError('Base Pose parallel implementation differs from the native proof')
    for name in ('motionbricks/data/dilated_graph_convolution.py', 'motionbricks/data/dilated_graph_context.py'):
        if read(zero)['code_sha256'][name] != digest(code/name):
            raise ValueError('Label graph execution differs from the zero-update proof')
    state = dict(pid=os.getpid(), production_budget_updates=0, production_modified=False,
        native_reference_reused=str(native), native_reference_sha256=digest(native),
        zero_update_proof_sha256=digest(zero))
    for phase in ('parallel', 'resume'):
        destination = run/('pose_graph_optimizer_'+phase+'_v1')
        if destination.exists():
            raise ValueError('Do not overwrite or repeat a graph optimizer diagnostic')
        stable = set()
        while True:
            snapshot, available = gpu_snapshot()
            candidates = sorted(stable.intersection(available))
            if len(candidates) >= 2:
                break
            stable = set(available)
            state.update(status='waiting_for_two_idle_gpus', phase=phase, gpu_snapshot=snapshot, available=available)
            save(output/'state.json', state)
            time.sleep(3 if len(available) >= 2 else 15)
        gpus = candidates[:2]
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                       '--format=csv,noheader'], text=True)
        command = ['/usr/bin/timeout', '--kill-after=10s', '900s', sys.executable, '-u', '-m',
            'scripts.check_pose_graph_optimizer', '--phase', phase, '--gpus', *map(str, gpus),
            '--checkpoint', str(run/'codec_pipeline_inputs_v1/checkpoint.pt'), '--output', str(destination)]
        if phase == 'resume':
            command += ['--resume-state', str(run/'pose_graph_optimizer_parallel_v1/midpoint.pt')]
        with (output/(phase+'.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=code, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, TMPDIR='${PRIVATE_STORAGE}/mbtmp', PYTHONPATH='.:out/vendor/kimodo',
                         OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
            state.update(status='running', phase=phase, child_pid=child.pid, gpus=gpus, command=command,
                         gpu_snapshot_before=snapshot, compute_processes_before=apps)
            save(output/'state.json', state)
            result = child.wait()
        if result or not (destination/'complete.json').exists():
            raise RuntimeError(phase+' graph diagnostic failed; inspect retained evidence')
        report = read(destination/'complete.json')
        if not report['passed']:
            raise ValueError(phase+' graph optimizer comparison failed')
        save(output/(phase+'-verified.json'), dict(path=str(destination/'complete.json'),
                                                 sha256=digest(destination/'complete.json')))
    state.update(status='proofs_complete_production_integration_required', actual_diagnostic_updates=6,
                 reused_native_reference_updates=4, production_approved=False, quality_approved=False)
    save(output/'state.json', state)
    save(output/'complete.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        save(args.base/'pfnn_mixed_20261015/pose_graph_optimizer_workflow_v1/failure.json', dict(error=repr(error)))
        raise
