"""Bounded actual-producer native/graph/cold proofs under real loader conditions."""

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
    output = run/'pose_actual_loader_optimizer_workflow_v1'
    output.mkdir(exist_ok=False)
    controls = read(run/'pose_graph_loader_isolation_v2/complete.json')
    for name in ('global_unpinned', 'threadlocal_pin', 'native_pin'):
        if not controls['results'][name]['completed'] or controls['results'][name]['native_computation_changes']:
            raise ValueError('Actual-loader zero-update controls did not establish numerical equivalence')
    state = dict(pid=os.getpid(), production_modified=False, production_budget_updates=0,
        control_report_sha256=digest(run/'pose_graph_loader_isolation_v2/complete.json'))
    reference = run/'pose_loader_optimizer_native_v1'
    candidate = run/'pose_loader_optimizer_graph_v1'
    for phase, mode, destination, extra in (
            ('native', 'native', reference, []),
            ('graph', 'graph', candidate, ['--reference', str(reference)]),
            ('resume', 'graph', run/'pose_loader_optimizer_resume_v1',
             ['--reference', str(reference), '--resume-state', str(candidate/'midpoint.pt')])):
        if destination.exists():
            raise ValueError('Do not repeat an actual-loader optimizer phase')
        stable = set()
        while True:
            snapshot, available = gpu_snapshot()
            selected = sorted(stable.intersection(available))
            if len(selected) >= 2:
                break
            stable = set(available)
            state.update(status='waiting_for_two_idle_gpus', phase=phase, gpu_snapshot=snapshot, available=available)
            save(output/'state.json', state)
            time.sleep(10 if len(available) >= 2 else 15)
        selected = selected[-2:]
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                       '--format=csv,noheader'], text=True)
        command = ['/usr/bin/timeout', '--kill-after=10s', '1200s', sys.executable, '-u', '-m',
            'scripts.check_pose_actual_loader_optimizer', '--gpus', *map(str, selected), '--mode', mode,
            '--checkpoint', str(run/'full_training_v13/pose/graph_handoff.pt'), '--output', str(destination), *extra]
        with (output/(phase+'.log')).open('x') as stream:
            child = subprocess.Popen(command, cwd=code, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, TMPDIR='${PRIVATE_STORAGE}/mbtmp', PYTHONPATH='.:out/vendor/kimodo',
                         OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
            state.update(status='running', phase=phase, child_pid=child.pid, gpus=selected, command=command,
                         gpu_snapshot_before=snapshot, compute_processes_before=apps)
            save(output/'state.json', state)
            result = child.wait()
        if result or not (destination/'complete.json').exists():
            raise RuntimeError(phase+' actual-producer diagnostic failed; retain evidence for review')
        if not read(destination/'complete.json')['passed']:
            raise ValueError(phase+' actual-producer optimizer comparison failed')
        save(output/(phase+'-verified.json'), dict(path=str(destination/'complete.json'),
                                                 sha256=digest(destination/'complete.json')))
    state.update(status='proofs_complete_production_integration_required', actual_diagnostic_updates=10,
                 production_approved=False, quality_approved=False)
    save(output/'state.json', state)
    save(output/'complete.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        save(args.base/'pfnn_mixed_20261015/pose_actual_loader_optimizer_workflow_v1/failure.json', dict(error=repr(error)))
        raise
