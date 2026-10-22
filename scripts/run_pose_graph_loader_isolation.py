"""Compare real-loader capture controls while verified native production continues."""

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
    suffix = 'v'+str(a.version)
    output = run/('pose_graph_loader_isolation_'+suffix)
    output.mkdir(exist_ok=False)
    if not read(run/'full_training_v15/recovery.json')['no_optimizer_updates_lost']:
        raise ValueError('Verified native recovery receipt is required first')
    state = dict(pid=os.getpid(), actual_optimizer_updates=0, production_budget_updates=0, production_modified=False)
    cases = [('global_pin', 'global', 'on'), ('global_unpinned', 'global', 'off'),
             ('threadlocal_pin', 'thread_local', 'on'), ('native_pin', 'native', 'on')]
    results = {}
    for label, mode, pin in cases:
        destination = run/('pose_graph_loader_'+label+'_'+suffix)
        if destination.exists():
            raise ValueError('Do not repeat a completed or failed loader control')
        stable = set()
        while True:
            snapshot, available = gpu_snapshot()
            choices = sorted(stable.intersection(available))
            if len(choices) >= 2:
                break
            stable = set(available)
            state.update(status='waiting_for_two_idle_gpus', case=label, gpu_snapshot=snapshot, available=available)
            save(output/'state.json', state)
            time.sleep(10 if len(available) >= 2 else 15)
        selected = choices[-2:]
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                       '--format=csv,noheader'], text=True)
        command = ['/usr/bin/timeout', '--kill-after=10s', '600s', sys.executable, '-u', '-m',
            'scripts.check_pose_graph_loader', '--gpus', *map(str, selected), '--mode', mode, '--pin-memory', pin,
            '--checkpoint', str(run/'full_training_v13/pose/graph_handoff.pt'), '--output', str(destination)]
        with (output/(label+'.log')).open('x') as stream:
            child = subprocess.Popen(command, cwd=code, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, TMPDIR='${PRIVATE_STORAGE}/mbtmp', PYTHONPATH='.:out/vendor/kimodo',
                         OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
            state.update(status='running', case=label, child_pid=child.pid, command=command, gpus=selected,
                         gpu_snapshot_before=snapshot, compute_processes_before=apps)
            save(output/'state.json', state)
            code_result = child.wait()
        if (destination/'complete.json').exists():
            results[label] = dict(completed=True, report=str(destination/'complete.json'))
        elif (destination/'failure.json').exists():
            failure = read(destination/'failure.json')
            results[label] = dict(completed=False, report=str(destination/'failure.json'))
            if failure['actual_optimizer_updates'] != 0 or 'capture' not in failure['error'].lower():
                raise RuntimeError(label+' failed outside the capture controls; inspect before continuing')
        else:
            raise RuntimeError(label+' exited without a report: '+str(code_result))
        save(output/(label+'-result.json'), dict(**results[label], sha256=digest(Path(results[label]['report']))))
    if not results['native_pin']['completed']:
        raise ValueError('Native real-loader reference did not complete')
    import numpy as np
    import torch
    torch.set_num_threads(4)

    def changes(left, right, path=''):
        if torch.is_tensor(left):
            return [] if torch.equal(left, right) else [dict(path=path,
                max_abs=float((left.to(torch.float64)-right.to(torch.float64)).abs().max()))]
        if isinstance(left, np.ndarray):
            return [] if np.array_equal(left, right) else [dict(path=path)]
        if isinstance(left, dict):
            return [item for key in left for item in changes(left[key], right[key], path+'.'+str(key))]
        if isinstance(left, (list, tuple)):
            return [item for i, (x, y) in enumerate(zip(left, right)) for item in changes(x, y, path+'.'+str(i))]
        return [] if left == right else [dict(path=path)]

    reference = torch.load(run/('pose_graph_loader_native_pin_'+suffix)/'observed.pt', map_location='cpu', weights_only=False)
    for label, value in results.items():
        if value['completed']:
            observed = torch.load(run/('pose_graph_loader_'+label+'_'+suffix)/'observed.pt', map_location='cpu', weights_only=False)
            value['native_computation_changes'] = changes(reference, observed)
            del observed
    state.update(status='controls_complete_review_required', results=results, production_approved=False,
                 optimizer_continuation_proven=False, quality_approved=False)
    save(output/'state.json', state)
    save(output/'complete.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--version', type=int, choices=(1,2), default=1)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        save(args.base/('pfnn_mixed_20261015/pose_graph_loader_isolation_v'+str(args.version))/'failure.json', dict(error=repr(error)))
        raise
