"""Run four bounded decoder ablations on the two available4090D devices."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_mixed_control_workflow import available_devices
from scripts.run_terrain_decoder_pilot import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    out = args.base/'runs'
    previous = None
    if args.resume:
        previous = json.loads((out/'state.json').read_text())
        if previous['status'] != 'failed':
            raise ValueError('Only a failed coordinator can be recovered')
    out.mkdir(exist_ok=args.resume)
    code = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TMPDIR='/tmp')
    state = {'status': 'starting', 'pid': os.getpid(), 'steps_per_arm': 500, 'arms': {},
             'source': str(args.base/'input/source.pt'), 'code': str(code), 'quality_approved': False}
    pending = []
    for arm in ('baseline', 'feedback', 'terrain', 'both'):
        if previous and previous['arms'].get(arm, {}).get('status') == 'complete':
            result = json.loads((out/arm/'training_result.json').read_text())
            if result['updates'] != 500 or not (out/arm/'review/metrics.json').exists():
                raise ValueError('Completed pilot lacks results')
            state['arms'][arm] = previous['arms'][arm]
        else:
            pending.append(arm)
    save(out/'state.json', state)
    jobs = {}
    failed = False
    while pending or jobs:
        for gpu, (arm, process, log) in list(jobs.items()):
            result = process.poll()
            if result is not None:
                log.close();del jobs[gpu]
                lines = (out/(arm+'.log')).read_text(errors='replace').strip().splitlines()
                text = lines[-1] if lines else ''
                device_race = (result != 0 and not (out/arm).exists()
                    and 'RuntimeError: GPU ' in text and ' is occupied:' in text)
                state['arms'][arm].update(status='retrying_device_preflight' if device_race else
                    ('complete' if result == 0 else 'failed'), exit_code=result)
                if device_race:
                    pending.insert(0, arm)
                else:
                    failed = failed or result != 0
                save(out/'state.json', state)
        if failed:
            pending.clear()
        for gpu in (0, 1):
            if gpu in jobs or not pending:
                continue
            try:
                available_devices([gpu])
            except RuntimeError:
                continue
            arm = pending.pop(0)
            command = [sys.executable, '-u', '-m', 'scripts.run_terrain_decoder_pilot',
                '--source', str(args.base/'input/source.pt'), '--data', str(args.base/'input/data'),
                '--output', str(out/arm), '--arm', arm, '--gpu', str(gpu), '--steps', '500']
            log = (out/(arm+'.log')).open('a')
            process = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            jobs[gpu] = (arm, process, log)
            state['arms'][arm] = {'status': 'running', 'pid': process.pid, 'gpu': gpu, 'command': command}
        state['status'] = 'running' if jobs else 'waiting_for_free_4090d_device'
        save(out/'state.json', state)
        if pending or jobs:time.sleep(10)
    state['status'] = 'failed' if failed else 'completed_review_required'
    save(out/'state.json', state)


if __name__ == '__main__':
    main()
