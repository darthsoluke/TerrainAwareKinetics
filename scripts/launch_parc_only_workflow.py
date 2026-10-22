"""Launch an authorized durable training workflow, with an explicit PID receipt."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def launch(args):
    root = Path(__file__).resolve().parents[1]
    receipt = args.receipt.resolve()
    if receipt.exists() or args.output.exists():
        raise FileExistsError('Use a new receipt and workflow output')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(root / 'scripts/run_parc_only_training.py'),
        '--dataset', str(args.dataset.resolve()), '--output', str(args.output.resolve()),
        '--steps', str(args.steps), '--gpus', *map(str, args.gpus), '--poll-seconds', '10']
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join([str(root), str(root / 'out/vendor/kimodo')])
    environment['MPLBACKEND'] = 'Agg'
    log = receipt.with_suffix('.log')
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    result = {'status': 'dispatched_not_completed', 'pid': child.pid, 'command': command,
              'log': str(log), 'code_root': str(root), 'started_at_unix': time.time(),
              'workflow_state': str(args.output.resolve() / 'state.json'), 'quality_approved': False}
    receipt.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()
    if args.steps < 1 or not args.gpus or min(args.gpus) < 0:
        parser.error('Steps and GPU indices must be valid')
    print(json.dumps(launch(args)), flush=True)


if __name__ == '__main__':
    main()
