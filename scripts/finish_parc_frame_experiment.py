"""Complete fixed reviews after an owned matched-stage run has fully exported."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_native_mb import select_gpu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    while True:
        state = json.loads((args.run / 'state.json').read_text())
        if state['status'] == 'completed':
            break
        if state['status'] == 'failed':
            raise RuntimeError('Training failed')
        time.sleep(20)
    while True:
        try:
            gpu = select_gpu(args.gpu)
            break
        except RuntimeError:
            time.sleep(10)
    root = Path(__file__).resolve().parents[1]
    python = [sys.executable, '-u']
    bundle = args.run / 'pipeline.pt'
    jobs = {
        'fixed_windows': python + [str(root / 'scripts/evaluate_parc_online_control.py'),
            '--dataset', str(args.dataset), '--bundle', str(bundle), '--output', str(args.run / 'fixed_windows'),
            '--decoder-review', '--inference-steps', '4', '--max-windows', '8', '--device', 'cuda:0'],
        'feedback': python + [str(root / 'scripts/review_parc_feedback_ablation.py'),
            '--dataset', str(args.dataset), '--bundle', str(bundle), '--output', str(args.run / 'feedback'), '--gpu', '0'],
        'contract': python + [str(root / 'scripts/review_parc_online_contract.py'),
            '--bundle', str(bundle), '--dataset', str(args.dataset), '--output', str(args.run / 'contract_review.json'), '--gpu', '0'],
        'long_source': python + [str(root / 'scripts/review_parc_online_feedback.py'),
            '--bundle', str(bundle), '--output', str(args.run / 'long_source.json'), '--scene', 'source'],
        'long_flat': python + [str(root / 'scripts/review_parc_online_feedback.py'),
            '--bundle', str(bundle), '--output', str(args.run / 'long_flat.json'), '--scene', 'flat'],
    }
    # select_gpu uses physical nvidia-smi indices, not CUDA_VISIBLE_DEVICES indices.
    for name in ('feedback', 'contract'):
        jobs[name][-1] = str(args.gpu)
    result = {'status': 'running', 'gpu_at_launch': gpu, 'jobs': {}, 'quality_approved': False}
    for name, command in jobs.items():
        print(json.dumps({'review': name, 'status': 'running'}), flush=True)
        with (args.run / (name + '.log')).open('w') as stream:
            code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
        result['jobs'][name] = {'exit_code': code}
        (args.run / 'review_state.json').write_text(json.dumps(result, indent=2))
        if code:
            raise RuntimeError(f'{name} failed; see retained log')
    result['status'] = 'completed'
    (args.run / 'review_state.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
