"""Finish the already authorized small experiment and publish private reports."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_terrain_decoder_pilot import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--drive-config', type=Path, required=True)
    args = p.parse_args()
    state = {'status': 'waiting_for_pilots', 'pid': os.getpid()}
    status_path = args.base/'finish.json'
    save(status_path, state)
    try:
        while True:
            run = json.loads((args.base/'runs/state.json').read_text())
            if run['status'] == 'failed':
                raise RuntimeError('A pilot failed; see its saved log')
            if run['status'] == 'completed_review_required':
                break
            time.sleep(10)
        state['status'] = 'summarizing_and_rendering';save(status_path, state)
        with (args.base/'summary.log').open('w') as log:
            subprocess.run([sys.executable, '-u', '-m', 'scripts.summarize_terrain_decoder_pilot',
                '--runs', str(args.base/'runs'), '--output', str(args.base/'comparison'), '--render'],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        state['status'] = 'exporting_private_reports';save(status_path, state)
        env = dict(os.environ, HTTP_PROXY='http://127.0.0.1:27890', HTTPS_PROXY='http://127.0.0.1:27890')
        with (args.base/'report_upload.log').open('w') as log:
            subprocess.run(['rclone', '--config', str(args.drive_config), 'copy', str(args.base),
                'mb-transfer:pilots/terrain-decoder-20261010/results',
                '--include', '/comparison/**', '--include', '/runs/*/contract.json',
                '--include', '/runs/*/training_result.json', '--include', '/runs/*/review/metrics.json',
                '--include', '/runs/baseline/initial_review/metrics.json', '--include', '/runs/state.json',
                '--immutable'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['status'] = 'completed_review_required';save(status_path, state)
    except Exception as error:
        state.update(status='failed', error=type(error).__name__+': '+str(error));save(status_path, state)
        raise


if __name__ == '__main__':
    main()
