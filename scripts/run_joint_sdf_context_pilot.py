"""Supervise one bounded full-context arm and the retained official-stair review."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('source', 'feedback'), required=True)
    args = parser.parse_args()
    task = Path.cwd().parent
    root = task / 'joint_sdf_20261012'
    output = root / ('context_' + args.arm + '_v1')
    state_file = task / ('joint-sdf-context-' + args.arm + '-state.json')
    if output.exists():
        raise ValueError('Run already exists; inspect before attempting any restart')

    def status(value, **details):
        state_file.write_text(json.dumps({'status': value, 'arm': args.arm, 'output': str(output),
                                         'quality_approved': False, **details}, indent=2))

    try:
        command = [sys.executable, '-u', '-m', 'scripts.pilot_joint_sdf_full_context',
                   '--arm', args.arm, '--checkpoint', str(root / 'decoder_ab_A/weights.pt'),
                   '--cases', str(root / 'context_pilot_data_v3/cases.pt'),
                   '--output', str(output), '--updates', '300']
        status('baseline_preflight_train_and_review', command=command, update_limit=300, example_limit=1200)
        subprocess.run(command, check=True)
        command = [sys.executable, '-u', '-m', 'scripts.review_joint_sdf_stair_control',
                   '--runs', args.arm + '=' + str(output / 'weights.pt'),
                   '--cases', str(root / 'decoder_ab_A/cases.pt'), '--output', str(output / 'official_start_review')]
        status('official_stair_review', command=command)
        subprocess.run(command, check=True)
        status('complete', actual_updates=300, primitive_examples=1200)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
