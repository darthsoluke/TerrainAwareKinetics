"""Bounded factual-Root normalization probe with unchanged runtime and long review."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(args):
    task = Path.cwd().parent
    root = task / 'joint_sdf_20261012'
    output = root / ('contact_factual_' + args.arm + '_v6')
    state = task / ('joint-sdf-contact-factual-' + args.arm + '-state.json')
    if output.exists():
        raise ValueError('Never repeat or overwrite an existing bounded trial')
    def status(value):
        state.write_text(json.dumps(dict(status=value, output=str(output), quality_approved=False), indent=2))
    command = [sys.executable, '-u', '-m', 'scripts.pilot_joint_sdf_contact_preserve',
        '--arm', 'contact', '--phase-support', '--separate-factual-root', '--updates', '200',
        '--reference-trial', str(root / ('contact_support_v3' if args.arm == 'support' else 'contact_direction_v5') / 'complete.json'),
        '--checkpoint', str(root / 'factual_prefix_aux_v1/weights.pt'),
        '--cases', str(root / 'context_pilot_data_v3/cases.pt'),
        '--accepted-recording', str(root / 'factual_prefix_aux_v1/post_review/reported/reported-stairs.npz'),
        '--output', str(output)]
    if args.arm == 'direction':
        command.append('--direction-only-preservation')
    try:
        status('training')
        subprocess.run(command, check=True)
        status('long_route_review')
        subprocess.run([sys.executable, '-u', '-m', 'scripts.review_joint_sdf_natural_stair_time',
            '--runs', args.arm + '=' + str(output / 'weights.pt'),
            '--cases', str(root / 'context_pilot_data_v3/cases.pt'),
            '--output', str(output / 'long_review')], check=True)
        status('complete')
    except Exception:
        status('failed')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('support', 'direction'), required=True)
    main(parser.parse_args())
