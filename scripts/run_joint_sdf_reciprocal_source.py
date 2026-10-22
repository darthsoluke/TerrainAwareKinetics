"""Run one source-imitation capacity probe and an unchanged closed-loop route review."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(args):
    task = Path.cwd().parent
    root = task / 'joint_sdf_20261012'
    output = root / ('reciprocal_source_' + args.arm + '_v1')
    state = task / ('joint-sdf-reciprocal-source-' + args.arm + '-state.json')
    if output.exists():
        raise ValueError('Refusing to repeat/overwrite an existing source capability test')
    def status(value):
        state.write_text(json.dumps(dict(status=value, output=str(output), quality_approved=False), indent=2))
    try:
        status('training')
        subprocess.run([sys.executable, '-u', '-m', 'scripts.pilot_joint_sdf_reciprocal_source',
            '--arm', args.arm, '--updates', '200', '--checkpoint', str(root / 'factual_prefix_aux_v1/weights.pt'),
            '--cases', str(root / 'context_pilot_data_v3/cases.pt'),
            '--accepted-recording', str(root / 'factual_prefix_aux_v1/post_review/reported/reported-stairs.npz'),
            '--output', str(output)], check=True)
        status('long_route_review')
        subprocess.run([sys.executable, '-u', '-m', 'scripts.review_joint_sdf_natural_stair_time',
            '--runs', args.arm+'='+str(output / 'weights.pt'),
            '--cases', str(root / 'context_pilot_data_v3/cases.pt'),
            '--output', str(output / 'long_review')], check=True)
        status('complete')
    except Exception:
        status('failed')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('decoder', 'coordinated'), required=True)
    main(parser.parse_args())
