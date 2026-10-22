"""Run a bounded matched whole-sequence credit-assignment probe."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(args):
    task = Path.cwd().parent
    root = task/'joint_sdf_20261012'
    output = root/('sequence_credit_'+args.arm+'_v1')
    state = task/('joint-sdf-sequence-credit-'+args.arm+'-state.json')
    if output.exists():
        raise ValueError('Existing sequence test must not be repeated')
    def status(value):
        state.write_text(json.dumps(dict(status=value, output=str(output), quality_approved=False), indent=2))
    try:
        status('training')
        subprocess.run([sys.executable, '-u', '-m', 'scripts.pilot_joint_sdf_sequence_credit',
            '--arm',args.arm,'--updates','20','--checkpoint',str(root/'reciprocal_source_coordinated_v1/weights.pt'),
            '--cases',str(root/'context_pilot_data_v3/cases.pt'),'--output',str(output)],check=True)
        status('long_route_review')
        subprocess.run([sys.executable,'-u','-m','scripts.review_joint_sdf_natural_stair_time',
            '--runs',args.arm+'='+str(output/'weights.pt'),'--cases',str(root/'context_pilot_data_v3/cases.pt'),
            '--output',str(output/'long_review')],check=True)
        status('complete')
    except Exception:
        status('failed')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('detached','through_history'),required=True)
    main(parser.parse_args())
