"""Matched200-update source-contact XYZ probe, starting from the same1000-step model."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(args):
    task=Path.cwd().parent
    root=task/'joint_sdf_20261012'
    output=root/('source_support_'+args.arm+'_v1')
    state=task/('joint-sdf-source-support-'+args.arm+'-state.json')
    if output.exists():
        raise ValueError('Refusing to overwrite an existing matched probe')
    def status(value):
        state.write_text(json.dumps(dict(status=value, output=str(output), quality_approved=False),indent=2))
    try:
        status('training')
        command=[sys.executable,'-u','-m','scripts.pilot_joint_sdf_reciprocal_source',
            '--arm','coordinated','--updates','200','--checkpoint',str(root/'factual_prefix_aux_v1/weights.pt'),
            '--resume-from',str(root/'reciprocal_source_coordinated_1000_v1/weights.pt'),
            '--cases',str(root/'context_pilot_data_v3/cases.pt'),
            '--accepted-recording',str(root/'reciprocal_source_coordinated_1000_v1/final/generated_8030ae7a2a7ad640.npz'),
            '--output',str(output)]
        if args.arm=='xyz':
            command.append('--support-position')
        subprocess.run(command,check=True)
        status('conditional_fit_review')
        subprocess.run([sys.executable,'-u','-m','scripts.review_joint_sdf_source_fit',
            '--checkpoint',str(output/'weights.pt'),'--cases',str(root/'context_pilot_data_v3/cases.pt'),
            '--output',str(output/'source_fit')],check=True)
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
    parser.add_argument('--arm',choices=('control','xyz'),required=True)
    main(parser.parse_args())
