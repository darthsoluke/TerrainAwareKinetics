"""Matched generated-only versus native source+generated decoder objectives."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm',choices=('control','auxiliary'),required=True)
args=parser.parse_args();task=Path.cwd().parent;root=task/'joint_sdf_20261012'
output=root/('sourcecodec_'+args.arm+'_v1');state_file=task/('joint-sdf-sourcecodec-'+args.arm+'-state.json')
if output.exists():raise ValueError('Experiment exists; do not duplicate it')
def status(value,**data):
    state_file.write_text(json.dumps({'status':value,'arm':args.arm,'output':str(output),'quality_approved':False,**data},indent=2))
try:
    command=[sys.executable,'-u',str(task/'pilot_joint_sdf_sourcecodec.py'),
        '--checkpoint',str(root/'train_1600k_v1/last.pt'),'--output',str(output),'--updates','200','--unfreeze-decoder']
    if args.arm=='auxiliary':command.append('--native-codec-reconstruction')
    status('training',command=command);subprocess.run(command,check=True)
    command=[sys.executable,'-u',str(task/'review_joint_sdf_stair_control.py'),
        '--runs',args.arm+'='+str(output/'weights.pt'),'--cases',str(output/'cases.pt'),
        '--output',str(output/'start_review')]
    status('review',command=command);subprocess.run(command,check=True)
    status('complete',actual_updates=200,primitive_examples=2400,
        auxiliary_reconstruction_examples=2400 if args.arm=='auxiliary' else 0)
except Exception as error:
    status('failed',error=str(error));raise
