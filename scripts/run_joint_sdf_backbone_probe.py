"""Locate decision-making limits by unfreezing one native MB backbone at a time."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm',choices=('root','pose'),required=True)
args=parser.parse_args();task=Path.cwd().parent;root=task/'joint_sdf_20261012'
output=root/('backbone_'+args.arm+'_v1');state_file=task/('joint-sdf-backbone-'+args.arm+'-state.json')
if output.exists():raise ValueError('Experiment exists; inspect before restart')
def status(value,**data):
    state_file.write_text(json.dumps({'status':value,'arm':args.arm,'output':str(output),'quality_approved':False,**data},indent=2))
try:
    command=[sys.executable,'-u',str(task/'pilot_joint_sdf_backbone.py'),
        '--checkpoint',str(root/'train_1600k_v1/last.pt'),'--output',str(output),'--updates','300',
        '--unfreeze-'+args.arm,'--reference-baseline',str(root/'decoder_ab_A/complete.json')]
    status('training',command=command);subprocess.run(command,check=True)
    command=[sys.executable,'-u',str(task/'review_joint_sdf_stair_control.py'),
        '--runs',args.arm+'='+str(output/'weights.pt'),'--cases',str(output/'cases.pt'),
        '--output',str(output/'start_review')]
    status('review',command=command);subprocess.run(command,check=True)
    status('complete',actual_updates=300,primitive_examples=3600,
        frozen_reference='decoder_ab_A: same initialization, seed, cases, loss, budget; baseline parity checked')
except Exception as error:
    status('failed',error=str(error));raise
