"""Bounded decoder-history supervision trial against an archived matched control."""

import json
from pathlib import Path
import subprocess
import sys


task = Path.cwd().parent
root = task / 'joint_sdf_20261012'
output = root / 'decoded_history_root_v1'
state_file = task / 'joint-sdf-decoded-history-root-state.json'
if output.exists():
    raise ValueError('History-root trial already exists; inspect before restarting')


def status(value, **details):
    state_file.write_text(json.dumps({'status': value, 'output': str(output),
                                     'quality_approved': False, **details}, indent=2))


try:
    command = [sys.executable, '-u', '-m', 'scripts.pilot_joint_sdf_full_context',
               '--arm', 'source', '--checkpoint', str(root / 'decoder_ab_A/weights.pt'),
               '--cases', str(root / 'context_pilot_data_v3/cases.pt'),
               '--output', str(output), '--updates', '300', '--decoded-history-root',
               '--reference-control', str(root / 'context_source_v1/complete.json')]
    status('baseline_preflight_train_and_review', command=command, actual_update_limit=300, exposure_limit=1200)
    subprocess.run(command, check=True)
    command = [sys.executable, '-u', '-m', 'scripts.review_joint_sdf_stair_control',
               '--runs', 'history_root=' + str(output / 'weights.pt'),
               '--cases', str(root / 'decoder_ab_A/cases.pt'), '--output', str(output / 'official_start_review')]
    status('official_stair_review', command=command)
    subprocess.run(command, check=True)
    status('complete', actual_updates=300, primitive_examples=1200,
           matched_control=str(root / 'context_source_v1'), inference_graph_unchanged=True)
except Exception as error:
    status('failed', error=str(error))
    raise
