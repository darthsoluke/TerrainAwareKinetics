"""Evaluate the reported clip with matched control, history objective and a flat scene."""

import json
from pathlib import Path
import subprocess
import sys

task = Path.cwd().parent
root = task / 'joint_sdf_20261012'
output = root / 'reported_hop_compare_v1'
output.mkdir(exist_ok=False)
state = json.loads((task / 'joint-sdf-decoded-history-root-state.json').read_text())
if state['status'] != 'complete':
    raise ValueError('The bounded trial and its required reviews must finish first')
reviews = {}
for label, checkpoint, flat in (
        ('control', root / 'context_source_v1/weights.pt', False),
        ('history_root', root / 'decoded_history_root_v1/weights.pt', False),
        ('flat', root / 'decoder_ab_A/weights.pt', True)):
    command = [sys.executable, '-u', str(task / 'audit_joint_sdf_bilateral_hops_review.py'),
               '--checkpoint', str(checkpoint), '--cases', str(root / 'context_pilot_data_v3/cases.pt'),
               '--recording', str(root / 'source_context_v2/constant_approach/generated_8030ae7a2a7ad640.npz'),
               '--output', str(output / label)]
    if flat:
        command.append('--flat-scene')
    subprocess.run(command, check=True)
    reviews[label] = json.loads((output / label / 'complete.json').read_text())
original = json.loads((root / 'bilateral_hop_audit_v2/complete.json').read_text())
reviews['original'] = original
summary = {}
for label, report in reviews.items():
    summary[label] = {key: report.get(key) for key in ('generated', 'failure', 'frames', 'counterfactual_flat_scene')}
    offsets = [row['decoded_past_xz_offset_norm_m'] for row in report['blocks']]
    summary[label]['past_offset_mean_m'] = sum(offsets) / len(offsets)
    summary[label]['past_offset_max_m'] = max(offsets)
    feet = report['series']['generated']['foot_foreaft_relative_m'][-32:]
    summary[label]['last32_close_feet_fraction'] = sum(abs(left - right) < .1 for left, right in feet) / len(feet)
    summary[label]['last32_foot_foreaft_separation_rms_m'] = (sum((left - right) ** 2 for left, right in feet) / len(feet)) ** .5
result = {'summary': summary, 'quality_approved': False,
          'scope': 'Reported constant-input route at its original80-frame horizon. Flat is a counterfactual '
                   'measured-height scene with identical initial body and command, not a paired flat source demonstration.'}
(output / 'complete.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
