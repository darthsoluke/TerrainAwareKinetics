"""Finish the bounded factual-prefix trial without promoting a failed candidate."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.summarize_joint_sdf_context_pair import assert_same
from scripts.train_joint_terrain_v3 import write_json


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm', choices=('control', 'aux'), required=True)
args = parser.parse_args()
task = Path.cwd().parent
root = task / 'joint_sdf_20261012'
run = root / ('factual_prefix_' + args.arm + '_v1')
output = run / 'post_review'
output.mkdir(exist_ok=False)
pair = [json.loads((root / ('factual_prefix_' + arm + '_v1/complete.json')).read_text()) for arm in ('control', 'aux')]
if pair[0]['factual_history_bank_sha256'] != pair[1]['factual_history_bank_sha256'] or pair[0]['schedule_sha256'] != pair[1]['schedule_sha256']:
    raise ValueError('Paired history/exposure schedules differ')
assert_same(pair[0]['baseline'], pair[1]['baseline'])
write_json(output / 'pair_preflight.json', dict(baseline_match=True, bank_match=True, schedule_match=True,
    actual_updates_per_arm=100, actor_prefix_exposures_per_arm=400, source_codec_examples_per_arm=400))


def execute(label, command):
    write_json(output / 'state.json', dict(status=label, command=command, quality_approved=False))
    subprocess.run(command, check=True)


python = sys.executable
checkpoint = str(run / 'weights.pt')
common = ['--checkpoint', checkpoint]
execute('official_stair', [python, '-u', '-m', 'scripts.review_joint_sdf_stair_control',
    '--runs', args.arm + '=' + checkpoint, '--cases', str(root / 'decoder_ab_A/cases.pt'),
    '--output', str(output / 'official')])
execute('bones_runs', [python, '-u', str(task / 'audit_joint_sdf_gait_chain_v2.py'), *common,
    '--output', str(output / 'bones_runs')])
for label, flags in (('reported', []), ('counterfactual_flat', ['--flat-scene'])):
    execute(label, [python, '-u', str(task / 'audit_joint_sdf_bilateral_hops_review.py'), *common,
        '--cases', str(root / 'context_pilot_data_v3/cases.pt'),
        '--recording', str(root / 'source_context_v2/constant_approach/generated_8030ae7a2a7ad640.npz'),
        '--output', str(output / label), *flags])
execute('rotation_boundary_recording', [python, '-u', str(task / 'audit_joint_sdf_rotation_recovery_v1.py'), *common,
    '--cases', str(root / 'context_pilot_data_v3/cases.pt'), '--recording', str(output / 'reported/reported-stairs.npz'),
    '--pose-helper', str(task / 'audit_joint_sdf_pose_conditions_v3.py'),
    '--gait-helper', str(task / 'audit_joint_sdf_gait_chain_v2.py'),
    '--recovery-helper', str(task / 'audit_joint_sdf_recovery_candidates_v1.py'),
    '--samples', '0', '--output', str(output / 'rotation_recording')])
execute('rotation_boundary_metrics', [python, '-u', str(task / 'audit_joint_sdf_rotation_boundary_v1.py'),
    '--stats', str(task.parent / 'long_route_mixed_10k_ab_v1/mixed_dataset/stats'),
    '--helper', str(task / 'audit_joint_sdf_rotation_recovery_v1.py'),
    '--input', str(output / 'rotation_recording'), '--output', str(output / 'rotation_boundary')])
reports = {name: json.loads((output / name / 'complete.json').read_text()) for name in
    ('official', 'bones_runs', 'reported', 'counterfactual_flat', 'rotation_recording', 'rotation_boundary')}
write_json(output / 'complete.json', dict(reports=reports, quality_approved=False))
write_json(output / 'state.json', dict(status='complete', quality_approved=False))
