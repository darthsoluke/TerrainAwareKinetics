"""Private export of factual history sensitivity and causal candidate diagnostics."""

import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.train_native_mb import sha256


task = Path.cwd().parent
base = task.parent
root = task / 'joint_sdf_20261012'
output = root / 'pose_diagnostic_export_v1'
output.mkdir(exist_ok=False)
reviews = {}
labels = {}


def add(label, clip, file, description):
    directory = output / 'recordings' / label
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(file, directory / (clip + '.npz'))
    reviews[label] = directory
    labels[label] = description


for stage in ('pose_condition_v2', 'pose_components_v3', 'recovery_candidates_v1', 'rotation_recovery_v1'):
    shutil.copyfile(root / stage / 'complete.json', output / (stage + '-report.json'))
add('source', 'reported-stairs', root / 'gait_stair_v2/run-0/source.npz', 'Unchanged source')
add('actor', 'reported-stairs', root / 'pose_condition_v2/generated_history/actor.npz', 'Original continuous actor')
for arm, description in (
    ('source_root_to_pose', 'Offline: source Root plan into Pose only'),
    ('source_pose_to_pose', 'Offline: source past pose into Pose only'),
    ('source_root_and_pose_to_pose', 'Offline: source Root and past pose into Pose')):
    add(arm, 'reported-stairs', root / 'pose_condition_v2/generated_history' / (arm + '.npz'), description)
for arm, description in (
    ('source_hip_height_to_pose', 'Offline: source past height channel'),
    ('source_positions_to_pose', 'Offline: source past position channels'),
    ('source_rotations_to_pose', 'Offline: source past rotation channels')):
    add(arm, 'reported-stairs', root / 'pose_components_v3/generated_history' / (arm + '.npz'), description)
rotation_report = json.loads((root / 'rotation_recovery_v1/complete.json').read_text())
for block in rotation_report['blocks']:
    index = block['block']
    for row in block['candidates']:
        candidate = row['candidate']
        label = 'native_argmax' if candidate == 0 else f'native_sample_{candidate:02}'
        add(label, f'eight-frame-block-{index}', root / f'rotation_recovery_v1/block-{index}/candidate-{candidate:02}.npz',
            'Offline: native argmax' if candidate == 0 else f'Offline: native random candidate {candidate:02}')
subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals',
    '--stats', str(base / 'long_route_mixed_10k_ab_v1/mixed_dataset/stats'),
    '--reviews', *[name + '=' + str(path) for name, path in reviews.items()],
    '--output', str(output / 'visuals')], check=True)
manifest_path = output / 'visuals/manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest['labels'].update(labels)
manifest['diagnostic_scope'] = ('Partial pose-channel substitutions are inconsistent-input sensitivity tests, '
    'not recovery poses. Native random candidates are separate8-frame proposals, not continuous actors; '
    'all68 are retained, with complete metrics and no admitted labels or quality approval.')
manifest_path.write_text(json.dumps(manifest, separators=(',', ':'), allow_nan=False))
receipt = dict(quality_approved=False, optimizer_updates=0,
    files={str(path.relative_to(output)): sha256(path) for path in output.rglob('*') if path.is_file()})
(output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
config = base / 'drive-auth/private-client.conf'
settings = configparser.ConfigParser(interpolation=None)
settings.read(config)
if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
    raise ValueError('User-owned OAuth client mismatch')
environment = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='',
    http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
subprocess.run(['rclone', '--config', str(config), 'copy', str(output),
    'private-store:motionbricks/joint_sdf_20261012/' + output.name, '--immutable', '--checksum',
    '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'], env=environment, check=True)
print(json.dumps(dict(status='uploaded', files=len(receipt['files']), quality_approved=False)), flush=True)
