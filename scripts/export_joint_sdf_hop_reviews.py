"""Export unmodified hopping diagnostics through the user's private Drive."""

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
output = root / 'hop_export_v1'
output.mkdir(exist_ok=False)
labels = {}


def add(label, name, path):
    directory = output / 'recordings' / label
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, directory / (name + '.npz'))
    labels[label] = directory


for directory, index, name in (('gait_stair_v2', 0, 'reported-stairs'),
                               ('gait_flat_v2', 0, 'counterfactual-flat'),
                               ('gait_chain_v1', 0, 'bones-run-0'),
                               ('gait_chain_v1', 1, 'bones-run-1')):
    audit = root / directory
    if not (audit / 'complete.json').exists():
        raise ValueError('Incomplete diagnostic')
    raw = audit / f'run-{index}'
    add('actor', name, raw / 'generated_history-actor.npz')
    if directory != 'gait_flat_v2':
        add('source', name, raw / 'source.npz')
    add('offline_source_history', name, raw / 'source_history-actor.npz')
    add('offline_source_tokens', name, raw / 'generated_history-source_tokens.npz')
    add('offline_source_root', name, raw / 'generated_history-source_root.npz')
    shutil.copyfile(audit / 'complete.json', output / (directory + '-report.json'))
for label, directory in (('matched_control', 'control'), ('history_root_loss', 'history_root')):
    add(label, 'reported-stairs', root / 'reported_hop_compare_v1' / directory / 'reported-stairs.npz')
for path, name in ((root / 'bilateral_hop_audit_v2/complete.json', 'original-gait-report.json'),
                   (root / 'reported_hop_compare_v1/complete.json', 'matched-comparison.json'),
                   (root / 'decoded_history_root_v1/complete.json', 'history-root-trial.json'),
                   (root / 'decoded_history_root_v1/official_start_review/complete.json', 'history-root-official-review.json')):
    shutil.copyfile(path, output / name)
subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals',
    '--stats', str(base / 'long_route_mixed_10k_ab_v1/mixed_dataset/stats'),
    '--reviews', *[key + '=' + str(value) for key, value in labels.items()],
    '--output', str(output / 'visuals')], check=True)
manifest_path = output / 'visuals/manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest['labels'].update(offline_source_history='Offline: source history each window',
    offline_source_tokens='Offline: source tokens, actor history and Root',
    offline_source_root='Offline: source Root, actor history and tokens',
    actor='Original actor, continuous feedback', source='Unchanged source',
    matched_control='Matched source-history training control', history_root_loss='Observed decoder Root loss trial')
manifest['diagnostic_scope'] = 'Offline arms are independent windows, never live actors. Flat counterfactual has no matched flat source.'
manifest_path.write_text(json.dumps(manifest, separators=(',', ':'), allow_nan=False))
receipt = dict(quality_approved=False, postprocessing=None,
    files={str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()})
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
