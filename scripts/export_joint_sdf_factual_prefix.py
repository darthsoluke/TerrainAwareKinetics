"""Deliver all bounded prefix results privately, without promoting their weights."""

import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from scripts.train_native_mb import sha256


task = Path.cwd().parent
base = task.parent
root = task / 'joint_sdf_20261012'
output = root / 'factual_prefix_export_v1'
output.mkdir(exist_ok=False)
reviews = {}


def add(label, clip, file):
    directory = output / 'recordings' / label
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(file, directory / (clip + '.npz'))
    reviews[label] = directory


for label, run_name in (('control', 'factual_prefix_control_v1'), ('prefix', 'factual_prefix_aux_v1')):
    run = root / run_name
    if json.loads((run / 'post_review/state.json').read_text())['status'] != 'complete':
        raise ValueError('Post reviews are incomplete')
    for name in ('complete', 'contract', 'preflight', 'training'):
        shutil.copyfile(run / (name + '.json'), output / (label + '-' + name + '.json'))
    shutil.copyfile(run / 'post_review/complete.json', output / (label + '-post-review.json'))
    add(label, 'reported-stairs', run / 'post_review/reported/reported-stairs.npz')
    add(label, 'counterfactual-flat', run / 'post_review/counterfactual_flat/reported-stairs.npz')
    for index in (0, 1):
        add(label, 'bones-run-' + str(index), run / f'post_review/bones_runs/run-{index}/generated_history-actor.npz')
    for category in ('idle', 'walk'):
        add(label, category, run / f'final/{category}-flat-104f.npz')
add('initial', 'reported-stairs', root / 'gait_stair_v2/run-0/generated_history-actor.npz')
add('source', 'reported-stairs', root / 'gait_stair_v2/run-0/source.npz')
add('initial', 'counterfactual-flat', root / 'reported_hop_compare_v1/flat/reported-stairs.npz')
for index in (0, 1):
    add('initial', 'bones-run-' + str(index), root / f'gait_chain_v1/run-{index}/generated_history-actor.npz')
    add('source', 'bones-run-' + str(index), root / f'gait_chain_v1/run-{index}/source.npz')
for category in ('idle', 'walk'):
    add('initial', category, root / f'factual_prefix_control_v1/baseline/{category}-flat-104f.npz')
stats = str(base / 'long_route_mixed_10k_ab_v1/mixed_dataset/stats')
subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', stats,
    '--reviews', *[label + '=' + str(path) for label, path in reviews.items()], '--output', str(output / 'visuals')], check=True)
subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', stats,
    '--reviews', 'control=' + str(root / 'factual_prefix_control_v1/final'),
    'prefix=' + str(root / 'factual_prefix_aux_v1/final'), '--output', str(output / 'full_review_visuals')], check=True)
manifest_path = output / 'visuals/manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest['labels'].update(initial='Initial actor', control='Source reconstruction control',
    prefix='Source reconstruction plus factual prefix', source='Unchanged source')
manifest['quality_approved'] = False
manifest_path.write_text(json.dumps(manifest, separators=(',', ':'), allow_nan=False))
receipt = dict(quality_approved=False, postprocessing=None,
    checkpoint_sha256={label: sha256(root / name / 'weights.pt') for label, name in
                       (('control', 'factual_prefix_control_v1'), ('prefix', 'factual_prefix_aux_v1'))},
    files={str(path.relative_to(output)): sha256(path) for path in output.rglob('*') if path.is_file()})
(output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
archive = output.with_suffix('.tar.gz')
with tarfile.open(archive, 'w:gz') as stream:
    stream.add(output, arcname=output.name)
archive_receipt = archive.with_suffix('.receipt.json')
archive_receipt.write_text(json.dumps(dict(filename=archive.name, sha256=sha256(archive),
    bytes=archive.stat().st_size, receipt_files=len(receipt['files'])), indent=2))
config = base / 'drive-auth/private-client.conf'
settings = configparser.ConfigParser(interpolation=None)
settings.read(config)
if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
    raise ValueError('User-owned OAuth client mismatch')
environment = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='',
    http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
for file in (archive, archive_receipt):
    subprocess.run(['rclone', '--config', str(config), 'copyto', str(file),
        'private-store:motionbricks/joint_sdf_20261012/' + file.name, '--immutable', '--checksum',
        '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'], env=environment, check=True)
print(json.dumps(dict(status='uploaded', archive=str(archive), files=len(receipt['files']), quality_approved=False)), flush=True)
