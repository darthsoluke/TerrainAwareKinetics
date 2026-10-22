"""Private before/after package; rejected floating outputs stay explicitly labelled."""

import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from scripts.summarize_joint_sdf_context_pair import assert_same
from scripts.train_native_mb import sha256


task = Path.cwd().parent
base = task.parent
root = task / 'joint_sdf_20261012'
runs = dict(control=root / 'contact_preserve_control_v2',
            rejected_float=root / 'contact_preserve_skin_v2', after=root / 'contact_support_v3')
reports = {label: json.loads((run / 'complete.json').read_text()) for label, run in runs.items()}
assert_same(reports['control']['baseline'], reports['after']['baseline'])
assert_same(reports['rejected_float']['baseline'], reports['after']['baseline'])
for value in reports.values():
    if value['actual_updates'] != 200 or value['schedule_sha256'] != reports['after']['schedule_sha256'] or value['exposure'] != reports['after']['exposure']:
        raise ValueError('Matched budgets/source/command schedules differ')
output = root / 'contact_preserve_export_v3'
output.mkdir(exist_ok=False)
labels = {'before': 'Before: accepted stair gait', 'control': 'Matched training control',
          'rejected_float': 'Rejected: collision-only floating', 'after': 'After: contact and weak support', 'source': 'Unchanged source'}
reviews = ['before=' + str(runs['after'] / 'baseline')]
for label, run in runs.items():
    reviews.append(label + '=' + str(run / 'final'))
    for name in ('complete', 'contract', 'preflight', 'training'):
        shutil.copyfile(run / (name + '.json'), output / (label + '-' + name + '.json'))
    raw = output / 'recordings' / label
    shutil.copytree(run / 'final', raw)
shutil.copytree(runs['after'] / 'baseline', output / 'recordings/before')
stats = str(base / 'long_route_mixed_10k_ab_v1/mixed_dataset/stats')
subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', stats,
    '--reviews', *reviews, '--output', str(output / 'visuals')], check=True)
subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', stats,
    '--reviews', 'after=' + str(runs['after'] / 'varied'),
    '--output', str(output / 'varied_visuals')], check=True)
manifest_path = output / 'visuals/manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest['labels'].update(labels)
manifest['quality_approved'] = False
manifest['scope'] = 'Same commands, original durations, no IK/projection/root replacement/smoothing. Quality and floating tests are reported separately.'
manifest_path.write_text(json.dumps(manifest, separators=(',', ':'), allow_nan=False))
receipt = dict(quality_approved=False, postprocessing=None, source_schedule_match=True,
    checkpoint_sha256={label: sha256(run / 'weights.pt') for label, run in runs.items()},
    files={str(path.relative_to(output)): sha256(path) for path in output.rglob('*') if path.is_file()})
(output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
archive = output.with_suffix('.tar.gz')
with tarfile.open(archive, 'w:gz') as stream:
    stream.add(output, arcname=output.name)
archive_receipt = archive.with_suffix('.receipt.json')
archive_receipt.write_text(json.dumps(dict(filename=archive.name, sha256=sha256(archive),
    bytes=archive.stat().st_size, files=len(receipt['files'])), indent=2))
config = base / 'drive-auth/private-client.conf'
settings = configparser.ConfigParser(interpolation=None)
settings.read(config)
if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
    raise ValueError('User-owned OAuth client mismatch')
environment = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='',
                   http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
for path in (archive, archive_receipt):
    subprocess.run(['rclone', '--config', str(config), 'copyto', str(path),
        'private-store:motionbricks/joint_sdf_20261012/' + path.name, '--immutable', '--checksum',
        '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'], env=environment, check=True)
print(json.dumps(dict(status='uploaded', archive=str(archive), files=len(receipt['files']), quality_approved=False)), flush=True)
