"""Export unequal-length route recordings without padding or playback retiming."""

import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np

from scripts.train_native_mb import sha256


task = Path.cwd().parent
base = task.parent
root = task / 'joint_sdf_20261012'
prior = root / 'natural_stair_time_v1'
current = root / 'natural_stair_time_v5'
reports = [json.loads((directory / 'complete.json').read_text()) for directory in (prior, current)]
output = root / 'natural_contact_comparison_v5'
output.mkdir(exist_ok=False)
directories = {arm: prior / arm for arm in ('before', 'slower', 'root_preserve')}
directories['direction_only'] = current / 'direction_only'
subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals',
    '--stats', str(base / 'long_route_mixed_10k_ab_v1/mixed_dataset/stats'),
    '--reviews', *[arm + '=' + str(path) for arm, path in directories.items()],
    '--output', str(output / 'visuals')], check=True)
manifest_path = output / 'visuals/manifest.json'
old = json.loads(manifest_path.read_text())
merged = dict(fps=30, postprocessing=None, clips={}, audit=old['audit'],
    labels={'before': 'Before: accepted gait', 'slower': 'Slower contact candidate',
            'root_preserve': 'Speed-preserving candidate', 'direction_only': 'Direction-only contact candidate'},
    scope='Each arm retains its own original frame count. Finished panels display their last frame with Ended status; '
          'the source arrays are never extended or retimed. The common clock is only a viewer clock.',
    quality_approved=False)
source_map = {'generated_8030ae7a2a7ad640': 'reported-stairs',
              'generated_415ccb3cd0a84f27': 'stairs-415', 'generated_babca93468f737e6': 'stairs-babca'}
all_rows = [row for report in reports for row in report['rows']]
for row in all_rows:
    stem = f'{row["label"]}-{row["arm"]}-{row["frames"]}f'
    entry = dict(old['clips'][stem]['arms'][row['arm']], frames=row['frames'], fps=30,
                 duration_seconds=row['frames'] / 30)
    if len(entry['roots']) != row['frames']:
        raise ValueError('Unequal-duration export changed an original recording length')
    name = source_map[row['label']]
    clip = merged['clips'].setdefault(name, dict(frames=0, fps=30, arms={}, source_id=row['label']))
    clip['frames'] = max(clip['frames'], row['frames'])
    clip['arms'][row['arm']] = entry
    raw_dir = output / 'recordings' / row['arm']
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(directories[row['arm']] / (stem + '.npz'), raw_dir / (stem + '.npz'))
for clip in merged['clips'].values():
    row = next(row for row in all_rows if row['label'] == clip['source_id'])
    file = directories[row['arm']] / f'{row["label"]}-{row["arm"]}-{row["frames"]}f.npz'
    with np.load(file, allow_pickle=False) as archive:
        hf = archive['heightfield_y_m'].copy()
        origin = archive['min_point_world_xyz_m'].copy()
        dx = float(archive['dx_m'])
    roots = np.concatenate([entry['roots'] for entry in clip['arms'].values()])
    lo, hi = roots[:, [0, 2]].min(0) - 5, roots[:, [0, 2]].max(0) + 5
    x0, z0 = np.maximum(0, np.floor((lo - origin[[0, 2]]) / dx).astype(int))
    x1, z1 = np.minimum([hf.shape[1], hf.shape[0]], np.ceil((hi - origin[[0, 2]]) / dx).astype(int) + 1)
    clip['scene'] = dict(heightfield_y_m=[[float(value) if np.isfinite(value) else None for value in line]
                                         for line in hf[z0:z1, x0:x1]],
                         origin=(origin + np.array([x0 * dx, 0., z0 * dx])).tolist(), dx=dx)
manifest_path.write_text(json.dumps(merged, separators=(',', ':'), allow_nan=False))
for name, directory in (('timing-before-v3-v4', prior), ('timing-direction-v5', current)):
    shutil.copyfile(directory / 'complete.json', output / (name + '.json'))
for label, name in (('v3', 'contact_support_v3'), ('v4', 'contact_support_v4'), ('v5', 'contact_direction_v5')):
    for artifact in ('complete', 'contract', 'preflight', 'training'):
        shutil.copyfile(root / name / (artifact + '.json'), output / f'{label}-{artifact}.json')
receipt = dict(quality_approved=False, frame_padding=False, playback_retiming=False,
    files={str(path.relative_to(output)): sha256(path) for path in output.rglob('*') if path.is_file()})
(output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
archive = output.with_suffix('.tar.gz')
with tarfile.open(archive, 'w:gz') as stream:
    stream.add(output, arcname=output.name)
archive_receipt = archive.with_suffix('.receipt.json')
archive_receipt.write_text(json.dumps(dict(filename=archive.name, sha256=sha256(archive),
    bytes=archive.stat().st_size, files=len(receipt['files'])), indent=2))
settings = configparser.ConfigParser(interpolation=None)
config = base / 'drive-auth/private-client.conf'
settings.read(config)
if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
    raise ValueError('User-owned OAuth client mismatch')
environment = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='',
                   http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
for path in (archive, archive_receipt):
    subprocess.run(['rclone', '--config', str(config), 'copyto', str(path),
        'private-store:motionbricks/joint_sdf_20261012/' + path.name, '--immutable', '--checksum',
        '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'], env=environment, check=True)
print(json.dumps(dict(status='uploaded', files=len(receipt['files']), quality_approved=False)), flush=True)
