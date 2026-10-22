"""Publish clearly labeled source-history diagnostics through owned private Drive."""
import argparse
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np

from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((a.run / 'complete.json').read_text())
    if report['optimizer_updates'] or not report['weights_unchanged']:
        raise ValueError('Expected completed read-only diagnosis')
    labels = dict(source='Real source replay', actor='Model from recorded history',
        offline_source_root='OFFLINE: source Root', offline_source_tokens='OFFLINE: source tokens',
        offline_source_root_tokens='OFFLINE: source Root and tokens')
    reviews = []
    for arm in labels:
        folder = a.output / 'recordings' / arm
        folder.mkdir(parents=True)
        for path in sorted((a.run / 'recordings' / arm).glob('*.npz')):
            with np.load(path, allow_pickle=False) as data:
                values = {key: data[key].copy() for key in data.files}
            metadata = json.loads(str(values['metadata_json']))
            if arm != 'source':
                with np.load(a.run / 'recordings/source' / path.name, allow_pickle=False) as source:
                    for key in ('local_rot_mats', 'roots'):
                        future = values[key][4:].copy()
                        # Display the factual played prefix; never publish reconstructed past frames.
                        values[key][:4] = source[key][:4]
                        np.testing.assert_array_equal(values[key][4:], future)
                metadata['history_display'] = 'four actual source-history frames; future output unchanged'
            metadata['role_description'] = labels[arm]
            values['metadata_json'] = np.array(json.dumps(metadata))
            np.savez_compressed(folder / path.name, **values)
        reviews.append(arm + '=' + str(folder))
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', str(a.stats),
        '--reviews', *reviews, '--output', str(a.output / 'visuals')], check=True)
    manifest = json.loads((a.output / 'visuals/manifest.json').read_text())
    for clip in manifest['clips'].values():
        for arm in clip['arms'].values():
            if len(arm['roots']) != clip['frames']:
                raise ValueError('Independent primitive frame count mismatch')
            arm['frames'] = clip['frames']
            arm['duration_seconds'] = clip['frames'] / 30
    manifest.update(labels=labels, quality_approved=False,
        scope='28 real-time frames per independent primitive. Four factual source-history frames then24generated '
        'frames. Source Root/token oracles use future truth OFFLINE ONLY. No WASD or full-route acceptance.')
    write_json(a.output / 'visuals/manifest.json', manifest)
    shutil.copyfile(a.run / 'complete.json', a.output / 'diagnostic-report.json')
    files = {str(p.relative_to(a.output)): sha256(p) for p in a.output.rglob('*') if p.is_file()}
    write_json(a.output / 'receipt.json', dict(files=files, frame_padding=False, quality_approved=False))
    archive = a.output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(a.output, arcname=a.output.name)
    receipt = archive.with_suffix('.receipt.json')
    write_json(receipt, dict(filename=archive.name, sha256=sha256(archive), bytes=archive.stat().st_size, files=len(files)))
    settings = configparser.ConfigParser(interpolation=None); settings.read(a.drive_config)
    if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('Owned private OAuth client required')
    env = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
    for path in (archive, receipt):
        subprocess.run(['rclone', '--config', str(a.drive_config), 'copyto', str(path),
            'private-store:motionbricks/parc_rugged32_20261013_v2/' + path.name, '--immutable', '--checksum',
            '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'], env=env, check=True)
    print(json.dumps(dict(status='uploaded', sha256=sha256(archive), files=len(files))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'stats', 'output', 'drive-config'):
        p.add_argument('--' + key, type=Path, required=True)
    main(p.parse_args())
