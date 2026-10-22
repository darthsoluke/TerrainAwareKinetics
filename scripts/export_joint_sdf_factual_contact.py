"""Package original-clock factual-Root contact probes and their archived controls."""

import configparser
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np

from scripts.train_native_mb import sha256


def main(mode='factual'):
    task = Path.cwd().parent
    base, root = task.parent, task / 'joint_sdf_20261012'
    reports = {
        'before': (root / 'natural_stair_time_v1', 'before'),
        'slower': (root / 'natural_stair_time_v1', 'slower'),
        'direction_control': (root / 'natural_stair_time_v5', 'direction_only'),
        'factual_support': (root / 'contact_factual_support_v6/long_review', 'support'),
        'factual_direction': (root / 'contact_factual_direction_v6/long_review', 'direction'),
    }
    labels = dict(before='Accepted gait baseline', slower='Before: backward-seam failure',
                  direction_control='Direction-only control', factual_support='Factual-history repair',
                  factual_direction='Factual history + direction-only')
    artifacts = {arm: root / f'contact_factual_{arm}_v6' for arm in ('support', 'direction')}
    output_name = 'natural_factual_comparison_v6'
    if mode == 'reciprocal':
        reports = {'before': (root / 'natural_stair_time_v1', 'before'),
            'previous': (root / 'contact_factual_direction_v6/long_review', 'direction'),
            'decoder': (root / 'reciprocal_source_decoder_v1/long_review', 'decoder'),
            'coordinated': (root / 'reciprocal_source_coordinated_v1/long_review', 'coordinated')}
        labels = dict(before='Accepted gait baseline', previous='Previous: step-to failure',
                      decoder='Source training: decoder only', coordinated='Source training: coordinated')
        artifacts = {arm: root / f'reciprocal_source_{arm}_v1' for arm in ('decoder', 'coordinated')}
        output_name = 'reciprocal_source_comparison_v1'
    if mode == 'head':
        reports = {'previous': (root / 'contact_factual_direction_v6/long_review', 'direction'),
            'root': (root / 'render_head_fit_v1/root/long_review', 'root'),
            'rendered': (root / 'render_head_fit_v1/rendered/long_review', 'rendered')}
        labels = dict(previous='Previous: step-to failure', root='Learned Root head: alternating landings',
                      rendered='Root + rotation head: incomplete landing evidence')
        artifacts = {arm: root / 'render_head_fit_v1' / arm for arm in ('root', 'rendered')}
        output_name = 'render_head_comparison_v1'
    if mode == 'legs':
        reports = {'previous': (root / 'contact_factual_direction_v6/long_review', 'direction'),
            'root': (root / 'render_head_fit_v1/root/long_review', 'root'),
            'legs': (root / 'leg_head_v1/long_review', 'legs'),
            'sequence': (root / 'leg_head_sequence_v2/long_review', 'legs')}
        labels = dict(previous='Previous: step-to failure', root='Before leg fit: Root-head candidate',
                      legs='Source-window leg fit: rejected', sequence='Continuous leg fit: not approved')
        artifacts = {'legs': root / 'leg_head_v1', 'root': root / 'render_head_fit_v1/root',
                     'sequence': root / 'leg_head_sequence_v2'}
        output_name = 'leg_head_comparison_v2'
    output = root / output_name
    output.mkdir(exist_ok=False)
    directories = {arm: path / original for arm, (path, original) in reports.items()}
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals',
        '--stats', str(base / 'long_route_mixed_10k_ab_v1/mixed_dataset/stats'),
        '--reviews', *[arm + '=' + str(path) for arm, path in directories.items()],
        '--output', str(output / 'visuals')], check=True)
    manifest_path = output / 'visuals/manifest.json'
    old = json.loads(manifest_path.read_text())
    merged = dict(fps=30, postprocessing=None, clips={}, audit=old['audit'],
        labels=labels, quality_approved=False,
        scope='Independent original30Hz recordings, no retiming or padding. Ended arms hold final pose in UI only.')
    source_map = {'generated_8030ae7a2a7ad640': 'reported-stairs',
                  'generated_415ccb3cd0a84f27': 'stairs-415', 'generated_babca93468f737e6': 'stairs-babca'}
    examples = {}
    for arm, (directory, original) in reports.items():
        report = json.loads((directory / 'complete.json').read_text())
        shutil.copyfile(directory / 'complete.json', output / (arm + '-route.json'))
        for row in report['rows']:
            if row['arm'] != original:
                continue
            if row['frames'] == 0:
                merged.setdefault('failures_without_motion', []).append(dict(row, display_arm=arm))
                continue
            stem = f'{row["label"]}-{original}-{row["frames"]}f'
            entry = dict(old['clips'][stem]['arms'][arm], frames=row['frames'], fps=30,
                         duration_seconds=row['frames']/30)
            if len(entry['roots']) != row['frames']:
                raise ValueError('Original recording duration changed')
            name = source_map[row['label']]
            clip = merged['clips'].setdefault(name, dict(frames=0, fps=30, arms={}, source_id=row['label']))
            clip['frames'] = max(clip['frames'], row['frames'])
            clip['arms'][arm] = entry
            raw_dir = output / 'recordings' / arm
            raw_dir.mkdir(parents=True, exist_ok=True)
            recording = directories[arm] / (stem + '.npz')
            shutil.copyfile(recording, raw_dir / recording.name)
            examples[name] = recording
    for name, clip in merged['clips'].items():
        with np.load(examples[name], allow_pickle=False) as archive:
            hf, origin, dx = archive['heightfield_y_m'], archive['min_point_world_xyz_m'], float(archive['dx_m'])
            roots = np.concatenate([entry['roots'] for entry in clip['arms'].values()])
            lo, hi = roots[:, [0, 2]].min(0)-5, roots[:, [0, 2]].max(0)+5
            x0, z0 = np.maximum(0, np.floor((lo-origin[[0, 2]])/dx).astype(int))
            x1, z1 = np.minimum([hf.shape[1], hf.shape[0]], np.ceil((hi-origin[[0, 2]])/dx).astype(int)+1)
            clip['scene'] = dict(heightfield_y_m=[[float(v) if np.isfinite(v) else None for v in line]
                for line in hf[z0:z1, x0:x1]], origin=(origin+np.array([x0*dx, 0., z0*dx])).tolist(), dx=dx)
    manifest_path.write_text(json.dumps(merged, separators=(',', ':'), allow_nan=False))
    for arm, directory in artifacts.items():
        names = ('complete', 'contract') if mode == 'head' or (mode == 'legs' and arm == 'root') else ('complete', 'contract', 'preflight', 'training')
        for name in names:
            shutil.copyfile(directory / f'{name}.json', output / f'{arm}-{name}.json')
    if mode == 'head':
        shutil.copyfile(root / 'render_head_footfalls_v1/complete.json', output / 'footfalls.json')
    if mode == 'legs':
        shutil.copyfile(root / 'leg_head_footfalls_v2/complete.json', output / 'footfalls.json')
        shutil.copyfile(root / 'leg_head_penetration_v2/complete.json', output / 'penetration-locations.json')
        shutil.copyfile(root / 'leg_head_sequence_v2/baseline_parity.json', output / 'baseline-parity.json')
    receipt = dict(quality_approved=False, frame_padding=False, playback_retiming=False,
        files={str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()})
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
    print(json.dumps(dict(status='uploaded', files=len(receipt['files']), quality_approved=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('factual', 'reciprocal', 'head', 'legs'), default='factual')
    main(parser.parse_args().mode)
