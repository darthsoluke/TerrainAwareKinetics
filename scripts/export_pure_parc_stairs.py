"""Private official SOMA comparisons of original, PARC-window and feedback fits."""

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

from scripts.train_native_mb import sha256


def main(args):
    if not (args.stats / 'mean.npy').is_file() or not (args.stats / 'std.npy').is_file():
        raise ValueError('Use the verified parent dataset stats before creating an export')
    if args.causal_review and not args.terrain_run:
        raise ValueError('Three-arm causal export requires the matched terrain run')
    result = json.loads((args.run / 'complete.json').read_text())
    if not result['frozen_hashes_match'] or not result['parent_file_unchanged']:
        raise ValueError('Parent/frozen state verification failed')
    args.output.mkdir(parents=True, exist_ok=False)
    stages = dict(before='baseline', window='window_review', feedback='final')
    runs = {arm: args.run for arm in stages}
    if args.terrain_run:
        alternate = json.loads((args.terrain_run / 'complete.json').read_text())
        if not alternate['frozen_hashes_match'] or not alternate['parent_file_unchanged']:
            raise ValueError('Joint terrain run failed its immutable checks')
        stages['terrain'] = 'final'
        runs['terrain'] = args.terrain_run
    for arm, stage in stages.items():
        folder = args.output / 'recordings' / arm
        folder.mkdir(parents=True)
        for path in (runs[arm] / stage).glob('*-speed*-*f.npz'):
            shutil.copy2(path, folder / path.name)
    if args.causal_review:
        shutil.copy2(args.causal_review / 'complete.json', args.output / 'causal-review.json')
        for causal, arm in dict(before='before', frozen='feedback', joint='terrain').items():
            for path in (args.causal_review / causal).glob('*-flat.npz'):
                with np.load(path, allow_pickle=False) as archive:
                    frames = len(archive['roots'])
                label = path.stem.removesuffix('-flat')
                shutil.copy2(path, args.output / 'recordings' / arm / (label + f'-flat-counterfactual-{frames}f.npz'))
    source = args.output / 'recordings' / 'source'
    source.mkdir(parents=True)
    for path in (args.run / 'baseline').glob('*-source.npz'):
        with np.load(path, allow_pickle=False) as archive:
            frames = len(archive['roots'])
        label = path.stem.removesuffix('-source')
        for speed in ('1.0', '0.7'):
            shutil.copy2(path, source / (label + '-speed' + speed + f'-{frames}f.npz'))
    for name in ('complete.json', 'contract.json', 'preflight.json', 'training.json'):
        shutil.copy2(args.run / name, args.output / name)
        if args.terrain_run:
            shutil.copy2(args.terrain_run / name, args.output / ('terrain-' + name))
    if args.terrain_run:
        shutil.copy2(args.terrain_run / 'baseline_parity.json', args.output / 'terrain-baseline_parity.json')
        shutil.copy2(args.terrain_run / 'final' / 'terrain_response.json', args.output / 'terrain-response.json')
    for stage in stages.values():
        shutil.copy2(args.run / stage / 'terrain_response.json', args.output / (stage + '-terrain_response.json'))
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', str(args.stats),
        '--reviews', *[arm + '=' + str(args.output / 'recordings' / arm) for arm in (*stages, 'source')],
        '--output', str(args.output / 'visuals')], check=True)
    path = args.output / 'visuals' / 'manifest.json'
    original = json.loads(path.read_text())
    manifest = dict(fps=30, postprocessing=None, quality_approved=False, audit=original['audit'], clips={},
        labels=dict(before='Approved parent: new routes unverified', window='PARC-only source-window fit',
                    feedback='PARC-only continuous-history fit', terrain='PARC-only terrain+motion joint fit',
                    source='Unmodified PARC source'),
        scope='Each actual recording retains its own30Hz length. Source is not a live actor. No padding/retiming.')
    for name, clip in original['clips'].items():
        base = name.rsplit('-', 1)[0]
        merged = manifest['clips'].setdefault(base, dict(frames=0, fps=30, arms={}, scene=clip['scene']))
        merged['frames'] = max(merged['frames'], clip['frames'])
        for arm, record in clip['arms'].items():
            record.update(frames=clip['frames'], fps=30, duration_seconds=clip['frames'] / 30)
            merged['arms'][arm] = record
    for name, clip in manifest['clips'].items():
        example = next(path for arm in (*stages, 'source')
                       for path in (args.output / 'recordings' / arm).glob(name + '-*f.npz'))
        with np.load(example, allow_pickle=False) as archive:
            hf, origin, dx = archive['heightfield_y_m'], archive['min_point_world_xyz_m'], float(archive['dx_m'])
        roots = np.concatenate([record['roots'] for record in clip['arms'].values()])
        lo, hi = roots[:, [0, 2]].min(0) - 5, roots[:, [0, 2]].max(0) + 5
        x0, z0 = np.maximum(0, np.floor((lo - origin[[0, 2]]) / dx).astype(int))
        x1, z1 = np.minimum([hf.shape[1], hf.shape[0]], np.ceil((hi - origin[[0, 2]]) / dx).astype(int) + 1)
        clip['scene'] = dict(heightfield_y_m=[[float(v) if np.isfinite(v) else None for v in row]
            for row in hf[z0:z1, x0:x1]], origin=(origin + np.array([x0 * dx, 0, z0 * dx])).tolist(), dx=dx)
    path.write_text(json.dumps(manifest, separators=(',', ':'), allow_nan=False))
    receipt = dict(quality_approved=False, frame_padding=False, playback_retiming=False,
        files={str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob('*') if p.is_file()})
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    archive = args.output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(args.output, arcname=args.output.name)
    archive_receipt = archive.with_suffix('.receipt.json')
    archive_receipt.write_text(json.dumps(dict(filename=archive.name, sha256=sha256(archive),
        bytes=archive.stat().st_size, files=len(receipt['files'])), indent=2))
    settings = configparser.ConfigParser(interpolation=None)
    settings.read(args.drive_config)
    if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('User-owned private client mismatch')
    environment = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', http_proxy='',
                       https_proxy='', all_proxy='', NO_PROXY='*')
    for file in (archive, archive_receipt):
        subprocess.run(['rclone', '--config', str(args.drive_config), 'copyto', str(file),
            'private-store:motionbricks/parc_stair64_20261013/' + file.name, '--immutable', '--checksum',
            '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'],
            env=environment, check=True)
    print(json.dumps(dict(status='uploaded', files=len(receipt['files']), quality_approved=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'stats', 'output', 'drive-config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--terrain-run', type=Path)
    parser.add_argument('--causal-review', type=Path)
    main(parser.parse_args())
