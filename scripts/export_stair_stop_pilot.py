"""Privately export a rejected STOP pilot without hiding before/after failures."""

import argparse
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from scripts.train_native_mb import sha256


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals',
        '--stats', str(args.stats), '--reviews', 'before='+str(args.run/'baseline'),
        'candidate='+str(args.run/'final'), '--output', str(args.output/'visuals')], check=True)
    manifest_path = args.output/'visuals/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(quality_approved=False,
        labels=dict(before='Approved moving weights: stopping still fails',
                    candidate='Rejected STOP-only trial: not a live controller'),
        scope='Recorded causal model output at30Hz. No IK, output correction, padding or retiming. Continuous MOVE outputs are identical; STOP still fails.')
    for clip in manifest['clips'].values():
        for arm in clip['arms'].values():
            arm.update(frames=clip['frames'], fps=30, duration_seconds=clip['frames']/30)
    manifest_path.write_text(json.dumps(manifest, separators=(',', ':'), allow_nan=False))
    for name in ('complete.json', 'contract.json', 'preflight.json', 'source_screen.json'):
        shutil.copy2(args.run/name, args.output/name)
    for arm, folder in [('before', 'baseline'), ('candidate', 'final')]:
        target = args.output/'recordings'/arm
        target.mkdir(parents=True)
        for path in (args.run/folder).glob('*.npz'):
            shutil.copy2(path, target/path.name)
    receipt = dict(quality_approved=False, frame_padding=False, playback_retiming=False,
        files={str(p.relative_to(args.output)):sha256(p) for p in args.output.rglob('*') if p.is_file()})
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2))
    archive = args.output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(args.output, arcname=args.output.name)
    archive_receipt = archive.with_suffix('.receipt.json')
    archive_receipt.write_text(json.dumps(dict(filename=archive.name, sha256=sha256(archive),
        bytes=archive.stat().st_size, files=len(receipt['files'])), indent=2))
    settings = configparser.ConfigParser(interpolation=None)
    settings.read(args.drive_config)
    if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('Not the user-owned OAuth client')
    environment = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='',
        http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
    for path in (archive, archive_receipt):
        subprocess.run(['rclone', '--config', str(args.drive_config), 'copyto', str(path),
            'private-store:motionbricks/parc_stair64_20261013/'+path.name, '--immutable', '--checksum',
            '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'],
            env=environment, check=True)
    print(json.dumps(dict(status='uploaded', files=len(receipt['files']), quality_approved=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'stats', 'output', 'drive-config'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
