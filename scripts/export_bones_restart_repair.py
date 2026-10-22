"""Export matched restart probes, including all failed reviews and raw outputs."""

import argparse
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def main(args):
    reports = {name:json.loads((folder/'complete.json').read_text())
               for name,folder in [('control', args.control), ('phase', args.phase)]}
    a, b = reports['control'], reports['phase']
    for key in ('baseline', 'exposure', 'schedule', 'actual_updates', 'actor_primitives'):
        if a[key] != b[key]:
            raise ValueError('Matched probe differs: '+key)
    if not all(report['frozen_hashes_match'] for report in reports.values()):
        raise ValueError('Frozen module mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    for name,folder in [('control', args.control), ('phase', args.phase)]:
        for filename in ('complete.json', 'contract.json', 'preflight.json'):
            shutil.copyfile(folder/filename, args.output/(name+'-'+filename))
    arms = [('before', a['baseline']), ('control', a['final']), ('phase', b['final'])]
    summary = dict(matched_schedule=True, frozen_hashes_match=True,
        actual_updates_each=a['actual_updates'], actor_primitives_each=a['actor_primitives'],
        quality_approved=False, postprocessing=None, arms={})
    for name,data in arms:
        selected = [row for row in data['flat'] if row['request_speed']==1.6 and row['idle_seconds']==16]
        summary['arms'][name] = dict(default_restart_rows=selected,
            critical=next(row for row in selected if abs(row['yaw']-3.141592653589793)<1e-6),
            stairs_pass_count=sum(row['passed'] for row in data['stairs']),
            restarts=data['restarts'])
    write_json(args.output/'summary.json', summary)
    reviews = {}
    for label,folder in [('before', args.control/'baseline_flat'), ('control', args.control/'final_flat'),
                         ('phase', args.phase/'final_flat')]:
        target = args.output/'recordings'/label
        target.mkdir(parents=True)
        for filename,name in [('idle60-case4.npz', 'idle16s-forward-yaw180'),
                              ('idle60-case0.npz', 'idle16s-forward-yaw0'),
                              ('idle0-case0.npz', 'immediate-forward-yaw0')]:
            shutil.copyfile(folder/filename, target/(name+'.npz'))
        reviews[label] = target
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', str(args.stats),
        '--reviews', *[label+'='+str(folder) for label,folder in reviews.items()],
        '--output', str(args.output/'visuals')], check=True)
    manifest_path = args.output/'visuals/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(quality_approved=False, postprocessing=None,
        scope='Actual continuous actor output. Zero input0..479,forward480..719. No reset,future pose or padding.',
        selection='Predeclared yaw180 failure,yaw0 restart and immediate controls; all6restart/16flat/12stair results retained.')
    manifest['labels'].update(before='Prior orange command model', control='Corrected graph + real restart data',
                              phase='Same + real-source restart actor prior')
    write_json(manifest_path, manifest)
    recordings = [name+'='+str(path/'idle16s-forward-yaw180.npz') for name,path in reviews.items()]
    subprocess.run([sys.executable, '-m', 'scripts.review_command_probe_mechanics', '--recordings', *recordings,
        '--output', str(args.output/'mechanics'), '--begin', '480', '--direction', '0', '-1'], check=True)
    files = {str(path.relative_to(args.output)):sha256(path) for path in args.output.rglob('*') if path.is_file()}
    write_json(args.output/'receipt.json', dict(files=files, quality_approved=False, frame_padding=False))
    archive = args.output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(args.output, arcname=args.output.name)
    receipt = archive.with_suffix('.receipt.json')
    write_json(receipt, dict(filename=archive.name, sha256=sha256(archive), bytes=archive.stat().st_size, files=len(files)))
    settings = configparser.ConfigParser(interpolation=None)
    settings.read(args.drive_config)
    if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('User-owned private Drive client required')
    env = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
    for path in (archive, receipt):
        subprocess.run(['rclone', '--config', str(args.drive_config), 'copyto', str(path),
            'private-store:motionbricks/parc_rugged32_20261013_v2/'+path.name, '--immutable', '--checksum',
            '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'], env=env, check=True)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control', 'phase', 'stats', 'output', 'drive-config'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
