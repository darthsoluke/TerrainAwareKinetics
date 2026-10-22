"""Receive and verify PFNN archives exclusively through private Google Drive."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    archives = args.archives if args.archives is not None else args.output / 'archives'
    archives.mkdir(parents=True, exist_ok=args.archives is not None)
    prefix = [args.rclone, '--config', str(args.config), '--contimeout', '20s',
              '--timeout', '2m', '--retries', '2', '--low-level-retries', '2']
    ready_at = time.monotonic() + args.wait_seconds
    save(args.output / 'status.json', dict(stage='waiting_for_drive_manifest', pid=os.getpid()))
    while True:
        call = subprocess.run(prefix + ['lsf', args.remote, '--max-depth', '1', '--files-only',
                                       '--include', 'manifest.json'], capture_output=True, text=True)
        if call.returncode == 0 and 'manifest.json' in call.stdout.splitlines():
            break
        if call.returncode not in (0, 3):
            raise RuntimeError('Drive manifest query failed: ' + call.stderr[-2000:])
        if time.monotonic() >= ready_at:
            raise TimeoutError('Drive ready manifest did not arrive within the transfer window')
        time.sleep(30)

    def copy(name):
        target = archives / name
        subprocess.run(prefix + ['copyto', args.remote.rstrip('/') + '/' + name, str(target),
            '--immutable', '--multi-thread-streams', '4', '--stats', '30s', '--stats-one-line'], check=True)
        return target

    manifest = json.loads(copy('manifest.json').read_text())
    expected = {'dataset_world_v2.tar.gz', 'cache_pfnn_world_v2.tar.gz', 'runtime_assets.tar.gz'}
    if (manifest['schema'] != 'motionbricks.pfnn_pro6000_transfer.v1'
            or len(manifest['packages']) != 3 or {p['name'] for p in manifest['packages']} != expected):
        raise ValueError('Unexpected transfer manifest/package set')
    data = args.output / 'data'
    data.mkdir()
    for entry in manifest['packages']:
        save(args.output / 'status.json', dict(stage='downloading', package=entry['name'], pid=os.getpid()))
        archive = copy(entry['name'])
        if archive.stat().st_size != entry['bytes'] or digest(archive) != entry['sha256']:
            raise ValueError('Downloaded archive identity differs: ' + entry['name'])
        save(args.output / 'status.json', dict(stage='extracting', package=entry['name'], pid=os.getpid()))
        with tarfile.open(archive, 'r:gz') as stream:
            stream.extractall(data, filter='data')
        print(json.dumps(dict(verified=entry)), flush=True)
    paths = {
        'pfnn_mixed_20261015/dataset_world_v2/manifest.json': 'dataset_world_v2/manifest.json',
        'pfnn_mixed_20261015/cache_pfnn_world_v2/complete.json': 'cache_pfnn_world_v2/complete.json',
        'pfnn_mixed_20261015/cache_pfnn_world_v2/data/records.jsonl': 'cache_pfnn_world_v2/data/records.jsonl',
        'pfnn_mixed_20261015/full_training_v16/pose/weights.pt': 'parent/weights.pt',
    }
    if set(manifest['source_identities']) != set(paths):
        raise ValueError('Source identity set differs')
    for source, target in paths.items():
        if digest(data / target) != manifest['source_identities'][source]:
            raise ValueError('Extracted source identity differs: ' + target)
    proof = json.loads((data / 'proofs/preflight_a100/complete.json').read_text())
    for path, expected_hash in proof['binding']['sources'].items():
        if digest(data / 'code_base' / path) != expected_hash:
            raise ValueError('Extracted source snapshot differs: ' + path)
    save(args.output / 'complete.json', dict(passed=True, data=str(data), packages=manifest['packages'],
        source_identities=manifest['source_identities'], optimizer_updates=0, dataset_rebuilt=False,
        transport='private Google Drive', finished_unix_time=time.time()))
    save(args.output / 'status.json', dict(stage='complete', pid=os.getpid()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--archives', type=Path, help='Reuse previously downloaded immutable archives')
    parser.add_argument('--rclone', required=True)
    parser.add_argument('--remote', required=True)
    parser.add_argument('--wait-seconds', type=int, default=10800)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            save(args.output / 'failure.json', dict(error=repr(error), unix_time=time.time()))
        raise
