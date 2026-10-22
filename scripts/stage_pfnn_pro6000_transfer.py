"""Upload existing PFNN training artifacts from their host to private Drive."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def main(args):
    run = args.base / 'pfnn_mixed_20261015'
    code = run / 'code_pfnn_frame_actor_v3'
    plan = json.loads((code / 'configs/pfnn_frame_actor_v1.json').read_text())
    identities = {
        run / 'dataset_world_v2/manifest.json': plan['manifest_sha256'],
        run / 'cache_pfnn_world_v2/complete.json': plan['cache_complete_sha256'],
        run / 'cache_pfnn_world_v2/data/records.jsonl': plan['cache_records_sha256'],
        run / 'full_training_v16/pose/weights.pt': plan['parent_sha256'],
    }
    for path, expected in identities.items():
        if digest(path) != expected:
            raise ValueError('Source artifact identity differs: ' + str(path))
    preflight = json.loads((run / 'pfnn_frame_actor_v1/preflight_v3/complete.json').read_text())
    for path, expected in preflight['binding']['sources'].items():
        if digest(code / path) != expected:
            raise ValueError('Source snapshot differs: ' + path)
    args.output.mkdir(parents=True, exist_ok=False)
    packages = {
        'dataset_world_v2.tar.gz': [(run / 'dataset_world_v2', 'dataset_world_v2')],
        'cache_pfnn_world_v2.tar.gz': [(run / 'cache_pfnn_world_v2', 'cache_pfnn_world_v2')],
        'runtime_assets.tar.gz': [(code, 'code_base'),
            (run / 'full_training_v16/pose/weights.pt', 'parent/weights.pt'),
            (args.base / 'long_route_mixed_10k_ab_v1/mixed_dataset/stats', 'stats'),
            (run / 'pfnn_frame_actor_v1/coverage', 'proofs/coverage'),
            (run / 'pfnn_frame_actor_v1/preflight_v3', 'proofs/preflight_a100'),
            (run / 'pfnn_frame_actor_v1/adam_smoke_v3/complete.json', 'proofs/adam_smoke_a100.json')],
    }
    versions = {entry.metadata['Name']: entry.version for entry in importlib.metadata.distributions()}
    receipt = dict(schema='motionbricks.pfnn_pro6000_transfer.v1',
        source_identities={str(p.relative_to(args.base)): h for p, h in identities.items()},
        source_snapshot=str(code), plan_sha256=digest(code / 'configs/pfnn_frame_actor_v1.json'),
        source_package_versions=versions, packages=[], transport='private Google Drive',
        optimizer_updates=0, dataset_rebuilt=False)
    env = dict(os.environ)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)

    def upload(path):
        subprocess.run([args.rclone, '--config', str(args.config), 'copyto', str(path),
            args.remote.rstrip('/') + '/' + path.name, '--immutable', '--drive-chunk-size', '64M',
            '--tpslimit', '2', '--retries', '3', '--low-level-retries', '3', '--contimeout', '30s',
            '--timeout', '5m', '--stats', '30s', '--stats-one-line'], check=True, env=env)

    for name, entries in packages.items():
        save(args.output / 'status.json', dict(stage='packaging', package=name, unix_time=time.time()))
        archive = args.output / name

        def include(info):
            parts = Path(info.name).parts
            if '__pycache__' in parts or '.pytest_cache' in parts:
                return None
            if info.issym() or info.islnk():
                raise ValueError('Archive source contains a link: ' + info.name)
            return info

        with tarfile.open(archive, 'w:gz', compresslevel=1) as stream:
            for source, member in entries:
                stream.add(source, arcname=member, filter=include)
        record = dict(name=name, bytes=archive.stat().st_size, sha256=digest(archive))
        receipt['packages'].append(record)
        save(args.output / 'manifest.preparing.json', receipt)
        save(args.output / 'status.json', dict(stage='uploading', package=name, **record, unix_time=time.time()))
        upload(archive)
        print(json.dumps(dict(uploaded=record)), flush=True)
    # The manifest is the ready marker; publish it only after every package.
    save(args.output / 'manifest.json', receipt)
    upload(args.output / 'manifest.json')
    save(args.output / 'status.json', dict(stage='complete', unix_time=time.time(), packages=receipt['packages']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'output', 'config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--rclone', default='rclone')
    parser.add_argument('--remote', required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            save(args.output / 'failure.json', dict(error=repr(error), unix_time=time.time()))
        raise
