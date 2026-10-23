"""Manifest, archive and verify original BONES BVHs through private Drive."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time


GIB = 1024**3


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def relative_path(value):
    path = PurePosixPath(value)
    if (path.is_absolute() or '..' in path.parts or '\\' in value
            or len(path.parts) < 2 or path.suffix != '.bvh' or path.stem.endswith('_M')):
        raise ValueError('Invalid original BVH path: ' + value)
    return path


def file_record(root, path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('Expected a regular source file: ' + str(path))
    name = path.relative_to(root).as_posix()
    relative_path(name)
    sha = digest(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed while hashing: ' + name)
    return dict(path=name, size=before.st_size, sha256=sha)


def read_manifest(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    names = set()
    for row in rows:
        relative_path(row['path'])
        if row['path'] in names or row['size'] < 0 or len(row['sha256']) != 64:
            raise ValueError('Invalid manifest entry: ' + row['path'])
        names.add(row['path'])
    return rows


def require_space(path, incoming, reserve):
    available = shutil.disk_usage(path).free
    if available - incoming < reserve:
        raise ValueError(f'Insufficient space: free={available}, incoming={incoming}, reserve={reserve}')
    return available


def index_source(args):
    root = args.source.resolve(strict=True)
    args.work.mkdir(parents=True, exist_ok=False)
    paths = sorted(p for p in root.rglob('*.bvh') if not p.stem.endswith('_M'))
    if len(paths) != args.expected_files:
        raise ValueError(f'Expected {args.expected_files} originals, found {len(paths)}')
    rows = []
    temporary = args.work / 'manifest.jsonl.partial'
    with temporary.open('x', encoding='utf-8') as stream, ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(lambda path: file_record(root, path), paths):
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
            if len(rows) % 1000 == 0:
                stream.flush()
                write_json(args.work / 'progress.json', dict(phase='hashing', files=len(rows), expected=len(paths)))
                print(f'Hashed {len(rows)}/{len(paths)}', flush=True)
    manifest = args.work / 'manifest.jsonl'
    temporary.replace(manifest)
    summary = dict(scope='all_originals', files=len(rows), total_bytes=sum(r['size'] for r in rows),
                   dates=len({PurePosixPath(r['path']).parts[0] for r in rows}),
                   manifest_sha256=digest(manifest))
    write_json(args.work / 'index_summary.json', summary)
    print(json.dumps(summary), flush=True)


def rclone(args, source, destination):
    subprocess.run([args.rclone, '--config', str(args.config), 'copyto', str(source), str(destination),
                    '--immutable', '--drive-chunk-size', args.drive_chunk_size, '--disable-http2', '--contimeout', '20s',
                    '--timeout', '120s', '--retries', '5', '--low-level-retries', '3',
                    '--stats', '30s', '--stats-one-line', '--stats-log-level', 'NOTICE'], check=True)


def pack_upload(args):
    root = args.source.resolve(strict=True)
    manifest = args.work / 'manifest.jsonl'
    rows = read_manifest(manifest)
    index = json.loads((args.work / 'index_summary.json').read_text())
    if digest(manifest) != index['manifest_sha256']:
        raise ValueError('Source manifest differs from completed index')
    grouped = defaultdict(list)
    for row in rows:
        grouped[PurePosixPath(row['path']).parts[0]].append(row)
    journal = args.work / 'packages.json'
    packages = json.loads(journal.read_text()) if journal.exists() else []
    completed_dates = set()
    for package in packages:
        members = grouped.get(package['date'], [])
        if (not members or package['date'] in completed_dates
                or package['name'] != package['date'] + '.tar.gz'
                or package['files'] != len(members)
                or package['raw_bytes'] != sum(row['size'] for row in members)):
            raise ValueError('Completed package journal does not match the source manifest')
        completed_dates.add(package['date'])
    remote = args.remote.rstrip('/')
    rclone(args, manifest, remote + '/manifest.jsonl')
    rclone(args, args.work / 'index_summary.json', remote + '/index_summary.json')
    for date, members in sorted(grouped.items()):
        if date in completed_dates:
            print(f'Keeping completed upload {date}', flush=True)
            continue
        archive = args.work / (date + '.tar.gz')
        require_space(args.work, sum(r['size'] for r in members), 10 * GIB)
        listing = args.work / (date + '.files0')
        listing.write_bytes(b''.join(r['path'].encode() + b'\0' for r in members))
        subprocess.run(['tar', '-C', str(root), '-I', f'pigz -1 -p {args.workers}',
                        '-cf', str(archive), '--null', '-T', str(listing)], check=True)
        package = dict(name=archive.name, date=date, files=len(members),
                       raw_bytes=sum(r['size'] for r in members), bytes=archive.stat().st_size,
                       sha256=digest(archive))
        rclone(args, archive, remote + '/' + archive.name)
        packages.append(package)
        write_json(args.work / 'packages.json', packages)
        archive.unlink()
        listing.unlink()
        write_json(args.work / 'progress.json', dict(phase='uploaded', dates=len(packages),
                   expected_dates=len(grouped), files=sum(p['files'] for p in packages)))
        print(f'Uploaded {date}: {len(packages)}/{len(grouped)} dates', flush=True)
    complete = dict(index, packages=packages, compressed_bytes=sum(p['bytes'] for p in packages))
    write_json(args.work / 'staging_complete.json', complete)
    rclone(args, args.work / 'staging_complete.json', remote + '/staging_complete.json')
    print(json.dumps(dict(files=complete['files'], total_bytes=complete['total_bytes'], dates=len(packages))), flush=True)


def matches(path, row):
    return (path.is_file() and not path.is_symlink() and path.stat().st_size == row['size']
            and digest(path) == row['sha256'])


def extract_archive(archive, destination, members):
    expected = {r['path']: r for r in members}
    seen = set()
    with tarfile.open(archive, 'r|gz') as stream:
        for item in stream:
            if not item.isfile() or item.name not in expected or item.name in seen:
                raise ValueError('Unexpected archive member: ' + item.name)
            row = expected[item.name]
            if item.size != row['size']:
                raise ValueError('Archive size differs: ' + item.name)
            target = destination.joinpath(*relative_path(item.name).parts)
            if not target.parent.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Destination escapes through a symlink: ' + item.name)
            if target.exists() or target.is_symlink():
                if not matches(target, row):
                    raise ValueError('Existing destination differs: ' + item.name)
                seen.add(item.name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.', suffix='.partial', delete=False) as output:
                with stream.extractfile(item) as source:
                    shutil.copyfileobj(source, output, 1024**2)
                temporary = Path(output.name)
            if not matches(temporary, row):
                raise ValueError('Extracted BVH hash differs: ' + item.name)
            temporary.replace(target)
            seen.add(item.name)
    if seen != set(expected):
        raise ValueError('Archive is missing expected BVHs')


def prefetch_archives(args):
    if (args.work / 'staging_complete.json').exists():
        return
    remote = args.remote.rstrip('/')
    rclone(args, remote + '/index_summary.json', args.work / 'index_summary.json')
    summary = json.loads((args.work / 'index_summary.json').read_text())
    if summary['files'] != args.expected_files or summary['scope'] != 'all_originals':
        raise ValueError('Unexpected source scope for archive prefetch')
    while not (args.work / 'staging_complete.json').exists():
        archives = list(args.work.glob('*.tar.gz'))
        cached = sum(path.stat().st_size for path in archives)
        # Reserve room for all raw BVHs, worst-case remaining archives and tar overhead.
        require_space(args.work, max(0, 2 * summary['total_bytes'] + 10 * GIB - cached), args.reserve_gib * GIB)
        subprocess.run([args.rclone, '--config', str(args.config), 'copy', remote, str(args.work),
                        '--include', '/*.tar.gz', '--include', '/manifest.jsonl',
                        '--include', '/staging_complete.json', '--immutable', '--transfers', str(args.download_transfers),
                        '--max-depth', '1', '--contimeout', '20s', '--timeout', '120s',
                        '--retries', '5', '--low-level-retries', '3', '--stats', '30s',
                        '--stats-one-line', '--stats-log-level', 'NOTICE'], check=True)
        print(f'Prefetched {len(list(args.work.glob("*.tar.gz")))}/{summary["dates"]} date archives', flush=True)
        if not (args.work / 'staging_complete.json').exists():
            time.sleep(45)


def receive_verify(args):
    args.work.mkdir(parents=True, exist_ok=True)
    args.destination.mkdir(parents=True, exist_ok=True)
    if args.prefetch:
        prefetch_archives(args)
    remote = args.remote.rstrip('/')
    for name in ('manifest.jsonl', 'staging_complete.json'):
        if not (args.work / name).exists():
            rclone(args, remote + '/' + name, args.work / name)
    manifest = args.work / 'manifest.jsonl'
    complete = json.loads((args.work / 'staging_complete.json').read_text())
    if digest(manifest) != complete['manifest_sha256']:
        raise ValueError('Downloaded manifest hash differs')
    rows = read_manifest(manifest)
    if len(rows) != complete['files'] or sum(r['size'] for r in rows) != complete['total_bytes']:
        raise ValueError('Manifest totals differ from staging receipt')
    grouped = defaultdict(list)
    remaining = 0
    for row in rows:
        grouped[PurePosixPath(row['path']).parts[0]].append(row)
        target = args.destination / row['path']
        if target.exists() or target.is_symlink():
            if not matches(target, row):
                raise ValueError('Existing destination differs: ' + row['path'])
        else:
            remaining += row['size']
    reserve = args.reserve_gib * GIB
    before = require_space(args.destination, remaining + max(p['bytes'] for p in complete['packages']), reserve)
    for number, package in enumerate(complete['packages'], 1):
        if package['name'] != package['date'] + '.tar.gz' or package['date'] not in grouped:
            raise ValueError('Archive name does not match a manifest date')
        members = grouped[package['date']]
        missing = [r for r in members if not (args.destination / r['path']).exists()]
        if missing:
            require_space(args.destination, remaining + package['bytes'], reserve)
            archive = args.work / package['name']
            if not archive.exists():
                rclone(args, remote + '/' + package['name'], archive)
            if archive.stat().st_size != package['bytes'] or digest(archive) != package['sha256']:
                raise ValueError('Archive hash differs: ' + package['name'])
            extract_archive(archive, args.destination, members)
            remaining -= sum(r['size'] for r in missing)
            archive.unlink()
        write_json(args.work / 'progress.json', dict(phase='extracted', dates=number, expected_dates=len(complete['packages'])))
        print(f'Extracted {package["date"]}: {number}/{len(complete["packages"])} dates', flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        checks = list(pool.map(lambda row: matches(args.destination / row['path'], row), rows))
    failures = [r['path'] for r, ok in zip(rows, checks) if not ok]
    expected_names = {r['path'] for r in rows}
    extra = sorted(p.relative_to(args.destination).as_posix() for p in args.destination.rglob('*.bvh')
                   if p.relative_to(args.destination).as_posix() not in expected_names)
    report = dict(files_expected=len(rows), files_verified=sum(checks), mismatches=failures,
                  unexpected_bvhs=extra, total_bytes=complete['total_bytes'], manifest_sha256=digest(manifest),
                  free_bytes_before=before, free_bytes_after=shutil.disk_usage(args.destination).free,
                  minimum_remaining_bytes=reserve)
    write_json(args.work / 'verification.json', report)
    print(json.dumps(report), flush=True)
    if failures or extra or report['free_bytes_after'] < reserve:
        raise ValueError('Final BVH verification failed; see verification.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['index', 'upload', 'receive'])
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--expected-files', type=int, default=71132)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--reserve-gib', type=int, default=200)
    parser.add_argument('--drive-chunk-size', default='32M')
    parser.add_argument('--prefetch', action='store_true', help='Download finished date archives while the source uploader is active')
    parser.add_argument('--download-transfers', type=int, default=4)
    parser.add_argument('--rclone')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--remote')
    args = parser.parse_args()
    {'index': index_source, 'upload': pack_upload, 'receive': receive_verify}[args.mode](args)


if __name__ == '__main__':
    main()
