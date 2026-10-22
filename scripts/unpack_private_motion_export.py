"""Verify an owned-Drive export and extract regular files within its named root."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main(args):
    receipt = json.loads(args.receipt.read_text())
    if args.archive.name != receipt['filename'] or args.archive.stat().st_size != receipt['bytes']:
        raise ValueError('Archive name/size differs from its private receipt')
    if digest(args.archive) != receipt['sha256']:
        raise ValueError('Archive SHA256 differs')
    expected = args.archive.name.removesuffix('.tar.gz')
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive) as stream:
        members = stream.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError('Duplicate archive path')
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or path.parts[0] != expected
                    or '\\' in member.name or ':' in member.name
                    or not (member.isfile() or member.isdir())):
                raise ValueError('Unsafe archive path/type: '+member.name)
            target = destination.joinpath(*path.parts).resolve()
            target.relative_to(destination)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source:
                if target.exists():
                    if hashlib.sha256(source.read()).hexdigest() != digest(target):
                        raise ValueError('Refusing changed existing export: '+member.name)
                    continue
                with target.open('xb') as output:
                    shutil.copyfileobj(source, output)
    root = destination/expected
    files = json.loads((root/'receipt.json').read_text())['files']
    if len(files) != receipt['files']:
        raise ValueError('Receipt file count differs')
    for name, checksum in files.items():
        path = (root/name).resolve()
        path.relative_to(root)
        if digest(path) != checksum:
            raise ValueError('File checksum differs: '+name)
    print(json.dumps(dict(verified_files=len(files), root=str(root), archive_sha256=receipt['sha256'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'receipt', 'destination'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
