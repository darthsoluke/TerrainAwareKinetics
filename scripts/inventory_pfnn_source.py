"""Verify an existing PFNN corpus against the author's original ZIP directory."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile
import zlib

import numpy as np

from motionbricks.data.lafan_resolved_motion import _read_header


OFFICIAL_URL = 'https://theorangeduck.com/media/uploads/other_stuff/pfnn.zip'


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position, self.cache, self.downloaded = url, 0, {}, 0
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
            self.size = int(response.headers['Content-Length'])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if self.position < 0:
            raise ValueError('Negative ZIP offset')
        return self.position

    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size-self.position)
        chunks = []
        while size > 0:
            block = self.position//65536
            start, end = block*65536, min((block+1)*65536, self.size)-1
            if block not in self.cache:
                request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206:
                        raise ValueError('Official server did not honor bounded ZIP range request')
                    data = response.read(end-start+2)
                if len(data) != end-start+1:
                    raise ValueError('Truncated official ZIP range')
                self.cache[block] = data
                self.downloaded += len(data)
            offset = self.position-start
            count = min(size, len(self.cache[block])-offset)
            chunks.append(self.cache[block][offset:offset+count])
            self.position += count
            size -= count
        return b''.join(chunks)


def identity(path):
    digest, crc, count = hashlib.sha256(), 0, 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
            count += len(chunk)
    return dict(sha256=digest.hexdigest(), crc32=crc & 0xffffffff, bytes=count)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    reader = RangeReader(OFFICIAL_URL)
    archive = zipfile.ZipFile(reader)
    entries = archive.infolist()
    candidates = [item.filename[:-len('generate_database.py')] for item in entries if item.filename.endswith('/generate_database.py')]
    if len(candidates) != 1:
        raise ValueError('Ambiguous official PFNN source root')
    prefix = candidates[0]
    lookup = {item.filename[len(prefix):]: item for item in entries if item.filename.startswith(prefix)}
    manifest, extras = [], []
    for path in sorted((args.source/'data/animations').glob('*.bvh')):
        relative = path.relative_to(args.source).as_posix()
        actual = identity(path)
        item = lookup.get(relative)
        official = item is not None and actual['bytes'] == item.file_size and actual['crc32'] == item.CRC
        with path.open('rb') as stream:
            parser, frames, fps = _read_header(stream)
        stem = path.stem.replace('_mirror', '')
        group = re.sub(r'_\d{3}$', '', stem)
        annotations = {}
        for suffix in ('.phase', '.gait', '_footsteps.txt'):
            companion = path.with_name(path.stem+suffix)
            if not companion.is_file():
                annotations[suffix] = dict(present=False)
                continue
            companion_relative = companion.relative_to(args.source).as_posix()
            companion_id = identity(companion)
            reference = lookup.get(companion_relative)
            annotation = dict(present=True, **companion_id, official_crc_match=reference is not None
                and companion_id['crc32'] == reference.CRC and companion_id['bytes'] == reference.file_size)
            if suffix in ('.phase', '.gait'):
                values = np.loadtxt(companion, ndmin=2)
                annotation.update(shape=list(values.shape), finite=bool(np.isfinite(values).all()),
                                  frame_count_matches=values.shape[0] == frames)
            annotations[suffix] = annotation
        names = parser.get_joints_names()
        manifest.append(dict(filename=relative, **actual, official_crc_match=official,
            frames=frames, fps=fps, seconds=frames/fps, joints=len(names), joint_names=names,
            role='reference_pose' if path.stem == 'rest' else 'motion',
            source_recording_group=group, mirrored='_mirror' in path.stem, annotations=annotations))
        if not official:
            extras.append(relative)
    source_folder = args.output/'official_source'
    for name in ('generate_database.py', 'generate_patches.py', 'README.md', 'LICENSE.md',
                 'motion/BVH.py', 'motion/Animation.py', 'motion/AnimationStructure.py',
                 'motion/Quaternions.py', 'motion/Learning.py', 'motion/Pivots.py'):
        if name not in lookup:
            continue
        path = source_folder/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(lookup[name]))
    official_bvh = [name for name in lookup if name.startswith('data/animations/') and name.endswith('.bvh')]
    present = {row['filename'] for row in manifest if row['official_crc_match']}
    heights = []
    for path in sorted((args.source/'data/heightmaps').glob('*.txt')):
        relative = path.relative_to(args.source).as_posix()
        actual = identity(path)
        entry = lookup.get(relative)
        normalized = path.read_bytes().replace(b'\r\n', b'\n')
        heights.append(dict(filename=relative, **actual, official_crc_match=entry is not None
            and actual['bytes'] == entry.file_size and actual['crc32'] == entry.CRC,
            official_lf_crc_match=entry is not None and len(normalized) == entry.file_size
            and (zlib.crc32(normalized) & 0xffffffff) == entry.CRC))
    result = dict(schema='motionbricks.pfnn_source_inventory.v1', official_url=OFFICIAL_URL,
        official_zip_bytes=reader.size, official_zip_directory_entries=len(entries),
        metadata_and_code_downloaded_bytes=reader.downloaded,
        source=str(args.source.resolve()), clips=manifest, heightmaps=heights,
        official_bvh_count=len(official_bvh), local_bvh_count=len(manifest),
        verified_official_clips=len(present), missing_official_clips=sorted(set(official_bvh)-present),
        unverified_local_clips=extras, total_frames=sum(row['frames'] for row in manifest),
        unique_recording_groups=len({row['source_recording_group'] for row in manifest if row['role'] == 'motion'}),
        motion_clips=sum(row['role'] == 'motion' for row in manifest),
        original_hours=sum(row['seconds'] for row in manifest if not row['mirrored'])/3600,
        mirrored_hours=sum(row['seconds'] for row in manifest if row['mirrored'])/3600,
        original_source_code={str(path.relative_to(source_folder)): identity(path)['sha256']
                              for path in source_folder.rglob('*') if path.is_file()},
        no_source_modified=True, training_started=False)
    (args.output/'inventory.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if args.package:
        if extras or result['missing_official_clips'] or not all(row['official_lf_crc_match'] for row in heights):
            raise ValueError('Cannot package unverified PFNN source data')
        archive_path = args.output/'pfnn_source.zip'
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as package:
            for row in manifest:
                path = args.source/row['filename']
                package.write(path, row['filename'])
                for suffix, annotation in row['annotations'].items():
                    if annotation['present']:
                        companion = path.with_name(path.stem+suffix)
                        relative = companion.relative_to(args.source).as_posix()
                        if annotation['official_crc_match']:
                            package.write(companion, relative)
                            annotation['package_source'] = 'verified_local_copy'
                        else:
                            if relative not in lookup:
                                raise ValueError('Source annotation absent from the official release')
                            canonical = archive.read(lookup[relative])
                            if suffix in ('.phase', '.gait'):
                                values = np.loadtxt(io.BytesIO(canonical), ndmin=2)
                                if values.shape[0] != row['frames'] or not np.isfinite(values).all():
                                    raise ValueError('Official annotation does not match source frame clock')
                            package.writestr(relative, canonical)
                            annotation['package_source'] = 'restored_official_zip_annotation'
                            annotation['packaged_sha256'] = hashlib.sha256(canonical).hexdigest()
            for row in heights:
                package.writestr(row['filename'], (args.source/row['filename']).read_bytes().replace(b'\r\n', b'\n'))
            for path in source_folder.rglob('*'):
                if path.is_file():
                    package.write(path, path.relative_to(args.output).as_posix())
            result['metadata_and_code_downloaded_bytes'] = reader.downloaded
            (args.output/'inventory.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            package.write(args.output/'inventory.json', 'inventory.json')
        receipt = dict(filename=archive_path.name, **identity(archive_path),
            source_inventory_sha256=identity(args.output/'inventory.json')['sha256'],
            original_motion_unchanged=True, heightmap_change='CRLF to original LF only',
            annotation_policy='Official bytes, local originals preserved; any local mismatch recorded')
        (args.output/'source.receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    summary = {key: value for key, value in result.items() if key not in ('clips', 'heightmaps', 'original_source_code')}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--package', action='store_true')
    main(parser.parse_args())
