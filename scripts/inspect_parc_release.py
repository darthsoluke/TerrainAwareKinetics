"""Bounded HTTP-range inspection of the official PARC release, without training."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import pickle
import urllib.request
import zipfile

import numpy as np


def get_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.downloaded = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError('Negative archive position')
        return self.position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b''
        if count > 128 * 1024**2 or self.downloaded + count > 256 * 1024**2:
            raise ValueError('Inspection download budget exceeded')
        first, last = self.position, self.position + count - 1
        request = urllib.request.Request(self.url + f'?audit_range={first}-{last}',
                                         headers={'Range': f'bytes={first}-{last}'})
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{self.size}':
                raise ValueError('Server did not honor the exact bounded range')
            value = response.read(count + 1)
        if len(value) != count:
            raise ValueError('Incorrect range length')
        self.position += count
        self.downloaded += count
        return value


class ArrayUnpickler(pickle.Unpickler):
    """Only load primitive containers and NumPy arrays; never import PARC code."""

    def find_class(self, module, name):
        allowed = {
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy.core.multiarray', 'scalar'): np.core.multiarray.scalar,
            ('numpy._core.multiarray', 'scalar'): np.core.multiarray.scalar,
            ('numpy.core.numeric', '_frombuffer'): np.core.numeric._frombuffer,
            ('numpy._core.numeric', '_frombuffer'): np.core.numeric._frombuffer,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unapproved type {module}.{name}')
        return allowed[module, name]


def summarize(value):
    if isinstance(value, np.ndarray):
        result = dict(shape=list(value.shape), dtype=str(value.dtype))
        if value.dtype.kind in 'fiub' and value.size:
            finite = np.isfinite(value)
            result.update(finite=bool(finite.all()), minimum=float(value[finite].min()) if finite.any() else None,
                          maximum=float(value[finite].max()) if finite.any() else None)
            if value.size <= 16:
                result['values'] = value.tolist()
        return result
    if isinstance(value, dict):
        return {str(key): summarize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return dict(type=type(value).__name__, length=len(value))
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, bytes):
        return dict(serialized_bytes=len(value))
    return dict(type=type(value).__name__)


def inspect_payload(payload):
    container = ArrayUnpickler(io.BytesIO(payload)).load()
    result = {'container': summarize(container)}
    for key in ('motion_data', 'terrain_data'):
        value = container.get(key)
        if isinstance(value, bytes):
            value = ArrayUnpickler(io.BytesIO(value)).load()
        result[key] = summarize(value)
        if key == 'motion_data' and isinstance(value, dict):
            root = value.get('root_pos', value.get('root_position'))
            fps = value.get('fps')
            if root is not None and fps:
                result['motion_summary'] = dict(frames=len(root), fps=float(fps),
                    duration_s=(len(root) - 1) / float(fps), first_root=root[0].tolist(),
                    last_root=root[-1].tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--member', action='append', default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = get_json('https://huggingface.co/api/datasets/mxucg/PARC')['sha']
    tree = get_json(f'https://huggingface.co/api/datasets/mxucg/PARC/tree/{revision}')
    entry = next(row for row in tree if row['path'] == 'dec_release.zip')
    remote = RemoteZip(f'https://huggingface.co/datasets/mxucg/PARC/resolve/{revision}/dec_release.zip', entry['size'])
    report = dict(revision=revision, archive=entry, download_mode='HTTP ranges, not full archive', samples=[])
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        index = [dict(name=row.filename, size=row.file_size, compressed_size=row.compress_size, crc32=row.CRC)
                 for row in infos]
        (args.output / 'archive_index.json').write_text(json.dumps(index, indent=2))
        report['entries'] = len(index)
        report['extensions'] = dict(Counter(Path(row.filename).suffix for row in infos if not row.is_dir()))
        for member in args.member:
            info = archive.getinfo(member)
            if info.file_size > 128 * 1024**2:
                raise ValueError('Member exceeds inspection budget')
            payload = archive.read(info)
            digest = hashlib.sha256(payload).hexdigest()
            # Never use an archive member path as a filesystem destination.
            local = args.output / (digest[:12] + '_' + Path(member).name)
            local.write_bytes(payload)
            sample = dict(member=member, sha256=digest, local_file=str(local.resolve()), size=len(payload))
            if member.endswith('.pkl'):
                try:
                    sample.update(inspect_payload(payload))
                except (pickle.UnpicklingError, ValueError, KeyError, AttributeError) as error:
                    sample['inspection_error'] = str(error)
            report['samples'].append(sample)
    report['downloaded_bytes'] = remote.downloaded
    (args.output / 'inspection.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
