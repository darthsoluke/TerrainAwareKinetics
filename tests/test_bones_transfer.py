import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from scripts.transfer_bones_originals import (
    GIB, extract_archive, index_source, read_manifest, relative_path, require_space,
)


def archive(path, name, data):
    with tarfile.open(path, 'w:gz') as stream:
        item = tarfile.TarInfo(name)
        item.size = len(data)
        stream.addfile(item, io.BytesIO(data))


def test_index_preserves_source_and_date_paths_and_excludes_mirrors(tmp_path):
    source = tmp_path / 'source'
    (source / '230118').mkdir(parents=True)
    for name in ('walk__A1.bvh', 'walk__A1_M.bvh', 'crawl__A2.bvh'):
        (source / '230118' / name).write_bytes(name.encode())
    before = {p.name: p.read_bytes() for p in source.rglob('*.bvh')}
    work = tmp_path / 'work'
    index_source(SimpleNamespace(source=source, work=work, expected_files=2, workers=2))
    rows = read_manifest(work / 'manifest.jsonl')
    assert [r['path'] for r in rows] == ['230118/crawl__A2.bvh', '230118/walk__A1.bvh']
    assert {p.name: p.read_bytes() for p in source.rglob('*.bvh')} == before
    assert json.loads((work / 'index_summary.json').read_text())['files'] == 2


@pytest.mark.parametrize('name', ['/x.bvh', '../x.bvh', '230118/../x.bvh', '230118/x_M.bvh'])
def test_manifest_rejects_escape_or_mirror_paths(name):
    with pytest.raises(ValueError):
        relative_path(name)


def test_exact_extraction_and_existing_file_protection(tmp_path):
    data = b'original motion bytes\n'
    row = dict(path='230118/walk.bvh', size=len(data), sha256=hashlib.sha256(data).hexdigest())
    package = tmp_path / '230118.tar.gz'
    archive(package, row['path'], data)
    destination = tmp_path / 'destination'
    destination.mkdir()
    extract_archive(package, destination, [row])
    assert (destination / row['path']).read_bytes() == data
    extract_archive(package, destination, [row])
    (destination / row['path']).write_bytes(b'keep existing content')
    with pytest.raises(ValueError, match='Existing destination differs'):
        extract_archive(package, destination, [row])
    assert (destination / row['path']).read_bytes() == b'keep existing content'


def test_wrong_hash_does_not_publish_destination(tmp_path):
    package = tmp_path / 'date.tar.gz'
    archive(package, '230118/walk.bvh', b'changed')
    destination = tmp_path / 'destination'
    destination.mkdir()
    row = dict(path='230118/walk.bvh', size=7, sha256=hashlib.sha256(b'correct').hexdigest())
    with pytest.raises(ValueError, match='hash differs'):
        extract_archive(package, destination, [row])
    assert not (destination / row['path']).exists()


def test_space_gate_keeps_reserve(monkeypatch, tmp_path):
    monkeypatch.setattr('scripts.transfer_bones_originals.shutil.disk_usage',
                        lambda path: SimpleNamespace(free=300 * GIB))
    assert require_space(tmp_path, 100 * GIB, 200 * GIB) == 300 * GIB
    with pytest.raises(ValueError, match='Insufficient space'):
        require_space(tmp_path, 101 * GIB, 200 * GIB)
