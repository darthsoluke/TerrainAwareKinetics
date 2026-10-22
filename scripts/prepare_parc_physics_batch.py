"""Build a create-once local input bundle for official PARC physics recording."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "out/parc_complete_20261008"
VENDOR = "out/vendor/PARC"
TRACKER = "out/parc_legacy_models_20261008/extracted/april272025/iter_3/p3_tracker"
DEFAULT_OUTPUT = "out/parc_physics_20261009/inputs_v1"
EXPECTED_MOTION_COUNT = 160
EXPECTED_REPRESENTATIVE_COUNT = 7
MAX_BUNDLE_BYTES = 1_000_000_000
BLOCK_SIZE = 4 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceFile:
    source: Path
    archive_path: str
    expected_sha256: str | None = None
    candidate_key: str | None = None
    source_frames: int | None = None


@dataclass(frozen=True)
class PlannedFile:
    source: Path
    archive_path: str
    sha256: str
    size_bytes: int
    candidate_key: str | None = None
    source_frames: int | None = None


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.hasher = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.hasher.update(block)
        self.bytes_read += len(block)
        return block


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            value.update(block)
    return value.hexdigest()


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_archive_path(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or "\x00" in value or "\r" in value or "\n" in value):
        raise ValueError(f"Non-canonical archive path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Non-canonical archive path: {value!r}")
    return value


def is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def checked_source(root: Path, relative: str) -> Path:
    relative = canonical_archive_path(relative)
    root = Path(root).absolute()
    path = root.joinpath(*PurePosixPath(relative).parts)
    current = path
    while True:
        if is_link(current):
            raise ValueError(f"Symlinked source rejected: {current}")
        if current == root:
            break
        current = current.parent
    resolved = path.resolve(strict=True)
    if resolved != root.resolve(strict=True).joinpath(*PurePosixPath(relative).parts) or not resolved.is_file():
        raise ValueError(f"Missing or non-regular source file: {path}")
    return path


def read_json_snapshot(root: Path, relative: str) -> tuple[dict, str]:
    path = checked_source(root, relative)
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"Invalid SHA-256 in {label}")
    return value


def walk_official_python(repo_root: Path) -> list[Path]:
    package_root = repo_root / "PARC"
    if is_link(package_root) or not package_root.is_dir():
        raise ValueError(f"Missing or symlinked official PARC tree: {package_root}")
    python_files = []
    for current, dirs, files in os.walk(package_root, followlinks=False):
        current_path = Path(current)
        for name in list(dirs):
            child = current_path / name
            if is_link(child):
                raise ValueError(f"Symlinked official PARC directory rejected: {child}")
        for name in files:
            child = current_path / name
            if is_link(child):
                raise ValueError(f"Symlinked official PARC file rejected: {child}")
            if child.suffix == ".py":
                python_files.append(child)
    if not python_files:
        raise ValueError("Official PARC Python tree is empty")
    return sorted(python_files, key=lambda path: path.relative_to(repo_root).as_posix())


def referenced_xml_files(root: Path, repo_root: Path, xml_path: Path) -> list[Path]:
    """Return regular files referenced by MJCF file/include attributes."""
    root = Path(root).absolute()
    repo_root = Path(repo_root).absolute()
    pending = [xml_path]
    visited_xml = set()
    referenced = set()

    while pending:
        current_xml = pending.pop()
        resolved_xml = current_xml.resolve(strict=True)
        if resolved_xml in visited_xml:
            continue
        visited_xml.add(resolved_xml)
        tree = ET.parse(resolved_xml)
        xml_root = tree.getroot()
        compiler = next((node for node in xml_root.iter() if node.tag == "compiler"), None)
        compiler_dirs = dict(compiler.attrib) if compiler is not None else {}

        for node in xml_root.iter():
            value = node.attrib.get("file")
            if not value:
                continue
            raw = PurePosixPath(value)
            if raw.is_absolute() or "\\" in value or ".." in raw.parts:
                raise ValueError(f"Unsafe XML asset reference in {resolved_xml}: {value!r}")
            if node.tag == "include":
                base = resolved_xml.parent
            else:
                dir_key = {"mesh": "meshdir", "texture": "texturedir"}.get(node.tag, "assetdir")
                directory = compiler_dirs.get(dir_key, "")
                directory_path = PurePosixPath(directory) if directory else PurePosixPath()
                if (directory_path.is_absolute() or "\\" in directory
                        or ".." in directory_path.parts or ":" in directory):
                    raise ValueError(f"Unsafe XML asset directory in {resolved_xml}: {directory!r}")
                base = resolved_xml.parent / Path(*directory_path.parts)
            candidate = base / Path(*raw.parts)
            try:
                candidate.relative_to(repo_root)
            except ValueError as error:
                raise ValueError(f"XML asset escapes official PARC tree: {value!r}") from error
            asset_relative = candidate.relative_to(root).as_posix()
            asset_path = checked_source(root, asset_relative)
            asset_path.resolve(strict=True).relative_to(repo_root.resolve(strict=True))
            referenced.add(asset_path)
            if node.tag == "include":
                pending.append(asset_path)

    return sorted(referenced, key=lambda path: path.relative_to(repo_root).as_posix())


def collect_sources(root: Path, expected_motion_count: int) -> tuple[list[SourceFile], dict, dict]:
    root = Path(root).resolve()
    collection_root = root / COLLECTION
    catalog_relative = f"{COLLECTION}/gallery/catalog.json"
    summary_relative = f"{COLLECTION}/summary.json"
    catalog_path = checked_source(root, catalog_relative)
    summary_path = checked_source(root, summary_relative)
    catalog, catalog_sha = read_json_snapshot(root, catalog_relative)
    summary, summary_sha = read_json_snapshot(root, summary_relative)

    candidates = catalog.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != expected_motion_count:
        raise ValueError(f"Expected {expected_motion_count} catalog candidates")
    if catalog.get("candidate_count") != len(candidates):
        raise ValueError("Catalog candidate_count does not match candidate records")

    candidates_by_key = {}
    source_paths = set()
    for row in candidates:
        key = row.get("candidate_key")
        if not isinstance(key, str) or not key or key in candidates_by_key:
            raise ValueError(f"Missing or duplicate candidate ID: {key!r}")
        source_rel = canonical_archive_path(row.get("source_path"))
        if PurePosixPath(source_rel).name != "generated.pkl":
            raise ValueError(f"Candidate source is not generated.pkl: {source_rel}")
        if source_rel in source_paths:
            raise ValueError(f"Duplicate candidate source path: {source_rel}")
        source_paths.add(source_rel)
        frames = row.get("frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
            raise ValueError(f"Invalid source frame count for candidate {key}")
        expected_sha = validate_sha256(row.get("reported_source_sha256"), key)
        report_entry = row.get("report_entry")
        if isinstance(report_entry, dict):
            report_sha = report_entry.get("sha256")
            report_frames = report_entry.get("frames")
            if report_sha is not None and validate_sha256(report_sha, key) != expected_sha:
                raise ValueError(f"Catalog/report hash disagreement for candidate {key}")
            if report_frames is not None and report_frames != frames:
                raise ValueError(f"Catalog/report frame disagreement for candidate {key}")
        candidates_by_key[key] = (row, source_rel, expected_sha, frames)

    showcase = summary.get("showcase")
    if not isinstance(showcase, list) or len(showcase) != EXPECTED_REPRESENTATIVE_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_REPRESENTATIVE_COUNT} summary showcase candidates")
    showcase_keys = []
    showcase_names = set()
    for entry in showcase:
        key = entry.get("candidate_key")
        name = entry.get("name")
        if key not in candidates_by_key or key in showcase_keys:
            raise ValueError(f"Missing or duplicate showcase candidate ID: {key!r}")
        if not isinstance(name, str) or not name or name in showcase_names:
            raise ValueError(f"Missing or duplicate showcase name: {name!r}")
        showcase_keys.append(key)
        showcase_names.add(name)

    motion_files = []
    motion_path_by_key = {}
    for number, key in enumerate(sorted(candidates_by_key)):
        _, source_rel, expected_sha, frames = candidates_by_key[key]
        archive_path = f"motions/motion_{number:03d}_{expected_sha[:12]}.pkl"
        motion_path_by_key[key] = archive_path
        motion_files.append(SourceFile(
            source=checked_source(root, f"{COLLECTION}/downloads/{source_rel}"),
            archive_path=archive_path,
            expected_sha256=expected_sha,
            candidate_key=key,
            source_frames=frames,
        ))

    repo_root = checked_source(root, f"{VENDOR}/PARC/__init__.py").parents[1]
    sources = list(motion_files)
    for path in walk_official_python(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        sources.append(SourceFile(path, f"PARC/{relative}"))

    required_repo_files = (
        "scripts/run_tracker.py",
        "scripts/__init__.py",
        "data/assets/humanoid.xml",
    )
    for relative in required_repo_files:
        sources.append(SourceFile(checked_source(repo_root, relative), f"PARC/{relative}"))

    xml_path = repo_root / "data/assets/humanoid.xml"
    for path in referenced_xml_files(root, repo_root, xml_path):
        relative = path.relative_to(repo_root).as_posix()
        sources.append(SourceFile(path, f"PARC/{relative}"))

    for name in ("model.pt", "dm_env.yaml", "agent_config.yaml"):
        relative = f"{TRACKER}/{name}"
        sources.append(SourceFile(checked_source(root, relative), f"tracker/{name}"))

    archive_paths = [canonical_archive_path(item.archive_path) for item in sources]
    if len(set(archive_paths)) != len(archive_paths):
        raise ValueError("Duplicate archive paths in bundle plan")

    metadata = {
        "catalog_path": f"{COLLECTION}/gallery/catalog.json",
        "catalog_sha256": catalog_sha,
        "summary_path": f"{COLLECTION}/summary.json",
        "summary_sha256": summary_sha,
        "motion_path_by_key": motion_path_by_key,
        "showcase": showcase,
    }
    return sources, metadata, {"catalog": catalog_path, "summary": summary_path}


def plan_files(sources: list[SourceFile]) -> list[PlannedFile]:
    planned = []
    for item in sources:
        if is_link(item.source) or not item.source.is_file():
            raise ValueError(f"Missing or symlinked bundle input: {item.source}")
        size = item.source.stat().st_size
        before = digest(item.source)
        if item.expected_sha256 is not None and before != item.expected_sha256:
            raise ValueError(f"Catalog hash mismatch: {item.archive_path}")
        planned.append(PlannedFile(item.source, item.archive_path, before, size,
                                   item.candidate_key, item.source_frames))
    return sorted(planned, key=lambda item: item.archive_path)


def file_records(planned: list[PlannedFile]) -> list[dict]:
    return [{"path": item.archive_path, "sha256": item.sha256, "bytes": item.size_bytes}
            for item in planned]


def source_fingerprint(records: list[dict]) -> str:
    encoded = json.dumps(records, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def estimate_tar_size(planned: list[PlannedFile], bundle_index: bytes) -> int:
    payload_bytes = sum(item.size_bytes for item in planned) + len(bundle_index)
    file_count = len(planned) + 1
    padded = sum(((item.size_bytes + 511) // 512) * 512 for item in planned)
    padded += ((len(bundle_index) + 511) // 512) * 512
    return padded + file_count * 512 + 1024


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(canonical_archive_path(name))
    info.size = len(data)
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


def add_planned_file(archive: tarfile.TarFile, item: PlannedFile) -> None:
    info = tarfile.TarInfo(canonical_archive_path(item.archive_path))
    info.size = item.size_bytes
    info.mode = stat.S_IMODE(item.source.stat().st_mode) & 0o777 or 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    with item.source.open("rb") as stream:
        reader = HashingReader(stream)
        archive.addfile(info, reader)
    if reader.bytes_read != item.size_bytes or reader.hasher.hexdigest() != item.sha256:
        raise RuntimeError(f"Input changed or copy hash mismatch: {item.archive_path}")
    if digest(item.source) != item.sha256:
        raise RuntimeError(f"Source changed during bundle copy: {item.archive_path}")


def verify_archive(archive_path: Path, planned: list[PlannedFile], bundle_index: bytes) -> None:
    expected = {item.archive_path: item for item in planned}
    expected["_bundle.json"] = None
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise RuntimeError("Archive member set is not the planned canonical file set")
        for member in members:
            canonical_archive_path(member.name)
            if not member.isfile() or member.issym() or member.islnk():
                raise RuntimeError(f"Non-regular archive member rejected: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Unable to read archive member: {member.name}")
            data_hash = hashlib.sha256()
            collected = bytearray() if member.name == "_bundle.json" else None
            for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
                data_hash.update(block)
                if collected is not None:
                    collected.extend(block)
            if member.name == "_bundle.json":
                if bytes(collected) != bundle_index:
                    raise RuntimeError("Embedded bundle index mismatch")
            elif data_hash.hexdigest() != expected[member.name].sha256:
                raise RuntimeError(f"Archive content hash mismatch: {member.name}")


def write_exclusive_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_bundle(root: Path, output_dir: Path, expected_motion_count: int = EXPECTED_MOTION_COUNT,
                 max_bundle_bytes: int = MAX_BUNDLE_BYTES) -> dict:
    root = Path(root).absolute()
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or is_link(output_dir):
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    for parent in output_dir.parents:
        if is_link(parent):
            raise ValueError(f"Symlinked output parent rejected: {parent}")

    sources, metadata, metadata_paths = collect_sources(root, expected_motion_count)
    planned = plan_files(sources)
    records = file_records(planned)
    by_archive_path = {item.archive_path: item for item in planned}
    motion_map = {}
    for item in planned:
        if item.candidate_key is not None:
            motion_map[item.archive_path] = {
                "candidate_key": item.candidate_key,
                "sha256": item.sha256,
                "source_frames": item.source_frames,
            }
    representatives = []
    for entry in metadata["showcase"]:
        archive_path = metadata["motion_path_by_key"][entry["candidate_key"]]
        item = by_archive_path[archive_path]
        representatives.append({
            "name": entry["name"],
            "path": archive_path,
            "candidate_key": item.candidate_key,
        })

    source_sha = source_fingerprint(records)
    created_at = stamp()
    embedded = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "source_sha256": source_sha,
        "metadata_inputs": {
            "catalog": {"path": metadata["catalog_path"], "sha256": metadata["catalog_sha256"]},
            "summary": {"path": metadata["summary_path"], "sha256": metadata["summary_sha256"]},
        },
        "files": records,
        "motions": motion_map,
        "representatives": representatives,
    }
    embedded_bytes = (json.dumps(embedded, indent=2, sort_keys=True) + "\n").encode("utf-8")
    estimate = estimate_tar_size(planned, embedded_bytes)
    if estimate >= max_bundle_bytes:
        raise ValueError(f"Estimated tar size {estimate} bytes reaches the {max_bundle_bytes}-byte limit")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    archive_path = output_dir / "parc_physics_inputs_v1.tar"
    manifest_path = output_dir / "manifest.json"
    try:
        with tarfile.open(archive_path, "x") as archive:
            for item in planned:
                add_planned_file(archive, item)
            add_bytes(archive, "_bundle.json", embedded_bytes)

        verify_archive(archive_path, planned, embedded_bytes)
        archive_size = archive_path.stat().st_size
        if archive_size >= max_bundle_bytes:
            raise ValueError(f"Tar size {archive_size} bytes reaches the {max_bundle_bytes}-byte limit")
        archive_sha = digest(archive_path)
        for name, path in metadata_paths.items():
            expected = metadata[f"{name}_sha256"]
            if digest(path) != expected:
                raise RuntimeError(f"Catalog metadata changed during bundle preparation: {path}")

        manifest = {
            "schema_version": 1,
            "created_at_utc": created_at,
            "archive": {
                "path": archive_path.name,
                "sha256": archive_sha,
                "size_bytes": archive_size,
            },
            "source_sha256": source_sha,
            "metadata_inputs": embedded["metadata_inputs"],
            "motion_count": len(motion_map),
            "representatives": representatives,
            "motions": motion_map,
            "files": records,
        }
        write_exclusive_json(manifest_path, manifest)
        return manifest
    except Exception:
        for path in (archive_path, manifest_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="motionbricks repository root containing out/")
    parser.add_argument("--output-dir", type=Path, default=ROOT / DEFAULT_OUTPUT,
                        help="new output directory; existing paths are never overwritten")
    args = parser.parse_args(argv)
    manifest = build_bundle(args.root, args.output_dir)
    print(json.dumps({
        "archive": str((args.output_dir / manifest["archive"]["path"]).resolve()),
        "archive_sha256": manifest["archive"]["sha256"],
        "archive_size_bytes": manifest["archive"]["size_bytes"],
        "motion_count": manifest["motion_count"],
        "representative_count": len(manifest["representatives"]),
        "manifest": str((args.output_dir / "manifest.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
