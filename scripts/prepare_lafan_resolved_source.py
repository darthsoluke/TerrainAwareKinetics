"""Audit and stage an already-downloaded LAFAN1-Resolved BVH archive.

This is source preparation only: it does not retarget, resample, or train.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import zipfile
import zlib
from pathlib import Path, PurePosixPath


_REPO_ROOT = Path(__file__).resolve().parents[1]
_KIMODO_ROOT = _REPO_ROOT / "out" / "vendor" / "kimodo"
if str(_KIMODO_ROOT) not in sys.path:
    sys.path.insert(0, str(_KIMODO_ROOT))


def _load_bvh_parser():
    """Load the vendor parser without importing Kimodo's model package."""
    parser_path = _KIMODO_ROOT / "kimodo" / "skeleton" / "bvh.py"
    if not parser_path.is_file():
        package_spec = importlib.util.find_spec("kimodo")
        if package_spec is None or package_spec.origin is None:
            raise ImportError("The official Kimodo BVH parser is required")
        parser_path = Path(package_spec.origin).parent / "skeleton" / "bvh.py"
    spec = importlib.util.spec_from_file_location("_lafan_source_bvh_parser", parser_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official BVH parser at {parser_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Bvh


Bvh = _load_bvh_parser()


_CANDIDATE = re.compile(r"(?:walk[1-4]|run[1-2])_subject[1-5]\.bvh\Z")
_SUBJECT = re.compile(r"_subject([1-5])\.bvh\Z")
_CHUNK_SIZE = 1024 * 1024


def _safe_member_name(name):
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (not normalized or normalized.startswith("/") or ".." in path.parts
            or re.match(r"^[A-Za-z]:", normalized)):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    return normalized


def _validate_archive_names(infos):
    seen = set()
    candidates = []
    output_names = set()
    for info in infos:
        name = _safe_member_name(info.filename)
        if name in seen:
            raise ValueError(f"Duplicate ZIP member: {name}")
        seen.add(name)
        basename = PurePosixPath(name).name
        if _CANDIDATE.fullmatch(basename):
            folded = basename.casefold()
            if folded in output_names:
                raise ValueError(f"Duplicate candidate output filename: {basename}")
            output_names.add(folded)
            candidates.append((info, basename))
    if not candidates:
        raise ValueError("Archive contains no supported walk/run candidate BVHs")
    return sorted(candidates, key=lambda item: (item[1], item[0].filename))


def _read_bvh_header(stream):
    """Feed only hierarchy and motion-header lines to the structured parser."""
    lines = []
    for raw_line in stream:
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("BVH header is not UTF-8") from error
        lines.append(line)
        if line.strip().startswith("Frame Time:"):
            break
    else:
        raise ValueError("BVH is missing its Frame Time header")
    parser = Bvh("".join(lines), backend="graph")
    frame_time = parser.frame_time
    frame_count = parser.nframes
    if not math.isfinite(frame_time) or frame_time <= 0 or frame_count <= 0:
        raise ValueError("BVH frame time and frame count must be positive")
    return parser, frame_count, 1.0 / frame_time


def _skeleton(parser):
    result = []
    joints = parser.get_joints()
    names = [joint.name for joint in joints]
    if len(names) != len(set(names)):
        raise ValueError("BVH skeleton contains duplicate joint names")
    for joint in joints:
        try:
            offset = [float(value) for value in joint["OFFSET"][:3]]
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid OFFSET for joint {joint.name}") from error
        if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
            raise ValueError(f"Invalid OFFSET for joint {joint.name}")
        parent = None if joint.parent == parser.root else joint.parent.name
        result.append({"name": joint.name, "parent": parent, "offset": offset,
                       "channels": parser.joint_channels(joint.name)})
    if not result:
        raise ValueError("BVH contains no root joint")
    return result


def _verify_skeleton(candidate, bind, filename):
    if len(candidate) != len(bind):
        raise ValueError(f"Skeleton joint count differs from Geno_bind.bvh: {filename}")
    for actual, expected in zip(candidate, bind):
        if actual["name"] != expected["name"] or actual["parent"] != expected["parent"]:
            raise ValueError(f"Skeleton hierarchy differs from Geno_bind.bvh: {filename}")
        if actual["channels"] != expected["channels"]:
            raise ValueError(f"Skeleton channels differ from Geno_bind.bvh: {filename}/{actual['name']}")
    # Upstream exports offsets=bone_local_positions[0], not invariant bind offsets.
    differences = [math.sqrt(sum((a - b) ** 2 for a, b in zip(actual["offset"], expected["offset"])))
                   for actual, expected in zip(candidate, bind)]
    return {"root_offset_difference_raw_units": differences[0],
            "max_nonroot_offset_difference_raw_units": max(differences[1:], default=0.0),
            "offsets_identical_to_bind": all(value == 0 for value in differences)}


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_source(archive_path, bind_path, output_dir):
    """Validate and stage supported files from local inputs into a new directory."""
    archive_path = Path(archive_path)
    bind_path = Path(bind_path)
    output_dir = Path(output_dir)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive does not exist: {archive_path}")
    if not bind_path.is_file():
        raise FileNotFoundError(f"Bind BVH does not exist: {bind_path}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new: {output_dir}")

    with bind_path.open("rb") as stream:
        bind_parser, _, _ = _read_bvh_header(stream)
    bind_skeleton = _skeleton(bind_parser)

    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        candidates = _validate_archive_names(infos)
        try:
            bad_member = archive.testzip()
        except (zipfile.BadZipFile, EOFError, RuntimeError, zlib.error) as error:
            raise ValueError("ZIP CRC validation failed while reading a member") from error
        if bad_member is not None:
            raise ValueError(f"ZIP CRC validation failed for member: {bad_member}")
        archive_digest = _sha256_file(archive_path)

        # Parse all headers and compare hierarchy/channels before creating files.
        candidate_headers = []
        for info, basename in candidates:
            with archive.open(info, "r") as stream:
                parser, frame_count, fps = _read_bvh_header(stream)
            skeleton = _skeleton(parser)
            offset_comparison = _verify_skeleton(skeleton, bind_skeleton, basename)
            subject = int(_SUBJECT.search(basename).group(1))
            candidate_headers.append((info, basename, frame_count, fps, subject, skeleton, offset_comparison))

        output_dir.mkdir(parents=True, exist_ok=False)
        records = []
        try:
            for info, basename, frame_count, fps, subject, skeleton, offset_comparison in candidate_headers:
                destination = output_dir / basename
                digest = hashlib.sha256()
                with archive.open(info, "r") as source, destination.open("xb") as target:
                    for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
                        digest.update(chunk)
                        target.write(chunk)
                records.append({
                    "archive_member": info.filename,
                    "filename": basename,
                    "recording_id": Path(basename).stem,
                    "subject_id": f"subject{subject}",
                    "heldout_split_key": f"subject{subject}",
                    "staged_path": basename,
                    "sha256": digest.hexdigest(),
                    "source_fps": fps,
                    "motion_frame_count": frame_count,
                    "header_offsets_raw_units": [joint["offset"] for joint in skeleton],
                    "offset_comparison_to_bind": offset_comparison,
                })
        except Exception:
            # Preserve the fresh output directory for inspection; no manifest claims success.
            raise

    manifest = {
        "manifest_version": 1,
        "source_only": True,
        "training_eligible": False,
        "SOMA_retarget_verified": False,
        "processing_performed": {"retargeting": False, "resampling": False, "training": False},
        "source": {
            "repository": "https://github.com/orangeduck/lafan1-resolved",
            "archive_path": str(archive_path.resolve()),
            "archive_sha256": archive_digest,
            "crc_validated": True,
            "archive_bvh_count": sum(info.filename.lower().endswith(".bvh") for info in infos),
            "bind_bvh_path": str(bind_path.resolve()),
            "bind_bvh_sha256": _sha256_file(bind_path),
            "candidate_pattern": _CANDIDATE.pattern,
        },
        "skeleton": {
            "joint_names": [joint["name"] for joint in bind_skeleton],
            "parents": [joint["parent"] for joint in bind_skeleton],
            "offsets": [joint["offset"] for joint in bind_skeleton],
            "channels": [joint["channels"] for joint in bind_skeleton],
            "offset_units": (
                "Raw BVH distance units. Upstream export_bvh.py writes MotionBuilder FBMatrix "
                "translations directly as offsets without a conversion; example bind offsets "
                "are centimeter-scale. No unit conversion performed."
            ),
            "hierarchy_and_channels_verified_against_bind": True,
            "constant_bind_offsets_assumed": False,
            "offset_semantics": "Upstream export_bvh.py uses bone_local_positions[0] for each clip's offsets",
            "nonroot_translation_channels_present": any(
                channel.endswith("position") for joint in bind_skeleton[1:]
                for channel in joint["channels"]),
            "full_animation_channel_import_verified": False,
            "conversion_requirement": (
                "Preserve per-joint animated local translations and native Euler order; "
                "do not double-add OFFSET to absolute local position channels. "
                "Header parsing is not proof of correct motion conversion."
            ),
            "exporter_reference": "https://github.com/orangeduck/lafan1-resolved/blob/main/export_bvh.py",
        },
        "license": {
            "identifier": "CC BY-NC-ND 4.0",
            "commercial_use_approved": False,
            "distribution_approved": False,
            "note": "No additional commercial or distribution approval has been obtained.",
            "source_license_url": "https://github.com/orangeduck/lafan1-resolved/blob/main/license.txt",
        },
        "candidate_files": records,
        "candidate_duration_seconds": sum(record["motion_frame_count"] / record["source_fps"]
                                          for record in records),
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Existing bvh.zip")
    parser.add_argument("--bind", type=Path, required=True, help="Existing Geno_bind.bvh")
    parser.add_argument("--output", type=Path, required=True, help="New, empty destination directory")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_source(args.archive, args.bind, args.output)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({"staged_candidates": len(manifest["candidate_files"]),
                      "manifest": str(args.output / "source_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
