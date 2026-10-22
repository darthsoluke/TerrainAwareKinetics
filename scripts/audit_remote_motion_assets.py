"""Inventory existing motion assets without loading checkpoints or modifying data."""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def bvh_header(path):
    joints = []
    parents = []
    stack = []
    pending_joint = None
    channels = 0
    frames = None
    frame_time = None
    with path.open(encoding="utf-8-sig", errors="strict") as handle:
        for index, line in enumerate(handle):
            if index > 10000:
                raise ValueError("BVH header exceeds audit limit")
            fields = line.strip().split()
            if not fields:
                continue
            if fields[0] in ("ROOT", "JOINT"):
                pending_joint = len(joints)
                joints.append(" ".join(fields[1:]))
                parents.append(stack[-1] if stack else -1)
            elif fields[:2] == ["End", "Site"]:
                pending_joint = None
            elif fields[0] == "{":
                stack.append(pending_joint)
                pending_joint = None
            elif fields[0] == "}":
                stack.pop()
            elif fields[0] == "CHANNELS":
                channels += int(fields[1])
            elif fields[0] == "Frames:":
                frames = int(fields[1])
            elif fields[:2] == ["Frame", "Time:"]:
                frame_time = float(fields[2])
                break
    if not joints or frames is None or frame_time is None:
        raise ValueError("Incomplete BVH header")
    return {
        "path": str(path), "joint_names": joints, "parents": parents,
        "channels": channels, "frames": frames, "fps": 1.0 / frame_time,
        "duration_seconds": frames * frame_time,
    }


def inventory_directory(root, sample_count=8, filename_prefix=None):
    root = Path(root)
    result = {"path": str(root), "exists": root.exists(), "filename_prefix": filename_prefix}
    if not root.exists():
        return result
    extensions = Counter()
    total_bytes = 0
    bvh_samples = []
    errors = []
    links = broken_links = 0

    def record_error(error):
        if len(errors) < 10:
            errors.append(str(error))

    for directory, _, filenames in os.walk(root, onerror=record_error):
        for name in filenames:
            if filename_prefix is not None and not name.startswith(filename_prefix):
                continue
            path = Path(directory) / name
            try:
                links += int(path.is_symlink())
                if path.is_symlink() and not path.exists():
                    broken_links += 1
                    continue
                size = path.stat().st_size
                total_bytes += size
                extension = ".tar.zst" if name.endswith(".tar.zst") else path.suffix.lower()
                extensions[extension or "<none>"] += 1
                if extension == ".bvh" and len(bvh_samples) < sample_count:
                    try:
                        bvh_samples.append(bvh_header(path))
                    except (OSError, ValueError, IndexError, UnicodeError) as error:
                        record_error(error)
            except OSError as error:
                record_error(error)
    result.update({
        "files_by_extension": dict(extensions), "total_bytes": total_bytes,
        "file_symlinks": links, "broken_file_symlinks": broken_links,
        "sample_bvh_headers": bvh_samples, "errors": errors,
        "validation_scope": f"File inventory and up to {sample_count} BVH headers; no full integrity or motion quality validation",
    })
    return result


def corpus_sources(path):
    groups = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 3:
                groups[row[0]].append(row[2])
    results = {}
    for name, files in sorted(groups.items()):
        step = max(1, len(files) // 32)
        sampled = files[::step][:32]
        existing = missing = inaccessible = 0
        for item in sampled:
            try:
                if Path(item).is_file():
                    existing += 1
                else:
                    missing += 1
            except OSError:
                inaccessible += 1
        results[name] = {
            "listed_files": len(files), "unique_paths": len(set(files)),
            "common_root": os.path.commonpath(files),
            "sampled_paths": len(sampled),
            "sampled_existing_paths": existing,
            "sampled_missing_paths": missing,
            "sampled_inaccessible_paths": inaccessible,
            "first_path": files[0],
        }
    return results


def persona_manifest(path):
    bodies = Counter()
    clips = set()
    fps = Counter()
    duration = 0.0
    rows = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            bodies[row["body"]] += 1
            clips.add(row["clip"])
            fps[row["fps"]] += 1
            duration += float(row["duration_s"])
    archive_root = Path(path).parent / "smpl_skeleton"
    formats = {}
    for name in ("smpl", "bvh"):
        available = {item.name.removesuffix(".tar.zst") for item in (archive_root / name).glob("*.tar.zst")}
        formats[name] = {
            "archives": len(available),
            "missing_body_archives": sorted(set(bodies) - available),
            "unexpected_body_archives": sorted(available - set(bodies)),
        }
    return {
        "path": str(path), "rows": rows, "bodies": len(bodies),
        "unique_source_clips": len(clips), "fps_distribution": dict(fps),
        "expanded_duration_hours": duration / 3600.0, "formats": formats,
        "validation_scope": "Manifest parsed and archive names matched; archives not fully decompressed or checksummed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-existing-inventory", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save_report():
        output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    paths = {
        "bones_seed_soma_uniform_originals": "${PRIVATE_STORAGE}/bones-seed/uniform_orig/soma_uniform/bvh",
        "motionpersona_originals": "${PRIVATE_STORAGE}/t7/motionpersona/100style++/refined/BVH",
        "motionpersona_x_release": "${PRIVATE_STORAGE}/release/MotionPersonaX",
        "motion100_processed": "${PRIVATE_STORAGE}/t7/motion100/motion100/processed_add_T_pose",
        "lafan1_test_samples": "${PRIVATE_STORAGE}/code/motionfit/test_data",
        "motionpersona_public_code": "${PRIVATE_STORAGE}/code/persona_pub_data/code",
    }
    report = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "datasets": {}, "training_started": False,
    }
    if args.reuse_existing_inventory and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("host") != report["host"]:
            raise ValueError("Cannot reuse inventory from another host")
        report["datasets"] = previous.get("datasets", {})
        report["reused_inventory_observed_at_utc"] = previous.get("observed_at_utc")
    for name, path in paths.items():
        prefix = "lafan1_" if name == "lafan1_test_samples" else None
        sample_count = 32 if prefix is not None else 8
        cached = report["datasets"].get(name)
        sample_target = min(sample_count, cached.get("files_by_extension", {}).get(".bvh", 0)) if cached else 0
        if not (cached and cached.get("path") == path and cached.get("filename_prefix") == prefix
                and len(cached.get("sample_bvh_headers", [])) >= sample_target):
            report["datasets"][name] = inventory_directory(path, sample_count=sample_count, filename_prefix=prefix)
        save_report()
        print(json.dumps({"completed": name}, ensure_ascii=True), flush=True)
    report["corpus_source_index"] = corpus_sources("${PRIVATE_STORAGE}/full_corpus/farm.tsv")
    report["motionpersona_x_manifest"] = persona_manifest("${PRIVATE_STORAGE}/release/MotionPersonaX/manifest.csv")
    status = Path("${PRIVATE_STORAGE}/full_corpus/ingest_status.txt")
    report["existing_ingest_status"] = status.read_text(encoding="utf-8").splitlines()
    save_report()
    print(json.dumps({"report": str(output), "corpus_source_index": report["corpus_source_index"],
                      "motionpersona_x_manifest": report["motionpersona_x_manifest"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
