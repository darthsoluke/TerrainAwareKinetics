"""Expand only BONES training recordings, preserving the frozen pilot evaluation data."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def select_training_expansion(pilot, candidates, max_train):
    """Keep recording/propagation families intact, then balance behaviors and families."""
    pilot_train = sum(row["split"] == "train" for row in pilot)
    if max_train <= pilot_train:
        raise ValueError("max_train must exceed the existing training clip count")
    families = defaultdict(set)
    candidate_paths = {}
    for row in candidates + pilot:
        if row["split"] not in {"train", "validation", "test"} or not row["recording_group"]:
            raise ValueError("Missing recording family or invalid split")
        families[row["recording_group"]].add(row["split"])
    if any(len(splits) != 1 for splits in families.values()):
        raise ValueError("Recording/propagation family crosses splits")
    for row in candidates:
        if row["path"] in candidate_paths:
            raise ValueError("Duplicate candidate path")
        candidate_paths[row["path"]] = row
    for row in pilot:
        candidate = candidate_paths.get(row["path"])
        if candidate is None or any(candidate[key] != row[key] for key in
                                    ("filename", "split", "recording_group", "behavior")):
            raise ValueError("Pilot identity does not match the frozen candidate manifest")

    reserved = {group for group, splits in families.items() if "train" not in splits}
    existing_names = {row["filename"].casefold() for row in pilot}
    buckets = defaultdict(list)
    for row in candidates:
        if (row["split"] == "train" and row["recording_group"] not in reserved
                and row["filename"].casefold() not in existing_names):
            buckets[(row["behavior"], row["recording_group"])].append(row)
    for key, rows in buckets.items():
        buckets[key] = deque(sorted(rows, key=lambda row: hashlib.sha256(
            row["path"].encode("utf-8")).hexdigest()))
    group_queues = defaultdict(deque)
    for behavior, group in sorted(buckets):
        group_queues[behavior].append(group)
    selected = []
    while len(selected) < max_train - pilot_train and any(group_queues.values()):
        for behavior in sorted(group_queues):
            groups = group_queues[behavior]
            if not groups:
                continue
            group = groups.popleft()
            rows = buckets[(behavior, group)]
            selected.append(rows.popleft())
            if rows:
                groups.append(group)
            if len(selected) == max_train - pilot_train:
                break
    return selected, {
        "existing_train_clips": pilot_train, "target_train_clips": max_train,
        "selected_additional_candidates": len(selected),
        "reserved_families": sorted(reserved),
        "selection": "round robin over behavior, then recording/propagation family; stable path hash within family",
        "selected_behavior_counts": dict(Counter(row["behavior"] for row in selected)),
    }


def validate_source_paths(records, source):
    source = Path(source).resolve()
    for row in records:
        path = Path(row["path"]).resolve()
        if source not in path.parents or path.stem != row["filename"] or not path.is_file():
            raise ValueError(f"Invalid or unavailable BONES source: {path}")


def seed_reserved_hashes(existing, reserved):
    """Also protect unselected held-out candidates from byte-identical train copies."""
    hashes = dict(existing)
    for row in reserved:
        digest = row["source_sha256"]
        if hashes.get(digest) == "train":
            raise ValueError("Existing training content duplicates a reserved held-out candidate")
        hashes.setdefault(digest, row["split"])
    return hashes


def _initialize_conversion(contract):
    import numpy as np
    import torch
    from kimodo.skeleton.registry import build_skeleton
    from motionbricks.data.bones_seed import build_representation, official_sole_offsets

    global _FULL, _REP, _OFFSETS
    torch.set_num_threads(1)
    _FULL, _REP = build_skeleton(77), build_representation()
    if (_REP.skeleton.bone_order_names != contract["joint_names"]
            or _REP.skeleton.joint_parents.tolist() != contract["parents"]
            or contract["up_axis"] != "Y" or contract["fps"] != 60.0):
        raise ValueError("Pilot skeleton/coordinate contract mismatch")
    np.testing.assert_allclose(_REP.skeleton.neutral_joints.numpy(),
                               contract["neutral_joints_m"], atol=1e-7, rtol=0)
    # This is a data-expansion control: retain the pilot's legacy fixed-plane
    # geometry explicitly, without silently using the newer default proxy model.
    offsets, mesh_hash = official_sole_offsets(_REP.skeleton, geometry="position_only_v0")
    np.testing.assert_allclose(offsets.numpy(), contract["sole_offsets_m"], atol=1e-7, rtol=0)
    if mesh_hash != contract["sole_mesh_sha256"]:
        raise ValueError("Pilot SOMA mesh hash mismatch")
    _OFFSETS = torch.tensor(contract["sole_offsets_m"], dtype=torch.float32)


def _convert_record(record):
    from scripts.prepare_bones_locomotion import convert

    before = sha256(record["path"])
    try:
        arrays, report = convert(record, _FULL, _REP, _OFFSETS)
    except (ValueError, RuntimeError, IndexError, KeyError) as error:
        return {"error": str(error), "source_sha256": before}
    if report["source_sha256"] != before:
        raise ValueError("Source file changed during conversion")
    return {"arrays": arrays, "report": report}


def verify_converter_parity(pilot_dir, records):
    import numpy as np

    selected = []
    for behavior in ("walk", "run"):
        selected.append(next(row for row in records
                             if row["split"] == "train" and row["behavior"] == behavior))
    results = []
    for record in selected:
        result = _convert_record(record)
        if "error" in result:
            raise ValueError(f"Existing pilot conversion rejected: {result['error']}")
        if result["report"]["source_sha256"] != record["source_sha256"]:
            raise ValueError("Frozen pilot BVH source content has changed")
        errors = {}
        with np.load(pilot_dir / record["shard"], allow_pickle=False) as previous:
            for name, array in result["arrays"].items():
                old = previous[name]
                if array.dtype == np.bool_:
                    np.testing.assert_array_equal(array, old)
                else:
                    np.testing.assert_allclose(array, old, atol=1e-6, rtol=0)
                errors[name] = float(np.abs(array.astype(float) - old.astype(float)).max())
        results.append({"filename": record["filename"], "max_array_error": errors})
    return results


def expand(pilot_dir, source, output, max_train=2048, workers=8):
    import numpy as np

    pilot_dir, source, output = map(lambda p: Path(p).resolve(), (pilot_dir, source, output))
    if output.exists() or pilot_dir == output or pilot_dir in output.parents or source in output.parents:
        raise ValueError("Use a new output directory outside both source and pilot")
    if workers < 1:
        raise ValueError("workers must be positive")
    pilot = json.loads((pilot_dir / "accepted.json").read_text())
    candidates = [json.loads(line) for line in (pilot_dir / "candidate_manifest.jsonl").read_text().splitlines()]
    contract = json.loads((pilot_dir / "skeleton_contract.json").read_text())
    selected, plan = select_training_expansion(pilot, candidates, max_train)
    reserved_candidates = [row for row in candidates if row["split"] != "train"]
    validate_source_paths(selected + pilot + reserved_candidates, source)
    _initialize_conversion(contract)
    parity = verify_converter_parity(pilot_dir, pilot)
    output.mkdir(parents=True)
    (output / "shards").mkdir()
    write_json(output / "selection_plan.json", {**plan, "candidates": selected})
    write_json(output / "converter_parity.json", parity)
    write_json(output / "skeleton_contract.json", {**contract,
        "sole_proxy_geometry": "position_only_v0",
        "expansion_note": "Historical pilot geometry preserved for data-coverage comparison; not a new contact-quality approval"})
    accepted, rejected, hashes = [], [], {}
    copied_hashes = {}
    for row in pilot:
        source_shard = (pilot_dir / row["shard"]).resolve()
        target_shard = (output / row["shard"]).resolve()
        if pilot_dir not in source_shard.parents or output not in target_shard.parents:
            raise ValueError("Pilot shard escapes its dataset directory")
        digest = row["source_sha256"]
        if digest in hashes:
            raise ValueError("Pilot already contains duplicate source content")
        hashes[digest] = row["split"]
        target_shard.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_shard, target_shard)
        copied_hashes[row["shard"]] = sha256(source_shard)
        if sha256(target_shard) != copied_hashes[row["shard"]]:
            raise ValueError("Copied pilot shard identity mismatch")
        accepted.append(dict(row))
    with ThreadPoolExecutor(max_workers=2) as hashing:
        reserved = [{"path": row["path"], "split": row["split"], "source_sha256": digest}
                    for row, digest in zip(reserved_candidates,
                        hashing.map(sha256, [row["path"] for row in reserved_candidates]))]
    hashes = seed_reserved_hashes(hashes, reserved)
    write_json(output / "reserved_content_hashes.json", reserved)
    print(f"RESERVED_CONTENT protected={len(reserved)} pilot_parity=passed", flush=True)
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_initialize_conversion, initargs=(contract,)) as pool:
        # Bounded batches cap resident BVH results; consume in planned order so
        # duplicate handling and manifests do not depend on worker completion time.
        for start in range(0, len(selected), workers * 2):
            batch = selected[start:start + workers * 2]
            for record, result in zip(batch, pool.map(_convert_record, batch)):
                if "error" in result:
                    rejected.append({**record, "reason": result["error"]})
                    continue
                digest = result["report"]["source_sha256"]
                if digest in hashes:
                    rejected.append({**record, "reason": "Duplicate source content",
                                     "duplicate_split": hashes[digest]})
                    continue
                shard = f"shards/{digest[:20]}.npz"
                if (output / shard).exists():
                    raise ValueError("Shard filename collision")
                np.savez_compressed(output / shard, **result["arrays"])
                hashes[digest] = "train"
                accepted.append({**record, **result["report"], "shard": shard})
            write_json(output / "accepted.in_progress.json", accepted)
            write_json(output / "rejected.in_progress.json", rejected)
            print(f"EXPANSION {min(start + len(batch), len(selected))}/{len(selected)} "
                  f"accepted_train={sum(r['split'] == 'train' for r in accepted)} "
                  f"rejected={len(rejected)}", flush=True)
    if [r for r in accepted if r["split"] != "train"] != [r for r in pilot if r["split"] != "train"]:
        raise ValueError("Frozen validation/test manifest changed")
    write_json(output / "accepted.json", accepted)
    write_json(output / "rejected.json", rejected)
    report = {
        "status": "completed", "dataset_role": "intermediate 60 Hz data; native builder regenerates 30 Hz train-only statistics",
        "source_pilot": str(pilot_dir), "source_root": str(source), "selection": plan,
        "source_accepted_sha256": sha256(pilot_dir / "accepted.json"),
        "candidate_manifest_sha256": sha256(pilot_dir / "candidate_manifest.jsonl"),
        "reserved_content_hashes_sha256": sha256(output / "reserved_content_hashes.json"),
        "copied_pilot_shard_sha256": copied_hashes,
        "code_sha256": {name: sha256(Path(__file__).parent / name) for name in
                        (Path(__file__).name, "prepare_bones_locomotion.py")},
        "bones_seed_sha256": sha256(Path(__file__).parents[1] / "motionbricks/data/bones_seed.py"),
        "sole_proxy_geometry": "position_only_v0", "converter_parity": parity,
        "split_counts": dict(Counter(row["split"] for row in accepted)),
        "train_behaviors": dict(Counter(row["behavior"] for row in accepted if row["split"] == "train")),
        "train_recording_families": len({row["recording_group"] for row in accepted if row["split"] == "train"}),
        "train_duration_hours": sum(row["duration_seconds"] for row in accepted if row["split"] == "train") / 3600,
        "rejected": len(rejected), "elapsed_seconds": time.monotonic() - started,
        "quality_approved": False,
        "limitations": ["Existing fixed-ground proxy approximation is retained, not repaired.",
                        "Candidate filenames/timelines and existing numerical gates are conservative automatic screening, not semantic ground truth.",
                        "Existing validation covers five behavior groups, not lateral motion."]}
    write_json(output / "expansion_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(expand(args.pilot, args.source, args.output, args.max_train, args.workers), indent=2))
