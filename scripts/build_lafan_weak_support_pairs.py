"""Build flat-prior weak-support evidence from accepted 60 Hz LaFAN SOMA shards."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DATASET = Path("${PRIVATE_STORAGE}/motionbricks-lafan-resolved/soma60hz-v3")
SCHEMA_VERSION = "motionbricks.lafan_weak_support_pairs.v1"
VALID_SPLITS = {"train", "validation", "test"}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _init_runtime():
    global np, torch, complete_motion_dict, build_representation
    global official_sole_surfaces, sole_region_indices, load_official_skinner
    global SupportConfig, reconstruct_support
    import numpy as np
    import torch
    project = Path(__file__).resolve().parents[1]
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from kimodo.exports.motion_io import complete_motion_dict
    from motionbricks.data.bones_seed import build_representation, official_sole_surfaces
    from motionbricks.data.sole_region_contact import sole_region_indices
    from motionbricks.data.soma_skin_reference import load_official_skinner
    from motionbricks.data.stair_support import SupportConfig, reconstruct_support


def _read_manifest(dataset):
    accepted_path = dataset / "accepted.json"
    if not accepted_path.is_file():
        raise FileNotFoundError(f"Missing accepted shard manifest: {accepted_path}")
    records = json.loads(accepted_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("accepted.json must contain a nonempty record list")
    counts = Counter(row.get("split") if row.get("split") in VALID_SPLITS else "unassigned"
                     for row in records)
    group_splits = defaultdict(set)
    for row in records:
        group = row.get("recording_group")
        if group and row.get("split") in VALID_SPLITS:
            group_splits[group].add(row["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("A recording_group appears in multiple inherited splits")
    return records, accepted_path, counts


def _validate_source_arrays(record, shard_path):
    with np.load(shard_path, allow_pickle=False) as source:
        required = {"local_rot_mats", "root_positions", "source_positions", "target_fps"}
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"Shard missing required accepted arrays: {sorted(missing)}")
        rotations = source["local_rot_mats"].copy()
        root = source["root_positions"].copy()
        source_positions = source["source_positions"].copy()
        target_fps = float(np.asarray(source["target_fps"]).reshape(()))
        optional = {key: source[key].copy() for key in ("source_names", "source_parents")
                    if key in source.files}
    if rotations.ndim != 4 or rotations.shape[1:] != (30, 3, 3):
        raise ValueError(f"Expected SOMA30 local_rot_mats [T,30,3,3], got {rotations.shape}")
    if root.shape != (len(rotations), 3) or len(rotations) < 3:
        raise ValueError("root_positions must be [T,3] and contain at least three frames")
    if source_positions.shape[0] != len(rotations) or source_positions.shape[-1] != 3:
        raise ValueError("source_positions must align with shard frames and xyz")
    if abs(target_fps - 60.0) > 1e-6:
        raise ValueError(f"Expected accepted 60 Hz target arrays, got {target_fps} Hz")
    declared = record.get("frames_60hz")
    if declared is not None and int(declared) != len(rotations):
        raise ValueError("accepted.json frame count differs from shard arrays")
    if not (np.isfinite(rotations).all() and np.isfinite(root).all()
            and np.isfinite(source_positions).all()):
        raise ValueError("Source motion arrays contain nonfinite values")
    return rotations, root, source_positions, target_fps, optional


def _skinning_regions(local, posed, skeleton, surfaces, skinner, chunk_frames):
    surface_indices = {side: sole_region_indices(surfaces, side) for side in ("Left", "Right")}
    selected = np.unique(np.concatenate([surface_indices[side] for side in ("Left", "Right")]))
    if len(selected) != 196 or any(len(surface_indices[side]) != 98 for side in ("Left", "Right")):
        raise ValueError("Official SOMA sole region changed: expected 98 vertices per side, 196 unique total")
    selected_lookup = {int(index): order for order, index in enumerate(selected)}
    side_local = {side.lower(): np.asarray([selected_lookup[int(index)] for index in surface_indices[side]])
                  for side in ("Left", "Right")}

    full_tensors = {attribute: getattr(skinner, attribute)
                    for attribute in ("bind_vertices", "lbs_indices", "lbs_weights")}
    try:
        with torch.inference_mode():
            sample_local = torch.from_numpy(local[:2])
            sample_posed = posed[:2]
            full = skinner.skin(sample_local, sample_posed, rot_is_global=False)
            expected = full.index_select(1, torch.as_tensor(selected, dtype=torch.long))
            for attribute, tensor in full_tensors.items():
                setattr(skinner, attribute, tensor.index_select(
                    0, torch.as_tensor(selected, dtype=torch.long)))
            actual = skinner.skin(sample_local, sample_posed, rot_is_global=False)
            parity_error = float((expected - actual).abs().max())
            if parity_error > 1e-6:
                raise ValueError(f"Selected 196-vertex LBS differs from full-mesh LBS: {parity_error:.9g} m")

            selected_vertices = []
            for start in range(0, len(local), chunk_frames):
                end = min(start + chunk_frames, len(local))
                vertices = skinner.skin(torch.from_numpy(local[start:end]), posed[start:end], rot_is_global=False)
                selected_vertices.append(vertices.cpu().numpy())
    finally:
        for attribute, tensor in full_tensors.items():
            setattr(skinner, attribute, tensor)
    selected_vertices = np.concatenate(selected_vertices, axis=0)
    by_side = {side: selected_vertices[:, indices].astype(np.float32, copy=False)
               for side, indices in side_local.items()}
    return by_side, parity_error, surface_indices


def process_record(record, dataset, output, representation, surfaces, skinner, chunk_frames,
                   accepted_sha256, builder_sha256, core_sha256, mesh_sha256):
    item_name = str(record.get("filename") or Path(record["shard"]).stem)
    if (not item_name or item_name in {".", ".."} or "/" in item_name or "\\" in item_name
            or Path(item_name).name != item_name):
        raise ValueError(f"Unsafe item filename: {item_name!r}")
    relative = Path(record["shard"])
    shard_path = (dataset / relative).resolve()
    if dataset.resolve() not in shard_path.parents:
        raise ValueError(f"Shard path escapes dataset root: {relative}")
    actual_shard_sha = _sha256_file(shard_path)
    declared_shard_sha = record.get("shard_sha256")
    if declared_shard_sha and actual_shard_sha != declared_shard_sha:
        raise ValueError("Shard bytes do not match accepted.json shard_sha256")
    rotations, root, source_positions, fps, optional = _validate_source_arrays(record, shard_path)
    root_sha = _sha256_array(root)
    rotations_sha = _sha256_array(rotations)
    source_positions_sha = _sha256_array(source_positions)
    with torch.inference_mode():
        local_t = torch.from_numpy(rotations)
        root_t = torch.from_numpy(root)
        motion = complete_motion_dict(local_t, root_t, representation.skeleton, fps)
        posed = motion["posed_joints"]
    soles, skin_parity_error, side_indices = _skinning_regions(
        rotations, posed, representation.skeleton, surfaces, skinner, chunk_frames)
    weak_arrays, support_report = reconstruct_support(
        sole_vertices=soles, root_positions=root, fps=fps,
        config=SupportConfig(require_stairs=False), body_joints=posed.cpu().numpy())
    if len(rotations) != int(record.get("frames_60hz", len(rotations))):
        raise ValueError("Input frame count changed during support reconstruction")
    if _sha256_array(root) != root_sha or _sha256_array(rotations) != rotations_sha:
        raise RuntimeError("Source root/rotation arrays changed during processing")

    item_relative = Path("items") / item_name
    item_dir = output / item_relative
    item_dir.mkdir(parents=True, exist_ok=False)
    source_frame_count = len(rotations)
    arrays = {
        "local_rot_mats": rotations,
        "root_positions": root,
        "posed_joints": posed.cpu().numpy().astype(np.float32),
        "sole_vertices_left": soles["left"],
        "sole_vertices_right": soles["right"],
        "sole_vertex_indices_left": side_indices["Left"].astype(np.int64, copy=False),
        "sole_vertex_indices_right": side_indices["Right"].astype(np.int64, copy=False),
        "timestamps_seconds": np.arange(source_frame_count, dtype=np.float64) / fps,
        "source_frame_indices": np.arange(source_frame_count, dtype=np.int64),
        "fps": np.asarray(fps, dtype=np.float64),
        **optional,
        **weak_arrays,
    }
    arrays["source_positions"] = source_positions
    np.savez_compressed(item_dir / "pair.npz", **arrays)
    support_summary = support_report.get("summary", {})
    patches = support_report.get("patches", [])
    collision_rejections = sum(bool(patch.get("free_space_violation_frames", 0)) for patch in patches)
    report = {
        "schema_version": SCHEMA_VERSION,
        "filename": item_name,
        "source_dataset": str(dataset.resolve()),
        "source_shard": str(relative.as_posix()),
        "source_shard_sha256": actual_shard_sha,
        "accepted_manifest_sha256": accepted_sha256,
        "source_rotations_sha256": rotations_sha,
        "source_root_positions_sha256": root_sha,
        "source_positions_sha256": source_positions_sha,
        "native_fps": fps,
        "frames": source_frame_count,
        "duration_seconds": (source_frame_count - 1) / fps,
        "source_arrays_unchanged": True,
        "resampling": None,
        "additional_floor_translation_m": 0.0,
        "prior_retarget_fixed_plane_y_m": record.get("fixed_plane_source_y_m"),
        "source_calibration_crop": record.get("source_calibration_crop"),
        "official_selected_lbs_parity_max_error_m": skin_parity_error,
        "sole_mesh_sha256": mesh_sha256,
        "geometry_source": "official SOMA77 skin_standard.npz LBS; fixed 98-vertex sole region per side",
        "sole_vertex_counts": {side: len(indices) for side, indices in side_indices.items()},
        "split": record.get("split") if record.get("split") in VALID_SPLITS else "unassigned",
        "recording_group": record.get("recording_group"),
        "subject_id": record.get("subject_id"),
        "behavior": record.get("behavior"),
        "support": support_report,
        "recording_plane_y_m": support_summary.get("recording_plane_y_m"),
        "pair_status": support_summary.get("pair_status", "unknown"),
        "collision_gate": {
            "passed": collision_rejections == 0,
            "patches_with_free_space_violations": collision_rejections,
            "scope": "sole regions and SOMA30 joint landmarks only; not full-body mesh collision",
        },
        "geometry_scope": "planar weak support only; no physical vertical support/stair evidence asserted",
        "training_eligible": False,
        "skeleton_parents": representation.skeleton.joint_parents.tolist(),
        "skeleton_joint_names": representation.skeleton.bone_order_names,
        "prior_retarget_assessment": record.get("retarget_assessment"),
        "builder_sha256": builder_sha256,
        "support_core_sha256": core_sha256,
    }
    (item_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "filename": item_name,
        "item_relative_path": item_relative.as_posix(),
        "source_shard": relative.as_posix(),
        "source_shard_sha256": actual_shard_sha,
        "split": report["split"],
        "recording_group": record.get("recording_group"),
        "subject_id": record.get("subject_id"),
        "behavior": record.get("behavior"),
        "execution_status": "processed",
        "pair_status": report["pair_status"],
        "collision_gate_passed": report["collision_gate"]["passed"],
        "frames": source_frame_count,
        "patch_count": len(patches),
        "event_count": len(support_report.get("events", [])),
        "fixed_plane_source_y_m": record.get("fixed_plane_source_y_m"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=128)
    args = parser.parse_args()
    if args.chunk_frames < 1:
        parser.error("--chunk-frames must be positive")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    records, accepted_path, split_counts = _read_manifest(dataset)
    if len(records) != 9:
        raise ValueError(f"Expected 9 accepted shards, found {len(records)}")
    if "unassigned" not in split_counts and split_counts != Counter({"train": 5, "validation": 2, "test": 2}):
        raise ValueError(f"Unexpected accepted dataset inventory: {len(records)} rows, splits={dict(split_counts)}")
    _init_runtime()
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    (output / "items").mkdir()
    accepted_sha = _sha256_file(accepted_path)
    builder_sha = _sha256_file(Path(__file__).resolve())
    import motionbricks.data.stair_support as support_module
    core_sha = _sha256_file(Path(support_module.__file__).resolve())
    representation = build_representation()
    surfaces, _, _, mesh_sha = official_sole_surfaces(representation.skeleton)
    skinner = load_official_skinner(representation.skeleton)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "processing",
        "source_dataset": str(dataset),
        "source_manifest": "accepted.json",
        "accepted_manifest_sha256": accepted_sha,
        "source_retarget_review": "retarget_review.json",
        "native_fps": 60.0,
        "source_shard_count": len(records),
        "inherited_split_counts": dict(split_counts),
        "split_policy": "copied only from accepted.json; missing/invalid metadata remains unassigned",
        "support_geometry_scope": "flat-prior planar weak support only; physical vertical support not established",
        "additional_floor_translation_m": 0.0,
        "sole_mesh_sha256": mesh_sha,
        "builder_sha256": builder_sha,
        "support_core_sha256": core_sha,
        "training_eligible": False,
        "records": [],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# LaFAN Flat-Prior Weak Support Sidecar\n\n"
        "This is an offline discovery/review artifact, not a training-ready label set. It reuses accepted SOMA60 "
        "shards and their source coordinates. It does not resample, alter root/rotation arrays, or translate the "
        "motion. The support core estimates a flat weak-support plane in those existing coordinates; the prior "
        "retarget plane is provenance only. Support is planar weak evidence only; motion does not "
        "establish physical load or vertical stair support. Source splits and recording groups are inherited.\n",
        encoding="utf-8")
    rows = []
    for index, record in enumerate(records, 1):
        try:
            result = process_record(record, dataset, output, representation, surfaces, skinner,
                                    args.chunk_frames, accepted_sha, builder_sha, core_sha, mesh_sha)
        except Exception as error:
            result = {
                "filename": record.get("filename", Path(record.get("shard", "unknown")).stem),
                "item_relative_path": None,
                "source_shard": record.get("shard"),
                "source_shard_sha256": record.get("shard_sha256"),
                "split": record.get("split") if record.get("split") in VALID_SPLITS else "unassigned",
                "recording_group": record.get("recording_group"),
                "subject_id": record.get("subject_id"),
                "behavior": record.get("behavior"),
                "execution_status": "failed",
                "failure_type": type(error).__name__, "failure": str(error),
                "pair_status": None, "collision_gate_passed": None,
                "frames": None, "patch_count": 0, "event_count": 0,
                "fixed_plane_source_y_m": record.get("fixed_plane_source_y_m"),
            }
        rows.append(result)
        manifest["records"] = rows
        manifest["status"] = "processing"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"item": index, "total": len(records), "filename": result["filename"],
                          "execution_status": result["execution_status"],
                          "split": result["split"], "pair_status": result["pair_status"]}), flush=True)
    manifest["status"] = "completed_with_failures" if any(
        row["execution_status"] == "failed" for row in rows) else "completed"
    manifest["summary"] = {
        "processed": sum(row["execution_status"] == "processed" for row in rows),
        "failed": sum(row["execution_status"] == "failed" for row in rows),
        "inherited_split_counts": {split: sum(row["split"] == split for row in rows)
                                   for split in ("train", "validation", "test")},
        "pair_status_counts": dict(Counter(row["pair_status"] for row in rows if row.get("pair_status"))),
        "collision_gate_failures": sum(row.get("collision_gate_passed") is False for row in rows),
        "training_eligible": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], sort_keys=True), flush=True)
    if manifest["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
