"""Automatic artifact contract checks and a compact weak-support dataset receipt."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def array_sha256(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def validate_pair(data, report):
    """Check alignment/unknowns/identity, not whether inferred terrain is real."""
    frames = int(report["frames"])
    fps = float(report["native_fps"])
    if frames < 3 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid frame/FPS contract")
    for key, hash_key in (("root_positions", "source_root_positions_sha256"),
                          ("local_rot_mats", "source_rotations_sha256"),
                          ("source_positions", "source_positions_sha256")):
        if len(data[key]) != frames or array_sha256(data[key]) != report[hash_key]:
            raise ValueError(f"Source array identity changed: {key}")
    np.testing.assert_array_equal(data["source_frame_indices"], np.arange(frames))
    np.testing.assert_allclose(data["timestamps_seconds"], np.arange(frames) / fps, atol=1e-12)
    patches = {p["id"]: p for p in report["support"]["patches"]}
    result = {}
    for side in ("left", "right"):
        vertices = data[f"sole_vertices_{side}"]
        vertex_ids = data[f"sole_vertex_indices_{side}"]
        ids = data[f"patch_id_{side}"]
        confidence = data[f"support_confidence_{side}"]
        gaps = data[f"support_gap_m_{side}"]
        measured = data[f"vertex_measured_{side}"]
        slip = data[f"slip_mps_{side}"]
        if vertices.shape != (frames, 98, 3) or measured.shape != (frames - 1, 98) or slip.shape != measured.shape:
            raise ValueError("Material vertex frame/interval axes do not align")
        if vertex_ids.shape != (98,) or len(np.unique(vertex_ids)) != 98 or (vertex_ids < 0).any():
            raise ValueError("Material vertices require 98 unique official mesh IDs per foot")
        if any(value.shape != (frames,) for value in (ids, confidence, gaps)):
            raise ValueError("Frame label length differs from source motion")
        known = ids >= 0
        if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > .8)).any():
            raise ValueError("Confidence outside declared heuristic range")
        if (confidence[~known] != 0).any() or not np.isnan(gaps[~known]).all():
            raise ValueError("Unknown cannot be a zero-height or confident support label")
        if not np.isfinite(gaps[known]).all() or (confidence[known] <= 0).any():
            raise ValueError("Known weak support requires finite gap and positive confidence")
        for pid in np.unique(ids[known]):
            if int(pid) not in patches or patches[int(pid)]["status"] != "weak_support":
                raise ValueError("Frame references missing/rejected patch")
        same_patch = (ids[:-1] >= 0) & (ids[:-1] == ids[1:])
        if (measured & ~same_patch[:, None]).any():
            raise ValueError("Measured material-point interval crosses a support transition")
        if not np.isnan(slip[~measured]).all() or not np.isfinite(slip[measured]).all():
            raise ValueError("Unmeasured slip must remain unknown, not zero")
        expected = np.linalg.norm(np.diff(vertices.astype(np.float64)[..., [0, 2]], axis=0), axis=-1) * fps
        np.testing.assert_allclose(slip[measured], expected[measured], atol=1e-10, rtol=1e-10)
        result[side] = {
            "weak_support_frames": int(known.sum()), "weak_support_frame_fraction": float(known.mean()),
            "measured_intervals": int(measured.any(axis=1).sum()), "measured_vertex_intervals": int(measured.sum()),
            "confidence_median": float(np.median(confidence[known])) if known.any() else None,
            "gap_abs_p95_m": float(np.quantile(abs(gaps[known]), .95)) if known.any() else None,
            "material_point_speed_p95_mps": float(np.quantile(slip[measured], .95)) if measured.any() else None,
        }
    return result


def review_dataset(dataset, verify_source=False):
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or any(r["execution_status"] != "processed" for r in manifest["records"]):
        raise ValueError("Input dataset processing is not complete")
    rows = []
    for record in manifest["records"]:
        item = (dataset / record["item_relative_path"]).resolve()
        if dataset.resolve() not in item.parents:
            raise ValueError("Item escapes dataset directory")
        report = json.loads((item / "report.json").read_text(encoding="utf-8"))
        with np.load(item / "pair.npz", allow_pickle=False) as data:
            metrics = validate_pair(data, report)
        if verify_source:
            source = Path(report["source_dataset"]) / report["source_shard"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != report["source_shard_sha256"]:
                raise ValueError(f"Original source shard changed: {source}")
        patches = report["support"]["patches"]
        rows.append({"filename": record["filename"], "split": record["split"],
                     "behavior": record["behavior"], "recording_group": record["recording_group"],
                     "subject_id": record["subject_id"], "frames": report["frames"],
                     "duration_seconds": report["duration_seconds"], "pair_status": report["pair_status"],
                     "source_arrays_unchanged": True, "selected_lbs_parity_max_error_m": report["official_selected_lbs_parity_max_error_m"],
                     "support_plane_y_m": report["support"]["summary"]["recording_plane_y_m"],
                     "weak_patches": sum(p["status"] == "weak_support" for p in patches),
                     "rejected_patches": sum(p["status"] == "rejected" for p in patches),
                     "rejection_reasons": dict(Counter(reason for p in patches for reason in p["reasons"])),
                     "sides": metrics})
    return {"schema_version": "weak-support-receipt-v1", "artifact_consistency_status": "passed",
            "source_bytes_reverified": verify_source, "source_manifest_sha256": manifest["accepted_manifest_sha256"],
            "recordings": len(rows), "minutes": sum(r["duration_seconds"] for r in rows) / 60,
            "split_counts": dict(Counter(r["split"] for r in rows)),
            "weak_patches": sum(r["weak_patches"] for r in rows), "rejected_patches": sum(r["rejected_patches"] for r in rows),
            "records": rows, "geometry_quality_status": "weak_evidence_not_ground_truth",
            "model_improvement_status": "not_assessed_no_training",
            "unvisited_space": "unknown; no completed terrain or heightmap",
            "evaluation_scope": "all accepted frames/intervals, not selected preview windows"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-source", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing receipt")
    report = review_dataset(args.dataset.resolve(), args.verify_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))


if __name__ == "__main__":
    main()
