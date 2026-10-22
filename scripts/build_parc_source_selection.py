"""Build a hash-bound manifest for a small official PARC source selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from motionbricks.data.parc_paired_source import (
    _CONTACT_BODY_NAMES,
    _decode_section,
    _restricted_load,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = ROOT / "out" / "parc_release_audit_20261008"
DEFAULT_OUTPUT = ROOT / "out" / "parc_source_selection_20261008" / "manifest.json"
MAX_SOURCE_BYTES = 128 * 1024 * 1024
CONTACT_THRESHOLD = 0.5
SOURCE_SPECS = (
    {
        "behavior": "stairs_ascent",
        "member": "dec_release/initial_aug/stairs/beyond_stairs_001_before_opt_dm.pkl",
        "filename": "7789fd14f77f_beyond_stairs_001_before_opt_dm.pkl",
        "source_group": "parc_release_dec_release_initial_aug_stairs_selection_20261008",
        "description": "Official released stair-ascent motion candidate.",
    },
    {
        "behavior": "stairs_descent",
        "member": "dec_release/initial_aug/stairs/beyond_stairs_002_before_opt_dm.pkl",
        "filename": "a92806a66c95_beyond_stairs_002_before_opt_dm.pkl",
        "source_group": "parc_release_dec_release_initial_aug_stairs_selection_20261008",
        "description": "Official released stair-descent motion candidate.",
    },
    {
        "behavior": "stair_hand_contact",
        "member": "dec_release/initial_aug/misc_mdm/mdm_castle_stairs_dm.pkl",
        "filename": "f900cf9e6fd0_mdm_castle_stairs_dm.pkl",
        "source_group": "parc_release_dec_release_initial_aug_stairs_selection_20261008",
        "description": "Official released mixed stair motion with source hand-contact labels available for inspection.",
    },
)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid inspection JSON at {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Inspection JSON root must be an object: {path}")
    return value


def _inspection_sample(inspection: dict, member: str) -> dict:
    samples = inspection.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Inspection is missing its samples list")
    matches = [row for row in samples if isinstance(row, dict) and row.get("member") == member]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one inspected archive member {member!r}")
    return matches[0]


def _load_and_summarize(path: Path) -> dict:
    size = path.stat().st_size
    if not 0 < size <= MAX_SOURCE_BYTES:
        raise ValueError(f"Source size is empty or exceeds the bounded loader limit: {path}")
    container = _restricted_load(path.read_bytes())
    if not isinstance(container, dict):
        raise ValueError(f"PARC pickle root must be a dictionary: {path}")
    motion = _decode_section(container, "motion_data")
    terrain = _decode_section(container, "terrain_data")

    arrays = {}
    for key, tail in (("root_pos", (3,)), ("root_rot", (4,)),
                      ("joint_rot", (14, 4)), ("body_contacts", (len(_CONTACT_BODY_NAMES),))):
        raw = motion.get(key)
        value = np.asarray(raw)
        if value.dtype.kind not in "fiu" or value.ndim != len(tail) + 1 or value.shape[1:] != tail:
            raise ValueError(f"{key} must be a numeric [frames, {tail}] array in {path}")
        if not 0 < value.shape[0] <= 1_000_000 or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain bounded finite frame data in {path}")
        arrays[key] = value.astype(np.float64, copy=False)

    frame_count = len(arrays["root_pos"])
    if any(len(value) != frame_count for value in arrays.values()):
        raise ValueError(f"PARC motion arrays have inconsistent frame counts in {path}")
    fps = float(motion.get("fps", 0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"PARC motion fps must be finite and positive in {path}")

    heightfield = np.asarray(terrain.get("hf"))
    min_point = np.asarray(terrain.get("min_point"))
    dx = float(terrain.get("dx", 0))
    if (heightfield.ndim != 2 or min(heightfield.shape) <= 0
            or heightfield.dtype.kind not in "fiu" or not np.isfinite(heightfield).all()
            or min_point.shape != (2,) or min_point.dtype.kind not in "fiu"
            or not np.isfinite(min_point).all() or not np.isfinite(dx) or dx <= 0):
        raise ValueError(f"PARC terrain section is malformed in {path}")

    contacts = arrays["body_contacts"] > CONTACT_THRESHOLD
    body_counts = {
        name: int(contacts[:, index].sum())
        for index, name in enumerate(_CONTACT_BODY_NAMES)
    }
    root_z = arrays["root_pos"][:, 2]
    return {
        "file_size_bytes": size,
        "frame_count": frame_count,
        "fps": fps,
        "duration_s": (frame_count - 1) / fps,
        "root_z_displacement_m_source_coordinates": float(root_z[-1] - root_z[0]),
        "contact_labels": {
            "source_field": "motion_data.body_contacts",
            "body_names_order": list(_CONTACT_BODY_NAMES),
            "threshold_for_count_only": CONTACT_THRESHOLD,
            "positive_frame_count_by_body": body_counts,
            "bodies_with_any_positive_frame": sum(value > 0 for value in body_counts.values()),
            "provenance": "Source labels carried as released; not verified physical contact truth.",
        },
    }


def build_manifest(audit_dir: Path = DEFAULT_AUDIT_DIR) -> dict:
    audit_dir = audit_dir.resolve()
    stairs = _read_json(audit_dir / "stairs" / "inspection.json")
    samples = _read_json(audit_dir / "samples" / "inspection.json")
    if stairs.get("revision") != samples.get("revision") or not stairs.get("revision"):
        raise ValueError("Stairs and samples inspections must identify the same release revision")
    archive = stairs.get("archive")
    if not isinstance(archive, dict) or archive != samples.get("archive"):
        raise ValueError("Stairs and samples inspections must have matching archive metadata")

    selected = []
    for spec in SOURCE_SPECS:
        inspected = _inspection_sample(stairs, spec["member"])
        if inspected.get("inspection_error"):
            raise ValueError(f"Source inspection failed for {spec['member']}")
        local_path = (audit_dir / "stairs" / spec["filename"]).resolve()
        recorded_path = Path(str(inspected.get("local_file", ""))).resolve()
        if recorded_path != local_path:
            raise ValueError(f"Inspection local path does not match expected source {spec['filename']}")
        expected_size = inspected.get("size")
        if not isinstance(expected_size, int) or expected_size <= 0 or expected_size > MAX_SOURCE_BYTES:
            raise ValueError(f"Inspection has an invalid bounded source size for {spec['member']}")
        if not local_path.is_file() or local_path.stat().st_size != expected_size:
            raise ValueError(f"Local byte count does not match inspection for {spec['member']}")
        payload = local_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != inspected.get("sha256"):
            raise ValueError(f"Local SHA-256 does not match inspection for {spec['member']}")
        if len(payload) != expected_size:
            raise ValueError(f"Local byte count does not match inspection for {spec['member']}")
        metrics = _load_and_summarize(local_path)
        inspected_motion = inspected.get("motion_data")
        inspected_root = inspected_motion.get("root_pos") if isinstance(inspected_motion, dict) else None
        inspected_shape = inspected_root.get("shape") if isinstance(inspected_root, dict) else None
        if (not isinstance(inspected_shape, list) or not inspected_shape
                or metrics["frame_count"] != inspected_shape[0]):
            raise ValueError(f"Decoded frame count does not match inspection for {spec['member']}")

        selected.append({
            "behavior": spec["behavior"],
            "description": spec["description"],
            "source_group": spec["source_group"],
            "split": "unassigned_not_training_data",
            "source": {
                "local_file": str(local_path),
                "sha256": digest,
                "size_bytes": len(payload),
                "archive_member": spec["member"],
                "archive_member_sha256": inspected["sha256"],
                "release_revision": stairs["revision"],
                "archive_path": archive.get("path"),
                "archive_lfs_sha256": archive.get("lfs", {}).get("oid"),
                "archive_member_size_bytes": inspected["size"],
                "inspection_record": str((audit_dir / "stairs" / "inspection.json").resolve()),
            },
            "provenance_stage_inferred": {
                "archive_directory": "initial_aug",
                "interpretation": "Initial augmentation collection by archive path only; exact generation/correction stage is not established.",
                "filename_signal": Path(spec["member"]).stem,
                "physics_corrected": False,
                "basis": "Neither the directory nor filename establishes a successful physics-recording or correction stage.",
            },
            "metrics": metrics,
            "training_approved": False,
            "retargeted": False,
        })

    return {
        "schema_version": 1,
        "artifact": "bounded_official_parc_source_selection",
        "status": "source_selection_only_no_generation_or_training",
        "release": {
            "revision": stairs["revision"],
            "archive_path": archive.get("path"),
            "archive_lfs_sha256": archive.get("lfs", {}).get("oid"),
            "archive_size_bytes": archive.get("size"),
        },
        "selection_policy": {
            "scope": "Only the three already-inspected local official release members listed below; no additional downloads.",
            "source_group_policy": "All selected items share one indivisible source_group and must never be split across train/validation/test.",
            "split_policy": "No split assigned; items are not training or validation data.",
            "no_train_validation_leakage": True,
            "training_approved": False,
            "retargeted": False,
            "physics_correction_claim": False,
        },
        "selected_items": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest(args.audit_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "selected": len(manifest["selected_items"])}, indent=2))


if __name__ == "__main__":
    main()
