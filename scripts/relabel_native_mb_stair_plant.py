"""Clone a native stair dataset and filter weak positives with shipped plant semantics.

The native derivative is the forward joint interval at frames 0..T-2; frame T-1
repeats the final backward interval. Rejected/unknown evidence is not airborne
truth. Native training still consumes zero-coded contact features without masks.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_data import build_native_representation, REPRESENTATION_NAME
from motionbricks.motionlib.core.motion_reps.tools import feet, motion_features


VERSION = "native-mb-stair-stationary-plant-v1"
FPS, VELOCITY_THRESHOLD, HEIGHT_THRESHOLD = 30.0, 0.15, 0.10
REPORT_FILE = "native_stationary_plant_relabel.json"
PARENT_FILE = "native_stationary_plant_parent_manifest.json"
NEW_ARRAYS = ("native_stationary_plant", "native_stationary_plant_mask",
              "native_stationary_plant_provenance")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _confined_file(folder, relative):
    path = (folder / relative).resolve()
    if Path(relative).is_absolute() or folder not in path.parents:
        raise ValueError(f"Dataset artifact escapes dataset: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _foot_order(skeleton):
    left = [int(i) for i in skeleton.left_foot_joint_idx]
    right = [int(i) for i in skeleton.right_foot_joint_idx]
    native = [int(i) for i in skeleton.foot_joint_idx]
    helper = left + right
    if (len(left) != 2 or len(right) != 2 or len(set(helper)) != 4
            or len(native) != 4 or set(native) != set(helper)):
        raise ValueError("Native foot joint IDs must be a permutation of helper left + right IDs")
    return native, helper, [helper.index(joint) for joint in native]


def stationary_plant(positions, original_contacts, patch_ids, support, skeleton):
    """Filter [T,4] positives in native joint order using same-foot patch heights."""
    native, _, order = _foot_order(skeleton)
    if (positions.ndim != 3 or positions.shape[-1] != 3 or len(positions) < 2
            or not positions.is_floating_point() or not torch.isfinite(positions).all()):
        raise ValueError("Require finite FK joint positions with at least two frames")
    original = torch.as_tensor(original_contacts, device=positions.device)
    if original.shape != (len(positions), 4) or not torch.isfinite(original).all():
        raise ValueError("Require finite original native contact channels [frames,4]")
    patches = support.get("support", support)["patches"]
    lookup = {}
    for patch in patches:
        if patch["id"] in lookup:
            raise ValueError("Support patch IDs must be unique")
        height = float(patch["height_m"])
        lookup[patch["id"]] = height if patch["status"] == "weak_support" else np.nan

    # Differentiate world FK first: changes in assigned support height must not
    # become joint velocity, and a rolling foot joint must retain its motion.
    velocity = motion_features.compute_vel_xyz(positions, FPS)
    relative = positions.clone()
    height_report = {}
    for side, joints in (("left", skeleton.left_foot_joint_idx),
                         ("right", skeleton.right_foot_joint_idx)):
        ids = np.asarray(patch_ids[side])
        if ids.shape != (len(positions),) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f"Require aligned integer patch_id_{side} per frame")
        heights = np.array([lookup.get(int(i), np.nan) if i >= 0 else np.nan for i in ids])
        heights[~np.isfinite(heights)] = np.nan
        relative[:, list(joints), 1] -= torch.as_tensor(
            heights, device=positions.device, dtype=positions.dtype
        )[:, None]
        known = heights[np.isfinite(heights)]
        height_report[side] = {
            "known_frames": int(len(known)), "unknown_frames": int(len(heights) - len(known)),
            "min_m": float(known.min()) if len(known) else None,
            "max_m": float(known.max()) if len(known) else None,
        }
    left, right = feet.foot_detect_from_pos_and_vel(
        relative[None], velocity[None], skeleton, VELOCITY_THRESHOLD, HEIGHT_THRESHOLD
    )
    detected = torch.cat((left, right), dim=-1)[0, :, order].bool()
    source_positive = original > 0.5
    plant = detected & source_positive
    speed = torch.linalg.vector_norm(velocity[:, native], dim=-1)
    exceed = source_positive & (speed > VELOCITY_THRESHOLD)
    before, after = int(source_positive.sum()), int(plant.sum())
    diagnostics = {
        "positive_joint_frames_before": before, "positive_joint_frames_after": after,
        "positive_joint_frames_removed": before - after,
        "positive_counts_before_by_native_channel": source_positive.sum(0).cpu().tolist(),
        "positive_counts_after_by_native_channel": plant.sum(0).cpu().tolist(),
        "source_labeled_joint_speed_exceed_0_15_mps_count": int(exceed.sum()),
        "source_labeled_joint_speed_exceed_0_15_mps_fraction": float(exceed.sum()) / before if before else None,
        "source_labeled_joint_speed_exceed_counts_by_native_channel": exceed.sum(0).cpu().tolist(),
        "support_heights": height_report,
    }
    return plant.cpu().numpy(), diagnostics


def relabel_shard(arrays, support, representation, provenance):
    """Preserve every existing array except the global native foot-contact slice."""
    if any(name in arrays for name in NEW_ARRAYS):
        raise ValueError("Shard already contains stationary-plant arrays")
    features = arrays["global_features"]
    if (features.ndim != 2 or features.shape[1] != representation.motion_rep_dim
            or not np.issubdtype(features.dtype, np.floating) or not np.isfinite(features).all()):
        raise ValueError("Invalid unnormalized native global_features")
    if float(arrays["fps"]) != FPS:
        raise ValueError("Stationary-plant relabeling requires 30 Hz source shards")
    contacts = np.asarray(representation.indices["foot_contacts"], dtype=np.int64)
    if len(contacts) != 4 or len(set(contacts.tolist())) != 4:
        raise ValueError("Native representation must expose four distinct contact channels")
    with torch.inference_mode():
        positions = representation.inverse(
            torch.from_numpy(features.copy()), is_normalized=False,
            joint_positions_from="global_rot_data", run_fk=True,
        )["posed_joints"]
        plant, diagnostics = stationary_plant(
            positions, features[:, contacts],
            {side: arrays[f"patch_id_{side}"] for side in ("left", "right")},
            support, representation.skeleton,
        )
    result = dict(arrays)
    result["global_features"] = features.copy()
    result["global_features"][:, contacts] = plant.astype(features.dtype)
    result["native_stationary_plant"] = plant
    result["native_stationary_plant_mask"] = plant.copy()
    result["native_stationary_plant_provenance"] = np.asarray(json.dumps({
        **provenance, "version": VERSION,
        "arrays": {name: VERSION for name in NEW_ARRAYS[:2]},
        "mask_semantics": "positive evidence only; false is unknown/rejected evidence, not known airborne",
        "original_evidence_arrays": sorted(name for name in arrays if name != "global_features"),
    }, sort_keys=True, allow_nan=False))
    return result, diagnostics


def _verify_source(dataset, manifest, records):
    artifacts = manifest.get("artifact_sha256", {})
    required = {"accepted.json", "stats/mean.npy", "stats/std.npy"}
    required.update(row["shard"] for row in records)
    stairs = [row for row in records if row.get("behavior", "").startswith("stairs_climbing")]
    if not stairs:
        raise ValueError("No accepted stair records to relabel")
    if len({row["shard"] for row in records}) != len(records):
        raise ValueError("Accepted shard paths must be unique")
    for row in stairs:
        report = row.get("support_report") or row.get("report_path")
        if not report:
            raise ValueError(f"Missing source support report: {row['shard']}")
        required.add(report)
    if not required.issubset(artifacts):
        raise ValueError("Source manifest must bind all accepted shards, support reports, splits and stats")
    for relative, expected in artifacts.items():
        if sha256_file(_confined_file(dataset, relative)) != expected:
            raise ValueError(f"Source artifact hash mismatch: {relative}")
    return stairs


def relabel_dataset(dataset, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output must not exist: {output}")
    if dataset in output.parents or output in dataset.parents:
        raise ValueError("Output must be separate from the source dataset tree")
    manifest_path = dataset / "native_manifest.json"
    parent_bytes = manifest_path.read_bytes()
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()
    manifest = json.loads(parent_bytes)
    contract = manifest["contract"]
    if contract["target_fps"] != FPS or contract["representation"] != REPRESENTATION_NAME:
        raise ValueError("Require the native SOMA30 30 Hz representation")
    records = _json(dataset / "accepted.json")
    stairs = _verify_source(dataset, manifest, records)
    for reserved in (REPORT_FILE, PARENT_FILE):
        if (dataset / reserved).exists():
            raise ValueError(f"Source already contains relabel provenance: {reserved}")

    representation = build_native_representation()
    native, helper, order = _foot_order(representation.skeleton)
    protocol = {
        "version": VERSION, "fps": FPS,
        "velocity_threshold_mps_strict_less_than": VELOCITY_THRESHOLD,
        "height_threshold_m_strict_less_than": HEIGHT_THRESHOLD,
        "velocity_function": "tools.motion_features.compute_vel_xyz",
        "detector_function": "tools.feet.foot_detect_from_pos_and_vel",
        "velocity_source_sha256": sha256_file(motion_features.__file__),
        "detector_source_sha256": sha256_file(feet.__file__),
        "derivative": "forward joint interval at frames 0..T-2; final frame repeats the final backward interval",
        "height": "FK foot joint Y minus its own foot's source weak_support patch height; unknown is NaN",
        "native_foot_joint_ids": native, "helper_left_right_joint_ids": helper,
        "helper_columns_in_native_order": order,
        "native_foot_joint_names": [representation.skeleton.bone_order_names[i] for i in native],
        "positive_rule": "official detector positive AND parent global_features foot_contacts > 0.5; never promote",
        "array_versions": {name: VERSION for name in NEW_ARRAYS[:2]},
        "original_evidence": "All preexisting sidecars, weak labels, masks, support geometry and dual_features are retained verbatim as original evidence. Only global_features foot_contacts are relabeled.",
        "interpretation": "Heuristic stationary-plant evidence, not force/contact or airborne truth. New mask is true only for positives; native training does not consume it and still penalizes zero-coded feature targets.",
    }
    shutil.copytree(dataset, output)
    if (output / "native_manifest.json").read_bytes() != parent_bytes:
        raise ValueError("Source native manifest changed during clone")
    _verify_source(output, manifest, records)
    (output / PARENT_FILE).write_bytes(parent_bytes)
    report = {"version": VERSION, "dataset": str(dataset), "output": str(output),
              "parent_native_manifest_sha256": parent_hash,
              "converter_sha256": sha256_file(__file__), "protocol": protocol, "shards": []}
    artifacts = dict(manifest["artifact_sha256"])
    for row in stairs:
        relative = row["shard"]
        support_path = row.get("support_report") or row["report_path"]
        path = _confined_file(output, relative)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        provenance = {
            "parent_native_manifest_sha256": parent_hash,
            "source_shard_sha256": artifacts[relative],
            "source_support_report_sha256": artifacts[support_path],
            "protocol": protocol,
        }
        updated, diagnostics = relabel_shard(arrays, _json(output / support_path),
                                             representation, provenance)
        np.savez_compressed(path, **updated)
        artifacts[relative] = sha256_file(path)
        report["shards"].append({
            "shard": relative, "split": row["split"], "behavior": row["behavior"],
            "source_shard_sha256": provenance["source_shard_sha256"],
            "relabel_shard_sha256": artifacts[relative], **diagnostics,
        })
    report["status"] = "completed_heuristic_relabel"
    _write_json(output / REPORT_FILE, report)
    artifacts[REPORT_FILE] = sha256_file(output / REPORT_FILE)
    artifacts[PARENT_FILE] = parent_hash
    child = copy.deepcopy(manifest)
    child.update(artifact_sha256=artifacts, parent_native_manifest_sha256=parent_hash,
                 parent_native_manifest=PARENT_FILE, parent_dataset=str(dataset),
                 stationary_plant_relabel={"version": VERSION, "report": REPORT_FILE,
                                          "report_sha256": artifacts[REPORT_FILE],
                                          "protocol": protocol})
    _write_json(output / "native_manifest.json", child)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh destination; must not exist")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    report = relabel_dataset(args.dataset, args.output)
    print(json.dumps({"status": report["status"], "shards": len(report["shards"]),
                      "report": str(args.output / REPORT_FILE)}, indent=2))


if __name__ == "__main__":
    main()
