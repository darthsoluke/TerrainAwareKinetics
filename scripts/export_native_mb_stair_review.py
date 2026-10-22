"""Export bounded source/generated native-MB stair replay fixtures for UE review."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motionbricks.data.native_mb_data import FPS, build_native_representation
from motionbricks.data.soma_replay import skeleton_manifest
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30


SCHEMA = "motionbricks.soma.replay.v2"
PREPROCESS_VERSION = "native-mb-stair-review-v1"
MODEL_FEATURES = "features_official_landings_c1_t1_i1"
CLIP_IDS = ("up_source", "up_model", "down_source", "down_model")
MAX_FIXTURE_BYTES = 32 * 1024 * 1024


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _ue_point(point):
    x, y, z = point
    return [float(z * 100), float(-x * 100), float(y * 100)]


def _support_patches(records):
    patches = []
    for record in records:
        polygon = record.get("polygon_xz_m")
        if not polygon or len(polygon) < 3:
            continue
        height = float(record["height_m"])
        patches.append({
            "id": int(record["id"]),
            "polygon_ue_cm": [[float(z * 100), float(-x * 100), height * 100]
                              for x, z in polygon],
            "confidence": float(record["confidence"]),
        })
    return patches


def _decode_features(features, representation, skeleton):
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32).copy())[None]
    if tensor.ndim != 3 or tensor.shape[1] < 2 or tensor.shape[2] != len(representation.indices["all"]):
        raise ValueError("Expected native global feature frames with the official 366-channel layout")
    if not torch.isfinite(tensor).all():
        raise ValueError("Native feature array contains nonfinite values")
    decoded = representation.inverse(tensor, False, return_all=True, return_quat=False)
    local = decoded["local_joint_rots"][0]
    root = decoded["posed_joints"][0, :, skeleton.root_idx]
    global_rot, official_positions, _ = skeleton.fk(local, root)
    feature_positions = decoded["posed_joints"][0]
    max_fk_error_m = float((official_positions - feature_positions).norm(dim=-1).max().item())
    if not np.isfinite(max_fk_error_m) or max_fk_error_m > 0.0002:
        raise ValueError(f"Official SOMA FK does not match decoded native features: {max_fk_error_m:.8f} m")
    return (root.numpy(), local.numpy(), global_rot.numpy(), official_positions.numpy(),
            decoded["foot_contacts"][0].numpy(), max_fk_error_m)


def _fixture(*, clip, record, window, feature_name, features, shard_sha, checkpoint, representation, skeleton):
    root, local, global_rot, positions, contacts, fk_error = _decode_features(features, representation, skeleton)
    local_q = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat().reshape(-1, 30, 4)
    global_q = Rotation.from_matrix(global_rot.reshape(-1, 3, 3)).as_quat().reshape(-1, 30, 4)
    contacts = np.asarray(contacts, dtype=np.float32)
    if contacts.shape != (len(root), 4) or not np.isin(contacts, [0, 1]).all():
        raise ValueError("Expected four native binary foot-contact channels")
    frames = []
    for i in range(len(root)):
        frames.append({
            "root_m": root[i].tolist(),
            "local_xyzw": local_q[i].tolist(),
            "global_xyzw": global_q[i].tolist(),
            "positions_m": positions[i].tolist(),
            "contacts": contacts[i].tolist(),
            "contact_confidence": contacts[i].tolist(),
        })
    runtime_state = "source_replay" if clip.endswith("_source") else "generated_replay"
    source_hash = record["source"]["source_sha256"]
    fixture = {
        "schema": SCHEMA,
        "runtime_state": runtime_state,
        "model_inference": False,
        "offline_generated": runtime_state == "generated_replay",
        "motion_postprocessing": "none; native frame order and source/global origin preserved, with no time-warp or root correction",
        "source_id": skeleton_manifest()["source_id"],
        "coordinates": "RH_Y_UP_Z_FORWARD_X_LEFT_METERS",
        "rotation_space": "local",
        "quaternion_order": "xyzw",
        "fps": FPS,
        "preprocess_version": PREPROCESS_VERSION,
        "joint_names": skeleton.bone_order_names,
        "parents": skeleton.joint_parents.tolist(),
        "neutral_positions_m": skeleton.neutral_joints.tolist(),
        "foot_channels": skeleton.foot_joint_names,
        "sole_offsets_m": skeleton_manifest()["sole_offsets_m"],
        "contact_status": "native automatic labels; not manual contact truth",
        "clip": {
            "filename": record["source"]["filename"],
            "behavior": record["source"]["behavior"],
            "source_sha256": source_hash,
            "split": record["source"]["split"],
            "artifact": window["artifact"],
            "feature_array": feature_name,
        },
        "shard_sha256": shard_sha,
        "model_provenance": checkpoint if runtime_state == "generated_replay" else None,
        "official_fk_max_error_m": fk_error,
        "frames": frames,
    }
    payload = _json_bytes(fixture)
    if len(payload) > MAX_FIXTURE_BYTES:
        raise ValueError(f"Fixture exceeds {MAX_FIXTURE_BYTES} byte UE bound")
    return payload, positions


def _bounds(positions):
    ue = np.asarray([_ue_point(point) for point in positions.reshape(-1, 3)])
    return ue.min(axis=0).tolist(), ue.max(axis=0).tolist()


def export(dataset: Path, output: Path):
    dataset, output = dataset.resolve(), output.resolve()
    report_path = dataset / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    windows = {window["artifact"]: window for window in report["windows"]}
    artifact_for = {"up": "00_up_start000000", "down": "03_down_start000000"}
    checkpoint = {
        "source": "plant-v4-whole-warm/report.json",
        "status": report["status"],
        "quality_approved": report["quality_approved"],
        "root_checkpoint_sha256": report["bundle"]["root_checkpoint_sha256"],
        "pose_checkpoint_sha256": report["bundle"]["pose_checkpoint_sha256"],
        "tokenizer_checkpoint_sha256": report["bundle"]["tokenizer"]["checkpoint_sha256"],
        "root_step": report["bundle"]["root_step"],
        "pose_step": report["bundle"]["pose_step"],
        "tokenizer_step": report["bundle"]["tokenizer"]["global_step"],
        "dataset_manifest_sha256": report["dataset_manifest_sha256"],
    }
    representation, skeleton = build_native_representation(), SOMASkeleton30()
    output.mkdir(parents=True, exist_ok=True)
    exported = {}
    manifest_clips = []
    for direction, artifact in artifact_for.items():
        window = windows[artifact]
        record = window
        shard_path = (dataset / f"{artifact}.npz").resolve()
        if not shard_path.is_relative_to(dataset):
            raise ValueError("Native stair artifact path escaped the review directory")
        shard_bytes = shard_path.read_bytes()
        shard_sha = _sha256(shard_bytes)
        with np.load(shard_path, allow_pickle=False) as arrays:
            if float(arrays["fps"]) != FPS:
                raise ValueError("Native stair shard is not 30 Hz")
            pair = {}
            for suffix, feature_name in (("source", "features_source"), ("model", MODEL_FEATURES)):
                clip_id = f"{direction}_{suffix}"
                payload, positions = _fixture(
                    clip=clip_id, record=record, window=window, feature_name=feature_name,
                    features=arrays[feature_name], shard_sha=shard_sha, checkpoint=checkpoint,
                    representation=representation, skeleton=skeleton)
                filename = f"{clip_id}.json"
                (output / filename).write_bytes(payload)
                pair[clip_id] = {"filename": filename, "positions": positions,
                                 "sha256": _sha256(payload), "feature_sha256": _sha256(np.ascontiguousarray(arrays[feature_name]).tobytes())}
            support = _support_patches(window["support_patches"])
            for clip_id in (f"{direction}_source", f"{direction}_model"):
                entry = pair[clip_id]
                bounds_min, bounds_max = _bounds(entry["positions"])
                manifest_clips.append({
                    "id": clip_id,
                    "fixture": entry["filename"],
                    "bounds_min_ue_cm": bounds_min,
                    "bounds_max_ue_cm": bounds_max,
                    "support_patches": support,
                    "support_evidence": "Weak source-derived support footprints only; incomplete observations, not measured full terrain or manual contact truth.",
                    "review_artifact_sha256": _sha256(shard_bytes),
                    "motion_features_sha256": entry["feature_sha256"],
                    "fixture_sha256": entry["sha256"],
                    "native_source_sha256": window["source"]["source_sha256"],
                    "feature_array": "features_source" if clip_id.endswith("_source") else MODEL_FEATURES,
                })
    manifest = {
        "schema": "motionbricks.soma.stair_review_manifest.v1",
        "description": "Offline UE stair review sidecar. Four bounded recorded/generated replays; not live inference.",
        "runtime_state": "offline_review_sidecar",
        "model_inference": False,
        "motion_postprocessing": "none; native frame order and source/global origin preserved, with no time-warp or root correction",
        "fps": FPS,
        "preprocess_version": PREPROCESS_VERSION,
        "coordinates": "RH_Y_UP_Z_FORWARD_X_LEFT_METERS; UE centimeters map (x,y,z) to (z,-x,y)*100",
        "checkpoint_provenance": checkpoint,
        "source_support_evidence": "Weak observed support footprints only. Rectangular context platforms are synthetic and non-colliding; polygons are not full measured terrain.",
        "clips": manifest_clips,
    }
    manifest_payload = json.dumps(manifest, indent=2, allow_nan=False).encode("utf-8")
    (output / "manifest.json").write_bytes(manifest_payload)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    manifest = export(args.dataset, args.output)
    print(json.dumps({"manifest": str(args.output / "manifest.json"), "clips": CLIP_IDS,
                      "fixture_count": len(manifest["clips"]),
                      "fixture_bytes": {path.name: path.stat().st_size for path in args.output.glob("*.json")}}, indent=2))


if __name__ == "__main__":
    main()
