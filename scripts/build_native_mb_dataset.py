"""Build an independently versioned 30 Hz native MotionBricks SOMA dataset."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from kimodo.exports.motion_io import complete_motion_dict
from kimodo.skeleton.definitions import SOMASkeleton30 as KimodoSOMA30
from motionbricks.data.bones_seed import resample_rotations_root
from motionbricks.data.native_mb_data import (
    CONTRACT_VERSION,
    FPS,
    REPRESENTATION_NAME,
    build_native_representation,
    build_train_stats,
)


SOURCE_FPS = 60.0


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_hashes():
    repo_motionbricks = Path(__file__).resolve().parents[1] / "motionbricks"
    official_sources = {
        "dual_root_global_joints": repo_motionbricks / "motionlib/core/motion_reps/dual_root_global_joints.py",
        "dual_root_local_body": repo_motionbricks / "motionlib/core/motion_reps/motion_reps_base/dual_root_local_body.py",
        "global_root_local_body": repo_motionbricks / "motionlib/core/motion_reps/motion_reps_base/global_root_local_body.py",
        "separated_root_local_body": repo_motionbricks / "motionlib/core/motion_reps/motion_reps_base/seperate_root_local_body.py",
        "motion_features": repo_motionbricks / "motionlib/core/motion_reps/tools/motion_features.py",
        "official_contact_features": repo_motionbricks / "motionlib/core/motion_reps/tools/feet.py",
        "soma_skeleton_adapter": repo_motionbricks / "motionlib/core/skeletons/soma.py",
        "kimodo_motion_io": Path(inspect.getsourcefile(complete_motion_dict)),
        "kimodo_soma30": Path(inspect.getsourcefile(KimodoSOMA30)),
    }
    return {
        "converter": _sha256(__file__),
        "native_mb_data": _sha256(repo_motionbricks / "data/native_mb_data.py"),
        "bones_seed": _sha256(repo_motionbricks / "data/bones_seed.py"),
        "official_representation": {name: _sha256(path) for name, path in official_sources.items()},
    }


def _validate_expansion_source(source, records):
    report_path = source / "expansion_report.json"
    skeleton_path = source / "skeleton_contract.json"
    expanded = skeleton_path.exists() and "expansion_note" in json.loads(skeleton_path.read_text())
    if not report_path.exists():
        if expanded:
            raise ValueError("Expanded source has no completed expansion report")
        return None
    report = json.loads(report_path.read_text())
    counts = {split: sum(row["split"] == split for row in records)
              for split in ("train", "validation", "test")}
    if report.get("status") != "completed" or report.get("split_counts") != counts:
        raise ValueError("Expansion report completion/count mismatch")
    original = Path(report["source_pilot"]) / "accepted.json"
    if _sha256(original) != report["source_accepted_sha256"]:
        raise ValueError("Frozen pilot manifest identity changed")
    prior = json.loads(original.read_text())
    if ([row for row in records if row["split"] != "train"] !=
            [row for row in prior if row["split"] != "train"]):
        raise ValueError("Expanded source changed frozen held-out records")
    for relative, expected in report["copied_pilot_shard_sha256"].items():
        path = (source / relative).resolve()
        if source not in path.parents or _sha256(path) != expected:
            raise ValueError("Expanded source changed a copied pilot shard")
    if _sha256(source / "reserved_content_hashes.json") != report["reserved_content_hashes_sha256"]:
        raise ValueError("Reserved held-out source identities changed")
    return {"report_sha256": _sha256(report_path), "split_counts": counts,
            "frozen_pilot_manifest_sha256": report["source_accepted_sha256"],
            "heldout_identity": "exact pilot record dictionaries and copied shard bytes preserved"}


def build_dataset(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest_path = source / "accepted.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    if output == source or source in output.parents:
        raise ValueError("Output must be separate from and outside the source dataset")

    source_manifest_hash = _sha256(manifest_path)
    source_records = json.loads(manifest_path.read_text(encoding="utf-8"))
    expansion_identity = _validate_expansion_source(source, source_records)
    missing = [record for record in source_records if not (source / record["shard"]).is_file()]
    if missing:
        sample = ", ".join(record.get("filename", record["shard"]) for record in missing[:8])
        raise FileNotFoundError(
            f"{len(missing)} of {len(source_records)} accepted source shards are missing; "
            f"refusing a partial native dataset. Examples: {sample}"
        )
    code_hashes = _code_hashes()
    contract = {
        "version": CONTRACT_VERSION,
        "representation": REPRESENTATION_NAME,
        "source_manifest_sha256": source_manifest_hash,
        "source_shard_fps": SOURCE_FPS,
        "target_fps": FPS,
        "resampling": "bones_seed.resample_rotations_root; quaternion SLERP; 4th-order zero-phase low-pass before downsampling",
        "features": "official MotionBricks representation recomputed at target FPS from local rotations and root translation",
        "ground": "source fixed ground plane retained; no per-frame fitting or vertical correction",
        "splits": "copied from source accepted.json; unavailable source shards are listed as omitted",
        "statistics": "train split only; four-way yaw augmentation; 65 global frames provide 64 local intervals",
        "statistics_folder": "stats",
        "custom_supervision": "none",
        "code_sha256": code_hashes,
        "expansion_identity": expansion_identity,
    }
    if output.exists():
        existing = output / "native_manifest.json"
        if not existing.is_file():
            raise FileExistsError(f"Refusing to use existing output without a native manifest: {output}")
        old_contract = json.loads(existing.read_text(encoding="utf-8")).get("contract")
        if old_contract != contract:
            raise FileExistsError("Refusing to overwrite dataset with a differing conversion contract")
        raise FileExistsError("Dataset with this conversion contract already exists")

    representation = build_native_representation()
    output.mkdir(parents=True)
    shard_dir = output / "shards"
    shard_dir.mkdir()
    accepted = []
    for source_record in source_records:
        source_shard = source / source_record["shard"]
        with np.load(source_shard, allow_pickle=False) as data:
            if not {"local_rot_mats", "root_positions", "global_features"}.issubset(data.files):
                raise ValueError(f"{source_shard} lacks source rotations/root/global features")
            rotations = data["local_rot_mats"].copy()
            root = data["root_positions"].copy()
            source_features = data["global_features"]
        if rotations.ndim != 4 or rotations.shape[1:] != (30, 3, 3):
            raise ValueError(f"Unexpected local rotation shape in {source_shard}: {rotations.shape}")
        if root.shape != (len(rotations), 3) or len(rotations) < 20:
            raise ValueError(f"Unexpected root shape or clip shorter than resampling filter: {source_shard}")
        if source_features.shape != (len(rotations), representation.motion_rep_dim):
            raise ValueError(f"Source global features do not align with 60 Hz rotations: {source_shard}")
        if not (np.isfinite(rotations).all() and np.isfinite(root).all()
                and np.isfinite(source_features).all()):
            raise ValueError(f"Nonfinite source arrays: {source_shard}")
        declared_frames = source_record.get("frames_60hz")
        if declared_frames is not None and declared_frames != len(rotations):
            raise ValueError(f"Manifest/shard frame count mismatch: {source_shard}")

        rotations30, root30 = resample_rotations_root(rotations, root, SOURCE_FPS, FPS)
        local_rots_t = torch.from_numpy(rotations30)
        root_t = torch.from_numpy(root30)
        motion30 = complete_motion_dict(local_rots_t, root_t, representation.skeleton, FPS)
        dual, initial_transform = representation.dual_rep({
            "posed_joints": motion30["posed_joints"],
            "global_joint_rots": motion30["global_rot_mats"],
            "local_joint_rots": motion30["local_rot_mats"],
            "translation": motion30["root_positions"],
        }, to_normalize=False, return_init_heading_info=True)
        global_features = representation.dual_rep.get_feature_subset(dual, mode="global")
        if not torch.isfinite(dual).all() or len(dual) != len(rotations30):
            raise ValueError(f"Invalid regenerated features in {source_shard}")
        roundtrip = representation.inverse(
            global_features, is_normalized=False, init_heading_info=initial_transform,
            return_all=True, return_quat=False,
        )["posed_joints"]
        roundtrip_error = (roundtrip - motion30["posed_joints"]).norm(dim=-1).max().item()
        if roundtrip_error > 0.002:
            raise ValueError(f"Feature/FK roundtrip mismatch for {source_shard}: {roundtrip_error:.6f} m")
        canonical = representation.inverse(
            global_features, is_normalized=False, return_all=True, return_quat=False,
        )
        saved_rotations = canonical["local_joint_rots"].cpu().numpy().astype(np.float32)
        saved_root = canonical["posed_joints"][:, representation.skeleton.root_idx].cpu().numpy().astype(np.float32)
        relative_shard = Path("shards") / (Path(source_record["shard"]).stem + "_30hz.npz")
        np.savez_compressed(
            output / relative_shard,
            global_features=global_features.numpy().astype(np.float32),
            dual_features=dual.numpy().astype(np.float32),
            local_rot_mats=saved_rotations,
            root_positions=saved_root,
            fps=np.asarray(FPS, dtype=np.float32),
            feature_fk_roundtrip_error_m=np.asarray(roundtrip_error, dtype=np.float32),
        )
        converted_shard_hash = _sha256(output / relative_shard)
        accepted.append({
            "filename": source_record["filename"],
            "recording_group": source_record.get("recording_group"),
            "behavior": source_record.get("behavior"),
            "split": source_record["split"],
            "source_shard": source_record["shard"],
            "source_shard_sha256": _sha256(source_shard),
            "converted_shard_sha256": converted_shard_hash,
            "source_shard_fps": SOURCE_FPS,
            "source_original_fps": source_record.get("source_fps"),
            "source_fixed_plane_y_m": source_record.get("fixed_plane_source_y_m"),
            "shard": relative_shard.as_posix(),
            "frames_30hz": len(dual),
            "duration_seconds": (len(dual) - 1) / FPS,
        })

    if not accepted:
        raise RuntimeError("No source shards were available to convert")
    (output / "accepted.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    accepted_hash = _sha256(output / "accepted.json")
    stats_contract = build_train_stats(output, accepted, source_manifest_hash)
    stats_hashes = {
        "mean.npy": _sha256(output / "stats" / "mean.npy"),
        "std.npy": _sha256(output / "stats" / "std.npy"),
    }
    artifact_hashes = {
        "accepted.json": accepted_hash,
        "stats/mean.npy": stats_hashes["mean.npy"],
        "stats/std.npy": stats_hashes["std.npy"],
        **{row["shard"]: row["converted_shard_sha256"] for row in accepted},
    }
    if expansion_identity is not None:
        shutil.copy2(source / "expansion_report.json", output / "expansion_report.json")
        artifact_hashes["expansion_report.json"] = _sha256(output / "expansion_report.json")
    provenance = {
        "contract": contract,
        "source_dataset": str(source),
        "source_manifest": "accepted.json",
        "source_manifest_sha256": source_manifest_hash,
        "accepted_json_sha256": accepted_hash,
        "stats_artifacts_sha256": stats_hashes,
        "artifact_sha256": artifact_hashes,
        "code_sha256": code_hashes,
        "source_rows": len(source_records),
        "converted_rows": len(accepted),
        "split_counts": {split: sum(row["split"] == split for row in accepted)
                         for split in ("train", "validation", "test")},
        "stats_contract": stats_contract,
        "limitations": [
            "Official foot-contact features are regenerated from SOMA joint positions and velocities; no custom contact confidence or sole-region supervision is included.",
            "The existing fixed source ground plane is preserved; this adapter does not refit it.",
        ],
    }
    (output / "native_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source pilot-v7 dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="New 30 Hz dataset directory")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(build_dataset(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
