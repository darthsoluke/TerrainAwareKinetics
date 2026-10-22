"""Prepare a reproducible real-data pilot without modifying source BONES files."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from kimodo.exports.bvh import bvh_to_kimodo_motion
from kimodo.exports.motion_io import complete_motion_dict
from kimodo.skeleton.bvh import Bvh, SkeletonBvh
from kimodo.skeleton.registry import build_skeleton
from motionbricks.data.bones_seed import (
    FPS, SOMA_REVISION, build_representation, calibration_crop, estimate_contacts,
    official_sole_offsets, recording_group, resample_rotations_root,
    sole_positions, split_for_group,
    validate_planar_posture,
)


EXCLUDED = ("stair", "ladder", "climb", "chair", "seat", "sitting", "jump", "vault",
            "crawl", "kneel", "lying", "lie down", "hill", "slope", "dance", "swim",
            "injur", "crouch", "crate", "box", "carry", "carrying", "lift", "weapon",
            "rifle", "pistol", "sword", "shoot", "attack", "fight", "combat", "punch",
            "kick", "drunk", "zombie", "scream", "push", "pull", "broom", "cart", "impro",
            "handstand", "somersault", "acrobat", "grab", "reach", "relax", "bend", "bow",
            "salute", "waving", "wave", "yoga", "tpose", "t-pose", "t_pose", "holding")

PLANAR_NAME_TOKENS = {
    "neutral", "walk", "jog", "run", "sprint", "idle", "turn", "strafe", "sidestep",
    "sideway", "sideways", "backward", "backwards", "forward", "ff", "start", "stop",
    "loop", "arc", "circle", "cw", "ccw", "r", "l", "to", "right", "left", "normal",
    "fast", "slow", "very", "speed", "straight",
}


def behavior(annotation, filename):
    if not re.match(r"^(?:neutral_)?(?:walk|jog|run|sprint|idle|turn|strafe|sidestep)[_-]",
                    filename, flags=re.IGNORECASE):
        return None
    stem = re.sub(r"__A\d+(?:_M)?$", "", filename, flags=re.IGNORECASE).lower()
    # A conservative filename grammar keeps new unreviewed actions out of V1.
    if any(token not in PLANAR_NAME_TOKENS and not token.isdigit()
           for token in re.split(r"[_-]+", stem) if token):
        return None
    text = (annotation.get("overview_description", "") + " " + filename + " " +
            " ".join(event.get("description", "") for event in annotation.get("events", []))).lower()
    if any(word in text for word in EXCLUDED):
        return None
    for label, words in (
        ("backward", ("backward", "backwards")), ("lateral", ("strafe", "sideway", "sidestep")),
        ("turn", ("turn", "arc", "circle")), ("run", ("run", "jog", "sprint")),
        ("walk", ("walk",)), ("idle", ("idle", "standing still", "stands still")),
    ):
        # Descriptions alone label many unrelated gestures as standing/walking.
        # Require an explicit locomotion signal in the source filename too.
        if any(word in filename.lower() for word in words):
            return label
    return None


def inventory(source, timeline_path, output, limit):
    annotations = {}
    with timeline_path.open(encoding="utf-8") as stream:
        for line in stream:
            annotation = json.loads(line)
            annotations[annotation["filename"]] = annotation
    # Propagated annotation families stay together, in addition to recording groups.
    parents = {}

    def find(group):
        parents.setdefault(group, group)
        if parents[group] != group:
            parents[group] = find(parents[group])
        return parents[group]

    for name, annotation in annotations.items():
        propagated = annotation.get("propagated_from_filename")
        if propagated:
            left, right = find(recording_group(name)), find(recording_group(propagated))
            parents[max(left, right)] = min(left, right)
    candidates = []
    unlabelled = 0
    for path in sorted(source.rglob("*.bvh")):
        annotation = annotations.get(path.stem)
        if annotation is None:
            unlabelled += 1
            continue
        label = behavior(annotation, path.stem)
        if label is None:
            continue
        group = find(recording_group(path.stem))
        candidates.append({
            "path": str(path), "filename": path.stem, "recording_group": group,
            "split": split_for_group(group), "behavior": label,
            "actor": path.stem.rsplit("__", 1)[-1],
            "description": annotation["overview_description"],
            "events": annotation.get("events", []),
        })
    with (output / "candidate_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for candidate in candidates:
            stream.write(json.dumps(candidate) + "\n")
    # Round-robin over behaviors and split targets; selection is seeded by file hashes.
    buckets = defaultdict(list)
    for record in candidates:
        buckets[(record["split"], record["behavior"])].append(record)
    for values in buckets.values():
        values.sort(key=lambda record: hashlib.sha256(record["path"].encode()).hexdigest())
    selected = []
    targets = {"train": int(limit * 0.8), "validation": max(1, int(limit * 0.1))}
    targets["test"] = limit - targets["train"] - targets["validation"]
    for split, target in targets.items():
        split_records = []
        while len(split_records) < target:
            added = False
            for key in sorted(buckets):
                if key[0] == split and buckets[key]:
                    split_records.append(buckets[key].pop())
                    added = True
                    if len(split_records) >= target:
                        break
            if not added:
                break
        selected.extend(split_records)
    return selected, {"eligible_candidates": len(candidates), "unlabelled_bvh_count": unlabelled,
                      "candidate_behaviors": dict(Counter(record["behavior"] for record in candidates))}


def convert(record, full_skeleton, representation, offsets):
    path = Path(record["path"])
    contents = path.read_text()
    bvh = Bvh(contents, backend="np")
    validate_header(bvh, full_skeleton)
    wrapper = np.asarray(bvh.frames_joint_channels("Root", bvh.joint_channels("Root")))
    if np.max(np.abs(wrapper)) > 1e-5:
        raise ValueError("Animated Root wrapper cannot be silently discarded")
    parsed = SkeletonBvh()
    parsed.load_from_bvh(str(path), exclude_bones={"Root"}, mocap=bvh)
    neutral = torch.as_tensor(parsed.get_neutral_joints() * 0.01)
    neutral -= neutral[0].clone()
    offset_error = (neutral - full_skeleton.bvh_neutral_joints.cpu()).norm(dim=-1).max().item()
    if offset_error > 0.002:
        raise ValueError(f"BVH reference geometry differs from official SOMA ({offset_error:.4f} m)")
    del bvh, contents
    return convert_motion(path, full_skeleton, representation, offsets, offset_error)


def validate_header(bvh, full_skeleton):
    names = bvh.get_joints_names()
    if names != ["Root"] + full_skeleton.bone_order_names:
        raise ValueError("Joint order does not match official SOMA77 plus Root wrapper")
    expected_rotation_channels = [channel for channel in bvh.joint_channels("Hips") if channel.endswith("rotation")]
    for name, parent in full_skeleton.bone_order_names_with_parents:
        expected_parent = "Root" if parent is None else parent
        if bvh.joint_parent(name).value[1] != expected_parent:
            raise ValueError(f"SOMA parent mismatch for {name}")
        channels = bvh.joint_channels(name)
        if [channel for channel in channels if channel.endswith("rotation")] != expected_rotation_channels:
            raise ValueError(f"Inconsistent Euler rotation order for {name}")
        if name != "Hips" and any(channel.endswith("position") for channel in channels):
            raise ValueError(f"Unsupported non-root position channels for {name}")


def convert_motion(path, full_skeleton, representation, offsets, offset_error):
    motion, source_fps = bvh_to_kimodo_motion(path, full_skeleton)
    rotations, root = resample_rotations_root(motion["local_rot_mats"].numpy(),
                                             motion["root_positions"].numpy(), source_fps)
    if len(root) < 129:
        raise ValueError("Less than 129 frames after 60 Hz resampling")
    reduced = representation.skeleton.from_SOMASkeleton77(torch.from_numpy(rotations))
    motion30 = complete_motion_dict(reduced, torch.from_numpy(root), representation.skeleton, FPS)
    motion77 = complete_motion_dict(torch.from_numpy(rotations).double(),
                                    torch.from_numpy(root).double(), full_skeleton, FPS)
    indices77 = representation.skeleton.get_skel_slice(full_skeleton)
    body_names = [i for i, name in enumerate(representation.skeleton.bone_order_names)
                  if "HandThumbEnd" not in name and "HandMiddleEnd" not in name]
    fk_error = (motion30["posed_joints"][:, body_names].double() -
                motion77["posed_joints"][:, indices77][:, body_names]).norm(dim=-1).max().item()
    if fk_error > 0.002:
        raise ValueError(f"SOMA30/77 body FK mismatch: {fk_error:.4f} m")
    crop_start, crop_end = calibration_crop(motion30["posed_joints"], representation.skeleton)
    motion30 = {key: value[crop_start:crop_end].clone() for key, value in motion30.items()}
    for key, tensor in motion30.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite tensor: {key}")
    sole = sole_positions(motion30["posed_joints"], motion30["global_rot_mats"],
                          representation.skeleton, offsets)
    contacts, confidence, plane, contact_report = estimate_contacts(sole)
    posture_report = validate_planar_posture(motion30["posed_joints"], representation.skeleton, plane, offsets)
    # Translate only once by the fixed support plane; preserve vertical root motion.
    motion30["posed_joints"][..., 1] -= plane
    motion30["root_positions"][..., 1] -= plane
    motion30["foot_contacts"] = contacts
    features, initial_transform = representation.dual_rep({
        "posed_joints": motion30["posed_joints"],
        "global_joint_rots": motion30["global_rot_mats"],
        "local_joint_rots": motion30["local_rot_mats"],
        "translation": motion30["root_positions"], "foot_contacts": contacts,
    }, to_normalize=False, return_init_heading_info=True)
    if not torch.isfinite(features).all():
        raise ValueError("Nonfinite MB features")
    # MB canonicalizes initial heading/position. Validate in the original frame,
    # then save the canonical frame consistently for both features and replay.
    global_features = representation.dual_rep.get_feature_subset(features, "global")
    reconstructed = representation.inverse(global_features, is_normalized=False,
                                           init_heading_info=initial_transform,
                                           return_all=True, return_quat=False)["posed_joints"]
    roundtrip = (reconstructed - motion30["posed_joints"]).norm(dim=-1).max().item()
    if roundtrip > 0.002:
        raise ValueError(f"MB feature roundtrip mismatch: {roundtrip:.4f} m")
    # Dual statistics must describe the exact recentered motion saved in the shard.
    local_features = representation.dual_rep.global_to_local(global_features, False, False)
    usable = len(local_features)
    canonical = representation.inverse(global_features, is_normalized=False,
                                       return_all=True, return_quat=False)
    dual_features = torch.cat([global_features[:usable, representation.indices["root"]],
                               local_features[:, :4], global_features[:usable, 5:]], -1)
    arrays = {
        "global_features": global_features[:usable].numpy().astype(np.float32),
        "dual_features": dual_features.numpy().astype(np.float32),
        "contact_confidence": confidence[:usable].numpy(),
        "local_rot_mats": canonical["local_joint_rots"][:usable].numpy(),
        "root_positions": canonical["posed_joints"][:usable, 0].numpy(),
    }
    report = {"source_fps": source_fps, "frames_60hz": usable, "duration_seconds": usable / FPS,
              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "fixed_plane_source_y_m": plane, "reference_geometry_error_m": offset_error,
              "source_crop_start_seconds": crop_start / FPS, "source_crop_end_seconds": crop_end / FPS,
              "body_subset_fk_error_m": fk_error, "feature_roundtrip_error_m": roundtrip,
              **contact_report, **posture_report}
    return arrays, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--timelines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=192)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "shards").mkdir()
    selected, summary = inventory(args.source, args.timelines, args.output, args.limit)
    full_skeleton = build_skeleton(77)
    representation = build_representation()
    offsets, mesh_hash = official_sole_offsets(representation.skeleton)
    skeleton_spec = {
        "preprocess_version": "soma60-planar-v7",
        "kimodo_revision": SOMA_REVISION, "joint_names": representation.skeleton.bone_order_names,
        "parents": representation.skeleton.joint_parents.tolist(),
        "neutral_joints_m": representation.skeleton.neutral_joints.tolist(),
        "up_axis": "Y", "fps": FPS, "foot_channels": representation.skeleton.foot_joint_names,
        "sole_offsets_m": offsets.tolist(), "sole_mesh_sha256": mesh_hash,
        "sole_geometry_status": "rigid mesh proxies; manual replay review still required",
        "contact_thresholds": {"speed_mps": 0.20, "height_m": 0.035, "persistence_seconds": 0.05},
    }
    (args.output / "skeleton_contract.json").write_text(json.dumps(skeleton_spec, indent=2))
    accepted, rejected, hashes = [], [], {}
    count, mean, m2 = 0, None, None
    for number, record in enumerate(selected):
        try:
            arrays, report = convert(record, full_skeleton, representation, offsets)
            digest = report["source_sha256"]
            if digest in hashes:
                raise ValueError("Duplicate source content; original kept in " + hashes[digest])
            hashes[digest] = record["split"]
            shard = f"shards/{digest[:20]}.npz"
            np.savez_compressed(args.output / shard, **arrays)
            accepted.append({**record, **report, "shard": shard})
            if record["split"] == "train":
                # Match the official training loop's random heading augmentation.
                # Four uniform yaw angles reproduce its first/second moments,
                # instead of giving near-zero X/std to canonically forward motion.
                heading_augmented = []
                for heading in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
                    global_augmented = representation.change_first_heading(
                        torch.from_numpy(arrays["global_features"]), heading, False, False)
                    local_augmented = representation.dual_rep.global_to_local(global_augmented, False, False)
                    heading_augmented.append(torch.cat([global_augmented[:, :5], local_augmented[:, :4],
                                                         global_augmented[:, 5:]], -1).numpy())
                data = np.concatenate(heading_augmented).astype(np.float64)
                n = len(data)
                clip_mean, clip_m2 = data.mean(0), ((data - data.mean(0)) ** 2).sum(0)
                if count == 0:
                    count, mean, m2 = n, clip_mean, clip_m2
                else:
                    delta = clip_mean - mean
                    m2 += clip_m2 + delta ** 2 * count * n / (count + n)
                    mean += delta * n / (count + n)
                    count += n
            print(f"{number + 1}/{len(selected)} accepted {record['filename']} ({record['split']})", flush=True)
        except (ValueError, RuntimeError, IndexError, KeyError) as error:
            rejected.append({**record, "reason": str(error)})
            print(f"REJECT {record['filename']}: {error}", flush=True)
    (args.output / "accepted.json").write_text(json.dumps(accepted, indent=2))
    (args.output / "rejected.json").write_text(json.dumps(rejected, indent=2))
    if not count:
        raise RuntimeError("No accepted training frames; inspect rejection report")
    stats = args.output / "stats"
    stats.mkdir()
    np.save(stats / "mean.npy", mean.astype(np.float32))
    np.save(stats / "std.npy", np.sqrt(m2 / count).astype(np.float32))
    summary.update({"accepted": len(accepted), "rejected": len(rejected),
                    "splits": dict(Counter(record["split"] for record in accepted)),
                    "accepted_behaviors": dict(Counter(record["behavior"] for record in accepted)),
                    "training_stats_frames": count, "full_training_gate_passed": False,
                    "timeline_sha256": hashlib.sha256(args.timelines.read_bytes()).hexdigest()})
    summary["semantic_policy"] = {"filename_token_allowlist": sorted(PLANAR_NAME_TOKENS),
                                  "automatic_rejection_only": True, "manual_semantic_review_passed": False}
    (args.output / "quality_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
