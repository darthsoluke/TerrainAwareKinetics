"""Build a native-feature stair sidecar without changing MotionBricks training."""

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from kimodo.exports.bvh import bvh_to_kimodo_motion
from kimodo.exports.motion_io import complete_motion_dict
from kimodo.skeleton.bvh import Bvh, SkeletonBvh
from kimodo.skeleton.registry import build_skeleton
from motionbricks.data.bones_seed import official_sole_surfaces, resample_rotations_root
from motionbricks.data.native_mb_data import (
    CONTRACT_VERSION, FPS, REPRESENTATION_NAME, build_native_representation,
)
from motionbricks.data.sole_region_contact import sole_region_indices
from motionbricks.data.soma_skin_reference import load_official_skinner
from motionbricks.data.stair_support import SupportConfig, reconstruct_support
from motionbricks.motionlib.core.utils.rotations import quat_apply, quat_conjugate


MIN_FRAMES = 65
BODY_FK_EXCLUDED = ("HandThumbEnd", "HandMiddleEnd")
REJECT_NAME = ("dance", "slip", "slide")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def validate_header(bvh, full_skeleton):
    if bvh.get_joints_names() != ["Root"] + full_skeleton.bone_order_names:
        raise ValueError("Joint order does not match official SOMA77 plus Root wrapper")
    expected = [c for c in bvh.joint_channels("Hips") if c.endswith("rotation")]
    for name, parent in full_skeleton.bone_order_names_with_parents:
        expected_parent = "Root" if parent is None else parent
        if bvh.joint_parent(name).value[1] != expected_parent:
            raise ValueError(f"SOMA parent mismatch for {name}")
        channels = bvh.joint_channels(name)
        if [c for c in channels if c.endswith("rotation")] != expected:
            raise ValueError(f"Inconsistent Euler rotation order for {name}")
        if name != "Hips" and any(c.endswith("position") for c in channels):
            raise ValueError(f"Unsupported non-root position channels for {name}")


def stair_actor(path):
    """Return (actor, direction) only for the requested climbing stair clips."""
    stem = path.stem.lower()
    if "stairs_climbing" not in stem or any(token in stem for token in REJECT_NAME):
        return None
    match = re.search(r"__a(\d+)(?:_m)?$", stem)
    if not match or match.group(1) not in {"300", "301"}:
        return None
    if re.search(r"(?:^|[_-])(?:up|upstairs)(?:[_-]|$)", stem):
        direction = "up"
    elif re.search(r"(?:^|[_-])(?:down|downstairs)(?:[_-]|$)", stem):
        direction = "down"
    else:
        return None
    return int(match.group(1)), direction


def inherited_flat_rows(reference_dataset, output):
    rows = json.loads((reference_dataset / "accepted.json").read_text(encoding="utf-8"))
    copied = []
    for row in rows:
        source = (reference_dataset / row["shard"]).resolve()
        if reference_dataset.resolve() not in source.parents:
            raise ValueError(f"Flat shard escapes reference dataset: {row['shard']}")
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / row["shard"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(dict(row))
    return copied


def _skin_sole_regions(local77, posed77, skinner, surfaces, chunk_frames=96):
    local77 = torch.as_tensor(local77, dtype=torch.float32)
    posed77 = torch.as_tensor(posed77, dtype=torch.float32)
    side_indices = {side: sole_region_indices(surfaces, side.title())
                    for side in ("left", "right")}
    selected = np.unique(np.concatenate(list(side_indices.values())))
    if len(selected) != 196 or any(len(ids) != 98 for ids in side_indices.values()):
        raise ValueError("Official SOMA sole regions must contain 98 unique vertices per side")
    tensors = {name: getattr(skinner, name) for name in
               ("bind_vertices", "lbs_indices", "lbs_weights")}
    try:
        with torch.inference_mode():
            expected = skinner.skin(local77[:2], posed77[:2], rot_is_global=False)
            expected = expected.index_select(1, torch.as_tensor(selected, dtype=torch.long))
            for name, tensor in tensors.items():
                setattr(skinner, name, tensor.index_select(0, torch.as_tensor(selected)))
            actual = skinner.skin(local77[:2], posed77[:2], rot_is_global=False)
            parity = float((expected - actual).abs().max())
            if parity > 1e-6:
                raise ValueError(f"Selected 196-vertex LBS parity error: {parity:.9g} m")
            selected_frames = []
            for start in range(0, len(local77), chunk_frames):
                end = min(start + chunk_frames, len(local77))
                selected_frames.append(skinner.skin(
                    local77[start:end], posed77[start:end], rot_is_global=False
                ).cpu().numpy())
    finally:
        for name, tensor in tensors.items():
            setattr(skinner, name, tensor)
    selected_frames = np.concatenate(selected_frames)
    lookup = {int(value): index for index, value in enumerate(selected)}
    return ({side: selected_frames[:, [lookup[int(i)] for i in indices]].astype(np.float32)
             for side, indices in side_indices.items()}, side_indices, parity)


def _weak_contact_channels(weak_arrays, surfaces, side_indices, frames, foot_joint_names):
    labels = np.zeros((frames, 4), dtype=np.uint8)
    mask = np.zeros((frames, 4), dtype=np.uint8)
    confidence = np.zeros((frames, 4), dtype=np.float32)
    if len(foot_joint_names) != 4 or set(foot_joint_names) != {
            "LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"}:
        raise ValueError(f"Unexpected official four-channel foot order: {foot_joint_names}")
    for column, channel in enumerate(foot_joint_names):
        side = "left" if channel.startswith("Left") else "right"
        side_lookup = {int(vertex): i for i, vertex in enumerate(side_indices[side])}
        surface = next(item for item in surfaces if item["channel"] == channel)
        vertices = np.asarray([side_lookup[int(index)] for index in surface["selected_indices"]])
        evidence = weak_arrays[f"vertex_measured_{side}"][:, vertices].any(axis=1)
        evidence_frames = np.zeros(frames, dtype=bool)
        evidence_frames[:-1] |= evidence
        evidence_frames[1:] |= evidence
        labels[:, column] = evidence_frames
        mask[:, column] = evidence_frames
        confidence[:, column] = evidence_frames.astype(np.float32) * 0.5
    return labels, mask, confidence


def _canonicalize_points(points, initial_heading_info):
    points = torch.as_tensor(points, dtype=torch.float32)
    heading = initial_heading_info["init_heading_quat"].to(torch.float32).reshape(-1, 4)[0]
    heading_inv = quat_conjugate(heading)
    origin_xz = initial_heading_info["root_pos_init_xz"].to(torch.float32).reshape(-1, 2)[0]
    offset = torch.stack((origin_xz[0], origin_xz.new_zeros(()), origin_xz[1]))
    return quat_apply(heading_inv, points - offset).cpu().numpy()


def _skin_canonical_regions(local30, posed30, skinner30, side_indices,
                            expected_source, initial_heading_info, chunk_frames=96):
    local30 = torch.as_tensor(local30, dtype=torch.float32)
    posed30 = torch.as_tensor(posed30, dtype=torch.float32)
    selected = np.unique(np.concatenate(list(side_indices.values())))
    tensors = {name: getattr(skinner30, name) for name in
               ("bind_vertices", "lbs_indices", "lbs_weights")}
    try:
        with torch.inference_mode():
            first = skinner30.skin(local30[:2], posed30[:2], rot_is_global=False)
            expected = np.concatenate([expected_source[s][:2] for s in ("left", "right")], axis=1)
            ordered_indices = np.concatenate([side_indices[s] for s in ("left", "right")])
            actual = first.index_select(1, torch.as_tensor(ordered_indices, dtype=torch.long))
            expected_t = torch.from_numpy(_canonicalize_points(expected, initial_heading_info))
            parity = float((actual - expected_t).abs().max())
            if parity > 0.003:
                raise ValueError(f"Canonical SOMA30/source SOMA77 sole parity exceeds 3 mm: {parity:.6f} m")
            for name, tensor in tensors.items():
                setattr(skinner30, name, tensor.index_select(0, torch.as_tensor(selected)))
            frames = []
            for start in range(0, len(local30), chunk_frames):
                end = min(start + chunk_frames, len(local30))
                frames.append(skinner30.skin(local30[start:end],
                                             posed30[start:end], rot_is_global=False).cpu().numpy())
    finally:
        for name, tensor in tensors.items():
            setattr(skinner30, name, tensor)
    vertices = np.concatenate(frames)
    lookup = {int(vertex): index for index, vertex in enumerate(selected)}
    return ({side: vertices[:, [lookup[int(i)] for i in ids]].astype(np.float32)
             for side, ids in side_indices.items()}, parity)


def convert_stair(path, actor, direction, full_skeleton, representation, skinner,
                  surfaces, output, source_hash=None):
    contents = path.read_text(encoding="utf-8", errors="strict")
    bvh = Bvh(contents, backend="np")
    validate_header(bvh, full_skeleton)
    wrapper = np.asarray(bvh.frames_joint_channels("Root", bvh.joint_channels("Root")))
    if wrapper.size and np.max(np.abs(wrapper)) > 1e-5:
        raise ValueError("Animated Root wrapper cannot be silently discarded")
    parsed = SkeletonBvh()
    parsed.load_from_bvh(str(path), exclude_bones={"Root"}, mocap=bvh)
    neutral = torch.as_tensor(parsed.get_neutral_joints() * 0.01)
    neutral -= neutral[0].clone()
    geometry_error = float((neutral - full_skeleton.bvh_neutral_joints.cpu()).norm(dim=-1).max())
    if geometry_error > 0.002:
        raise ValueError(f"BVH reference geometry differs from official SOMA77 ({geometry_error:.4f} m)")
    del bvh, contents

    native, source_fps = bvh_to_kimodo_motion(path, full_skeleton)
    native_rot = native["local_rot_mats"].cpu().numpy().astype(np.float32)
    native_root = native["root_positions"].cpu().numpy().astype(np.float32)
    native_root_hash = sha256_array(native_root)
    native_rot_hash = sha256_array(native_rot)
    rotations30, root30 = resample_rotations_root(native_rot, native_root, source_fps, FPS)
    if len(root30) < MIN_FRAMES:
        raise ValueError(f"Fewer than {MIN_FRAMES} target frames")

    motion77 = complete_motion_dict(torch.from_numpy(rotations30).double(),
                                    torch.from_numpy(root30).double(), full_skeleton, FPS)
    motion30 = complete_motion_dict(
        representation.skeleton.from_SOMASkeleton77(torch.from_numpy(rotations30)),
        torch.from_numpy(root30), representation.skeleton, FPS)
    indices77 = representation.skeleton.get_skel_slice(full_skeleton)
    body = [i for i, name in enumerate(representation.skeleton.bone_order_names)
            if not any(excluded in name for excluded in BODY_FK_EXCLUDED)]
    fk_error = float((motion30["posed_joints"][:, body].double() -
                      motion77["posed_joints"][:, indices77][:, body]).norm(dim=-1).max())
    if fk_error > 0.002:
        raise ValueError(f"SOMA30/77 body FK mismatch: {fk_error:.6f} m")

    posed77_source = motion77["posed_joints"].cpu().numpy().astype(np.float32)
    soles, side_indices, parity = _skin_sole_regions(
        rotations30, posed77_source, skinner, surfaces)
    weak_arrays, support = reconstruct_support(
        soles, root30, FPS, SupportConfig(require_stairs=True),
        posed77_source)
    if support["summary"]["pair_status"] != "weak_pair_candidate":
        raise ValueError("insufficient_geometry: " + ", ".join(support["summary"]["reasons"]))
    source_world_patches = copy.deepcopy(support["patches"])
    source_world_soles = {side: value.copy() for side, value in soles.items()}
    weak_patches = [p for p in support["patches"] if p["status"] == "weak_support"]
    floor_shift = float(np.median([p["height_m"] for p in weak_patches
                                  if p["level_id"] == min(q["level_id"] for q in weak_patches)]))
    root30 = root30.copy()
    root30[:, 1] -= floor_shift
    for side in soles:
        soles[side][..., 1] -= floor_shift
    for patch in support["patches"]:
        patch["height_m"] -= floor_shift
    for event in support["events"]:
        for key in ("height_m", "event_fit_height_m"):
            if key in event:
                event[key] -= floor_shift
    support["summary"]["source_lowest_weak_level_y_m"] = floor_shift
    support["summary"]["usable_level_heights_m"] = [height - floor_shift
                                                        for height in support["summary"]["usable_level_heights_m"]]
    support["summary"]["recording_plane_y_m"] = 0.0
    support["protocol"]["constant_ground_translation_m"] = -floor_shift

    contacts, contact_mask, contact_confidence = _weak_contact_channels(
        weak_arrays, surfaces, side_indices, len(root30),
        [representation.skeleton.bone_order_names[i]
         for i in representation.skeleton.foot_joint_idx])
    motion30 = complete_motion_dict(
        representation.skeleton.from_SOMASkeleton77(torch.from_numpy(rotations30)),
        torch.from_numpy(root30), representation.skeleton, FPS)
    # The native representation has four foot-state channels. Populate only positive
    # weak stationary-patch evidence; unknowns are zero tokens, not airborne truth.
    motion30["foot_contacts"] = torch.from_numpy(contacts.astype(np.float32))
    # Only official global features are stored; stair labels remain sidecar metadata.
    dual, initial = representation.dual_rep({
        "posed_joints": motion30["posed_joints"],
        "global_joint_rots": motion30["global_rot_mats"],
        "local_joint_rots": motion30["local_rot_mats"],
        "translation": motion30["root_positions"],
        "foot_contacts": motion30["foot_contacts"],
    }, to_normalize=False, return_init_heading_info=True)
    global_features = representation.dual_rep.get_feature_subset(dual, "global")
    roundtrip = representation.inverse(global_features, is_normalized=False,
                                        init_heading_info=initial, return_all=True,
                                        return_quat=False)["posed_joints"]
    roundtrip_error = float((roundtrip - motion30["posed_joints"]).norm(dim=-1).max())
    if roundtrip_error > 0.002:
        raise ValueError(f"Native feature/FK roundtrip mismatch: {roundtrip_error:.6f} m")
    canonical = representation.inverse(global_features, is_normalized=False,
                                       return_all=True, return_quat=False)
    canonical_root = canonical["posed_joints"][:, representation.skeleton.root_idx].cpu().numpy().astype(np.float32)
    canonical_joints = canonical["posed_joints"].cpu().numpy().astype(np.float32)
    canonical_rotations = canonical["local_joint_rots"].cpu().numpy().astype(np.float32)
    skinner30 = load_official_skinner(representation.skeleton)
    canonical_soles, canonical_parity = _skin_canonical_regions(
        canonical_rotations, canonical_joints, skinner30, side_indices,
        soles, initial)
    for patch in support["patches"]:
        polygon = np.asarray(patch["polygon_xz_m"], dtype=np.float32)
        points = np.column_stack((polygon[:, 0], np.zeros(len(polygon), dtype=np.float32), polygon[:, 1]))
        patch["polygon_xz_m"] = _canonicalize_points(points, initial)[:, [0, 2]].tolist()
    local_features = representation.dual_rep.global_to_local(global_features, False, False)
    dual_features = torch.cat([global_features[:, representation.indices["root"]],
                               local_features[:, :4], global_features[:, 5:]], dim=-1)
    source_digest = source_hash or sha256_file(path)
    stem = f"stair_{direction}_A{actor}_{source_digest[:16]}_30hz"
    shard_rel = f"shards/{stem}.npz"
    support_path = f"reports/{stem}.json"
    archive = {
        "global_features": global_features.cpu().numpy().astype(np.float32),
        "dual_features": dual_features.cpu().numpy().astype(np.float32),
        "local_rot_mats": canonical_rotations,
        "root_positions": canonical_root,
        "posed_joints": canonical_joints,
        "local_rot_mats_canonical": canonical_rotations,
        "root_positions_canonical": canonical_root,
        "posed_joints_source_world": motion30["posed_joints"].cpu().numpy().astype(np.float32),
        "local_rot_mats_source30": rotations30,
        "root_positions_source30": root30,
        "local_rot_mats_native": native_rot,
        "root_positions_native": native_root,
        "sole_vertices_left": canonical_soles["left"], "sole_vertices_right": canonical_soles["right"],
        "sole_vertices_left_source_world": source_world_soles["left"],
        "sole_vertices_right_source_world": source_world_soles["right"],
        "sole_vertex_indices_left": side_indices["left"],
        "sole_vertex_indices_right": side_indices["right"],
        "stair_contact_labels": contacts, "stair_contact_label_mask": contact_mask,
        "stair_contact_confidence": contact_confidence,
        "patch_id_left": weak_arrays["patch_id_left"],
        "patch_id_right": weak_arrays["patch_id_right"],
        "support_gap_m_left": weak_arrays["support_gap_m_left"],
        "support_gap_m_right": weak_arrays["support_gap_m_right"],
        "vertex_measured_left": weak_arrays["vertex_measured_left"],
        "vertex_measured_right": weak_arrays["vertex_measured_right"],
        "support_confidence_left": weak_arrays["support_confidence_left"],
        "support_confidence_right": weak_arrays["support_confidence_right"],
        "fps": np.asarray(FPS, dtype=np.float32),
        "feature_fk_roundtrip_error_m": np.asarray(roundtrip_error, dtype=np.float32),
    }
    output_path = output / shard_rel
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **archive)
    support.update({
        "filename": path.stem, "actor": actor, "direction": direction,
        "source_path": str(path.resolve()), "source_sha256": source_digest,
        "source_native_root_sha256": native_root_hash,
        "source_native_local_rotations_sha256": native_rot_hash,
        "source_fps": float(source_fps), "target_fps": FPS,
        "patches": support["patches"],
        "actor_root_vertical_displacement_m": float(root30[-1, 1] - root30[0, 1]),
        "directional_motion": "up" if root30[-1, 1] > root30[0, 1] else "down",
        "native_converted_root_positions_unchanged": True,
        "target30_constant_ground_translation_m": -floor_shift,
        "target30_root_sha256_before_ground_translation": sha256_array(root30 + np.array([0, floor_shift, 0], dtype=np.float32)),
        "official_soma77_geometry_error_m": geometry_error,
        "official_selected_lbs_parity_max_error_m": parity,
        "canonical_soma30_source_soma77_sole_parity_max_error_m": canonical_parity,
        "body_fk_30_vs_77_max_error_m": fk_error,
        "native_feature_fk_roundtrip_max_error_m": roundtrip_error,
        "geometry_source": "official SOMA77 skin_standard.npz full-mesh LBS; selected 196 vertices; 98 per side",
        "sole_vertex_indices": {side: ids.tolist() for side, ids in side_indices.items()},
        "source_patch_frame_ids": {side: np.flatnonzero(weak_arrays[f"patch_id_{side}"] >= 0).tolist()
                                    for side in ("left", "right")},
        "contact_labels": "positive-only stationary material heel/toe evidence on accepted weak patches; unknown is retained in the evaluation label mask",
        "native_foot_feature_encoding": "0 for unknown channels; positive weak evidence uses 1; this heuristic is not contact/load truth",
        "label_mask": "stair_contact_label_mask; false means unknown, never inferred airborne; evaluation metadata only",
        "native_loader_note": "NativeMBMotionDataset/collate currently omit label masks. Unknown feature channels are zero-coded and therefore participate as heuristic zero targets in the unchanged reconstruction loss; no masking is applied.",
        "report_path": support_path,
        "support_report": support_path,
        "source_world_patches": source_world_patches,
    })
    report_path = output / support_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(support, indent=2, allow_nan=False), encoding="utf-8")
    row = {
        "filename": stem, "recording_group": f"stairs_climbing_{direction}_A{actor}",
        "behavior": f"stairs_climbing_{direction}",
        "split": "train" if actor == 300 else "validation", "actor": f"A{actor}",
        "source_bvh": str(path.resolve()), "source_sha256": source_digest,
        "shard": shard_rel, "frames_30hz": len(global_features),
        "duration_seconds": (len(global_features) - 1) / FPS,
        "source_fps": float(source_fps), "report_path": support_path,
        "support_report": support_path,
        "source_patch_frame_ids_path": support_path,
        "feature_fk_roundtrip_error_m": roundtrip_error,
        "support_pair_status": "weak_pair_candidate",
        "actor_root_vertical_displacement_m": float(root30[-1, 1] - root30[0, 1]),
        "directional_motion": "up" if root30[-1, 1] > root30[0, 1] else "down",
    }
    return row


def prepare(source, output, reference_dataset, flat_source=None, *, limit=None,
            actors=(300, 301), directions=("up", "down")):
    source, output = Path(source).resolve(), Path(output).resolve()
    reference = Path(flat_source or reference_dataset).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if output == reference or reference in output.parents:
        raise ValueError("Output must be separate from and outside the flat reference dataset")
    if output == source or source in output.parents:
        raise ValueError("Output must not be created inside the raw BVH source tree")
    if not (reference / "accepted.json").is_file():
        raise FileNotFoundError(reference / "accepted.json")
    output.mkdir(parents=True)
    (output / "shards").mkdir()
    flat_rows = inherited_flat_rows(reference, output)
    flat_manifest_hash = sha256_file(reference / "accepted.json")
    baseline_stats = Path(reference_dataset).resolve() / "stats"
    (output / "stats").mkdir()
    for name in ("mean.npy", "std.npy"):
        shutil.copy2(baseline_stats / name, output / "stats" / name)
    frozen_stats = {name: sha256_file(output / "stats" / name) for name in ("mean.npy", "std.npy")}

    full_skeleton = build_skeleton(77)
    representation = build_native_representation()
    surfaces, _, _, _ = official_sole_surfaces(representation.skeleton)
    skinner = load_official_skinner(full_skeleton)
    paths = []
    for path in sorted(source.rglob("*.bvh")):
        actor_direction = stair_actor(path)
        if actor_direction and actor_direction[0] in actors and actor_direction[1] in directions:
            paths.append((path, *actor_direction))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        paths = paths[:limit]
    if not paths:
        raise ValueError("No eligible stairs_climbing up/down clips for actors A300/A301")
    stairs, rejected = [], []
    for path, actor, direction in paths:
        try:
            stairs.append(convert_stair(path, actor, direction, full_skeleton,
                                        representation, skinner, surfaces, output))
        except (ValueError, RuntimeError, IndexError, KeyError) as error:
            rejected.append({"source_bvh": str(path.resolve()), "actor": f"A{actor}",
                             "direction": direction, "reason": str(error)})
    if limit is None and 300 in actors and not any(row["split"] == "train" for row in stairs):
        raise RuntimeError("No accepted A300 training stair clips")
    if limit is None and 301 in actors and not any(row["split"] == "validation" for row in stairs):
        raise RuntimeError("No accepted A301 validation stair clips")
    if not stairs:
        raise RuntimeError("No stair clips passed conversion and geometry gates")
    accepted = flat_rows + stairs
    accepted_path = output / "accepted.json"
    accepted_path.write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    representation = build_native_representation(output / "stats")
    del representation
    manifest_hash = sha256_file(accepted_path)
    artifact_hashes = {
        "accepted.json": manifest_hash,
        "stats/mean.npy": frozen_stats["mean.npy"],
        "stats/std.npy": frozen_stats["std.npy"],
    }
    artifact_hashes.update({row["shard"]: sha256_file(output / row["shard"]) for row in accepted})
    artifact_hashes.update({row["report_path"]: sha256_file(output / row["report_path"])
                            for row in stairs})
    manifest = {
        "contract": {
            "version": CONTRACT_VERSION, "representation": REPRESENTATION_NAME,
            "source_manifest_sha256": flat_manifest_hash,
            "source_shard_fps": "native BVH converted and resampled to 30 Hz",
            "target_fps": FPS,
            "resampling": "bones_seed.resample_rotations_root; quaternion SLERP and declared 4th-order zero-phase low-pass",
            "features": "official MotionBricks representation recomputed at target FPS from local rotations and root translation",
            "ground": "single constant translation from lowest accepted weak stair level; no per-frame fitting",
            "splits": "flat rows inherited exactly; stairs split by actor A300 train and A301 validation",
            "statistics": "byte-copied frozen reference normalization; no stair contribution",
            "statistics_folder": "stats", "custom_supervision": "sidecar only; native trainer/loss unchanged",
            "geometry": "official SOMA77 skin_standard.npz LBS; weak static horizontal patches",
        },
        "source_dataset": str(reference), "source_manifest": "accepted.json",
        "source_manifest_sha256": flat_manifest_hash, "accepted_json_sha256": manifest_hash,
        "stats_artifacts_sha256": frozen_stats, "artifact_sha256": artifact_hashes,
        "source_rows": len(flat_rows), "converted_rows": len(accepted),
        "split_counts": {split: sum(row["split"] == split for row in accepted)
                         for split in ("train", "validation", "test")},
        "frozen_normalization": {"copied_byte_for_byte": True, "source_dataset": str(Path(reference_dataset).resolve()),
                                 "mean_sha256": frozen_stats["mean.npy"], "std_sha256": frozen_stats["std.npy"]},
        "stair_source_inventory": {"eligible": len(paths), "accepted": len(stairs), "rejected": rejected,
                                   "test_stairs": "none; no test rows invented"},
        "limitations": ["Stair weak contacts are metadata only; native loader/loss do not consume their masks.",
                        "Unknown contact frames are masked, not labeled airborne.",
                        "Support surfaces are inferred weak horizontal patches, not physical load measurements."],
    }
    (output / "native_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local BONES BVH source tree")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--reference-dataset", type=Path, required=True)
    parser.add_argument("--flat-source", type=Path, help="Flat native dataset to inherit; defaults to reference dataset")
    parser.add_argument("--limit", type=int, help="Process only the first N eligible BVHs (smoke runs)")
    parser.add_argument("--actors", type=int, nargs="+", choices=(300, 301), default=(300, 301))
    parser.add_argument("--directions", nargs="+", choices=("up", "down"), default=("up", "down"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(prepare(args.source, args.output, args.reference_dataset, args.flat_source,
                             limit=args.limit, actors=tuple(args.actors),
                             directions=tuple(args.directions)), indent=2))


if __name__ == "__main__":
    main()
