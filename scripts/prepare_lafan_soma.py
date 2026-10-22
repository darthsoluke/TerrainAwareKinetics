"""Decode and retarget LAFAN-Resolved, publishing training eligibility only after checks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.spatial.transform import Rotation
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.bones_seed import build_representation, calibration_crop, official_sole_surfaces
from motionbricks.data.lafan_resolved_motion import decode_geno_bind_bvh, decode_lafan_resolved_bvh
from motionbricks.data.lafan_soma_retarget import GenoSomaRetargeter, JOINT_MAP, Y_TO_Z
from motionbricks.data.soma_skin_reference import load_official_skinner, surface_heights
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from kimodo.exports.motion_io import complete_motion_dict


# These are declared engineering acceptance bounds, not measured force labels.
# They are saved unchanged before decoding any evaluation motion.
BOUNDS = {"reference_fk_max_m": 2e-6, "reference_pose_max_m": .001,
          "root_tracking_p95_m": .02, "root_tracking_max_m": .05,
          "endpoint_tracking_p95_m": .03, "endpoint_tracking_max_m": .08,
          "orientation_tracking_p95_deg": 10.,
          "added_angular_step_p99_deg": 5., "added_angular_step_max_deg": 15.,
          "stationary_anchor_added_speed_p95_mps": .05,
          "stationary_sole_gap_p95_m": .03, "penetration_p99_m": .03,
          "stationary_anchor_min_intervals": 30}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def angular_steps(rotations):
    delta = rotations[:-1].swapaxes(-1, -2) @ rotations[1:]
    return np.degrees(Rotation.from_matrix(delta.reshape(-1, 3, 3)).magnitude()).reshape(delta.shape[:2])


def assess_motion(source_positions, target_positions, desired_positions, target_rotations,
                  desired_rotations, sole_heights, source_indices, target_indices, scale):
    errors = np.linalg.norm(target_positions[:, target_indices] - desired_positions, axis=-1)
    feet = [index for index, name in enumerate(JOINT_MAP) if name.endswith(("Foot", "ToeBase"))]
    source_anchors = source_positions[:, np.asarray(source_indices)[feet]]
    source_speed = np.linalg.norm(np.diff(source_anchors, axis=0), axis=-1) * 60.
    stationary = minimum_filter1d(source_speed < .10, size=5, axis=0, mode="constant", cval=0)
    target_speed = np.linalg.norm(np.diff(target_positions[:, np.asarray(target_indices)[feet]], axis=0), axis=-1) * 60.
    added_speed = np.maximum(target_speed - scale * source_speed, 0.)
    stationary_frames = np.zeros_like(sole_heights, dtype=bool)
    stationary_frames[:-1] |= stationary
    stationary_frames[1:] |= stationary
    count = int(stationary.sum())
    if count < BOUNDS["stationary_anchor_min_intervals"]:
        raise ValueError("Insufficient stationary kinematic foot anchors; no contact claim can be assessed")
    # A single translation for the entire recording, never a per-frame fit.
    floor = float(np.median(sole_heights[stationary_frames]))
    gap = sole_heights - floor
    relative = target_rotations[:, target_indices].swapaxes(-1, -2) @ desired_rotations
    angles = np.degrees(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude())
    added_step = np.maximum(angular_steps(target_rotations[:, target_indices]) - angular_steps(desired_rotations), 0.)
    metrics = {"root_tracking_p95_m": float(np.quantile(errors[:, 0], .95)),
               "root_tracking_max_m": float(errors[:, 0].max()),
               "endpoint_tracking_p95_m": float(np.quantile(errors[:, feet], .95)),
               "endpoint_tracking_max_m": float(errors[:, feet].max()),
               "orientation_tracking_p95_deg": float(np.quantile(angles, .95)),
               "added_angular_step_p99_deg": float(np.quantile(added_step, .99)),
               "added_angular_step_max_deg": float(added_step.max()),
               "stationary_anchor_added_speed_p95_mps": float(np.quantile(added_speed[stationary], .95)),
               "stationary_sole_gap_p95_m": float(np.quantile(np.abs(gap[stationary_frames]), .95)),
               "penetration_p99_m": float(np.quantile(np.maximum(-gap, 0.), .99)),
               "stationary_anchor_intervals": count, "fixed_plane_source_y_m": floor,
               "source_stationary_speed_p95_mps": float(np.quantile(source_speed[stationary], .95)),
               "target_stationary_speed_p95_mps": float(np.quantile(target_speed[stationary], .95))}
    failures = [name for name, bound in BOUNDS.items() if name in metrics and metrics[name] > bound]
    return metrics, failures, floor


def convert_clip(task):
    torch.set_num_threads(1)
    row, args = task
    output = Path(args["output"])
    filename = row["filename"]
    source_path = Path(args["source"]) / filename
    if sha256(source_path) != row["sha256"]:
        raise ValueError(f"Staged source BVH identity changed: {filename}")
    bind = decode_geno_bind_bvh(args["bind"], distance_unit="cm")
    motion = decode_lafan_resolved_bvh(source_path, distance_unit="cm")
    original_count = len(motion["posed_joints"])
    source_view = SimpleNamespace(bone_index={name: index for index, name in enumerate(motion["source_skeleton"]["names"])})
    crop_start, crop_end = calibration_crop(torch.from_numpy(motion["posed_joints"]).float(), source_view)
    for key in ("local_rot_mats", "local_positions", "global_rot_mats", "posed_joints"):
        motion[key] = motion[key][crop_start:crop_end]
    skeleton = SOMASkeleton30()
    retargeter = GenoSomaRetargeter(bind, skeleton, args["gmr"], output / "calibration" / Path(filename).stem,
                                  endpoint_position_cost=args["endpoint_position_cost"])
    names = list(skeleton.bone_order_names)
    rng = np.random.default_rng(2333)
    random_local = Rotation.from_rotvec(rng.normal(0., .2, (30, 3))).as_matrix()
    fk_error = retargeter.validate_fk(random_local, np.array([.2, 1., -.4]))
    reference_global = np.asarray(retargeter.calibration["reference_global_rot_mats"])
    parents = skeleton.joint_parents.numpy()
    reference_local = reference_global.copy()
    reference_local[1:] = reference_global[parents[1:]].swapaxes(-1, -2) @ reference_global[1:]
    root_ref = retargeter.calibration["uniform_scale"] * bind["posed_joints"][0, 0]
    retargeter.engine.configuration.update(retargeter.qpos_from_motion(reference_local, root_ref))
    ref_local, ref_root = retargeter.solve(bind["posed_joints"][0], bind["global_rot_mats"][0])
    _, reference_joints, _ = skeleton.fk(torch.tensor(ref_local[None], dtype=torch.float32),
                                       torch.tensor(ref_root[None], dtype=torch.float32))
    reference_error = float(np.abs(reference_joints[0].numpy() -
                                   (np.asarray(retargeter.calibration["reference_positions_relative_root_m"]) + root_ref)).max())
    if reference_error > BOUNDS["reference_pose_max_m"]:
        raise ValueError(f"Reference pose mapping mismatch: {reference_error}")
    total = len(motion["posed_joints"])
    count = min(total, args["max_frames"]) if args["max_frames"] else total
    local, roots, desired_positions, desired_rotations = [], [], [], []
    source_indices = [motion["source_skeleton"]["names"].index(name) for name in JOINT_MAP.values()]
    target_indices = [names.index(name) for name in JOINT_MAP]
    start = time.monotonic()
    for frame in range(count):
        lr, root = retargeter.solve(motion["posed_joints"][frame], motion["global_rot_mats"][frame])
        if frame == 0:
            # Fully converge the first pose; subsequent frames use upstream warm-start IK.
            for _ in range(10):
                lr, root = retargeter.solve(motion["posed_joints"][frame], motion["global_rot_mats"][frame])
        local.append(lr)
        roots.append(root)
        desired_positions.append([retargeter.engine.scaled_human_data[name][0] @ Y_TO_Z for name in JOINT_MAP.values()])
        desired_rotations.append([Y_TO_Z.T @ Rotation.from_quat(
            retargeter.engine.scaled_human_data[name][1], scalar_first=True).as_matrix() @ Y_TO_Z
                                  for name in JOINT_MAP.values()])
        if frame % 1000 == 0:
            print(json.dumps({"clip": filename, "frame": frame, "total": count}), flush=True)
    local, roots = np.asarray(local, dtype=np.float32), np.asarray(roots, dtype=np.float32)
    rotations, positions, _ = skeleton.fk(torch.from_numpy(local), torch.from_numpy(roots))
    surfaces, _, _, _ = official_sole_surfaces(skeleton)
    skinner = load_official_skinner(skeleton)
    with torch.inference_mode():
        full_sample = skinner.skin(torch.from_numpy(local[:4]), positions[:4], rot_is_global=False)
        full_sample_heights = torch.stack(list(surface_heights(full_sample, surfaces).values()), -1)
    selected = np.unique(np.concatenate([surface["selected_indices"] for surface in surfaces]))
    # Run the exact official LBS on only the required material vertices.
    # Rig transforms and per-vertex weights are unchanged; faces are not used.
    for attribute in ("bind_vertices", "lbs_indices", "lbs_weights"):
        setattr(skinner, attribute, getattr(skinner, attribute)[selected])
    selected_lookup = {int(index): order for order, index in enumerate(selected)}
    selected_surfaces = [dict(surface, selected_indices=np.asarray(
        [selected_lookup[int(index)] for index in surface["selected_indices"]])) for surface in surfaces]
    with torch.inference_mode():
        subset_sample = skinner.skin(torch.from_numpy(local[:4]), positions[:4], rot_is_global=False)
        subset_heights = torch.stack(list(surface_heights(subset_sample, selected_surfaces).values()), -1)
    subset_error = float((full_sample_heights - subset_heights).abs().max())
    if subset_error > 1e-6:
        raise ValueError(f"Selected official skinning differs from the full mesh: {subset_error}")
    heights = []
    for first in range(0, count, 128):
        with torch.inference_mode():
            vertices = skinner.skin(torch.from_numpy(local[first:first + 128]),
                                    positions[first:first + 128], rot_is_global=False)
        channel_heights = surface_heights(vertices, selected_surfaces)
        heights.append(torch.stack(list(channel_heights.values()), -1).numpy())
    heights = np.concatenate(heights)
    metrics, failures, floor = assess_motion(
        motion["posed_joints"][:count], positions.numpy(), np.asarray(desired_positions),
        rotations.numpy(), np.asarray(desired_rotations), heights, source_indices, target_indices,
        retargeter.calibration["uniform_scale"])
    roots[:, 1] -= floor
    positions[:, :, 1] -= floor
    source_positions = motion["posed_joints"][:count].copy()
    # Source coordinates retained verbatim; renderer applies only the recorded scale.
    relative = "shards/" + Path(filename).stem + ".npz"
    motion_dict = complete_motion_dict(torch.from_numpy(local), torch.from_numpy(roots), skeleton, 60.)
    representation = build_representation()
    dual = representation.dual_rep({"posed_joints": motion_dict["posed_joints"],
        "global_joint_rots": motion_dict["global_rot_mats"], "local_joint_rots": motion_dict["local_rot_mats"],
        "translation": motion_dict["root_positions"]}, to_normalize=False)
    features = representation.dual_rep.get_feature_subset(dual, mode="global").numpy()
    np.savez_compressed(output / relative, local_rot_mats=local, root_positions=roots,
                        global_features=features, target_positions=positions.numpy(),
                        source_positions=source_positions.astype(np.float32),
                        source_global_rot_mats=motion["global_rot_mats"][:count].astype(np.float32),
                        source_names=np.asarray(motion["source_skeleton"]["names"]),
                        source_parents=np.asarray(motion["source_skeleton"]["parents"]),
                        target_fps=60., uniform_scale=retargeter.calibration["uniform_scale"],
                        sole_heights=heights - floor)
    subject = "subject" + Path(filename).stem.rsplit("subject", 1)[1]
    result = {"filename": Path(filename).stem, "shard": relative, "subject_id": subject,
              "split": "validation" if subject == "subject4" else "test" if subject == "subject5" else "train",
              "behavior": "run" if filename.startswith("run") else "walk",
              "recording_group": Path(filename).stem, "source_fps": motion["source_fps"],
              "frames_60hz": count, "original_frames": total, "fixed_plane_source_y_m": floor,
              "raw_source_frames": original_count, "source_calibration_crop": [crop_start, crop_end],
              "source_bvh_sha256": sha256(source_path), "shard_sha256": sha256(output / relative),
              "retarget_assessment": {"metrics": metrics, "failures": failures,
                                      "status": "passed" if not failures else "failed",
                                      "reference_fk_max_m": fk_error, "reference_pose_max_m": reference_error,
                                      "selected_skinning_max_error_m": subset_error},
              "calibration": retargeter.calibration, "elapsed_seconds": time.monotonic() - start}
    dump(output / "reports" / (Path(filename).stem + ".json"), result)
    print(json.dumps({"clip": filename, "assessment": result["retarget_assessment"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bind", type=Path, required=True)
    parser.add_argument("--gmr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0, help="Preflight only; never publishes training eligibility")
    parser.add_argument("--clips", nargs="*", help="Preflight subset; never publishes training eligibility")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--endpoint-position-cost", type=float, default=50.)
    parser.add_argument("--exclude-failed-clips", action="store_true",
                        help="Publish only clips that meet every bound; rejected clips and reasons remain archived")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Retarget output must be a new versioned directory")
    if args.max_frames < 0 or args.workers < 1:
        raise ValueError("Invalid frame/worker count")
    manifest_path = args.source / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["candidate_files"]
    if sha256(args.bind) != manifest["source"]["bind_bvh_sha256"]:
        raise ValueError("Geno reference BVH identity changed")
    if args.clips:
        rows = [row for row in rows if row["filename"] in args.clips]
        if len(rows) != len(args.clips):
            raise ValueError("Preflight clip names do not match source manifest")
    args.output.mkdir(parents=True)
    for name in ("shards", "reports"):
        (args.output / name).mkdir()
    config = {name: str(value.resolve()) if isinstance(value, Path) else value for name, value in vars(args).items()}
    dump(args.output / "preparation_contract.json", {"status": "preparing", "bounds": BOUNDS,
        "source_manifest_sha256": sha256(manifest_path), "args": config,
        "code_sha256": {str(path.name): sha256(path) for path in [Path(__file__),
                         Path(sys.modules[GenoSomaRetargeter.__module__].__file__),
                         Path(sys.modules[decode_lafan_resolved_bvh.__module__].__file__)]},
        "floor": "one fixed plane per recording from official target sole surfaces at stationary source anchors",
        "contact_evidence": "stationary source foot/toe kinematic anchors; not measured force or exact Geno shoe contact",
        "runtime_postprocessing": "none"})
    accepted = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(convert_clip, [(row, config) for row in rows]):
                accepted.append(result)
        failures = {row["filename"]: row["retarget_assessment"]["failures"] for row in accepted
                    if row["retarget_assessment"]["status"] != "passed"}
        all_results = accepted
        rejected = [row for row in all_results if row["retarget_assessment"]["status"] != "passed"]
        if args.exclude_failed_clips:
            accepted = [row for row in all_results if row["retarget_assessment"]["status"] == "passed"]
        dump(args.output / "conversion_results.json", all_results)
        dump(args.output / "accepted.json", accepted)
        dump(args.output / "rejected.json", rejected)
        full = not args.max_frames and not args.clips and len(all_results) == len(manifest["candidate_files"])
        eligible_splits = all({row["behavior"] for row in accepted if row["split"] == split} == {"walk", "run"}
                              for split in ("train", "validation", "test"))
        eligible = full and eligible_splits and (not failures or args.exclude_failed_clips)
        report = {"status": "passed" if eligible else "failed" if failures else "preflight_only",
            "training_eligible": eligible, "source_accepted_sha256": sha256(args.output / "accepted.json"),
            "accepted_shard_sha256": {row["shard"]: row["shard_sha256"] for row in accepted},
            "heldout_subjects": {"validation": ["subject4"], "test": ["subject5"]},
            "bounds": BOUNDS, "rejected_clip_failures": failures, "clips": len(accepted),
            "source_inventory_clips": len(all_results), "rejected_clips": len(rejected),
            "acceptance_scope": "passed clips only" if args.exclude_failed_clips else "complete candidate collection",
            "source_inventory_quality_status": "partial_rejection" if rejected else "all_passed",
            "accepted_contains_failed_clips": any(row["retarget_assessment"]["status"] != "passed" for row in accepted),
            "minutes_by_split": {split: sum(row["frames_60hz"] / 3600. for row in accepted if row["split"] == split)
                                 for split in ("train", "validation", "test")},
            "assessment_limitations": ["Geno mesh is unavailable; stationary foot/toe anchors are kinematic evidence, not true contact labels.",
                "One fixed recording floor estimated from official SOMA sole surfaces, not a calibrated capture plane.",
                "Jaw/eyes/finger endpoints without anatomical mapping stay neutral relative to their parent.",
                "Custom anatomically calibrated GMR adapter, not an official MB Geno-to-SOMA converter.",
                "Retarget quality acceptance is not evidence of trained model quality."]}
        dump(args.output / "retarget_review.json", report)
        print(json.dumps(report, indent=2), flush=True)
        if failures and not eligible:
            raise RuntimeError("Retarget automatic assessment failed; training is blocked")
    except BaseException as error:
        dump(args.output / "preparation_failure.json", {"status": "failed", "error": repr(error), "training_eligible": False})
        raise


if __name__ == "__main__":
    main()
