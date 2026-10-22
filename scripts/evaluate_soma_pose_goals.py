"""Fixed validation and raw control tests for a native sparse-goal trial."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256, require_same_snapshot
from motionbricks.data.bones_rollout import SomaLocomotionBackend
from motionbricks.data.bones_seed import official_sole_surfaces, validation_window_records
from motionbricks.data.sole_region_contact import sole_region_indices
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner, skinned_vertices
from scripts.diagnose_soma_continuity import command, motion_metrics, stats
from scripts.evaluate_sole_region_ab import source_labels, motion_metrics as contact_metrics
from scripts.portable_bones_control import load_bundle


@torch.no_grad()
def compare_contact(source, prediction, rep, skinner, indices):
    move_skinner(skinner, "cpu")
    source_pose = rep.inverse(source, False, return_all=False)
    pred_pose = rep.inverse(prediction, False, return_all=False)
    source_rot = source_pose["local_joint_rots"][0].cpu()
    # Official skin() takes all posed joints, NOT a [T,3] root array.
    source_joints = source_pose["posed_joints"][0].cpu()
    source_root = source_joints[:, 0]
    labels, _ = source_labels(skinned_vertices(skinner, source_rot, source_joints), indices, rep.fps)
    return {
        "source_contact": contact_metrics(source_rot, source_root, skinner, indices, labels, rep.fps),
        "predicted_contact": contact_metrics(pred_pose["local_joint_rots"][0].cpu(),
            pred_pose["posed_joints"][0, :, 0].cpu(), skinner, indices, labels, rep.fps)}


@torch.no_grad()
def teacher_prediction(raw, inference, library):
    """Source root is privileged here; source future BODY is never a condition."""
    rep, pose = inference.motion_rep, inference._pose_model
    frames = 64
    local = rep.dual_rep.global_to_local(raw[:, :frames + 1], False, True,
        lengths=torch.tensor([frames + 1], device=raw.device))[:, :frames]
    goal_local, selected_ids = None, None
    if library is not None:
        velocity = (raw[:, frames - 1, [0, 2]] - raw[:, frames - 5, [0, 2]]) * rep.fps / 4
        heading = torch.atan2(raw[:, frames - 1, 4], raw[:, frames - 1, 3])
        target, _, selected_ids = library.select(velocity, heading, raw[:, frames - 4:frames, :5])
        goal_local = rep.dual_rep.global_to_local(target, False, True,
                                                 lengths=torch.tensor([4], device=raw.device))
    _, decoded, _ = pose.predict_local(local, rep.normalize(raw[:, :frames]), goal_local=goal_local)
    result = rep.dual_rep.local_to_global(decoded, True, False,
                                         lengths=torch.tensor([frames], device=raw.device))
    result[..., [0, 2]] += raw[:, :1, [0, 2]]
    yaw = torch.atan2(result[..., 4], result[..., 3]) + torch.atan2(raw[:, :1, 4], raw[:, :1, 3])
    result[..., 3], result[..., 4] = yaw.cos(), yaw.sin()
    return result, selected_ids


@torch.no_grad()
def raw_rollout(inference, initial, library, delay, frames=600):
    backend = SomaLocomotionBackend(inference, initial, argmax=True)
    backend.goal_library = library
    active = backend.predict_plan([[0., 0.]], [0.])["planned_motion"]
    chunks, boundaries, timings, ids = [], [], [], []
    cursor, published = 0, 0
    while published < frames:
        velocity, heading = command(published)
        result = backend.predict_plan([velocity], [heading])
        timings.append(result["generation_seconds"] * 1000)
        ids.append(None if result["reference_pose_ids"] is None else result["reference_pose_ids"].item())
        new = result["planned_motion"]
        if delay:
            preceding = active[:, cursor:cursor + delay]
            if preceding.shape[1] != delay:
                raise ValueError("Delayed buffer exhausted")
            chunks.append(preceding)
            published += delay
        chosen = new[:, delay:delay + 4]
        boundaries.append(published)
        chunks.append(chosen)
        published += 4
        backend.history = chosen.clone()
        active, cursor = new, delay + 4
    motion = torch.cat(chunks, 1)[:, :frames]
    commands = motion.new_tensor([command(frame)[0] for frame in range(frames - 1)])
    actual = motion[0, 1:, [0, 2]] - motion[0, :-1, [0, 2]]
    metrics = motion_metrics(motion, inference.motion_rep, boundaries)
    metrics["root_command_velocity_error_mps"] = stats((actual * inference.motion_rep.fps - commands).norm(dim=-1))
    metrics["model_generation_ms"] = stats(timings)
    metrics["reference_pose_ids"] = ids
    metrics["scope"] = "raw model output; delay=12 emulates stale-prefix skipping, not a wall-clock benchmark"
    return motion, boundaries, metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--disable-goals", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh evaluation directory")
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    inference, bundle = load_bundle(args.bundle, "cuda:0")
    rep = inference.motion_rep
    require_same_snapshot(bundle["contracts"]["pose"]["dataset_snapshot"], dataset_snapshot(args.dataset))
    library = None if args.disable_goals else getattr(inference, "goal_library", None)
    records = validation_window_records([r for r in json.loads((args.dataset / "accepted.json").read_text())
                                          if r["split"] == "validation"], 128, 3)
    skinner = load_official_skinner(rep.skeleton)
    surfaces, _, _, _ = official_sole_surfaces(rep.skeleton)
    indices = {side: sole_region_indices(surfaces, side) for side in ("Left", "Right")}
    args.output.mkdir(parents=True)
    report = {"bundle_sha256": file_sha256(args.bundle), "test_used": False, "post_processing": None,
        "evaluation_sha256": file_sha256(Path(__file__)),
        "body_goals": "train_library_one_endpoint" if library else "none", "root_stage": "released_MB",
        "validation_scope": "teacher history/source root; endpoint root supplies diagnostic command, never future body",
        "contact_scope": "official skin, fixed ground y=0, shared SOURCE-derived candidate masks; skip4",
        "quality_approved": False, "ue_realtime_assessed": False, "windows": [], "rollouts": {}}
    for index, record in enumerate(records):
        start = record["window_start_frame"]
        path = args.dataset / record["shard"]
        with np.load(path, allow_pickle=False) as data:
            raw = torch.from_numpy(data["global_features"][start:start + 129].copy())[None].cuda()
        chunks, selected = [], []
        for offset in range(0, 64, 4):
            prediction, ids = teacher_prediction(raw[:, offset:offset + 65], inference, library)
            chunks.append(prediction[:, 4:8])
            selected.append(None if ids is None else ids.item())
        motion = torch.cat(chunks, 1)
        source = raw[:, 4:68]
        metrics = motion_metrics(motion, rep, range(4, 64, 4), source)
        metrics.update(compare_contact(source, motion, rep, skinner, indices))
        report["windows"].append({"filename": record["filename"], "start": start,
            "input_sha256": file_sha256(path), "reference_pose_ids": selected, **metrics})
        if index % 10 == 0:
            print(json.dumps({"validation_windows": index + 1, "body_goals": report["body_goals"]}), flush=True)
    for delay in (0, 12):
        torch.manual_seed(2333)
        motion, boundaries, measured = raw_rollout(inference, bundle["initial_history"].cuda(), library, delay)
        report["rollouts"][str(delay)] = measured
        np.savez_compressed(args.output / f"delay_{delay}.npz", global_features=motion[0].cpu().numpy(),
                            boundaries=np.asarray(boundaries), fps=np.asarray(rep.fps))
    rows = report["windows"]
    measured_slip = [r["predicted_contact"]["measured_slip_mps"]["mean"]
                    for r in rows if r["predicted_contact"]["measured_slip_mps"] is not None]
    report["aggregate"] = {
        "windows": len(rows),
        "mean_window_mpjpe_mm": float(np.mean([r["root_relative_mpjpe_mm"]["mean"] for r in rows])),
        "mean_window_boundary_angular_p95_deg": float(np.mean([r["boundary_angular_step_deg"]["p95"] for r in rows])),
        "worst_joint_step_deg": max(r["angular_step_deg"]["max"] for r in rows),
        "mean_window_penetration_m": float(np.mean([r["predicted_contact"]["region_penetration_m"]["mean"] for r in rows])),
        "mean_window_measured_slip_mps": float(np.mean(measured_slip)) if measured_slip else None}
    # Persist metrics before any optional media tool, so rendering cannot lose evidence.
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output), "aggregate": report["aggregate"],
                      "rollouts": {k: r["boundary_joint_step_mm"]["p95"] for k, r in report["rollouts"].items()}}), flush=True)


if __name__ == "__main__":
    main()
