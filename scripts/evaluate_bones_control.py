"""Score goal-driven SOMA motion separately from the source-root diagnostic upper bound."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256, require_same_snapshot
from motionbricks.data.bones_evaluation import rotation_error_degrees
from motionbricks.data.bones_rollout import SomaLocomotionBackend, control_inputs, load_control_models
from motionbricks.data.bones_seed import build_representation, official_sole_surfaces, validation_window_records
from motionbricks.data.sole_region_contact import material_point_slip, sole_region_indices
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner, skinned_vertices
from scripts.evaluate_sole_region_ab import motion_metrics, source_labels


COMMANDS = ((0, 4, "idle", (0, 0), 0), (4, 12, "walk", (0, 1.2), 0),
            (12, 20, "run", (0, 2.5), 0), (20, 24, "stop", (0, 0), 0),
            (24, 32, "turn90", (1.2, 0), np.pi / 2),
            (32, 40, "turn180", (-1.2, 0), np.pi * 1.5),
            (40, 46, "backward", (0.8, 0), np.pi * 1.5),
            (46, 52, "lateral", (0, 0.8), np.pi * 1.5),
            (52, 56, "stop", (0, 0), np.pi * 1.5),
            (56, 60, "start", (-1.2, 0), np.pi * 1.5))


def command_at(seconds):
    for start, end, name, velocity, heading in COMMANDS:
        if start <= seconds < end:
            return name, velocity, heading
    raise ValueError("Commands are defined for [0, 60) seconds")


def distribution(values):
    values = torch.as_tensor(values).flatten().float().cpu()
    if not values.numel():
        return {"count": 0, "mean": None, "p95": None, "peak": None}
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite scored motion")
    return {"count": values.numel(), "mean": float(values.mean()),
            "p95": float(torch.quantile(values, 0.95)), "peak": float(values.max())}


def sole_vertices(pose, skinner, indices, device):
    combined = np.concatenate([indices["Left"], indices["Right"]])
    selection = torch.as_tensor(combined, device=device)
    result = []
    for start in range(0, len(pose["local_joint_rots"]), 48):
        vertices = skinned_vertices(skinner, pose["local_joint_rots"][start:start + 48].to(device),
                                    pose["posed_joints"][start:start + 48].to(device))
        result.append(vertices.index_select(1, selection).cpu())
    return torch.cat(result)


def free_metrics(motion, velocity, heading, initial_root, representation, vertices, indices):
    root = motion[:, :3]
    actual_velocity = torch.diff(torch.cat((initial_root[None], root)), dim=0)[:, [0, 2]] * 60
    desired_position = initial_root[[0, 2]] + velocity.cumsum(0) / 60
    error = (actual_velocity - velocity).norm(dim=-1)
    yaw = torch.atan2(motion[:, 4], motion[:, 3])
    heading_error = torch.atan2((yaw - heading).sin(), (yaw - heading).cos()).abs() * 180 / np.pi
    joints = representation.inverse(motion[None].to(next(representation.buffers()).device), False,
                                    return_all=False)["posed_joints"][0].cpu()
    displacement = joints[1:] - joints[:-1]
    boundary = torch.arange(1, len(motion)) % 4 == 0
    soles = {}
    offset = 0
    gaps = []
    for side in ("Left", "Right"):
        count = len(indices[side])
        points = vertices[:, offset:offset + count]
        offset += count
        heights = points[..., 1]
        gap = heights.amin(-1)
        gaps.append(gap)
        measured = material_point_slip(heights, points[..., [0, 2]], 60)
        mask = measured["vertex_measured"]
        speeds = (points[1:, :, [0, 2]] - points[:-1, :, [0, 2]]).norm(dim=-1) * 60
        drift = []
        # Stable-geometry intervals are a diagnostic, never labels fed to the generator.
        for vertex in range(count):
            flags = mask[:, vertex].numpy()
            transitions = np.diff(np.r_[False, flags, False].astype(np.int8))
            for start, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
                drift.append((points[end, vertex, [0, 2]] - points[start, vertex, [0, 2]]).norm())
        soles[side] = {"near_ground_material_speed_mps": distribution(speeds[mask]),
                       "measured_vertex_interval_fraction": float(mask.float().mean()),
                       "near_ground_frame_fraction": float((gap <= 0.01).float().mean()),
                       "undecided_band_frame_fraction": float(((gap > 0.01) & (gap <= 0.03)).float().mean()),
                       "clearance_frame_fraction": float((gap > 0.03).float().mean()),
                       "contact_segment_net_drift_m": distribution(torch.stack(drift) if drift else [])}
    gaps = torch.stack(gaps, -1)
    return {"velocity_error_mps": distribution(error), "actual_speed_mps": distribution(actual_velocity.norm(dim=-1)),
            "heading_error_deg": distribution(heading_error),
            "displacement_tracking_error_m": distribution((root[:, [0, 2]] - desired_position).norm(dim=-1)),
            "sole_penetration_m": distribution(torch.relu(-gaps)),
            "sole_clearance_m": distribution(torch.relu(gaps)),
            "both_feet_above_3cm_fraction": float((gaps.amin(-1) > 0.03).float().mean()),
            "root_height_m": distribution(root[:, 1]), "feet": soles,
            "boundary_joint_step_m": distribution(displacement.norm(dim=-1)[boundary]),
            "interior_joint_step_m": distribution(displacement.norm(dim=-1)[~boundary]),
            "joint_acceleration_mps2": distribution(torch.diff(joints, n=2, dim=0).norm(dim=-1) * 3600),
            "quality_approved": False,
            "scoring": "free geometry diagnostics, no source masks or time-aligned MPJPE"}


@torch.inference_mode()
def paired_windows(dataset, records, inference, representation, skinner, indices):
    results = []
    device = inference.device
    for record in validation_window_records([r for r in records if r["split"] == "validation"], 128, 3):
        start = record["window_start_frame"]
        with np.load(dataset / record["shard"], allow_pickle=False) as data:
            raw = torch.from_numpy(data["global_features"][start:start + 129].copy())[None].to(device)
        local = representation.dual_rep.global_to_local(raw, False, True, lengths=torch.tensor([129], device=device))[:, :128]
        local_raw = representation.dual_rep.local_motion_rep.unnormalize(local)
        velocity = local_raw[:, 124:128, 1:3].mean(1)
        prediction, _ = inference.predict(*control_inputs(raw[:, :4], raw[:, 124:128, :5], velocity, representation),
                                           32, config={"pose_token_sampling_use_argmax": True})
        _, upper, _ = inference._pose_model.predict_local(local, representation.normalize(raw[:, :128]))
        upper = representation.dual_rep.local_to_global(upper, True, False, lengths=torch.tensor([128], device=device))
        upper[..., [0, 2]] += raw[:, :1, [0, 2]]
        first_yaw = torch.atan2(raw[:, 0, 4], raw[:, 0, 3])
        upper_yaw = torch.atan2(upper[..., 4], upper[..., 3]) + first_yaw[:, None]
        upper[..., 3], upper[..., 4] = upper_yaw.cos(), upper_yaw.sin()
        poses = {"source": representation.inverse(raw[:, :128], False, return_all=True),
                 "control": representation.inverse(prediction[:, :128], False, return_all=True),
                 "source_root_diagnostic": representation.inverse(upper, False, return_all=True)}
        source = {key: value[0] for key, value in poses["source"].items() if torch.is_tensor(value)}
        source_mesh = skinned_vertices(skinner, source["local_joint_rots"], source["posed_joints"]).cpu()
        labels, _ = source_labels(source_mesh, indices, 60)
        cpu_skinner = move_skinner(load_official_skinner(representation.skeleton), "cpu")
        entry = {"filename": record["filename"], "behavior": record.get("behavior"), "start": start}
        for name in ("control", "source_root_diagnostic"):
            pose = poses[name]
            geometry = motion_metrics(pose["local_joint_rots"][0].cpu(), pose["posed_joints"][0, :, 0].cpu(),
                                       cpu_skinner, indices, labels, 60)
            geometry["mpjpe_m"] = float((pose["posed_joints"][:, 4:] - poses["source"]["posed_joints"][:, 4:]).norm(dim=-1).mean())
            geometry["local_rotation_error_deg"] = float(rotation_error_degrees(
                pose["local_joint_rots"][:, 4:], poses["source"]["local_joint_rots"][:, 4:]).mean())
            geometry["root_xz_error_m"] = float((pose["posed_joints"][:, 4:, 0, [0, 2]] -
                                                  poses["source"]["posed_joints"][:, 4:, 0, [0, 2]]).norm(dim=-1).mean())
            geometry["joint_acceleration_error_mps2"] = float((torch.diff(pose["posed_joints"][:, 4:], n=2, dim=1) -
                                                                torch.diff(poses["source"]["posed_joints"][:, 4:], n=2, dim=1)).norm(dim=-1).mean() * 3600)
            entry[name] = geometry
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset", "root-run", "pose-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--seeds", default="2333,2334,2335")
    parser.add_argument("--skip-paired", action="store_true")
    parser.add_argument("--argmax", action="store_true", help="Separate greedy-sampling diagnostic, not a new trained arm")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.seconds <= 60:
        raise ValueError("Use a fresh output directory and 1-60 second command duration")
    torch.set_num_threads(4)
    snapshot = dataset_snapshot(args.dataset)
    rep = build_representation(args.dataset / snapshot["stats_folder"])
    inference, contracts = load_control_models(args.root_run, args.pose_run, rep, args.device)
    require_same_snapshot(contracts["pose"]["dataset_snapshot"], snapshot)
    records = json.loads((args.dataset / "accepted.json").read_text())
    initial_record = next(r for r in records if r["split"] == "validation" and r["behavior"] == "idle")
    with np.load(args.dataset / initial_record["shard"], allow_pickle=False) as data:
        initial = torch.from_numpy(data["global_features"][:4].copy())[None].to(args.device)
    initial = rep.change_first_heading(initial, torch.zeros(1, device=args.device), False, False)
    skinner = move_skinner(load_official_skinner(rep.skeleton), args.device)
    surfaces, _, _, _ = official_sole_surfaces(rep.skeleton)
    indices = {side: sole_region_indices(surfaces, side) for side in ("Left", "Right")}
    args.output.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    sources = ("scripts/evaluate_bones_control.py", "scripts/train_bones_control.py",
               "motionbricks/data/bones_rollout.py", "motionbricks/data/bones_control.py",
               "motionbricks/data/bones_models.py", "motionbricks/data/bones_seed.py",
               "motionbricks/motion_backbone/inference/motion_inference.py",
               "motionbricks/motion_backbone/models/pose_model.py", "motionbricks/motion_backbone/models/root_model.py")
    source_hashes = {}
    for name in sources:
        destination = args.output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, destination)
        source_hashes[name] = file_sha256(destination)
    report = {"schema": "motionbricks.soma_control_review.v1", "dataset_sha256": snapshot["sha256"],
              "runtime_source_sha256": source_hashes,
              "root_checkpoint_sha256": file_sha256(args.root_run / "checkpoints/final.ckpt"),
              "pose_checkpoint_sha256": file_sha256(args.pose_run / "checkpoints/final.ckpt"),
              "initial_history_source": initial_record["filename"], "initial_history_frames": 4,
              "commands": COMMANDS, "post_processing": None, "horizon_frames": 64, "commit_frames": 4,
              "intent_goal_acceleration_mps2": 3.0,
              "intent_goal_semantics": "integrate observed velocity towards requested velocity; endpoints only, not a model-output clamp",
              "source_fps": 60, "sampling": "one-pass argmax" if args.argmax else "one-pass official Gumbel; fixed seed per independent sequence",
              "quality_approved": False, "test_split_used": False, "rollouts": []}
    if not args.skip_paired:
        report["paired_validation_windows"] = paired_windows(args.dataset, records, inference, rep, skinner, indices)
        (args.output / "paired_metrics.json").write_text(json.dumps(report["paired_validation_windows"], indent=2, allow_nan=False))
    for seed in [int(value) for value in args.seeds.split(",")]:
        torch.manual_seed(seed)
        backend = SomaLocomotionBackend(inference, initial, argmax=args.argmax)
        chunks, histories, goals, velocities, headings, latency = [], [], [], [], [], []
        for frame in range(0, args.seconds * 60, 4):
            name, velocity, heading = command_at(frame / 60)
            histories.append(backend.history.cpu().clone())
            result = backend.step([velocity], [heading])
            chunks.append(result["motion"].cpu())
            goals.append(result["goal_root"].cpu())
            velocities.extend([velocity] * 4)
            headings.extend([heading] * 4)
            latency.append(result["generation_seconds"])
        motion = torch.cat(chunks, 1)[0]
        posed = rep.inverse(motion[None].to(args.device), False, return_all=True)
        pose = {key: value[0].cpu() for key, value in posed.items() if torch.is_tensor(value)}
        vertices = sole_vertices(pose, skinner, indices, args.device)
        metrics = free_metrics(motion, torch.tensor(velocities), torch.tensor(headings), initial[0, -1, :3].cpu(),
                               rep, vertices, indices)
        metrics["generation_seconds_per_4_committed_frames"] = distribution(latency)
        metrics["seed"] = seed
        metrics["frames"] = len(motion)
        np.savez_compressed(args.output / f"rollout_{seed}.npz", global_features=motion.numpy(),
                            local_rot_mats=pose["local_joint_rots"].numpy(), root_positions=pose["posed_joints"][:, 0].numpy(),
                            joints=pose["posed_joints"].numpy(), executed_history=torch.cat(histories).numpy(),
                            goal_root=torch.cat(goals).numpy(), command_velocity_xz=np.asarray(velocities),
                            command_heading_rad=np.asarray(headings), generation_seconds=np.asarray(latency),
                            sole_vertices=vertices.numpy(), fps=np.asarray(60))
        report["rollouts"].append(metrics)
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({"seed": seed, "frames": len(motion), "velocity_error_mps": metrics["velocity_error_mps"],
                          "penetration_m": metrics["sole_penetration_m"]}), flush=True)


if __name__ == "__main__":
    main()
