"""Bounded command rollout through the official Root -> Pose -> decoder pipeline.

This is a start-only stress probe, not an interactive locomotion acceptance test.
Only four seed frames are read from a held-out motion; all later body poses are
model-generated. Command endpoints are conditions, never output replacements.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from scripts.train_native_mb import select_gpu, sha256, write_json
from scripts.train_native_mb_control import load_control_checkpoint


COMMANDS = (
    ("idle", (0.0, 0.0), 0.0),
    ("forward", (0.0, 1.0), 0.0),
    ("turn_right", (1.0, 0.0), math.pi / 2),
    ("stop", (0.0, 0.0), math.pi / 2),
    ("backward", (-0.6, 0.0), math.pi / 2),
)


def load_bundle(root_run, pose_run, device):
    root, root_contract, root_result = load_control_checkpoint(root_run, device)
    pose, pose_contract, pose_result = load_control_checkpoint(pose_run, device)
    if root_contract["stage"] != "native_root" or pose_contract["stage"] != "native_pose":
        raise ValueError("Bundle requires Root and Pose checkpoints in that order")
    root_data_compatibility = None
    if root_contract["dataset_manifest_sha256"] != pose_contract["dataset_manifest_sha256"]:
        from motionbricks.data.native_mb_dataset_compatibility import verify_root_contact_only_revision
        root_data_compatibility = verify_root_contact_only_revision(
            root_contract['dataset'], pose_contract['dataset'], root)
    for key in ("min_tokens", "max_tokens", "down_t"):
        if root.args[key] != pose.args[key]:
            raise ValueError(f"Root/Pose temporal contract mismatch: {key}")
    # The factory accepts an already loaded tokenizer and disables the original
    # path loader. Set its bookkeeping flag only after the strict loader has
    # verified the external checkpoint and the frozen tensor-state digest.
    pose._vqvae_model_loaded = True
    infer = motion_inference({"root": root, "pose": pose}, pose.args, device=device)
    metadata = {"root_checkpoint_sha256": root_result["final_checkpoint_sha256"],
                "pose_checkpoint_sha256": pose_result["final_checkpoint_sha256"],
                "root_step": root_result["optimizer_updates"],
                "pose_step": pose_result["optimizer_updates"],
                "tokenizer": pose_contract["tokenizer"],
                "dataset_manifest_sha256": pose_contract["dataset_manifest_sha256"],
                "root_training_dataset_manifest_sha256": root_contract['dataset_manifest_sha256'],
                "root_data_compatibility": root_data_compatibility}
    return infer, Path(pose_contract["dataset"]), metadata


def command_constraints(infer, history, velocity_xz, heading, num_tokens=8):
    """Use the upstream eight-slot convention: four past and four goal frames."""
    if history.ndim != 3 or history.shape[:2] != (1, 4) or not torch.isfinite(history).all():
        raise ValueError("Expected four finite unnormalized global seed/history frames")
    minimum, maximum = infer._args["min_tokens"], infer._args["max_tokens"]
    if not minimum <= num_tokens <= maximum:
        raise ValueError("Requested horizon is outside the native token range")
    device, dtype = history.device, history.dtype
    velocity = torch.as_tensor(velocity_xz, device=device, dtype=dtype)
    if velocity.shape != (2,) or not torch.isfinite(velocity).all() or not math.isfinite(heading):
        raise ValueError("Command velocity and heading must be finite")
    global_rep, local_rep = infer.global_motion_rep, infer.local_motion_rep
    local = infer.motion_rep.dual_rep.global_to_local(
        history, False, False, lengths=torch.tensor([4], device=device))
    start_global = extract_feature_from_motion_rep(history, global_rep, "root")
    start_local = extract_feature_from_motion_rep(local, local_rep, "root")
    start_pose = extract_feature_from_motion_rep(local, local_rep, infer.EXTERNAL_POSE_FEATURE_MODE)
    target = start_global[:, -1:].repeat(1, 4, 1)
    # Relative to the last context frame (index 3), not to an artificial origin.
    offsets = torch.arange(num_tokens * 4 - 7, num_tokens * 4 - 3,
                           device=device, dtype=dtype) / global_rep.fps
    target[:, :, 0] += offsets[None] * velocity[0]
    target[:, :, 2] += offsets[None] * velocity[1]
    target[:, :, 3] = math.cos(heading)
    target[:, :, 4] = math.sin(heading)
    has_global = torch.ones(1, 8, device=device, dtype=torch.bool)
    has_local = torch.zeros_like(has_global)
    has_local[:, :3] = True  # Last context velocity lacks a future frame.
    has_pose = torch.zeros_like(has_global)
    has_pose[:, :4] = True
    allowed = torch.zeros(1, maximum - minimum + 1, device=device, dtype=torch.long)
    allowed[:, num_tokens - minimum] = 1
    return {"global_root_values": torch.cat((start_global, target), 1),
            "has_global_root_values": has_global,
            "local_root_values": torch.cat((start_local, torch.zeros_like(start_local)), 1),
            "has_local_root_values": has_local,
            "local_poses": torch.cat((start_pose, torch.zeros_like(start_pose)), 1),
            "has_local_poses": has_pose,
            "num_tokens": None, "allowed_pred_num_tokens": allowed}


@full_precision_evaluation()
def rollout(infer, seed_frames, plans_per_command=6, commit_frames=8, num_tokens=8):
    if plans_per_command < 1 or not 1 <= commit_frames <= num_tokens * 4 - 4:
        raise ValueError("Invalid rollout/commit length")
    history = seed_frames.clone()
    pieces, plans, samples = [history], [], []
    total_frames = 4
    config = {"num_inference_step": 1, "pose_token_sampling_use_argmax": True,
              "allow_pred_out_of_reach_num_tokens": False, "smooth_root_traj": False,
              "skip_ending_target_cond": True, "final_root_pred_mode": "from_pose_module"}
    with torch.inference_mode():
        for name, velocity, heading in COMMANDS:
            for _ in range(plans_per_command):
                inputs = command_constraints(infer, history, velocity, heading, num_tokens)
                if history.is_cuda:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                prediction, count = infer.predict(**inputs, config=config, info={})
                if history.is_cuda:
                    torch.cuda.synchronize()
                latency = time.perf_counter() - started
                if int(count.item()) != num_tokens or not torch.isfinite(prediction).all():
                    raise ValueError("Invalid native pipeline horizon or nonfinite prediction")
                committed = prediction[:, 4:4 + commit_frames].clone()
                if committed.shape[1] != commit_frames:
                    raise ValueError("Generated clip is shorter than the commit horizon")
                plans.append({"command": name, "velocity_xz_mps": list(velocity),
                              "heading_rad": heading, "first_generated_frame": total_frames,
                              "committed_frames": commit_frames, "predicted_tokens": int(count.item()),
                              "latency_seconds": latency,
                              "future_pose_constraints": 0, "future_source_tokens": 0})
                samples.extend([velocity] * commit_frames)
                pieces.append(committed)
                total_frames += commit_frames
                # Preserve played history. Discarding the predicted context does
                # not blend, shift or replace any generated future frame.
                history = torch.cat((history, committed), 1)[:, -4:].clone()
    features = torch.cat(pieces, 1)
    posed = infer.global_motion_rep.inverse(features, False, return_all=True)["posed_joints"]
    root = posed[0, :, infer.motion_rep.skeleton.root_idx]
    actual_velocity = (root[4:, [0, 2]] - root[3:-1, [0, 2]]) * infer.motion_rep.fps
    requested = torch.tensor(samples, device=root.device, dtype=root.dtype)
    tracking = (actual_velocity - requested).norm(dim=-1)
    joints = posed[0]
    joint_velocity = (joints[1:] - joints[:-1]) * infer.motion_rep.fps
    seam_changes = []
    for plan in plans:
        index = plan["first_generated_frame"]
        seam_changes.append(float((joint_velocity[index - 1] - joint_velocity[index - 2]).norm(dim=-1).mean()))
    latencies = np.asarray([plan["latency_seconds"] for plan in plans])
    summary = {"generated_frames": total_frames - 4,
               "generated_seconds": (total_frames - 4) / infer.motion_rep.fps,
               "command_velocity_error_mean_mps": float(tracking.mean()),
               "command_velocity_error_p95_mps": float(torch.quantile(tracking, 0.95)),
               "root_height_min_m": float(root[4:, 1].min()),
               "root_height_max_m": float(root[4:, 1].max()),
               "replan_joint_velocity_change_mean_mps": float(np.mean(seam_changes)),
               "replan_joint_velocity_change_max_mps": max(seam_changes),
               "inference_latency_median_ms": float(np.median(latencies) * 1000),
               "inference_latency_p95_ms": float(np.quantile(latencies, 0.95) * 1000)}
    return features, posed, plans, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-run", type=Path, required=True)
    parser.add_argument("--pose-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--plans-per-command", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    if args.device == "cuda":
        if args.gpu is None:
            parser.error("CUDA evaluation requires an explicitly checked --gpu")
        select_gpu(args.gpu)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    np.random.seed(2333)
    infer, dataset, bundle = load_bundle(args.root_run, args.pose_run, args.device)
    records = json.loads((dataset / "accepted.json").read_text())
    record = next(row for row in records if row["split"] == "validation" and row["behavior"] == "idle")
    with np.load(dataset / record["shard"], allow_pickle=False) as data:
        raw = torch.from_numpy(data["global_features"][:4].copy()).to(args.device)[None]
    seed = infer.global_motion_rep.change_first_heading(raw, 0.0, False, False)
    features, posed, plans, summary = rollout(infer, seed, args.plans_per_command)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "command_rollout.npz", global_features=features.cpu().numpy()[0],
                        prediction=posed.cpu().numpy()[0], fps=np.float32(infer.motion_rep.fps),
                        initial_source_frames=seed.cpu().numpy()[0])
    report = {"task": "start-only native Root/Pose closed-loop command stress probe",
              "quality_approved": False, "ue_deployment": False, "bundle": bundle,
              "seed_shard": record["shard"], "seed_shard_sha256": sha256(dataset / record["shard"]),
              "source_frames_used": [0, 1, 2, 3], "subsequent_history": "generated frames only",
              "goal_generation": "constant command velocity endpoints; previous root height; requested heading",
              "future_pose_condition": "none; stress probe outside always-endpose native training distribution",
              "foot_slip_and_penetration": "not_assessed", "postprocessing": "none",
              "root_overwrite": False, "fps": infer.motion_rep.fps, "plans": plans, "summary": summary,
              "device": args.device, "gpu_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
              "runtime_note": "model-call timing on the recorded host only, not UE or RTX3080 acceptance",
              "source_sha256": sha256(Path(__file__))}
    write_json(args.output / "metrics.json", report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
