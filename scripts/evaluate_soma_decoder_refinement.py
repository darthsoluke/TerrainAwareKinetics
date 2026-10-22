"""Fixed held-out continuations plus three raw 60-second control sequences."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256, require_same_snapshot
from motionbricks.data.bones_control import state_digest
from motionbricks.data.bones_rollout import SomaLocomotionBackend
from motionbricks.data.bones_seed import official_sole_surfaces, validation_window_records
from motionbricks.data.soma_decoder_refinement import deployed_request
from motionbricks.data.soma_skin_reference import load_official_skinner
from motionbricks.data.sole_region_contact import sole_region_indices
from scripts.diagnose_soma_continuity import motion_metrics, stats
from scripts.evaluate_soma_pose_goals import compare_contact, raw_rollout, teacher_prediction
from scripts.portable_bones_control import load_bundle


def command_at(frame):
    # Five-second blocks; no interpolated pose/root or command filtering.
    commands = (("idle", (0., 0.), 0.), ("walk", (0., 1.2), 0.),
        ("run", (0., 2.7), 0.), ("stop", (0., 0.), 0.),
        ("turn90", (1.2, 0.), np.pi / 2), ("turn180", (-1.2, 0.), -np.pi / 2),
        ("backward", (0., -1.), 0.), ("strafe_left", (-1., 0.), 0.),
        ("strafe_right", (1., 0.), 0.), ("walk_again", (0., 1.2), 0.),
        ("reverse", (0., -1.2), np.pi), ("stop_again", (0., 0.), np.pi))
    return commands[min(frame // 300, len(commands) - 1)]


@torch.no_grad()
def long_rollout(inference, initial, frames=3600, commit_frames=4):
    backend = SomaLocomotionBackend(inference, initial, argmax=True, commit_frames=commit_frames)
    chunks, timings = [], []
    for frame in range(0, frames, commit_frames):
        _, velocity, heading = command_at(frame)
        result = backend.step([velocity], [heading])
        chunks.append(result["motion"])
        timings.append(result["generation_seconds"] * 1000)
    motion = torch.cat(chunks, 1)
    metrics = motion_metrics(motion, inference.motion_rep, range(commit_frames, frames, commit_frames))
    speed = motion[0, :, [0, 2]].diff(dim=0) * inference.motion_rep.fps
    requested = speed.new_tensor([command_at(frame)[1] for frame in range(frames - 1)])
    errors = (speed - requested).norm(dim=-1)
    metrics["root_command_velocity_error_mps"] = stats(errors)
    metrics["per_command"] = {command_at(start)[0]: {
        "velocity_error_mps": stats(errors[start:min(start + 299, len(errors))]),
        "speed_mps": stats(speed[start:min(start + 299, len(speed))].norm(dim=-1))}
        for start in range(0, frames, 300)}
    metrics["model_generation_ms"] = stats(timings)
    metrics["scope"] = "raw deterministic argmax; no ground-truth reset; different observed initial histories, not training seeds"
    return motion, metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--commit-frames", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh evaluation directory")
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    inference, bundle = load_bundle(args.bundle, "cuda:0")
    if getattr(inference, "goal_library", None) is not None:
        raise ValueError("This decoder comparison freezes the original goal-free Root/Pose")
    rep = inference.motion_rep
    snapshot = dataset_snapshot(args.dataset)
    require_same_snapshot(bundle["contracts"]["pose"]["dataset_snapshot"], snapshot)
    records = validation_window_records([r for r in json.loads((args.dataset / "accepted.json").read_text())
                                         if r["split"] == "validation"], 128, 3)
    skinner = load_official_skinner(rep.skeleton)
    surfaces, _, _, _ = official_sole_surfaces(rep.skeleton)
    indices = {side: sole_region_indices(surfaces, side) for side in ("Left", "Right")}
    args.output.mkdir(parents=True)
    report = {"bundle_sha256": file_sha256(args.bundle), "dataset_sha256": snapshot["sha256"],
        "evaluation_sha256": file_sha256(Path(__file__)), "test_used": False, "post_processing": None,
        "root_condition_policy": inference._vqvae_pose_model.root_condition_policy,
        "history_geometry": getattr(inference, "history_geometry", "predicted_features"),
        "commit_frames": args.commit_frames,
        "tokenizer_state": state_digest(inference._vqvae_pose_model),
        "frozen_state": {"root": state_digest(inference._root_model), "pose": state_digest(inference._pose_model.backbone_net),
                         "encoder": state_digest(inference._vqvae_pose_model.encoder),
                         "quantizer": state_digest(inference._vqvae_pose_model.quantizer),
                         "decoder": state_digest(inference._vqvae_pose_model.decoder)},
        "scoring": "fixed 57 windows, shared source labels, skip4; native goals use source root ONLY",
        "source_root_scope": "privileged source trajectory diagnosis, not free control",
        "quality_approved": False, "ue_deadline_assessed": False, "windows": [], "rollouts": {}}
    initial_histories = []
    for index, record in enumerate(records):
        start, path = record["window_start_frame"], args.dataset / record["shard"]
        with np.load(path, allow_pickle=False) as data:
            raw = torch.from_numpy(data["global_features"][start:start + 129].copy())[None].cuda()
        native_chunks, teacher_chunks = [], []
        for offset in range(0, 64, args.commit_frames):
            source = raw[:, offset:offset + 65]
            teacher_chunks.append(teacher_prediction(source, inference, None)[0][:, 4:4 + args.commit_frames])
            _, native = deployed_request(inference, source, source[:, :4],
                                        commit_frames=args.commit_frames, return_committed=True)
            native_chunks.append(native)
        teacher, native = torch.cat(teacher_chunks, 1), torch.cat(native_chunks, 1)
        target = raw[:, 4:68]
        scores = {}
        for name, motion in (("teacher_source_root", teacher), ("native_root", native)):
            measured = motion_metrics(motion, rep, range(args.commit_frames, 64, args.commit_frames), target)
            measured["root_position_error_m"] = stats((motion[:, 4:, :3] - target[:, 4:, :3]).norm(dim=-1))
            measured.update(compare_contact(target, motion, rep, skinner, indices))
            scores[name] = measured
        report["windows"].append({"filename": record["filename"], "start": start, "behavior": record.get("behavior"),
            "input_sha256": file_sha256(path), **scores})
        if record.get("behavior") == "idle" and len(initial_histories) < 3:
            initial_histories.append((raw[:, :4].clone(), record["filename"], start))
        if index % 10 == 0:
            print(json.dumps({"validation_windows": index + 1}), flush=True)
    report["aggregate"] = {}
    for name in ("teacher_source_root", "native_root"):
        rows = [r[name] for r in report["windows"]]
        report["aggregate"][name] = {
            "mean_window_mpjpe_mm": float(np.mean([r["root_relative_mpjpe_mm"]["mean"] for r in rows])),
            "mean_window_root_position_error_m": float(np.mean([r["root_position_error_m"]["mean"] for r in rows])),
            "worst_joint_step_deg": max(r["angular_step_deg"]["max"] for r in rows),
            "mean_window_boundary_joint_step_mm": float(np.mean([r["boundary_joint_step_mm"]["p95"] for r in rows])),
            "mean_window_penetration_m": float(np.mean([r["predicted_contact"]["region_penetration_m"]["mean"] for r in rows])),
            "mean_window_measured_slip_mps": float(np.mean([r["predicted_contact"]["measured_slip_mps"]["mean"]
                for r in rows if r["predicted_contact"]["measured_slip_mps"] is not None]))}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if len(initial_histories) != 3:
        raise ValueError("Three fixed held-out idle histories are required")
    for index, (initial, filename, start) in enumerate(initial_histories):
        motion, metrics = long_rollout(inference, initial, commit_frames=args.commit_frames)
        report["rollouts"][str(index)] = {"initial_filename": filename, "initial_frame": start, **metrics}
        np.savez_compressed(args.output / f"long_{index}.npz", global_features=motion[0].cpu().numpy(),
                            boundaries=np.arange(args.commit_frames, 3600, args.commit_frames), fps=np.asarray(60))
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({"rollout": index, "joint_step_p95_mm": metrics["boundary_joint_step_mm"]["p95"],
                         "command_error_mps": metrics["root_command_velocity_error_mps"]["mean"]}), flush=True)
    for delay in (0, 12):
        motion, boundaries, measured = raw_rollout(inference, bundle["initial_history"].cuda(), None, delay)
        report["rollouts"]["delay_" + str(delay)] = measured
        np.savez_compressed(args.output / f"delay_{delay}.npz", global_features=motion[0].cpu().numpy(),
                            boundaries=np.asarray(boundaries), fps=np.asarray(60))
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"completed": str(args.output), "aggregate": report["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
