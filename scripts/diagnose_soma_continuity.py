"""Separate raw codec, token prediction and replanning errors; never filter poses."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.bones_control import decode_tokens
from motionbricks.data.bones_rollout import SomaLocomotionBackend
from motionbricks.data.bones_seed import validation_window_records
from scripts.portable_bones_control import load_bundle


def stats(values):
    values = torch.as_tensor(values).detach().double().flatten().cpu()
    if not values.numel():
        return {"n": 0, "mean": None, "p95": None, "max": None}
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite motion metric")
    return {"n": len(values), "mean": values.mean().item(),
            "p95": values.quantile(.95).item(), "max": values.max().item()}


def angular_steps(rotations):
    relative = rotations[:-1].transpose(-1, -2) @ rotations[1:]
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return cosine.clamp(-1, 1).acos().rad2deg()


def motion_metrics(motion, rep, boundaries=(), source=None):
    pose = rep.inverse(motion, False, return_all=False)
    positions, rotations = pose["posed_joints"][0], pose["local_joint_rots"][0]
    angular = angular_steps(rotations)
    steps = positions.diff(dim=0).norm(dim=-1) * 1000
    boundary = torch.zeros(len(steps), dtype=torch.bool, device=motion.device)
    for frame in boundaries:
        if 0 < frame < motion.shape[1]:
            boundary[frame - 1] = True
    peak = angular.flatten().argmax().item()
    frame, joint = divmod(peak, angular.shape[1])
    result = {"angular_step_deg": stats(angular), "joint_step_mm": stats(steps),
              "boundary_angular_step_deg": stats(angular[boundary]),
              "interior_angular_step_deg": stats(angular[~boundary]),
              "boundary_joint_step_mm": stats(steps[boundary]),
              "interior_joint_step_mm": stats(steps[~boundary]),
              "joint_acceleration_mps2": stats(positions.diff(n=2, dim=0).norm(dim=-1) * rep.fps ** 2),
              "worst_angular_step": {"destination_frame": frame + 1,
                                     "joint": rep.skeleton.bone_order_names[joint],
                                     "deg": angular[frame, joint].item()},
              "per_joint_angular_step_deg": {name: stats(angular[:, index]) for index, name in
                                              enumerate(rep.skeleton.bone_order_names)}}
    features6d = motion[0, :, rep.indices["global_rot_data"]].reshape(-1, 30, 6)
    first, second = features6d[..., :3], features6d[..., 3:]
    result["rotation6d_min_vector_norm"] = min(first.norm(dim=-1).min().item(), second.norm(dim=-1).min().item())
    result["rotation6d_min_normalized_cross_norm"] = torch.cross(
        torch.nn.functional.normalize(first, dim=-1), torch.nn.functional.normalize(second, dim=-1), dim=-1).norm(dim=-1).min().item()
    if source is not None:
        target = rep.inverse(source, False, return_all=False)["posed_joints"][0]
        relative_error = (positions - positions[:, :1]) - (target - target[:, :1])
        result["root_relative_mpjpe_mm"] = stats(relative_error[4:].norm(dim=-1) * 1000)
    return result


def source_features(path, rep, device):
    with np.load(path, allow_pickle=False) as data:
        local = torch.from_numpy(data["source_local_rot_mats"].copy()).to(device)
        root = torch.from_numpy(data["source_positions_m"][:, 0].copy()).to(device)
        contacts = torch.from_numpy(data["source_contacts"].copy()).float().to(device)
    global_rot, joints, _ = rep.skeleton.fk(local, root)
    return rep({"posed_joints": joints, "global_joint_rots": global_rot,
                "local_joint_rots": local, "translation": root,
                "foot_contacts": contacts}, to_normalize=False)[None]


def decode_source(raw, inference, predicted=False, frames=64):
    rep, pose = inference.motion_rep, inference._pose_model
    local = rep.dual_rep.global_to_local(raw[:, :frames + 1], False, True,
                                        lengths=torch.tensor([frames + 1], device=raw.device))[:, :frames]
    if predicted:
        _, decoded, _ = pose.predict_local(local, rep.normalize(raw[:, :frames]))
    else:
        tokens = pose.pose_net.encode_into_idx(local, fetch_overall_indices=False)
        mask = torch.zeros(local.shape[:2], dtype=torch.bool, device=local.device)
        mask[:, :4] = True
        decoded = decode_tokens(pose.pose_net, local, mask, tokens, local)
    result = rep.dual_rep.local_to_global(decoded, True, False,
                                         lengths=torch.tensor([frames], device=raw.device))
    result[..., [0, 2]] += raw[:, :1, [0, 2]]
    yaw = torch.atan2(result[..., 4], result[..., 3]) + torch.atan2(raw[:, :1, 4], raw[:, :1, 3])
    result[..., 3], result[..., 4] = yaw.cos(), yaw.sin()
    return result


def command(frame):
    if frame < 120:
        return [0., 0.], 0.
    if frame < 360:
        return [0., 1.2], 0.
    if frame < 480:
        return [1.2, 0.], np.pi / 2
    return [0., 0.], np.pi / 2


def rollout(inference, initial, delay, frames=600):
    backend = SomaLocomotionBackend(inference, initial, argmax=True)
    chunks, boundaries = [], []
    active = backend.predict_plan([[0., 0.]], [0.])["planned_motion"]
    cursor, published = 0, 0
    while published < frames:
        velocity, heading = command(published)
        new = backend.predict_plan([velocity], [heading])["planned_motion"]
        if delay:
            preceding = active[:, cursor:cursor + delay]
            if preceding.shape[1] != delay:
                raise ValueError("Old plan exhausted in delayed experiment")
            chunks.append(preceding)
            published += delay
            # Reproduce the engine: it skips the new prediction's unexecuted prefix.
            chosen = new[:, delay:delay + 4]
        else:
            chosen = new[:, :4]
        boundaries.append(published)
        chunks.append(chosen)
        published += 4
        backend.history = chosen.clone()
        active, cursor = new, delay + 4
    return torch.cat(chunks, 1)[:, :frames], boundaries


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--native-root-stage", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new report directory")
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    inference, bundle = load_bundle(args.bundle, "cuda:0")
    inference.use_native_root_stage = args.native_root_stage
    rep = inference.motion_rep
    args.output.mkdir(parents=True)
    report = {"bundle_sha256": file_sha256(args.bundle), "post_processing": None,
              "root_stage": "released_MB" if args.native_root_stage else "legacy_prefix_override",
              "test_used": False, "quality_approved": False, "windows": {}, "rollouts": {},
              "single_plan_metric_scope": "all frames including reconstructed history and padded end; use 4frame boundary results for deployed prefix",
              "validated_continuation_windows": []}
    for behavior in ("idle", "walk", "run", "turn", "backward"):
        path = args.windows / (behavior + "_review_window.npz")
        raw = source_features(path, rep, inference.device)
        codec = decode_source(raw, inference)
        predicted = decode_source(raw, inference, True)
        oracle_chunks, predicted_chunks = [], []
        for start in range(0, 64, 4):
            window = raw[:, start:start + 65]
            oracle_chunks.append(decode_source(window, inference)[:, 4:8])
            predicted_chunks.append(decode_source(window, inference, True)[:, 4:8])
        stitched_source = raw[:, 4:68]
        report["windows"][behavior] = {"input_sha256": file_sha256(path),
            "source": motion_metrics(raw[:, :64], rep),
            "oracle_tokens": motion_metrics(codec, rep, source=raw[:, :64]),
            "predicted_tokens_source_root": motion_metrics(predicted, rep, source=raw[:, :64]),
            "teacher_history_oracle_tokens_4frame": motion_metrics(torch.cat(oracle_chunks, 1), rep,
                                                                  range(4, 64, 4), stitched_source),
            "teacher_history_predicted_tokens_4frame": motion_metrics(torch.cat(predicted_chunks, 1), rep,
                                                                     range(4, 64, 4), stitched_source)}
        print(json.dumps({"behavior": behavior, "oracle_mpjpe_mm": report["windows"][behavior]["oracle_tokens"]["root_relative_mpjpe_mm"]["mean"],
                          "predicted_mpjpe_mm": report["windows"][behavior]["predicted_tokens_source_root"]["root_relative_mpjpe_mm"]["mean"]}), flush=True)
    if args.dataset:
        records = json.loads((args.dataset / "accepted.json").read_text())
        for record in validation_window_records([r for r in records if r["split"] == "validation"], 128, 3):
            start = record["window_start_frame"]
            with np.load(args.dataset / record["shard"], allow_pickle=False) as data:
                raw = torch.from_numpy(data["global_features"][start:start + 129].copy())[None].to(inference.device)
            chunks = [decode_source(raw[:, offset:offset + 65], inference, True)[:, 4:8]
                      for offset in range(0, 64, 4)]
            measured = motion_metrics(torch.cat(chunks, 1), rep, range(4, 64, 4), raw[:, 4:68])
            report["validated_continuation_windows"].append({"filename": record["filename"],
                "start": start, "behavior": record.get("behavior"), **measured})
    for delay in (0, 12):
        motion, boundaries = rollout(inference, bundle["initial_history"].cuda(), delay)
        report["rollouts"][str(delay)] = motion_metrics(motion, rep, boundaries)
        np.savez_compressed(args.output / f"delay_{delay}.npz", global_features=motion[0].cpu().numpy(),
                            boundaries=np.asarray(boundaries), fps=np.asarray(60))
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output), "rollouts": {k: v["angular_step_deg"] for k, v in report["rollouts"].items()}}), flush=True)


if __name__ == "__main__":
    main()
