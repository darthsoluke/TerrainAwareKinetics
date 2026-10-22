"""Matched Root/Pose adaptation with native MB losses and sparse body goals."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256, require_same_snapshot
from motionbricks.data.bones_control import state_digest
from motionbricks.data.soma_goal_conditioning import library_records
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.portable_bones_control import load_bundle


@torch.no_grad()
def build_library(dataset, records, rep, device):
    features, velocities, provenance = [], [], []
    for record in library_records(records):
        path = dataset / record["shard"]
        with np.load(path, allow_pickle=False) as data:
            raw = torch.from_numpy(data["global_features"].copy()).to(device)
        ids = torch.arange(4, len(raw), 4, device=device)
        poses = raw[ids, None]
        canonical = rep.change_first_heading(poses, torch.zeros(len(ids), device=device), False, False)
        velocity = (raw[ids][:, [0, 2]] - raw[ids - 4][:, [0, 2]]) * rep.fps / 4
        probe = poses.expand(-1, 2, -1).clone()
        probe[:, 1, [0, 2]] += velocity
        local_probe = rep.change_first_heading(probe, torch.zeros(len(ids), device=device), False, False)
        features.append(canonical[:, 0].cpu())
        velocities.append(local_probe[:, 1, [0, 2]].cpu())
        digest = file_sha256(path)
        provenance.extend({"split": "train", "shard": record["shard"], "frame": frame,
                           "shard_sha256": digest} for frame in ids.tolist())
    return {"schema": "motionbricks.train_pose_library.v1", "features": torch.cat(features),
            "local_velocity_xz_mps": torch.cat(velocities), "provenance": provenance,
            "selection": "nearest heading-local XZ velocity; one pose; no phase fitting",
            "frame_stride": 4, "velocity_intervals": 4}


def native_losses(root, pose, global_motion, local):
    frames, batch_size = local.shape[1], len(local)
    lengths = torch.full((batch_size, 1), frames // 4, dtype=torch.long, device=local.device)
    global_values, global_mask = root._sample_the_conditions(
        extract_feature_from_motion_rep(global_motion, root.global_motion_rep, root.args["global_root_feature"]), frames)
    local_values, local_mask = root._sample_the_conditions(
        extract_feature_from_motion_rep(local, root.local_motion_rep, root.args["local_root_feature"]), frames)
    body_values, body_mask = root._sample_the_conditions(
        extract_feature_from_motion_rep(local, root.local_motion_rep, root.args["local_pose_feature"]), frames)
    root_output = root.backbone_net(global_values, global_mask, local_values, local_mask,
                                   body_values, body_mask, lengths, groundtruth_num_tokens=lengths)
    root_losses = root.loss({"global_motions": global_motion, "local_motions": local,
                             "groundtruth_num_tokens": lengths}, root_output)
    with torch.no_grad():
        tokens = pose.pose_net.encode_into_idx(local, fetch_overall_indices=False)
    values, mask, _, _ = pose._sample_the_local_pose_conditions(local, frames)
    ids = torch.full_like(tokens, pose.backbone_net.POSE_MASK_ID)
    trajectory = extract_feature_from_motion_rep(global_motion, pose.global_motion_rep, pose.args["cond_root_feature"])
    output = pose.backbone_net(ids, trajectory, values, mask, lengths, None, None)
    pose_losses = pose.loss({"groundtruth_pose_tokens": tokens,
                             "focused_token_mask": torch.ones_like(tokens, dtype=torch.bool)}, output)
    return root_losses["loss"], pose_losses["loss"], {
        **{"root_" + name: value for name, value in root_losses.items()},
        "pose_ce": pose_losses["pose_loss"], "body_goal_frames": mask[:, 4:].sum()}


def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 2000:
        raise ValueError("Fresh bounded experiment required")
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    rng = np.random.default_rng(2333)
    inference, bundle = load_bundle(args.bundle, "cuda:0")
    snapshot = dataset_snapshot(args.dataset)
    require_same_snapshot(bundle["contracts"]["pose"]["dataset_snapshot"], snapshot)
    gpu = str(torch.cuda.get_device_properties(0).uuid)
    gpu = gpu if gpu.startswith("GPU-") else "GPU-" + gpu
    if gpu not in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","):
        raise ValueError("Training requires an explicitly checked GPU")
    records = json.loads((args.dataset / "accepted.json").read_text())
    selected = library_records(records)
    clips = []
    for record in selected:
        with np.load(args.dataset / record["shard"], allow_pickle=False) as data:
            clips.append(data["global_features"].copy())
    root, pose, rep = inference._root_model, inference._pose_model, inference.motion_rep
    library = build_library(args.dataset, records, rep, "cuda:0")
    policy = "single_endpoint" if args.arm == "B" else "history_only"
    for model in (root, pose):
        model.args["body_goal_policy"] = policy
        model.backbone_net.requires_grad_(True)
    frozen = state_digest(pose.pose_net)
    optimizers = [torch.optim.Adam(model.backbone_net.parameters(), lr=5e-5) for model in (root, pose)]
    args.output.mkdir(parents=True)
    contract = {"schema": "motionbricks.soma_pose_goals.v1", "arm": args.arm, "body_goal_policy": policy,
        "initial_bundle_sha256": file_sha256(args.bundle), "dataset_sha256": snapshot["sha256"],
        "train_clips": len(selected), "library_frames": len(library["features"]), "seed": 2333,
        "optimizer": "Adam", "lr": 5e-5, "batch": 4, "steps_per_stage": args.steps, "gpu_uuid": gpu,
        "objectives": "released root_model.loss and pose_model.loss unchanged; tokenizer fully frozen",
        "source_endpoint_scope": "TRAINING only; validation/inference use train-library poses",
        "pose_training_trajectory": "source root; raw closed-loop evaluates predicted root",
        "decoder_root_policy": "existing provided_velocity; native refinement NOT restored in this trial",
        "source_mechanism": "MotionBricks 3/6.1 sparse context plus reference-pose goals",
        "post_processing": None, "quality_approved": False,
        "source_sha256": {name: file_sha256(Path(name)) for name in (
            "scripts/train_soma_pose_goals.py", "motionbricks/data/soma_goal_conditioning.py",
            "motionbricks/data/bones_control.py", "motionbricks/data/bones_rollout.py",
            "scripts/portable_bones_control.py")}}
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
    for name in contract["source_sha256"]:
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    sampling = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(args.steps):
        frames = int(rng.choice((64, 128)))
        chosen = rng.integers(len(clips), size=4)
        starts = [int(rng.integers(len(clips[index]) - frames)) for index in chosen]
        sampling.update(np.asarray([frames, *chosen, *starts], dtype=np.int64).tobytes())
        raw = torch.from_numpy(np.stack([clips[index][start:start + frames + 1]
                                         for index, start in zip(chosen, starts)])).cuda()
        yaw = torch.from_numpy(rng.uniform(0, 2 * np.pi, 4).astype(np.float32)).cuda()
        with torch.no_grad():
            global_motion = rep.change_first_heading(raw, yaw, False, True)
            local = rep.dual_rep.global_to_local(global_motion, True, True,
                        lengths=torch.full((4,), frames + 1, device=raw.device))[:, :frames]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        root_loss, pose_loss, terms = native_losses(root, pose, global_motion[:, :frames], local)
        loss = root_loss + pose_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite native objective")
        loss.backward()
        for model, optimizer in zip((root, pose), optimizers):
            parameters = list(model.backbone_net.parameters())
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("Nonfinite gradient")
            torch.nn.utils.clip_grad_norm_(parameters, .5)
            optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            event = {"step": step + 1, **{name: float(value.detach()) for name, value in terms.items()}}
            with (args.output / "training.jsonl").open("a") as log:
                log.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps(event), flush=True)
    if state_digest(pose.pose_net) != frozen:
        raise RuntimeError("Tokenizer/EMA changed")
    adapted = copy.deepcopy(bundle)
    for stage, model in (("root", root), ("pose", pose)):
        adapted[stage + "_state"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        adapted[stage + "_config"]["model"]["args"]["body_goal_policy"] = policy
    adapted["body_goal_policy"] = policy
    if args.arm == "B":
        adapted["reference_pose_library"] = library
    adapted["pose_goal_adaptation"] = {**contract, "sampling_sha256": sampling.hexdigest()}
    adapted["quality_approved"] = False
    torch.save(adapted, args.output / "backend.pt")
    torch.save({"steps": args.steps, "root_optimizer": optimizers[0].state_dict(),
        "pose_optimizer": optimizers[1].state_dict(), "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
        args.output / "training_state.pt")
    result = {"steps_per_stage": args.steps, "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
        "sampling_sha256": sampling.hexdigest(), "tokenizer_unchanged": True,
        "bundle_sha256": file_sha256(args.output / "backend.pt"), "quality_approved": False}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
