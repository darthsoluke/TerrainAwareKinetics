"""Portable inference-only export of the strictly loaded native MB SOMA stages."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.native_mb_data import FPS, REPRESENTATION_NAME
from motionbricks.data.native_mb_models import build_native_model
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from motionbricks.motionlib.core.utils.stats import Stats
from scripts.train_native_mb_control import state_digest


SCHEMA = "motionbricks.native_mb_soma_bundle.v1"


def select_references(dataset, rep):
    """Select real contiguous train segments, without stationary/reversed synthesis."""
    records = json.loads((dataset / "accepted.json").read_text())
    manifest = json.loads((dataset / "native_manifest.json").read_text())
    candidates = {"idle": [], "walk": [], "run": []}
    position_indices = rep.indices["ric_data"]
    for row in records:
        if row["split"] != "train":
            continue
        path = dataset / row["shard"]
        if file_sha256(path) != manifest["artifact_sha256"][row["shard"]]:
            raise ValueError("Reference shard hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            raw = data["global_features"].copy()
        speed = np.linalg.norm(np.diff(raw[:, [0, 2]], axis=0), axis=-1) * FPS
        body_speed = np.linalg.norm(np.diff(raw[:, position_indices].reshape(-1, 29, 3), axis=0), axis=-1).mean(-1) * FPS
        quiet = np.convolve(speed + body_speed, np.ones(29) / 29, "valid")
        start = int(quiet.argmin())
        if speed[start:start + 29].max() <= 0.03 and body_speed[start:start + 29].mean() <= 0.03:
            candidates["idle"].append((float(quiet[start]), row, start, raw[start:start + 30]))
        style = row["behavior"]
        if style not in ("walk", "run"):
            continue
        low, high = (0.6, 1.8) if style == "walk" else (1.8, 3.5)
        for start in range(0, len(raw) - 120 + 1, 15):
            velocities = np.diff(raw[start:start + 120, [0, 2]], axis=0) * FPS
            speeds = np.linalg.norm(velocities, axis=-1)
            if np.percentile(speeds, 10) < low or np.percentile(speeds, 90) > high:
                continue
            direction = velocities.mean(0)
            straightness = np.linalg.norm(direction) / max(float(speeds.mean()), 1e-8)
            score = float(1 - straightness + speeds.std() / speeds.mean())
            candidates[style].append((score, row, start, raw[start:start + 120]))
    clips, provenance = {}, {}
    for style, available in candidates.items():
        if not available:
            raise ValueError(f"No genuine train-only {style} reference met selection requirements")
        score, row, start, raw = min(available, key=lambda value: value[0])
        clips[style] = torch.from_numpy(raw)
        provenance[style] = {"split": "train", "shard": row["shard"], "filename": row["filename"],
                             "shard_sha256": file_sha256(dataset / row["shard"]),
                             "start_frame": start, "frames": len(raw), "selection_score": score,
                             "role": "input reference keyframes, never played animation"}
    return clips, provenance


def load_bundle(path, device="cuda:0"):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    return instantiate_bundle(bundle, device)


def instantiate_bundle(bundle, device="cuda:0"):
    """Strict shared model construction; callers retain their own artifact gates."""
    if bundle.get("schema") != SCHEMA or bundle.get("fps") != FPS:
        raise ValueError("Expected native MB SOMA 30 Hz bundle")
    stats = Stats()
    stats.register_from_tensors(bundle["stats_mean"], bundle["stats_std"])
    rep = GlobalRootGlobalJoints(FPS, SOMASkeleton30(), REPRESENTATION_NAME, stats=stats)
    if (list(rep.skeleton.bone_order_names) != bundle["bone_names"]
            or not torch.allclose(rep.skeleton.neutral_joints.cpu(), bundle["neutral_joints"], atol=1e-6, rtol=0)):
        raise ValueError("Portable source skeleton mismatch")
    tokenizer = build_native_model("tokenizer", rep, OmegaConf.create(bundle["tokenizer_config"]))
    tokenizer.pose_net.load_state_dict(bundle["tokenizer_state"], strict=True)
    if state_digest(tokenizer.pose_net) != bundle["tokenizer_state_sha256"]:
        raise ValueError("Frozen tokenizer tensor binding mismatch")
    models = {}
    for stage in ("root", "pose"):
        model = build_native_model(stage, rep, OmegaConf.create(bundle[stage + "_config"]),
                                   tokenizer=tokenizer if stage == "pose" else None)
        model.load_state_dict(bundle[stage + "_state"], strict=True)
        if state_digest(model.backbone_net) != bundle[stage + "_backbone_sha256"]:
            raise ValueError(f"Portable {stage} tensor binding mismatch")
        model.to(device).eval()
        model.move_supporting_nets_to_device(torch.device(device))
        models[stage] = model
    if state_digest(models["pose"].supporting_nets["pose_net"]) != bundle["tokenizer_state_sha256"]:
        raise ValueError("Pose tokenizer changed during loading")
    models["pose"]._vqvae_model_loaded = True
    for key in ("min_tokens", "max_tokens", "down_t"):
        if models["root"].args[key] != models["pose"].args[key]:
            raise ValueError("Native temporal stage contracts disagree")
    infer = motion_inference(models, models["pose"].args, device=device)
    infer.requires_grad_(False)
    return infer, bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-run", type=Path, required=True)
    parser.add_argument("--pose-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse to overwrite an inference bundle")
    torch.set_num_threads(4)
    from scripts.evaluate_native_mb_pipeline import load_bundle as load_runs
    infer, dataset, metadata = load_runs(args.root_run, args.pose_run, "cpu")
    clips, provenance = select_references(dataset, infer.global_motion_rep)
    initial = infer.global_motion_rep.change_first_heading(clips["idle"][None, :4], 0.0, False, False)
    # Coordinate-origin change only; preserve the source hip height and all poses.
    initial[..., 0] -= initial[0, -1, 0].clone()
    initial[..., 2] -= initial[0, -1, 2].clone()
    tokenizer_run = Path(metadata["tokenizer"]["run"])
    bundle = {"schema": SCHEMA, "fps": FPS, "metadata": metadata,
              "initial_history": initial, "reference_clips": clips, "reference_provenance": provenance,
              "bone_names": list(infer.motion_rep.skeleton.bone_order_names),
              "neutral_joints": infer.motion_rep.skeleton.neutral_joints.cpu(),
              "stats_mean": infer.motion_rep.dual_rep.stats.mean.cpu(),
              "stats_std": infer.motion_rep.dual_rep.stats.std.cpu(),
              "tokenizer_state": infer._vqvae_pose_model.state_dict(),
              "tokenizer_state_sha256": state_digest(infer._vqvae_pose_model),
              "tokenizer_config": OmegaConf.to_container(OmegaConf.load(tokenizer_run / "native_config.yaml"), resolve=False),
              "quality_approved": False, "post_processing": None}
    for stage, run, model in (("root", args.root_run, infer._root_model), ("pose", args.pose_run, infer._pose_model)):
        bundle[stage + "_state"] = model.state_dict()
        bundle[stage + "_backbone_sha256"] = state_digest(model.backbone_net)
        bundle[stage + "_config"] = OmegaConf.to_container(OmegaConf.load(run / "native_config.yaml"), resolve=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    receipt = {"bundle": str(args.output), "sha256": file_sha256(args.output),
               "bytes": args.output.stat().st_size, "metadata": metadata,
               "reference_provenance": provenance, "quality_approved": False}
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
