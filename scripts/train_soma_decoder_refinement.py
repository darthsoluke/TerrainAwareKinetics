"""Matched 10k-update decoder refinement, with frozen original Root/Pose/codebooks."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256, require_same_snapshot
from motionbricks.data.bones_control import state_digest
from motionbricks.data.bones_models import BonesTokenizer
from motionbricks.data.soma_decoder_refinement import (deployed_request, refinement_loss,
    rollout_probability, source_label_window, source_target_with_history, warmup_request)
from scripts.portable_bones_control import load_bundle


def save_bundle(bundle, pose, output, contract, sampling, completed_updates):
    adapted = copy.copy(bundle)
    adapted["pose_config"] = copy.deepcopy(bundle["pose_config"])
    adapted["pose_config"]["model"]["pose_vqvae_network"]["root_condition_policy"] = pose.pose_net.root_condition_policy
    adapted["pose_state"] = {key: value.detach().cpu() for key, value in pose.state_dict().items()}
    adapted["decoder_refinement"] = {**contract, "sampling_sha256": sampling,
                                     "completed_updates": completed_updates}
    adapted["quality_approved"] = False
    torch.save(adapted, output)


def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset", "labels", "bundle", "tokenizer-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or (args.steps != 10000 and not (args.smoke_test and args.steps == 2)):
        raise ValueError("Fresh fixed 10k trial required (or explicit two-update smoke test)")
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    rng = np.random.default_rng(2333)
    inference, bundle = load_bundle(args.bundle, "cuda:0")
    rep, pose = inference.motion_rep, inference._pose_model
    snapshot = dataset_snapshot(args.dataset)
    require_same_snapshot(bundle["contracts"]["pose"]["dataset_snapshot"], snapshot)
    gpu = str(torch.cuda.get_device_properties(0).uuid)
    gpu = gpu if gpu.startswith("GPU-") else "GPU-" + gpu
    if gpu not in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","):
        raise ValueError("An explicitly checked idle GPU is required")
    source_config = OmegaConf.load(args.tokenizer_config)
    coefficients = OmegaConf.to_container(source_config.model.args, resolve=True)
    coefficients["rollout_commit_frames"] = 4
    if not coefficients.get("sole_region_supervision"):
        raise ValueError("Reuse the original region-supervised tokenizer objective")
    records = [r for r in json.loads((args.dataset / "accepted.json").read_text())
               if r["split"] == "train" and r["frames_60hz"] >= 129]
    manifest = json.loads((args.labels / "manifest.json").read_text())
    if ({r["filename"] for r in records} != {r["filename"] for r in manifest["items"]}
            or any(r["split"] != "train" for r in manifest["items"])):
        raise ValueError("Labels must cover exactly the complete TRAIN split")
    clips = []
    for record in records:
        with np.load(args.dataset / record["shard"], allow_pickle=False) as data:
            raw, confidence = data["global_features"].copy(), data["contact_confidence"].copy()
        with np.load(args.labels / (record["filename"] + ".npz"), allow_pickle=False) as data:
            labels = {f"sole_{key}_{side}": torch.from_numpy(data[f"{key}_{side}"].copy())
                      for side in ("left", "right") for key in
                      ("frame_state", "interval_state", "region_measured", "vertex_measured")}
        clips.append((raw, confidence, labels))
    frozen_modules = {"root": inference._root_model, "pose": pose.backbone_net,
                      "encoder": pose.pose_net.encoder, "quantizer": pose.pose_net.quantizer}
    frozen = {name: state_digest(module) for name, module in frozen_modules.items()}
    pose.pose_net.root_condition_policy = "provided_velocity" if args.arm == "A" else "learned_refinement"
    trainer = BonesTokenizer(pose_vqvae_network=pose.pose_net, root_vqvae_network=None,
        motion_rep=rep, optimizer=None, scheduler=None, args=coefficients).cuda().eval()
    pose.pose_net.decoder.requires_grad_(True)
    parameters = list(pose.pose_net.decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=2e-5)
    args.output.mkdir(parents=True)
    files = ("scripts/train_soma_decoder_refinement.py", "motionbricks/data/soma_decoder_refinement.py",
             "motionbricks/data/bones_models.py", "motionbricks/data/bones_control.py",
             "motionbricks/data/sole_geometry_losses.py", "scripts/portable_bones_control.py")
    contract = {"schema": "motionbricks.decoder_refinement.v1", "arm": args.arm,
        "initial_bundle_sha256": file_sha256(args.bundle), "dataset_sha256": snapshot["sha256"],
        "labels_manifest_sha256": file_sha256(args.labels / "manifest.json"),
        "tokenizer_config_sha256": file_sha256(args.tokenizer_config), "train_clips": len(records),
        "seed": 2333, "steps": args.steps, "optimizer": "Adam", "learning_rate": 2e-5,
        "batch": 4, "primitives_per_update": 2, "frames_per_primitive": 64,
        "warmup_steps": 2500, "history_ramp_steps": 2500,
        "warmup": "source root and frozen encoder tokens; first four observed body frames",
        "later_inputs": "exact frozen runtime Root/Pose argmax; TRAIN endpoint root goals only",
        "history_policy": "generated_second_primitive" if args.arm == "C" else "teacher",
        "target_frame": "dataset future unchanged; same origin translation applied to context and target",
        "loss_arguments": coefficients, "frozen_state": frozen, "gpu_uuid": gpu,
        "root_condition_policy": pose.pose_net.root_condition_policy,
        "source_mechanism": "MotionBricks 4.2 native joint root/body decoder; DART D.2 scheduled history (adapted)",
        "not_reproduced": "not DART diffusion; two primitive decoder-only adaptation, not full Pose retraining",
        "test_used": False, "post_processing": None, "quality_approved": False,
        "source_sha256": {name: file_sha256(Path(name)) for name in files}}
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
    for name in files:
        destination = args.output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(name).read_bytes())
    sampling, history_counts = hashlib.sha256(), {"teacher": 0, "generated": 0}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(args.steps):
        chosen = rng.integers(len(clips), size=4)
        starts = [int(rng.integers(len(clips[index][0]) - 68)) for index in chosen]
        yaw = rng.uniform(0, 2 * np.pi, 4).astype(np.float32)
        draws = rng.random(4)
        sampling.update(np.asarray([*chosen, *starts], dtype=np.int64).tobytes())
        sampling.update(yaw.tobytes())
        sampling.update(draws.tobytes())
        raw = torch.from_numpy(np.stack([clips[i][0][s:s + 69] for i, s in zip(chosen, starts)])).cuda()
        with torch.no_grad():
            raw = rep.change_first_heading(raw, torch.from_numpy(yaw).cuda(), False, False)
        history = raw[:, :4].clone()
        curriculum_step = (0 if step == 0 else 9999) if args.smoke_test else step
        warmup = curriculum_step < 2500
        probability = rollout_probability(curriculum_step)
        use_generated = torch.from_numpy(draws < probability).cuda() if args.arm == "C" else torch.zeros(4, dtype=torch.bool, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        latest, objective = {}, 0.
        for primitive in range(2):
            offset = 4 * primitive
            source = raw[:, offset:offset + 65]
            observed = source[:, :4].clone() if primitive == 0 else torch.where(use_generated[:, None, None], history, source[:, :4])
            with torch.no_grad():
                global_target, local_target = source_target_with_history(source, observed, rep)
                if warmup:
                    request = warmup_request(pose, local_target)
                    generated = None
                else:
                    request, generated = deployed_request(inference, source, observed)
            labels = {}
            for row, (index, start) in enumerate(zip(chosen, starts)):
                sampled = source_label_window(clips[index][2], start + offset, 64,
                    context_is_generated=primitive > 0 and bool(use_generated[row]))
                for key, value in sampled.items():
                    labels.setdefault(key, []).append(value)
            labels = {key: torch.stack(value).cuda() for key, value in labels.items()}
            confidence = torch.from_numpy(np.stack([clips[i][1][s + offset:s + offset + 64]
                                                      for i, s in zip(chosen, starts)])).cuda().bool()
            if primitive:
                confidence[use_generated, :4] = False
            losses, decoded = refinement_loss(trainer, local_target, global_target, confidence, labels, request, history=observed)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Nonfinite native decoder objective")
            (losses["loss"] / 2).backward()
            objective += float(losses["loss"].detach()) / 2
            latest = {key: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                      for key, value in losses.items()}
            if generated is not None:
                history = generated
            if primitive:
                history_counts["generated"] += int(use_generated.sum())
                history_counts["teacher"] += int((~use_generated).sum())
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("Nonfinite decoder gradient")
        torch.nn.utils.clip_grad_norm_(parameters, .5)
        optimizer.step()
        if step == 0 or (step + 1) % 500 == 0:
            event = {"step": step + 1, "objective": objective, "rollout_probability": probability,
                     "history_counts": dict(history_counts), **latest}
            with (args.output / "training.jsonl").open("a") as log:
                log.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps({k: event[k] for k in ("step", "objective", "rollout_probability", "history_counts")}), flush=True)
        if (step + 1) % 2500 == 0:
            save_bundle(bundle, pose, args.output / f"step_{step + 1}.pt", contract, sampling.hexdigest(), step + 1)
    if {name: state_digest(module) for name, module in frozen_modules.items()} != frozen:
        raise RuntimeError("A frozen network/encoder/EMA changed")
    save_bundle(bundle, pose, args.output / "backend.pt", contract, sampling.hexdigest(), args.steps)
    torch.save({"steps": args.steps, "optimizer": optimizer.state_dict(), "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, args.output / "training_state.pt")
    result = {"steps": args.steps, "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2 ** 20, "sampling_sha256": sampling.hexdigest(),
        "history_counts": history_counts, "frozen_unchanged": True, "quality_approved": False,
        "bundle_sha256": file_sha256(args.output / "backend.pt")}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
