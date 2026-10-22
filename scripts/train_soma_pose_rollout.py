"""Matched rollout trials; optional Root/Pose coupling keeps the tokenizer frozen."""

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
from motionbricks.data.bones_rollout import UE_PUBLISH_FRAMES as PUBLISH_FRAMES
from motionbricks.data.soma_decoder_refinement import (rollout_probability, source_label_window,
                                                      source_target_with_history)
from motionbricks.data.soma_pose_rollout import (deployed_pose_request, pose_rollout_loss,
                                              require_deployed_tokens_equal)
from motionbricks.data.soma_joint_rollout import deployed_reference, joint_rollout_loss
from scripts.portable_bones_control import load_bundle


def save_bundle(bundle, pose, output, contract, sampling, steps, root=None):
    adapted = copy.copy(bundle)
    adapted["pose_state"] = {key: value.detach().cpu() for key, value in pose.state_dict().items()}
    if root is not None:
        adapted["root_state"] = {key: value.detach().cpu() for key, value in root.state_dict().items()}
    key = "joint_rollout_adaptation" if contract.get("coupled_root_trial") else "pose_rollout_adaptation"
    adapted[key] = {**contract, "sampling_sha256": sampling, "completed_updates": steps}
    adapted["history_geometry"] = contract.get("history_geometry", "predicted_features")
    adapted["quality_approved"] = False
    torch.save(adapted, output)


def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset", "labels", "bundle", "tokenizer-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Explicit matched 1000-update trial, not a full training run")
    parser.add_argument("--supervision-policy", choices=("source_v2", "legacy_splice"), default="source_v2")
    parser.add_argument("--history-geometry", choices=("predicted_features", "executed_fk"), default="predicted_features")
    parser.add_argument("--primitives-per-update", type=int, default=4)
    parser.add_argument("--train-decoder", action="store_true",
                        help="Adapt the native decoder too; encoder/codebooks and root-output policy stay fixed")
    parser.add_argument("--coupled-root-trial", action="store_true",
                        help="Matched A=frozen Root vs B=optimized Root; both optimize Pose and decoded motion")
    args = parser.parse_args()
    if args.coupled_root_trial and args.arm not in ("A", "B"):
        raise ValueError("Coupled Root trial has only matched A/B arms")
    if args.train_decoder and (not args.coupled_root_trial or args.supervision_policy != "source_v2"):
        raise ValueError("Decoder adaptation requires the corrected coupled supervision")
    if args.history_geometry == "executed_fk" and (not args.coupled_root_trial or args.supervision_policy != "source_v2"):
        raise ValueError("Observed FK training requires the corrected coupled path")
    if PUBLISH_FRAMES != 8:
        raise ValueError("Training must match the current UE immutable-prefix contract")
    valid_budget = (args.steps == 10000 and not args.pilot and not args.smoke_test
                   or args.pilot and args.steps == 1000 and not args.smoke_test
                   or args.smoke_test and args.steps == 2 and not args.pilot)
    if args.output.exists() or not valid_budget or args.primitives_per_update < 2:
        raise ValueError("Fresh 10k run, explicit 1k pilot or two-update smoke required; at least two primitives")
    primitives = args.primitives_per_update
    source_frames = 65 + PUBLISH_FRAMES * (primitives - 1)
    teacher_steps = ramp_steps = 250 if args.pilot else 2500
    torch.set_num_threads(4)
    if args.coupled_root_trial:
        # CUDA convolution kernels must not switch TF32 precision with autograd.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(2333)
    rng = np.random.default_rng(2333)
    inference, bundle = load_bundle(args.bundle, "cuda:0")
    inference.history_geometry = args.history_geometry
    inference.eval()
    rep, pose = inference.motion_rep, inference._pose_model
    if getattr(inference, "goal_library", None) is not None or pose.pose_net.root_condition_policy != "provided_velocity":
        raise ValueError("Use the installed original, not a rejected decoder or reference-goal trial")
    snapshot = dataset_snapshot(args.dataset)
    require_same_snapshot(bundle["contracts"]["pose"]["dataset_snapshot"], snapshot)
    gpu = str(torch.cuda.get_device_properties(0).uuid)
    gpu = gpu if gpu.startswith("GPU-") else "GPU-" + gpu
    if gpu not in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","):
        raise ValueError("An explicitly checked idle GPU is required")
    source_config = OmegaConf.load(args.tokenizer_config)
    coefficients = OmegaConf.to_container(source_config.model.args, resolve=True)
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
    excluded = [records[i]["filename"] for i, clip in enumerate(clips) if len(clip[0]) < source_frames]
    clips = [clip for clip in clips if len(clip[0]) >= source_frames]
    if not clips:
        raise ValueError("No genuine training clips meet the rollout length; padding is forbidden")
    coefficients["rollout_commit_frames"] = PUBLISH_FRAMES
    coefficients["rollout_rotation_increment_coeff"] = 1.0 if args.supervision_policy == "source_v2" else 0.0
    train_root = args.coupled_root_trial and args.arm == "B"
    initial_root = state_digest(inference._root_model)
    initial_decoder = state_digest(pose.pose_net.decoder)
    frozen_modules = ({"encoder": pose.pose_net.encoder, "quantizer": pose.pose_net.quantizer}
                      if args.train_decoder else {"tokenizer": pose.pose_net})
    if not train_root:
        frozen_modules["root"] = inference._root_model
    frozen = {name: state_digest(module) for name, module in frozen_modules.items()}
    initial_pose = state_digest(pose.backbone_net)
    trainer = BonesTokenizer(pose_vqvae_network=pose.pose_net, root_vqvae_network=None,
        motion_rep=rep, optimizer=None, scheduler=None, args=coefficients).cuda().eval()
    pose.backbone_net.requires_grad_(True)
    parameters = list(pose.backbone_net.parameters())
    if args.train_decoder:
        pose.pose_net.decoder.requires_grad_(True)
        parameters += list(pose.pose_net.decoder.parameters())
    if train_root:
        inference._root_model.backbone_net.requires_grad_(True)
        parameters += list(inference._root_model.backbone_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=2e-5)
    args.output.mkdir(parents=True)
    files = ("scripts/train_soma_pose_rollout.py", "motionbricks/data/soma_pose_rollout.py",
             "motionbricks/data/soma_decoder_refinement.py", "motionbricks/data/bones_models.py",
             "motionbricks/data/bones_control.py", "motionbricks/data/sole_geometry_losses.py",
             "scripts/soma_ue_live_server.py", "motionbricks/data/bones_rollout.py",
             "motionbricks/data/soma_joint_rollout.py",
             "scripts/portable_bones_control.py",
             "motionbricks/data/soma_continuation.py", "motionbricks/vqvae/models/motion_vqvae.py",
             "motionbricks/motion_backbone/neural_modules/root_backbone.py",
             "motionbricks/motion_backbone/neural_modules/pose_backbone.py")
    contract = {"schema": "motionbricks.joint_rollout.v1" if args.coupled_root_trial else "motionbricks.pose_rollout.v1",
        "arm": args.arm, "coupled_root_trial": args.coupled_root_trial, "train_root": train_root,
        "train_decoder": args.train_decoder, "initial_decoder_state": initial_decoder,
        "initial_bundle_sha256": file_sha256(args.bundle), "dataset_sha256": snapshot["sha256"],
        "labels_manifest_sha256": file_sha256(args.labels / "manifest.json"),
        "tokenizer_config_sha256": file_sha256(args.tokenizer_config), "train_clips": len(clips),
        "excluded_short_clips": excluded, "supervision_policy": args.supervision_policy,
        "history_geometry": args.history_geometry,
        "source_frames_per_sample": source_frames, "executed_rollout_seconds": primitives * PUBLISH_FRAMES / rep.fps,
        "pilot": args.pilot,
        "seed": 2333, "steps": args.steps, "optimizer": "Adam", "learning_rate": 2e-5,
        "batch": 4, "primitives_per_update": primitives, "frames_per_primitive": 64,
        "commit_frames": PUBLISH_FRAMES, "history_frames": 4,
        "teacher_steps": teacher_steps, "history_ramp_steps": ramp_steps, "motion_ce_weight": 1.,
        "decoded_motion_weight": 1. if args.coupled_root_trial or args.arm == "C" else 0.,
        "history_policy": "teacher" if args.arm == "A" and not args.coupled_root_trial else "scheduled_generated",
        "inputs": "exact deployed native Root and Pose; source endpoint root goals only",
        "targets": ("source-only motion and derivatives; generated history is input only; actual executed boundary"
                    if args.supervision_policy == "source_v2" else "LEGACY mixed-history targets, known splice defect; comparison only"),
        "gradient_estimator": "ACTUAL deployed hard argmax forward; autograd softmax/codebook straight-through backward; biased",
        "kernel_equivalence": "active logits float32 allclose(atol=rtol=1e-4); record differences; actual deployed IDs are never substituted",
        "loss_arguments": coefficients, "frozen_state": frozen, "initial_pose_state": initial_pose,
        "initial_root_state": initial_root, "optimized_parameter_count": sum(p.numel() for p in parameters),
        "motion_fps": rep.fps, "gradient_clip_norm": .5,
        "comparison_variable": ("history observation geometry" if args.history_geometry == "executed_fk" else
                                "decoder optimizer membership" if args.train_decoder else
                                "source-only targets and executed-boundary supervision" if args.pilot else
                                "Root backbone frozen/optimized" if args.coupled_root_trial else "history/objective"),
        "gpu_uuid": gpu, "network_mode": "eval: deterministic dropout-free paired adaptation",
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "paired_attention_kernel": "unfused MHA and math SDPA" if args.coupled_root_trial else "default",
        "root_condition_policy": pose.pose_net.root_condition_policy,
        "source_mechanism": "MB native Root/Pose/decoder path and masked-token CE; DART-style generated histories; native decoded-motion objective",
        "not_reproduced": "not DART diffusion; straight-through token predictor objective is our adaptation",
        "test_used": False, "post_processing": None, "quality_approved": False,
        "source_sha256": {name: file_sha256(Path(name)) for name in files}}
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
    for name in files:
        destination = args.output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(name).read_bytes())
    sampling, counts = hashlib.sha256(), {"teacher": 0, "generated": 0}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(args.steps):
        chosen = rng.integers(len(clips), size=4)
        starts = [int(rng.integers(len(clips[index][0]) - source_frames + 1)) for index in chosen]
        yaw = rng.uniform(0, 2 * np.pi, 4).astype(np.float32)
        draws = rng.random((primitives - 1, 4))
        sampling.update(np.asarray([*chosen, *starts], dtype=np.int64).tobytes())
        sampling.update(yaw.tobytes())
        sampling.update(draws.tobytes())
        raw = torch.from_numpy(np.stack([clips[i][0][s:s + source_frames] for i, s in zip(chosen, starts)])).cuda()
        with torch.no_grad():
            raw = rep.change_first_heading(raw, torch.from_numpy(yaw).cuda(), False, False)
        history = raw[:, :4].clone()
        curriculum_step = (0 if step == 0 else 9999) if args.smoke_test else step
        probability = rollout_probability(curriculum_step, teacher_steps, ramp_steps)
        optimizer.zero_grad(set_to_none=True)
        objective, latest = 0., {}
        for primitive in range(primitives):
            offset = PUBLISH_FRAMES * primitive
            source = raw[:, offset:offset + 65]
            use_generated = (torch.from_numpy(draws[primitive - 1] < probability).cuda()
                if primitive and (args.coupled_root_trial or args.arm != "A")
                else torch.zeros(4, dtype=torch.bool, device="cuda"))
            observed = torch.where(use_generated[:, None, None], history, source[:, :4])
            with torch.no_grad():
                global_target, local_target = source_target_with_history(source, observed, rep,
                    legacy_splice=args.supervision_policy == "legacy_splice")
                _, source_local = source_target_with_history(source, source[:, :4], rep)
                if args.coupled_root_trial:
                    reference = deployed_reference(inference, source, observed, commit_frames=PUBLISH_FRAMES)
                    generated = reference["history"]
                else:
                    pose_request, decoder_request, generated, deployed_logits = deployed_pose_request(
                        inference, source, observed, commit_frames=PUBLISH_FRAMES)
            labels = {}
            for row, (index, start) in enumerate(zip(chosen, starts)):
                sampled = source_label_window(clips[index][2], start + offset, 64, bool(use_generated[row]))
                for key, value in sampled.items():
                    labels.setdefault(key, []).append(value)
            labels = {key: torch.stack(value).cuda() for key, value in labels.items()}
            confidence = torch.from_numpy(np.stack([clips[i][1][s + offset:s + offset + 64]
                                                      for i, s in zip(chosen, starts)])).cuda().bool()
            confidence[use_generated, :4] = False
            if args.coupled_root_trial:
                total, terms = joint_rollout_loss(inference, trainer, source, observed, reference,
                    source_local, local_target, global_target, confidence, labels, args.supervision_policy)
            else:
                total, terms, ids = pose_rollout_loss(pose, trainer, pose_request, decoder_request,
                    source_local, local_target, global_target, confidence, labels, contract["decoded_motion_weight"], deployed_logits,
                    history=observed if args.supervision_policy == "source_v2" else None)
                require_deployed_tokens_equal(ids, decoder_request)
            (total / primitives).backward()
            objective += float(total.detach()) / primitives
            latest = {key: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                      for key, value in terms.items()}
            history = generated.detach()
            if primitive:
                counts["generated"] += int(use_generated.sum())
                counts["teacher"] += int((~use_generated).sum())
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("Nonfinite Pose gradients")
        if args.coupled_root_trial:
            for name, module, required in (("root", inference._root_model.backbone_net, train_root),
                                           ("pose", pose.backbone_net, True),
                                           ("decoder", pose.pose_net.decoder, args.train_decoder)):
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])) if grads else torch.tensor(0.)
                if required and (not grads or norm.item() <= 0):
                    raise RuntimeError(f"Optimized {name} has no nonzero gradient")
                if not required and grads:
                    raise RuntimeError("Frozen Root received a gradient")
                latest[f"{name}_gradient_norm"] = float(norm)
        torch.nn.utils.clip_grad_norm_(parameters, .5)
        optimizer.step()
        if step == 0 or (step + 1) % 500 == 0:
            event = {"step": step + 1, "objective": objective, "rollout_probability": probability,
                     "history_counts": dict(counts), **latest}
            with (args.output / "training.jsonl").open("a") as log:
                log.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps({key: event[key] for key in ("step", "objective", "rollout_probability", "history_counts")}), flush=True)
        if (step + 1) % 2500 == 0:
            save_bundle(bundle, pose, args.output / f"step_{step + 1}.pt", contract, sampling.hexdigest(), step + 1,
                        root=inference._root_model if args.coupled_root_trial else None)
    if {name: state_digest(module) for name, module in frozen_modules.items()} != frozen:
        raise RuntimeError("Frozen Root or complete tokenizer/EMA changed")
    if state_digest(pose.backbone_net) == initial_pose:
        raise RuntimeError("Pose network did not change")
    root_changed = state_digest(inference._root_model) != initial_root
    if root_changed != train_root:
        raise RuntimeError("Root state change does not match the A/B contract")
    decoder_changed = state_digest(pose.pose_net.decoder) != initial_decoder
    if decoder_changed != args.train_decoder:
        raise RuntimeError("Decoder state change does not match the experiment")
    save_bundle(bundle, pose, args.output / "backend.pt", contract, sampling.hexdigest(), args.steps,
                root=inference._root_model if args.coupled_root_trial else None)
    torch.save({"steps": args.steps, "optimizer": optimizer.state_dict(), "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, args.output / "training_state.pt")
    result = {"steps": args.steps, "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2 ** 20, "sampling_sha256": sampling.hexdigest(),
        "history_counts": counts, "frozen_unchanged": True, "pose_changed": True, "quality_approved": False,
        "root_changed": root_changed, "coupled_root_trial": args.coupled_root_trial,
        "decoder_changed": decoder_changed,
        "bundle_sha256": file_sha256(args.output / "backend.pt")}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
