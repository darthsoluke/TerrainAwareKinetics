"""Bounded SOMA run of the shipped MB tokenizer, without custom model subclasses."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from motionbricks.data.native_mb_evaluation import full_precision_evaluation, verify_tokenizer_decode_parity


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_lf_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def apply_native_matmul_precision(config, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    precision = config.get("matmul_precision")
    if precision not in {"highest", "high", "medium"}:
        raise ValueError("Native config must declare a valid matmul_precision")
    torch_module.set_float32_matmul_precision(precision)
    if torch_module.get_float32_matmul_precision() != precision:
        raise RuntimeError("Native matmul precision was not applied")
    return precision


def runtime_numerics(config, configured_matmul_precision, torch_module):
    return {
        "trainer_precision": config.trainer.precision,
        "configured_matmul_precision": configured_matmul_precision,
        "effective_matmul_precision": torch_module.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch_module.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch_module.backends.cudnn.allow_tf32,
        "bf16": False,
        "official_precision_setting_applied": True,
        "precision_policy_note": (
            "Lightning precision=32; float32 matmuls use the configured policy, which may "
            "use TF32. This is not a strict IEEE FP32 claim."
        ),
    }


def resume_numerics_transition(previous_contract, current_numerics):
    previous = previous_contract.get("runtime_numerics")
    previous_precision = (previous or {}).get("effective_matmul_precision", "not_recorded")
    return {
        "status": "prior_not_recorded_current_explicit" if previous is None else "recorded_transition",
        "previous_runtime_numerics": previous,
        "previous_effective_matmul_precision": previous_precision,
        "current_runtime_numerics": current_numerics,
        "current_effective_matmul_precision": current_numerics["effective_matmul_precision"],
        "prior_value_inferred": False,
        "scope": "runner numerical policy only; model, objective, optimizer and schedule unchanged",
    }


def verify_resume_training_sources(previous, code_root):
    """Allow only line-ending differences in the model/data/objective source."""
    required = (
        "motionbricks/data/native_mb_data.py", "motionbricks/data/native_mb_models.py",
        "motionbricks/vqvae/models/motion_vqvae.py",
        "motionbricks/vqvae/neural_modules/vqvae.py",
        "motionbricks/helper/data_training_util.py",
    )
    if previous.get("training_profile", {}).get("training_profile", "").startswith("paper_"):
        required += ("motionbricks/data/native_mb_paper_protocol.py",)
    recorded = {key.replace("\\", "/"): value
                for key, value in previous.get("source_sha256", {}).items()}
    recorded_lf = {key.replace("\\", "/"): value
                   for key, value in previous.get("source_lf_sha256", {}).items()}
    verified = {}
    for relative in required:
        if relative not in recorded:
            raise ValueError(f"Native resume lacks training source identity: {relative}")
        path = Path(code_root) / relative
        raw, normalized = sha256(path), source_lf_sha256(path)
        if raw != recorded[relative] and normalized != recorded_lf.get(relative, recorded[relative]):
            raise ValueError(f"Native resume training source changed: {relative}")
        verified[relative] = {"sha256": raw, "lf_sha256": normalized,
                              "line_endings_only": raw != recorded[relative]}
    return verified


def verify_dataset(dataset_dir, manifest):
    artifacts = manifest.get("artifact_sha256", {})
    required = {"accepted.json", "stats/mean.npy", "stats/std.npy"}
    if not required.issubset(artifacts):
        raise ValueError("Dataset manifest must bind the accepted split list and normalization")
    records = json.loads((dataset_dir / "accepted.json").read_text())
    if not {r["shard"] for r in records}.issubset(artifacts):
        raise ValueError("Dataset manifest must bind every converted shard")
    for relative, expected in artifacts.items():
        path = (dataset_dir / relative).resolve()
        if dataset_dir.resolve() not in path.parents or sha256(path) != expected:
            raise ValueError(f"Dataset artifact mismatch: {relative}")


def verify_training_source_policy(dataset_dir, manifest, *, parc_only=False):
    if not parc_only and "parc_sources" not in manifest:
        return None
    from motionbricks.data.parc_training_sources import verify_parc_dataset_sources
    records = json.loads((dataset_dir / "accepted.json").read_text())
    return verify_parc_dataset_sources(dataset_dir, manifest, records)


def validate_native_resume(run, manifest_hash, config_hash, seed, batch_size, stop_steps):
    """Only resume a completed native run with its complete Lightning state."""
    run = Path(run).resolve()
    previous = json.loads((run / "run_contract.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    if previous.get("stage") != "native_tokenizer" or previous.get("status") != "completed":
        raise ValueError("Resume requires a completed native tokenizer run, not legacy weights")
    for key, expected in (("dataset_manifest_sha256", manifest_hash),
                          ("config_sha256", config_hash), ("seed", seed), ("batch_size", batch_size)):
        if previous.get(key) != expected:
            raise ValueError(f"Native resume contract mismatch: {key}")
    if stop_steps <= result["optimizer_updates"]:
        raise ValueError("Resume --steps is a total update target and must exceed the saved step")
    checkpoint = run / "checkpoints" / "final.ckpt"
    digest = sha256(checkpoint)
    if digest != result["final_checkpoint_sha256"]:
        raise ValueError("Native resume checkpoint hash mismatch")
    return {"run": str(run), "checkpoint": str(checkpoint), "sha256": digest,
            "saved_step": result["optimizer_updates"],
            "restore": "Lightning model, optimizer, scheduler and training loop state",
            "sampling_note": "RNGs reseeded; exact uninterrupted sample order is not guaranteed"}


def select_gpu(index):
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    for line in result.splitlines():
        number, uuid, memory, utilization = [value.strip() for value in line.split(",")]
        if int(number) == index:
            if int(memory) > 1024 or int(utilization) > 5:
                raise RuntimeError(f"GPU {index} is occupied: {memory} MiB, {utilization}%")
            os.environ["CUDA_VISIBLE_DEVICES"] = uuid
            return {"index": index, "uuid": uuid, "used_mib": int(memory),
                    "utilization_percent": int(utilization), "checked_at_unix": time.time()}
    raise ValueError(f"No GPU {index}")


@full_precision_evaluation()
def evaluate_tokenizer(model, dataset_dir, output, windows_per_clip=3):
    """Fixed held-out reconstruction, not closed-loop locomotion or quality approval."""
    import numpy as np
    import torch
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    device = next(model.parameters()).device
    rep, local_rep = model.motion_rep, model.local_motion_rep
    network = model.pose_net
    frames = model.args["max_tokens"] * model.get_num_frames_per_code()
    records = [row for row in json.loads((dataset_dir / "accepted.json").read_text())
               if row["split"] == "validation"]
    rows, examples, excluded = [], {}, []
    code_counts = torch.zeros(network.quantizer.num_heads, network.quantizer.nb_code_per_head,
                              dtype=torch.long, device=device)
    max_decode_error = 0.0
    roundoff_replays = []
    with torch.inference_mode():
        for record in records:
            with np.load(dataset_dir / record["shard"], allow_pickle=False) as data:
                features = torch.from_numpy(data["global_features"].copy()).to(device)
            if len(features) < frames + 1:
                excluded.append({"shard": record["shard"], "frames": len(features),
                                 "reason": "shorter than required global evaluation window"})
                continue
            starts = sorted(set(np.linspace(0, len(features) - frames - 1,
                                           windows_per_clip).astype(int).tolist()))
            for start in starts:
                raw = features[start:start + frames + 1][None]
                # Training also translates/reorients each crop before global->local.
                raw = rep.change_first_heading(raw, 0.0, False, False)
                global_motion = rep.normalize(raw)
                local = rep.dual_rep.global_to_local(
                    global_motion, True, True,
                    lengths=torch.tensor([frames + 1], device=device))[:, :frames]
                external = extract_feature_from_motion_rep(
                    local, local_rep, network.decoder_external_cond_feature_mode)
                gt = local_rep.inverse(local, True, return_all=True)
                indices = network.encode_into_idx(local, fetch_overall_indices=False)
                for head in range(len(code_counts)):
                    code_counts[head] += torch.bincount(indices[..., head].flatten(),
                                                       minlength=code_counts.shape[1])
                for conditioning in ("none", "context4"):
                    mask = torch.zeros(local.shape[:2], dtype=torch.bool, device=device)
                    if conditioning == "context4":
                        mask[:, :4] = True
                    target = local * mask[..., None] if mask.any() else None
                    has_target = mask if mask.any() else None
                    prediction = network(local, target_cond=target, has_target_cond=has_target,
                                         external_cond=external)["recon_state"]
                    decoded = network.forward_decoder(
                        indices, target_cond=target, has_target_cond=has_target,
                        external_cond=external, use_overall_indices=False)["recon_state"]
                    parity = verify_tokenizer_decode_parity(
                        network, local, indices, prediction, decoded,
                        target_cond=target, has_target_cond=has_target, external_cond=external)
                    error = parity["raw_max_error"]
                    max_decode_error = max(max_decode_error, error)
                    if parity["roundoff_replay_required"]:
                        roundoff_replays.append({"shard": record["shard"], "start": start,
                                                 "conditioning": conditioning, **parity})
                    pred = local_rep.inverse(prediction, True, return_all=True)
                    p, g = pred["posed_joints"][:, 4:], gt["posed_joints"][:, 4:]
                    root = rep.skeleton.root_idx
                    metrics = {
                        "mpjpe_m": float((p - g).norm(dim=-1).mean()),
                        "root_error_m": float((p[:, :, root] - g[:, :, root]).norm(dim=-1).mean()),
                        "joint_velocity_error_mps": float(((p[:, 1:] - p[:, :-1]) -
                            (g[:, 1:] - g[:, :-1])).norm(dim=-1).mean() * rep.fps),
                        "normalized_smooth_l1": float(torch.nn.functional.smooth_l1_loss(
                            prediction[:, 4:], local[:, 4:])),
                    }
                    row = {"shard": record["shard"], "behavior": record.get("behavior", "unknown"),
                           "start": start, "conditioning": conditioning, **metrics}
                    rows.append(row)
                    key = row["behavior"]
                    if conditioning == "context4" and key not in examples:
                        examples[key] = row
                        np.savez_compressed(output / f"{key}_reconstruction.npz",
                            prediction=pred["posed_joints"][0].cpu().numpy(),
                            source=gt["posed_joints"][0].cpu().numpy(), fps=np.array(rep.fps))
    if not rows:
        raise ValueError("No held-out windows")
    summaries = {}
    for condition in ("none", "context4"):
        selected = [row for row in rows if row["conditioning"] == condition]
        summaries[condition] = {name: float(np.mean([row[name] for row in selected]))
                               for name in metrics}
        summaries[condition]["windows"] = len(selected)
    result = {"task": "source-encoded tokenizer reconstruction; not WASD generation",
              "token_decode_parity_policy": "raw 1e-5; above this requires exact token identity and unchanged decoder replays of proven official STE roundoff",
              "roundoff_replays": roundoff_replays,
              "evaluation_precision": "float32; CUDA matmul and cuDNN TF32 disabled during evaluation only",
              "quality_approved": False, "fps": rep.fps,
              "frame_scoring": [4, frames], "interval_scoring": [4, frames - 1],
              "postprocessing": "none", "root_overwrite": False,
              "token_decode_max_feature_error": max_decode_error,
              "active_codes_per_head": (code_counts > 0).sum(-1).cpu().tolist(),
              "summary": summaries, "windows": rows, "examples": examples,
              "excluded_validation_clips": excluded}
    write_json(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--schedule-steps", type=int, default=2000001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--training-profile", choices=("release", "paper"), default="release")
    parser.add_argument("--parc-only", action="store_true",
        help="Require both audited PARC origins; reject cross-dataset initialization")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume-run", type=Path,
        help="Completed native run to resume; --steps is the total update target")
    initialization.add_argument("--warmstart-run", type=Path,
        help="Verified 50k same-stage run for weights-only transfer; --steps are additional updates")
    initialization.add_argument("--continue-dataset-run", type=Path,
        help="Verified 50k tokenizer with full optimizer/curriculum state on new data; --steps is total")
    args = parser.parse_args()
    if args.parc_only and (args.warmstart_run or args.continue_dataset_run):
        parser.error("PARC-only training cannot import a non-PARC dataset checkpoint")
    if (args.warmstart_run or args.continue_dataset_run) and args.training_profile != "paper":
        parser.error("Cross-dataset initialization requires --training-profile paper")
    if args.training_profile == "paper":
        from motionbricks.data.native_mb_paper_protocol import validate_paper_arguments
        validate_paper_arguments(args.schedule_steps, args.batch_size)
    if args.steps < 1 or args.schedule_steps < args.steps or args.batch_size < 1:
        parser.error("Require 1 <= steps <= schedule-steps and batch-size > 0")
    if args.output.exists():
        parser.error("Use a new output directory; existing runs are never overwritten")
    gpu = select_gpu(args.gpu)

    # Select an idle physical GPU before importing project modules with CUDA state.
    import numpy as np
    import pytorch_lightning as pl
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from motionbricks.data.native_mb_data import (
        NativeMBMotionDataset, build_native_representation, collate_native_batch,
    )
    from motionbricks.data.native_mb_models import native_config, build_native_model
    from motionbricks.data.native_mb_warmstart import (
        load_warmstart_weights, validate_warmstart_run, validate_dataset_continuation,
    )
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    torch.set_num_threads(4)
    pl.seed_everything(args.seed, workers=True)
    dataset_dir = args.dataset.resolve()
    manifest_path = dataset_dir / "native_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_dataset(dataset_dir, manifest)
    parc_sources = verify_training_source_policy(dataset_dir, manifest, parc_only=args.parc_only)
    rep = build_native_representation(dataset_dir / "stats")
    profile = None
    if args.training_profile == "paper":
        from motionbricks.data.native_mb_paper_protocol import (
            configure_paper_optimizer, paper_config, profile_notes,
        )
        config = paper_config("tokenizer")
        profile = profile_notes("tokenizer", parc_only=parc_sources is not None)
    else:
        config = native_config("tokenizer", args.schedule_steps)
    configured_matmul_precision = apply_native_matmul_precision(config, torch)
    numerics = runtime_numerics(config, configured_matmul_precision, torch)
    model = build_native_model("tokenizer", rep, config)
    if profile is not None:
        configure_paper_optimizer(model, config)
    dimensions = {}
    dummy = torch.zeros(1, 1, len(model.local_motion_rep.indices["all"]))
    for field, mode in zip(("encoder_state_dim", "decoder_state_dim", "decoder_target_cond_dim",
                            "decoder_external_cond_dim"), config.model.pose_vqvae_network.feature_mode):
        dimensions[field] = extract_feature_from_motion_rep(dummy, model.local_motion_rep, mode).shape[-1]
        config.model.pose_vqvae_network[field] = dimensions[field]
    from motionbricks.data.native_mb_data import native_training_loader
    is_paper = args.training_profile == "paper"
    dataset = NativeMBMotionDataset(dataset_dir, "train", max_frames=200, min_frames=65,
                                    cache_max_bytes=256 * 1024 * 1024 if is_paper else 0)
    if parc_sources is not None:
        from motionbricks.data.parc_training_sources import require_parc_training_origins
        parc_sources["actual_loader_origin_counts"] = require_parc_training_origins(dataset.records)
    records = json.loads((dataset_dir / "accepted.json").read_text())
    if not any(r["split"] == "validation" and r["frames_30hz"] >= 65 for r in records):
        raise ValueError("No validation clips satisfy the 65-frame contract")
    loader = native_training_loader(dataset, args.batch_size, args.workers, paper=is_paper)
    args.output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, args.output / "native_config.yaml")
    resume = validate_native_resume(args.resume_run, sha256(manifest_path),
        sha256(args.output / "native_config.yaml"), args.seed, args.batch_size, args.steps
    ) if args.resume_run else None
    if args.continue_dataset_run:
        resume = validate_dataset_continuation(args.continue_dataset_run, dataset_dir, config,
            seed=args.seed, batch_size=args.batch_size, stop_steps=args.steps)
    warmstart = validate_warmstart_run(args.warmstart_run, "tokenizer", dataset_dir, config) \
        if args.warmstart_run else None
    if warmstart:
        checkpoint_state = warmstart.pop("_checkpoint_state")
        load_warmstart_weights(model, checkpoint_state, "tokenizer", torch)
        del checkpoint_state
        warmstart.update({
            "new_updates_target": args.steps,
            "projected_inherited_plus_new_updates": warmstart["baseline_optimizer_updates"] + args.steps,
            "fresh_optimizer": OmegaConf.to_container(config.model.optimizer, resolve=True),
            "fresh_scheduler": OmegaConf.to_container(config.model.scheduler, resolve=True),
        })
    code_root = Path(__file__).resolve().parents[1]
    numerical_transition = None
    if resume:
        previous = json.loads((Path(resume["run"]) / "run_contract.json").read_text())
        resume["source_verification"] = verify_resume_training_sources(previous, code_root)
        numerical_transition = resume_numerics_transition(previous, numerics)
    sources = [Path(__file__), code_root / "motionbricks/data/native_mb_data.py",
               code_root / "motionbricks/data/native_mb_models.py",
               code_root / "motionbricks/vqvae/models/motion_vqvae.py",
               code_root / "motionbricks/vqvae/neural_modules/vqvae.py",
               code_root / "motionbricks/helper/data_training_util.py"]
    if profile is not None:
        sources.append(code_root / "motionbricks/data/native_mb_paper_protocol.py")
    if args.continue_dataset_run or warmstart:
        sources.append(code_root / 'motionbricks/data/native_mb_warmstart.py')
    contract = {"stage": "native_tokenizer", "status": "running", "seed": args.seed,
        "dataset": str(dataset_dir), "dataset_manifest_sha256": sha256(manifest_path),
        "initialization": ("native_state_continuation_new_dataset" if args.continue_dataset_run else
                           "native_checkpoint_resume" if resume else
                           "verified_weights_only_warmstart" if warmstart else
                           "from_scratch; no legacy checkpoints"),
        "resume": resume,
        "warmstart": warmstart,
        "stop_steps": args.steps, "original_schedule_steps": args.schedule_steps,
        "batch_size": args.batch_size, "batchsize_mul_factor": model.args["batchsize_mul_factor"],
        "training_clips": len(dataset), "fps": rep.fps, "gpu_at_launch": gpu,
        "runtime_feature_dimensions": dimensions,
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "network_class": type(model.pose_net).__module__ + "." + type(model.pose_net).__name__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "config_sha256": sha256(args.output / "native_config.yaml"),
        "source_sha256": {str(p.relative_to(code_root)): sha256(p) for p in sources},
        "source_lf_sha256": {str(p.relative_to(code_root)): source_lf_sha256(p) for p in sources},
        "runtime_versions": {"torch": torch.__version__, "lightning": pl.__version__,
                             "numpy": np.__version__},
        "runtime_numerics": numerics,
        "resume_numerics_transition": numerical_transition,
        "evaluation_source_lf_sha256": source_lf_sha256(code_root / "motionbricks/data/native_mb_evaluation.py"),
        "adaptations": ["SOMA30 skeleton and feature dimensions",
                        "train-split normalization statistics bound by the dataset manifest",
                        "60Hz source downsampled offline to official 30Hz temporal contract",
                        "fixed source ground: floor_estimation disabled",
                        f"single GPU with declared input batch size {args.batch_size}; bounded stop; "
                        "original curriculum not accelerated"],
        "excluded": ["custom region losses", "root velocity overwrite", "rollout losses",
                     "IK", "output smoothing", "old trained weights"],
        "not_full_paper_reproduction": True, "ue_deployment": False,
        "dataset_provenance": manifest}
    contract["parc_only_sources"] = parc_sources
    if parc_sources is not None:
        sources.append(code_root / "motionbricks/data/parc_training_sources.py")
        contract["source_sha256"] = {str(p.relative_to(code_root)): sha256(p) for p in sources}
        contract["source_lf_sha256"] = {str(p.relative_to(code_root)): source_lf_sha256(p) for p in sources}
        contract["adaptations"][2:4] = ["PARC paired motion already at native 30 Hz; no resampling",
            "source terrain retained; no floor estimator or pose projection"]
    contract["training_profile"] = profile or {"name": "released_preview_optimizer_and_schedule"}
    if is_paper:
        contract["sampling_contract"] = {"input_batch_size": args.batch_size,
            "official_augmentation_factor": model.args["batchsize_mul_factor"],
            "augmented_window_batch_size": int(args.batch_size * model.args["batchsize_mul_factor"]),
            "sampler": "torch RandomSampler replacement; independent real recording crops; drop_last",
            "cache_cpu_bytes_per_worker": dataset.cache_max_bytes,
            "epoch_steps": len(loader), "invented_motion": False}
    write_json(args.output / "run_contract.json", contract)

    class FiniteTraining(pl.Callback):
        def __init__(self):
            self.actual_updates = 0
            self.skipped_batches = 0
            self.input_batch_sizes = set()
            self.initial_state_checked = False

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if not self.initial_state_checked and args.continue_dataset_run:
                optimizer = trainer.optimizers[0]
                actual_lr = optimizer.param_groups[0]['lr']
                actual_epoch = trainer.lr_scheduler_configs[0].scheduler.last_epoch
                if (trainer.global_step != resume['saved_step']
                        or abs(actual_lr - resume['expected_initial_lr']) > 1e-12
                        or len(optimizer.state) != resume['expected_optimizer_state_entries']
                        or actual_epoch != resume['expected_scheduler_last_epoch']):
                    raise RuntimeError('Original optimizer/scheduler/curriculum state was not restored')
                pose_prob, _ = module._construct_keyframe_prob()
                write_json(args.output / 'initial_state_verified.json', dict(
                    global_step=trainer.global_step, lr=actual_lr, scheduler_last_epoch=actual_epoch,
                    optimizer_state_entries=len(optimizer.state),
                    decoder_keyframe_counts=np.flatnonzero(pose_prob).tolist(),
                    source_checkpoint_sha256=resume['sha256']))
                self.initial_state_checked = True
            self.input_batch_sizes.add(int(batch["batch_size"]))
            if is_paper and batch["batch_size"] != args.batch_size:
                raise RuntimeError("Paper training input batch is smaller than the declared per-GPU batch")

        def on_before_optimizer_step(self, trainer, module, optimizer):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()) if gradients else False
            if not finite:
                raise RuntimeError("Missing or nonfinite native training gradients")
            self.actual_updates += 1

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if outputs is None:
                self.skipped_batches += 1
                return
            if trainer.global_step % 500 == 0:
                value = outputs["loss"] if isinstance(outputs, dict) else outputs
                print(f"NATIVE_STEP {trainer.global_step} loss={float(value):.6f}", flush=True)

    checks = FiniteTraining()
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.output / "checkpoints",
        filename="native-{step}", every_n_train_steps=5000, save_top_k=1, save_last=True)
    trainer = pl.Trainer(max_steps=args.steps, devices=1, accelerator="gpu", strategy="auto",
        precision=config.trainer.precision, gradient_clip_val=config.trainer.gradient_clip_val,
        logger=pl.loggers.CSVLogger(str(args.output), name="logs"),
        callbacks=[checks, checkpoint], log_every_n_steps=25,
        enable_progress_bar=False, num_sanity_val_steps=0, enable_model_summary=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.fit(model, train_dataloaders=loader, ckpt_path=resume["checkpoint"] if resume else None)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    expected_updates = trainer.global_step - (resume["saved_step"] if resume else 0)
    if is_paper and (checks.actual_updates != expected_updates or checks.skipped_batches):
        raise RuntimeError("Paper optimizer updates must be real updates with no skipped batches")
    final_checkpoint = args.output / "checkpoints" / "final.ckpt"
    trainer.save_checkpoint(final_checkpoint)
    result = {"optimizer_updates": trainer.global_step,
        "updates_this_run": checks.actual_updates,
        "skipped_batches": checks.skipped_batches,
        "observed_input_batch_sizes": sorted(checks.input_batch_sizes),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "final_checkpoint_sha256": sha256(final_checkpoint),
        "runtime_numerics": numerics,
        "resume_numerics_transition": numerical_transition,
        "last_metrics": {key: float(value) for key, value in trainer.callback_metrics.items()
                         if np.ndim(value.cpu()) == 0}, "quality_approved": False}
    result["optimizer_class"] = type(trainer.optimizers[0]).__module__ + "." + type(trainer.optimizers[0]).__name__
    if profile is not None and not isinstance(trainer.optimizers[0], torch.optim.Adam):
        raise RuntimeError("Paper profile did not instantiate the declared Adam optimizer")
    write_json(args.output / "training_result.json", result)
    evaluation = evaluate_tokenizer(model, dataset_dir, args.output / "validation")
    contract["status"] = "completed"
    contract["validation_summary"] = evaluation["summary"]
    write_json(args.output / "run_contract.json", contract)
    print(json.dumps({**result, "validation": evaluation["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
