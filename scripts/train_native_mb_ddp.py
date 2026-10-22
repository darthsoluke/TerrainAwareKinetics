"""Torchrun/Lightning DDP adapter; unchanged native MB Pose model and objective."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_native_mb import sha256, source_lf_sha256, verify_dataset
from scripts.train_native_mb_control import SOURCE_FILES, state_digest, validate_resume, verify_tokenizer_binding
from scripts.export_native_mb_snapshot import write_json

DDP_SOURCE = "scripts/train_native_mb_ddp.py"


def distributed_environment(env=None, visible=None, *, allow_single_rank=False):
    env = os.environ if env is None else env
    world = int(env.get("WORLD_SIZE", "1"))
    rank = int(env.get("RANK", "-1"))
    local_rank = int(env.get("LOCAL_RANK", "-1"))
    local_world = int(env.get("LOCAL_WORLD_SIZE", str(world)))
    devices = (env.get("CUDA_VISIBLE_DEVICES", "") if visible is None else visible).split(",")
    if (world < (1 if allow_single_rank else 2) or local_world != world or not 0 <= rank < world or local_rank != rank
            or len(devices) != world or len(set(devices)) != world or any(not x.strip() for x in devices)):
        raise ValueError("Requires single-node torchrun with distinct visible GPUs and valid rank mapping")
    return {"world_size": world, "rank": rank, "local_rank": local_rank, "visible_devices": devices}


def ddp_batch_plan(profile, batch_size, world_size, global_batch_size=None):
    from motionbricks.data.native_mb_paper_protocol import PAPER_BATCH_PER_GPU

    if type(world_size) is not int or world_size < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch and world size must be positive integers")
    if profile == "paper":
        if batch_size != PAPER_BATCH_PER_GPU or global_batch_size is not None:
            raise ValueError("Paper profile requires batch256 per GPU, without a global override")
    elif profile == "fixed_global_batch":
        if (type(global_batch_size) is not int or global_batch_size != 256
                or global_batch_size % world_size or batch_size * world_size != global_batch_size):
            raise ValueError("Fixed global batch256 must split equally into the declared per-GPU batch")
    else:
        raise ValueError("Unknown DDP batch profile")
    return {"profile": profile, "per_gpu": batch_size, "global": batch_size * world_size,
            "gradient_accumulation": 1}


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


def native_ddp_loader(dataset, batch_size, workers, seed, *, rank=0, world_size=2):
    import torch
    from torch.utils.data import DataLoader, RandomSampler
    from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
    from lightning_fabric.utilities.seed import pl_worker_init_function
    from motionbricks.data.native_mb_data import native_training_loader, collate_native_batch

    original = native_training_loader(dataset, batch_size, workers, paper=True)
    sampler = RandomSampler(dataset, replacement=True, num_samples=len(original.sampler) * world_size,
                            generator=torch.Generator().manual_seed(seed))
    distributed = DistributedSamplerWrapper(sampler, num_replicas=world_size, rank=rank,
                                             shuffle=False, seed=seed, drop_last=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=distributed, num_workers=workers,
                      collate_fn=collate_native_batch, drop_last=True, pin_memory=True,
                      persistent_workers=workers > 0,
                      worker_init_fn=partial(pl_worker_init_function, rank=rank),
                      generator=torch.Generator().manual_seed(seed + rank))


def validate_ddp_resume(run, expected, code_root, runner_change_reason=None, *, allow_batch_change=False):
    previous = json.loads((Path(run) / "run_contract.json").read_text())
    strict_expected = dict(expected)
    batch_changed = previous.get("batch_size") != expected.get("batch_size")
    if allow_batch_change:
        if not runner_change_reason:
            raise ValueError("Batch transition requires an explicit reviewed reason")
        sampling = expected.get("sampling_contract") or {}
        distributed = expected.get("distributed_training") or {}
        ddp_batch_plan("fixed_global_batch", expected["batch_size"], distributed.get("world_size"),
                       sampling.get("global_input_batch_size"))
        strict_expected["batch_size"] = previous["batch_size"]
    binding = validate_resume(run, strict_expected, code_root)
    if previous.get("benchmark_only") and not expected.get("benchmark_only"):
        raise ValueError("Production cannot resume an isolated timing benchmark")
    old_world = (previous.get("distributed_training") or {}).get("world_size", 1)
    new_world = (expected.get("distributed_training") or {}).get("world_size", 1)
    if batch_changed or (allow_batch_change and old_world != new_world):
        binding["batch_transition"] = {"previous_per_gpu": previous["batch_size"],
            "current_per_gpu": expected["batch_size"], "previous_global": previous["batch_size"] * old_world,
            "current_global": expected["batch_size"] * new_world, "reason": runner_change_reason}
    if previous.get("distributed_training"):
        previous_hash = previous["source_lf_sha256"].get(DDP_SOURCE)
        current_hash = source_lf_sha256(Path(code_root) / DDP_SOURCE)
        if previous_hash != current_hash:
            if not previous_hash or not runner_change_reason:
                raise ValueError("DDP entrypoint changed during continuation")
            binding["ddp_entrypoint_revision"] = {"previous_sha256": previous_hash,
                "current_sha256": current_hash, "reviewed_change_reason": runner_change_reason}
    if previous.get("original_schedule_steps") != expected["original_schedule_steps"]:
        raise ValueError("DDP resume cannot change the learning-rate schedule")
    binding["sampling_note"] = "Full optimizer/scheduler resumed; workers reseeded; larger global batch after single-to-DDP switch."
    if allow_batch_change:
        binding["sampling_note"] = "Full optimizer/scheduler resumed; workers reseeded; audited fixed global batch256 continuation."
        binding["runtime_numerics_transition"] = {
            "previous": previous.get("runtime_numerics"), "current": expected.get("runtime_numerics"),
            "reason": runner_change_reason}
    return binding


def verify_rank_receipts(receipts, start_step, total_steps, world_size, tokenizer_digest):
    if len(receipts) != world_size or {x["rank"] for x in receipts} != set(range(world_size)):
        raise ValueError("Missing or duplicate DDP rank receipts")
    if len({x["batch_size"] for x in receipts}) != 1:
        raise ValueError("DDP ranks must use the same per-GPU batch")
    for receipt in receipts:
        restored = receipt.get("restored_training_state") or {}
        if (restored.get("global_step") != start_step or restored.get("max_adam_step") != start_step
                or restored.get("scheduler_last_epoch") != start_step
                or restored.get("schedule_steps") != 2_000_000):
            raise ValueError("DDP rank receipt does not prove restored optimizer/scheduler state")
        if (receipt["optimizer_updates"] != total_steps or receipt["start_step"] != start_step
                or receipt["updates_this_run"] != total_steps - start_step or receipt["skipped_batches"] != 0
                or receipt["tokenizer_state_sha256"] != tokenizer_digest
                or receipt["observed_input_batch_sizes"] != [receipt["batch_size"]]
                or receipt["sampler_rank"] != receipt["rank"]
                or receipt["sampler_replicas"] != world_size):
            raise ValueError("DDP rank update/batch/sampler/frozen-tokenizer contract failed")
    if len({x["backbone_state_sha256"] for x in receipts}) != 1:
        raise ValueError("DDP backbone tensors differ between ranks")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-run", type=Path, required=True)
    parser.add_argument("--resume-run", type=Path, required=True)
    parser.add_argument("--runner-change-reason", help="Audited DDP adapter-only revision; model/objective sources stay strict")
    parser.add_argument("--gpu-launch-report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--schedule-steps", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batch-profile", choices=("paper", "fixed_global_batch"), default="paper")
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--benchmark-only", action="store_true", help="Isolated timing fork, never a production checkpoint")
    parser.add_argument("--benchmark-matmul-precision", choices=("highest", "high", "medium"))
    parser.add_argument("--timing-warmup-updates", type=int, default=25)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--find-unused-parameters", action="store_true")
    args = parser.parse_args(argv)
    ranks = distributed_environment(allow_single_rank=args.benchmark_only)
    if args.output.exists():
        parser.error("DDP requires a fresh run directory before all ranks launch")
    from motionbricks.data.native_mb_paper_protocol import validate_paper_arguments
    batch_plan = ddp_batch_plan(args.batch_profile, args.batch_size, ranks["world_size"], args.global_batch_size)
    validate_paper_arguments(args.schedule_steps, 256)
    if args.benchmark_matmul_precision and not args.benchmark_only:
        parser.error("Matmul precision overrides are allowed only in isolated benchmarks")
    if not 1 <= args.steps <= args.schedule_steps or args.workers < 0 or args.timing_warmup_updates < 0:
        parser.error("Invalid update/worker budget")
    gpu_report = json.loads(args.gpu_launch_report.read_text())
    if [gpu["uuid"] for gpu in gpu_report["gpus"]] != ranks["visible_devices"]:
        raise ValueError("Visible GPUs differ from the launch availability audit")

    import numpy as np
    import pytorch_lightning as pl
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from motionbricks.data.native_mb_data import NativeMBMotionDataset, build_native_representation
    from motionbricks.data.native_mb_models import build_native_model
    from motionbricks.data.native_mb_paper_protocol import configure_paper_optimizer, paper_config, profile_notes
    from scripts.evaluate_native_mb_checkpoint import load_native_checkpoint

    torch.set_num_threads(4)
    pl.seed_everything(args.seed, workers=True)
    torch.cuda.set_device(ranks["local_rank"])
    dist.init_process_group(backend="nccl", init_method="env://")
    code_root = Path(__file__).resolve().parents[1]
    dataset_dir = args.dataset.resolve()
    manifest = json.loads((dataset_dir / "native_manifest.json").read_text())
    verify_dataset(dataset_dir, manifest)
    dataset_hash = sha256(dataset_dir / "native_manifest.json")
    tokenizer, tokenizer_info, _ = load_native_checkpoint(args.tokenizer_run)
    verify_tokenizer_binding(tokenizer_info, dataset_hash)
    config = paper_config("pose")
    configured_matmul = apply_native_matmul_precision(config, torch)
    if args.benchmark_matmul_precision:
        torch.set_float32_matmul_precision(args.benchmark_matmul_precision)
    rep = build_native_representation(dataset_dir / "stats")
    model = build_native_model("pose", rep, config, tokenizer=tokenizer)
    configure_paper_optimizer(model, config)
    model._vqvae_model_loaded = True
    support = model.supporting_nets["pose_net"]
    frozen_digest = state_digest(support)
    minimum_frames = config.model.args.max_tokens * (2 ** config.model.args.down_t) + 1
    dataset = NativeMBMotionDataset(dataset_dir, "train", max_frames=max(200, minimum_frames),
                                    min_frames=minimum_frames, cache_max_bytes=256 * 1024 * 1024)
    loader = native_ddp_loader(dataset, args.batch_size, args.workers, args.seed,
                               rank=ranks["rank"], world_size=ranks["world_size"])
    if ranks["rank"] == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(config, args.output / "native_config.yaml")
    dist.barrier()
    sources = (*SOURCE_FILES, "motionbricks/data/native_mb_paper_protocol.py", DDP_SOURCE)
    contract = {"stage": "native_pose", "status": "running", "seed": args.seed,
                "batch_size": args.batch_size, "dataset": str(dataset_dir),
                "dataset_manifest_sha256": dataset_hash, "config_sha256": sha256(args.output / "native_config.yaml"),
                "tokenizer": tokenizer_info, "tokenizer_state_sha256": frozen_digest,
                "stop_steps": args.steps, "original_schedule_steps": args.schedule_steps,
                "training_clips": len(dataset), "min_global_frames": minimum_frames, "fps": rep.fps,
                "model_class": type(model).__module__ + "." + type(model).__name__,
                "network_class": type(model.backbone_net).__module__ + "." + type(model.backbone_net).__name__,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "source_lf_sha256": {path: source_lf_sha256(code_root / path) for path in sources},
                "runtime_versions": {"torch": torch.__version__, "lightning": pl.__version__, "numpy": np.__version__},
                "original_training_step": True, "custom_losses": False, "quality_approved": False,
                "ue_deployment": False, "training_profile": profile_notes("pose"),
                "batch_profile": batch_plan, "benchmark_only": args.benchmark_only,
                "runtime_numerics": {"trainer_precision": config.trainer.precision,
                    "configured_matmul_precision": configured_matmul,
                    "effective_matmul_precision": torch.get_float32_matmul_precision(),
                    "benchmark_override": args.benchmark_matmul_precision,
                    "bf16": False, "official_precision_setting_applied": not args.benchmark_matmul_precision},
                "distributed_training": {"world_size": ranks["world_size"], "launcher": "torchrun",
                    "strategy": "ddp", "find_unused_parameters": args.find_unused_parameters,
                    "gpu_at_launch": gpu_report["gpus"], "gradient_reduction": "PyTorch DDP mean",
                    "lr_scaling": "none; unchanged official Adam and 2M schedule"},
                "sampling_contract": {"input_batch_size": args.batch_size,
                    "global_input_batch_size": args.batch_size * ranks["world_size"],
                    "official_augmentation_factor": model.args["batchsize_mul_factor"],
                    "augmented_window_batch_size": int(args.batch_size * model.args["batchsize_mul_factor"]),
                    "global_augmented_window_batch_size": int(args.batch_size * model.args["batchsize_mul_factor"] * ranks["world_size"]),
                    "sampler": "Lightning DistributedSamplerWrapper over torch replacement RandomSampler",
                    "epoch_steps_per_rank": len(loader), "invented_motion": False}}
    if args.batch_profile == "fixed_global_batch":
        contract["training_profile"]["explicit_deviations"].append(
            "User-authorized efficiency continuation: fixed global batch256, not paper batch256 per GPU")
    resume = validate_ddp_resume(args.resume_run, contract, code_root, args.runner_change_reason,
                                 allow_batch_change=args.batch_profile == "fixed_global_batch")
    contract.update(resume=resume, initialization="full_lightning_resume_single_to_ddp")
    if ranks["rank"] == 0:
        write_json(args.output / "run_contract.json", contract)
        shutil.copy2(Path(__file__), args.output / "train_native_mb_ddp.executed.py")
    dist.barrier()

    class TrainingChecks(pl.Callback):
        def __init__(self):
            self.actual_updates = 0
            self.skipped_batches = 0
            self.input_batch_sizes = set()
            self.last_time = time.monotonic()
            self.restored_training_state = None
            self.timed_start = None
            self.token_length_counts = {}

        def on_train_start(self, trainer, module):
            if (trainer.world_size != ranks["world_size"] or trainer.global_rank != ranks["rank"]
                    or str(module.device) != f"cuda:{ranks['local_rank']}"
                    or trainer.train_dataloader.sampler is not loader.sampler):
                raise RuntimeError("DDP rank/device/sampler mapping changed")
            optimizer_steps = [float(value["step"]) for value in trainer.optimizers[0].state.values() if "step" in value]
            schedule = trainer.lr_scheduler_configs[0].scheduler.state_dict()
            if (trainer.global_step != resume["saved_step"] or not optimizer_steps
                    or max(optimizer_steps) != resume["saved_step"]
                    or schedule["last_epoch"] != resume["saved_step"]
                    or schedule["num_training_steps"] != args.schedule_steps):
                raise RuntimeError("DDP failed to restore optimizer updates or the original learning-rate schedule")
            self.restored_training_state = {"global_step": trainer.global_step,
                "max_adam_step": max(optimizer_steps), "scheduler_last_epoch": schedule["last_epoch"],
                "schedule_steps": schedule["num_training_steps"]}
            print(f"DDP_RANK_READY rank={trainer.global_rank} world={trainer.world_size} device={module.device}", flush=True)

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            self.input_batch_sizes.add(int(batch["batch_size"]))
            if batch["batch_size"] != args.batch_size:
                raise RuntimeError("DDP input batch differs from official per-GPU batch")

        def on_before_optimizer_step(self, trainer, module, optimizer):
            gradients = [p.grad for p in module.backbone_net.parameters() if p.grad is not None]
            if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
                raise RuntimeError("Nonfinite or missing native Pose gradients")
            if support.training or any(p.requires_grad or p.grad is not None for p in support.parameters()):
                raise RuntimeError("Supporting tokenizer is not frozen")
            self.actual_updates += 1

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if outputs is None:
                self.skipped_batches += 1
            if outputs is not None and "num_tokens" in batch:
                length = int(batch["num_tokens"][0, 0])
                self.token_length_counts[length] = self.token_length_counts.get(length, 0) + 1
            if self.actual_updates == args.timing_warmup_updates:
                torch.cuda.synchronize()
                self.timed_start = time.monotonic()
            if outputs is not None and self.actual_updates % 100 == 0 and trainer.is_global_zero:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                now = time.monotonic()
                print(f"DDP_POSE_STEP {trainer.global_step} loss={float(loss):.6f} updates_per_second={100/(now-self.last_time):.3f}", flush=True)
                self.last_time = now

    checks = TrainingChecks()
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.output / "checkpoints",
        filename="native-{step}", every_n_train_steps=5000, save_top_k=1, save_last=True)
    strategy = pl.strategies.DDPStrategy(find_unused_parameters=args.find_unused_parameters,
                                        process_group_backend="nccl")
    trainer = pl.Trainer(max_steps=args.steps, devices=ranks["world_size"], accelerator="gpu", strategy=strategy,
        precision=config.trainer.precision, gradient_clip_val=config.trainer.gradient_clip_val,
        logger=pl.loggers.CSVLogger(str(args.output), name="logs", version=0), callbacks=[checks, checkpoint],
        log_every_n_steps=25, enable_progress_bar=False, num_sanity_val_steps=0, enable_model_summary=False,
        use_distributed_sampler=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.fit(model, train_dataloaders=loader, ckpt_path=resume["checkpoint"])
    torch.cuda.synchronize()
    finished = time.monotonic()
    elapsed = finished - started
    timed_updates = max(0, checks.actual_updates - args.timing_warmup_updates)
    timed_elapsed = finished - checks.timed_start if checks.timed_start is not None and timed_updates else None
    if not isinstance(trainer.optimizers[0], torch.optim.Adam):
        raise RuntimeError("DDP did not instantiate original paper Adam optimizer")
    receipt = {"rank": ranks["rank"], "optimizer_updates": trainer.global_step,
               "start_step": resume["saved_step"], "updates_this_run": checks.actual_updates,
               "skipped_batches": checks.skipped_batches, "batch_size": args.batch_size,
               "observed_input_batch_sizes": sorted(checks.input_batch_sizes),
               "sampler_rank": loader.sampler.rank, "sampler_replicas": loader.sampler.num_replicas,
               "elapsed_seconds": elapsed, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
               "tokenizer_state_sha256": state_digest(support), "backbone_state_sha256": state_digest(model.backbone_net)}
    receipt["restored_training_state"] = checks.restored_training_state
    receipt["runtime_numerics"] = contract["runtime_numerics"]
    receipt["token_length_counts"] = checks.token_length_counts
    receipt["timing"] = {"warmup_updates": args.timing_warmup_updates, "updates": timed_updates,
                         "elapsed_seconds": timed_elapsed}
    write_json(args.output / f"rank_{ranks['rank']}_result.json", receipt)
    dist.barrier()
    receipts = [json.loads((args.output / f"rank_{rank}_result.json").read_text()) for rank in range(ranks["world_size"])]
    verify_rank_receipts(receipts, resume["saved_step"], args.steps, ranks["world_size"], frozen_digest)
    final = args.output / "checkpoints/final.ckpt"
    trainer.save_checkpoint(final)
    dist.barrier()
    if trainer.is_global_zero:
        timing = {"warmup_updates": args.timing_warmup_updates, "updates": timed_updates,
                  "elapsed_seconds": max(x["timing"]["elapsed_seconds"] for x in receipts) if timed_elapsed else None}
        timing["updates_per_second"] = timed_updates / timing["elapsed_seconds"] if timing["elapsed_seconds"] else None
        timing["input_samples_per_second"] = timing["updates_per_second"] * batch_plan["global"] if timing["updates_per_second"] else None
        result = {"optimizer_updates": trainer.global_step, "updates_this_run": checks.actual_updates,
                  "skipped_batches": 0, "observed_input_batch_sizes": sorted(checks.input_batch_sizes),
                  "elapsed_seconds": max(x["elapsed_seconds"] for x in receipts),
                  "peak_gpu_memory_mb": max(x["peak_gpu_memory_mb"] for x in receipts),
                  "rank_receipts": receipts, "final_checkpoint_sha256": sha256(final),
                  "tokenizer_state_sha256": frozen_digest, "quality_approved": False,
                  "optimizer_class": "torch.optim.adam.Adam", "world_size": ranks["world_size"],
                  "batch_profile": batch_plan, "runtime_numerics": contract["runtime_numerics"],
                  "benchmark_only": args.benchmark_only, "timing": timing,
                  "last_metrics_scope": "Unchanged upstream logging semantics; printed step loss is rank0-local.",
                  "last_metrics": {key: float(value) for key, value in trainer.callback_metrics.items()
                                   if np.ndim(value.detach().cpu()) == 0}}
        write_json(args.output / "training_result.json", result)
        contract["status"] = "trained_not_reviewed"
        write_json(args.output / "run_contract.json", contract)
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
