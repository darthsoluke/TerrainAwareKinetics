"""Bounded Root/Pose training using the official MB networks on real SOMA clips."""

import json
import os
import shutil
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256, require_same_snapshot
from motionbricks.data.bones_control import BonesControlRootModel, BonesPoseModel, load_tokenizer, state_digest
from motionbricks.data.bones_seed import BonesMotionDataset, build_representation, collate_bones_batch, validation_window_records


def build_control_model(stage, config, representation, tokenizer_run=None, expected_sha256=None, tokenizer_state=None):
    optimizer = instantiate(config.model.optimizer)
    scheduler = instantiate(config.model.scheduler)
    args = OmegaConf.to_container(config.model.args, resolve=True)
    network = instantiate(config.model.backbone_network, motion_rep=representation, _recursive_=False)
    if stage == "root":
        return BonesControlRootModel(pose_vqvae_network=None, root_vqvae_network=None,
                                     backbone_network=network, motion_rep=representation,
                                     optimizer=optimizer, scheduler=scheduler, args=args)
    if tokenizer_state is None:
        tokenizer, _ = load_tokenizer(tokenizer_run, representation, expected_sha256)
    else:
        tokenizer = instantiate(config.model.pose_vqvae_network, motion_rep=representation.dual_rep.local_motion_rep)
        tokenizer.load_state_dict(tokenizer_state, strict=True)
        tokenizer.requires_grad_(False).eval()
    pose_class = BonesPoseModel
    if args.get("history_policy"):
        from motionbricks.data.bones_history import BonesHistoryPoseModel
        pose_class = BonesHistoryPoseModel
    return pose_class(pose_vqvae_network=tokenizer, root_vqvae_network=None,
                          backbone_network=network, motion_rep=representation,
                          optimizer=optimizer, scheduler=scheduler, args=args)


def validate_control_protocol(args, snapshot, protocol):
    bounds = {"root": 10000, "pose": 20000}
    if (args.stage not in bounds or not protocol
            or protocol.get("experiment_kind") != "bounded_soma_control"
            or protocol.get("stage") != args.stage or not 1 <= args.max_steps <= bounds[args.stage]
            or protocol.get("max_steps") != args.max_steps
            or protocol.get("dataset_snapshot_sha256") != snapshot["sha256"]
            or protocol.get("batch_size") != args.batch_size
            or not args.tokenizer_run):
        raise ValueError("Control training needs its frozen bounded protocol and tokenizer identity")
    for name in ("scheduler_warmup_steps", "validation_interval", "validation_windows_per_clip"):
        if protocol.get(name) != getattr(args, name):
            raise ValueError("Control protocol differs: " + name)
    checkpoint = args.tokenizer_run / "checkpoints/final.ckpt"
    if file_sha256(checkpoint) != protocol.get("tokenizer_sha256"):
        raise ValueError("Frozen tokenizer checkpoint differs")
    contract = json.loads((args.tokenizer_run / "run_contract.json").read_text())
    require_same_snapshot(contract["dataset_snapshot"], snapshot)
    if file_sha256(args.tokenizer_run / "upstream_config_overrides.yaml") != contract["config_sha256"]:
        raise ValueError("Tokenizer configuration changed")
    if args.history_policy:
        if (args.stage != "pose" or not args.control_init_run or not args.history_root_run
                or not args.history_generator_pose_run or args.max_steps != 10000
                or protocol.get("history_policy") != args.history_policy
                or protocol.get("root_checkpoint_sha256") != file_sha256(args.history_root_run / "checkpoints/final.ckpt")
                or protocol.get("generator_pose_checkpoint_sha256") != file_sha256(
                    args.history_generator_pose_run / "checkpoints/final.ckpt")):
            raise ValueError("History learning requires its paired checkpoint identities and budget")


def run_control_training(args):
    snapshot = dataset_snapshot(args.dataset)
    protocol = json.loads(args.protocol.read_text()) if args.protocol else None
    validate_control_protocol(args, snapshot, protocol)
    report = json.loads((args.dataset / "quality_report.json").read_text())
    representation = build_representation(args.dataset / snapshot["stats_folder"])
    torch.set_num_threads(4)
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not actual_uuid.startswith("GPU-"):
        actual_uuid = "GPU-" + actual_uuid
    if actual_uuid not in os.environ["CUDA_VISIBLE_DEVICES"].split(","):
        raise RuntimeError("Selected CUDA device differs from the idle GPU check")
    pl.seed_everything(2333, workers=True)
    project = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(project / f"out/motionbricks_{args.stage}/version_1/hparams.yaml")
    source = OmegaConf.load(args.tokenizer_run / "upstream_config_overrides.yaml")
    with open_dict(config):
        config.data = {"text_embeddings": None}
        config.model.args.min_tokens = 6
        config.model.args.max_tokens = 32
        config.model.args.batchsize_mul_factor = 1
        config.model.args.floor_estimation = None
        config.model.args.keyframe_num_warmup_steps = 100
        config.model.scheduler.num_warmup_steps = args.scheduler_warmup_steps
        config.model.scheduler.num_training_steps = args.max_steps
        config.model.optimizer._target_ = "torch.optim.Adam"
        config.model.optimizer.lr = 5e-5
        config.model.scheduler.final_lr = 2e-6
        if args.stage == "pose":
            config.model.pose_vqvae_network = source.model.pose_vqvae_network
            config.model.backbone_network.args.n_embd = 512
            config.model.backbone_network.args.n_head = 8
            config.model.backbone_network.args.n_layers = 8
            config.model.args.vqvae_model_ckpt_path = None
        else:
            config.model.args.prob_provide_num_tokens = 1.0
        if args.history_policy:
            config.model.args.history_policy = args.history_policy
            config.model.args.history_root_run = str(args.history_root_run.resolve())
            config.model.args.history_generator_pose_run = str(args.history_generator_pose_run.resolve())
            config.model.args.history_ramp_steps = 5000
            config.model.args.history_max_probability = 0.5
    model = build_control_model(args.stage, config, representation, args.tokenizer_run,
                                protocol["tokenizer_sha256"])
    if args.control_init_run:
        previous = json.loads((args.control_init_run / "run_contract.json").read_text())
        require_same_snapshot(previous["dataset_snapshot"], snapshot)
        if previous["stage"] != args.stage or previous["tokenizer_sha256"] != protocol["tokenizer_sha256"]:
            raise ValueError("Initial control model uses different stage/tokenizer")
        initial = args.control_init_run / "checkpoints/final.ckpt"
        if protocol.get("initial_checkpoint_sha256") != file_sha256(initial):
            raise ValueError("Initial control checkpoint differs from protocol")
        model.load_state_dict(torch.load(initial, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    initial_digest = state_digest(model)
    records = json.loads((args.dataset / "accepted.json").read_text())
    datasets, loaders = {}, {}
    for split in ("train", "validation"):
        selected = [record for record in records if record["split"] == split and record["frames_60hz"] >= 129]
        if split == "validation":
            selected = validation_window_records(selected, 128, args.validation_windows_per_clip)
        datasets[split] = BonesMotionDataset(args.dataset, selected, representation.stats,
                                             crop_frames=360 if split == "train" else 129,
                                             training=split == "train")
        loaders[split] = DataLoader(datasets[split], batch_size=args.batch_size, shuffle=split == "train",
                                    num_workers=args.workers, collate_fn=collate_bones_batch,
                                    persistent_workers=args.workers > 0, pin_memory=True)
    if not all(len(dataset) for dataset in datasets.values()):
        raise ValueError("Empty control dataset")
    policy_files = [Path(__file__), project / "scripts/train_bones_locomotion.py",
                    project / "motionbricks/data/bones_control.py", project / "motionbricks/data/bones_models.py",
                    project / "motionbricks/data/bones_seed.py", project / "motionbricks/helper/data_training_util.py",
                    project / "motionbricks/motion_backbone/models/pose_model.py",
                    project / "motionbricks/motion_backbone/models/root_model.py",
                    project / "motionbricks/motion_backbone/inference/motion_inference.py"]
    hashes = {path.relative_to(project).as_posix(): file_sha256(path) for path in policy_files}
    if args.history_policy:
        for name in ("motionbricks/data/bones_history.py", "motionbricks/data/bones_rollout.py"):
            path = project / name
            policy_files.append(path)
            hashes[name] = file_sha256(path)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        previous_path = args.resume.resolve().parent.parent / "run_contract.json"
        previous = json.loads(previous_path.read_text())
        require_same_snapshot(previous["dataset_snapshot"], snapshot)
        if (previous_path.parent != args.output.resolve()
                or previous["stage"] != args.stage
                or previous["training_source_sha256"] != hashes
                or previous["protocol_sha256"] != file_sha256(args.protocol)
                or previous["tokenizer_sha256"] != protocol["tokenizer_sha256"]
                or previous["config_sha256"] != file_sha256(args.output / "upstream_config_overrides.yaml")):
            raise ValueError("Resume requires the unchanged original control run")
        if OmegaConf.to_container(config, resolve=False) != OmegaConf.to_container(
                OmegaConf.load(args.output / "upstream_config_overrides.yaml"), resolve=False):
            raise ValueError("Resume configuration differs")
        initial_digest = previous["initial_weights_sha256"]
    OmegaConf.save(config, args.output / "upstream_config_overrides.yaml")
    contract = {"schema": "motionbricks.soma_control.v1", "stage": args.stage,
                "experiment_scope": "bounded_soma_control", "dataset_snapshot": snapshot,
                "tokenizer_run": str(args.tokenizer_run.resolve()), "tokenizer_sha256": protocol["tokenizer_sha256"],
                "initial_weights_sha256": initial_digest, "seed": 2333,
                "config_sha256": file_sha256(args.output / "upstream_config_overrides.yaml"),
                "training_source_sha256": hashes, "protocol_sha256": file_sha256(args.protocol),
                "train_clips": len(datasets["train"]), "validation_windows": len(datasets["validation"]),
                "full_training_gate_passed": report["full_training_gate_passed"], "quality_approved": False,
                "body_conditions": "four actually observed frames; no future body constraints",
                "pose_input_tokens": "all MASK, including context; never source-encoded history tokens",
                "pose_target_tokens": "all positions, including the context token that affects future decoding",
                "root_conditions": "observed root/history velocity and requested endpoint root/velocity",
                "pose_training_root": "source trajectory (teacher baseline)" if args.stage == "pose" else None,
                "history_policy": args.history_policy or "teacher",
                "history_ramp_steps": 5000 if args.history_policy else None,
                "history_max_probability": 0.5 if args.history_policy else None,
                "future_height_condition": "carried from history, never source future height",
                "frozen_tokenizer_state_sha256": getattr(model, "frozen_tokenizer_digest", None),
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "gpu_uuid": actual_uuid, "max_steps": args.max_steps,
                "batch_size": args.batch_size, "post_processing": None}
    if not args.resume:
        (args.output / "run_contract.json").write_text(json.dumps(contract, indent=2))
        (args.output / "experiment_protocol.json").write_bytes(args.protocol.read_bytes())
        for path in policy_files:
            destination = args.output / "source" / path.relative_to(project)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    class Progress(Callback):
        def on_before_backward(self, trainer, module, loss):
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite control training loss")

        def on_validation_end(self, trainer, module):
            if not trainer.sanity_checking:
                print(json.dumps({"event": "validation", "step": trainer.global_step,
                                  "metrics": {key: float(value) for key, value in trainer.callback_metrics.items()
                                              if key.startswith("val/")}}, allow_nan=False), flush=True)

    monitor = "val/loss" if args.stage == "pose" else "val/root_xz_error_m"
    checkpoint = ModelCheckpoint(dirpath=args.output / "checkpoints", filename="best-{step}",
                                 monitor=monitor, save_last=True, save_top_k=1, mode="min")
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="32-true", max_steps=args.max_steps,
                         gradient_clip_val=0.5, log_every_n_steps=20,
                         val_check_interval=args.validation_interval, check_val_every_n_epoch=None,
                         num_sanity_val_steps=1, enable_progress_bar=False,
                         callbacks=[checkpoint, LearningRateMonitor(logging_interval="step"), Progress()],
                         logger=CSVLogger(str(args.output), name="metrics", version=0))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.fit(model, loaders["train"], loaders["validation"],
                ckpt_path=str(args.resume) if args.resume else None)
    metrics = trainer.validate(model, loaders["validation"], verbose=False)
    trainer.save_checkpoint(args.output / "checkpoints/final.ckpt")
    result = {"steps": trainer.global_step, "elapsed_seconds": time.monotonic() - started,
              "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
              "validation": metrics, "quality_approved": False,
              "history_counts": getattr(model, "history_counts", None),
              "frozen_tokenizer_unchanged": None if args.stage != "pose" else
                  state_digest(model.pose_net) == model.frozen_tokenizer_digest}
    (args.output / "pilot_result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)
