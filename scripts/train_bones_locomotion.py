"""Checkpointed real-SOMA pilot using official MB VQ-VAE or Root Transformer."""

import argparse
import json
import hashlib
import importlib.metadata
import os
import subprocess
import time
import sys
from pathlib import Path


def select_idle_gpus(indices):
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    inventory = {}
    for line in result.strip().splitlines():
        index, uuid, memory, utilization = [value.strip() for value in line.split(",")]
        inventory[int(index)] = (uuid, int(memory), int(utilization))
    selected = []
    for index in indices:
        uuid, memory, utilization = inventory[index]
        if memory > 1024 or utilization > 5:
            raise RuntimeError(f"GPU {index} is not idle ({memory} MiB, {utilization}%); no job launched")
        selected.append(uuid)
    return selected


def canonical_cuda_uuid(value):
    value = str(value)
    return value if value.startswith("GPU-") else "GPU-" + value


def validate_training_scope(args, report, protocol, geometry_preflight):
    if getattr(args, "support_repair_pilot", False):
        if (args.pipeline_smoke or args.convergence_pilot or args.stage != "tokenizer"
                or not 1000 < args.max_steps <= 10000 or not args.initialize_from or args.resume
                or not protocol or protocol.get("experiment_kind") != "bounded_tokenizer_support_repair"
                or not geometry_preflight or not geometry_preflight.get("automatic_preconditions_passed")):
            raise RuntimeError("Support repair needs a separate frozen <=10000-step tokenizer protocol and reviewed initial weights")
        return "bounded_tokenizer_support_repair"
    if args.pipeline_smoke and args.convergence_pilot:
        raise ValueError("Choose pipeline smoke or convergence, not both")
    if args.convergence_pilot:
        if (args.stage != "tokenizer" or not 1000 < args.max_steps <= 10000
                or not protocol or protocol.get("experiment_kind") != "bounded_tokenizer_convergence"
                or not geometry_preflight or not geometry_preflight.get("automatic_preconditions_passed")
                or not (args.initialize_from or args.resume)):
            raise RuntimeError("Convergence needs an audited tokenizer, frozen <=10000-step protocol and initial weights")
        return "bounded_tokenizer_convergence"
    if report["full_training_gate_passed"]:
        return "quality_training"
    if args.pipeline_smoke and args.max_steps <= 1000:
        return "pipeline_smoke"
    raise RuntimeError("Unapproved data permits only pipeline smoke or explicitly audited tokenizer convergence")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("tokenizer", "root", "pose"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="5", help="Explicit physical GPU indices, checked for idleness")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path, help="Weights-only warm start; optimizer/schedule are fresh and provenance is recorded")
    parser.add_argument("--initialization-review", type=Path, help="Source checkpoint token-decode parity report for convergence initialization")
    parser.add_argument("--convergence-pilot", action="store_true", help="Audited tokenizer-only experiment, never full training or deployment approval")
    parser.add_argument("--support-repair-pilot", action="store_true", help="Explicitly bounded change of root authority and support objective, not unchanged-objective convergence")
    parser.add_argument("--scheduler-warmup-steps", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--validation-windows-per-clip", type=int, default=3)
    parser.add_argument("--protocol", type=Path, help="Frozen experiment protocol recorded with the checkpoint")
    parser.add_argument("--tokenizer-profile", choices=("preview", "paper-informed"), default="paper-informed")
    parser.add_argument("--geometry-loss-profile", choices=("upstream", "contact-aware", "support-repair"), default="contact-aware")
    parser.add_argument("--geometry-preflight", type=Path, help="Content-bound source geometry checks for contact-aware training")
    parser.add_argument("--pipeline-smoke", action="store_true", help="Permit a bounded, not quality-approved pilot")
    parser.add_argument("--sole-region-labels", type=Path,
                        help="Folder of sole-region label files for the training split; enables the region supervision")
    parser.add_argument("--loss-overrides", type=Path,
                        help="JSON dict of model.args overrides, frozen into the run contract before training starts")
    parser.add_argument("--control-pilot", action="store_true", help="Bounded SOMA Root/Pose learning with frozen tokenizer")
    parser.add_argument("--tokenizer-run", type=Path)
    parser.add_argument("--control-init-run", type=Path)
    parser.add_argument("--history-policy", choices=("teacher", "generated"))
    parser.add_argument("--history-root-run", type=Path)
    parser.add_argument("--history-generator-pose-run", type=Path)
    args = parser.parse_args()
    if args.max_steps < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Positive steps/batch size and nonnegative workers are required")
    if not 0 < args.scheduler_warmup_steps <= args.max_steps or args.validation_interval < 1 or args.validation_windows_per_clip < 1:
        parser.error("Warmup, validation interval and window count must be positive and warmup must fit the budget")
    if args.resume and args.initialize_from:
        parser.error("Resume and weights-only initialization are distinct operations")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("Use a fresh output directory; existing experiment evidence must not be overwritten")
    # This must precede imports of project/dependency modules: some of them query
    # CUDA at import time, after which changing visibility cannot select a new card.
    indices = [int(index) for index in args.gpus.split(",")]
    if "LOCAL_RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(select_idle_gpus(indices))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if args.control_pilot:
        if args.stage not in ("root", "pose") or len(indices) != 1 or args.initialize_from:
            raise ValueError("Control pilot uses one idle card per run and a Root/Pose stage")
        from scripts.train_bones_control import run_control_training
        return run_control_training(args)
    if args.stage == "pose":
        raise ValueError("SOMA Pose training requires the bounded --control-pilot route")
    from motionbricks.data.bones_contract import dataset_snapshot, file_sha256, require_same_snapshot
    snapshot = dataset_snapshot(args.dataset)
    protocol = json.loads(args.protocol.read_text()) if args.protocol else None
    if protocol and (protocol["stage"] != args.stage or args.max_steps > protocol["max_steps"]):
        raise ValueError("Run stage/step budget differs from the frozen protocol")
    if protocol and (protocol["accepted_sha256"] != snapshot["accepted_sha256"]
                     or protocol["stats_folder"] != snapshot["stats_folder"]):
        raise ValueError("Run dataset/statistics differ from the frozen protocol")
    if protocol and (protocol["tokenizer_profile"] != args.tokenizer_profile
                     or protocol["geometry_loss_profile"] != args.geometry_loss_profile):
        raise ValueError("Run profiles differ from the frozen protocol")
    if protocol and args.batch_size != protocol["batch_size"]:
        raise ValueError("Run batch size differs from the frozen protocol")
    for name in ("scheduler_warmup_steps", "validation_interval", "validation_windows_per_clip"):
        if protocol and name in protocol and protocol[name] != getattr(args, name):
            raise ValueError("Run differs from the frozen protocol: " + name)
    protocol_sha256 = file_sha256(args.protocol) if args.protocol else None
    geometry_preflight = None
    geometry_preflight_sha256 = None
    if args.stage == "tokenizer" and args.geometry_loss_profile in ("contact-aware", "support-repair"):
        if not args.geometry_preflight:
            raise ValueError("Contact-aware training requires --geometry-preflight from audit_bones_geometry_losses.py")
        geometry_preflight = json.loads(args.geometry_preflight.read_text())
        if (geometry_preflight.get("schema") != "motionbricks.geometry_preflight.v1"
                or not geometry_preflight.get("automatic_preconditions_passed")
                or not geometry_preflight.get("checks") or not all(geometry_preflight["checks"].values())
                or geometry_preflight.get("dataset_snapshot_sha256") != snapshot["sha256"]):
            raise ValueError("Source geometry preflight failed or describes a different dataset")
        if (geometry_preflight.get("ground_y_m") != 0.0 or geometry_preflight.get("fps") != 60.0
                or geometry_preflight.get("penetration_tolerance_m") != 0.005):
            raise ValueError("Geometry preflight uses different ground, time or sole-tolerance units")
        geometry_preflight_sha256 = file_sha256(args.geometry_preflight)
    report = json.loads((args.dataset / "quality_report.json").read_text())
    stats_folder = report.get("stats_folder", "stats")
    stats_digest = hashlib.sha256(b"".join(
        (args.dataset / stats_folder / name).read_bytes() for name in ("mean.npy", "std.npy"))).hexdigest()
    if args.resume:
        previous_contract = args.resume.resolve().parent.parent / "run_contract.json"
        previous = json.loads(previous_contract.read_text())
        if previous["dataset"] != str(args.dataset.resolve()) or previous["stage"] != args.stage:
            raise RuntimeError("Resume must retain the original stage and dataset")
        if previous.get("stats_sha256") != stats_digest:
            raise RuntimeError("Resume normalization mismatch, or old checkpoint lacks the required contract")
        require_same_snapshot(previous.get("dataset_snapshot"), snapshot)
        if previous.get("protocol_sha256") != protocol_sha256:
            raise ValueError("Resume must retain its frozen experiment protocol")
        if previous.get("geometry_preflight_sha256") != geometry_preflight_sha256:
            raise ValueError("Resume must retain its source geometry preflight")
        if (previous.get("tokenizer_profile") != args.tokenizer_profile
                or previous.get("geometry_loss_profile") != args.geometry_loss_profile):
            raise ValueError("Resume must retain its architecture and loss profiles")
    experiment_scope = validate_training_scope(args, report, protocol, geometry_preflight)
    initial_contract = None
    initial_config = None
    initialization = {"kind": "fresh"}
    if args.initialize_from:
        if args.stage != "tokenizer":
            raise ValueError("Weights-only initialization is currently supported only for the tokenizer")
        initial_run = args.initialize_from.resolve().parent.parent
        initial_contract = json.loads((initial_run / "run_contract.json").read_text())
        if (initial_contract["stage"] != args.stage or initial_contract["tokenizer_profile"] != args.tokenizer_profile
                or (initial_contract["geometry_loss_profile"] != args.geometry_loss_profile
                    and not (args.support_repair_pilot and initial_contract["geometry_loss_profile"] == "contact-aware"))):
            raise ValueError("Initial checkpoint uses a different stage, architecture or geometry policy")
        require_same_snapshot(initial_contract.get("dataset_snapshot"), snapshot)
        initial_config = initial_run / "upstream_config_overrides.yaml"
        if file_sha256(initial_config) != initial_contract["config_sha256"]:
            raise ValueError("Initial checkpoint configuration changed")
        initialization = {"kind": "weights_only", "checkpoint": str(args.initialize_from.resolve()),
                          "checkpoint_sha256": file_sha256(args.initialize_from),
                          "source_run_contract_sha256": file_sha256(initial_run / "run_contract.json"),
                          "optimizer_restored": False, "schedule_restored": False}
        if protocol and protocol.get("initialize_sha256") != initialization["checkpoint_sha256"]:
            raise ValueError("Initial checkpoint differs from the frozen convergence protocol")
        if args.convergence_pilot or args.support_repair_pilot:
            if not args.initialization_review:
                raise ValueError("Convergence initialization requires the source checkpoint's token-decode parity review")
            initial_review = json.loads(args.initialization_review.read_text())
            if (initial_review.get("checkpoint_sha256") != initialization["checkpoint_sha256"]
                    or initial_review.get("dataset_snapshot_sha256") != snapshot["sha256"]
                    or not initial_review.get("token_decoder_parity", {}).get("passed")):
                raise ValueError("Initial token-decode review failed or describes different weights/data")
            initialization["review_sha256"] = file_sha256(args.initialization_review)
            initialization["source_geometry_coefficients"] = initial_contract.get("geometry_coefficients")
            if args.support_repair_pilot:
                initialization["root_condition_policy_changed"] = initial_contract.get("root_condition_policy", "decoded") != "provided_velocity"
    elif args.resume:
        initialization = previous.get("initialization", {"kind": "fresh"})
    import numpy as np
    import pytorch_lightning as pl
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.strategies import DDPStrategy
    from torch.utils.data import DataLoader
    from motionbricks.data.bones_models import BonesRootModel, BonesTokenizer
    from motionbricks.data.bones_seed import BonesMotionDataset, build_representation, collate_bones_batch, validation_window_records

    torch.set_num_threads(4)
    cuda_device_uuid = canonical_cuda_uuid(torch.cuda.get_device_properties(0).uuid)
    if cuda_device_uuid not in os.environ["CUDA_VISIBLE_DEVICES"].split(","):
        raise RuntimeError("CUDA initialized on a different GPU; refusing to train")
    pl.seed_everything(2333, workers=True)
    representation = build_representation(args.dataset / report.get("stats_folder", "stats"))
    records = json.loads((args.dataset / "accepted.json").read_text())
    datasets = {}
    for split in ("train", "validation"):
        selected = [record for record in records if record["split"] == split and record["frames_60hz"] >= 129]
        if not selected:
            raise RuntimeError(f"No usable {split} clips")
        if split == "validation":
            selected = validation_window_records(selected, 128, args.validation_windows_per_clip)
        # Only the training split carries sole-region labels; validation is scored with the
        # same fixed windows for both arms and never reads them.
        datasets[split] = BonesMotionDataset(
            args.dataset, selected, representation.stats,
            crop_frames=360 if split == "train" else 129, training=split == "train",
            sole_region_folder=args.sole_region_labels if split == "train" else None)
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                          num_workers=args.workers, collate_fn=collate_bones_batch,
                          persistent_workers=args.workers > 0, pin_memory=True)
        for split, dataset in datasets.items()
    }
    project = Path(__file__).resolve().parents[1]
    policy_files = [Path(__file__), project / "motionbricks/data/bones_models.py",
                    project / "motionbricks/data/bones_seed.py", project / "motionbricks/data/bones_evaluation.py",
                    project / "motionbricks/data/bones_contract.py", project / "motionbricks/helper/data_training_util.py",
                    project / "motionbricks/data/soma_replay.py", project / "motionbricks/motionlib/core/skeletons/soma.py",
                    project / "motionbricks/vqvae/models/motion_vqvae.py", project / "scripts/audit_bones_geometry_losses.py",
                    project / "motionbricks/data/kinematic_consistency.py"]
    if args.sole_region_labels:
        policy_files += [project / "motionbricks/data/sole_geometry_losses.py",
                         project / "motionbricks/data/sole_region_contact.py",
                         project / "motionbricks/data/soma_skin_reference.py"]
    policy_hashes = {path.relative_to(project).as_posix(): file_sha256(path) for path in policy_files}
    if args.resume and previous.get("training_source_sha256") != policy_hashes:
        raise ValueError("Training policy changed; do not silently resume an older experiment")
    template_stage = "vqvae" if args.stage == "tokenizer" else "root"
    conf = OmegaConf.load(project / f"out/motionbricks_{template_stage}/version_1/hparams.yaml")
    with open_dict(conf):
        conf.data = {"text_embeddings": None}
        conf.fps = representation.fps
        conf.skeleton = {"_target_": "motionbricks.motionlib.core.skeletons.soma.SOMASkeleton30"}
        conf.motion_rep = {"_target_": "motionbricks.motionlib.core.motion_reps.dual_root_global_joints.GlobalRootGlobalJoints",
                           "fps": representation.fps, "skeleton": "${skeleton}", "name": "somaskel30_dual_root_global_joints",
                           "stats": {"_target_": "motionbricks.motionlib.core.utils.stats.Stats",
                                     "folder": str(args.dataset / stats_folder)}}
        conf.model._target_ = "motionbricks.data.bones_models.BonesTokenizer" if args.stage == "tokenizer" else "motionbricks.data.bones_models.BonesRootModel"
        conf.trainer = {"_target_": "pytorch_lightning.Trainer", "max_steps": args.max_steps,
                        "accelerator": "gpu", "devices": len(indices), "num_nodes": 1, "precision": "32-true",
                        "gradient_clip_val": 0.5, "log_every_n_steps": 10,
                        "val_check_interval": min(args.validation_interval, args.max_steps),
                        "check_val_every_n_epoch": None, "num_sanity_val_steps": 2,
                        "enable_progress_bar": False, "strategy": "ddp_find_unused_parameters_true" if len(indices) > 1 else "auto"}
        conf.model.args.min_tokens = 6
        conf.model.args.max_tokens = 32
        conf.model.args.batchsize_mul_factor = 1
        conf.model.args.keyframe_num_warmup_steps = 100
        conf.model.args.floor_estimation = None
        conf.model.scheduler.num_warmup_steps = args.scheduler_warmup_steps
        conf.model.scheduler.num_training_steps = args.max_steps
        if args.stage == "tokenizer":
            conf.model.args.trusted_skate_coeff = 0.0
            conf.model.args.penetration_coeff = 0.0
            if args.geometry_loss_profile in ("contact-aware", "support-repair"):
                conf.model.args.skate_contact_loss_coeff = 0
                conf.model.args.trusted_skate_coeff = 0.01
                conf.model.args.penetration_coeff = 1.0
            if args.geometry_loss_profile == "support-repair":
                if not args.support_repair_pilot or not protocol:
                    raise ValueError("Support-repair policy requires its own audited bounded protocol")
                conf.model.args.support_repair = True
                for name, value in protocol["geometry_coefficients"].items():
                    if value < 0:
                        raise ValueError("Loss coefficients must be nonnegative")
                    conf.model.args[name] = value
                conf.model.pose_vqvae_network._target_ = "motionbricks.data.bones_models.ConditionedBonesVQVAE"
                conf.model.pose_vqvae_network.root_condition_policy = "provided_velocity"
            if args.tokenizer_profile == "paper-informed":
                network = conf.model.pose_vqvae_network
                network.feature_mode = ["joint_positions_and_rotations", "pose",
                                        "joint_positions_and_rotations", "root_without_hip_height"]
                # Four learned 128-entry heads follow the paper's per-head recommendation.
                # Width/latent dimension retain the released compact implementation.
                network.num_heads = 4
                network.nb_code = 128 ** 4
                conf.model.optimizer._target_ = "torch.optim.Adam"
                conf.model.optimizer.lr = 5e-5
                conf.model.scheduler.final_lr = 2e-6
        else:
            conf.model.args.prob_provide_num_tokens = 1.0
    loss_overrides = json.loads(args.loss_overrides.read_text()) if args.loss_overrides else {}
    if loss_overrides:
        with open_dict(conf):
            for name, value in loss_overrides.items():
                conf.model.args[name] = value
    optimizer = instantiate(conf.model.optimizer)
    scheduler = instantiate(conf.model.scheduler)
    model_args = OmegaConf.to_container(conf.model.args, resolve=True)
    coefficient_names = ["skate_contact_loss_coeff", "trusted_skate_coeff", "penetration_coeff", "joint_vel_loss_coeff"]
    if model_args.get("sole_region_supervision", False):
        coefficient_names += ["sole_region_coeff_contact_height_m", "sole_region_coeff_penetration_m",
                              "sole_region_coeff_slip_mps"]
        if model_args.get("sole_region_coeff_stance_displacement_m", 0.0):
            coefficient_names += ["sole_region_coeff_stance_displacement_m"]
    if args.support_repair_pilot:
        coefficient_names += ["stance_anchor_coeff", "support_height_coeff", "fk_position_coeff", "sole_position_coeff"]
    for name in ("kinematic_position_consistency_coeff", "kinematic_velocity_consistency_coeff"):
        if name in model_args:
            coefficient_names.append(name)
    geometry_coefficients = {name: model_args.get(name, 0.0) for name in coefficient_names}
    if initial_contract and args.support_repair_pilot:
        old_coefficients = initial_contract.get("geometry_coefficients", {})
        initialization["objective_changed"] = any(old_coefficients.get(name, 0.0) != geometry_coefficients.get(name, 0.0)
                                                  for name in set(old_coefficients) | set(geometry_coefficients))
    if protocol and protocol.get("geometry_coefficients") != geometry_coefficients and "geometry_coefficients" in protocol:
        raise ValueError("Loss weights differ from the frozen experiment protocol")
    if initial_contract and initial_contract.get("geometry_coefficients") != geometry_coefficients and not args.support_repair_pilot:
        raise ValueError("Weights-only convergence must preserve the source run's geometry weights")
    if args.stage == "tokenizer":
        network = instantiate(conf.model.pose_vqvae_network, motion_rep=representation.dual_rep.local_motion_rep)
        if initial_config:
            old_network = OmegaConf.to_container(OmegaConf.load(initial_config).model.pose_vqvae_network, resolve=False)
            new_network = OmegaConf.to_container(conf.model.pose_vqvae_network, resolve=False)
            calculated_dims = {"encoder_state_dim", "decoder_state_dim", "decoder_target_cond_dim", "decoder_external_cond_dim"}
            if args.support_repair_pilot:
                calculated_dims |= {"_target_", "root_condition_policy"}
            if ({key: value for key, value in old_network.items() if key not in calculated_dims}
                    != {key: value for key, value in new_network.items() if key not in calculated_dims}):
                raise ValueError("Initial checkpoint network definition differs, even if parameter shapes match")
        with open_dict(conf):
            dummy = torch.zeros(1, 1, len(network.motion_rep.indices["all"]))
            for dim_key, mode in (("encoder_state_dim", network.encoder_input_feature_mode),
                                  ("decoder_state_dim", network.decoder_input_feature_mode),
                                  ("decoder_target_cond_dim", network.decoder_target_cond_feature_mode),
                                  ("decoder_external_cond_dim", network.decoder_external_cond_feature_mode)):
                conf.model.pose_vqvae_network[dim_key] = network.extract_feature(dummy, mode).shape[-1]
        model = BonesTokenizer(pose_vqvae_network=network, root_vqvae_network=None,
                               motion_rep=representation, optimizer=optimizer, scheduler=scheduler, args=model_args)
    else:
        network = instantiate(conf.model.backbone_network, motion_rep=representation, _recursive_=False)
        model = BonesRootModel(pose_vqvae_network=None, root_vqvae_network=None, backbone_network=network,
                               motion_rep=representation, optimizer=optimizer, scheduler=scheduler, args=model_args)
    if args.initialize_from:
        initial_state = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        # Sole indices are deterministic asset buffers, not learned weights. A warm
        # start from pre-region models may lack only these two derived buffers.
        incompatible = model.load_state_dict(initial_state["state_dict"], strict=False)
        allowed_missing = {"sole_region_indices_left", "sole_region_indices_right"}
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise ValueError(f"Initial learned state differs: {incompatible}")
        initialization["source_global_step"] = initial_state["global_step"]
        del initial_state
    initial_weights_sha256 = hashlib.sha256(b"".join(
        name.encode() + parameter.detach().cpu().numpy().tobytes()
        for name, parameter in sorted(model.state_dict().items())
        if name not in {"sole_region_indices_left", "sole_region_indices_right"})).hexdigest()
    # Asset setup or checkpoint loading must not change the data/keyframe RNG stream.
    pl.seed_everything(2333, workers=True)
    args.output.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(str(args.output), name="metrics", version=0)
    selection_metric = "val/max_joint_error_mean_m" if args.stage == "tokenizer" else "val/root_xz_error_m"
    if args.support_repair_pilot:
        selection_metric = "val/support_repair_score"
    checkpoint = ModelCheckpoint(dirpath=args.output / "checkpoints", filename="best-{step}",
                                 monitor=selection_metric, mode="min", save_top_k=1, save_last=True,
                                 save_on_train_epoch_end=False)
    class TrainingProgress(Callback):
        def on_before_backward(self, trainer, module, loss):
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite training loss; stop this run instead of saving it as successful")

        def on_validation_end(self, trainer, module):
            if trainer.is_global_zero and not trainer.sanity_checking:
                names = ("val/loss", "val/mpjpe_m", "val/max_joint_error_mean_m",
                         "val/trusted_skate_mps", "val/sole_penetration_max_m", "val/contact_f1_auto")
                values = {name: float(trainer.callback_metrics[name]) for name in names
                          if name in trainer.callback_metrics}
                print(json.dumps({"event": "validation", "step": trainer.global_step, "metrics": values},
                                 allow_nan=False), flush=True)

    trainer_args = OmegaConf.to_container(conf.trainer, resolve=True)
    trainer_args.pop("_target_")
    trainer_args["strategy"] = DDPStrategy(find_unused_parameters=True) if len(indices) > 1 else "auto"
    trainer = pl.Trainer(
        **trainer_args,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step"), TrainingProgress()],
        logger=logger, default_root_dir=str(args.output),
    )
    if trainer.is_global_zero:
        OmegaConf.save(conf, args.output / "upstream_config_overrides.yaml")
        provenance = {
            "stage": args.stage, "dataset": str(args.dataset.resolve()),
            "experiment_scope": experiment_scope, "initialization": initialization,
            "full_training_gate_passed": report["full_training_gate_passed"],
            "train_clips": len(datasets["train"]), "validation_windows": len(datasets["validation"]),
            "stats_folder": report.get("stats_folder", "stats"),
            "stats_sha256": stats_digest,
            "dataset_snapshot": snapshot, "training_source_sha256": policy_hashes,
            "protocol_sha256": protocol_sha256, "checkpoint_selection_metric": selection_metric,
            "config_sha256": file_sha256(args.output / "upstream_config_overrides.yaml"),
            "tokenizer_profile": args.tokenizer_profile, "geometry_loss_profile": args.geometry_loss_profile,
            "geometry_coefficients": geometry_coefficients,
            "loss_overrides": loss_overrides,
            "loss_overrides_sha256": file_sha256(args.loss_overrides) if args.loss_overrides else None,
            "sole_region_labels_folder": str(args.sole_region_labels) if args.sole_region_labels else None,
            "sole_region_manifest_sha256": file_sha256(args.sole_region_labels / "manifest.json")
                                            if args.sole_region_labels else None,
            "sole_region_labels_version": model_args.get("sole_region_labels_version"),
            "root_condition_policy": "provided_velocity" if args.support_repair_pilot else "decoded",
            "reconstruction_only": args.stage == "tokenizer",
            "encoder_observes_entire_source_window": args.stage == "tokenizer",
            "validation_root_condition_source": "recorded full-window planar velocities; not player input",
            "geometry_preflight_sha256": geometry_preflight_sha256,
            "contact_supervision": "saved confidence aligned through official segment sampling",
            "pipeline_smoke_only": experiment_scope == "pipeline_smoke",
            "quality_approved": False,
            "physical_gpus": indices, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "actual_cuda_device_uuid": cuda_device_uuid,
            "frames_per_token": 4, "max_frames": 128, "horizon_seconds": 128 / 60,
            "training_body_keyframes": "upstream variable 0-to-10" if args.stage == "tokenizer" else "observed history only",
            "validation_body_keyframes": "first four observed frames only; no future target poses",
            "future_root_height": "carried from observed history for Root; reconstructed by tokenizer",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "initial_weights_sha256": initial_weights_sha256,
            "initial_weights_hash_scope": "loaded parameters and EMA state, excluding deterministic sole-region index buffers",
            "seed": 2333,
            "sampling_order": "pl.seed_everything(2333) with identical loader settings, dataset snapshot and "
                              "batch size, so the row resampling and window starts are identical across arms",
            "torch_version": torch.__version__, "lightning_version": pl.__version__,
            "dependencies": {package: importlib.metadata.version(package) for package in
                             ("vector-quantize-pytorch", "adam-atan2-pytorch", "kimodo", "numpy", "scipy")},
        }
        (args.output / "run_contract.json").write_text(json.dumps(provenance, indent=2))
        if args.protocol:
            (args.output / "experiment_protocol.json").write_bytes(args.protocol.read_bytes())
        if args.geometry_preflight:
            (args.output / "geometry_preflight.json").write_bytes(args.geometry_preflight.read_bytes())
        if args.initialization_review:
            (args.output / "initialization_review.json").write_bytes(args.initialization_review.read_bytes())
        (args.output / "environment-freeze.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.fit(model, train_dataloaders=loaders["train"], val_dataloaders=loaders["validation"],
                ckpt_path=str(args.resume) if args.resume else None)
    # Always validate/save final state, including runs that end between validation intervals.
    metrics = trainer.validate(model, loaders["validation"], verbose=False)
    trainer.save_checkpoint(args.output / "checkpoints/final.ckpt")
    if trainer.is_global_zero:
        result = {"steps": trainer.global_step, "elapsed_seconds": time.monotonic() - started,
                  "validation": metrics, "best_checkpoint": checkpoint.best_model_path,
                  "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                                         if torch.cuda.is_available() else None),
                  "batch_size": args.batch_size, "max_steps": args.max_steps,
                  "learning_rate": conf.model.optimizer.lr, "physical_gpus": indices,
                  "quality_approved": False}
        (args.output / "pilot_result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
