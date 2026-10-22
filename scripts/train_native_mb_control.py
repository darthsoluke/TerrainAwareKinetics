"""Train the shipped Root/Pose classes on versioned native SOMA features."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_native_mb import (
    apply_native_matmul_precision, resume_numerics_transition, runtime_numerics,
    select_gpu, sha256, source_lf_sha256, verify_dataset, write_json,
)


SOURCE_FILES = (
    "scripts/train_native_mb_control.py",
    "scripts/train_native_mb.py",
    "scripts/evaluate_native_mb_checkpoint.py",
    "motionbricks/data/native_mb_data.py",
    "motionbricks/data/native_mb_models.py",
    "motionbricks/helper/data_training_util.py",
    "motionbricks/motion_backbone/models/root_model.py",
    "motionbricks/motion_backbone/models/pose_model.py",
    "motionbricks/motion_backbone/neural_modules/root_backbone.py",
    "motionbricks/motion_backbone/neural_modules/pose_backbone.py",
    "motionbricks/vqvae/neural_modules/vqvae.py",
)

TERRAIN_SOURCE_FILES = (
    "motionbricks/data/native_mb_terrain_batch.py",
    "motionbricks/data/native_mb_terrain_protocol.py",
    "motionbricks/motion_backbone/neural_modules/local_terrain_encoder.py",
)

ONLINE_SOURCE_FILES = (
    "motionbricks/data/native_mb_online_conditions.py",
    "motionbricks/data/native_mb_online_training.py",
    "motionbricks/data/native_mb_online_terrain.py",
    "motionbricks/data/native_mb_online_checkpoint.py",
    "motionbricks/data/local_terrain_observation.py",
)


def state_digest(module, keys=None):
    """Bind tensor values, not pickle layout or tensor devices."""
    digest = hashlib.sha256()
    state = module.state_dict()
    names = sorted(state if keys is None else keys)
    for name in names:
        value = state[name]
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def terrain_gradient_evidence(network):
    """Measure active training paths without changing the upstream objective."""
    import torch

    result = {}
    for label, prefix in (('encoder', '_terrain_encoder.'),
                          ('attention', '_terrain_cross_attention.'),
                          ('height', '_terrain_root_height')):
        gradients = [parameter.grad for name, parameter in network.named_parameters()
                     if name.startswith(prefix) and parameter.grad is not None]
        if any(not bool(torch.isfinite(value).all()) for value in gradients):
            raise RuntimeError('Nonfinite terrain gradient: ' + label)
        result[label] = any(bool(torch.count_nonzero(value)) for value in gradients)
    return result


def verify_tokenizer_binding(metadata, dataset_hash):
    if metadata.get("run_status_at_review") != "completed":
        raise ValueError("Pose requires a completed, immutable native tokenizer run")
    if metadata.get("dataset_manifest_sha256") != dataset_hash:
        raise ValueError("Pose tokenizer and control data must have identical normalization/data")
    if not metadata.get("checkpoint_sha256") or metadata.get("global_step", 0) < 1:
        raise ValueError("Pose requires a trained tokenizer checkpoint identity")


def validate_resume(run, expected, code_root, runner_change_reason=None,
                    allowed_source_changes=()):
    run = Path(run).resolve()
    contract = json.loads((run / "run_contract.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    if contract.get("status") not in {"completed", "trained_not_reviewed"}:
        raise ValueError("Resume requires a completed training run")
    for key in ("stage", "dataset_manifest_sha256", "config_sha256", "seed", "batch_size"):
        if contract.get(key) != expected[key]:
            raise ValueError(f"Control resume mismatch: {key}")
    previous_tokenizer = (contract.get("tokenizer") or {}).get("checkpoint_sha256")
    current_tokenizer = (expected.get("tokenizer") or {}).get("checkpoint_sha256")
    if previous_tokenizer != current_tokenizer:
        raise ValueError("Control resume cannot change its frozen tokenizer")
    if contract.get("training_profile", {}).get("training_profile", "").startswith("paper_"):
        relative = "motionbricks/data/native_mb_paper_protocol.py"
        if contract.get("source_lf_sha256", {}).get(relative) != source_lf_sha256(Path(code_root) / relative):
            raise ValueError("Paper resume protocol source changed")
    if expected["stop_steps"] <= result["optimizer_updates"]:
        raise ValueError("Resume steps must exceed the previous total update count")
    entrypoint_revision = None
    runner_source_revisions = []
    for relative in SOURCE_FILES:
        current = source_lf_sha256(Path(code_root) / relative)
        previous = contract.get("source_lf_sha256", {}).get(relative)
        if previous != current:
            if relative in allowed_source_changes and previous:
                revision = {"source": relative, "previous_sha256": previous,
                            "current_sha256": current,
                            "reviewed_change_reason": "online-only continuation plumbing"}
                runner_source_revisions.append(revision)
                if relative == "scripts/train_native_mb_control.py":
                    entrypoint_revision = revision
            elif (relative in {"scripts/train_native_mb_control.py", "scripts/train_native_mb.py"}
                    and previous and runner_change_reason):
                revision = {"source": relative, "previous_sha256": previous,
                            "current_sha256": current,
                            "reviewed_change_reason": runner_change_reason}
                if relative == "scripts/train_native_mb.py":
                    revision["change_scope"] = (
                        "Shared Tokenizer/Root runner now applies and records configured native "
                        "matmul precision plus the resume numerical transition."
                    )
                runner_source_revisions.append(revision)
                if relative == "scripts/train_native_mb_control.py":
                    entrypoint_revision = revision
            else:
                raise ValueError(f"Control resume source changed: {relative}")
    checkpoint = run / "checkpoints/final.ckpt"
    if sha256(checkpoint) != result["final_checkpoint_sha256"]:
        raise ValueError("Control resume checkpoint hash mismatch")
    return {"run": str(run), "checkpoint": str(checkpoint),
            "sha256": result["final_checkpoint_sha256"],
            "saved_step": result["optimizer_updates"],
            "entrypoint_revision": entrypoint_revision,
            "runner_source_revisions": runner_source_revisions,
            "sampling_note": "RNGs reseeded; not bitwise uninterrupted sampling"}


def load_control_checkpoint(run, device="cpu"):
    import torch
    from omegaconf import OmegaConf
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.data.native_mb_models import build_native_model
    from scripts.evaluate_native_mb_checkpoint import load_native_checkpoint

    run = Path(run).resolve()
    contract = json.loads((run / "run_contract.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    if contract.get("stage") not in {"native_root", "native_pose"}:
        raise ValueError("Expected a native control-stage run")
    if contract.get("status") not in {"completed", "trained_not_reviewed"}:
        raise ValueError("Control training is incomplete")
    dataset = Path(contract["dataset"])
    if sha256(dataset / "native_manifest.json") != contract["dataset_manifest_sha256"]:
        raise ValueError("Control dataset manifest changed")
    verify_dataset(dataset, json.loads((dataset / "native_manifest.json").read_text()))
    if sha256(run / "native_config.yaml") != contract["config_sha256"]:
        raise ValueError("Control configuration changed")
    config = OmegaConf.load(run / "native_config.yaml")
    stage = contract["stage"].removeprefix("native_")
    online_enabled = bool(config.model.args.get("online_conditioning"))
    if online_enabled:
        from motionbricks.data.native_mb_online_checkpoint import (
            validate_online_config, validate_online_control_run, verify_online_root_dependency,
        )
        from motionbricks.data.native_mb_terrain_protocol import validate_terrain_dataset
        from scripts.train_native_mb import verify_training_source_policy

        validate_online_config(config, stage)
        parc_sources = verify_training_source_policy(dataset, json.loads(
            (dataset / "native_manifest.json").read_text()), parc_only=True)
        terrain = validate_terrain_dataset(
            json.loads((dataset / "native_manifest.json").read_text()),
            json.loads((dataset / "accepted.json").read_text()),
        )
        validate_online_control_run(
            contract, result, stage=stage,
            dataset_manifest_sha256=contract["dataset_manifest_sha256"],
            parc_only_sources=parc_sources, terrain_conditioning=terrain,
        )
        if stage == "pose":
            verify_online_root_dependency(
                (contract.get("online_contract") or {}).get("root_dependency"),
                dataset_manifest_sha256=contract["dataset_manifest_sha256"],
                parc_only_sources=parc_sources, terrain_conditioning=terrain,
            )
        elif (contract.get("online_contract") or {}).get("root_dependency") is not None:
            raise ValueError("Online Root run cannot bind an upstream Root dependency")
    elif contract.get("online_contract") is not None:
        raise ValueError("Online contract is present but the native config is not online-enabled")
    tokenizer = None
    if contract["stage"] == "native_pose":
        dependency = contract["tokenizer"]
        tokenizer, metadata, _ = load_native_checkpoint(dependency["run"], device="cpu")
        verify_tokenizer_binding(metadata, contract["dataset_manifest_sha256"])
        if metadata["checkpoint_sha256"] != dependency["checkpoint_sha256"]:
            raise ValueError("Frozen tokenizer checkpoint changed")
    rep = build_native_representation(dataset / "stats")
    model = build_native_model(stage, rep, config, tokenizer=tokenizer)
    checkpoint = run / "checkpoints/final.ckpt"
    if sha256(checkpoint) != result["final_checkpoint_sha256"]:
        raise ValueError("Control checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["global_step"] != result["optimizer_updates"]:
        raise ValueError("Control saved update count mismatch")
    model.load_state_dict(saved["state_dict"], strict=True)
    if online_enabled and stage == "root":
        expected_component_keys = sorted(
            key for key in model.backbone_net.state_dict() if "component_mask" in key)
        recorded_component_keys = (contract.get("online_contract") or {}).get(
            "component_mask_parameter_keys")
        if not expected_component_keys or recorded_component_keys != expected_component_keys:
            raise ValueError("Online Root component-mask parameter listing mismatch")
    model.to(device).eval()
    model.move_supporting_nets_to_device(torch.device(device))
    if stage == "pose" and state_digest(model.supporting_nets["pose_net"]) != result["tokenizer_state_sha256"]:
        raise ValueError("Reloaded frozen tokenizer differs from the trained dependency")
    if stage == "pose":
        model._vqvae_model_loaded = True
    return model, contract, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("root", "pose"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-run", type=Path)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume-run", type=Path)
    initialization.add_argument("--warmstart-run", type=Path,
        help="Verified 50k same-stage run for weights-only transfer; --steps are additional updates")
    initialization.add_argument("--online-warmstart-run", type=Path,
        help="Verified same-stage, same-dataset PARC terrain run; optimizer and schedule reset")
    initialization.add_argument("--online-resume-run", type=Path,
        help="Exact online Root run to resume with its full Lightning optimizer and scheduler state")
    parser.add_argument("--online-root-run", type=Path,
        help="Fully trained verified online Root dependency for online Pose training")
    parser.add_argument("--runner-change-reason", help="Explicit audit note for an entrypoint-only resume revision; model/data changes remain forbidden")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--schedule-steps", type=int, default=2000001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--training-profile", choices=("release", "paper"), default="release")
    parser.add_argument("--skip-review", action="store_true", help="Explicit smoke-only delivery, no quality claim")
    parser.add_argument("--terrain-conditioning", action="store_true",
        help="Opt-in local-scene Root/Pose extension; requires accepted paired SOMA data")
    parser.add_argument("--online-conditioning", action="store_true",
        help="Use observed history and a planar endpoint goal with causal terrain; requires PARC-only data")
    parser.add_argument("--parc-only", action="store_true",
        help="Require only official/generated PARC data, local terrain and fresh initialization")
    args = parser.parse_args(argv)
    if args.parc_only and (not args.terrain_conditioning or args.warmstart_run):
        parser.error("PARC-only controls require terrain conditioning and no cross-dataset warmstart")
    if args.online_conditioning:
        if not (args.terrain_conditioning and args.parc_only and args.skip_review):
            parser.error("Online conditioning requires --terrain-conditioning --parc-only --skip-review")
        if args.resume_run or args.warmstart_run:
            parser.error("Online conditioning forbids generic resume and cross-dataset warmstart; use online-only options")
        if args.online_resume_run and args.stage != "root":
            parser.error("--online-resume-run is Root-only; online Pose must use weights-only warmstart")
        if args.online_resume_run and args.training_profile != "paper":
            parser.error("--online-resume-run requires the original paper schedule")
        if args.stage == "pose" and args.online_root_run is None:
            parser.error("Online Pose training requires --online-root-run")
        if args.stage == "root" and args.online_root_run is not None:
            parser.error("--online-root-run is only valid for online Pose training")
    elif args.online_warmstart_run or args.online_resume_run or args.online_root_run:
        parser.error("Online warmstart and Root dependency options require --online-conditioning")
    if args.terrain_conditioning and args.resume_run:
        parser.error("Terrain extension starts a new versioned run; do not resume a flat run")
    if args.terrain_conditioning and not args.skip_review:
        parser.error("Flat-only review is not terrain acceptance; use --skip-review and a separate terrain review")
    if args.warmstart_run and args.training_profile != "paper":
        parser.error("--warmstart-run requires --training-profile paper")
    if args.training_profile == "paper":
        from motionbricks.data.native_mb_paper_protocol import validate_paper_arguments
        validate_paper_arguments(args.schedule_steps, args.batch_size)
    if not 1 <= args.steps <= args.schedule_steps or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid training budget, batch size or workers")
    if (args.stage == "pose") != (args.tokenizer_run is not None):
        parser.error("Only Pose requires --tokenizer-run")
    if args.output.exists():
        parser.error("Use a new output directory")
    gpu = select_gpu(args.gpu)

    import numpy as np
    import pytorch_lightning as pl
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from motionbricks.data.native_mb_data import (
        NativeMBMotionDataset, build_native_representation, collate_native_batch,
    )
    from motionbricks.data.native_mb_models import build_native_model, native_config
    from motionbricks.data.native_mb_warmstart import (
        load_warmstart_weights, validate_warmstart_run,
    )
    from scripts.evaluate_native_mb_checkpoint import load_native_checkpoint
    from motionbricks.data.native_mb_online_checkpoint import (
        ONLINE_CHECKPOINT_VERSION, ONLINE_RESUME_PLUMBING_SOURCES,
        copy_online_warmstart_backbone, validate_online_config, validate_online_control_run,
        validate_online_resume_run, validate_online_source_compatibility,
        validate_parc_terrain_warmstart_run, state_tree_digest, verify_online_root_dependency,
        verify_same_tokenizer_dependency,
    )

    torch.set_num_threads(4)
    pl.seed_everything(args.seed, workers=True)
    dataset_dir = args.dataset.resolve()
    manifest_path = dataset_dir / "native_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_dataset(dataset_dir, manifest)
    from scripts.train_native_mb import verify_training_source_policy
    parc_sources = verify_training_source_policy(dataset_dir, manifest, parc_only=args.parc_only)
    manifest_hash = sha256(manifest_path)
    terrain_contract = None
    if args.terrain_conditioning:
        from motionbricks.data.native_mb_terrain_protocol import validate_terrain_dataset
        terrain_contract = validate_terrain_dataset(
            manifest, json.loads((dataset_dir / 'accepted.json').read_text()))
    tokenizer, tokenizer_metadata = None, None
    if args.tokenizer_run:
        tokenizer, tokenizer_metadata, _ = load_native_checkpoint(args.tokenizer_run)
        verify_tokenizer_binding(tokenizer_metadata, manifest_hash)
    profile = None
    if args.training_profile == "paper":
        from motionbricks.data.native_mb_paper_protocol import (
            configure_paper_optimizer, paper_config, profile_notes,
        )
        config = paper_config(args.stage)
        profile = profile_notes(args.stage, parc_only=parc_sources is not None)
    else:
        config = native_config(args.stage, args.schedule_steps)
    flat_config = config
    if args.terrain_conditioning:
        from motionbricks.data.native_mb_terrain_protocol import enable_terrain_conditioning
        config = enable_terrain_conditioning(config)
    if args.online_conditioning:
        from motionbricks.data.native_mb_online_conditions import enable_online_conditioning
        config = enable_online_conditioning(config)
        validate_online_config(config, args.stage)
    configured_matmul_precision = apply_native_matmul_precision(config, torch)
    numerics = runtime_numerics(config, configured_matmul_precision, torch)
    rep = build_native_representation(dataset_dir / "stats")
    model = build_native_model(args.stage, rep, config, tokenizer=tokenizer)
    if profile is not None:
        configure_paper_optimizer(model, config)
    if tokenizer is not None:
        # The original path loader was disabled because the verified native
        # checkpoint was loaded above. Reflect that completed load for inference.
        model._vqvae_model_loaded = True
    online_warmstart = None
    online_warmstart_source = None
    online_warmstart_old_root_dependency = None
    online_root_model = None
    online_root_binding = None
    if args.online_warmstart_run:
        source_model, source_contract, source_result = load_control_checkpoint(
            args.online_warmstart_run, device="cpu")
        source_is_online = source_contract.get("online_contract") is not None
        if source_is_online:
            validate_online_control_run(
                source_contract, source_result, stage=args.stage,
                dataset_manifest_sha256=manifest_hash, parc_only_sources=parc_sources,
                terrain_conditioning=terrain_contract,
            )
            if args.stage == "root" and source_contract["online_contract"].get("root_dependency") is not None:
                raise ValueError("Online Root warmstart source cannot bind an upstream Root dependency")
            if args.stage == "pose":
                online_warmstart_old_root_dependency = verify_online_root_dependency(
                    source_contract["online_contract"].get("root_dependency"),
                    dataset_manifest_sha256=manifest_hash, parc_only_sources=parc_sources,
                    terrain_conditioning=terrain_contract,
                )
        else:
            validate_parc_terrain_warmstart_run(
                source_contract, source_result, stage=args.stage,
                dataset_manifest_sha256=manifest_hash, parc_only_sources=parc_sources,
                terrain_conditioning=terrain_contract,
            )
        if args.stage == "pose":
            verify_same_tokenizer_dependency(source_contract.get("tokenizer"), tokenizer_metadata)
        online_warmstart_source = (source_model, source_contract, source_result, source_is_online)
    if args.online_root_run:
        online_root_model, root_contract, root_result = load_control_checkpoint(
            args.online_root_run, device="cpu")
        validate_online_control_run(
            root_contract, root_result, stage="root",
            dataset_manifest_sha256=manifest_hash, parc_only_sources=parc_sources,
            terrain_conditioning=terrain_contract,
        )
        online_root_model.eval()
        for parameter in online_root_model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        if "online_root_model" in model._modules:
            raise RuntimeError("Online Root dependency must not be registered on the Pose model")
        object.__setattr__(model, "online_root_model", online_root_model)
        if any(id(parameter) in {id(value) for value in model.parameters()}
               for parameter in online_root_model.parameters()):
            raise RuntimeError("Online Root parameters entered the Pose optimizer parameter set")
        online_root_binding = {
            "run": str(args.online_root_run.resolve()),
            "checkpoint_sha256": root_result["final_checkpoint_sha256"],
            "state_digest": state_digest(online_root_model),
            "dataset_manifest_sha256": manifest_hash,
            "status": root_contract["status"],
            "frozen": True,
            "eval_mode": True,
            "registered_with_pose_model": False,
        }
    minimum_frames = config.model.args.max_tokens * (2 ** config.model.args.down_t) + 1
    if (not args.online_conditioning and args.stage == "root"
            and config.model.args.prob_off_range_target_token > 0):
        minimum_frames = max(minimum_frames, config.model.args.max_off_target_tokens_to_sample *
                             (2 ** config.model.args.down_t) + 1)
    from motionbricks.data.native_mb_data import native_training_loader
    is_paper = args.training_profile == "paper"
    dataset = NativeMBMotionDataset(dataset_dir, "train", max_frames=max(200, minimum_frames),
                                    min_frames=minimum_frames,
                                    cache_max_bytes=256 * 1024 * 1024 if is_paper else 0,
                                    require_terrain=args.terrain_conditioning,
                                    require_online_terrain=args.online_conditioning)
    if parc_sources is not None:
        from motionbricks.data.parc_training_sources import require_parc_training_origins
        parc_sources["actual_loader_origin_counts"] = require_parc_training_origins(dataset.records)
    loader = native_training_loader(dataset, args.batch_size, args.workers, paper=is_paper)
    args.output.mkdir(parents=True)
    OmegaConf.save(config, args.output / "native_config.yaml")
    code_root = Path(__file__).resolve().parents[1]
    contract = {"stage": f"native_{args.stage}", "status": "running",
        "seed": args.seed, "batch_size": args.batch_size,
        "dataset": str(dataset_dir), "dataset_manifest_sha256": manifest_hash,
        "config_sha256": sha256(args.output / "native_config.yaml"),
        "tokenizer": tokenizer_metadata, "stop_steps": args.steps,
        "original_schedule_steps": args.schedule_steps, "gpu_at_launch": gpu,
        "training_clips": len(dataset), "min_global_frames": minimum_frames, "fps": rep.fps,
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "network_class": type(model.backbone_net).__module__ + "." + type(model.backbone_net).__name__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "source_lf_sha256": {path: source_lf_sha256(code_root / path) for path in SOURCE_FILES},
        "runtime_versions": {"torch": torch.__version__, "lightning": pl.__version__, "numpy": np.__version__},
        "runtime_numerics": numerics,
        "training_profile": profile or {"name": "released_preview_optimizer_and_schedule"},
        "evaluation_source_lf_sha256": source_lf_sha256(code_root / "motionbricks/data/native_mb_evaluation.py"),
        "original_training_step": not args.terrain_conditioning, "custom_losses": False,
        "terrain_conditioning": terrain_contract,
        "quality_approved": False, "ue_deployment": False,
        "adaptations": ["SOMA30/data dimensions",
                        "train-split normalization statistics bound by the dataset manifest",
                        "fixed source floor",
                        f"single GPU with declared input batch size {args.batch_size}",
                        "bounded stop, unchanged schedules"]}
    if args.terrain_conditioning:
        contract['source_lf_sha256'].update({
            path: source_lf_sha256(code_root / path) for path in TERRAIN_SOURCE_FILES})
        contract['adaptations'].append('local terrain and relative root height conditioning; original objectives')
    if args.online_conditioning:
        contract['source_lf_sha256'].update({
            path: source_lf_sha256(code_root / path) for path in ONLINE_SOURCE_FILES})
        contract['adaptations'].append(
            'causal observed-history and planar endpoint conditioning; original objectives')
    contract['parc_only_sources'] = parc_sources
    if parc_sources is not None:
        relative = 'motionbricks/data/parc_training_sources.py'
        contract['source_lf_sha256'][relative] = source_lf_sha256(code_root / relative)
        contract['adaptations'][2] = 'paired PARC source terrain; no flat floor estimator or projection'
    if profile is not None:
        relative = "motionbricks/data/native_mb_paper_protocol.py"
        contract["source_lf_sha256"][relative] = source_lf_sha256(code_root / relative)
    online_resume_source = None
    if args.online_resume_run:
        from motionbricks.data.native_mb_paper_protocol import PAPER_STEPS

        online_resume_source = validate_online_resume_run(
            args.online_resume_run, contract,
            parc_only_sources=parc_sources,
            terrain_conditioning=terrain_contract,
            expected_model_state=model.state_dict(),
            component_mask_parameter_keys=sorted(
                key for key in model.backbone_net.state_dict() if "component_mask" in key),
            max_updates=PAPER_STEPS,
        )
        resume = validate_resume(
            args.online_resume_run, contract, code_root,
            allowed_source_changes=ONLINE_RESUME_PLUMBING_SOURCES,
        )
        resume.update({
            "online_source_revisions": online_resume_source["source_revisions"],
            "component_mask_state_keys": online_resume_source["component_mask_state_keys"],
            "component_mask_state_sha256": online_resume_source["component_mask_state_sha256"],
            "optimizer_state_sha256": online_resume_source["optimizer_state_sha256"],
            "scheduler_state_sha256": online_resume_source["scheduler_state_sha256"],
            "optimizer_state": online_resume_source["optimizer_state"],
            "scheduler_state": online_resume_source["scheduler_state"],
        })
    else:
        resume = validate_resume(args.resume_run, contract, code_root, args.runner_change_reason) \
            if args.resume_run else None

    if online_warmstart_source is not None:
        source_model, source_contract, source_result, source_is_online = online_warmstart_source
        if source_is_online:
            revisions = validate_online_source_compatibility(
                source_contract, contract["source_lf_sha256"])
            if source_contract.get("config_sha256") != contract["config_sha256"]:
                raise ValueError("Same-online warmstart config mismatch")
            if source_contract.get("runtime_numerics") != numerics:
                raise ValueError("Same-online warmstart runtime numerics mismatch")
            if source_contract.get("training_profile") != (profile or {
                    "name": "released_preview_optimizer_and_schedule"}):
                raise ValueError("Same-online warmstart training profile mismatch")
        else:
            revisions = []
        online_warmstart = copy_online_warmstart_backbone(
            model.backbone_net, source_model.backbone_net, args.stage,
            verified_online_source=source_is_online)
        online_warmstart.update({
            "run": str(args.online_warmstart_run.resolve()),
            "checkpoint_sha256": source_result["final_checkpoint_sha256"],
            "dataset_manifest_sha256": manifest_hash,
            "same_stage": True,
            "same_parc_only_source_provenance": True,
            "same_terrain_protocol": True,
            "source_optimizer_updates": source_result["optimizer_updates"],
            "source_is_verified_online_run": source_is_online,
            "source_code_revisions": revisions,
            "learning_rate_reset": OmegaConf.to_container(config.model.optimizer, resolve=True),
            "fresh_scheduler": OmegaConf.to_container(config.model.scheduler, resolve=True),
            "new_updates_target": args.steps,
        })
        if args.stage == "pose":
            online_warmstart["tokenizer_dependency_checkpoint_sha256"] = \
                tokenizer_metadata["checkpoint_sha256"]
            online_warmstart["old_root_dependency"] = online_warmstart_old_root_dependency
        del source_model
        online_warmstart_source = None
    warmstart = validate_warmstart_run(args.warmstart_run, args.stage, dataset_dir, flat_config) \
        if args.warmstart_run else None
    if warmstart:
        checkpoint_state = warmstart.pop("_checkpoint_state")
        if args.terrain_conditioning:
            from motionbricks.data.native_mb_terrain_protocol import load_flat_backbone_into_terrain
            warmstart['new_zero_gated_parameter_keys'] = load_flat_backbone_into_terrain(model, checkpoint_state)
            warmstart['strict_key_and_shape_match'] = 'all original parameters; new terrain branches explicitly listed'
        else:
            load_warmstart_weights(model, checkpoint_state, args.stage, torch)
        del checkpoint_state
        warmstart.update({
            "new_updates_target": args.steps,
            "projected_inherited_plus_new_updates": warmstart["baseline_optimizer_updates"] + args.steps,
            "fresh_optimizer": OmegaConf.to_container(config.model.optimizer, resolve=True),
            "fresh_scheduler": OmegaConf.to_container(config.model.scheduler, resolve=True),
        })
    numerical_transition = None
    if resume:
        previous_contract = json.loads((Path(resume["run"]) / "run_contract.json").read_text())
        numerical_transition = resume_numerics_transition(previous_contract, numerics)
    contract["resume"] = resume
    contract["resume_numerics_transition"] = numerical_transition
    contract["initialization"] = ("full_lightning_resume" if resume else
        "verified_online_weights_only_warmstart" if online_warmstart else
        "verified_weights_only_warmstart" if warmstart else "from_scratch_native_backbone")
    contract["warmstart"] = warmstart
    contract["initial_backbone_state_sha256"] = state_digest(model.backbone_net) if not resume else None
    if args.online_conditioning:
        from motionbricks.data.native_mb_online_conditions import ONLINE_PROTOCOL

        optimizer_state = ({
            "state": "restored from verified full Lightning checkpoint",
            "source_run": resume["run"],
            "config": OmegaConf.to_container(config.model.optimizer, resolve=True),
        } if online_resume_source is not None else {
            "state": "fresh; source optimizer state not loaded",
            "config": OmegaConf.to_container(config.model.optimizer, resolve=True),
        })
        scheduler_state = ({
            "state": "restored from verified full Lightning checkpoint",
            "source_run": resume["run"],
            "config": OmegaConf.to_container(config.model.scheduler, resolve=True),
        } if online_resume_source is not None else {
            "state": "fresh; source scheduler state not loaded",
            "config": OmegaConf.to_container(config.model.scheduler, resolve=True),
        })
        online_contract = {
            "version": ONLINE_CHECKPOINT_VERSION,
            "protocol_version": ONLINE_PROTOCOL["version"],
            "protocol": ONLINE_PROTOCOL,
            "warmstart": online_warmstart,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
        }
        if args.stage == "root":
            online_contract["component_mask_parameter_keys"] = sorted(
                key for key in model.backbone_net.state_dict() if "component_mask" in key)
            if not online_contract["component_mask_parameter_keys"]:
                raise RuntimeError("Online Root model has no component-mask parameters")
        if online_root_binding is not None:
            online_contract["root_dependency"] = online_root_binding
            if online_warmstart is not None and args.stage == "pose":
                online_warmstart["new_root_dependency"] = online_root_binding
                online_contract["warmstart"] = online_warmstart
        contract["online_contract"] = online_contract
    support = model.supporting_nets["pose_net"] if args.stage == "pose" else None
    frozen_digest = state_digest(support) if support is not None else None
    contract["tokenizer_state_sha256"] = frozen_digest
    if is_paper:
        contract["sampling_contract"] = {"input_batch_size": args.batch_size,
            "official_augmentation_factor": model.args["batchsize_mul_factor"],
            "augmented_window_batch_size": int(args.batch_size * model.args["batchsize_mul_factor"]),
            "sampler": "torch RandomSampler replacement; independent real recording crops; drop_last",
            "cache_cpu_bytes_per_worker": dataset.cache_max_bytes,
            "epoch_steps": len(loader), "invented_motion": False}
    write_json(args.output / "run_contract.json", contract)
    shutil.copy2(Path(__file__), args.output / "train_native_mb_control.executed.py")

    class NativeTrainingChecks(pl.Callback):
        def __init__(self):
            self.actual_updates = 0
            self.skipped_batches = 0
            self.input_batch_sizes = set()
            self.terrain_gradient_updates = {'encoder': 0, 'attention': 0, 'height': 0}
            self.component_mask_gradient_updates = 0
            self.component_mask_gradient_update_counts = {}
            self.component_mask_forward_calls = 0
            self.online_root_end_state_digest = None
            self._online_root_hook = None
            if args.online_conditioning:
                watched_backbone = online_root_model.backbone_net if online_root_model is not None \
                    else model.backbone_net

                def count_component_mask_use(_module, _inputs, kwargs):
                    if kwargs.get("global_root_component_mask") is not None:
                        self.component_mask_forward_calls += 1

                self._online_root_hook = watched_backbone.register_forward_pre_hook(
                    count_component_mask_use, with_kwargs=True)

        def _check_frozen_online_root(self):
            if online_root_model is None:
                return
            if online_root_model.training:
                raise RuntimeError("Frozen online Root must remain in eval mode")
            if any(parameter.requires_grad or parameter.grad is not None
                   for parameter in online_root_model.parameters()):
                raise RuntimeError("Frozen online Root must have no trainable parameters or gradients")

        def on_fit_start(self, trainer, module):
            self._check_frozen_online_root()
            if online_root_model is not None and "online_root_model" in module._modules:
                raise RuntimeError("Frozen online Root was registered on the Pose model")

        def on_train_start(self, trainer, module):
            if online_resume_source is not None:
                if trainer.global_step != online_resume_source["saved_step"]:
                    raise RuntimeError("Lightning did not restore the saved Root update step")
                restored_mask_digest = state_digest(
                    module, online_resume_source["component_mask_state_keys"])
                if restored_mask_digest != online_resume_source["component_mask_state_sha256"]:
                    raise RuntimeError("Lightning did not restore the exact online Root component-mask state")
                optimizer_digest = state_tree_digest([
                    optimizer.state_dict() for optimizer in trainer.optimizers])
                if optimizer_digest != online_resume_source["optimizer_state_sha256"]:
                    raise RuntimeError("Lightning did not restore the exact Root optimizer state")
                scheduler_digest = state_tree_digest([
                    config.scheduler.state_dict() for config in trainer.lr_scheduler_configs])
                if scheduler_digest != online_resume_source["scheduler_state_sha256"]:
                    raise RuntimeError("Lightning did not restore the exact Root scheduler state")
                restoration = {
                    "restored_global_step": trainer.global_step,
                    "component_mask_state_sha256": restored_mask_digest,
                    "optimizer_state_sha256": optimizer_digest,
                    "scheduler_state_sha256": scheduler_digest,
                    "exact_state_restore_verified": True,
                    "sampling_note": "RNGs reseeded; not bitwise uninterrupted sampling",
                }
                write_json(args.output / "online_resume_restoration.json", restoration)
                print("ONLINE_ROOT_RESTORE " + json.dumps(restoration, sort_keys=True), flush=True)

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            self._check_frozen_online_root()
            self.input_batch_sizes.add(int(batch["batch_size"]))
            if is_paper and batch["batch_size"] != args.batch_size:
                raise RuntimeError("Paper training input batch is smaller than the declared per-GPU batch")

        def on_before_optimizer_step(self, trainer, module, optimizer):
            gradients = [p.grad for p in module.backbone_net.parameters() if p.grad is not None]
            if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
                raise RuntimeError("Missing/nonfinite native backbone gradients")
            if support is not None:
                if support.training or any(p.requires_grad or p.grad is not None for p in support.parameters()):
                    raise RuntimeError("Pose tokenizer is not frozen in eval mode")
            if args.terrain_conditioning:
                for key, observed in terrain_gradient_evidence(module.backbone_net).items():
                    self.terrain_gradient_updates[key] += int(observed)
            if args.online_conditioning and args.stage == "root":
                component_parameters = [(name, parameter) for name, parameter in module.backbone_net.named_parameters()
                                        if "component_mask" in name]
                if not component_parameters:
                    raise RuntimeError("Online Root has no component-mask parameters to train")
                observed = False
                for name, parameter in component_parameters:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    if not bool(torch.isfinite(gradient).all()):
                        raise RuntimeError("Nonfinite online component-mask gradient: " + name)
                    if bool(torch.count_nonzero(gradient)):
                        observed = True
                        self.component_mask_gradient_update_counts[name] = \
                            self.component_mask_gradient_update_counts.get(name, 0) + 1
                self.component_mask_gradient_updates += int(observed)
            self._check_frozen_online_root()
            self.actual_updates += 1

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if outputs is None:
                self.skipped_batches += 1
            if outputs is not None and trainer.global_step % 500 == 0:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                print(f"NATIVE_{args.stage.upper()}_STEP {trainer.global_step} loss={float(loss):.6f}", flush=True)

        def on_train_end(self, trainer, module):
            self._check_frozen_online_root()
            if online_root_model is not None:
                self.online_root_end_state_digest = state_digest(online_root_model)
                if self.online_root_end_state_digest != online_root_binding["state_digest"]:
                    raise RuntimeError("Frozen online Root changed during Pose training")

        def close(self):
            if self._online_root_hook is not None:
                self._online_root_hook.remove()

    checks = NativeTrainingChecks()
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.output / "checkpoints",
        filename="native-{step}", every_n_train_steps=5000, save_top_k=1, save_last=True)
    trainer = pl.Trainer(max_steps=args.steps, devices=1, accelerator="gpu", strategy="auto",
        precision=config.trainer.precision, gradient_clip_val=config.trainer.gradient_clip_val,
        logger=pl.loggers.CSVLogger(str(args.output), name="logs"),
        callbacks=[checks, checkpoint], log_every_n_steps=25, enable_progress_bar=False,
        num_sanity_val_steps=0, enable_model_summary=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.fit(model, train_dataloaders=loader, ckpt_path=resume["checkpoint"] if resume else None)
    checks.close()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    expected_updates = trainer.global_step - (resume["saved_step"] if resume else 0)
    if checks.actual_updates != expected_updates or checks.skipped_batches:
        raise RuntimeError("Reported global steps must equal real optimizer updates, with no skipped batches")
    if support is not None and state_digest(support) != frozen_digest:
        raise RuntimeError("Frozen tokenizer tensors changed during Pose training")
    if args.online_conditioning:
        if checks.component_mask_forward_calls < checks.actual_updates:
            raise RuntimeError("Online component masks were not used for every optimizer update")
        if args.stage == "root" and checks.component_mask_gradient_updates == 0:
            raise RuntimeError("Online Root component-mask parameters received no nonzero gradients")
        if online_root_model is not None:
            checks._check_frozen_online_root()
            final_root_digest = state_digest(online_root_model)
            if final_root_digest != online_root_binding["state_digest"]:
                raise RuntimeError("Frozen online Root changed during Pose training")
            online_root_binding["end_state_digest"] = final_root_digest
    final = args.output / "checkpoints/final.ckpt"
    trainer.save_checkpoint(final)
    result = {"optimizer_updates": trainer.global_step, "updates_this_run": checks.actual_updates,
        "skipped_batches": checks.skipped_batches,
        "observed_input_batch_sizes": sorted(checks.input_batch_sizes), "elapsed_seconds": elapsed,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "final_checkpoint_sha256": sha256(final), "tokenizer_state_sha256": frozen_digest,
        "runtime_numerics": numerics,
        "resume_numerics_transition": numerical_transition,
        "last_metrics": {key: float(value) for key, value in trainer.callback_metrics.items()
                         if np.ndim(value.detach().cpu()) == 0}, "quality_approved": False}
    if args.online_conditioning:
        result["online_training_evidence"] = {
            "component_mask_forward_use_count": checks.component_mask_forward_calls,
            "component_mask_gradient_updates": checks.component_mask_gradient_updates
            if args.stage == "root" else None,
            "component_mask_gradient_update_counts": checks.component_mask_gradient_update_counts,
            "component_mask_parameter_keys": contract["online_contract"].get(
                "component_mask_parameter_keys", []),
            "root_dependency": online_root_binding,
        }
        if args.stage == "root":
            contract["online_contract"]["component_mask_gradient_evidence"] = {
                "nonzero_gradient_updates": checks.component_mask_gradient_updates,
                "nonzero_gradient_update_counts_by_parameter": checks.component_mask_gradient_update_counts,
                "forward_use_count": checks.component_mask_forward_calls,
            }
    if online_warmstart is not None:
        result["online_warmstart"] = online_warmstart
    if online_resume_source is not None:
        result["online_resume"] = {
            "run": resume["run"],
            "checkpoint_sha256": resume["sha256"],
            "saved_total_updates": resume["saved_step"],
            "additional_updates": checks.actual_updates,
            "component_mask_state_sha256_at_restore": online_resume_source[
                "component_mask_state_sha256"],
        }
    if online_root_binding is not None:
        result["online_root_dependency"] = online_root_binding
    result["optimizer_class"] = type(trainer.optimizers[0]).__module__ + "." + type(trainer.optimizers[0]).__name__
    if args.terrain_conditioning:
        result['terrain_training_evidence'] = {
            'nonzero_gradient_updates': checks.terrain_gradient_updates,
            'final_gates': {name: float(parameter.detach().cpu())
                            for name, parameter in model.backbone_net.named_parameters()
                            if name.startswith('_terrain_') and (name.endswith('.gate') or name.endswith('_gate'))},
            'meaning': 'Participation in optimization, not learned skill or behavior acceptance',
        }
        if args.parc_only and args.steps >= 10 and not all(
                checks.terrain_gradient_updates[key] for key in ('encoder', 'attention')):
            raise RuntimeError('PARC terrain conditions did not participate in optimization')
    if profile is not None and not isinstance(trainer.optimizers[0], torch.optim.Adam):
        raise RuntimeError("Paper profile did not instantiate the declared Adam optimizer")
    write_json(args.output / "training_result.json", result)
    contract["status"] = "trained_not_reviewed"
    write_json(args.output / "run_contract.json", contract)
    if not args.skip_review:
        from motionbricks.data.native_mb_control_review import evaluate_control
        review = evaluate_control(model, args.stage, dataset_dir, args.output / "validation")
        contract["status"] = "completed"
        contract["validation_summary"] = review["summary"]
        write_json(args.output / "run_contract.json", contract)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
