"""Factories for the unmodified MotionBricks VQ-VAE, root, and pose models."""

from copy import deepcopy
from functools import partial
from pathlib import Path

import yaml
from omegaconf import OmegaConf


_CONFIGS = {
    "tokenizer": "motionbricks_vqvae",
    "root": "motionbricks_root",
    "pose": "motionbricks_pose",
}


def _official_config(stage):
    if stage not in _CONFIGS:
        raise ValueError(f"Unknown native MotionBricks stage: {stage!r}")
    path = Path(__file__).resolve().parents[2] / "out" / _CONFIGS[stage] / "version_1" / "hparams.yaml"
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def native_config(stage, max_steps):
    """Return a copy of the shipped stage config with SOMA's fixed-floor adaptation.

    The official feature modes, architecture, sampling, losses, optimizer, and
    scheduler settings remain intact. Only the run length and moving-floor
    estimator are specialized for fixed-plane SOMA data.
    """
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    config = deepcopy(_official_config(stage))
    config["trainer"]["max_steps"] = max_steps
    config["model"]["scheduler"]["num_training_steps"] = max_steps
    config["model"]["args"]["floor_estimation"] = None
    return OmegaConf.create(config)


def _optimizer_and_scheduler(config):
    from adam_atan2_pytorch import AdamAtan2
    from motionbricks.motionlib.train.scheduler import WarmupCosineScheduler

    optim_cfg = config["model"]["optimizer"]
    sched_cfg = config["model"]["scheduler"]
    optimizer = partial(AdamAtan2, lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])
    scheduler = partial(
        WarmupCosineScheduler,
        num_warmup_steps=sched_cfg["num_warmup_steps"],
        num_training_steps=sched_cfg["num_training_steps"],
        final_lr=sched_cfg["final_lr"],
        last_epoch=sched_cfg["last_epoch"],
        lt_kwargs=deepcopy(sched_cfg["lt_kwargs"]),
    )
    return optimizer, scheduler


def _adapt_vqvae_dimensions(config, motion_rep):
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
    import torch

    net_cfg = config["model"]["pose_vqvae_network"]
    modes = net_cfg["feature_mode"]
    dummy = torch.zeros(1, 1, len(motion_rep.indices["all"]))
    dims = [extract_feature_from_motion_rep(dummy, motion_rep, mode).shape[-1] for mode in modes]
    keys = ("encoder_state_dim", "decoder_state_dim", "decoder_target_cond_dim", "decoder_external_cond_dim")
    for key, dim in zip(keys, dims):
        net_cfg[key] = dim
    return dims


def _vqvae_config(config, motion_rep):
    from motionbricks.vqvae.neural_modules.vqvae import VQVAE

    _adapt_vqvae_dimensions(config, motion_rep)
    net = deepcopy(config["model"]["pose_vqvae_network"])
    for key in ("_target_", "motion_rep", "pose_root_mode"):
        net.pop(key, None)
    return VQVAE(pose_root_mode="pose", motion_rep=motion_rep, **net)


def _resolved_backbone_args(stage, config, tokenizer=None):
    model = config["model"]
    args = deepcopy(model["backbone_network"]["args"])
    train_args = model["args"]
    if stage == "root":
        links = {
            "min_tokens": "min_tokens", "max_tokens": "max_tokens", "down_t": "down_t",
            "use_hard_num_token_emb_for_root_prediction": "use_hard_num_token_emb_for_root_prediction",
            "local_root_feature": "local_root_feature", "global_root_feature": "global_root_feature",
            "local_pose_feature": "local_pose_feature",
        }
        for target, source in links.items():
            args[target] = train_args[source]
        args["text_embeddings"] = None
    else:
        for key in ("pose_root_mode", "min_tokens", "max_tokens", "down_t", "local_pose_feature",
                    "cond_root_feature", "cond_root_feature_is_from_motion_rep"):
            args[key] = train_args[key]
        args["text_embeddings"] = None
        if tokenizer is not None:
            pose_net = getattr(tokenizer, "pose_net", tokenizer)
            args["pose_vqvae"].update({
                "nb_code": pose_net.num_code,
                "code_dim": pose_net.code_dim,
                "num_heads": pose_net._num_heads,
            })
    return args


def build_native_model(stage, motion_rep, config, tokenizer=None, root_model=None):
    """Build an upstream Lightning model using the official SOMA-adapted config.

    ``motion_rep`` is the global dual-root representation. For the pose stage,
    ``tokenizer`` must be the official tokenizer model (or its VQVAE network).
    ``root_model`` is accepted for staged-runner call compatibility; the official
    pose trainer conditions on supplied root features and does not own the root
    model, so it is deliberately not wrapped into or optimized with the pose net.
    """
    if stage not in _CONFIGS:
        raise ValueError(f"Unknown native MotionBricks stage: {stage!r}")
    fps = float(motion_rep.fps)
    if abs(fps - 30.0) > 1e-6:
        raise ValueError(f"Native MotionBricks models require a 30 Hz motion_rep; received {fps:g} Hz")
    model_cfg = config["model"]
    optimizer, scheduler = _optimizer_and_scheduler(config)
    args = deepcopy(model_cfg["args"])

    if stage == "tokenizer":
        from motionbricks.vqvae.models.motion_vqvae import MotionVQVAEModel

        local_rep = motion_rep.dual_rep.local_motion_rep
        pose_net = _vqvae_config(config, local_rep)
        return MotionVQVAEModel(
            pose_vqvae_network=pose_net,
            root_vqvae_network=None,
            motion_rep=motion_rep,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
        )

    if stage == "root":
        from motionbricks.motion_backbone.models.root_model import MotionModel
        from motionbricks.motion_backbone.neural_modules.root_backbone import root_backbone_network

        backbone = root_backbone_network(_resolved_backbone_args(stage, config), motion_rep)
        return MotionModel(
            pose_vqvae_network=None,
            root_vqvae_network=None,
            backbone_network=backbone,
            motion_rep=motion_rep,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
        )

    from motionbricks.motion_backbone.models.pose_model import MotionModel
    from motionbricks.motion_backbone.neural_modules.pose_backbone import pose_backbone_network

    if tokenizer is None:
        raise ValueError("The pose stage requires the official tokenizer model or VQVAE network")
    pose_net = getattr(tokenizer, "pose_net", tokenizer)
    # The upstream constructor shadows Module.requires_grad_; freeze the parameters directly.
    for parameter in pose_net.parameters():
        parameter.requires_grad_(False)
    pose_net.eval()
    _adapt_vqvae_dimensions(config, motion_rep.dual_rep.local_motion_rep)
    backbone = pose_backbone_network(motion_rep, _resolved_backbone_args(stage, config, pose_net))
    # The upstream constructor checks exists(path); an empty path disables loading.
    args["vqvae_model_ckpt_path"] = ""
    backbone.init_embedding_from_codebooks(pose_net.get_codebook(), None)
    return MotionModel(
        pose_vqvae_network=pose_net,
        root_vqvae_network=None,
        backbone_network=backbone,
        motion_rep=motion_rep,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
    )
