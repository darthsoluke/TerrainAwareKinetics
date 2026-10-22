from pathlib import Path
import sys

import pytest
import torch
from omegaconf import DictConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "out" / "vendor" / "kimodo"))

from motionbricks.data.native_mb_models import build_native_model, native_config
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from motionbricks.motion_backbone.models.root_model import MotionModel as RootMotionModel
from motionbricks.motion_backbone.models.pose_model import MotionModel as PoseMotionModel
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from motionbricks.motionlib.train.scheduler import WarmupCosineScheduler
from motionbricks.vqvae.models.motion_vqvae import MotionVQVAEModel
from motionbricks.vqvae.neural_modules.vqvae import VQVAE


def build_native_representation(fps=30.0):
    return GlobalRootGlobalJoints(
        fps,
        SOMASkeleton30(),
        "somaskel30_dual_root_global_joints",
    )


@pytest.fixture(scope="module")
def motion_rep():
    return build_native_representation()


def test_native_configs_keep_official_training_shape_and_expose_horizon():
    tokenizer = native_config("tokenizer", max_steps=2_000_001)
    root = native_config("root", max_steps=2_000_001)
    pose = native_config("pose", max_steps=2_000_001)

    assert all(isinstance(cfg, DictConfig) for cfg in (tokenizer, root, pose))
    for cfg in (tokenizer, root, pose):
        assert cfg.trainer.max_steps == 2_000_001
        assert cfg.model.scheduler.num_training_steps == 2_000_001
        assert cfg.model.scheduler.num_warmup_steps == 10_000
        assert cfg.model.optimizer._target_ == "adam_atan2_pytorch.AdamAtan2"
        assert cfg.model.args.floor_estimation is None
        assert cfg.model.args.batchsize_mul_factor == 2

    assert (tokenizer.model.args.min_tokens, tokenizer.model.args.max_tokens) == (6, 16)
    assert (root.model.args.min_tokens, root.model.args.max_tokens) == (6, 16)
    assert (pose.model.args.min_tokens, pose.model.args.max_tokens) == (6, 16)
    assert pose.model.args.masked_token_ratio == 0.8
    assert pose.model.args.incorrect_token_ratio_min == 0.0
    assert pose.model.args.incorrect_token_ratio_max == 1.0
    assert pose.model.args.max_num_middle_keyframes == 8
    assert pose.model.args.no_middle_keyframe_prob == 0.5
    assert pose.model.backbone_network.args.n_embd == 1024


def test_tokenizer_is_original_model_with_soma_feature_dimensions_and_no_extra_loss(motion_rep):
    config = native_config("tokenizer", max_steps=2_000_001)
    model = build_native_model("tokenizer", motion_rep, config)

    assert type(model) is MotionVQVAEModel
    assert type(model.pose_net) is VQVAE
    assert model.root_net is None
    assert model.pose_net.motion_rep.name == "local"
    assert model.pose_net.encoder.model[0].in_channels == 268
    assert model.pose_net.decoder.model[-1].out_channels == 365
    assert config.model.pose_vqvae_network.encoder_state_dim == 268
    assert config.model.pose_vqvae_network.decoder_state_dim == 365
    assert config.model.pose_vqvae_network.decoder_target_cond_dim == 268
    assert config.model.pose_vqvae_network.decoder_external_cond_dim == 2
    assert model.pose_net.decoder_target_cond_feature_mode == "joint_positions_and_rotations_and_hip_height"
    assert model.pose_net.decoder_external_cond_feature_mode == "root_without_hip_height_without_heading"
    assert model.args["min_tokens"] == 6 and model.args["max_tokens"] == 16
    assert model.args["batchsize_mul_factor"] == 2
    assert model.args["floor_estimation"] is None
    assert "contact_confidence_loss_coeff" not in model.args
    assert "sole_region_loss_coeff" not in model.args
    assert model.scheduler.func is WarmupCosineScheduler
    assert model.scheduler.keywords["num_training_steps"] == 2_000_001

    root_model = build_native_model("root", motion_rep, native_config("root", 2_000_001))
    pose_config = native_config("pose", 2_000_001)
    pose_model = build_native_model("pose", motion_rep, pose_config, tokenizer=model, root_model=root_model)
    assert type(pose_model) is PoseMotionModel
    assert pose_model.backbone_net._args["n_embd"] == 1024
    assert pose_model.backbone_net.initted.item()
    assert pose_model.args["vqvae_model_ckpt_path"] == ""
    assert pose_config.model.pose_vqvae_network.encoder_state_dim == 268
    pose_model.train()
    assert not pose_model.supporting_nets["pose_net"].training
    assert all(not parameter.requires_grad for parameter in pose_model.supporting_nets["pose_net"].parameters())


def test_root_factory_keeps_original_root_model_and_learned_decoder(motion_rep):
    config = native_config("root", max_steps=2_000_001)
    model = build_native_model("root", motion_rep, config)

    assert type(model) is RootMotionModel
    assert model.backbone_net._conv_output is not None
    assert model.backbone_net._conv_output.model[-1].out_channels == 5
    assert model.args["max_num_start_keyframes"] == 4
    assert model.args["max_num_end_keyframes"] == 4
    assert model.args["keyframe_num_warmup_steps"] == 200_000
    assert model.args["floor_estimation"] is None


def test_root_decoder_output_is_learned_and_differentiable(motion_rep):
    config = native_config("root", max_steps=100)
    backbone_args = config.model.backbone_network.args
    backbone_args.n_embd = 32
    backbone_args.n_head = 4
    backbone_args.n_layers_shared = 1
    backbone_args.n_layers_root_token = 1
    backbone_args.width = 32
    backbone_args.depth = 1
    model = build_native_model("root", motion_rep, config)
    backbone = model.backbone_net

    supplied_global_root = torch.full((1, 8, 5), 123.0)
    has_conditions = torch.ones((1, 8), dtype=torch.bool)
    output = backbone(
        global_root_values=supplied_global_root,
        has_global_root_values=has_conditions,
        local_root_values=torch.zeros(1, 8, 4),
        has_local_root_values=has_conditions,
        poses=torch.zeros(1, 8, 268),
        has_poses=has_conditions,
        num_tokens=torch.tensor([[6]]),
        groundtruth_num_tokens=torch.tensor([[6]]),
    )["pred_global_root_values"]
    middle_prediction = output[:, 16:20]
    middle_prediction.square().mean().backward()

    decoder_grad = backbone._conv_output.model[-1].weight.grad
    assert torch.isfinite(output).all()
    assert not torch.allclose(middle_prediction.detach(), supplied_global_root[:, :4])
    assert decoder_grad is not None and torch.isfinite(decoder_grad).all()
    assert decoder_grad.abs().sum() > 0


def test_soma_feature_widths_are_derived_from_representation(motion_rep):
    sample = torch.zeros(1, 1, len(motion_rep.indices["all"]))
    local = motion_rep.dual_rep.local_motion_rep
    assert len(motion_rep.indices["all"]) == 366
    assert extract_feature_from_motion_rep(sample, local, "joint_positions_and_rotations_and_hip_height").shape[-1] == 268
    assert extract_feature_from_motion_rep(sample, local, "root_without_hip_height_without_heading").shape[-1] == 2
    assert extract_feature_from_motion_rep(sample, motion_rep, "root_without_hip_height").shape[-1] == 4
    assert extract_feature_from_motion_rep(sample, motion_rep, "root").shape[-1] == 5
    assert motion_rep.skeleton.nbjoints == 30
    assert isinstance(motion_rep.skeleton, SOMASkeleton30)


def test_invalid_stage_and_horizon_fail_early():
    with pytest.raises(ValueError):
        native_config("unknown", 100)
    with pytest.raises(ValueError):
        native_config("pose", 0)


def test_native_factory_rejects_non_30hz_representation():
    with pytest.raises(ValueError, match="require a 30 Hz"):
        build_native_model("root", build_native_representation(fps=60.0), native_config("root", 100))
