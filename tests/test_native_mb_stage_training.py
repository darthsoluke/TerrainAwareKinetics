"""CPU integration checks for the shipped SOMA30 root and pose training paths."""

import gc
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "out" / "vendor" / "kimodo"))

from motionbricks.data.native_mb_data import collate_native_batch
from motionbricks.data.native_mb_models import build_native_model, native_config
from motionbricks.data.native_mb_paper_protocol import paper_config
from motionbricks.motion_backbone.models.pose_model import MotionModel as NativePoseModel
from motionbricks.motion_backbone.models.root_model import MotionModel as NativeRootModel
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from motionbricks.motionlib.core.utils.stats import Stats
from motionbricks.motionlib.train.scheduler import WarmupCosineScheduler
from motionbricks.vqvae.neural_modules.vqvae import VQVAE


@pytest.fixture(scope="module", autouse=True)
def _cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _motion_and_representation(frame_count=25):
    """Make a static, grounded SOMA clip via official FK and feature conversion."""
    skeleton = SOMASkeleton30()
    local_rots = torch.eye(3).expand(frame_count, skeleton.nbjoints, 3, 3).clone()
    root = torch.zeros(frame_count, 3)
    _, fk_positions, _ = skeleton.fk(local_rots, root)
    # Place the official foot joints on the floor. The static clip then has valid
    # stance labels and avoids injecting hand-authored pose features.
    root[:, 1] = -fk_positions[:, skeleton.foot_joint_idx, 1].amin()
    foot_contacts = torch.ones(frame_count, 4)
    unnormalized_rep = GlobalRootGlobalJoints(30.0, skeleton, "somaskel30_dual_root_global_joints")
    features = unnormalized_rep(
        {"local_joint_rots": local_rots, "translation": root,
         "foot_contacts": foot_contacts},
        to_normalize=False,
        lengths=torch.tensor(frame_count),
    )
    assert features.shape == (frame_count, 366)
    assert torch.isfinite(features).all()

    # Derive complete dual-representation statistics from this test motion, with
    # a nonzero floor for constant channels, then let the representation slice
    # the global and local statistics in its normal constructor path.
    local_features = unnormalized_rep.dual_rep.global_to_local(
        features, is_normalized=False, to_normalize=False,
        lengths=torch.full((1,), frame_count),
    )
    dual_features = unnormalized_rep.dual_rep.cat_global_and_local(features, local_features)
    mean = dual_features.mean(dim=0)
    std = dual_features.std(dim=0, unbiased=False).clamp_min(0.1)
    stats = Stats()
    stats.register_from_tensors(mean, std)
    representation = GlobalRootGlobalJoints(
        30.0, SOMASkeleton30(), "somaskel30_dual_root_global_joints", stats=stats
    )
    normalized = representation.stats.normalize(features)
    batch = collate_native_batch([{"motion": normalized}])
    return representation, batch


def _attach_minimal_training_context(model):
    # The original training_step needs only global_step for scheduled keyframes;
    # skip Lightning's logger plumbing while still exercising that method itself.
    model._trainer = SimpleNamespace(global_step=100_000)
    model.log = lambda *args, **kwargs: None


def _set_test_sampler_horizon(model):
    # Sample the official minimum (6 x 4 = 24 frames) without changing the
    # already-constructed backbone's 6..16-token embeddings or output heads.
    model.args["min_tokens"] = 6
    model.args["max_tokens"] = 6


def _assert_finite_nonzero_grad(parameter, label):
    assert parameter.grad is not None, f"{label} received no gradient"
    assert torch.isfinite(parameter.grad).all(), f"{label} gradient is nonfinite"
    assert parameter.grad.abs().sum() > 0, f"{label} gradient is identically zero"


def test_native_root_training_step_backpropagates_root_and_duration_losses():
    torch.manual_seed(701)
    np.random.seed(701)
    representation, batch = _motion_and_representation()
    config = native_config("root", max_steps=2_000_001)
    model = build_native_model("root", representation, config)
    assert type(model) is NativeRootModel
    expected_duration_classes = config.model.args.max_tokens - config.model.args.min_tokens + 2
    assert expected_duration_classes == 12
    assert model.backbone_net._proj_num_token_output_logit.out_features == expected_duration_classes
    assert model.backbone_net._args["max_tokens"] == config.model.args.max_tokens == 16
    assert model.backbone_net._args["n_embd"] == 512
    assert model.backbone_net._args["n_layers_shared"] == 3
    assert model.backbone_net._args["width"] == 512
    assert model.scheduler.func is WarmupCosineScheduler
    _set_test_sampler_horizon(model)
    model.train()
    _attach_minimal_training_context(model)

    loss = model.training_step(batch, 0)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()

    _assert_finite_nonzero_grad(model.backbone_net._conv_output.model[-1].weight,
                                "native root continuous decoder")
    _assert_finite_nonzero_grad(model.backbone_net._proj_num_token_output_logit.weight,
                                "native root duration classifier")


def test_native_pose_training_step_runs_original_mask_mix_with_frozen_tokenizer():
    torch.manual_seed(1701)
    np.random.seed(1701)
    representation, batch = _motion_and_representation()
    tokenizer_config = native_config("tokenizer", max_steps=2_000_001)
    tokenizer = build_native_model("tokenizer", representation, tokenizer_config)
    assert type(tokenizer.pose_net) is VQVAE
    config = native_config("pose", max_steps=2_000_001)
    model = build_native_model("pose", representation, config, tokenizer=tokenizer)
    assert type(model) is NativePoseModel
    assert model.backbone_net._args["n_embd"] == 1024
    assert model.backbone_net._args["n_layers"] == 16
    assert model.backbone_net._args["max_tokens"] == config.model.args.max_tokens == 16
    assert model.backbone_net.initted.item()
    assert model.supporting_nets["pose_net"] is tokenizer.pose_net
    _set_test_sampler_horizon(model)
    model.train()
    _attach_minimal_training_context(model)

    loss = model.training_step(batch, 0)
    assert loss is not None and torch.isfinite(loss)
    masked = batch["masked_token_mask"].view_as(batch["input_tokens"])
    incorrect = batch["incorrect_token_mask"].view_as(batch["input_tokens"])
    assert masked.any(), "original pose mask path did not mask any codebook tokens"
    assert incorrect.any(), "original pose mixing path did not sample incorrect tokens"
    assert torch.all(batch["input_tokens"][masked] == model.backbone_net.POSE_MASK_ID)
    assert torch.any(batch["input_tokens"][incorrect] != batch["groundtruth_pose_tokens"][incorrect])
    assert model.supporting_nets["pose_net"].training is False
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in model.supporting_nets["pose_net"].parameters())

    loss.backward()
    trainable_grads = [parameter.grad for parameter in model.backbone_net.parameters()
                       if parameter.requires_grad and parameter.grad is not None]
    assert trainable_grads, "native pose backbone received no gradients"
    assert all(torch.isfinite(grad).all() for grad in trainable_grads)
    assert any(grad.abs().sum() > 0 for grad in trainable_grads)
    assert all(parameter.grad is None for parameter in model.supporting_nets["pose_net"].parameters())

    del model, tokenizer
    gc.collect()


def test_paper_pose_training_step_accepts_twelve_frame_window():
    torch.manual_seed(4701)
    np.random.seed(4701)
    representation, batch = _motion_and_representation(frame_count=13)
    tokenizer = build_native_model("tokenizer", representation, paper_config("tokenizer"))
    config = paper_config("pose")
    assert config.model.args.min_tokens == 3
    model = build_native_model("pose", representation, config, tokenizer=tokenizer)
    model.args["max_tokens"] = model.args["min_tokens"]
    model.train()
    _attach_minimal_training_context(model)
    # Exercise the full keyframe curriculum, including up to eight middle poses
    # when this shortest window only provides four middle-frame slots.
    model._trainer.global_step = model.args["keyframe_num_warmup_steps"]

    loss = model.training_step(batch, 0)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    grads = [parameter.grad for parameter in model.backbone_net.parameters()
             if parameter.requires_grad and parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)
    assert all(parameter.grad is None for parameter in model.supporting_nets["pose_net"].parameters())
    del model, tokenizer
    gc.collect()
