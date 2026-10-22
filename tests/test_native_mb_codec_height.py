"""Core tests for native dense-root-height tokenizer conditioning."""

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "out" / "vendor" / "kimodo"))

from test_native_mb_stage_training import _motion_and_representation
from test_native_mb_terrain_integration import small_config
from motionbricks.data.native_mb_codec_height import (
    CODEC_HEIGHT_CONDITIONING,
    build_codec_height_tokenizer,
    enable_codec_height_conditioning,
    fixed_base_token_decoder_forward,
    rigid_vertical_pair,
)
from motionbricks.data.native_mb_models import build_native_model
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _small_tokenizer_config():
    config = small_config("tokenizer")
    network = config.model.pose_vqvae_network
    network.nb_code = 256
    network.code_dim = 32
    network.output_emb_width = 32
    network.width = 32
    network.depth = 1
    network.num_heads = 4
    network.dilation_growth_rate = 2
    return config


@pytest.fixture
def codec_pair():
    torch.manual_seed(2251)
    rep, batch = _motion_and_representation(frame_count=25)
    source_config = _small_tokenizer_config()
    source = build_native_model("tokenizer", rep, source_config)
    target_config = enable_codec_height_conditioning(source_config)
    target = build_codec_height_tokenizer(rep, source, target_config)
    frames = 24
    global_motion = batch["motion"][:, :frames]
    local_motion = rep.dual_rep.global_to_local(
        global_motion,
        is_normalized=True,
        to_normalize=True,
        lengths=torch.full((global_motion.shape[0],), frames),
    )
    return rep, source_config, source, target_config, target, global_motion, local_motion


def _state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_config_adapter_preserves_native_training_configuration():
    source = _small_tokenizer_config()
    adapted = enable_codec_height_conditioning(source)
    before = OmegaConf.to_container(source, resolve=False)
    after = OmegaConf.to_container(adapted, resolve=False)

    assert before["model"]["pose_vqvae_network"]["feature_mode"][3] == (
        "root_without_hip_height_without_heading"
    )
    assert after["model"]["pose_vqvae_network"]["feature_mode"][3] == "root_without_heading"
    assert after["model"]["args"]["codec_height_conditioning"] == CODEC_HEIGHT_CONDITIONING
    before["model"]["pose_vqvae_network"]["feature_mode"][3] = "root_without_heading"
    del after["model"]["args"]["codec_height_conditioning"]
    assert after == before
    assert source.model.pose_vqvae_network.feature_mode[3] == (
        "root_without_hip_height_without_heading"
    )


def test_rigid_vertical_pair_shifts_root_and_absolute_ric_y_for_fk_and_ric(codec_pair):
    rep, _, _, _, _, global_motion, local_motion = codec_pair
    global_motion = global_motion.repeat(2, 1, 1)
    local_motion = local_motion.repeat(2, 1, 1)
    delta = torch.tensor([0.35, -0.2])
    shifted_global, shifted_local = rigid_vertical_pair(
        global_motion, local_motion, rep, delta
    )
    global_position = np.asarray(rep.indices["global_root_pos"])
    planar_position = np.asarray(rep.indices["global_root_pos_2d"])
    global_y = int(next(index for index in global_position if index not in set(planar_position)))
    local_rep = rep.dual_rep.local_motion_rep
    local_y = int(local_rep.indices["global_root_y"][0])
    global_ric_y = np.asarray(rep.indices["ric_data"])[1::3]
    local_ric_y = np.asarray(local_rep.indices["ric_data"])[1::3]
    global_shift = np.concatenate(([global_y], global_ric_y))
    local_shift = np.concatenate(([local_y], local_ric_y))

    expected_global = global_motion.clone()
    expected_local = local_motion.clone()
    expected_global[..., global_shift] = shifted_global[..., global_shift]
    expected_local[..., local_shift] = shifted_local[..., local_shift]
    assert torch.equal(shifted_global, expected_global)
    assert torch.equal(shifted_local, expected_local)

    raw_before = rep.unnormalize(global_motion)
    raw_after = rep.unnormalize(shifted_global)
    expected_raw = raw_before.clone()
    expected_raw[..., global_shift] += delta[:, None, None]
    torch.testing.assert_close(raw_after, expected_raw, rtol=0, atol=2e-6)

    for motion_rep, before, after in (
        (rep, global_motion, shifted_global),
        (local_rep, local_motion, shifted_local),
    ):
        raw_before_rep = motion_rep.unnormalize(before)
        raw_after_rep = motion_rep.unnormalize(after)
        shift_indices = global_shift if motion_rep is rep else local_shift
        expected_raw_rep = raw_before_rep.clone()
        expected_raw_rep[..., shift_indices] += delta[:, None, None]
        torch.testing.assert_close(raw_after_rep, expected_raw_rep, rtol=0, atol=2e-6)
        unchanged = np.setdiff1d(motion_rep.indices["all"], shift_indices)
        assert torch.equal(before[..., unchanged], after[..., unchanged])
        for key in ("local_vel", "global_rot_data", "foot_contacts"):
            indices = motion_rep.indices[key]
            assert torch.equal(raw_before_rep[..., indices], raw_after_rep[..., indices])

        for joint_source in ("ric_data", "global_rot_data"):
            before_joints = motion_rep.inverse(
                before, is_normalized=True, return_all=True,
                joint_positions_from=joint_source,
            )["posed_joints"]
            after_joints = motion_rep.inverse(
                after, is_normalized=True, return_all=True,
                joint_positions_from=joint_source,
            )["posed_joints"]
            torch.testing.assert_close(
                after_joints[..., 1], before_joints[..., 1] + delta[:, None, None],
                rtol=0, atol=2e-5,
            )


def test_expanded_decoder_starts_at_exact_legacy_parity(codec_pair):
    rep, _, source, _, target, global_motion, local_motion = codec_pair
    source.pose_net.eval()
    target.pose_net.eval()
    target_net = target.pose_net
    source_net = source.pose_net
    target_indices = extract_feature_from_motion_rep(
        None, target_net.motion_rep, target_net.decoder_external_cond_feature_mode,
        fetch_feat_idx=True,
    )
    root_y_indices = np.asarray(target_net.motion_rep.indices["global_root_y"])
    root_y = int(root_y_indices[0])
    y_position = int(np.flatnonzero(target_indices == root_y)[0])

    with torch.no_grad():
        tokens = source_net.encode_into_idx(local_motion, fetch_overall_indices=False)
        has_target = torch.zeros(local_motion.shape[:2], dtype=torch.bool)
        has_target[:, (0, 11, 23)] = True
        source_target_cond = extract_feature_from_motion_rep(
            local_motion, source_net.motion_rep,
            source_net.decoder_target_cond_feature_mode,
        )
        source_external = extract_feature_from_motion_rep(
            local_motion, source_net.motion_rep,
            source_net.decoder_external_cond_feature_mode,
        )
        target_external = extract_feature_from_motion_rep(
            local_motion, target_net.motion_rep,
            target_net.decoder_external_cond_feature_mode,
        )
        target_external[..., y_position] = torch.randn_like(target_external[..., y_position]) * 7
        legacy = source_net.forward_decoder(
            tokens, target_cond=source_target_cond, has_target_cond=has_target,
            external_cond=source_external, use_overall_indices=False,
        )["recon_state"]
        adapted = target_net.forward_decoder(
            tokens, target_cond=source_target_cond, has_target_cond=has_target,
            external_cond=target_external, use_overall_indices=False,
        )["recon_state"]

    assert torch.equal(adapted, legacy)
    assert target.args["codec_height_conditioning"] == CODEC_HEIGHT_CONDITIONING
    assert target_net.decoder_external_cond_feature_mode == "root_without_heading"


def test_expansion_handles_optional_last_fusion_layer():
    rep, _ = _motion_and_representation(frame_count=25)
    config = _small_tokenizer_config()
    config.model.pose_vqvae_network.cond_fusion_last_layer = True
    source = build_native_model("tokenizer", rep, config)
    target = build_codec_height_tokenizer(
        rep, source, enable_codec_height_conditioning(config)
    )
    source_decoder = source.pose_net.decoder
    target_decoder = target.pose_net.decoder
    last_linear_index = source_decoder._down_t * 2
    source_linear = source_decoder.external_cond_blocks[last_linear_index]
    target_linear = target_decoder.external_cond_blocks[last_linear_index]
    source_indices = extract_feature_from_motion_rep(
        None, source.pose_net.motion_rep,
        source.pose_net.decoder_external_cond_feature_mode,
        fetch_feat_idx=True,
    )
    target_indices = extract_feature_from_motion_rep(
        None, target.pose_net.motion_rep,
        target.pose_net.decoder_external_cond_feature_mode,
        fetch_feat_idx=True,
    )
    y_index = int(target.pose_net.motion_rep.indices["global_root_y"][0])
    y_position = int(np.flatnonzero(target_indices == y_index)[0])
    hidden_width = source_linear.out_features

    assert source_decoder._COND_FUSION_LAST_LAYER
    assert len(target_decoder.external_cond_blocks) == 2 * (target_decoder._down_t + 1)
    torch.testing.assert_close(
        target_linear.weight[:, :hidden_width],
        source_linear.weight[:, :hidden_width], rtol=0, atol=0,
    )
    for source_position, source_index in enumerate(source_indices):
        target_position = int(np.flatnonzero(target_indices == source_index)[0])
        torch.testing.assert_close(
            target_linear.weight[:, hidden_width + target_position],
            source_linear.weight[:, hidden_width + source_position], rtol=0, atol=0,
        )
    assert torch.count_nonzero(target_linear.weight[:, hidden_width + y_position]) == 0


def test_decoder_only_step_trains_height_columns_and_preserves_frozen_state(codec_pair):
    rep, _, _, _, target, global_motion, local_motion = codec_pair
    shifted_global, shifted_local = rigid_vertical_pair(
        global_motion, local_motion, rep, torch.tensor([0.4])
    )
    net = target.pose_net
    target.train()
    trainable = [name for name, parameter in net.named_parameters() if parameter.requires_grad]
    assert trainable and all(name.startswith("decoder.") for name in trainable)

    encoder_before = _state_hash(net.encoder)
    quantizer_before = _state_hash(net.quantizer)
    has_target = torch.zeros(local_motion.shape[:2], dtype=torch.bool)
    has_target[:, (0, 11, 23)] = True
    output = fixed_base_token_decoder_forward(
        target, local_motion, shifted_local, has_target
    )
    assert set(output) == {"recon_state", "tokens"}
    assert output["tokens"].shape[-1] == net._num_heads

    # This is the native local-motion SmoothL1 reconstruction objective; no
    # synthetic commitment value is added by the offline fixed-token helper.
    loss = torch.nn.functional.smooth_l1_loss(
        output["recon_state"], shifted_local
    )
    assert torch.isfinite(loss)
    optimizer = torch.optim.SGD(
        [parameter for parameter in net.parameters() if parameter.requires_grad], lr=1e-2
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    external_indices = extract_feature_from_motion_rep(
        None, net.motion_rep, net.decoder_external_cond_feature_mode,
        fetch_feat_idx=True,
    )
    root_y_indices = np.asarray(net.motion_rep.indices["global_root_y"])
    root_y = int(root_y_indices[0])
    y_position = int(np.flatnonzero(external_indices == root_y)[0])
    first_linear = net.decoder.external_cond_blocks[0]
    hidden_width = first_linear.out_features
    frames_per_position = 2 ** net.decoder._down_t
    y_columns = [
        hidden_width + frame * len(external_indices) + y_position
        for frame in range(frames_per_position)
    ]
    y_gradient = first_linear.weight.grad[:, y_columns]
    assert torch.isfinite(y_gradient).all()
    assert y_gradient.abs().sum() > 0

    optimizer.step()
    assert _state_hash(net.encoder) == encoder_before
    assert _state_hash(net.quantizer) == quantizer_before
