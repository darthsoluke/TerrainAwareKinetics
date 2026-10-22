"""Experimental, isolated native VQ-VAE conditioning with dense root height.

This core is not runtime-bound or deployment-validated, and does not establish
that dense height alone resolves root-height error. Decoder reconstruction is
not rendered/played FK feedback; feedback training must first re-encode the
actual played FK through the official MotionRep path.
"""

from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from motionbricks.data.native_mb_models import build_native_model
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


CODEC_HEIGHT_CONDITIONING = "native-dense-root-height-v1"
SOURCE_EXTERNAL_FEATURE_MODE = "root_without_hip_height_without_heading"
HEIGHT_EXTERNAL_FEATURE_MODE = "root_without_heading"


def enable_codec_height_conditioning(tokenizer_config):
    """Clone a native tokenizer config and add only dense root-height input."""
    config = deepcopy(tokenizer_config)
    try:
        feature_modes = config["model"]["pose_vqvae_network"]["feature_mode"]
        args = config["model"]["args"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Expected a native tokenizer config") from exc

    if len(feature_modes) != 4:
        raise ValueError("Native tokenizer feature_mode must contain four entries")
    current_mode = feature_modes[3]
    if current_mode == SOURCE_EXTERNAL_FEATURE_MODE:
        feature_modes[3] = HEIGHT_EXTERNAL_FEATURE_MODE
    elif current_mode != HEIGHT_EXTERNAL_FEATURE_MODE:
        raise ValueError(
            "Dense height conditioning expects the native tokenizer's "
            f"{SOURCE_EXTERNAL_FEATURE_MODE!r} external feature mode"
        )
    args["codec_height_conditioning"] = CODEC_HEIGHT_CONDITIONING
    return config


def _pose_net(tokenizer):
    net = getattr(tokenizer, "pose_net", tokenizer)
    if not isinstance(net, nn.Module) or not hasattr(net, "decoder"):
        raise TypeError("Expected a native tokenizer model or VQVAE network")
    return net


def _feature_indices(net, feature_mode):
    indices = extract_feature_from_motion_rep(
        None, net.motion_rep, feature_mode, fetch_feat_idx=True
    )
    return np.asarray(indices, dtype=np.int64).reshape(-1)


def _one_index(indices, label):
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected exactly one {label} feature, got {values.size}")
    return int(values[0])


def _global_root_y_index(rep):
    global_position = np.asarray(rep.indices["global_root_pos"], dtype=np.int64).reshape(-1)
    planar_position = np.asarray(rep.indices["global_root_pos_2d"], dtype=np.int64).reshape(-1)
    height = [int(index) for index in global_position if index not in set(planar_position.tolist())]
    if len(height) != 1:
        raise ValueError("Expected exactly one vertical component in global_root_pos")
    return height[0]


def _expanded_external_weights(source_net, target_net):
    source_decoder, target_decoder = source_net.decoder, target_net.decoder
    if source_decoder._down_t != target_decoder._down_t:
        raise ValueError("Source and target tokenizer downsampling must match")
    if source_decoder._COND_FUSION_LAST_LAYER != target_decoder._COND_FUSION_LAST_LAYER:
        raise ValueError("Source and target tokenizer fusion layout must match")
    if source_net.decoder_external_cond_feature_mode != SOURCE_EXTERNAL_FEATURE_MODE:
        raise ValueError("Source tokenizer does not use the native planar external condition")
    if target_net.decoder_external_cond_feature_mode != HEIGHT_EXTERNAL_FEATURE_MODE:
        raise ValueError("Target tokenizer must use root_without_heading external conditioning")

    source_indices = _feature_indices(
        source_net, source_net.decoder_external_cond_feature_mode
    )
    target_indices = _feature_indices(
        target_net, target_net.decoder_external_cond_feature_mode
    )
    root_y_index = _one_index(target_net.motion_rep.indices["global_root_y"], "global_root_y")
    if source_indices.size != 2 or target_indices.size != source_indices.size + 1:
        raise ValueError("Expected the native planar condition plus one global root-height feature")
    if len(set(source_indices.tolist())) != source_indices.size:
        raise ValueError("Source external feature indices must be unique")
    extras = [int(index) for index in target_indices if index not in set(source_indices.tolist())]
    if extras != [root_y_index]:
        raise ValueError("The only added external feature must be global_root_y")

    target_positions = []
    for source_index in source_indices:
        matches = np.flatnonzero(target_indices == source_index)
        if matches.size != 1:
            raise ValueError("Target external mode must retain each source feature exactly once")
        target_positions.append(int(matches[0]))

    source_width = len(source_indices)
    target_width = len(target_indices)
    layer_count = source_decoder._down_t + int(source_decoder._COND_FUSION_LAST_LAYER)
    if not hasattr(source_decoder, "external_cond_blocks") or not hasattr(
        target_decoder, "external_cond_blocks"
    ):
        raise ValueError("Native tokenizer decoder must expose external condition blocks")

    expanded = {}
    for layer in range(layer_count):
        module_index = layer * 2
        source_linear = source_decoder.external_cond_blocks[module_index]
        target_linear = target_decoder.external_cond_blocks[module_index]
        if not isinstance(source_linear, nn.Linear) or not isinstance(target_linear, nn.Linear):
            raise TypeError("External condition blocks must alternate Linear and activation modules")
        if source_linear.out_features != target_linear.out_features:
            raise ValueError("External condition hidden width must be preserved")

        frames_per_position = 2 ** (source_decoder._down_t - layer)
        hidden_width = source_linear.out_features
        expected_source_inputs = hidden_width + frames_per_position * source_width
        expected_target_inputs = hidden_width + frames_per_position * target_width
        if source_linear.in_features != expected_source_inputs:
            raise ValueError("Unexpected source external condition layout")
        if target_linear.in_features != expected_target_inputs:
            raise ValueError("Unexpected target external condition layout")

        source_weight = source_linear.weight.detach()
        expanded_weight = torch.zeros_like(target_linear.weight)
        expanded_weight[:, :hidden_width].copy_(source_weight[:, :hidden_width])
        for frame in range(frames_per_position):
            for source_position, target_position in enumerate(target_positions):
                source_column = hidden_width + frame * source_width + source_position
                target_column = hidden_width + frame * target_width + target_position
                expanded_weight[:, target_column].copy_(source_weight[:, source_column])
        expanded[f"decoder.external_cond_blocks.{module_index}.weight"] = expanded_weight

    return expanded


def _copy_tokenizer_state(source_tokenizer, target_tokenizer):
    source_net, target_net = _pose_net(source_tokenizer), _pose_net(target_tokenizer)
    source_state = source_net.state_dict()
    target_state = target_net.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError("Source and target native tokenizer state keys do not match")

    expanded_weights = _expanded_external_weights(source_net, target_net)
    if set(expanded_weights) != {
        f"decoder.external_cond_blocks.{layer * 2}.weight"
        for layer in range(
            source_net.decoder._down_t
            + int(source_net.decoder._COND_FUSION_LAST_LAYER)
        )
    }:
        raise RuntimeError("Did not identify every external condition Linear weight")

    copied_state = {}
    for key, source_value in source_state.items():
        target_value = target_state[key]
        if key in expanded_weights:
            value = expanded_weights[key]
            if value.shape != target_value.shape:
                raise ValueError(f"Expanded weight has an unexpected shape: {key}")
            copied_state[key] = value
        else:
            if source_value.shape != target_value.shape:
                raise ValueError(f"Unexpected tokenizer state shape change: {key}")
            copied_state[key] = source_value.detach().clone()

    target_net.load_state_dict(copied_state, strict=True)
    return copied_state


def _freeze_to_decoder(tokenizer):
    net = _pose_net(tokenizer)
    for name, parameter in net.named_parameters():
        parameter.requires_grad_(name.startswith("decoder."))
    net.encoder.eval()
    net.quantizer.eval()


def build_codec_height_tokenizer(motion_rep, source_tokenizer, tokenizer_config):
    """Build the native tokenizer, transplant its weights, and freeze its encoder."""
    config = enable_codec_height_conditioning(tokenizer_config)
    tokenizer = build_native_model("tokenizer", motion_rep, config)
    _copy_tokenizer_state(source_tokenizer, tokenizer)
    _freeze_to_decoder(tokenizer)
    return tokenizer


def _batch_delta(delta_m, batch_size, device, dtype):
    delta = torch.as_tensor(delta_m, device=device, dtype=dtype)
    if not torch.isfinite(delta).all():
        raise ValueError("delta_m must be finite")
    if delta.numel() == 1:
        return delta.reshape(1).expand(batch_size)
    if delta.numel() != batch_size:
        raise ValueError("delta_m must be a scalar or contain one value per batch item")
    return delta.reshape(batch_size)


def rigid_vertical_pair(global_normalized, local_normalized, rep, delta_m):
    """Apply a rigid world-Y translation to normalized global and local motions.

    Native ``ric_data`` stores each non-root joint's world Y directly while X/Z
    are root-relative. Shift those Y channels with root height so both RIC
    reconstruction and rotation-based FK describe the same translated pose.
    """
    if (global_normalized.ndim != 3 or local_normalized.ndim != 3
            or global_normalized.shape[:2] != local_normalized.shape[:2]):
        raise ValueError("Expected aligned global and local motions with shape [B,T,D]")
    if not torch.is_floating_point(global_normalized) or not torch.is_floating_point(local_normalized):
        raise ValueError("Normalized motions must be floating-point tensors")

    local_rep = rep.dual_rep.local_motion_rep
    batch_size = global_normalized.shape[0]
    global_y = _global_root_y_index(rep)
    local_y = _one_index(local_rep.indices["global_root_y"], "local global_root_y")
    global_ric_y = np.asarray(rep.indices["ric_data"], dtype=np.int64).reshape(-1)[1::3]
    local_ric_y = np.asarray(local_rep.indices["ric_data"], dtype=np.int64).reshape(-1)[1::3]
    global_shift_indices = np.concatenate((np.asarray([global_y]), global_ric_y))
    local_shift_indices = np.concatenate((np.asarray([local_y]), local_ric_y))
    global_delta = _batch_delta(
        delta_m, batch_size, global_normalized.device, global_normalized.dtype
    )[:, None, None]
    local_delta = _batch_delta(
        delta_m, batch_size, local_normalized.device, local_normalized.dtype
    )[:, None, None]

    global_raw = rep.unnormalize(global_normalized.clone())
    local_raw = local_rep.unnormalize(local_normalized.clone())
    global_raw[..., global_shift_indices] += global_delta
    local_raw[..., local_shift_indices] += local_delta

    # Keep every non-world-Y normalized feature bit-identical to its input.
    shifted_global = global_normalized.clone()
    shifted_local = local_normalized.clone()
    shifted_global[..., global_shift_indices] = rep.normalize(global_raw)[..., global_shift_indices]
    shifted_local[..., local_shift_indices] = local_rep.normalize(local_raw)[..., local_shift_indices]
    return shifted_global, shifted_local


@contextmanager
def _frozen_encoder_eval(net):
    if any(parameter.requires_grad for parameter in net.encoder.parameters()):
        raise ValueError("Tokenizer encoder must be frozen before fixed-token decoding")
    if any(parameter.requires_grad for parameter in net.quantizer.parameters()):
        raise ValueError("Tokenizer quantizer must be frozen before fixed-token decoding")
    encoder_training = net.encoder.training
    quantizer_training = net.quantizer.training
    net.encoder.eval()
    net.quantizer.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        net.encoder.train(encoder_training)
        net.quantizer.train(quantizer_training)


def fixed_base_token_decoder_forward(
    tokenizer,
    source_local_normalized,
    shifted_local_normalized,
    has_target_cond,
    token_mask=None,
):
    """Test/offline helper: encode base tokens and decode shifted conditions.

    This helper uses a known shifted motion as decoder conditioning. It is not
    suitable for runtime inference, where only predicted root and pose values
    are available. Its reconstruction is not observed-FK feedback; any
    feedback-training path must re-encode the actual played FK through the
    official MotionRep first.
    """
    net = _pose_net(tokenizer)
    if source_local_normalized.shape != shifted_local_normalized.shape:
        raise ValueError("Base and shifted local motions must have the same shape")
    if source_local_normalized.ndim != 3:
        raise ValueError("Local motions must have shape [B,T,D]")
    batch_size, frame_count = source_local_normalized.shape[:2]
    if frame_count % (2 ** net._down_t):
        raise ValueError("Frame count must be divisible by the tokenizer downsampling factor")
    if has_target_cond.shape != (batch_size, frame_count) or has_target_cond.dtype != torch.bool:
        raise ValueError("has_target_cond must be a boolean [B,T] mask")
    if (net.decoder_external_cond_feature_mode != HEIGHT_EXTERNAL_FEATURE_MODE
            or "root_without_heading" not in net.ALLOWED_FEATURE_MODE):
        raise ValueError("Expected the dense root-height tokenizer feature mode")

    with _frozen_encoder_eval(net):
        tokens = net.encode_into_idx(source_local_normalized, fetch_overall_indices=False)

    external_cond = extract_feature_from_motion_rep(
        shifted_local_normalized,
        net.motion_rep,
        net.decoder_external_cond_feature_mode,
    )
    decoded = net.forward_decoder(
        tokens,
        target_cond=shifted_local_normalized,
        has_target_cond=has_target_cond,
        external_cond=external_cond,
        use_overall_indices=False,
        token_mask=token_mask,
    )
    return {"recon_state": decoded["recon_state"], "tokens": tokens}
