"""Shared precision policy for native MotionBricks evaluation."""

from contextlib import contextmanager


@contextmanager
def full_precision_evaluation():
    """Use highest matmul precision for evaluation, then restore training policy."""
    import torch

    previous_matmul = torch.get_float32_matmul_precision()
    previous_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(previous_matmul)
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def verify_tokenizer_decode_parity(network, local, indices, prediction, decoded,
                                   target_cond=None, has_target_cond=None,
                                   external_cond=None):
    """Check token identity and decoder routing without changing official outputs.

    MB's quantizer returns x + (code - x).detach(), even in eval mode. FP32
    cancellation can perturb that value before the decoder, whereas token
    inference retrieves code directly. A raw mismatch is accepted only after
    replaying both exact inputs through the unchanged official decoder.
    """
    import torch
    from motionbricks.helper.data_training_util import convert_sparse_cond_to_dense_cond_if_needed

    if network.training:
        raise ValueError("Token-decode parity requires eval mode")
    if not torch.isfinite(prediction).all() or not torch.isfinite(decoded).all():
        raise ValueError("Nonfinite tokenizer prediction or token decode")
    raw_error = float((prediction - decoded).abs().max())
    result = {"raw_max_error": raw_error, "roundoff_replay_required": False}
    if raw_error <= 1e-5:
        return result

    with torch.no_grad():
        encoded = network.encoder(network.extract_feature(
            local, network.encoder_input_feature_mode).permute(0, 2, 1))
        repeated_indices = network.quantizer.forward_into_idx(encoded, fetch_overall_indices=False)
        if not torch.equal(indices, repeated_indices):
            raise ValueError("Token-decode parity failure: encoder token identities differ")
        exact = network.quantizer.dequantize(indices, use_overall_indices=False).permute(0, 2, 1).contiguous()
        quantized = network.quantizer(encoded)[0]
        replay_input = encoded + (exact - encoded).detach()
        if not torch.isfinite(quantized).all() or not torch.equal(quantized, replay_input):
            raise ValueError("Token-decode parity failure: mismatch is not the official straight-through arithmetic")
        roundoff_bound = torch.finfo(encoded.dtype).eps * (2 * encoded.abs() + exact.abs())
        if not ((quantized - exact).abs() <= roundoff_bound).all():
            raise ValueError("Token-decode parity failure: latent difference exceeds floating-point roundoff bound")
        if target_cond is not None:
            target_cond, has_target_cond = convert_sparse_cond_to_dense_cond_if_needed(
                target_cond, has_target_cond, local.shape[1])
            target_cond = network.extract_feature(target_cond, network.decoder_target_cond_feature_mode)
        replay = network.decoder(replay_input, external_cond, target_cond, has_target_cond).permute(0, 2, 1)
        exact_replay = network.decoder(exact, external_cond, target_cond, has_target_cond).permute(0, 2, 1)
        replay_error = float((prediction - replay).abs().max())
        exact_error = float((decoded - exact_replay).abs().max())
        if not torch.isfinite(replay).all() or not torch.isfinite(exact_replay).all() or max(replay_error, exact_error) > 1e-5:
            raise ValueError(f"Token-decode parity failure: decoder replay errors {replay_error}, {exact_error}")
    result.update({"roundoff_replay_required": True,
                   "latent_max_error": float((quantized - exact).abs().max()),
                   "straight_through_decoder_replay_error": replay_error,
                   "token_decoder_replay_error": exact_error,
                   "token_identity_equal": True})
    return result
