"""Train MB token selection on executed history, without changing runtime decoding.

The optional decoded-motion objective uses hard argmax codes in its forward pass.
Its softmax/codebook straight-through gradient is a biased training surrogate,
not a claim that MotionBricks or DART uses this particular estimator.
"""

import torch

from motionbricks.data.bones_control import decode_tokens
from motionbricks.data.soma_decoder_refinement import deployed_request, history_in_input_frame
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


@torch.no_grad()
def deployed_pose_request(inference, source, history, commit_frames=4):
    calls, outputs = [], []

    def capture(module, args):
        calls.append(tuple(value.detach().clone() if isinstance(value, torch.Tensor) else value
                           for value in args))

    hook = inference._pose_model.backbone_net.register_forward_pre_hook(capture)
    output_hook = inference._pose_model.backbone_net.register_forward_hook(
        lambda module, args, output: outputs.append(output["pose_logits"].detach().clone()))
    try:
        decoder, generated = deployed_request(inference, source, history, commit_frames=commit_frames)
    finally:
        hook.remove()
        output_hook.remove()
    if len(calls) != 1:
        raise ValueError("This paired trial requires the deployed one-pass argmax Pose route")
    return calls[0], decoder, generated, outputs[0]


def straight_through_codes(logits, quantizer, hard_ids=None):
    if logits.ndim != 4 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite [batch, tokens, heads, codes] logits")
    codes = quantizer.vq.codebook.detach()
    if codes.shape[:2] != logits.shape[-2:]:
        raise ValueError("Logits and native per-head codebooks disagree")
    ids = logits.argmax(-1) if hard_ids is None else hard_ids.detach()
    if ids.shape != logits.shape[:-1]:
        raise ValueError("Deployed hard token shape differs from logits")
    hard = quantizer.dequantize(ids, use_overall_indices=False).detach()
    soft = torch.einsum("bthk,hkd->bthd", logits.softmax(-1), codes).flatten(-2)
    # The parenthesized difference is exactly zero in the forward pass.
    return hard + (soft - soft.detach()), ids


def decode_logits(tokenizer, logits, request, use_deployed_ids=False):
    full, mask, _, local_root, token_mask = request
    latent, ids = straight_through_codes(logits, tokenizer.quantizer, request[2] if use_deployed_ids else None)
    external = extract_feature_from_motion_rep(local_root, tokenizer.motion_rep,
                                               tokenizer.decoder_external_cond_feature_mode)
    condition = extract_feature_from_motion_rep(full, tokenizer.motion_rep,
                                                tokenizer.decoder_target_cond_feature_mode)
    decoded = tokenizer.decoder(latent.permute(0, 2, 1).contiguous(), external, condition,
                                mask, token_mask=token_mask).permute(0, 2, 1)
    output = tokenizer._apply_condition({"recon_state": decoded}, external)
    return output["recon_state"], ids


def require_deployed_tokens_equal(ids, request):
    active = request[4][..., None].expand_as(ids)
    if not torch.equal(ids[active], request[2][active]):
        raise RuntimeError("Differentiable training differs from deployed ACTIVE argmax tokens")


def pose_rollout_loss(pose, trainer, pose_request, decoder_request, source_local,
                      local_target, global_target, confidence, labels, decoded_weight, deployed_logits=None,
                      history=None):
    if decoded_weight not in (0., 1.):
        raise ValueError("The paired trial fixes decoded-motion coefficient at zero or one")
    output = pose.backbone_net(*pose_request)
    logits = output["pose_logits"]
    logit_delta, argmax_delta = 0., 0
    if deployed_logits is not None:
        active = decoder_request[4][..., None, None].expand_as(logits)
        # Fused inference and autograd kernels need not be bit-identical. Verify
        # float32 closeness, but keep the ACTUAL deployed hard IDs in the decoder.
        torch.testing.assert_close(logits.detach()[active], deployed_logits[active], atol=1e-4, rtol=1e-4)
        logit_delta = float((logits.detach()[active] - deployed_logits[active]).abs().max())
        token_active = decoder_request[4][..., None].expand_as(logits[..., 0])
        argmax_delta = int(((logits.argmax(-1) != decoder_request[2]) & token_active).sum())
    with torch.no_grad():
        target = pose.pose_net.encode_into_idx(source_local, fetch_overall_indices=False)
    if logits.shape[1] < target.shape[1]:
        raise ValueError("Runtime Pose padding is shorter than the source window")
    padded = torch.zeros_like(logits[..., 0], dtype=torch.long)
    padded[:, :target.shape[1]] = target
    focus = torch.zeros_like(padded, dtype=torch.bool)
    focus[:, :target.shape[1]] = True
    ce = pose.loss({"groundtruth_pose_tokens": padded, "focused_token_mask": focus}, output)["loss"]
    with torch.set_grad_enabled(bool(decoded_weight)):
        decoded, ids = decode_logits(pose.pose_net, logits, decoder_request,
                                    use_deployed_ids=deployed_logits is not None)
        decoded = decoded[:, :local_target.shape[1]]
        batch = {"local_motions": local_target.detach(),
            "global_motions": global_target.detach(), "contact_confidence": confidence.detach(),
            **labels}
        if history is not None:
            batch["executed_history"] = history_in_input_frame(history)
        losses = trainer.loss(batch, {"recon_state": decoded, "l_commit": decoded.sum() * 0., "perplexity": 0.}, None)
    total = ce + decoded_weight * losses["loss"]
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite Pose continuation objective")
    return total, {"pose_ce": ce.detach(), "decoded_motion_loss": losses["loss"].detach(),
                   "decoded_weight": decoded_weight,
                   "runtime_logit_max_error": logit_delta, "runtime_active_argmax_differences": argmax_delta,
                   **{key: value.detach() if isinstance(value, torch.Tensor) else value
                      for key, value in losses.items() if key != "loss"}}, ids
