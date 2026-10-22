"""Coupled MB Root/Pose continuation with the unchanged deployed forward path.

The only gradient surrogate replaces hard code embeddings with the existing
softmax straight-through estimator. Its forward values are the actual deployed
argmax codes; its backward gradient is biased, not an official MB/DART objective.
Root derivatives, conditions, masks and root replacement policy remain native.
"""

from contextlib import contextmanager

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from motionbricks.data.soma_decoder_refinement import history_in_input_frame, source_navigation_inputs
from motionbricks.data.soma_pose_rollout import straight_through_codes


CONFIG = {"pose_token_sampling_use_argmax": True, "final_root_pred_mode": "from_pose_module"}


@contextmanager
def paired_attention_kernel():
    """Use one FP32 attention algorithm on both sides of the autograd check."""
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


@torch.no_grad()
def deployed_reference(inference, source, history, commit_frames=8):
    inputs, frames = source_navigation_inputs(source, history, inference.motion_rep,
        getattr(inference, "history_geometry", "predicted_features"))
    if commit_frames != 8 or frames % 4 or frames < 12:
        raise ValueError("This trial requires the native eight-frame publication contract")
    roots, poses = [], []
    handles = [
        inference._root_model.backbone_net.register_forward_hook(
            lambda m, a, o: roots.append(o["pred_global_root_values"].detach().clone())),
        inference._pose_model.backbone_net.register_forward_hook(
            lambda m, a, o: poses.append(o["pose_logits"].detach().clone())),
    ]
    info = {"capture_decoder_inputs": True, "capture_decoder_output": True}
    try:
        with paired_attention_kernel():
            predicted, _ = inference.predict(*inputs, frames // 4, config=dict(CONFIG), info=info)
    finally:
        for handle in handles:
            handle.remove()
    if len(roots) != 1 or len(poses) != 1:
        raise ValueError("Only the deployed one-pass native Root/Pose route is supported")
    return {"root": roots[0], "logits": poses[0], "decoder": info["decoder_inputs"],
            "decoded": info["decoder_output"].detach().clone(),
            "history": predicted[:, commit_frames:commit_frames + 4].detach().clone()}


def differentiable_native_prediction(inference, source, history, reference):
    """Execute native predict, preserving the Root graph through both downstream paths."""
    if inference.training or any(m.training for m in (
            inference._root_model.backbone_net, inference._pose_model.backbone_net,
            inference._vqvae_pose_model)):
        raise ValueError("Paired continuation adaptation requires deterministic eval mode")
    inputs, frames = source_navigation_inputs(source, history, inference.motion_rep,
        getattr(inference, "history_geometry", "predicted_features"))
    root_outputs, pose_outputs = [], []
    tokenizer = inference._vqvae_pose_model

    def attach_token_gradient(module, args):
        if len(pose_outputs) != 1:
            raise ValueError("Decoder must follow exactly one native Pose prediction")
        logits = pose_outputs[0]["pose_logits"]
        codes, _ = straight_through_codes(logits, tokenizer.quantizer, reference["decoder"][2])
        return (codes.permute(0, 2, 1).contiguous(), *args[1:])

    handles = [
        inference._root_model.backbone_net.register_forward_hook(
            lambda m, a, o: root_outputs.append(o["pred_global_root_values"])),
        inference._pose_model.backbone_net.register_forward_hook(
            lambda m, a, o: pose_outputs.append(o)),
        tokenizer.decoder.register_forward_pre_hook(attach_token_gradient),
    ]
    info = {"capture_decoder_inputs": True, "capture_decoder_output": True}
    try:
        with paired_attention_kernel():
            inference.predict(*inputs, frames // 4, config=dict(CONFIG), info=info)
    finally:
        for handle in handles:
            handle.remove()
    if len(root_outputs) != 1 or len(pose_outputs) != 1:
        raise ValueError("Native Root/Pose prediction count changed")
    active = reference["decoder"][4]
    logits = pose_outputs[0]["pose_logits"]
    selected = active[..., None, None].expand_as(logits)
    decoded = info["decoder_output"]
    active_ids = active[..., None].expand_as(logits[..., 0])
    numerical = {
        "runtime_root_max_error": (root_outputs[0].detach()[:, :frames] - reference["root"][:, :frames]).abs().max(),
        "runtime_logit_max_error": (logits.detach()[selected] - reference["logits"][selected]).abs().max(),
        "runtime_decoded_max_error": (decoded.detach()[:, :frames] - reference["decoded"][:, :frames]).abs().max(),
        "runtime_active_argmax_differences": ((logits.argmax(-1) != reference["decoder"][2]) & active_ids).sum(),
    }
    try:
        torch.testing.assert_close(logits.detach()[selected], reference["logits"][selected], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(root_outputs[0].detach()[:, :frames], reference["root"][:, :frames], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(decoded.detach()[:, :frames], reference["decoded"][:, :frames], atol=1e-4, rtol=1e-4)
    except AssertionError as error:
        raise RuntimeError(f"Native gradient/reference mismatch: { {k: float(v) for k, v in numerical.items()} }") from error
    return decoded[:, :frames], pose_outputs[0], numerical


def joint_rollout_loss(inference, trainer, source, history, reference, source_local,
                       local_target, global_target, confidence, labels, supervision_policy="source_v2"):
    if supervision_policy not in ("source_v2", "legacy_splice"):
        raise ValueError("Unknown rollout supervision policy")
    decoded, pose_output, numerical = differentiable_native_prediction(inference, source, history, reference)
    pose = inference._pose_model
    with torch.no_grad():
        target = pose.pose_net.encode_into_idx(source_local.detach(), fetch_overall_indices=False)
    logits = pose_output["pose_logits"]
    if logits.shape[1] < target.shape[1] or decoded.shape != local_target.shape:
        raise ValueError("Decoded/source frame contract or token padding disagrees")
    padded = torch.zeros_like(logits[..., 0], dtype=torch.long)
    padded[:, :target.shape[1]] = target
    focus = torch.zeros_like(padded, dtype=torch.bool)
    focus[:, :target.shape[1]] = True
    ce = pose.loss({"groundtruth_pose_tokens": padded, "focused_token_mask": focus}, pose_output)["loss"]
    batch = {"local_motions": local_target.detach(), "global_motions": global_target.detach(),
                          "contact_confidence": confidence.detach(),
                          **{key: value.detach() for key, value in labels.items()}}
    if supervision_policy == "source_v2":
        batch["executed_history"] = history_in_input_frame(history)
    losses = trainer.loss(batch,
                         {"recon_state": decoded, "l_commit": decoded.sum() * 0., "perplexity": 0.}, None)
    total = ce + losses["loss"]
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite coupled continuation objective")
    return total, {"pose_ce": ce.detach(), "decoded_motion_loss": losses["loss"].detach(),
                   **numerical, **{key: value.detach() if isinstance(value, torch.Tensor) else value
                                  for key, value in losses.items() if key != "loss"}}
