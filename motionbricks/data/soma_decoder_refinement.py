"""MB joint root/body decoding with DART-inspired scheduled rollout history.

No output correction is performed. Future targets retain the dataset's world
frame and clock: generated drift is not hidden by transporting targets to it.
This two-primitive decoder adaptation is not DART's diffusion training algorithm.
"""

import torch

from motionbricks.data.bones_control import decode_tokens
from motionbricks.data.bones_rollout import control_inputs, observed_history_features


def rollout_probability(step, teacher_steps=2500, ramp_steps=2500):
    if step < 0 or teacher_steps < 1 or ramp_steps < 1:
        raise ValueError("Invalid scheduled training lengths")
    return min(1., max(0., (step - teacher_steps) / ramp_steps))


def source_target_with_history(source, history, representation, *, legacy_splice=False):
    """Source-only targets in the input origin; history is a condition, not truth.

    Local integration discards the source's initial XZ offset. Geometric losses
    must therefore use global_target, not inverse(local_target), during rollout.
    legacy_splice exists only for explicitly recorded regression comparisons.
    """
    if source.ndim != 3 or source.shape[1] < 9 or history.shape != source[:, :4].shape:
        raise ValueError("Expected four raw history frames and a longer source window")
    target = source.detach().clone()
    if legacy_splice:
        target[:, :4] = history.detach()
    # Native local-root integration starts at zero. Apply the SAME translation
    # to the complete target and actual context, without yaw or floor changes.
    target[..., [0, 2]] -= history[:, :1, [0, 2]].detach()
    frames = target.shape[1] - 1
    lengths = torch.full((len(target),), frames + 1, device=target.device)
    global_target = representation.normalize(target[:, :frames])
    local_target = representation.dual_rep.global_to_local(target, False, True, lengths=lengths)[:, :frames]
    return global_target, local_target


def history_in_input_frame(history):
    """The SAME origin as native inference; never change height or heading."""
    result = history.detach().clone()
    result[..., [0, 2]] -= history[:, :1, [0, 2]].detach()
    return result


def source_label_window(labels, start, frames, context_is_generated=False):
    """One source sampling index for frames/intervals/vertices; never prediction masks."""
    result = {}
    for side in ("left", "right"):
        for key in ("frame_state", "interval_state", "region_measured", "vertex_measured"):
            name = f"sole_{key}_{side}"
            count = frames if key == "frame_state" else frames - 1
            values = labels[name][start:start + count].detach().clone()
            if len(values) != count:
                raise ValueError(f"{name} shorter than the selected source window")
            if context_is_generated and key in ("region_measured", "vertex_measured"):
                # Source labels do not describe generated context or its splice.
                # This fixed exclusion is set BEFORE decoding, never by prediction.
                values[:4] = False
            result[name] = values
    return result


def source_navigation_inputs(source, history, representation, history_geometry="predicted_features"):
    """Shared deployed request construction; future body/height are not inputs."""
    if source.ndim != 3 or history.shape != source[:, :4].shape:
        raise ValueError("Expected a source window and four actual history frames")
    frames = source.shape[1] - 1
    history = observed_history_features(history.detach(), representation, history_geometry)
    velocity = (source[:, -4:, [0, 2]] - source[:, -5:-1, [0, 2]]) * representation.fps
    return control_inputs(history.detach(), source[:, frames - 4:frames, :5].detach(),
                          velocity.detach(), representation), frames


@torch.no_grad()
def deployed_request(inference, source, history, commit_frames=4, return_committed=False):
    """Use exact runtime Root -> Pose -> decoder inputs; source endpoint ROOT is a goal.

    The endpoint is a training navigation constraint, not hidden future body.
    Native source-root/oracle-token warmup is a separate training path.
    """
    inputs, frames = source_navigation_inputs(source, history, inference.motion_rep,
        getattr(inference, "history_geometry", "predicted_features"))
    if commit_frames not in (4, 8) or frames < commit_frames + 4:
        raise ValueError("Unsupported executed-prefix contract")
    info = {"capture_decoder_inputs": True}
    predicted, _ = inference.predict(*inputs, frames // 4,
        config={"pose_token_sampling_use_argmax": True, "final_root_pred_mode": "from_pose_module"}, info=info)
    selected = predicted[:, 4:4 + commit_frames] if return_committed else predicted[:, commit_frames:commit_frames + 4]
    return info["decoder_inputs"], selected.detach()


def refinement_loss(trainer, local_target, global_target, confidence, labels, request, history=None):
    """Reuse the verified tokenizer objective, including source-masked sole LBS."""
    full, mask, tokens, local_root, token_mask = request
    prediction = decode_tokens(trainer.pose_net, full, mask, tokens, local_root, token_mask)
    if prediction.shape[1] < local_target.shape[1]:
        raise ValueError("Deployed decoder output is shorter than the supervised window")
    # Deployment pads to max_tokens. Decode with that exact padding/mask, then
    # score only the requested window; do not change its convolution boundary.
    prediction = prediction[:, :local_target.shape[1]]
    output = {"recon_state": prediction, "l_commit": prediction.sum() * 0., "perplexity": 0.}
    batch = {"local_motions": local_target.detach(), "global_motions": global_target.detach(),
             "contact_confidence": confidence.detach(), **labels}
    if history is not None:
        batch["executed_history"] = history_in_input_frame(history)
    return trainer.loss(batch, output, None), prediction


@torch.no_grad()
def warmup_request(pose, local):
    mask = torch.zeros(local.shape[:2], dtype=torch.bool, device=local.device)
    mask[:, :4] = True
    tokens = pose.pose_net.encode_into_idx(local, fetch_overall_indices=False)
    active = torch.ones(tokens.shape[:2], dtype=torch.bool, device=local.device)
    return local * mask[..., None], mask, tokens, local, active
