"""Train native motion continuity against motion data, not against zero velocity."""

import torch
import torch.nn.functional as F

from motionbricks.data.bones_control import decode_tokens
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


def continuation_losses(prediction, target, observed, fps, history=4, future=4):
    """Score the published boundary using actual context, not reconstructed context.

    Inputs are official FK poses. Position losses are in m, velocity in m/s,
    acceleration in m/s^2, and rotation increment error is dimensionless chordal
    distance. Temporal targets include the source motion's nonzero differences.
    Only the first published future frames enter temporal supervision.
    """
    if fps <= 0 or history < 2 or future < 2:
        raise ValueError("Need a physical frame rate and sufficient context/future")
    end = history + future
    p, t, h = (value["posed_joints"] for value in (prediction, target, observed))
    r, tr, hr = (value["local_joint_rots"] for value in (prediction, target, observed))
    if p.shape != t.shape or r.shape != tr.shape or p.shape[1] < end or h.shape[1] != history:
        raise ValueError("Continuation poses do not match the frame contract")
    t, h, tr, hr = (value.detach() for value in (t, h, tr, hr))
    # Preserve observed motion in the loss construction only, never alter output.
    stitched = torch.cat((h[:, -2:], p[:, history:end]), 1)
    # Source derivatives are measured on the continuous SOURCE sequence. Using
    # generated h on this side would turn drift into a fake reference impulse.
    reference = t[:, history - 2:end]
    velocity_error = (stitched.diff(dim=1) - reference.diff(dim=1))[:, 1:] * fps
    acceleration_error = (stitched.diff(n=2, dim=1) - reference.diff(n=2, dim=1)) * fps ** 2
    rotation = torch.cat((hr[:, -1:], r[:, history:end]), 1)
    reference_rot = tr[:, history - 1:end]
    delta_rot = rotation[:, :-1].transpose(-1, -2) @ rotation[:, 1:]
    reference_delta = reference_rot[:, :-1].transpose(-1, -2) @ reference_rot[:, 1:]
    return {"l_boundary_position_m": (p[:, history:end] - t[:, history:end]).norm(dim=-1).mean(),
            "l_context_position_m": (p[:, :history] - h).norm(dim=-1).mean(),
            "l_continuation_velocity_mps": velocity_error.norm(dim=-1).mean(),
            "l_continuation_acceleration_mps2": acceleration_error.norm(dim=-1).mean(),
            "l_rotation_increment_chordal": (delta_rot - reference_delta).square().mean()}


@torch.no_grad()
def context_tokens(pose, local, global_motion, predicted):
    """Oracle tokens are training targets; predicted tokens get no future body."""
    if not predicted:
        return pose.pose_net.encode_into_idx(local, fetch_overall_indices=False)
    return context_logits(pose, local, global_motion).argmax(-1)


def context_logits(pose, local, global_motion):
    frames = local.shape[1]
    count = frames // pose.backbone_net.get_num_frames_per_token()
    heads = pose.backbone_net.get_num_heads()[0]
    ids = torch.full((len(local), count, heads), pose.backbone_net.POSE_MASK_ID,
                     device=local.device, dtype=torch.long)
    values, mask, _, _ = pose._sample_the_local_pose_conditions(local, frames)
    root = extract_feature_from_motion_rep(global_motion, pose.global_motion_rep, pose.args["cond_root_feature"])
    lengths = torch.full((len(local), 1), count, device=local.device, dtype=torch.long)
    return pose.backbone_net(ids, root, values, mask, lengths, None, None)["pose_logits"]


def decode_logits_straight_through(tokenizer, logits, local, mask):
    """Exact argmax forward; a biased softmax gradient surrogate in training only.

    There are no mixed-token outputs at inference. The differentiable path uses
    the official codebooks and decoder, without changing their parameters/EMA.
    """
    codebooks = tokenizer.get_codebook().detach()
    if (logits.shape[-2:] != codebooks.shape[:2]
            or not isinstance(tokenizer.quantizer.vq.project_out, torch.nn.Identity)):
        raise ValueError("Unsupported multihead codebook contract")
    probabilities = logits.softmax(-1)
    hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
    weights = hard + (probabilities - probabilities.detach())
    latent = torch.einsum("bthk,hkd->bthd", weights, codebooks).flatten(-2).transpose(1, 2).contiguous()
    external = extract_feature_from_motion_rep(local, tokenizer.motion_rep,
                                               tokenizer.decoder_external_cond_feature_mode)
    condition = extract_feature_from_motion_rep(local * mask[..., None], tokenizer.motion_rep,
                                                tokenizer.decoder_target_cond_feature_mode)
    decoded = tokenizer.decoder(latent, external, condition, mask).transpose(1, 2)
    return tokenizer._apply_condition({"recon_state": decoded}, external)["recon_state"]


def decoder_training_loss(pose, local, global_motion, confidence, temporal):
    """Same reconstruction objective in both arms; only temporal terms differ."""
    from motionbricks.data.bones_models import contact_masked_reconstruction

    mask = torch.zeros(local.shape[:2], dtype=torch.bool, device=local.device)
    mask[:, :4] = True
    # Alternate oracle and actual predicted tokens identically in both arms.
    predicted = temporal["predicted_tokens"]
    train_pose = temporal.get("train_pose", False)
    if train_pose:
        logits = context_logits(pose, local, global_motion)
        decoded = decode_logits_straight_through(pose.pose_net, logits, local, mask)
        targets = context_tokens(pose, local, global_motion, False)
        token_loss = F.cross_entropy(logits.flatten(0, -2), targets.flatten().long())
    else:
        tokens = context_tokens(pose, local, global_motion, predicted)
        decoded = decode_tokens(pose.pose_net, local * mask[..., None], mask, tokens, local)
    pred = pose.local_motion_rep.inverse(decoded, True, return_all=False)
    with torch.no_grad():
        target = pose.local_motion_rep.inverse(local, True, return_all=False)
    observed = {key: target[key][:, :4] for key in ("posed_joints", "local_joint_rots")}
    terms = continuation_losses(pred, target, observed, pose.motion_rep.fps)
    terms["l_reconstruction"] = contact_masked_reconstruction(
        decoded, local, confidence, pose.local_motion_rep.indices["foot_contacts"])
    terms["l_fk_position_m"] = (pred["posed_joints"][:, 4:] - target["posed_joints"][:, 4:]).norm(dim=-1).mean()
    terms["l_local_rotation_matrix"] = F.smooth_l1_loss(pred["local_joint_rots"][:, 4:], target["local_joint_rots"][:, 4:])
    weights = {"l_reconstruction": 1., "l_fk_position_m": 5., "l_local_rotation_matrix": 1.,
               "l_boundary_position_m": 5., "l_context_position_m": 5.,
               "l_continuation_velocity_mps": .05 if temporal["enabled"] else 0.,
               "l_continuation_acceleration_mps2": .001 if temporal["enabled"] else 0.,
               "l_rotation_increment_chordal": 1. if temporal["enabled"] else 0.}
    loss = sum(terms[key] * coefficient for key, coefficient in weights.items())
    if train_pose:
        # A keeps the existing CE-only objective. B adds native decoded supervision.
        terms["l_token_ce"] = token_loss
        weights = {key: coefficient if temporal["enabled"] else 0. for key, coefficient in weights.items()}
        weights["l_token_ce"] = 1.
        loss = token_loss + loss if temporal["enabled"] else token_loss
    return loss, terms, weights
