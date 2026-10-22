"""Meter/second codec targets gated by measured GT or accepted teacher labels."""

from dataclasses import dataclass, fields
import math
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class LossTerms:
    reconstruction: Tensor
    commitment: Tensor
    fk_m: Tensor
    contact_bce: Tensor
    contact_anchor_m: Tensor
    contact_gap_m: Tensor
    relative_slip_mps: Tensor
    penetration_m: Tensor
    boundary_mps: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def weighted_sum(self, weights: Mapping[str, float]) -> Tensor:
        terms = self.as_dict()
        if set(weights) != set(terms):
            raise ValueError("Provide an explicit weight for every named LossTerms field")
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("Loss weights must be finite and nonnegative")
        return sum(terms[name] * weights[name] for name in terms)


def masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    """Exclude invalid values before arithmetic, including unknown/NaN slots."""
    if valid.dtype != torch.bool or values.shape != valid.shape:
        raise ValueError("A boolean mask must exactly match the per-element loss")
    valid = valid.detach()
    return torch.where(valid, values, 0.0).sum() / valid.sum().clamp_min(1)


def _masked_vector(values: Tensor, valid: Tensor) -> Tensor:
    return torch.where(valid.detach()[..., None], values, 0.0)


def contact_surface_transforms(contacts: Mapping, scene: Mapping) -> tuple[Tensor, Tensor]:
    """Resolve persistent contact IDs against each sample's ordered object-ID table."""
    object_id = contacts["object_id"].detach().long()
    table_ids = scene["object_ids"].detach().to(object_id)
    transforms = scene["object_to_world"].detach()
    object_valid = scene["object_valid"].detach().bool()
    batch, time, channels = object_id.shape
    if table_ids.shape != (batch, transforms.shape[2]):
        raise ValueError("scene.object_ids must be [B,O], aligned with object_to_world columns")
    result = torch.eye(4, dtype=transforms.dtype, device=transforms.device).expand(
        batch, time, channels, 4, 4).clone()
    valid = torch.zeros_like(object_id, dtype=torch.bool)
    matches = (object_id[..., None] == table_ids[:, None, None, :]) & (object_id[..., None] >= 0)
    b, t, c, objects = matches.nonzero(as_tuple=True)
    result[b, t, c] = transforms[b, t, objects]
    valid[b, t, c] = object_valid[b, t, objects]
    return result, valid


def relative_contact_losses(
    body_samples_m: Tensor, contact_signed_distance_m: Tensor, contacts: Mapping,
    scene: Mapping, contact_sample_indices: Tensor, contact_surface_valid: Tensor,
    contact_interval_valid: Tensor, frame_valid: Tensor, *, fps: float,
) -> dict[str, Tensor]:
    """Fixed support/brace/grasp attach material points; slide/touch allow tangent motion.

    Contact kind IDs are the data contract: support=0, brace=1, grasp=2,
    touch=3, slide=4. Intended `state` and predicted contact logits are not gates.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    points = body_samples_m.index_select(2, contact_sample_indices.long())
    transform, object_valid = contact_surface_transforms(contacts, scene)
    actual_valid = contacts["actual_valid"].detach().bool() & frame_valid[..., None].detach()
    active = contacts["actual_state"].detach() > 0.5
    geometry_valid = (actual_valid & active & contacts["anchor_valid"].detach().bool()
                      & object_valid & contact_surface_valid.detach().bool())
    kind = contacts["kind"].detach()
    fixed = (kind == 0) | (kind == 1) | (kind == 2)
    # R^T (p - t) includes both object translation and rotation in relative slip.
    rotation = torch.where(geometry_valid[..., None, None], transform[..., :3, :3],
                           torch.eye(3, dtype=points.dtype, device=points.device))
    offset = _masked_vector(points, geometry_valid) - _masked_vector(transform[..., :3, 3], geometry_valid)
    local = (rotation.transpose(-1, -2) @ offset[..., None]).squeeze(-1)
    anchor = _masked_vector(contacts["anchor_local_m"].detach(), geometry_valid)
    normal = _masked_vector(contacts["normal_local"].detach(), geometry_valid)
    error = local - anchor
    anchor_error = torch.where(fixed, error.norm(dim=-1), (error * normal).sum(-1).abs())
    anchor_loss = masked_mean(anchor_error, geometry_valid)
    gap = masked_mean(torch.where(geometry_valid, contact_signed_distance_m, 0.0).abs(), geometry_valid)
    interval_valid = (geometry_valid[:, 1:] & geometry_valid[:, :-1]
                      & fixed[:, 1:] & fixed[:, :-1]
                      & contact_interval_valid.detach().bool()
                      & (contacts["object_id"][:, 1:] == contacts["object_id"][:, :-1])
                      & (kind[:, 1:] == kind[:, :-1])
                      & ((anchor[:, 1:] - anchor[:, :-1]).abs().amax(-1) <= 1e-6))
    velocity = _masked_vector(local[:, 1:] - local[:, :-1], interval_valid) * fps
    slip = masked_mean(velocity.norm(dim=-1), interval_valid)
    return {"contact_anchor_m": anchor_loss, "contact_gap_m": gap, "relative_slip_mps": slip}


def executed_boundary_loss(
    predicted_joints_m: Tensor, executed_joints_m: Tensor, executed_valid: Tensor,
    frame_valid: Tensor, *, fps: float, first_new_frame: int,
) -> Tensor:
    """Match seam velocity to actual executed velocity, without rewriting the prefix."""
    if fps <= 0 or not 0 <= first_new_frame < predicted_joints_m.shape[1]:
        raise ValueError("Positive fps and an in-range first_new_frame are required")
    if executed_joints_m.shape[1] < 2:
        raise ValueError("Boundary continuity requires at least two actually executed frames")
    valid = (executed_valid[:, -1].detach() & executed_valid[:, -2].detach()
             & frame_valid[:, first_new_frame, None].detach())
    previous = _masked_vector(executed_joints_m[:, -1].detach(), valid)
    older = _masked_vector(executed_joints_m[:, -2].detach(), valid)
    new = _masked_vector(predicted_joints_m[:, first_new_frame], valid)
    velocity_error = ((new - previous) - (previous - older)) * fps
    return masked_mean(velocity_error.norm(dim=-1), valid)


def codec_loss_terms(prediction: Mapping, target: Mapping, *, fps: float, first_new_frame: int) -> LossTerms:
    """Minimal codec objective: reconstruction/VQ, FK, contact, geometry, slip and seam.

    Nested `contacts` and `scene` are batched tensors following the paired.v1 data
    schema. actual_valid is reliable measured/accepted teacher evidence, not intent.
    SDF queries supplied in prediction must retain gradients to body_samples_m.
    """
    frames = target["frame_valid"].detach()
    recon_valid = frames[..., None].expand_as(prediction["recon_state"])
    recon = F.smooth_l1_loss(torch.where(recon_valid, prediction["recon_state"], 0.0),
                             torch.where(recon_valid, target["recon_state"].detach(), 0.0), reduction="none")
    reconstruction = masked_mean(recon, recon_valid)
    joint_valid = target["joint_valid"].detach() & frames[..., None]
    delta = _masked_vector(prediction["joint_positions_m"], joint_valid) - _masked_vector(
        target["joint_positions_m"].detach(), joint_valid)
    fk = masked_mean(delta.norm(dim=-1), joint_valid)
    contacts = target["contacts"]
    contact_valid = contacts["actual_valid"].detach().bool() & frames[..., None]
    logits = torch.where(contact_valid, prediction["contact_logits"], 0.0)
    labels = torch.where(contact_valid, contacts["actual_state"].detach(), 0.0).to(logits.dtype)
    contact = masked_mean(F.binary_cross_entropy_with_logits(logits, labels, reduction="none"), contact_valid)
    geometry = relative_contact_losses(
        prediction["body_samples_m"], prediction["contact_signed_distance_m"], contacts, target["scene"],
        target["contact_sample_indices"], target["contact_surface_valid"],
        target["contact_interval_valid"], frames, fps=fps)
    noncontact = target["body_noncontact_valid"].detach() & frames[..., None]
    signed_distance = torch.where(noncontact, prediction["body_signed_distance_m"], 0.0)
    penetration = masked_mean(F.relu(-signed_distance), noncontact)
    boundary = executed_boundary_loss(
        prediction["joint_positions_m"], target["executed_joints_m"], target["executed_valid"],
        frames, fps=fps, first_new_frame=first_new_frame)
    return LossTerms(reconstruction, prediction["l_commit"], fk, contact,
                     geometry["contact_anchor_m"], geometry["contact_gap_m"],
                     geometry["relative_slip_mps"], penetration, boundary)
