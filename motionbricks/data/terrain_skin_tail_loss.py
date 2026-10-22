"""Training-only exact full-skin penetration risk, with no motion correction."""

import math

import torch
from torch.nn import functional as F

from .terrain_distance_volume import swept_surface_points
from .terrain_response_loss import heightfield_gap


SKIN_TAIL_VERSION = 'exact-skin-published8-sweep4-cvar1pct-margin5mm-v1'
SUPPORT_PROXY_VERSION = 'frozen-teacher-two-stable-intervals-low-sole-v1'
PHASE_PROXY_VERSION = 'frozen-teacher-lower5cm-backward10cmps-two-intervals-v1'


def skin_tail_penetration(future_skin, factual_last_skin, scene, field, *,
                          substeps=4, tail_fraction=.01, margin_m=.005):
    if future_skin.ndim != 3 or future_skin.shape[-1] != 3 or factual_last_skin.shape != future_skin[:1].shape:
        raise ValueError('Expected future[T,V,XYZ] and one factual last skin frame')
    if not 0 < tail_fraction <= 1 or margin_m < 0 or substeps < 1:
        raise ValueError('Invalid penetration risk parameters')
    # The past point is fixed; only the new segment can receive a gradient.
    points = swept_surface_points(torch.cat((factual_last_skin.detach(), future_skin), 0), substeps)[1:]
    gap, known = heightfield_gap(points, scene)
    if not known.all():
        raise ValueError('Published skin leaves measured terrain; retain failure, never invent observations')
    inside = gap.detach() < 0
    depth = gap * 0
    if inside.any():
        distance, _, certified = field.query(points[inside])
        if not certified.all():
            raise ValueError('Nearest air boundary is uncertified; no silent geometric fallback')
        depth = depth.masked_scatter(inside, (-distance).clamp_min(0))
    penalty = (depth - margin_m).clamp_min(0).square().flatten()
    count = max(1, math.ceil(penalty.numel() * tail_fraction))
    loss = penalty.topk(count, sorted=False).values.mean()
    return loss, dict(points=penalty.numel(), tail_points=count,
        peak_m=float(depth.detach().max()), mean_m2=float(penalty.detach().mean()),
        fraction_over2cm=float((depth.detach() > .02).float().mean()))


@torch.no_grad()
def teacher_support_proxy(points, foot_sizes, *, speed_mps=.1, bottom_margin_m=.02):
    """Weak kinematic phase evidence, not measured force/contact labels.

    Two consecutive slow world-space intervals reject an instantaneous swing
    apex. Only near-bottom vertices of that foot are used, not lifted toe areas.
    The detached frozen teacher, never student predictions, chooses the mask.
    """
    if points.ndim != 4 or points.shape[-1] != 3 or points.shape[1] < 3 or sum(foot_sizes) != points.shape[2]:
        raise ValueError('Need[B,T,sole_vertices,XYZ] and the two complete foot groups')
    points = points.detach()
    stable = torch.diff(points, dim=1).norm(dim=-1) * 30 <= speed_mps
    paired = stable[:, :-1] & stable[:, 1:]
    active = torch.zeros(points.shape[:-1], device=points.device, dtype=torch.bool)
    active[:, :-2] |= paired
    active[:, 1:-1] |= paired
    active[:, 2:] |= paired
    start = 0
    for size in foot_sizes:
        heights = points[:, :, start:start + size, 1]
        active[:, :, start:start + size] &= heights <= heights.amin(-1, keepdim=True) + bottom_margin_m
        start += size
    return active


def teacher_support_gap(future_soles, support_mask, scenes):
    if future_soles.shape[:-1] != support_mask.shape or support_mask.requires_grad:
        raise ValueError('Support selection must be detached and match actual future sole vertices')
    numerator = future_soles.sum() * 0
    count = 0
    for points, mask, scene in zip(future_soles, support_mask, scenes):
        gap, known = heightfield_gap(points, scene)
        if not known.all():
            raise ValueError('Support query leaves known terrain')
        numerator = numerator + gap.abs().masked_select(mask).sum()
        count += int(mask.sum())
    return numerator / max(1, count), count


@torch.no_grad()
def teacher_phase_support_proxy(points, roots, direction, foot_sizes):
    """Conservative weak stance phase from backward, lower-foot motion.

    Moving feet need two backward intervals relative to the pelvis and a5cm
    height separation from the other foot. This is not a force/contact sensor.
    Zero-command cases retain the stricter stationary world-space proxy.
    """
    if len(foot_sizes) != 2 or sum(foot_sizes) != points.shape[2] or roots.shape != points.shape[:2] + (3,):
        raise ValueError('Need aligned roots and exactly two complete sole groups')
    points, roots, direction = points.detach(), roots.detach(), direction.detach()
    moving = direction.norm(dim=-1) > 1e-6
    unit = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    relative = points - roots[:, :, None]
    backward = (torch.diff(relative[..., [0, 2]], dim=1) * unit[:, None, None]).sum(-1) * 30 <= -.1
    paired = backward[:, :-1] & backward[:, 1:]
    active = torch.zeros(points.shape[:-1], device=points.device, dtype=torch.bool)
    active[:, :-2] |= paired
    active[:, 1:-1] |= paired
    active[:, 2:] |= paired
    left = foot_sizes[0]
    minimum = torch.stack((relative[:, :, :left, 1].amin(-1), relative[:, :, left:, 1].amin(-1)), -1)
    start = 0
    for foot, size in enumerate(foot_sizes):
        heights = relative[:, :, start:start + size, 1]
        active[:, :, start:start + size] &= heights <= minimum[:, :, foot, None] + .02
        active[:, :, start:start + size] &= (minimum[:, :, foot] <= minimum[:, :, 1 - foot] - .05)[..., None]
        start += size
    stationary = teacher_support_proxy(points, foot_sizes)
    return torch.where(moving[:, None, None], active, stationary)


def direction_without_speed_loss(velocity, requested, velocity_std):
    """Moving commands constrain direction, not pace; idle requests zero velocity.

    Mean negative progress is penalized, but positive speed can vary freely and
    zero progress during an individual preparation window is not forbidden.
    Route completion is evaluated independently over its real elapsed time.
    """
    unit = requested / requested.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    side = torch.stack((-unit[:, 1], unit[:, 0]), -1)
    std = torch.as_tensor(velocity_std, device=velocity.device, dtype=velocity.dtype).reshape(1, 2).clamp_min(1e-4)
    side_std = ((side * std).square().sum(-1)).sqrt().clamp_min(1e-4)
    forward_std = ((unit * std).square().sum(-1)).sqrt().clamp_min(1e-4)
    lateral = (velocity * side[:, None]).sum(-1) / side_std[:, None]
    forward = (velocity * unit[:, None]).sum(-1).mean(-1) / forward_std
    moving_loss = F.smooth_l1_loss(lateral, torch.zeros_like(lateral), reduction='none').mean(-1)
    moving_loss = moving_loss + F.smooth_l1_loss((-forward).clamp_min(0), torch.zeros_like(forward), reduction='none')
    idle_loss = F.smooth_l1_loss(velocity / std, torch.zeros_like(velocity), reduction='none').mean((1, 2))
    return torch.where(requested.norm(dim=-1) > 1e-6, moving_loss, idle_loss).mean()
