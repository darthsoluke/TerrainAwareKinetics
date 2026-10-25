"""Differentiable signed distance to a padded set of oriented boxes (scene manifest layout).

Used as a training cost: points of the generated body (FK joints or skin samples) are
pushed out of solids and feet are pulled onto support, with gradients through FK into
the model. Boxes use the same padded tuple as skeleton_ray_sampler.pack_boxes:
(centers [K,3], half extents [K,3], rotation_local_to_world [K,3,3], mask [K]).

Outside the solids the value is the exact distance to the nearest box. Inside, it is the
negative depth to the nearest face of the containing box (min over boxes), which is what
a penetration cost needs. No host synchronisation; fixed shapes.
"""

import numpy as np

#: Floor under the square root so the unused branch of torch.where stays finite.
_EPS = 1e-24


def box_sdf(points, boxes):
    """Signed distance [B, ...] of points [B, ..., 3]; boxes shared [K, ...] or per sample [B, K, ...]."""
    import torch
    from motionbricks.data.skeleton_ray_sampler import batched_boxes
    batch = points.shape[0]
    centers, half, rotation, mask = batched_boxes(boxes, batch)
    flat = points.reshape(batch, -1, 3)
    local = torch.einsum("bnki,bkij->bnkj", flat[:, :, None] - centers[:, None], rotation)
    q = local.abs() - half[:, None]
    outside = q.clamp_min(0)
    squared = outside.square().sum(-1)
    positive = squared > 0
    # Exact length outside; zero (with a finite, zero gradient) on and inside the box.
    length = torch.where(positive, squared.clamp_min(_EPS).sqrt(), torch.zeros_like(squared))
    component = length + q.amax(-1).clamp_max(0)
    component = torch.where(mask[:, None], component, torch.full_like(component, float("inf")))
    return component.amin(-1).reshape(points.shape[:-1])


def box_sdf_numpy(points, boxes):
    """float64 reference of box_sdf."""
    centers, half, rotation, mask = boxes
    local = np.einsum("...ki,kij->...kj", np.asarray(points, dtype=np.float64)[..., None, :] - centers, rotation)
    q = np.abs(local) - half
    component = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
    component = np.where(mask, component, np.inf)
    return component.min(-1)


def penetration_cost(points, boxes, radius=0.0, weight=None, max_depth=None):
    """Mean squared depth by which spheres (points with radius) enter any solid (depth capped at max_depth)."""
    depth = (radius - box_sdf(points, boxes)).clamp_min(0)
    if max_depth is not None:
        depth = depth.clamp_max(max_depth)
    cost = depth.square()
    if weight is not None:
        cost = cost * weight
    return cost.flatten(1).mean(-1)


def support_cost(points, boxes, contact, offset=0.0):
    """Contact-weighted squared gap between points and the nearest surface (at offset)."""
    gap = box_sdf(points, boxes) - offset
    return (contact * gap.square()).flatten(1).mean(-1)
