"""Body-attached ray observations: clearance around a simplified skeleton.

Each node of the controller's own skeleton (or the midpoint of two nodes) casts a
few rays in the character's heading frame (up, down, forward, back, left, right,
forward-up) and reports where it first meets a solid box, the surface normal there,
and whether the node itself is inside a solid. Boxes use the scene-manifest layout
(center, half extents, rotation_local_to_world with local axes as columns).

Performance contract, so the call can run every frame next to the controller:

* every shape is fixed by the layout and by the padded box capacity K;
* no host synchronisation (no .item(), .all() checks, Python branches on data);
* ray/box intersection is analytic (slab test), [batch, rays, K] work in one pass.

Inputs are therefore not validated in the hot path; ``check_inputs`` exists for
tests and offline cache building. ``cast_rays_numpy`` is the float64 reference.
These are observations, not the 2 cm full-skin geometry gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Heading-frame directions: x = right, y = up, z = forward.
DIRECTIONS = {
    "up": (0.0, 1.0, 0.0), "down": (0.0, -1.0, 0.0),
    "forward": (0.0, 0.0, 1.0), "back": (0.0, 0.0, -1.0),
    "left": (-1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0),
    "forward_up": (0.0, 0.7071067811865476, 0.7071067811865476),
    "forward_down": (0.0, -0.7071067811865476, 0.7071067811865476),
}


@dataclass(frozen=True)
class RayNode:
    """A skeleton node (one joint name) or the midpoint of two, with its rays."""
    joints: tuple
    rays: tuple                      # ((direction name, max length m), ...)

    @property
    def name(self):
        return "+".join(self.joints)


def _side(name, left, right):
    return left if name.startswith("Left") else right


def default_layout():
    """About 50 rays on the 30-joint controller skeleton (names as in joint_sdf_observation)."""
    nodes = [
        RayNode(("Head",), (("up", 1.0), ("forward_up", 1.0), ("forward", .6), ("back", .6),
                            ("left", .6), ("right", .6))),
        RayNode(("Chest",), (("up", 1.0), ("forward", .6), ("back", .6), ("left", .6), ("right", .6))),
        RayNode(("Hips",), (("up", 1.2), ("down", 1.2), ("forward", .6), ("back", .6),
                            ("left", .6), ("right", .6))),
    ]
    for side in ("Left", "Right"):
        out = _side(side, "left", "right")
        nodes += [
            RayNode((f"{side}Hand",), (("down", .6), ("up", .6), ("forward", .5), (out, .5))),
            RayNode((f"{side}ForeArm",), (("down", .6), (out, .5))),
            RayNode((f"{side}Shin",), (("down", .6), ("forward", .5))),
            RayNode((f"{side}Foot",), (("down", .5), ("forward_down", .5))),
            RayNode((f"{side}ToeBase",), (("down", .5), ("forward", .5))),
            RayNode(("Hips", f"{side}Shin"), (("down", .6), ("up", .6))),          # thigh
            RayNode((f"{side}Shin", f"{side}Foot"), (("down", .6),)),               # lower leg
            RayNode(("Chest", f"{side}ForeArm"), (("down", .6),)),                  # upper arm
        ]
    return tuple(nodes)


class RayLayout:
    """Fixed index tables for a node layout on a given joint-name order."""

    feature_per_ray = 6               # distance/max, normal (heading frame) xyz, hit, node inside

    def __init__(self, joint_names, nodes=None):
        nodes = default_layout() if nodes is None else tuple(nodes)
        names = list(joint_names)
        missing = sorted({j for node in nodes for j in node.joints if j not in names})
        if missing:
            raise ValueError(f"Skeleton lacks ray nodes {missing}")
        self.nodes = nodes
        first, second, direction, length, node_of_ray = [], [], [], [], []
        for index, node in enumerate(nodes):
            a = names.index(node.joints[0])
            b = names.index(node.joints[-1])
            for direction_name, max_length in node.rays:
                if direction_name not in DIRECTIONS or not max_length > 0:
                    raise ValueError(f"Bad ray {direction_name!r} {max_length!r} on {node.name}")
                first.append(a)
                second.append(b)
                direction.append(DIRECTIONS[direction_name])
                length.append(float(max_length))
                node_of_ray.append(index)
        self.first = np.asarray(first, dtype=np.int64)
        self.second = np.asarray(second, dtype=np.int64)
        self.direction = np.asarray(direction, dtype=np.float64)
        self.max_length = np.asarray(length, dtype=np.float64)
        self.node_of_ray = np.asarray(node_of_ray, dtype=np.int64)
        self.labels = [f"{nodes[n].name}:{nodes[n].rays[k][0]}"
                       for n in range(len(nodes)) for k in range(len(nodes[n].rays))]

    @property
    def rays(self):
        return len(self.max_length)

    @property
    def feature_dim(self):
        return self.rays * self.feature_per_ray


def pack_boxes(boxes, capacity):
    """Scene-manifest style box dicts -> padded arrays (centers, half, rotation, mask)."""
    if len(boxes) > capacity:
        raise ValueError(f"{len(boxes)} boxes exceed capacity {capacity}; raise it or cull first")
    centers = np.zeros((capacity, 3))
    half = np.full((capacity, 3), 1e-3)
    rotation = np.tile(np.eye(3), (capacity, 1, 1))
    mask = np.zeros(capacity, dtype=bool)
    for k, box in enumerate(boxes):
        centers[k] = box["center_xyz_m"]
        half[k] = box["half_extent_xyz_m"]
        rotation[k] = box["rotation_local_to_world"]
        mask[k] = True
    centers[~mask] = 1e6                                    # padding lies far away and is masked
    return centers, half, rotation, mask


def _heading_basis_numpy(forward_xz):
    forward_xz = np.asarray(forward_xz, dtype=np.float64)
    norm = np.linalg.norm(forward_xz, axis=-1, keepdims=True)
    f = forward_xz / np.maximum(norm, 1e-9)
    forward = np.stack([f[..., 0], np.zeros_like(f[..., 0]), f[..., 1]], -1)
    up = np.broadcast_to(np.array([0.0, 1.0, 0.0]), forward.shape)
    right = np.cross(up, forward)                           # x = right with y up, z forward
    return np.stack([right, up, forward], -1)               # columns: heading axes in world


def cast_rays_numpy(joints, forward_xz, layout, boxes):
    """float64 reference. joints [B, J, 3], forward_xz [B, 2], boxes = pack_boxes(...) (shared)."""
    centers, half, rotation, mask = boxes
    joints = np.asarray(joints, dtype=np.float64)
    basis = _heading_basis_numpy(forward_xz)                                    # [B, 3, 3]
    origin = 0.5 * (joints[:, layout.first] + joints[:, layout.second])         # [B, R, 3]
    direction = np.einsum("bij,rj->bri", basis, layout.direction)               # [B, R, 3]
    local_origin = np.einsum("brki,kij->brkj", origin[:, :, None] - centers, rotation)
    local_direction = np.einsum("bri,kij->brkj", direction, rotation)
    safe = np.where(np.abs(local_direction) < 1e-12, 1e-12, local_direction)
    t1 = (-half - local_origin) / safe
    t2 = (half - local_origin) / safe
    near_axis = np.minimum(t1, t2)
    t_near = near_axis.max(-1)
    t_far = np.maximum(t1, t2).min(-1)
    inside = (np.abs(local_origin) <= half).all(-1) & mask
    hit = (t_near <= t_far) & (t_far >= 0) & (t_near <= layout.max_length[None, :, None]) & mask
    t = np.where(hit, np.maximum(t_near, 0.0), np.inf)
    winner = t.argmin(-1)
    distance = np.take_along_axis(t, winner[..., None], -1)[..., 0]
    any_hit = np.isfinite(distance)
    axis = np.take_along_axis(near_axis.argmax(-1), winner[..., None], -1)[..., 0]
    sign = -np.sign(np.take_along_axis(local_direction, winner[..., None, None].repeat(3, -1), 2)[:, :, 0])
    local_normal = np.zeros(distance.shape + (3,))
    np.put_along_axis(local_normal, axis[..., None], np.take_along_axis(sign, axis[..., None], -1), -1)
    normal = np.einsum("brij,brj->bri", rotation[winner], local_normal)
    starts_inside = (np.take_along_axis(inside, winner[..., None], -1)[..., 0] & any_hit) | inside.any(-1)
    normal = np.where((any_hit & ~starts_inside)[..., None], normal, 0.0)
    distance = np.where(any_hit, np.where(starts_inside, 0.0, distance), layout.max_length[None])
    return {"origin": origin, "direction": direction, "distance": distance, "normal": normal,
            "hit": any_hit, "inside": starts_inside}


def features_numpy(result, forward_xz, layout):
    basis = _heading_basis_numpy(forward_xz)
    normal_heading = np.einsum("bij,bri->brj", basis, result["normal"])
    return np.concatenate([(result["distance"] / layout.max_length)[..., None], normal_heading,
                           result["hit"][..., None].astype(float), result["inside"][..., None].astype(float)],
                          -1).reshape(len(basis), -1)


class SkeletonRaySampler:
    """Torch runtime version of cast_rays_numpy plus features; fixed shapes, no sync."""

    def __init__(self, joint_names, nodes=None, *, box_capacity=64):
        self.layout = RayLayout(joint_names, nodes)
        self.box_capacity = int(box_capacity)
        self._cache = {}

    def _tables(self, like):
        import torch
        key = (like.device, like.dtype)
        if key not in self._cache:
            layout = self.layout
            self._cache[key] = (torch.as_tensor(layout.first, device=like.device),
                                torch.as_tensor(layout.second, device=like.device),
                                torch.as_tensor(layout.direction, device=like.device, dtype=like.dtype),
                                torch.as_tensor(layout.max_length, device=like.device, dtype=like.dtype))
        return self._cache[key]

    def boxes(self, boxes, device="cpu", dtype=None):
        """Pack box dicts once per scene into device tensors of the fixed capacity."""
        import torch
        dtype = dtype or torch.float32
        return tuple(torch.as_tensor(a, device=device, dtype=torch.bool if a.dtype == bool else dtype)
                     for a in pack_boxes(boxes, self.box_capacity))

    @staticmethod
    def check_inputs(joints, forward_xz, boxes):
        """Offline/test validation; never call in the per-frame path (it synchronises)."""
        import torch
        if joints.ndim != 3 or joints.shape[-1] != 3 or forward_xz.shape != (joints.shape[0], 2):
            raise ValueError("joints [B, J, 3] and forward_xz [B, 2] required")
        if not (torch.isfinite(joints).all() and torch.isfinite(forward_xz).all()):
            raise ValueError("Finite joints and heading required")
        if not (forward_xz.norm(dim=-1) > 1e-6).all() or not boxes[3].any():
            raise ValueError("Nonzero heading and at least one box required")

    def cast(self, joints, forward_xz, boxes):
        import torch
        centers, half, rotation, mask = boxes
        first, second, direction_heading, max_length = self._tables(joints)
        f = forward_xz / forward_xz.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        zero = torch.zeros_like(f[:, 0])
        forward = torch.stack((f[:, 0], zero, f[:, 1]), -1)
        up = torch.zeros_like(forward)
        up[:, 1] = 1
        right = torch.linalg.cross(up, forward)
        basis = torch.stack((right, up, forward), -1)                                    # [B, 3, 3]
        origin = 0.5 * (joints[:, first] + joints[:, second])                            # [B, R, 3]
        direction = torch.einsum("bij,rj->bri", basis, direction_heading)
        local_origin = torch.einsum("brki,kij->brkj", origin[:, :, None] - centers, rotation)
        local_direction = torch.einsum("bri,kij->brkj", direction, rotation)
        tiny = torch.finfo(joints.dtype).eps
        safe = torch.where(local_direction.abs() < tiny, torch.full_like(local_direction, tiny), local_direction)
        t1 = (-half - local_origin) / safe
        t2 = (half - local_origin) / safe
        near_axis = torch.minimum(t1, t2)
        t_near = near_axis.amax(-1)
        t_far = torch.maximum(t1, t2).amin(-1)
        inside = (local_origin.abs() <= half).all(-1) & mask
        hit = (t_near <= t_far) & (t_far >= 0) & (t_near <= max_length[None, :, None]) & mask
        t = torch.where(hit, t_near.clamp_min(0), torch.full_like(t_near, float("inf")))
        distance, winner = t.min(-1)
        any_hit = torch.isfinite(distance)
        axis = near_axis.argmax(-1).gather(-1, winner[..., None])                       # [B, R, 1]
        winner_direction = local_direction.gather(2, winner[..., None, None].expand(*winner.shape, 1, 3))[:, :, 0]
        sign = -torch.sign(winner_direction.gather(-1, axis))
        local_normal = torch.zeros_like(origin).scatter(-1, axis, sign)
        normal = torch.einsum("brij,brj->bri", rotation[winner], local_normal)
        starts_inside = (inside.gather(-1, winner[..., None])[..., 0] & any_hit) | inside.any(-1)
        normal = torch.where((any_hit & ~starts_inside)[..., None], normal, torch.zeros_like(normal))
        distance = torch.where(any_hit, torch.where(starts_inside, torch.zeros_like(distance), distance),
                               max_length[None].expand_as(distance))
        return {"origin": origin, "direction": direction, "distance": distance, "normal": normal,
                "hit": any_hit, "inside": starts_inside, "basis": basis}

    def observe(self, joints, forward_xz, boxes):
        """[B, rays * 6] features; joints come from past (already generated) frames only."""
        import torch
        result = self.cast(joints, forward_xz, boxes)
        max_length = self._tables(joints)[3]
        normal_heading = torch.einsum("bij,bri->brj", result["basis"], result["normal"])
        features = torch.cat(((result["distance"] / max_length)[..., None], normal_heading,
                              result["hit"][..., None].to(joints.dtype),
                              result["inside"][..., None].to(joints.dtype)), -1)
        result["features"] = features.flatten(1)
        return result
