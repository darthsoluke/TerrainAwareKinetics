"""Token inputs for the terrain-aware motion transformer.

Three kinds of environment tokens, all in the character's heading frame:

* part tokens: the skeleton rays of each body part (root, torso, arms, legs), from
  skeleton_ray_sampler, gathered into a fixed [parts, rays per part] table;
* station tokens: rays cast from the predicted path stations (where the character is
  about to be): floor below, ceiling above, walls left/right and ahead;
* optional volume tokens: the 24 spatial tokens of the existing SharedTerrainEncoder.

Body-part membership is decided from joint names, so the same code works on any rig
whose names follow the SOMA convention (Left*/Right*, Hips, Spine*, Chest, Neck*, Head).
"""

import numpy as np

from motionbricks.data.skeleton_ray_sampler import DIRECTIONS, cast_rays, heading_basis, ray_features

BODY_PARTS = ("root", "torso", "left_arm", "right_arm", "left_leg", "right_leg")
_ARM = ("Clavicle", "Shoulder", "Arm", "ForeArm", "Hand", "Thumb", "Index", "Middle", "Ring", "Pinky", "Finger")
_FACE = ("Eye", "Ear", "Jaw", "Lip", "Brow", "Cheek")
_LEG = ("UpLeg", "Leg", "Thigh", "Knee", "Shin", "Foot", "Toe", "Ankle")


def part_of_joint_name(name):
    """Index into BODY_PARTS for a SOMA-style joint name."""
    for prefix, arm, leg in (("Left", 2, 4), ("Right", 3, 5)):
        if name.startswith(prefix):
            rest = name[len(prefix):]
            if any(token in rest for token in _LEG):
                return leg
            if any(token in rest for token in _ARM):
                return arm
            if any(token in rest for token in _FACE):
                return 1
    if name.startswith(("Hips", "Root", "Pelvis")):
        return 0
    if name.startswith(("Spine", "Chest", "Neck", "Head", "Jaw", "Eye")):
        return 1
    raise ValueError(f"No body part for joint {name!r}")


def joint_parts(joint_names):
    """[J] body-part index per joint."""
    return np.array([part_of_joint_name(name) for name in joint_names], dtype=np.int64)


def ray_parts(layout):
    """[R] body-part index per ray; a midpoint node belongs to its limb end, not the hips."""
    parts = []
    for node_index in layout.node_of_ray:
        candidates = [part_of_joint_name(name) for name in layout.nodes[node_index].joints]
        limbs = [p for p in candidates if p >= 2]
        parts.append(limbs[-1] if limbs else candidates[0])
    return np.asarray(parts, dtype=np.int64)


class PartRayTable:
    """Fixed gather table [parts, slots] of ray indices, padded with an empty slot."""

    def __init__(self, layout):
        parts = ray_parts(layout)
        self.slots = int(max(int((parts == p).sum()) for p in range(len(BODY_PARTS))))
        table = np.full((len(BODY_PARTS), self.slots), layout.rays, dtype=np.int64)
        mask = np.zeros((len(BODY_PARTS), self.slots), dtype=bool)
        for p in range(len(BODY_PARTS)):
            rays = np.flatnonzero(parts == p)
            table[p, :len(rays)] = rays
            mask[p, :len(rays)] = True
        self.table, self.mask = table, mask
        self.feature_dim = self.slots * 7              # 6 ray features + 1 presence flag per slot

    def gather(self, features):
        """[B, R, 6] ray features -> [B, parts, slots * 7]."""
        import torch
        padded = torch.cat((features, features.new_zeros(features.shape[0], 1, features.shape[-1])), 1)
        table = torch.as_tensor(self.table, device=features.device)
        mask = torch.as_tensor(self.mask, device=features.device, dtype=features.dtype)
        gathered = padded[:, table]                                    # [B, parts, slots, 6]
        flags = mask[None, :, :, None].expand(features.shape[0], -1, -1, 1)
        return torch.cat((gathered, flags), -1).flatten(2)


#: Rays cast from each path station: (direction in the heading frame, maximum length m).
STATION_RAYS = (("down", 2.0), ("up", 2.5), ("left", 0.8), ("right", 0.8), ("forward", 0.8))


def fourier(values, bands=4, base=1.0):
    """[..., D] -> [..., D * 2 * bands] sin/cos at frequencies base * 2^k (k < bands)."""
    import torch
    frequencies = base * (2.0 ** torch.arange(bands, device=values.device, dtype=values.dtype)) * np.pi
    angles = values[..., None] * frequencies
    return torch.cat((angles.sin(), angles.cos()), -1).flatten(-2)


class StationProber:
    """Rays from predicted path stations at the current root height, in the heading frame."""

    def __init__(self, rays=STATION_RAYS, bands=4):
        self.rays = tuple(rays)
        self.bands = int(bands)
        self.feature_dim = len(self.rays) * 6 + 3 * 2 * self.bands

    def __call__(self, stations_world_xz, root_world, forward_xz, boxes):
        """stations [B, S, 2], root [B, 3], forward [B, 2] -> features [B, S, feature_dim]."""
        import torch
        batch, stations = stations_world_xz.shape[:2]
        basis = heading_basis(forward_xz)                                           # [B, 3, 3]
        direction_heading = stations_world_xz.new_tensor([DIRECTIONS[name] for name, _ in self.rays])
        max_length = stations_world_xz.new_tensor([length for _, length in self.rays])
        origin = torch.stack((stations_world_xz[..., 0], root_world[:, None, 1].expand(batch, stations),
                              stations_world_xz[..., 1]), -1)                       # [B, S, 3]
        origins = origin[:, :, None].expand(batch, stations, len(self.rays), 3).reshape(batch, -1, 3)
        directions = torch.einsum("bij,rj->bri", basis, direction_heading)
        directions = directions[:, None].expand(batch, stations, len(self.rays), 3).reshape(batch, -1, 3)
        lengths = max_length.repeat(stations)
        result = cast_rays(origins, directions, lengths, boxes)
        features = ray_features(result, basis, lengths).reshape(batch, stations, -1)
        relative = torch.einsum("bij,bsi->bsj", basis, origin - root_world[:, None])  # heading-frame offsets
        return torch.cat((features, fourier(relative, self.bands, base=0.25)), -1)
