"""BONES-SEED motion paired with fitted scenes, for the terrain-aware frame controller.

One shard per clip holds the unchanged source motion as native world SOMA30 frames
[F, 366] (as pfnn_frame_actor.advance builds them), the accepted scene variants packed
for skeleton_ray_sampler / scene_sdf, contact channels measured against the fitted
scene, and source-derived joystick commands (pfnn_full_training_data.input_intent_labels).

The only change of coordinates is a constant vertical translation of motion and scene
together, so that the fitted floor is y = 0; the motion itself is not edited.
Observations (body rays, path-station rays) are not cached: they are computed online on
the GPU with the same code as at runtime, from whatever history the trainer feeds.
"""

import hashlib
import re

import numpy as np

from motionbricks.controller.scene_sdf import box_sdf_numpy
from motionbricks.data.skeleton_ray_sampler import pack_boxes

FPS = 30
SPLIT_SALT = "bones-env-v1"
SPLIT_FRACTIONS = (("test", 0.08), ("validation", 0.04))      # by actor; the rest is train
BOX_CAPACITY = 64


# ---------------------------------------------------------------- identity and split
def actor_of(filename):
    match = re.search(r"__(A\d+)$", filename)
    if not match:
        raise ValueError(f"No actor suffix in {filename!r}")
    return match.group(1)


def assign_split(actor, salt=SPLIT_SALT):
    """Deterministic actor-level split: no actor appears in two splits."""
    value = int(hashlib.sha256(f"{salt}:{actor}".encode()).hexdigest()[:8], 16) / 2 ** 32
    edge = 0.0
    for name, fraction in SPLIT_FRACTIONS:
        edge += fraction
        if value < edge:
            return name
    return "train"


# ---------------------------------------------------------------- scenes
def translate_boxes(boxes, dy):
    moved = []
    for box in boxes:
        box = dict(box)
        center = list(box["center_xyz_m"])
        center[1] += dy
        box["center_xyz_m"] = center
        for key in ("top_m", "bottom_m"):
            if key in box:
                box[key] = box[key] + dy
        moved.append(box)
    return moved


def pack_variants(variants, dy=0.0, capacity=BOX_CAPACITY):
    """List of box lists -> centers [V,K,3], half [V,K,3], rotation [V,K,3,3], mask [V,K]."""
    packed = [pack_boxes(translate_boxes(boxes, dy), capacity) for boxes in variants]
    return tuple(np.stack([p[k] for p in packed]) for k in range(4))


def scene_variants(scene, include_geometry_only=False):
    """(original variant index, boxes) of the variants usable for training."""
    key = "geometry_accepted" if include_geometry_only else "accepted"
    return [(v["index"], v["boxes"]) for v in scene["variants"] if v.get(key, v["accepted"])]


# ---------------------------------------------------------------- contacts
def contact_channels(sole_left, sole_right, foot_points, boxes, fps=FPS, height_tol=0.03, speed_tol=0.2):
    """[F, 4] contacts (LeftFoot, LeftToe, RightFoot, RightToe) against the fitted scene.

    Each side's sole vertices are split per frame into a heel and a toe half by which of
    the foot / toe joints is horizontally nearer. A channel is in contact when its lowest
    vertex is within height_tol of a solid surface and its vertices move horizontally
    slower than speed_tol on average. foot_points is [F, 4, 3] in the same order as the channels.
    """
    packed = pack_boxes(boxes, max(1, len(boxes)))
    frames = len(foot_points)
    out = np.zeros((frames, 4), dtype=np.float32)
    for soles, heel_i, toe_i in ((np.asarray(sole_left), 0, 1), (np.asarray(sole_right), 2, 3)):
        sdf = box_sdf_numpy(soles.reshape(-1, 3), packed).reshape(soles.shape[:2])
        d_heel = np.linalg.norm(soles[..., [0, 2]] - foot_points[:, None, heel_i, [0, 2]], axis=-1)
        d_toe = np.linalg.norm(soles[..., [0, 2]] - foot_points[:, None, toe_i, [0, 2]], axis=-1)
        toe_half = d_toe < d_heel
        # Per-vertex horizontal speed (the sole vertex set is fixed, so vertices are tracked).
        vertex_speed = np.linalg.norm(np.diff(soles[..., [0, 2]], axis=0), axis=-1) * fps
        vertex_speed = np.concatenate([vertex_speed[:1], vertex_speed])
        for channel, half in ((heel_i, ~toe_half), (toe_i, toe_half)):
            count = half.sum(axis=1)
            lowest = np.where(half, sdf, np.inf).min(axis=1)
            speed = (vertex_speed * half).sum(axis=1) / np.maximum(count, 1)
            out[:, channel] = (count > 0) & (lowest <= height_tol) & (speed <= speed_tol)
    return out


# ---------------------------------------------------------------- native frames (torch / Kimodo)
def soma30_motion(rep, rotations77, root):
    """SOMA77 local rotations [F,77,3,3] + root [F,3] -> SOMA30 local rotations and posed joints."""
    import torch
    from kimodo.exports.motion_io import complete_motion_dict
    rotations30 = rep.skeleton.from_SOMASkeleton77(torch.as_tensor(np.asarray(rotations77, dtype=np.float32)))
    motion = complete_motion_dict(rotations30, torch.as_tensor(np.asarray(root, dtype=np.float32)), rep.skeleton, FPS)
    return motion["local_rot_mats"], motion["posed_joints"]


def world_features(rep, local_rotations30, root, contacts):
    """Native world [F, 366] frames, built exactly as pfnn_frame_actor.advance builds them."""
    import torch
    rotations = torch.as_tensor(local_rotations30)[None].float()
    roots = torch.as_tensor(np.asarray(root, dtype=np.float32))[None]
    foot = torch.as_tensor(np.asarray(contacts, dtype=np.float32))[None]
    planar_origin = roots[:, :1] * roots.new_tensor([1., 0., 1.])
    canonical, origin = rep(dict(local_joint_rots=rotations, translation=roots - planar_origin, foot_contacts=foot),
                            to_normalize=False, return_init_heading_info=True,
                            lengths=torch.full((1,), roots.shape[1], dtype=torch.long))
    quat = origin["init_heading_quat"].reshape(1, 4)
    yaw = 2 * torch.atan2(quat[:, 2], quat[:, 0])
    world = rep.change_first_heading(canonical, yaw, False, False)
    offset = torch.cat((planar_origin, world.new_zeros(1, 1, world.shape[-1] - 3)), dim=-1)
    return (world + offset)[0]


def roundtrip_error(rep, features, posed_joints):
    """Max distance between FK of the native frames and the reference posed joints."""
    import torch
    decoded = rep.inverse(features[None], False, return_all=False, joint_positions_from="global_rot_data")
    return float((decoded["posed_joints"][0] - torch.as_tensor(posed_joints)).norm(dim=-1).max())


# ---------------------------------------------------------------- training windows
def window_starts(frames, command_valid, history=16, horizon=16, stride=4):
    """Window starts s: history frames [s, s+history) and targets [s+history, s+history+horizon).

    The command for the step that emits frame t+1 is the command at frame t, so commands
    at frames s+history-1 .. s+history+horizon-2 must all be valid.
    """
    command_valid = np.asarray(command_valid, dtype=bool)
    starts = []
    for s in range(0, frames - history - horizon + 1, stride):
        if command_valid[s + history - 1:s + history + horizon - 1].all():
            starts.append(s)
    return np.asarray(starts, dtype=np.int64)


class BonesEnvWindows:
    """Torch-style dataset of (history, targets, commands, contacts, one scene variant) windows.

    rows: manifest rows (dicts with "shard", "frames", "split", "variants"); root: dataset
    directory. The variant of each item is fixed by its index, so an epoch is reproducible.
    """

    def __init__(self, root, rows, split, *, history=16, horizon=16, stride=4, cache_clips=64):
        from collections import OrderedDict
        from pathlib import Path
        self.root, self.history, self.horizon = Path(root), history, horizon
        self.rows = [r for r in rows if r["split"] == split]
        self.items = []
        for clip, row in enumerate(self.rows):
            valid = np.load(self.root / row["shard"])["command_valid"]
            for start in window_starts(row["frames"], valid, history, horizon, stride):
                self.items.append((clip, int(start)))
        self._cache, self._capacity = OrderedDict(), cache_clips

    def __len__(self):
        return len(self.items)

    def _shard(self, clip):
        if clip in self._cache:
            self._cache.move_to_end(clip)
            return self._cache[clip]
        with np.load(self.root / self.rows[clip]["shard"]) as data:
            shard = {k: data[k] for k in data.files}
        self._cache[clip] = shard
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return shard

    def __getitem__(self, index):
        import torch
        clip, start = self.items[index]
        shard = self._shard(clip)
        variant = int(hashlib.sha256(f"{index}".encode()).hexdigest()[:8], 16) % len(shard["box_mask"])
        h, t = self.history, self.horizon
        frames = shard["features"][start:start + h + t]
        commands = shard["commands"][start + h - 1:start + h + t - 1]
        return {
            "history": torch.from_numpy(frames[:h]), "history_mask": torch.ones(h, dtype=torch.bool),
            "targets": torch.from_numpy(frames[h:]), "commands": torch.from_numpy(commands),
            "contacts": torch.from_numpy(shard["contacts"][start + h:start + h + t]),
            "box_centers": torch.from_numpy(shard["box_centers"][variant]),
            "box_half": torch.from_numpy(shard["box_half"][variant]),
            "box_rotation": torch.from_numpy(shard["box_rotation"][variant]),
            "box_mask": torch.from_numpy(shard["box_mask"][variant]),
            "clip": clip, "start": start, "variant": variant,
        }


def boxes_of(batch):
    """Collated batch -> per-sample padded box tuple for cast_rays / box_sdf."""
    return batch["box_centers"], batch["box_half"], batch["box_rotation"], batch["box_mask"]
