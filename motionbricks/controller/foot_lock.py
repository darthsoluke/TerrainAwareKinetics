"""Runtime foot locking: a deterministic IK post-process on the model's emitted frame.

PFNN-style kinematic controllers do not ask the network to keep a stance foot perfectly
still; they read the predicted foot contacts and pin the foot with inverse kinematics.
This module does that for TerrainMotionTransformer, between ``model.act`` and
``pfnn_frame_actor.advance``:

* per side, the lock engages when the predicted contact (max of heel and toe channels)
  rises above ``on`` while the ankle moves slower than ``max_onset_speed_mps``; the
  ankle position at that frame becomes the anchor (whatever its height, so raised
  supports keep working);
* while locked, an analytic two-bone IK (hip -> knee -> ankle) moves the ankle back to
  the anchor; the foot keeps the global orientation the model predicted, so toes and
  sole follow the foot;
* the lock releases when the contact falls below ``off`` or the predicted ankle has slid
  more than ``max_slip_m`` from the anchor, and the target blends back to the predicted
  ankle over ``release_frames`` frames.

Only the hip, knee and ankle local rotations of a locked leg change; the root, the other
joints and the contact logits are the model's. The source motion is never touched.
"""

from dataclasses import dataclass

import torch

from motionbricks.motionlib.core.utils.rotations import cont6d_to_matrix, matrix_to_cont6d
from motionbricks.motionlib.core.utils.torch_utils import batch_rigid_transform

LEGS = (("LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase"),
        ("RightLeg", "RightShin", "RightFoot", "RightToeBase"))
#: Contact channels (LeftFoot, LeftToe, RightFoot, RightToe) used for each side.
SIDE_CHANNELS = ((0, 1), (2, 3))
FPS = 30


@dataclass(frozen=True)
class FootLockConfig:
    on: float = 0.6
    off: float = 0.4
    max_onset_speed_mps: float = 0.6
    max_slip_m: float = 0.2
    release_frames: int = 4
    reach_fraction: float = 0.999


def axis_angle_matrix(axis, angle):
    """Rodrigues rotation [B, 3, 3] about unit axes [B, 3] by angles [B]."""
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(k)
    s, c = angle.sin()[..., None, None], angle.cos()[..., None, None]
    return eye + s * k + (1 - c) * (k @ k)


def _unit(v, fallback=None, eps=1e-8):
    n = v.norm(dim=-1, keepdim=True)
    if fallback is None:
        return v / n.clamp_min(eps)
    return torch.where(n > 1e-6, v / n.clamp_min(eps), fallback)


def _angle(u, v):
    """Unsigned angle between vectors; atan2 stays exact near 0 and pi (acos does not)."""
    return torch.atan2(torch.cross(u, v, dim=-1).norm(dim=-1), (u * v).sum(-1))


def two_bone_ik(a, b, c, target, global_a, global_b, bend_hint, reach_fraction=0.999):
    """Analytic two-bone IK in world space.

    a, b, c: hip, knee, ankle positions [B, 3]; global_a/b: hip and knee global rotations.
    bend_hint [B, 3] chooses the bend plane when the leg is straight (e.g. ankle -> toe).
    Returns new hip and knee global rotations and the reached ankle position. The bend is
    applied first and the swing second (verified numerically: the reverse order misses).
    """
    lab, lcb = (b - a).norm(dim=-1), (c - b).norm(dim=-1)
    lat = (target - a).norm(dim=-1).clamp((lab - lcb).abs() + 1e-4, (lab + lcb) * reach_fraction)
    ac_ab_0, ba_bc_0 = _angle(c - a, b - a), _angle(a - b, c - b)
    ac_ab_1 = torch.acos(((lcb ** 2 - lab ** 2 - lat ** 2) / (-2 * lab * lat)).clamp(-1, 1))
    ba_bc_1 = torch.acos(((lat ** 2 - lab ** 2 - lcb ** 2) / (-2 * lab * lcb)).clamp(-1, 1))
    axis0 = _unit(torch.cross(c - a, b - a, dim=-1), _unit(torch.cross(c - a, bend_hint, dim=-1)))
    w0 = axis_angle_matrix(axis0, ac_ab_1 - ac_ab_0)
    w1 = axis_angle_matrix(axis0, ba_bc_1 - ba_bc_0)
    bent_c = a + (w0 @ (b - a)[..., None])[..., 0] + (w0 @ w1 @ (c - b)[..., None])[..., 0]
    # Swing the bent chain so the ankle points at the target.
    swing_axis = _unit(torch.cross(bent_c - a, target - a, dim=-1), axis0)
    w2 = axis_angle_matrix(swing_axis, _angle(bent_c - a, target - a))
    new_a = w2 @ w0 @ global_a
    new_b = w2 @ w0 @ w1 @ global_b
    reached = a + (w2 @ w0 @ (b - a)[..., None])[..., 0] + (w2 @ w0 @ w1 @ (c - b)[..., None])[..., 0]
    return new_a, new_b, reached


class FootLock:
    """Stateful per-sample foot locking for a batch of characters (no gradients needed)."""

    def __init__(self, rep, config=None):
        skeleton = rep.skeleton
        names = list(skeleton.bone_order_names)
        self.config = config or FootLockConfig()
        self.parents = skeleton.joint_parents.detach().cpu().long()
        self.root = int(skeleton.root_idx)
        self.neutral = skeleton.neutral_joints.detach().float()
        missing = [n for leg in LEGS for n in leg if n not in names]
        if missing:
            raise ValueError(f"Skeleton lacks leg joints {missing}")
        self.legs = [[names.index(n) for n in leg] for leg in LEGS]
        for hip, knee, ankle, toe in self.legs:
            if not (self.parents[knee] == hip and self.parents[ankle] == knee and self.parents[toe] == ankle):
                raise ValueError("Leg joints are not a hip -> knee -> ankle -> toe chain in this skeleton")

    def fk(self, rotations, root):
        """World joint positions [B, J, 3] and global rotations [B, J, 3, 3] (root joint at ``root``)."""
        neutral = self.neutral.to(rotations)[None].expand(rotations.shape[0], -1, -1)
        posed, global_rot = batch_rigid_transform(rotations, neutral, self.parents, self.root)
        return posed - posed[:, self.root:self.root + 1] + root[:, None], global_rot

    def initial_state(self, batch, device, dtype=torch.float32):
        return {"locked": torch.zeros(batch, 2, dtype=torch.bool, device=device),
                "anchor": torch.zeros(batch, 2, 3, dtype=dtype, device=device),
                "release": torch.zeros(batch, 2, dtype=torch.long, device=device),
                "release_from": torch.zeros(batch, 2, 3, dtype=dtype, device=device),
                "previous_ankle": None}

    @torch.no_grad()
    def apply(self, pred, state):
        """Returns (corrected pred, new state, info) for one emitted frame."""
        config = self.config
        rotations = cont6d_to_matrix(pred["local_rot6d"])
        joints, global_rot = self.fk(rotations, pred["root_next"])
        probability = pred["contact_logits"].sigmoid()
        new_rotations = rotations.clone()
        locked_out, anchor_out, release_out, from_out = [], [], [], []
        ankles, residual, active_any = [], [], []
        for side, (hip, knee, ankle, toe) in enumerate(self.legs):
            contact = probability[:, list(SIDE_CHANNELS[side])].amax(-1)
            ankle_pos = joints[:, ankle]
            previous = ankle_pos if state["previous_ankle"] is None else state["previous_ankle"][:, side]
            speed = (ankle_pos - previous)[:, [0, 2]].norm(dim=-1) * FPS
            locked = state["locked"][:, side]
            onset = ~locked & (contact > config.on) & (speed < config.max_onset_speed_mps)
            anchor = torch.where(onset[:, None], ankle_pos, state["anchor"][:, side])
            locked = locked | onset
            slip = (ankle_pos - anchor)[:, [0, 2]].norm(dim=-1)
            release_now = locked & ((contact < config.off) | (slip > config.max_slip_m))
            still_locked = locked & ~release_now
            release = torch.where(release_now, torch.full_like(state["release"][:, side], config.release_frames),
                                  state["release"][:, side])
            release_from = torch.where(release_now[:, None], anchor, state["release_from"][:, side])
            blending = ~still_locked & (release > 0)
            weight = (release.to(ankle_pos.dtype) / (config.release_frames + 1))[:, None]
            target = torch.where(still_locked[:, None], anchor, weight * release_from + (1 - weight) * ankle_pos)
            active = still_locked | blending
            parent = int(self.parents[hip])
            new_hip, new_knee, reached = two_bone_ik(
                joints[:, hip], joints[:, knee], ankle_pos, target, global_rot[:, hip], global_rot[:, knee],
                joints[:, toe] - ankle_pos, config.reach_fraction)
            local_hip = global_rot[:, parent].transpose(-1, -2) @ new_hip
            local_knee = new_hip.transpose(-1, -2) @ new_knee
            local_ankle = new_knee.transpose(-1, -2) @ global_rot[:, ankle]       # keep the foot's orientation
            mask = active[:, None, None]
            new_rotations[:, hip] = torch.where(mask, local_hip, rotations[:, hip])
            new_rotations[:, knee] = torch.where(mask, local_knee, rotations[:, knee])
            new_rotations[:, ankle] = torch.where(mask, local_ankle, rotations[:, ankle])
            emitted_ankle = torch.where(active[:, None], reached, ankle_pos)
            locked_out.append(still_locked)
            anchor_out.append(anchor)
            release_out.append(torch.where(blending, release - 1, release))
            from_out.append(release_from)
            ankles.append(emitted_ankle)
            residual.append(torch.where(active, (reached - target).norm(dim=-1), torch.zeros_like(slip)))
            active_any.append(active)
        corrected = dict(pred)
        corrected["local_rot6d"] = matrix_to_cont6d(new_rotations)
        new_state = {"locked": torch.stack(locked_out, 1), "anchor": torch.stack(anchor_out, 1),
                     "release": torch.stack(release_out, 1), "release_from": torch.stack(from_out, 1),
                     "previous_ankle": torch.stack(ankles, 1)}
        info = {"active": torch.stack(active_any, 1), "reach_residual_m": torch.stack(residual, 1)}
        return corrected, new_state, info


@torch.no_grad()
def rollout_frames(loop, batch, horizon, lock=None, mode="raw"):
    """Closed-loop frames [B, H, 366] for evaluation, with optional foot locking.

    mode "raw": the model's frames. "display": foot locking is applied to the emitted
    frame only; the model's own (unlocked) frame enters its history, as in PFNN. "feedback":
    the locked frame enters the history, so the model sees what was shown.
    Returns (frames, info) where info holds the lock activity and IK residual per frame.
    """
    from motionbricks.data.bones_env_dataset import boxes_of
    from motionbricks.data.pfnn_frame_actor import advance
    from .closed_loop import forward_xz
    if mode not in ("raw", "display", "feedback"):
        raise ValueError(f"Unknown mode {mode!r}")
    if mode != "raw" and lock is None:
        raise ValueError("Foot locking modes need a FootLock")
    history, mask, boxes = batch["history"], batch["history_mask"], boxes_of(batch)
    state = None if lock is None else lock.initial_state(history.shape[0], history.device, history.dtype)
    frames, active, residual = [], [], []
    for t in range(horizon):
        command = batch["commands"][:, t]
        _, part_rays, _ = loop.observe(history, command, boxes)
        plan = loop.model.plan(history, mask, command, part_rays)
        stations = loop.prober(plan["path_world_xz"], history[:, -1, :3], forward_xz(history), boxes)
        pred = loop.model.act(history, plan, stations)
        raw = advance(loop.rep, history[:, -4:], pred)[:, -1:]
        if mode == "raw":
            shown = fed = raw
        else:
            corrected, state, info = lock.apply(pred, state)
            shown = advance(loop.rep, history[:, -4:], corrected)[:, -1:]
            fed = shown if mode == "feedback" else raw
            active.append(info["active"])
            residual.append(info["reach_residual_m"])
        frames.append(shown)
        history = torch.cat((history[:, 1:], fed), 1)
        mask = torch.cat((mask[:, 1:], mask.new_ones(mask.shape[0], 1)), 1)
    info = {}
    if active:
        info = {"active": torch.stack(active, 1), "reach_residual_m": torch.stack(residual, 1)}
    return torch.cat(frames, 1), info
