"""Causal 30 Hz SOMA actor; scene queries belong to the calling runner.

History is unnormalized, FK-authoritative global SOMA30 [B,4,366]. Commands
are [VX,VZ,cos(world heading),sin(world heading)]. Root deltas are world metres
per frame; the eight learned path offsets have horizons 0.1 through 0.8 s.
"""

import torch
from torch import nn
import torch.nn.functional as F

from motionbricks.motionlib.core.utils.rotations import cont6d_to_matrix, matrix_to_cont6d
from .unified_terrain import SharedTerrainEncoder


FPS = 30
STATE_DIM = 256
LOSS_WEIGHTS = dict(root_delta=1., body_fk=1., rotation=.1,
                    boundary_joint_velocity=.25, contact=.1)


class FrameActor(nn.Module):
    """One explicit recurrent update followed by predicted-path pose refinement.

    ``rep`` supplies the native layout, hierarchy and optional fixed statistics;
    it is not retained as a model dependency. State belongs to the caller. Call
    initial_state only at an explicit episode start, never on command changes.
    Both terrain observations use SharedTerrainEncoder's grid8 dictionary API.
    """

    def __init__(self, rep, *, command_frame="world"):
        super().__init__()
        if command_frame not in ("world", "body"):
            raise ValueError("command_frame must be world or body")
        if (rep.motion_rep_dim != 366 or rep.nbjoints != 30
                or rep.root_mode != "global" or rep.removed_heading):
            raise ValueError("FrameActor requires native global SOMA30/366")
        self.command_frame = command_frame
        self.register_buffer("parents", rep.skeleton.joint_parents.detach().clone().long().clamp_min(0))
        self.register_buffer("root_mask", (torch.arange(30, device=self.parents.device)
                                          == rep.skeleton.root_idx)[:, None, None])
        self.register_buffer("rotation_indices", torch.as_tensor(rep.indices["global_rot_data"],
                                                                  device=self.parents.device))
        mean, scale = torch.zeros(366), torch.ones(366)
        if rep.stats is not None and rep.stats.is_loaded():
            mean = rep.stats.mean.detach().clone().reshape(366)
            std = rep.stats.std.detach().clone().reshape(366)
            scale = std.clamp_min(1e-3) if rep.stats.legacy else (std.square() + rep.stats.eps).sqrt()
        # Absolute XZ statistics have no meaning after recentering.
        mean[[0, 2]], scale[[0, 2]] = 0., 1.
        self.register_buffer("history_mean", mean)
        self.register_buffer("history_scale", scale)
        self.register_buffer("planar_mask", torch.tensor([1., 0., 1.]))
        self.register_buffer("path_times_s", torch.arange(1, 9, dtype=torch.float32) * .1)
        self.terrain_encoder = SharedTerrainEncoder(grid_size=8)
        self.history_encoder = nn.Sequential(nn.Linear(4 * 366, 256), nn.GELU())
        self.gru = nn.GRUCell(256 + 4 + 128, STATE_DIM)
        self.root_head = nn.Linear(STATE_DIM, 3)
        self.path_head = nn.Linear(STATE_DIM, 16)
        self.pose = nn.Sequential(nn.Linear(STATE_DIM + 4 + 128 + 3, 256), nn.GELU())
        self.rotation_head = nn.Linear(256, 30 * 6)
        self.contact_head = nn.Linear(256, 4)
        nn.init.zeros_(self.rotation_head.weight)
        nn.init.zeros_(self.rotation_head.bias)

    def initial_state(self, batch_size, *, device=None, dtype=None):
        """Explicit episode-start allocation; no hidden state is stored here."""
        weight = self.gru.weight_hh
        return torch.zeros(batch_size, STATE_DIM,
                           device=weight.device if device is None else device,
                           dtype=weight.dtype if dtype is None else dtype)

    def _history_input(self, history):
        root = history[..., :3] - history[:, -1:, :3] * self.planar_mask
        canonical = torch.cat((root, history[..., 3:]), dim=-1)
        return (canonical - self.history_mean) / self.history_scale

    def _command(self, history, command):
        if self.command_frame == "world":
            return command
        c, s = history[:, -1, 3], history[:, -1, 4]
        vx, vz = command[:, 0], command[:, 1]
        world = torch.stack((c * vx + s * vz, -s * vx + c * vz), dim=-1)
        return torch.cat((world, command[:, 2:]), dim=-1)

    def _terrain(self, observation):
        tokens, valid = self.terrain_encoder(observation)
        return tokens.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)

    def _last_local(self, history):
        global_rot = cont6d_to_matrix(history[:, -1].index_select(-1, self.rotation_indices)
                                     .reshape(-1, 30, 6))
        parent_rot = global_rot.index_select(1, self.parents)
        parent_rot = torch.where(self.root_mask, torch.eye(3, device=history.device,
                                                          dtype=history.dtype), parent_rot)
        return parent_rot.transpose(-1, -2) @ global_rot

    def root_step(self, history, command, coarse_observation, state):
        if state is None:
            raise ValueError("Explicit recurrent state required; use initial_state at episode start")
        encoded = self.history_encoder(self._history_input(history).flatten(1))
        value = torch.cat((encoded, self._command(history, command), self._terrain(coarse_observation)), -1)
        state = self.gru(value, state)
        delta = self.root_head(state) / FPS
        path_relative = self.path_head(state).reshape(-1, 8, 2) * self.path_times_s[None, :, None]
        return dict(state=state, root_delta=delta,
                    path_world_xz=history[:, -1, [0, 2]][:, None] + path_relative,
                    embedding=state)

    def pose_step(self, history, command, fine_observation, root_result):
        value = torch.cat((root_result["embedding"], self._command(history, command),
                           self._terrain(fine_observation), root_result["root_delta"]), -1)
        embedding = self.pose(value)
        local6d = matrix_to_cont6d(self._last_local(history))
        local6d = local6d + self.rotation_head(embedding).reshape(-1, 30, 6)
        return dict(root_next=history[:, -1, :3] + root_result["root_delta"],
                    root_delta=root_result["root_delta"], local_rot6d=local6d,
                    contact_logits=self.contact_head(embedding), state=root_result["state"])


def load_environment_encoder(actor, parent_state, *, freeze=True):
    """Copy only original checkpoint model_state's adaptation.encoder.* tensors.

    Pass the flat model_state mapping, not the checkpoint archive. Validate the
    entire encoder before mutating it. No Root/Pose/codec tensors are consumed.
    """
    prefix = "adaptation.encoder."
    state = {key[len(prefix):]: value for key, value in parent_state.items() if key.startswith(prefix)}
    expected = actor.terrain_encoder.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("Checkpoint adaptation.encoder keys differ from SharedTerrainEncoder(grid8)")
    if any(state[key].shape != expected[key].shape for key in expected):
        raise ValueError("Checkpoint adaptation.encoder tensor shapes differ")
    actor.terrain_encoder.load_state_dict(state, strict=True)
    actor.terrain_encoder.requires_grad_(not freeze)
    return tuple(prefix + key for key in state)


def advance(rep, history, pred):
    """Append the predicted Root/local rotations and rebuild all native features.

    The last three actual poses retain their rotations and Root exactly up to
    native conversion roundoff. Redundant positions/velocities/heading come
    from their FK; no source future, correction or pose filter is involved.
    """
    actual = rep.inverse(history[:, 1:], False, return_all=False,
                         joint_positions_from="global_rot_data")
    rotations = torch.cat((actual["local_joint_rots"],
                           cont6d_to_matrix(pred["local_rot6d"])[:, None]), dim=1)
    roots = torch.cat((history[:, 1:, :3], pred["root_next"][:, None]), dim=1)
    contacts = torch.cat((history[:, 1:, rep.indices["foot_contacts"]],
                          pred["contact_logits"].sigmoid()[:, None]), dim=1)
    planar_origin = roots[:, :1] * roots.new_tensor([1., 0., 1.])
    canonical, origin = rep(dict(local_joint_rots=rotations, translation=roots - planar_origin,
                                foot_contacts=contacts),
                            to_normalize=False, return_init_heading_info=True,
                            lengths=torch.full((history.shape[0],), 4, device=history.device, dtype=torch.long))
    quat = origin["init_heading_quat"].reshape(history.shape[0], 4)
    yaw = 2 * torch.atan2(quat[:, 2], quat[:, 0])
    world = rep.change_first_heading(canonical, yaw, False, False)
    offset = torch.cat((planar_origin,
                        world.new_zeros(world.shape[0], 1, 363)), dim=-1)
    return world + offset


def source_one_step_loss(rep, pred, nexttruth, history):
    """Source-aligned one-step losses; target is unnormalized [B,366].

    Boundary velocity uses the last actual FK pose, including genuine nonzero
    motion. Targets are detached; predicted state/history retain their graphs.
    """
    emitted = advance(rep, history, pred)[:, -1:]
    decoded = rep.inverse(emitted, False, return_all=False, joint_positions_from="global_rot_data")
    with torch.no_grad():
        truth = nexttruth.detach()[:, None]
        target = rep.inverse(truth, False, return_all=False, joint_positions_from="global_rot_data")
        boundary = rep.inverse(history[:, -1:].detach(), False, return_all=False,
                               joint_positions_from="global_rot_data")["posed_joints"]
    joints, target_joints = decoded["posed_joints"], target["posed_joints"]
    losses = dict(
        root_delta=F.smooth_l1_loss(pred["root_delta"] / .05,
                                   (truth[:, 0, :3] - history[:, -1, :3].detach()) / .05),
        body_fk=F.smooth_l1_loss((joints - pred["root_next"][:, None, None]) / .05,
                                (target_joints - truth[..., :3][:, :, None]) / .05),
        rotation=F.mse_loss(cont6d_to_matrix(pred["local_rot6d"]), target["local_joint_rots"][:, 0]),
        boundary_joint_velocity=F.smooth_l1_loss((joints - boundary) * FPS / .5,
                                                 (target_joints - boundary) * FPS / .5),
        contact=F.binary_cross_entropy_with_logits(pred["contact_logits"],
                                                   truth[:, 0, rep.indices["foot_contacts"]]),
    )
    losses["total"] = sum(LOSS_WEIGHTS[key] * value for key, value in losses.items())
    return losses
