"""Terrain-aware motion transformer: one next frame per call, body-part queries.

Inputs per frame (all built by the caller from already-emitted frames and the scene):

* history [B, T, 366]: native global SOMA30 frames (as P1 FrameActor), oldest first;
  history_mask [B, T] marks real frames (the last frame is always real);
* command [B, 4]: world [VX, VZ, cos(target heading), sin(target heading)], i.e. the left
  stick velocity and the right stick facing; converted here to the heading frame;
* part rays [B, 6, part_dim] from tokens.PartRayTable;
* station features [B, 8, station_dim] from tokens.StationProber, cast along the path the
  model itself proposes in ``plan``;
* optional volume tokens [B, N, volume_dim] (e.g. SharedTerrainEncoder's 24 tokens).

Two stages, so the caller can query the scene in between (as FrameActor root/pose steps):
``plan`` encodes history, command and body rays and proposes 8 path stations (0.1-0.8 s);
``act`` reads the stations' features and emits the next Root displacement, 30 local
rotations (6D, residual on the last frame, zero-initialised so an untrained model holds
the pose) and 4 foot-contact logits. The output dict is FrameActor-compatible, so
pfnn_frame_actor.advance and source_one_step_loss apply unchanged.
"""

from dataclasses import dataclass

import torch
from torch import nn

from motionbricks.motionlib.core.utils.rotations import cont6d_to_matrix, matrix_to_cont6d

from .tokens import BODY_PARTS, joint_parts

FPS = 30


@dataclass(frozen=True)
class TMTConfig:
    width: int = 128
    heads: int = 4
    ffn: int = 384
    temporal_layers: int = 3
    planner_layers: int = 2
    decoder_layers: int = 2
    history: int = 16
    stations: int = 8
    frame_dim: int = 366
    joints: int = 30
    dropout: float = 0.0


def _encoder(config, layers):
    layer = nn.TransformerEncoderLayer(config.width, config.heads, config.ffn, config.dropout,
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def _decoder(config, layers):
    layer = nn.TransformerDecoderLayer(config.width, config.heads, config.ffn, config.dropout,
                                       batch_first=True, norm_first=True)
    return nn.TransformerDecoder(layer, layers)


def _mlp(inputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, width))


def body_command(history, command):
    """World command -> heading frame [vx right, vz forward, cos dθ, sin dθ] (FrameActor convention)."""
    c, s = history[:, -1, 3], history[:, -1, 4]
    wx, wz = command[:, 0], command[:, 1]
    vx, vz = c * wx - s * wz, s * wx + c * wz
    cos_t, sin_t = command[:, 2], command[:, 3]
    return torch.stack((vx, vz, cos_t * c + sin_t * s, sin_t * c - cos_t * s), -1)


def heading_to_world(history, xz):
    """Inverse of body_command's planar rotation: heading-frame XZ [..., 2] -> world XZ."""
    c, s = history[:, -1, 3], history[:, -1, 4]
    while c.ndim < xz.ndim - 1:
        c, s = c[..., None], s[..., None]
    return torch.stack((c * xz[..., 0] + s * xz[..., 1], -s * xz[..., 0] + c * xz[..., 1]), -1)


class TerrainMotionTransformer(nn.Module):
    def __init__(self, rep, joint_names, part_dim, station_dim, *, volume_dim=None, config=None):
        super().__init__()
        config = config or TMTConfig()
        if rep.motion_rep_dim != config.frame_dim or rep.nbjoints != config.joints or rep.root_mode != "global":
            raise ValueError("TerrainMotionTransformer requires native global SOMA30/366 history")
        self.config = config
        width = config.width
        parts = joint_parts(joint_names)
        if len(parts) != config.joints or set(parts.tolist()) != set(range(len(BODY_PARTS))):
            raise ValueError("Every body part needs at least one joint")
        self.register_buffer("joint_part", torch.as_tensor(parts))
        self.register_buffer("parents", rep.skeleton.joint_parents.detach().clone().long().clamp_min(0))
        self.register_buffer("root_mask", (torch.arange(config.joints) == rep.skeleton.root_idx)[:, None, None])
        self.register_buffer("rotation_indices", torch.as_tensor(rep.indices["global_rot_data"]))
        mean, scale = torch.zeros(config.frame_dim), torch.ones(config.frame_dim)
        if rep.stats is not None and rep.stats.is_loaded():
            mean = rep.stats.mean.detach().clone().reshape(config.frame_dim)
            std = rep.stats.std.detach().clone().reshape(config.frame_dim)
            scale = std.clamp_min(1e-3) if rep.stats.legacy else (std.square() + rep.stats.eps).sqrt()
        mean[[0, 2]], scale[[0, 2]] = 0., 1.             # absolute XZ is meaningless after recentring
        self.register_buffer("history_mean", mean)
        self.register_buffer("history_scale", scale)
        self.register_buffer("path_times_s", torch.arange(1, config.stations + 1, dtype=torch.float32) * .1)

        self.frame_in = nn.Linear(config.frame_dim, width)
        self.age = nn.Parameter(torch.zeros(config.history, width))
        self.temporal = _encoder(config, config.temporal_layers)
        self.command_in = _mlp(4, width)
        self.part_in = _mlp(part_dim, width)
        self.part_embedding = nn.Parameter(torch.zeros(len(BODY_PARTS), width))
        self.station_in = _mlp(station_dim, width)
        self.station_embedding = nn.Parameter(torch.zeros(config.stations, width))
        self.volume_in = _mlp(volume_dim, width) if volume_dim else None
        self.plan_query = nn.Parameter(torch.zeros(1, width))
        self.planner = _decoder(config, config.planner_layers)
        self.path_head = nn.Linear(width, config.stations * 2)
        self.decoder = _decoder(config, config.decoder_layers)
        self.rotation_head = nn.Linear(width, config.joints * 6)
        self.root_head = nn.Linear(width, 3)
        self.contact_head = nn.Linear(2 * width, 4)
        for parameter in (self.age, self.part_embedding, self.station_embedding, self.plan_query):
            nn.init.normal_(parameter, std=0.02)
        for head in (self.rotation_head, self.root_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    # ------------------------------------------------------------------ helpers
    def normalized_history(self, history):
        recentred = history.clone()
        recentred[..., [0, 2]] = history[..., [0, 2]] - history[:, -1:, [0, 2]]
        return (recentred - self.history_mean) / self.history_scale

    def last_local_rotations(self, history):
        global_rot = cont6d_to_matrix(history[:, -1].index_select(-1, self.rotation_indices)
                                      .reshape(-1, self.config.joints, 6))
        parent_rot = global_rot.index_select(1, self.parents)
        eye = torch.eye(3, device=history.device, dtype=history.dtype)
        parent_rot = torch.where(self.root_mask, eye, parent_rot)
        return parent_rot.transpose(-1, -2) @ global_rot

    # ------------------------------------------------------------------ stages
    def plan(self, history, history_mask, command, part_rays, volume=None):
        """Encode context and propose the path; returns a dict consumed by ``act``."""
        if history.shape[1] != self.config.history:
            raise ValueError(f"History must have {self.config.history} frames (pad and mask the start)")
        frames = self.frame_in(self.normalized_history(history)) + self.age
        memory = self.temporal(frames, src_key_padding_mask=~history_mask)
        command_body = body_command(history, command)
        command_token = self.command_in(command_body)[:, None]
        parts = self.part_in(part_rays) + self.part_embedding
        context = [memory, command_token, parts]
        if self.volume_in is not None and volume is not None:
            context.append(self.volume_in(volume))
        context = torch.cat(context, 1)
        padding = torch.cat((~history_mask, history_mask.new_zeros(history.shape[0], context.shape[1] - history.shape[1])), 1)
        query = command_token + self.plan_query
        planned = self.planner(query, context, memory_key_padding_mask=padding)[:, 0]
        offsets = self.path_head(planned).reshape(-1, self.config.stations, 2) * self.path_times_s[None, :, None]
        path = history[:, -1][:, [0, 2]][:, None] + heading_to_world(history, offsets)
        return {"context": context, "padding": padding, "parts": parts, "command_token": command_token,
                "path_heading_xz": offsets, "path_world_xz": path}

    def act(self, history, plan, station_features):
        stations = self.station_in(station_features) + self.station_embedding
        context = torch.cat((plan["context"], stations), 1)
        padding = torch.cat((plan["padding"], plan["padding"].new_zeros(stations.shape[:2])), 1)
        queries = plan["parts"] + plan["command_token"]
        decoded = self.decoder(queries, context, memory_key_padding_mask=padding)   # [B, parts, W]
        per_part = self.rotation_head(decoded).reshape(-1, len(BODY_PARTS), self.config.joints, 6)
        index = self.joint_part[None, None, :, None].expand(per_part.shape[0], 1, -1, 6)
        delta = per_part.gather(1, index)[:, 0]                                     # each joint from its part
        local6d = matrix_to_cont6d(self.last_local_rotations(history)) + delta
        root_heading = self.root_head(decoded[:, 0]) / FPS                          # [dx right, dy, dz forward]
        planar = heading_to_world(history, root_heading[:, [0, 2]])
        root_delta = torch.stack((planar[:, 0], root_heading[:, 1], planar[:, 1]), -1)
        contact_logits = self.contact_head(torch.cat((decoded[:, 4], decoded[:, 5]), -1))
        return {"root_next": history[:, -1, :3] + root_delta, "root_delta": root_delta,
                "local_rot6d": local6d, "contact_logits": contact_logits,
                "path_world_xz": plan["path_world_xz"], "summary": decoded.mean(1)}
