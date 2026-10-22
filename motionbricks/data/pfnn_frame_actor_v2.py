"""Bounded A/B/C ablation actor: GRU or causal transformer, mean or spatial geometry.

Arm A reproduces the P1 actor (GRU over a flattened motion window, mean-pooled
spatial tokens). Arm B keeps the GRU and replaces the mean with cross-attention
to the encoder's 24 spatial tokens. Arm C replaces the temporal backbone with a
causal transformer over per-frame motion tokens. Frames, heads, losses and
exposure budget stay matched so architecture is the only variable.

The three arms share head shapes and the public root_step/pose_step contract of
motionbricks.data.pfnn_frame_actor, so advance() and source_one_step_loss() are
reused unchanged.
"""

import torch
from torch import nn

from motionbricks.motionlib.core.utils.rotations import cont6d_to_matrix, matrix_to_cont6d
from .unified_terrain import SharedTerrainEncoder
from .pfnn_frame_actor import FPS


STATE_DIM = 256
TOKEN_DIM = 128
BACKBONES = ("gru", "transformer")
GEOMETRIES = ("mean", "attention")


class SpatialTerrainAttention(nn.Module):
    """Residual cross-attention from one body query to the 24 spatial tokens.

    Token availability is the encoder's own mask. A row with no available token
    cannot be softmaxed, so one dummy key is un-masked and the residual for that
    sample is explicitly cleared instead of silently attending to zeros. The
    output projection starts at zero, so enabling attention never perturbs a
    warm-started actor before training.
    """

    def __init__(self, query_dim, token_dim=TOKEN_DIM, heads=4):
        super().__init__()
        self.query = nn.Linear(query_dim, token_dim)
        self.attention = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.output = nn.Linear(token_dim, token_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value, tokens, valid):
        if tokens.shape[-1] != self.attention.embed_dim:
            raise ValueError("Spatial token width differs from the declared token width")
        any_valid = valid.any(dim=1)
        safe_valid = valid.clone()
        safe_valid[:, 0] = safe_valid[:, 0] | ~any_valid
        safe_tokens = tokens.clone()
        safe_tokens[:, 0] = torch.where(any_valid[:, None], safe_tokens[:, 0],
                                        torch.zeros_like(safe_tokens[:, 0]))
        query = self.query(value).unsqueeze(1)
        update, _ = self.attention(query, safe_tokens, safe_tokens,
                                   key_padding_mask=~safe_valid, need_weights=False)
        update = update.squeeze(1) * any_valid[:, None].to(update.dtype)
        # Arm B is arm A plus this residual: the zero-initialised projection makes the
        # two arms identical at initialisation, so any difference is spatial attention.
        mean = tokens.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        return mean + self.output(update)


class MotionWindowEncoder(nn.Module):
    """One embedding per executed frame; the causal backbone consumes the sequence."""

    def __init__(self, width, hidden=96):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(366, hidden), nn.GELU())
        self.out = nn.Sequential(nn.Linear(hidden, width), nn.GELU())

    def forward(self, window):
        return self.out(self.point(window))


class CausalTemporalBackbone(nn.Module):
    """Causal self-attention over the motion window; only the last frame is read."""

    def __init__(self, width, layers=3, heads=4, feedforward=384):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, feedforward, batch_first=True,
                                       norm_first=True, dropout=0.0, activation="gelu")
            for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.width = width

    def forward(self, tokens):
        length = tokens.shape[1]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                     device=tokens.device), diagonal=1)
        for block in self.blocks:
            tokens = block(tokens, src_mask=mask, is_causal=True)
        return self.norm(tokens)[:, -1]


class FrameActorV2(nn.Module):
    """Ablation actor with an explicit backbone and geometry-consumption choice."""

    def __init__(self, rep, *, backbone="gru", geometry="mean", frames=16,
                 width=128, layers=3, heads=4, feedforward=384,
                 command_frame="world"):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError("backbone must be one of " + ", ".join(BACKBONES))
        if geometry not in GEOMETRIES:
            raise ValueError("geometry must be one of " + ", ".join(GEOMETRIES))
        if command_frame not in ("world", "body"):
            raise ValueError("command_frame must be world or body")
        if (rep.motion_rep_dim != 366 or rep.nbjoints != 30
                or rep.root_mode != "global" or rep.removed_heading):
            raise ValueError("FrameActorV2 requires native global SOMA30/366")
        if not 1 <= frames <= 32:
            raise ValueError("frames must lie in 1..32 for the frozen cache window")
        self.backbone = backbone
        self.geometry = geometry
        self.frames = frames
        self.width = width
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
        mean[[0, 2]], scale[[0, 2]] = 0., 1.
        self.register_buffer("history_mean", mean)
        self.register_buffer("history_scale", scale)
        self.register_buffer("planar_mask", torch.tensor([1., 0., 1.]))
        self.register_buffer("path_times_s", torch.arange(1, 9, dtype=torch.float32) * .1)

        self.terrain_encoder = SharedTerrainEncoder(grid_size=8, width=TOKEN_DIM)
        if geometry == "attention":
            self.coarse_attention = SpatialTerrainAttention(STATE_DIM, TOKEN_DIM, heads)
            self.fine_attention = SpatialTerrainAttention(STATE_DIM, TOKEN_DIM, heads)

        if backbone == "gru":
            self.history_encoder = nn.Sequential(nn.Linear(frames * 366, STATE_DIM), nn.GELU())
            self.gru = nn.GRUCell(STATE_DIM + 4 + TOKEN_DIM, STATE_DIM)
            self.project = None
        else:
            self.motion_encoder = MotionWindowEncoder(width)
            self.temporal = CausalTemporalBackbone(width, layers, heads, feedforward)
            self.project = nn.Sequential(nn.Linear(width, STATE_DIM), nn.GELU())
            self.fuse = nn.Sequential(nn.Linear(STATE_DIM + 4 + TOKEN_DIM, STATE_DIM), nn.GELU())

        self.root_head = nn.Linear(STATE_DIM, 3)
        self.path_head = nn.Linear(STATE_DIM, 16)
        self.pose = nn.Sequential(nn.Linear(STATE_DIM + 4 + TOKEN_DIM + 3, 256), nn.GELU())
        self.rotation_head = nn.Linear(256, 30 * 6)
        self.contact_head = nn.Linear(256, 4)
        nn.init.zeros_(self.rotation_head.weight)
        nn.init.zeros_(self.rotation_head.bias)

    def initial_state(self, batch_size, *, device=None, dtype=None):
        """GRU arms carry a hidden state; the transformer arm is stateless."""
        if self.backbone == "transformer":
            return None
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

    def _geometry(self, query, tokens, valid, stage):
        if self.geometry == "mean":
            return tokens.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        attention = self.coarse_attention if stage == "coarse" else self.fine_attention
        return attention(query, tokens, valid)

    def _last_local(self, history):
        global_rot = cont6d_to_matrix(history[:, -1].index_select(-1, self.rotation_indices)
                                     .reshape(-1, 30, 6))
        parent_rot = global_rot.index_select(1, self.parents)
        parent_rot = torch.where(self.root_mask, torch.eye(3, device=history.device,
                                                          dtype=history.dtype), parent_rot)
        return parent_rot.transpose(-1, -2) @ global_rot

    def root_step(self, history, command, coarse_observation, state):
        if self.backbone == "gru" and state is None:
            raise ValueError("GRU arms require an explicit recurrent state")
        if history.shape[1] != self.frames:
            raise ValueError(f"Expected {self.frames} history frames, received {history.shape[1]}")
        tokens, valid = self.terrain_encoder(coarse_observation)
        window = self._history_input(history)
        if self.backbone == "gru":
            encoded = self.history_encoder(window.flatten(1))
        else:
            encoded = self.project(self.temporal(self.motion_encoder(window)))
        geometry = self._geometry(encoded, tokens, valid, "coarse")
        value = torch.cat((encoded, self._command(history, command), geometry), -1)
        if self.backbone == "gru":
            state = self.gru(value, state)
            embedding = state
        else:
            embedding = self.fuse(value)
        delta = self.root_head(embedding) / FPS
        path_relative = self.path_head(embedding).reshape(-1, 8, 2) * self.path_times_s[None, :, None]
        return dict(state=state, root_delta=delta,
                    path_world_xz=history[:, -1, [0, 2]][:, None] + path_relative,
                    embedding=embedding)

    def pose_step(self, history, command, fine_observation, root_result):
        tokens, valid = self.terrain_encoder(fine_observation)
        geometry = self._geometry(root_result["embedding"], tokens, valid, "fine")
        value = torch.cat((root_result["embedding"], self._command(history, command),
                           geometry, root_result["root_delta"]), -1)
        embedding = self.pose(value)
        local6d = matrix_to_cont6d(self._last_local(history))
        local6d = local6d + self.rotation_head(embedding).reshape(-1, 30, 6)
        return dict(root_next=history[:, -1, :3] + root_result["root_delta"],
                    root_delta=root_result["root_delta"], local_rot6d=local6d,
                    contact_logits=self.contact_head(embedding), state=root_result["state"])
