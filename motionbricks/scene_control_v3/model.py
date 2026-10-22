"""One shared causal forward; randomly initialized and never optimized here."""

import math

import torch
from torch import nn

from .contracts import CAPACITY_SCOPE, PROBE_STATUS, SceneActorConfig


class _SceneEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.scene_token_dim
        self.features = nn.Sequential(
            nn.Linear(config.scene_dim, width), nn.SiLU(), nn.Linear(width, width),
        )
        self.queries = nn.Parameter(torch.randn(config.scene_token_count, width) / math.sqrt(width))
        self.unknown_tokens = nn.Parameter(torch.randn(config.scene_token_count, width) / math.sqrt(width))
        self.scale = width ** -0.5

    def forward(self, scene_features, scene_valid):
        clean = torch.where(scene_valid.unsqueeze(-1), scene_features, 0.0)
        features = self.features(clean)
        known = scene_valid.any(dim=1, keepdim=True).unsqueeze(-1)
        scores = torch.matmul(self.queries, features.transpose(1, 2)) * self.scale
        scores = scores.masked_fill(~scene_valid.unsqueeze(1), -torch.inf)
        # An unknown row has its own tokens, not a fabricated zero-distance plane.
        scores = torch.where(known, scores, 0.0)
        weights = scores.softmax(dim=-1) * scene_valid.unsqueeze(1)
        tokens = torch.matmul(weights, features)
        tokens = torch.where(known, tokens, self.unknown_tokens.unsqueeze(0))
        return tokens, scene_valid.to(scene_features.dtype).mean(dim=1, keepdim=True)


class CausalSceneActor(nn.Module):
    """Tensor-only UNTRAINED capacity probe with caller-owned recurrent memory.

    Root deltas are world XYZ metres/step. Joint rotations are world-global
    native column6D. Root previews are ego XYZ displacements at steps 1..N.
    Heading delta is an auxiliary radians/step prediction.
    """

    probe_status = PROBE_STATUS
    capacity_scope = CAPACITY_SCOPE

    def __init__(self, config=SceneActorConfig()):
        super().__init__()
        self.config = config
        width = config.hidden_dim
        self.scene_encoder = _SceneEncoder(config)
        self.history_encoder = nn.Sequential(
            nn.Linear(config.history_frames * config.history_dim, width), nn.SiLU(),
            nn.Linear(width, 128), nn.SiLU(),
        )
        self.command_encoder = nn.Sequential(nn.Linear(config.command_dim, 32), nn.SiLU())
        self.contact_encoder = nn.Sequential(
            nn.Linear(config.contact_slots * (config.contact_dim + 1), 96), nn.SiLU(),
        )
        self.scene_projection = nn.Sequential(
            nn.Linear(config.scene_token_count * config.scene_token_dim + 1, 128), nn.SiLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(384, width), nn.LayerNorm(width), nn.SiLU())
        self.backbone = nn.GRUCell(width, width)
        self.root_head = nn.Linear(width, 3)
        self.heading_head = nn.Linear(width, 1)
        self.body_head = nn.Linear(width, config.joint_count * 6)
        self.contact_head = nn.Linear(width, config.contact_slots)
        self.release_head = nn.Linear(width, config.contact_slots)
        self.preview_head = nn.Linear(width, config.preview_basis_dim * 3)

        times = torch.arange(1, config.preview_frames + 1, dtype=torch.float32) / config.fps
        frequencies = torch.arange(1, config.preview_basis_dim, dtype=torch.float32) * math.pi
        basis = torch.cat((times.unsqueeze(1), torch.sin(times.unsqueeze(1) * frequencies)), dim=1)
        self.register_buffer("preview_basis", basis, persistent=False)

    def init_hidden(self, batch_size, *, device=None, dtype=None):
        return self.root_head.weight.new_zeros((batch_size, self.config.hidden_dim), device=device, dtype=dtype)

    def forward(self, history, command, contact_state, contact_valid, scene_features, scene_valid, hidden):
        scene_tokens, scene_coverage = self.scene_encoder(scene_features, scene_valid)
        clean_contacts = torch.where(contact_valid.unsqueeze(-1), contact_state, 0.0)
        contacts = torch.cat((clean_contacts.flatten(1), contact_valid.to(history.dtype)), dim=1)
        scene = torch.cat((scene_tokens.flatten(1), scene_coverage), dim=1)
        fused = self.fusion(torch.cat((
            self.history_encoder(history.flatten(1)), self.command_encoder(command),
            self.contact_encoder(contacts), self.scene_projection(scene),
        ), dim=1))
        next_hidden = self.backbone(fused, hidden)
        coefficients = self.preview_head(next_hidden).reshape(-1, self.config.preview_basis_dim, 3)
        return {
            "root_delta": self.root_head(next_hidden) / self.config.fps,
            "heading_delta": self.heading_head(next_hidden) / self.config.fps,
            "global_joint_rot6d": self.body_head(next_hidden).reshape(-1, self.config.joint_count, 6),
            "contact_logits": self.contact_head(next_hidden),
            "release_logits": self.release_head(next_hidden),
            "root_preview": torch.matmul(self.preview_basis, coefficients),
            "hidden": next_hidden,
            "input_valid": scene_valid.any(dim=1),
        }
