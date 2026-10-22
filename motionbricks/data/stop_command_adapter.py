"""Learned STOP-only residual in command embedding space, never motion space."""

import torch
from torch import nn


STOP_ADAPTER_VERSION = 'zero-speed-command-embedding-residual-v1'


class StopCommandAdapter(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        width = base[-1].out_features
        self.stop = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.stop[-1].weight)
        nn.init.zeros_(self.stop[-1].bias)
        self.stop.to(base[-1].weight)

    def forward(self, controls):
        if controls.shape[-1] != 7:
            raise ValueError('Expected the original seven command-embedding features')
        value = self.base(controls)
        stopped = (controls[..., 2:4].abs().amax(-1) <= 1e-6).unsqueeze(-1)
        return value + self.stop(controls)*stopped
