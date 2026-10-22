"""Small local-heightfield encoder and gated cross-attention for MotionBricks."""

import torch
from torch import nn
from torch.nn import functional as F
from motionbricks.data.native_mb_terrain_protocol import TERRAIN_PROTOCOL

TERRAIN_PATCH_COUNT, TERRAIN_PATCH_HEIGHT, TERRAIN_PATCH_WIDTH, TERRAIN_COORDINATE_DIM = \
    TERRAIN_PROTOCOL["patch_shape"]
TERRAIN_MAX_HORIZONTAL_DISTANCE_M = TERRAIN_PROTOCOL["max_horizontal_query_radius_m"]


def terrain_average_pool(value, grid):
    if not torch.are_deterministic_algorithms_enabled():
        return F.adaptive_avg_pool2d(value, (grid, grid))
    # Same adaptive bins; separate reductions avoid overlapping atomic backward.
    height, width = value.shape[-2:]
    rows = []
    for y in range(grid):
        top, bottom = y*height//grid, ((y+1)*height+grid-1)//grid
        columns = []
        for x in range(grid):
            left, right = x*width//grid, ((x+1)*width+grid-1)//grid
            columns.append(value[..., top:bottom, left:right].mean((-2,-1)))
        rows.append(torch.stack(columns,-1))
    return torch.stack(rows,-2)


class LocalTerrainEncoder(nn.Module):
    """Encode two masked XYZ heightfields into spatially pooled terrain tokens."""

    def __init__(self, embedding_dim: int, token_grid_size: int = 3):
        super().__init__()
        self.token_grid_size = int(token_grid_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, embedding_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, terrain_observation, batch_size: int, device, dtype):
        if not isinstance(terrain_observation, dict):
            raise TypeError("terrain_observation must be a dict with 'points' and 'valid'.")
        if "points" not in terrain_observation or "valid" not in terrain_observation:
            raise ValueError("terrain_observation must contain 'points' and 'valid'.")

        cache = terrain_observation.get('_inference_token_cache')
        use_cache = isinstance(cache, dict) and not self.training and not torch.is_grad_enabled()
        cache_key = (self, batch_size, str(device), dtype)
        if use_cache and cache_key in cache:
            return cache[cache_key]

        points = terrain_observation["points"]
        valid = terrain_observation["valid"]
        expected_shape = (batch_size, TERRAIN_PATCH_COUNT, TERRAIN_PATCH_HEIGHT,
                          TERRAIN_PATCH_WIDTH, TERRAIN_COORDINATE_DIM)
        if not torch.is_tensor(points) or points.ndim != 5 or tuple(points.shape) != expected_shape:
            raise ValueError(
                "terrain points must have shape "
                f"[B, {TERRAIN_PATCH_COUNT}, {TERRAIN_PATCH_HEIGHT}, {TERRAIN_PATCH_WIDTH}, "
                f"{TERRAIN_COORDINATE_DIM}]."
            )
        if not torch.is_tensor(valid) or valid.shape != points.shape[:-1] or valid.dtype != torch.bool:
            raise ValueError("terrain valid must be bool with shape [B, 2, H, W].")

        # The observation is data, not a differentiable input to the policy.
        points = points.detach().to(device=device, dtype=dtype)
        valid = valid.detach().to(device=device)
        finite = torch.isfinite(points).all(dim=-1)
        valid = valid & finite
        horizontal_distance = torch.linalg.vector_norm(points[..., (0, 2)], dim=-1)
        envelope = terrain_observation.get('query_envelope_m')
        maximum = TERRAIN_MAX_HORIZONTAL_DISTANCE_M
        if envelope is not None:
            envelope = torch.as_tensor(envelope,device=device,dtype=dtype).detach()
            if envelope.shape!=(batch_size,) or not torch.isfinite(envelope).all() or ((envelope<.5)|(envelope>26.1)).any():
                raise ValueError('Invalid declared runtime query envelope')
            maximum = envelope[:,None,None,None]+1e-5
        if (valid & (horizontal_distance > maximum)).any():
            raise ValueError(
                "valid terrain points must be within "
                f"{TERRAIN_MAX_HORIZONTAL_DISTANCE_M} m horizontally from the current root."
            )
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        points = torch.where(valid[..., None], points, torch.zeros_like(points))

        batch, planes, height, width, _ = points.shape
        channels = torch.cat([points, valid[..., None].to(dtype)], dim=-1)
        channels = channels.permute(0, 1, 4, 2, 3).reshape(batch * planes, 4, height, width)
        encoded = self.encoder(channels)
        grid = self.token_grid_size
        tokens = terrain_average_pool(encoded, grid)
        occupancy = terrain_average_pool(valid.reshape(batch * planes, 1, height, width).to(dtype), grid)
        tokens = tokens.flatten(2).transpose(1, 2).reshape(batch, planes * grid * grid, -1)
        token_valid = occupancy.flatten(2).squeeze(1).reshape(batch, planes * grid * grid) > 0
        tokens = torch.where(token_valid[..., None], tokens, torch.zeros_like(tokens))
        result = tokens, token_valid
        if use_cache:
            cache[cache_key] = result
        return result


class TerrainCrossAttention(nn.Module):
    """Residual cross-attention whose zero gate preserves the original network."""

    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, queries, terrain_tokens, terrain_valid):
        # MHA softmax is undefined for an all-masked row. Unmask a zero dummy key,
        # then explicitly clear that sample's update.
        any_valid = terrain_valid.any(dim=1)
        safe_valid = terrain_valid.clone()
        safe_valid[:, 0] = safe_valid[:, 0] | ~any_valid
        safe_tokens = terrain_tokens.clone()
        safe_tokens[:, 0] = torch.where(any_valid[:, None], safe_tokens[:, 0], 0)
        update, _ = self.attention(queries, safe_tokens, safe_tokens,
                                   key_padding_mask=~safe_valid, need_weights=False)
        update = update * any_valid[:, None, None].to(update.dtype)
        return queries + self.gate * update
