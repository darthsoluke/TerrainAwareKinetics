"""Explicit per-head VQ. Tokens are always int64 [batch, length, heads]."""

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


@dataclass(frozen=True)
class QuantizerConfig:
    entries_per_head: int
    num_heads: int = 8
    code_dim: int = 256
    decay: float = 0.99
    dead_code_threshold: float = 1.0
    kmeans_init: bool = False

    def __post_init__(self):
        if self.entries_per_head not in (10, 128, 256):
            raise ValueError("entries_per_head must be explicitly 10, 128, or 256")
        if self.num_heads < 1 or self.code_dim < 1 or self.code_dim % self.num_heads:
            raise ValueError("code_dim must be divisible by positive num_heads")
        if not 0 <= self.decay < 1 or self.dead_code_threshold < 0:
            raise ValueError("Invalid EMA decay or dead-code threshold")


@dataclass
class QuantizedLatents:
    latents: Tensor
    indices: Tensor
    commitment: Tensor
    perplexity: Tensor


def tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Exact tensor content identity, independent of serialization container."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_head_indices(indices: Tensor, config: QuantizerConfig) -> None:
    if indices.ndim != 3 or indices.shape[-1] != config.num_heads:
        raise ValueError("Only explicit [B,L,H] indices are accepted; packed total indices are forbidden")
    if indices.dtype != torch.int64:
        raise TypeError("Head indices must be int64")
    if torch.any((indices < 0) | (indices >= config.entries_per_head)):
        raise ValueError("Head index outside entries_per_head; mask/padding IDs cannot be decoded")


class ExplicitHeadQuantizer(nn.Module):
    def __init__(self, config: QuantizerConfig):
        super().__init__()
        self.config = config
        self.register_buffer("vocabulary_frozen", torch.tensor(False))
        self.vq = VectorQuantize(
            dim=config.code_dim,
            codebook_dim=config.code_dim // config.num_heads,
            heads=config.num_heads,
            separate_codebook_per_head=True,
            codebook_size=config.entries_per_head,
            decay=config.decay,
            threshold_ema_dead_code=config.dead_code_threshold,
            kmeans_init=config.kmeans_init,
            rotation_trick=False,
        )

    def _quantize(self, latents: Tensor, *, freeze_codebook: bool):
        if latents.ndim != 3 or latents.shape[1] != self.config.code_dim:
            raise ValueError("Quantizer input must be [B,code_dim,L]")
        values = latents.transpose(1, 2).contiguous()
        quantized, indices, _ = self.vq(values, freeze_codebook=freeze_codebook)
        indices = indices.reshape(values.shape[0], values.shape[1], self.config.num_heads)
        return values, quantized, indices

    def forward(self, latents: Tensor) -> QuantizedLatents:
        values, quantized, indices = self._quantize(
            latents, freeze_codebook=bool(self.vocabulary_frozen)
        )
        commitment = F.mse_loss(values, quantized.detach())
        straight_through = values + (quantized - values).detach()
        return QuantizedLatents(straight_through.transpose(1, 2).contiguous(), indices,
                                commitment, self.per_head_perplexity(indices))

    @torch.no_grad()
    def encode(self, latents: Tensor) -> Tensor:
        # Label generation must never update EMA, even when the caller is training.
        if not bool(self.vq._codebook.initted):
            raise RuntimeError("Initialize the kmeans codebook in training before generating labels")
        return self._quantize(latents, freeze_codebook=True)[2]

    def dequantize(self, indices: Tensor) -> Tensor:
        validate_head_indices(indices, self.config)
        return self.vq.get_codes_from_indices(indices)

    @torch.no_grad()
    def per_head_perplexity(self, indices: Tensor) -> Tensor:
        validate_head_indices(indices, self.config)
        counts = torch.stack([
            torch.bincount(indices[..., head].reshape(-1), minlength=self.config.entries_per_head)
            for head in range(self.config.num_heads)
        ]).float()
        probabilities = counts / counts.sum(-1, keepdim=True).clamp_min(1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        return entropy.exp()

    def freeze(self) -> str:
        if not bool(self.vq._codebook.initted):
            raise RuntimeError("Cannot freeze an uninitialized kmeans codebook")
        self.vocabulary_frozen.fill_(True)
        self.eval().requires_grad_(False)
        return self.codebook_sha256()

    def codebook_sha256(self) -> str:
        state = {"codebook": self.vq.codebook}
        state.update({f"project_in.{key}": val for key, val in self.vq.project_in.state_dict().items()})
        state.update({f"project_out.{key}": val for key, val in self.vq.project_out.state_dict().items()})
        return tensor_state_sha256(state)


@dataclass(frozen=True)
class FrozenCodebookIdentity:
    quantizer_config: QuantizerConfig
    codebook_sha256: str
    encoder_sha256: str
    representation_id: str
    normalization_sha256: str
    parent_id: str | None = None
    schema: str = "motionbricks.scene_control_v2.frozen_vocabulary.v1"

    def __post_init__(self):
        if (not isinstance(self.normalization_sha256, str)
                or re.fullmatch("[0-9a-f]{64}", self.normalization_sha256) is None):
            raise ValueError("normalization_sha256 must be 64 lowercase hexadecimal characters")

    @property
    def identity_id(self) -> str:
        payload = asdict(self)
        # Content identity does not change when the same weights are re-frozen.
        payload.pop("parent_id")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def make_token_cache(indices: Tensor, identity: FrozenCodebookIdentity) -> dict:
    validate_head_indices(indices, identity.quantizer_config)
    return {"schema": "motionbricks.scene_control_v2.head_token_cache.v1",
            "vocabulary_id": identity.identity_id, "vocabulary": asdict(identity),
            "normalization_sha256": identity.normalization_sha256,
            "indices": indices.detach().cpu().clone()}


def read_token_cache(cache: Mapping, identity: FrozenCodebookIdentity) -> Tensor:
    if cache["schema"] != "motionbricks.scene_control_v2.head_token_cache.v1":
        raise ValueError("Unexpected token-cache schema")
    if cache["normalization_sha256"] != identity.normalization_sha256:
        raise ValueError("Token labels use different frozen normalization; regenerate labels")
    if cache["vocabulary_id"] != identity.identity_id or cache["vocabulary"] != asdict(identity):
        raise ValueError("Token labels belong to a different frozen vocabulary; regenerate labels")
    validate_head_indices(cache["indices"], identity.quantizer_config)
    return cache["indices"]
