"""Explicit sampling requests shared by scene-control training and inference."""

from dataclasses import dataclass
import math
import random


HORIZONS = (8, 12, 16)
TOKEN_MODES = ("all_mask", "random_mask", "self_iteration")
ITERATION_POLICY = "motionbricks.scene_control_v2.argmax_cosine4.v1"


def _probabilities(values, name):
    if len(values) != 3 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError(f"{name} requires three finite nonnegative probabilities")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
        raise ValueError(f"{name} probabilities must sum to one")


@dataclass(frozen=True)
class BatchRequest:
    horizon_tokens: int
    root_source: tuple[str, ...]
    history_source: tuple[str, ...]
    token_mode: tuple[str, ...]
    self_iteration: tuple[int, ...]
    policy_id: str
    iteration_policy: str = ITERATION_POLICY

    def __post_init__(self):
        size = len(self.root_source)
        if self.horizon_tokens not in HORIZONS or not size or not self.policy_id:
            raise ValueError("Request needs a policy, a positive batch and an 8/12/16 horizon")
        if any(len(x) != size for x in (self.history_source, self.token_mode, self.self_iteration)):
            raise ValueError("Request fields must describe the same batch")
        if set(self.root_source) - {"source", "predicted"}:
            raise ValueError("Unknown Root source")
        if set(self.history_source) - {"source", "generated"} or set(self.token_mode) - set(TOKEN_MODES):
            raise ValueError("Unknown history source or token mode")
        for mode, iteration in zip(self.token_mode, self.self_iteration):
            if iteration not in ((1, 2, 3) if mode == "self_iteration" else (0,)):
                raise ValueError("Self-iteration must be 1/2/3; other modes use zero")
        if self.iteration_policy != ITERATION_POLICY:
            raise ValueError("Unsupported iterative token sampling version")

    @property
    def batch_size(self):
        return len(self.root_source)


@dataclass(frozen=True)
class CurriculumPhase:
    start_exposure: int
    source_root_probability: float
    source_history_probability: float
    token_probabilities: tuple[float, float, float]

    def __post_init__(self):
        if type(self.start_exposure) is not int or self.start_exposure < 0:
            raise ValueError("Curriculum boundaries are nonnegative actual sample exposures")
        for p in (self.source_root_probability, self.source_history_probability):
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Root/history probabilities must be in [0,1]")
        _probabilities(self.token_probabilities, "Token mixture")


class Curriculum:
    def __init__(self, *, policy_id, phases, horizon_probabilities):
        self.policy_id = policy_id
        self.phases = tuple(phases)
        self.horizon_probabilities = tuple(horizon_probabilities)
        _probabilities(self.horizon_probabilities, "Horizon mixture")
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("An explicit observation sampling policy version is required")
        starts = [p.start_exposure for p in self.phases]
        if not starts or starts[0] != 0 or starts != sorted(set(starts)):
            raise ValueError("Curriculum starts at zero with strictly increasing exposure boundaries")
        if not all(p > 0 for p in self.horizon_probabilities):
            raise ValueError("Aligned training must cover all inference horizons: 8,12,16")
        if any(not any(p.token_probabilities[i] > 0 for p in self.phases) for i in range(3)):
            raise ValueError("The curriculum must cover all-MASK, random-mask and self-iteration inputs")
        if not any(p.source_root_probability < 1 for p in self.phases) or not any(
                p.source_root_probability > 0 for p in self.phases):
            raise ValueError("The curriculum must include source and predicted Root conditions")
        if not any(p.source_history_probability < 1 for p in self.phases) or not any(
                p.source_history_probability > 0 for p in self.phases):
            raise ValueError("The curriculum must include source history and accepted generated-history recovery targets")

    @classmethod
    def from_dict(cls, value, *, policy_id):
        return cls(policy_id=policy_id, horizon_probabilities=value["horizon_probabilities"],
                   phases=[CurriculumPhase(**p) for p in value["phases"]])

    def draw(self, batch_size, *, exposure, seed):
        if type(batch_size) is not int or batch_size < 1 or type(exposure) is not int or exposure < 0:
            raise ValueError("Positive batch size and nonnegative actual exposure required")
        phase = max((p for p in self.phases if p.start_exposure <= exposure),
                    key=lambda p: p.start_exposure)
        rng = random.Random(seed)
        horizon = rng.choices(HORIZONS, weights=self.horizon_probabilities, k=1)[0]
        roots = tuple("source" if rng.random() < phase.source_root_probability else "predicted"
                      for _ in range(batch_size))
        histories = tuple("source" if rng.random() < phase.source_history_probability else "generated"
                          for _ in range(batch_size))
        modes = tuple(rng.choices(TOKEN_MODES, weights=phase.token_probabilities, k=batch_size))
        iterations = tuple(rng.randint(1, 3) if m == "self_iteration" else 0 for m in modes)
        return BatchRequest(horizon, roots, histories, modes, iterations, self.policy_id)


def remask_argmax(logits, *, next_iteration, mask_id):
    """Input to pass 2/3/4 of the versioned, deterministic four-pass sampler."""
    import torch

    if next_iteration not in (1, 2, 3) or logits.ndim != 4 or mask_id != logits.shape[-1]:
        raise ValueError("Expected per-head logits and next iteration 1/2/3")
    tokens = logits.argmax(-1)
    confidence = logits.softmax(-1).gather(-1, tokens[..., None]).squeeze(-1)
    count = max(1, int(tokens[0].numel() * math.cos(next_iteration * math.pi / 8)))
    order = confidence.flatten(1).argsort(dim=1, stable=True)
    masked = torch.zeros_like(tokens.flatten(1), dtype=torch.bool)
    masked.scatter_(1, order[:, :count], True)
    return tokens.masked_fill(masked.reshape_as(tokens), mask_id)
