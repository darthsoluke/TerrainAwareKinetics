"""Align current-frame terrain observations with native MB sampled segments."""

from collections.abc import Mapping
import operator
from typing import Any

import torch


def gather_current_terrain_observation(
    batch: Mapping[str, torch.Tensor],
    sample_info: Mapping[str, Any],
    context_frames: int = 4,
) -> dict[str, torch.Tensor]:
    """Gather root-local terrain at the last context frame for sampled rows.

    ``batch`` contains padded per-source ``terrain_points [N,T,2,H,W,3]`` and
    ``terrain_valid [N,T,2,H,W]``. ``sample_info`` follows the MB segment sampler contract:
    ``chosen_ids [B]`` selects source rows and ``frame_indices [B,S]`` selects
    frames from each row. Only column ``context_frames - 1`` is used. Inputs are
    detached before gathering; the returned dictionary is ready for the backbone.
    """
    if not isinstance(batch, Mapping):
        raise ValueError("batch must contain terrain_points and terrain_valid")
    missing_arrays = {"terrain_points", "terrain_valid"} - batch.keys()
    if missing_arrays:
        raise ValueError(f"batch is missing required terrain arrays: {sorted(missing_arrays)}")
    if not isinstance(sample_info, Mapping):
        raise ValueError("sample_info must be a mapping containing chosen_ids and frame_indices")
    missing = {"chosen_ids", "frame_indices"} - sample_info.keys()
    if missing:
        raise ValueError(f"sample_info is missing required fields: {sorted(missing)}")
    if not isinstance(context_frames, int) or context_frames < 1:
        raise ValueError("context_frames must be a positive integer")

    points = torch.as_tensor(batch["terrain_points"]).detach()
    valid = torch.as_tensor(batch["terrain_valid"], device=points.device).detach()
    if points.ndim != 6 or points.shape[2] != 2 or points.shape[-1] != 3:
        raise ValueError("terrain_points must have shape [N,T,2,H,W,3]")
    if not points.is_floating_point() or min(points.shape[:2]) <= 0 or min(points.shape[3:5]) <= 0:
        raise ValueError("terrain_points must be nonempty floating-point data")
    if valid.dtype != torch.bool or valid.ndim != 5:
        raise ValueError("terrain_valid must be boolean with shape [N,T,2,H,W]")
    if valid.shape != points.shape[:-1]:
        raise ValueError("terrain_valid must align with terrain_points across every frame and query")

    chosen_ids = torch.as_tensor(sample_info["chosen_ids"])
    frame_indices = torch.as_tensor(sample_info["frame_indices"])
    if chosen_ids.dtype == torch.bool or chosen_ids.is_floating_point() or chosen_ids.ndim != 1:
        raise ValueError("sample_info chosen_ids must be a one-dimensional integer tensor")
    if (frame_indices.dtype == torch.bool or frame_indices.is_floating_point()
            or frame_indices.ndim != 2):
        raise ValueError("sample_info frame_indices must be a two-dimensional integer tensor")
    if chosen_ids.shape[0] == 0 or frame_indices.shape[0] != chosen_ids.shape[0]:
        raise ValueError("chosen_ids and frame_indices must have the same nonzero batch size")
    if frame_indices.shape[1] < context_frames:
        raise ValueError("frame_indices do not contain the full current context window")

    chosen_ids = chosen_ids.to(device=points.device, dtype=torch.long)
    frame_indices = frame_indices.to(device=points.device, dtype=torch.long)
    if (chosen_ids < 0).any() or (chosen_ids >= points.shape[0]).any():
        raise ValueError("sample_info chosen_ids index outside terrain source rows")
    if (frame_indices < 0).any() or (frame_indices >= points.shape[1]).any():
        raise ValueError("sample_info frame_indices index outside padded terrain frames")

    current_indices = frame_indices[:, context_frames - 1]
    return {
        "points": points[chosen_ids, current_indices],
        "valid": valid[chosen_ids, current_indices],
    }


def root_height_condition(
    global_root: torch.Tensor,
    global_rep,
    context_frames: int = 4,
    normalized: bool = True,
) -> torch.Tensor:
    """Return root Y relative to the last context frame as detached ``[B,T,1]``.

    Training may pass full-motion features for a 366-frame window; inference may
    pass five root features per frame across a trajectory of up to 64 frames. Both
    use frame ``context_frames-1`` as their height origin; no future pose is used
    to choose that origin.
    """
    if not isinstance(global_root, torch.Tensor) or global_root.ndim != 3:
        raise ValueError("global_root must have shape [B,T,D]")
    if not isinstance(context_frames, int) or context_frames < 1:
        raise ValueError("context_frames must be a positive integer")
    if global_root.shape[1] < context_frames:
        raise ValueError("global_root must contain at least context_frames samples")
    try:
        y_index = operator.index(global_rep.indices["global_root_pos"][1])
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError("global_rep must define global_root_pos XYZ indices") from exc
    if y_index < 0 or y_index >= global_root.shape[-1]:
        raise ValueError("global_root_pos Y index is outside global_root features")

    root = global_root.detach()
    if normalized:
        if not hasattr(global_rep, "unnormalize"):
            raise ValueError("normalized root features require global_rep.unnormalize")
        root = global_rep.unnormalize(root)
    y = root[..., y_index]
    relative_y = y - y[:, context_frames - 1:context_frames]
    return relative_y.unsqueeze(-1).detach()
