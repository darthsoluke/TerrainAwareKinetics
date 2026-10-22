"""Training-only XYZ imitation on kinematically verified source sole contacts."""

import torch
from .terrain_response_loss import source_support_masks


def source_sole_position_loss(predicted_soles, source_soles, heights):
    if predicted_soles.shape != source_soles.shape or predicted_soles.ndim != 4:
        raise ValueError('Aligned[B,T,sole_vertices,XYZ] source and predictions required')
    with torch.no_grad():
        source = source_soles.detach()
        gap, known = heights.query(source)
        active, _ = source_support_masks(source, gap, known)
    error = (predicted_soles-source).norm(dim=-1)
    loss = ((error*active).flatten(1).sum(1)/active.flatten(1).sum(1).clamp_min(1)).mean()
    return loss, active
