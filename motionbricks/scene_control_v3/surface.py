"""Fixed official skin vertices for surface-query capacity measurements."""

import torch
from torch import nn


class OfficialSurfaceSampler(nn.Module):
    """Selected official LBS vertices, not a complete contact sampling layout."""

    complete_contact_layout = False

    def __init__(self, skinner, sample_count=512, *, selected_vertex_ids=None):
        super().__init__()
        device = skinner.bind_vertices.device
        vertex_count = len(skinner.bind_vertices)
        if type(sample_count) is not int or not 0 < sample_count <= vertex_count:
            raise ValueError("sample_count must be an integer in [1, official vertex count]")
        if selected_vertex_ids is None:
            indices = torch.linspace(0, vertex_count - 1, sample_count, dtype=torch.float64).round().long()
            self.selection_kind = "uniform_vertex_index_capacity_selection"
        else:
            indices = torch.as_tensor(selected_vertex_ids).detach().cpu()
            self.selection_kind = "explicit_vertex_ids_capacity_selection"
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("selected_vertex_ids must contain integer indices")
        if indices.ndim != 1 or indices.numel() != sample_count:
            raise ValueError("selected_vertex_ids must be one-dimensional with sample_count entries")
        if (indices < 0).any() or (indices >= vertex_count).any():
            raise ValueError("selected_vertex_ids are outside the official vertex range")
        if indices.unique().numel() != sample_count:
            raise ValueError("selected_vertex_ids must be unique")
        indices = indices.to(device=device, dtype=torch.long)
        self.sample_count = sample_count
        self.register_buffer("selected_vertex_ids", indices.detach().clone())
        for name in ("bind_vertices", "lbs_indices", "lbs_weights"):
            self.register_buffer(name, getattr(skinner, name).detach().index_select(0, indices).clone())
        self.register_buffer("bind_rig_transform_inv", skinner.bind_rig_transform_inv.detach().clone())
        # The native adapter overrides the instance method at the rig boundary.
        self._official_lbs = type(skinner).lbs

    def forward(self, rig):
        """Map [B,T,77,4,4] to [B,T,sample_count,3] with every influence."""
        return self._official_lbs(self, rig)
