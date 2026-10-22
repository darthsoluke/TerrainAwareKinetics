"""Reference skinned SOMA surface from the official Kimodo implementation.

kimodo.viz.soma_skin cannot be imported through its package because kimodo.viz
imports the viser UI stack, so the module is loaded by file path. The implementation
itself is the official one: it already pads SOMA30 to the 77-joint rig and applies
the inverse bind transform, so it is the reference for judging the sole proxies
instead of a reimplementation or a manually aligned skeleton.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch

MODULE_NAME = "kimodo_soma_skin_official"


def official_skin_module_path():
    """Path of the vendored official skinning module."""
    import kimodo

    return Path(kimodo.__file__).resolve().parent / "viz" / "soma_skin.py"


def load_official_skinner(skeleton):
    """Instantiate the official SOMASkin for a SOMA30 or SOMA77 skeleton."""
    path = official_skin_module_path()
    if not path.exists():
        raise FileNotFoundError(f"Official SOMA skinning module not found at {path}")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SOMASkin(skeleton)


def skinned_vertices(skinner, local_rot_mats, root_positions):
    """Linear-blend-skinned vertices [frames, vertices, xyz] for a SOMA30 motion."""
    with torch.inference_mode():
        return skinner.skin(local_rot_mats, root_positions, rot_is_global=False)


def surface_heights(vertices, surfaces):
    """Lowest skinned height per channel over that channel's own sole vertex set.

    vertices is [frames, vertices, xyz] and surfaces comes from
    bones_seed.official_sole_surfaces. Returns {channel: [frames]} in metres above
    the world origin, so the caller decides the ground level.
    """
    heights = {}
    for surface in surfaces:
        indices = torch.as_tensor(np.asarray(surface["selected_indices"]), dtype=torch.long,
                                  device=vertices.device)
        heights[surface["channel"]] = vertices.index_select(1, indices)[..., 1].amin(dim=-1)
    return heights


def move_skinner(skinner, device, dtype=None):
    """Move the official skinner's tensors, which are plain attributes rather than buffers."""
    for name in ("bind_rig_transform", "bind_rig_transform_inv", "bind_vertices"):
        value = getattr(skinner, name)
        setattr(skinner, name, value.to(device=device, dtype=dtype or value.dtype))
    for name in ("lbs_indices", "faces"):
        value = getattr(skinner, name)
        setattr(skinner, name, value.to(device=device))
    setattr(skinner, "lbs_weights",
            getattr(skinner, "lbs_weights").to(device=device, dtype=dtype or getattr(skinner, "lbs_weights").dtype))
    return skinner
