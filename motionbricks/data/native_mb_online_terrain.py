"""Causal online terrain queries from static per-record heightfield sidecars."""

from collections.abc import Mapping

import torch


_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
MAX_PACKED_TERRAIN_CELLS = 16_000_000


def _integer_tensor(value, name, ndim):
    result = torch.as_tensor(value)
    if result.dtype not in _INTEGER_DTYPES or result.ndim != ndim:
        raise ValueError(f"{name} must be an integer tensor with {ndim} dimensions")
    return result


def gather_online_terrain_observation(
    batch,
    sample_info,
    direction_local_xz,
    context_frames=4,
    *, range_policy=None, batch_queries=True,
):
    """Query static source maps at each sampled row's last context root.

    ``chosen_ids`` selects original batch rows and only
    ``frame_indices[:, context_frames - 1]`` is read. Directions are relative to
    each selected root's body yaw and are never inferred from map validity or
    motion. Repeated samples from one source share a vectorized map query.
    """
    if not isinstance(batch, Mapping):
        raise ValueError("batch must contain terrain_scene and terrain_root_world")
    if "terrain_scene" not in batch or "terrain_root_world" not in batch:
        raise ValueError("batch is missing online terrain scenes or world roots")
    if not isinstance(sample_info, Mapping):
        raise ValueError("sample_info must contain chosen_ids and frame_indices")
    missing = {"chosen_ids", "frame_indices"} - sample_info.keys()
    if missing:
        raise ValueError(f"sample_info is missing required fields: {sorted(missing)}")
    if isinstance(context_frames, bool) or not isinstance(context_frames, int) or context_frames < 1:
        raise ValueError("context_frames must be a positive integer")

    from motionbricks.data.local_terrain_observation import extract_local_terrain_observation
    from motionbricks.data.native_mb_terrain_protocol import terrain_observation_config

    roots = torch.as_tensor(batch["terrain_root_world"]).detach()
    scenes = batch["terrain_scene"]
    if (roots.ndim != 3 or roots.shape[-1] != 5 or not roots.is_floating_point()
            or min(roots.shape[:2]) <= 0):
        raise ValueError("terrain_root_world must be nonempty floating-point [N,T,5]")
    if not isinstance(scenes, (list, tuple)) or len(scenes) != roots.shape[0]:
        raise ValueError("terrain_scene must contain one scene per original source row")

    chosen_ids = _integer_tensor(sample_info["chosen_ids"], "chosen_ids", 1)
    frame_indices = _integer_tensor(sample_info["frame_indices"], "frame_indices", 2)
    if (chosen_ids.numel() == 0 or frame_indices.shape[0] != chosen_ids.shape[0]
            or frame_indices.shape[1] < context_frames):
        raise ValueError("chosen_ids and frame_indices must describe rows with a full context window")

    commands = torch.as_tensor(direction_local_xz).detach()
    if (commands.ndim != 2 or tuple(commands.shape) != (chosen_ids.shape[0], 2)
            or not commands.is_floating_point() or not torch.isfinite(commands).all()):
        raise ValueError("direction_local_xz must be finite floating-point [sampled_batch,2]")

    device = roots.device
    chosen_ids = chosen_ids.to(device=device, dtype=torch.long)
    current_indices = frame_indices[:, context_frames - 1].to(device=device, dtype=torch.long)
    if (chosen_ids < 0).any() or (chosen_ids >= roots.shape[0]).any():
        raise ValueError("chosen_ids index outside online terrain source rows")
    if (current_indices < 0).any() or (current_indices >= roots.shape[1]).any():
        raise ValueError("current frame index outside padded world root trajectories")
    if "motion_len" in batch:
        lengths = torch.as_tensor(batch["motion_len"], device=device).detach()
        if lengths.ndim != 1 or lengths.shape[0] != roots.shape[0]:
            raise ValueError("motion_len must contain one length per source row")
        if (current_indices >= lengths[chosen_ids]).any():
            raise ValueError("current frame index selects padded root data")
    commands = commands.to(device=device, dtype=roots.dtype)

    # A single rectangular query avoids a Python/CUDA synchronization chain
    # for every sampled source. Unknown padding remains NaN, hence invalid;
    # no interpolation, edge extension, or change to query coordinates occurs.
    source_ids = chosen_ids.tolist()
    selected_scenes = [scenes[index] for index in source_ids]
    for scene in selected_scenes:
        if not isinstance(scene, Mapping) or {'heightfield_y_m', 'min_point_world_xyz_m', 'dx_m'} - scene.keys():
            raise ValueError('terrain_scene is missing required geometry')
    heightfields = [torch.as_tensor(scene['heightfield_y_m']).detach() for scene in selected_scenes]
    if any(field.ndim != 2 or min(field.shape) < 1 for field in heightfields):
        raise ValueError('Terrain maps must be nonempty two-dimensional arrays')
    height = max(field.shape[0] for field in heightfields)
    width = max(field.shape[1] for field in heightfields)
    if batch_queries and height * width <= MAX_PACKED_TERRAIN_CELLS:
        storage_device = heightfields[0].device
        rows_per_chunk = max(1, MAX_PACKED_TERRAIN_CELLS // (height * width))
        point_chunks, valid_chunks = [], []
        # Bound temporary storage without falling back to per-source CUDA queries.
        for start in range(0, len(source_ids), rows_per_chunk):
            stop = min(start + rows_per_chunk, len(source_ids))
            packed = torch.full((stop - start, height, width), torch.nan,
                                dtype=torch.float32, device=storage_device)
            for index, field in enumerate(heightfields[start:stop]):
                packed[index, :field.shape[0], :field.shape[1]] = field.to(storage_device)
            scenes_chunk = selected_scenes[start:stop]
            origins = torch.stack([torch.as_tensor(scene['min_point_world_xyz_m'], device=storage_device,
                                                  dtype=torch.float32) for scene in scenes_chunk])
            spacing = torch.stack([torch.as_tensor(scene['dx_m'], device=storage_device,
                                                  dtype=torch.float32).reshape(()) for scene in scenes_chunk])
            root = roots[chosen_ids[start:stop], current_indices[start:stop]].to(dtype=torch.float32)
            observation = extract_local_terrain_observation(
                packed.to(device), origins.to(device), spacing.to(device), root[:, :3],
                torch.atan2(root[:, 4], root[:, 3]), commands[start:stop].to(dtype=torch.float32),
                config=terrain_observation_config(range_policy))
            point_chunks.append(observation.points.detach())
            valid_chunks.append(observation.valid.detach())
        return {'points': torch.cat(point_chunks), 'valid': torch.cat(valid_chunks)}

    groups = {}
    for sample_id, source_id in enumerate(source_ids):
        groups.setdefault(source_id, []).append(sample_id)
    points_by_sample = [None] * len(chosen_ids)
    valid_by_sample = [None] * len(chosen_ids)
    config = terrain_observation_config(range_policy)

    for source_id, sample_ids in groups.items():
        scene = scenes[source_id]
        if not isinstance(scene, Mapping):
            raise ValueError(f"terrain_scene[{source_id}] must be a mapping")
        required = {"heightfield_y_m", "min_point_world_xyz_m", "dx_m"}
        if required - scene.keys():
            raise ValueError(f"terrain_scene[{source_id}] is missing required geometry")

        selected_ids = torch.as_tensor(sample_ids, device=device, dtype=torch.long)
        source_indices = current_indices[selected_ids]
        root = roots[source_id, source_indices].to(dtype=torch.float32)
        yaw = torch.atan2(root[:, 4], root[:, 3])
        heightfield = torch.as_tensor(scene["heightfield_y_m"], device=device,
                                      dtype=torch.float32).detach()
        origin = torch.as_tensor(scene["min_point_world_xyz_m"], device=device,
                                 dtype=torch.float32).detach()
        spacing = torch.as_tensor(scene["dx_m"], device=device, dtype=torch.float32).detach()
        command = commands[selected_ids].to(dtype=torch.float32)
        observation = extract_local_terrain_observation(
            heightfield,
            origin,
            spacing,
            root[:, :3],
            yaw,
            command,
            config=config,
        )
        for group_index, sample_id in enumerate(sample_ids):
            points_by_sample[sample_id] = observation.points[group_index].detach()
            valid_by_sample[sample_id] = observation.valid[group_index].detach()

    if any(value is None for value in points_by_sample + valid_by_sample):
        raise RuntimeError("Online terrain query did not produce every sampled row")
    return {
        "points": torch.stack(points_by_sample).detach(),
        "valid": torch.stack(valid_by_sample).detach(),
    }
