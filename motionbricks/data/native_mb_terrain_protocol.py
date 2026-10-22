"""Explicit opt-in extension of MB, not a claim of unmodified paper training."""

from copy import deepcopy


TERRAIN_PROTOCOL = {
    "version": "native-mb-local-terrain-v2",
    "coordinates": "current_root_origin_body_yaw_y_up_meters",
    "observation_time": "last_context_frame",
    "map_access": "near_support_and_command_direction_patches_only",
    "future_terrain_observations": False,
    "root_height_condition": "relative_to_last_context_frame_meters",
    "training_root_height_source": "ground_truth_root",
    "inference_root_height_source": "predicted_root",
    "geometry_scope": "single_valued_heightfield_no_overhangs",
    "patch_shape": [2, 9, 9, 3],
    "max_horizontal_query_radius_m": 3.6,
    "query_geometry": {
        "patch_height": 9, "patch_width": 9, "near_spacing_m": 0.10,
        "forward_start_m": 0.20, "forward_spacing_m": 0.40,
        "forward_lateral_spacing_m": 0.15,
    },
    "runtime_postprocessing": "none",
}


INTENT_RANGE_POLICY = 'intent-horizon-range-v1'


def terrain_observation_config(range_policy=None):
    """Shared offline/online metric queries; the forward patch covers gap landings."""
    from .local_terrain_observation import LocalTerrainObservationConfig

    if range_policy not in (None, INTENT_RANGE_POLICY):
        raise ValueError('Unknown terrain observation range policy')
    return LocalTerrainObservationConfig(**TERRAIN_PROTOCOL['query_geometry'],
                                         scale_forward_with_intent=range_policy == INTENT_RANGE_POLICY)


def enable_terrain_conditioning(config):
    """Copy config and add zero-gated adapters; preserve loss and optimizer fields."""
    result = deepcopy(config)
    if "backbone_network" not in result["model"]:
        raise ValueError("Terrain conditioning is a Root/Pose extension, not a tokenizer option")
    result["model"]["backbone_network"]["args"]["terrain_conditioning"] = True
    return result


def load_flat_backbone_into_terrain(model, checkpoint):
    """Exact base weights, with only explicitly new terrain parameters missing."""
    import torch

    network = model.backbone_net
    if not network._args.get("terrain_conditioning", False):
        raise ValueError("The target must enable terrain conditioning")
    state = checkpoint.get("state_dict", {})
    prefix = "backbone_net."
    backbone = {key[len(prefix):]: value for key, value in state.items()
                if key.startswith(prefix)}
    target = network.state_dict()
    new_keys = {key for key in target if key.startswith("_terrain_")}
    original_keys = set(target) - new_keys
    if set(backbone) != original_keys:
        raise ValueError("Flat checkpoint keys must match every original backbone parameter exactly")
    for key, value in backbone.items():
        if value.shape != target[key].shape:
            raise ValueError(f"Flat checkpoint shape mismatch: {key}")
    if any(torch.count_nonzero(value).item() for key, value in target.items()
           if key in new_keys and (key.endswith(".gate") or key.endswith("_gate"))):
        raise ValueError("Warmstart requires untouched zero-gated terrain branches")
    merged = dict(target)
    merged.update(backbone)
    network.load_state_dict(merged, strict=True)
    return sorted(new_keys)


def validate_terrain_dataset(manifest, records):
    """Training cannot silently consume raw PARC poses or motion-only scenes."""
    contract = manifest.get("terrain_conditioning", {})
    for key in TERRAIN_PROTOCOL:
        if contract.get(key) != TERRAIN_PROTOCOL[key]:
            raise ValueError(f"Terrain dataset contract mismatch: {key}")
    acceptance = contract.get("acceptance", {})
    if acceptance.get("retarget_status") != "passed":
        raise ValueError("Terrain training requires accepted SOMA retargeting")
    if acceptance.get("motion_terrain_pairing_status") != "passed":
        raise ValueError("Terrain training requires accepted motion/terrain alignment")
    report = acceptance.get("report")
    if not report or report not in manifest.get("artifact_sha256", {}):
        raise ValueError("Terrain acceptance report must be bound by the dataset manifest")
    groups = {}
    for record in records:
        group, split = record.get("source_group"), record.get("split")
        if not group or split not in {"train", "validation", "test"}:
            raise ValueError("Terrain records require source_group and a known split")
        if group in groups and groups[group] != split:
            raise ValueError("Base-motion/terrain-family group leaks between splits")
        groups[group] = split
    return deepcopy(contract)
