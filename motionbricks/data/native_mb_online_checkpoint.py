"""Verified checkpoint bindings for online-conditioned native Root/Pose runs."""

import hashlib
import json
from pathlib import Path


ONLINE_CHECKPOINT_VERSION = "native-mb-online-checkpoint-v1"
ONLINE_RESUME_PLUMBING_SOURCES = frozenset({
    "scripts/train_native_mb_control.py",
    "motionbricks/data/native_mb_online_checkpoint.py",
})


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol():
    from .native_mb_online_conditions import ONLINE_PROTOCOL

    return ONLINE_PROTOCOL


def validate_online_config(config, stage):
    """Require the saved native config to identify the same online protocol."""
    protocol_version = _protocol()["version"]
    model = config.get("model", {})
    model_args = model.get("args", {})
    backbone = model.get("backbone_network", {})
    backbone_args = backbone.get("args", {})
    if (model_args.get("online_conditioning") != protocol_version
            or backbone_args.get("online_conditioning") != protocol_version):
        raise ValueError("Native config does not enable the expected online-conditioning protocol")
    if stage == "root" and backbone_args.get("root_component_conditioning") is not True:
        raise ValueError("Online Root config must enable root component conditioning")
    return protocol_version


def _provenance_identity(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("Online control runs require verified PARC-only source provenance")
    return {key: item for key, item in value.items() if key != "actual_loader_origin_counts"}


def validate_online_control_run(
    contract,
    result,
    *,
    stage,
    dataset_manifest_sha256,
    parc_only_sources,
    terrain_conditioning,
    require_completed=False,
):
    """Bind a same-stage online checkpoint to the target data and protocols."""
    from .native_mb_terrain_protocol import TERRAIN_PROTOCOL

    expected_stage = f"native_{stage}"
    if contract.get("stage") != expected_stage:
        raise ValueError("Online checkpoint stage mismatch")
    allowed_status = {"completed"} if require_completed else {"completed", "trained_not_reviewed"}
    if contract.get("status") not in allowed_status:
        raise ValueError("Online checkpoint is not in an accepted fully trained state")
    if contract.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("Online checkpoint dataset manifest hash mismatch")
    if _provenance_identity(contract.get("parc_only_sources")) != _provenance_identity(parc_only_sources):
        raise ValueError("Online checkpoint PARC-only provenance mismatch")

    source_terrain = contract.get("terrain_conditioning")
    if (not isinstance(source_terrain, dict)
            or source_terrain.get("version") != TERRAIN_PROTOCOL["version"]
            or source_terrain != terrain_conditioning):
        raise ValueError("Online checkpoint terrain contract mismatch")

    online = contract.get("online_contract")
    protocol = _protocol()
    if (not isinstance(online, dict) or online.get("version") != ONLINE_CHECKPOINT_VERSION
            or online.get("protocol") != protocol
            or online.get("protocol_version") != protocol["version"]):
        raise ValueError("Online checkpoint protocol contract mismatch")
    if (not isinstance(result, dict) or type(result.get("optimizer_updates")) is not int
            or result["optimizer_updates"] < 1
            or result["optimizer_updates"] != contract.get("stop_steps")):
        raise ValueError("Online checkpoint did not reach its recorded bounded stop")
    checkpoint_sha256 = result.get("final_checkpoint_sha256")
    if not _is_sha256(checkpoint_sha256):
        raise ValueError("Online checkpoint result lacks a valid checkpoint SHA256")
    return True


def validate_online_source_compatibility(contract, current_source_lf_sha256):
    """Allow only the audited continuation-plumbing files to differ from a run."""
    saved = contract.get("source_lf_sha256")
    if not isinstance(saved, dict):
        raise ValueError("Online checkpoint lacks source provenance hashes")
    revisions = []
    for relative, current in sorted(current_source_lf_sha256.items()):
        previous = saved.get(relative)
        if previous == current:
            continue
        if (relative not in ONLINE_RESUME_PLUMBING_SOURCES
                or not _is_sha256(previous) or not _is_sha256(current)):
            raise ValueError(f"Online checkpoint source changed: {relative}")
        revisions.append({
            "source": relative,
            "previous_sha256": previous,
            "current_sha256": current,
            "change_scope": "online continuation validation and command plumbing only",
        })
    return revisions


def validate_online_resume_run(
    run,
    expected,
    *,
    parc_only_sources,
    terrain_conditioning,
    expected_model_state,
    component_mask_parameter_keys,
    max_updates,
):
    """Verify an exact Root resume source, including its complete Lightning state."""
    import torch

    run = Path(run).resolve()
    contract_path = run / "run_contract.json"
    result_path = run / "training_result.json"
    config_path = run / "native_config.yaml"
    checkpoint_path = run / "checkpoints" / "final.ckpt"
    if not all(path.is_file() for path in (contract_path, result_path, config_path, checkpoint_path)):
        raise ValueError("Online resume source is missing a required run artifact")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_online_control_run(
        contract, result, stage="root",
        dataset_manifest_sha256=expected["dataset_manifest_sha256"],
        parc_only_sources=parc_only_sources,
        terrain_conditioning=terrain_conditioning,
    )

    if (contract.get("dataset") is None
            or Path(contract["dataset"]).resolve() != Path(expected["dataset"]).resolve()):
        raise ValueError("Online resume dataset path mismatch")
    manifest_path = Path(contract["dataset"]) / "native_manifest.json"
    if (not manifest_path.is_file()
            or sha256_file(manifest_path) != expected["dataset_manifest_sha256"]):
        raise ValueError("Online resume dataset manifest changed")
    if sha256_file(config_path) != contract.get("config_sha256"):
        raise ValueError("Online resume source config SHA256 mismatch")

    exact_fields = (
        "dataset_manifest_sha256", "config_sha256", "seed", "batch_size",
        "original_schedule_steps", "training_profile", "runtime_numerics",
        "runtime_versions", "model_class", "network_class", "parameter_count",
        "tokenizer",
    )
    for key in exact_fields:
        if contract.get(key) != expected.get(key):
            raise ValueError(f"Online resume mismatch: {key}")
    if contract.get("online_contract", {}).get("root_dependency") is not None:
        raise ValueError("Online Root resume cannot bind an upstream Root dependency")
    if expected.get("stop_steps", 0) > max_updates or max_updates != expected.get("original_schedule_steps"):
        raise ValueError("Online resume target must stay within the original update schedule")

    saved_step = result.get("optimizer_updates")
    if (type(saved_step) is not int or saved_step < 1
            or saved_step != contract.get("stop_steps") or saved_step >= expected["stop_steps"]
            or saved_step > max_updates):
        raise ValueError("Online resume target must exceed the verified saved update count")
    checkpoint_sha256 = result.get("final_checkpoint_sha256")
    if not _is_sha256(checkpoint_sha256) or sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("Online resume checkpoint SHA256 mismatch")

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or saved.get("global_step") != saved_step:
        raise ValueError("Online resume checkpoint update count mismatch")
    state = saved.get("state_dict")
    if not isinstance(state, dict) or set(state) != set(expected_model_state):
        raise ValueError("Online resume model state keys mismatch")
    for key, expected_value in expected_model_state.items():
        if not hasattr(state[key], "shape") or state[key].shape != expected_value.shape:
            raise ValueError(f"Online resume model state shape mismatch: {key}")

    expected_mask_keys = sorted(
        key for key in expected_model_state if "component_mask" in key)
    relative_mask_keys = sorted(component_mask_parameter_keys)
    normalized_mask_keys = sorted(
        key.split("backbone_net.", 1)[1] for key in expected_mask_keys
        if "backbone_net." in key)
    if (not relative_mask_keys or normalized_mask_keys != relative_mask_keys
            or sorted(key for key in state if "component_mask" in key) != expected_mask_keys):
        raise ValueError("Online resume component-mask parameter keys mismatch")
    if contract.get("online_contract", {}).get("component_mask_parameter_keys") != relative_mask_keys:
        raise ValueError("Online resume contract component-mask parameter listing mismatch")

    optimizer_states = saved.get("optimizer_states")
    scheduler_states = saved.get("lr_schedulers")
    if (not isinstance(optimizer_states, list) or not optimizer_states
            or not isinstance(scheduler_states, list) or not scheduler_states):
        raise ValueError("Online resume requires full optimizer and scheduler checkpoint state")
    if any(not isinstance(item, dict) or not item.get("state") or not item.get("param_groups")
           for item in optimizer_states):
        raise ValueError("Online resume optimizer checkpoint state is incomplete")

    source_revisions = validate_online_source_compatibility(
        contract, expected.get("source_lf_sha256", {}))
    return {
        "run": str(run),
        "checkpoint": str(checkpoint_path),
        "sha256": checkpoint_sha256,
        "saved_step": saved_step,
        "source_revisions": source_revisions,
        "component_mask_state_keys": expected_mask_keys,
        "component_mask_state_sha256": _tensor_state_digest(state, expected_mask_keys),
        "optimizer_state_sha256": state_tree_digest(optimizer_states),
        "scheduler_state_sha256": state_tree_digest(scheduler_states),
        "optimizer_state": "restored_from_verified_full_lightning_checkpoint",
        "scheduler_state": "restored_from_verified_full_lightning_checkpoint",
        "sampling_note": "RNGs reseeded; not bitwise uninterrupted sampling",
    }


def _tensor_state_digest(state, keys):
    digest = hashlib.sha256()
    for name in sorted(keys):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def state_tree_digest(value):
    """Hash nested optimizer/scheduler state independent of tensor device."""
    import torch

    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=repr):
                digest.update(type(key).__name__.encode())
                digest.update(repr(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"list" if isinstance(item, list) else b"tuple")
            for child in item:
                update(child)
        else:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def validate_parc_terrain_warmstart_run(
    contract,
    result,
    *,
    stage,
    dataset_manifest_sha256,
    parc_only_sources,
    terrain_conditioning,
):
    """Accept a same-stage PARC terrain baseline without requiring online metadata."""
    from .native_mb_terrain_protocol import TERRAIN_PROTOCOL

    if contract.get("stage") != f"native_{stage}":
        raise ValueError("Online warmstart requires a same-stage PARC terrain run")
    if contract.get("status") not in {"completed", "trained_not_reviewed"}:
        raise ValueError("Online warmstart source training is incomplete")
    if contract.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("Online warmstart dataset manifest hash mismatch")
    if _provenance_identity(contract.get("parc_only_sources")) != _provenance_identity(parc_only_sources):
        raise ValueError("Online warmstart PARC-only provenance mismatch")
    source_terrain = contract.get("terrain_conditioning")
    if (not isinstance(source_terrain, dict)
            or source_terrain.get("version") != TERRAIN_PROTOCOL["version"]
            or source_terrain != terrain_conditioning):
        raise ValueError("Online warmstart terrain contract mismatch")
    if not isinstance(result, dict) or result.get("optimizer_updates", 0) < 1:
        raise ValueError("Online warmstart source has no completed optimizer updates")
    if not _is_sha256(result.get("final_checkpoint_sha256")):
        raise ValueError("Online warmstart source lacks a valid checkpoint SHA256")
    return True


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def verify_same_tokenizer_dependency(source, target):
    """Online Pose warmstarts may not substitute or rebind the native tokenizer."""
    if not isinstance(source, dict) or not isinstance(target, dict) or source != target:
        raise ValueError("Online Pose warmstart must preserve its exact tokenizer dependency")
    source_sha = source.get("checkpoint_sha256")
    if not _is_sha256(source_sha) or source_sha != target.get("checkpoint_sha256"):
        raise ValueError("Online Pose warmstart tokenizer checkpoint SHA256 mismatch")
    return source_sha


def copy_online_warmstart_backbone(
    target_backbone, source_backbone, stage, *, verified_online_source=False,
):
    """Copy a verified source, preserving learned masks only for online checkpoints."""
    import torch

    target_state = target_backbone.state_dict()
    source_state = (source_backbone.state_dict() if hasattr(source_backbone, "state_dict")
                    else source_backbone)
    if not isinstance(source_state, dict):
        raise ValueError("Online warmstart source has no backbone state dictionary")
    target_keys, source_keys = set(target_state), set(source_state)
    if stage == "root":
        new_keys = {key for key in target_keys if "component_mask" in key}
        source_new_keys = {key for key in source_keys if "component_mask" in key}
        if not new_keys:
            raise ValueError("Online Root target has no component-mask parameters")
        if verified_online_source:
            if new_keys != source_new_keys:
                raise ValueError("Verified online Root warmstart component-mask keys must match exactly")
        elif source_new_keys:
            raise ValueError("Online Root warmstart source must omit target component-mask parameters")
    elif stage == "pose":
        new_keys = set()
        if any("component_mask" in key for key in target_keys | source_keys):
            raise ValueError("Online Pose warmstart must not introduce component-mask parameters")
    else:
        raise ValueError(f"Unknown online warmstart stage: {stage!r}")

    original_keys = target_keys if verified_online_source else target_keys - new_keys
    if original_keys != source_keys:
        message = ("Online warmstart backbone keys must match exactly" if verified_online_source
                   else "Online warmstart original backbone keys must match exactly")
        raise ValueError(message)
    for key in sorted(source_keys):
        if target_state[key].shape != source_state[key].shape:
            raise ValueError(f"Online warmstart backbone shape mismatch: {key}")

    if stage == "root" and not verified_online_source:
        nonzero_new = [key for key in new_keys if torch.count_nonzero(target_state[key]).item()]
        if nonzero_new:
            raise ValueError("Online Root component-mask parameters must remain freshly zero-initialized")
        merged = dict(target_state)
        merged.update({key: source_state[key] for key in original_keys})
        target_backbone.load_state_dict(merged, strict=True)
        if any(torch.count_nonzero(target_backbone.state_dict()[key]).item() for key in new_keys):
            raise RuntimeError("Online Root warmstart modified a fresh component-mask parameter")
    else:
        if target_keys != source_keys:
            raise ValueError("Online Pose warmstart backbone keys must match exactly")
        target_backbone.load_state_dict(source_state, strict=True)

    return {
        "transfer_scope": ("all_backbone_parameters" if stage == "pose" or verified_online_source
                           else "original_backbone_parameters_only"),
        "strict_key_and_shape_match": True,
        "copied_parameter_count": len(source_keys),
        "new_parameter_keys": sorted(new_keys) if stage == "root" else [],
        "new_parameters_kept_fresh_zero": stage == "root" and not verified_online_source,
        "component_mask_source": ("preserved_from_verified_online_source" if verified_online_source
                                  and stage == "root" else "fresh_zero" if stage == "root" else None),
        "optimizer_state": "not_loaded_reset",
        "scheduler_state": "not_loaded_reset",
    }


def verify_online_root_dependency(dependency, *, dataset_manifest_sha256,
                                  parc_only_sources, terrain_conditioning):
    """Verify a Pose run's frozen Root identity without constructing another model."""
    from .native_mb_terrain_protocol import TERRAIN_PROTOCOL

    if not isinstance(dependency, dict):
        raise ValueError("Online Pose contract is missing its Root dependency")
    run = Path(dependency.get("run", "")).resolve()
    contract_path = run / "run_contract.json"
    result_path = run / "training_result.json"
    config_path = run / "native_config.yaml"
    checkpoint_path = run / "checkpoints" / "final.ckpt"
    if not all(path.is_file() for path in (contract_path, result_path, config_path, checkpoint_path)):
        raise ValueError("Online Root dependency is missing a required run artifact")

    root_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    root_result = json.loads(result_path.read_text(encoding="utf-8"))
    root_dataset_manifest = Path(root_contract.get("dataset", "")) / "native_manifest.json"
    if (not root_dataset_manifest.is_file()
            or sha256_file(root_dataset_manifest) != dataset_manifest_sha256):
        raise ValueError("Online Root dependency dataset manifest changed")
    if root_contract.get("config_sha256") != sha256_file(config_path):
        raise ValueError("Online Root dependency config SHA256 mismatch")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (checkpoint_sha256 != root_result.get("final_checkpoint_sha256")
            or checkpoint_sha256 != dependency.get("checkpoint_sha256")):
        raise ValueError("Online Root dependency checkpoint SHA256 mismatch")
    if (root_result.get("optimizer_updates") != root_contract.get("stop_steps")
            or root_result.get("optimizer_updates", 0) < 1):
        raise ValueError("Online Root dependency training did not reach its bounded stop")
    if dependency.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("Online Root dependency dataset identity mismatch")
    if not _is_sha256(dependency.get("state_digest")):
        raise ValueError("Online Root dependency state digest is invalid")
    if dependency.get("end_state_digest") != dependency.get("state_digest"):
        raise ValueError("Online Root dependency changed before Pose training completed")
    if (dependency.get("frozen") is not True or dependency.get("eval_mode") is not True
            or dependency.get("registered_with_pose_model") is not False):
        raise ValueError("Online Root dependency must be frozen, in eval mode and unregistered")

    validate_online_control_run(
        root_contract,
        root_result,
        stage="root",
        dataset_manifest_sha256=dataset_manifest_sha256,
        parc_only_sources=parc_only_sources,
        terrain_conditioning=terrain_conditioning,
        require_completed=False,
    )
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_online_config(config, "root")
    if (not root_contract.get("terrain_conditioning")
            or root_contract["terrain_conditioning"].get("version") != TERRAIN_PROTOCOL["version"]):
        raise ValueError("Online Root dependency must use the accepted terrain protocol")
    if root_contract.get("stage") != "native_root":
        raise ValueError("Online Pose dependency must be a Root run")
    return {
        "run": str(run),
        "checkpoint_sha256": checkpoint_sha256,
        "state_digest": dependency["state_digest"],
        "end_state_digest": dependency["end_state_digest"],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "status": root_contract["status"],
        "frozen": dependency["frozen"],
        "eval_mode": dependency["eval_mode"],
        "registered_with_pose_model": dependency["registered_with_pose_model"],
    }
