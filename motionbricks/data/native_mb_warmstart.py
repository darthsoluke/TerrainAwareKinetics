"""Strict, weights-only initialization from a verified 50k native checkpoint."""

import hashlib
import json
from pathlib import Path


INHERITED_UPDATES = 50000
STATS_FILES = ("mean.npy", "std.npy")
CONTINUATION_CONFIG_ALLOWED_DIFFERENCES = (
    "data.folder",
    "data_root",
    "motion_rep.stats.folder",
    "trainer.max_steps",
    "loggers.max_steps",
    "loggers.out_dir",
    "loggers.wandb_run",
    "callbacks.callback_dict.checkpoint_epoch_cb.dirpath",
    "callbacks.callback_dict.eval_visualize.out_dir",
    "out_dir",
    "run_dir",
    "id",
    "wandb_run",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_config(config):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=False)
    except ImportError:
        pass
    return config


def _architecture_signature(config):
    config = _plain_config(config)
    model = config.get("model", {})
    return {key: model.get(key) for key in (
        "args", "backbone_network", "pose_vqvae_network", "root_vqvae_network",
    )}


def _first_config_mismatch(source, target, allowed_differences, path=""):
    """Return the first differing dotted path outside an explicit allowlist."""
    if path in allowed_differences:
        return None
    if isinstance(source, dict) and isinstance(target, dict):
        for key in sorted(source.keys() | target.keys()):
            child = f"{path}.{key}" if path else str(key)
            if key not in source or key not in target:
                if child not in allowed_differences:
                    return child
                continue
            mismatch = _first_config_mismatch(source[key], target[key], allowed_differences, child)
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(source, list) and isinstance(target, list):
        if len(source) != len(target):
            return path
        for index, (source_item, target_item) in enumerate(zip(source, target)):
            mismatch = _first_config_mismatch(source_item, target_item, allowed_differences,
                                              f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    return None if source == target else path


def validate_warmstart_run(run, stage, target_dataset, target_config):
    """Validate a finished 50k source before loading any weights."""
    run = Path(run).resolve()
    contract = json.loads((run / "run_contract.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    source_stage = f"native_{stage}"
    allowed_status = {"completed"}
    if stage in {"root", "pose"}:
        allowed_status.add("trained_not_reviewed")
    if contract.get("stage") != source_stage or contract.get("status") not in allowed_status:
        raise ValueError("Warmstart requires a completed baseline run of the same stage")

    source_config_path = run / "native_config.yaml"
    if _sha256(source_config_path) != contract.get("config_sha256"):
        raise ValueError("Warmstart baseline config hash does not match run_contract")
    source_dataset = Path(contract["dataset"])
    source_manifest_path = source_dataset / "native_manifest.json"
    if _sha256(source_manifest_path) != contract.get("dataset_manifest_sha256"):
        raise ValueError("Warmstart baseline manifest hash does not match run_contract")
    source_manifest = json.loads(source_manifest_path.read_text())

    checkpoint = run / "checkpoints" / "final.ckpt"
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != result.get("final_checkpoint_sha256"):
        raise ValueError("Warmstart checkpoint hash does not match training_result")
    if result.get("optimizer_updates") != INHERITED_UPDATES:
        raise ValueError("Warmstart baseline must contain exactly 50000 optimizer updates")

    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("global_step") != INHERITED_UPDATES:
        raise ValueError("Warmstart checkpoint global_step must be exactly 50000")

    target_dataset = Path(target_dataset)
    artifacts = source_manifest.get("artifact_sha256", {})
    for name in STATS_FILES:
        source_stats = source_dataset / "stats" / name
        target_stats = target_dataset / "stats" / name
        relative = f"stats/{name}"
        if relative not in artifacts or _sha256(source_stats) != artifacts[relative]:
            raise ValueError(f"Warmstart baseline manifest does not bind {relative}")
        if source_stats.read_bytes() != target_stats.read_bytes():
            raise ValueError(f"Warmstart normalization mismatch: stats/{name}")

    import yaml

    source_config = yaml.safe_load(source_config_path.read_text())
    if _architecture_signature(source_config) != _architecture_signature(target_config):
        raise ValueError("Warmstart model architecture is incompatible with the target")

    return {
        "run": str(run),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "baseline_stage": source_stage,
        "baseline_dataset": str(source_dataset),
        "baseline_dataset_manifest_sha256": contract.get("dataset_manifest_sha256"),
        "baseline_optimizer_updates": INHERITED_UPDATES,
        "weights_only": True,
        "transfer_scope": ("full_tokenizer_state" if stage == "tokenizer"
                           else "backbone_net_only"),
        "strict_key_and_shape_match": True,
        "supporting_modules_preserved": stage in {"root", "pose"},
        "optimizer_state": "reset; fresh target-profile optimizer",
        "scheduler_state": "reset; fresh target-profile schedule",
        "exact_resume": False,
        "_checkpoint_state": saved,
        "normalization_byte_identical": True,
        "normalization_files": {
            name: _sha256(target_dataset / "stats" / name) for name in STATS_FILES
        },
    }


def load_warmstart_weights(model, checkpoint, stage, torch_module=None):
    """Load only approved model weights, leaving control support nets untouched."""
    if torch_module is None:
        import torch as torch_module
    saved = checkpoint if isinstance(checkpoint, dict) else torch_module.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    state = saved.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Warmstart checkpoint has no Lightning state_dict")
    if stage == "tokenizer":
        model.load_state_dict(state, strict=True)
        return
    if stage not in {"root", "pose"}:
        raise ValueError(f"Unknown native warmstart stage: {stage!r}")
    prefix = "backbone_net."
    backbone_state = {key[len(prefix):]: value for key, value in state.items()
                      if key.startswith(prefix)}
    if not backbone_state:
        raise ValueError("Warmstart checkpoint has no backbone_net state")
    model.backbone_net.load_state_dict(backbone_state, strict=True)


def validate_dataset_continuation(run, target_dataset, target_config, *, seed, batch_size, stop_steps):
    """Explicit data transition with full original training state, not a warm restart."""
    import yaml
    verified = validate_warmstart_run(run, 'tokenizer', target_dataset, target_config)
    saved = verified.pop('_checkpoint_state')
    previous = json.loads((Path(run) / 'run_contract.json').read_text())
    source_config = yaml.safe_load((Path(run) / 'native_config.yaml').read_text())
    target = _plain_config(target_config)
    for key, expected in (('seed', seed), ('batch_size', batch_size)):
        if previous.get(key) != expected:
            raise ValueError(f'Dataset continuation changes {key}')
    mismatch = _first_config_mismatch(source_config, target,
                                      set(CONTINUATION_CONFIG_ALLOWED_DIFFERENCES))
    if mismatch is not None:
        raise ValueError(f'Dataset continuation changes config at {mismatch}')
    if stop_steps <= verified['baseline_optimizer_updates']:
        raise ValueError('Dataset continuation needs a larger total update target')
    if (len(saved.get('optimizer_states', [])) != 1 or not saved['optimizer_states'][0].get('state')
            or len(saved.get('lr_schedulers', [])) != 1 or not saved.get('loops')):
        raise ValueError('Dataset continuation requires optimizer, scheduler and loop state')
    return dict(run=str(Path(run).resolve()), checkpoint=verified['checkpoint'],
        sha256=verified['checkpoint_sha256'], saved_step=verified['baseline_optimizer_updates'],
        restore='Lightning model, optimizer, scheduler and training loop state; deliberate dataset transition',
        source_dataset_manifest_sha256=verified['baseline_dataset_manifest_sha256'],
        target_dataset_manifest_sha256=_sha256(Path(target_dataset) / 'native_manifest.json'),
        allowed_config_differences=list(CONTINUATION_CONFIG_ALLOWED_DIFFERENCES),
        normalization_byte_identical=True, sampling_note='Dataset changed; RNGs reseeded; not the uninterrupted sample sequence',
        expected_initial_lr=saved['optimizer_states'][0]['param_groups'][0]['lr'],
        expected_optimizer_state_entries=len(saved['optimizer_states'][0]['state']),
        expected_scheduler_last_epoch=saved['lr_schedulers'][0]['last_epoch'])
