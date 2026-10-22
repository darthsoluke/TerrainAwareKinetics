"""Export the hash-pinned official legacy demo model to PARC's new checkpoint.

No fitting or model changes. Loading the legacy Python object requires the
unmodified v0.1 source; forward parity is required before writing any weights.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib
import io
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

SOURCE_SHA256 = "bbe4358cfc4f2059554c53bdb6f896d2fd4f6312b94fab17c8fcf52538ab5890"

NON_INFERENCE_CONFIG_KEYS = {
    "create_dataset_config",
    "epochs_per_checkpoint",
    "input_model_path",
    "motion_lib_file",
    "output_dir",
    "sampler_save_filepath",
    "sampler_stats_filepath",
    "device",
    "char_file",
    "use_wandb",
}
SOURCE_STAT_KEYS = ("_mdm_features_mean", "_mdm_features_std")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        return super().find_class(module, name)


def derive_config(original, reference_config, legacy_repo, current_repo):
    """Use the checkpoint's config and reject inference-relevant YAML drift."""
    source_cfg = getattr(original, "_cfg", None)
    if not isinstance(source_cfg, dict):
        raise ValueError("Pinned source object does not contain a dictionary _cfg")
    reference_cfg = yaml.safe_load(reference_config.read_text(encoding="utf-8"))
    if not isinstance(reference_cfg, dict):
        raise ValueError("Reference config must contain a YAML mapping")

    keys = (set(source_cfg) | set(reference_cfg)) - NON_INFERENCE_CONFIG_KEYS
    missing = object()
    mismatches = [
        key for key in sorted(keys)
        if source_cfg.get(key, missing) != reference_cfg.get(key, missing)
    ]
    if mismatches:
        raise ValueError("Reference config differs from source _cfg: " + ", ".join(mismatches))

    source_char_file = Path(source_cfg["char_file"])
    if not source_char_file.is_absolute():
        source_char_file = legacy_repo / source_char_file
    current_char_file = current_repo / "data" / "assets" / "humanoid.xml"
    if sha256(source_char_file) != sha256(current_char_file):
        raise ValueError("Legacy and current character assets differ")

    cfg = deepcopy(source_cfg)
    cfg["device"] = "cpu"
    cfg["char_file"] = str(current_char_file.resolve())
    cfg["use_wandb"] = False
    return cfg


def collect_source_extra(original):
    """Copy runtime tensors and scalars, requiring the learned feature stats."""
    extra = {
        key: value.detach().cpu().clone()
        for key, value in original.__dict__.items()
        if isinstance(value, torch.Tensor)
    }
    extra.update({
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in original.__dict__.items()
        if isinstance(value, (int, float, bool, str, np.ndarray)) and key != "_device"
    })
    for key in SOURCE_STAT_KEYS:
        if key not in extra or not isinstance(extra[key], torch.Tensor):
            raise ValueError(f"Source checkpoint is missing tensor statistic {key}")
    if extra[SOURCE_STAT_KEYS[0]].shape != extra[SOURCE_STAT_KEYS[1]].shape:
        raise ValueError("Source feature mean/std shapes differ")
    if extra[SOURCE_STAT_KEYS[0]].ndim != 2:
        raise ValueError("Source feature mean/std must be rank-2 tensors")
    if extra[SOURCE_STAT_KEYS[0]].shape[1] != original._motion_frame_dof:
        raise ValueError("Source feature statistics do not match the source feature width")
    if not all(torch.isfinite(extra[key]).all() for key in SOURCE_STAT_KEYS):
        raise ValueError("Source feature statistics contain non-finite values")
    extra["_device"] = "cpu"
    return extra


def verify_denoiser_parity(original, adapted, old_key, new_key, cfg):
    """Check raw and EMA forwards with shared deterministic inputs."""
    pairs = {"raw": (original._denoise_model, adapted._denoise_model)}
    source_has_ema = hasattr(original, "_ema_denoise_model")
    target_has_ema = hasattr(adapted, "_ema_denoise_model")
    if source_has_ema != target_has_ema:
        raise ValueError("Source and current model disagree about EMA availability")
    if source_has_ema:
        source_ema = original._ema_denoise_model.cpu().eval()
        target_ema = adapted._ema_denoise_model
        target_ema.load_state_dict(source_ema.state_dict(), strict=True)
        target_ema.eval()
        pairs["ema"] = (source_ema, target_ema)

    old_net = original._denoise_model.cpu().eval()
    adapted._denoise_model.eval()
    pairs["raw"] = (old_net, adapted._denoise_model)

    grid = cfg["heightmap"]["local_grid"]
    obs_shape = (1 + grid["num_x_neg"] + grid["num_x_pos"],
                 1 + grid["num_y_neg"] + grid["num_y_pos"])
    batch_size = 2
    torch.manual_seed(42)
    errors = {name: [] for name in pairs}
    steps = sorted({0, cfg["diffusion_timesteps"] // 2, cfg["diffusion_timesteps"] - 1})
    with torch.no_grad():
        for step in steps:
            x = torch.randn(batch_size, cfg["seq_len"], adapted._motion_frame_dof)
            conditions = {
                "OBS_KEY": torch.randn(batch_size, 1, *obs_shape),
                "OBS_FLAG_KEY": torch.tensor([True, True]),
                "TARGET_KEY": torch.randn(batch_size, 1, adapted._target_dim),
                "TARGET_FLAG_KEY": torch.tensor([True, True]),
                "PREV_STATE_KEY": torch.randn(
                    batch_size, adapted._num_prev_states, adapted._motion_frame_dof
                ),
                "PREV_STATE_FLAG_KEY": torch.tensor([True, False]),
                "PREV_STATE_NOISE_IND_KEY": torch.tensor([True, False]),
            }
            t = torch.full((batch_size,), step, dtype=torch.int64)
            for name, (source_net, current_net) in pairs.items():
                old_conditions = {
                    old_key[key]: value.clone() for key, value in conditions.items()
                }
                new_conditions = {
                    new_key[key]: value.clone() for key, value in conditions.items()
                }
                before = source_net(x.clone(), old_conditions, t.clone())
                after = current_net(x.clone(), new_conditions, t.clone())
                try:
                    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
                except AssertionError as error:
                    raise AssertionError(f"{name} denoiser parity failed at timestep {step}") from error
                errors[name].append(float((before - after).abs().max()))
    return errors


def convert(source, config, legacy_repo, current_repo, output):
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Only the verified official demo object may be unpickled")
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    sys.path.insert(0, str(legacy_repo.resolve()))
    sys.path.insert(0, str(current_repo.resolve()))
    sys.modules.setdefault("parc", importlib.import_module("PARC"))
    from diffusion.diffusion_util import MDMKeyType as OldKey
    from parc.motion_generator.diffusion_util import MDMKeyType as NewKey
    from parc.motion_generator.mdm import MDM

    with source.open("rb") as stream:
        original = CPUUnpickler(stream).load()
    print("Loaded pinned legacy object", type(original).__name__, flush=True)
    cfg = derive_config(original, config, legacy_repo, current_repo)
    adapted = MDM(cfg)
    old_net = original._denoise_model.cpu().eval()
    adapted._denoise_model.load_state_dict(old_net.state_dict(), strict=True)
    adapted._denoise_model.eval()
    old_slices = {k.name: (v.start, v.stop, v.step) for k, v in original._feature_slices.items()}
    new_slices = {k.name: (v.start, v.stop, v.step) for k, v in adapted._feature_slices.items()}
    if old_slices != new_slices:
        raise ValueError("Motion feature ordering changed")
    if cfg.get("normalize_dofs", False):
        raise ValueError("Normalized legacy checkpoints need an explicit statistics adapter")

    parity = verify_denoiser_parity(original, adapted, OldKey, NewKey, cfg)

    # Export tensor/scalar attributes using the current upstream checkpoint contract.
    extra = collect_source_extra(original)
    payload = {"cfg": cfg, "model": old_net.state_dict(), "extra": extra,
               "optimizer": adapted._optimizer.state_dict()}
    if hasattr(original, "_ema_denoise_model"):
        payload["ema_model"] = original._ema_denoise_model.state_dict()
        payload["ema_meta"] = {
            key: getattr(original, key, None)
            for key in ("ema_step", "ema_decay", "ema_start", "ema_update_rate")
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {
        "source": str(source.resolve()), "source_sha256": SOURCE_SHA256,
        "source_stage": "official april272025/iter_3/p1_train_gen/model_20000",
        "highest_stage_claim": "largest generator found in accessible official releases, not verified final paper iteration",
        "archive_sha256": "330e19a24334d903a9311f489b83ce161022534a917b42f13322096482d66391",
        "source_url": "https://github.com/mshoe/PARC#dataset-and-models",
        "output": str(output.resolve()), "output_sha256": sha256(output),
        "parameters": sum(p.numel() for p in old_net.parameters()),
        "architecture": {k: cfg[k] for k in ("d_model", "num_layers", "num_heads", "diffusion_timesteps", "test_ddim_stride", "seq_len", "sequence_fps", "use_hf_collision_loss")},
        "forward_max_abs_errors": parity["raw"],
        "ema_forward_max_abs_errors": parity.get("ema"),
        "feature_slices": new_slices,
        "preserved_source_stats": {
            key: {
                "shape": list(extra[key].shape),
                "sha256": hashlib.sha256(extra[key].contiguous().numpy().tobytes()).hexdigest(),
            }
            for key in SOURCE_STAT_KEYS
        },
        "weight_selection": "raw and EMA states preserved separately; selection is left to the consuming loader",
        "training_performed": False, "physics_tracking_performed": False,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "config", "legacy-repo", "current-repo", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.config, args.legacy_repo, args.current_repo, args.output), indent=2))
