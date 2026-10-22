"""Focused conversion safeguards for the pinned PARC legacy model."""

from enum import Enum
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from convert_parc_demo_checkpoint import (  # noqa: E402
    collect_source_extra,
    derive_config,
    verify_denoiser_parity,
)


class OldKey(Enum):
    OBS_KEY = "old_obs"
    OBS_FLAG_KEY = "old_obs_flag"
    TARGET_KEY = "old_target"
    TARGET_FLAG_KEY = "old_target_flag"
    PREV_STATE_KEY = "old_prev"
    PREV_STATE_FLAG_KEY = "old_prev_flag"
    PREV_STATE_NOISE_IND_KEY = "old_prev_noise"


class NewKey(Enum):
    OBS_KEY = "new_obs"
    OBS_FLAG_KEY = "new_obs_flag"
    TARGET_KEY = "new_target"
    TARGET_FLAG_KEY = "new_target_flag"
    PREV_STATE_KEY = "new_prev"
    PREV_STATE_FLAG_KEY = "new_prev_flag"
    PREV_STATE_NOISE_IND_KEY = "new_prev_noise"


class ScaledDenoiser(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x, conditions, timestep):
        return x * self.scale


class OffsetDenoiser(ScaledDenoiser):
    def forward(self, x, conditions, timestep):
        return super().forward(x, conditions, timestep) + 1


def test_config_comes_from_source_and_reference_may_only_differ_in_training_fields(tmp_path):
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    (legacy / "data" / "assets").mkdir(parents=True)
    (current / "data" / "assets").mkdir(parents=True)
    source_xml = legacy / "data" / "assets" / "humanoid.xml"
    current_xml = current / "data" / "assets" / "humanoid.xml"
    source_xml.write_text("same skeleton", encoding="utf-8")
    current_xml.write_text("same skeleton", encoding="utf-8")

    source_cfg = {
        "char_file": "data/assets/humanoid.xml",
        "device": "cuda:0",
        "use_wandb": True,
        "relative_z_style": "RELATIVE_TO_ROOT",
        "heightmap": {"max_h": 3.0},
        "test_ddim_stride": 50,
        "output_dir": "legacy-output",
        "epochs_per_checkpoint": 1000,
    }
    reference_cfg = dict(source_cfg)
    reference_cfg["output_dir"] = "reference-output"
    reference_cfg["epochs_per_checkpoint"] = 2500
    reference_cfg["sampler_stats_filepath"] = "training-only.txt"
    config_path = tmp_path / "reference.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(reference_cfg), encoding="utf-8")

    cfg = derive_config(SimpleNamespace(_cfg=source_cfg), config_path, legacy, current)

    assert cfg["relative_z_style"] == source_cfg["relative_z_style"]
    assert cfg["heightmap"] == source_cfg["heightmap"]
    assert cfg["test_ddim_stride"] == source_cfg["test_ddim_stride"]
    assert cfg["device"] == "cpu"
    assert cfg["char_file"] == str(current_xml.resolve())
    assert cfg["use_wandb"] is False


def test_config_rejects_inference_preprocessing_drift(tmp_path):
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    for repo in (legacy, current):
        (repo / "data" / "assets").mkdir(parents=True)
        (repo / "data" / "assets" / "humanoid.xml").write_text("same", encoding="utf-8")
    source_cfg = {
        "char_file": "data/assets/humanoid.xml",
        "device": "cpu",
        "use_wandb": False,
        "relative_z_style": "RELATIVE_TO_ROOT",
    }
    reference_cfg = dict(source_cfg, relative_z_style="RELATIVE_TO_ROOT_FLOOR")
    config_path = tmp_path / "reference.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(reference_cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="relative_z_style"):
        derive_config(SimpleNamespace(_cfg=source_cfg), config_path, legacy, current)


def test_source_feature_statistics_are_required_and_copied_to_cpu():
    mean = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    std = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1
    source = SimpleNamespace(
        _mdm_features_mean=mean,
        _mdm_features_std=std,
        _motion_frame_dof=4,
        _device="cuda:0",
        source_tag="pinned",
        source_array=np.array([1, 2, 3]),
    )

    extra = collect_source_extra(source)
    mean[0, 0] = -1

    assert extra["_mdm_features_mean"].device.type == "cpu"
    assert extra["_mdm_features_mean"][0, 0].item() == 0
    torch.testing.assert_close(extra["_mdm_features_std"], std)
    assert extra["source_tag"] == "pinned"
    np.testing.assert_array_equal(extra["source_array"], np.array([1, 2, 3]))
    assert extra["_device"] == "cpu"

    with pytest.raises(ValueError, match="_mdm_features_std"):
        collect_source_extra(SimpleNamespace(_mdm_features_mean=mean, _motion_frame_dof=4))


def test_denoiser_parity_checks_raw_and_ema_networks():
    source = SimpleNamespace(
        _denoise_model=ScaledDenoiser(2),
        _ema_denoise_model=ScaledDenoiser(3),
    )
    adapted = SimpleNamespace(
        _denoise_model=ScaledDenoiser(2),
        _ema_denoise_model=ScaledDenoiser(3),
        _motion_frame_dof=4,
        _num_prev_states=2,
        _target_dim=2,
    )
    cfg = {
        "seq_len": 3,
        "diffusion_timesteps": 4,
        "heightmap": {"local_grid": {
            "num_x_neg": 0, "num_x_pos": 0, "num_y_neg": 0, "num_y_pos": 0,
        }},
    }

    errors = verify_denoiser_parity(source, adapted, OldKey, NewKey, cfg)

    assert errors == {"raw": [0.0, 0.0, 0.0], "ema": [0.0, 0.0, 0.0]}

    adapted._ema_denoise_model = OffsetDenoiser(3)
    with pytest.raises(AssertionError, match="ema denoiser parity failed"):
        verify_denoiser_parity(source, adapted, OldKey, NewKey, cfg)
