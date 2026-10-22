"""Paper scalar settings, exact release preservation, and public optimizer wiring."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "motionbricks"))

from motionbricks.data.native_mb_models import native_config
from motionbricks.data.native_mb_paper_protocol import (
    PAPER_BATCH_PER_GPU,
    PAPER_SOURCE,
    PAPER_STEPS,
    configure_paper_optimizer,
    paper_config,
    profile_notes,
    validate_paper_arguments,
)


@pytest.mark.parametrize("stage", ["tokenizer", "root", "pose"])
def test_paper_scalars_preserve_every_other_release_setting(stage):
    config = paper_config(stage)
    assert PAPER_STEPS == 2000000
    assert PAPER_BATCH_PER_GPU == 256
    assert config.trainer.max_steps == PAPER_STEPS
    assert config.model.optimizer._target_ == "torch.optim.Adam"
    assert config.model.optimizer.lr == 5e-5
    assert config.model.optimizer.weight_decay == 0.0
    assert config.model.scheduler.num_warmup_steps == 10000
    assert config.model.scheduler.num_training_steps == PAPER_STEPS
    assert config.model.scheduler.final_lr == 2e-6
    assert config.model.args.min_tokens == 3 and config.model.args.max_tokens == 16

    # Compare the whole unresolved tree, including interpolations and missing fields.
    actual = OmegaConf.to_container(config, resolve=False)
    release = OmegaConf.to_container(native_config(stage, PAPER_STEPS), resolve=False)
    original_optimizer = release["model"]["optimizer"].copy()
    actual["model"]["args"]["min_tokens"] = release["model"]["args"]["min_tokens"]
    for section, fields in {
        "optimizer": ("_target_", "lr", "weight_decay"),
        "scheduler": ("num_warmup_steps", "num_training_steps", "final_lr"),
    }.items():
        for field in fields:
            actual["model"][section][field] = release["model"][section][field]
    assert actual == release
    assert native_config(stage, PAPER_STEPS).model.optimizer == original_optimizer


@pytest.mark.parametrize("stage", ["tokenizer", "root", "pose"])
def test_profile_notes_disclose_scope_without_fixed_dataset_size(stage):
    notes = profile_notes(stage)
    assert json.loads(json.dumps(notes)) == notes
    assert notes["training_profile"] == "paper_s7_4_optimizer_release_architecture_soma30"
    assert notes["paper_reference"] == PAPER_SOURCE
    assert PAPER_SOURCE == "https://arxiv.org/html/2604.24833v1#S7.SS4"
    assert notes["architecture_authority"] == "official executable source and release hparams"
    text = json.dumps(notes)
    assert "basic flat locomotion subset" in text and "full700h" in text
    assert "not global batch equivalence" in text
    assert "not claimed exact paper reimplementation" in text
    assert "betas/eps standard torch defaults" in text
    assert "4.7h" not in text and "2003" not in text
    assert "12 heads" in notes["architecture_ambiguity"]
    assert "without arbitrary edits" in notes["architecture_ambiguity"]


@pytest.mark.parametrize("stage", ["tokenizer", "root", "pose"])
def test_public_optimizer_callable_is_adam_and_scheduler_reaches_paper_endpoints(stage):
    import torch
    from motionbricks.motionlib.train.scheduler import WarmupCosineScheduler

    config = paper_config(stage)
    scheduler_callable = object()
    training_step = object()
    model = SimpleNamespace(optimizer=object(), scheduler=scheduler_callable, training_step=training_step)
    configure_paper_optimizer(model, config)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = model.optimizer([parameter])
    assert type(optimizer) is torch.optim.Adam
    assert optimizer.param_groups[0]["lr"] == 5e-5
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
    assert model.scheduler is scheduler_callable and model.training_step is training_step

    scheduler = WarmupCosineScheduler(
        optimizer,
        num_warmup_steps=config.model.scheduler.num_warmup_steps,
        num_training_steps=config.model.scheduler.num_training_steps,
        final_lr=config.model.scheduler.final_lr,
    )
    for step, expected in ((0, 0.0), (10000, 5e-5), (PAPER_STEPS, 2e-6)):
        scheduler.last_epoch = step
        assert scheduler.get_lr() == pytest.approx([expected])


@pytest.mark.parametrize("field,value", [
    ("model.optimizer._target_", "adam_atan2_pytorch.AdamAtan2"),
    ("model.optimizer.lr", 1e-4),
    ("model.optimizer.weight_decay", 0.01),
    ("model.scheduler._target_", "other.Scheduler"),
    ("model.scheduler.num_warmup_steps", 1000),
    ("model.scheduler.num_training_steps", 2000001),
    ("model.scheduler.final_lr", 4e-6),
])
def test_mismatched_profile_fails_before_replacing_optimizer(field, value):
    config = paper_config("root")
    OmegaConf.update(config, field, value)
    original = object()
    model = SimpleNamespace(optimizer=original)
    with pytest.raises(ValueError, match=field):
        configure_paper_optimizer(model, config)
    assert model.optimizer is original


@pytest.mark.parametrize("schedule_steps,batch_size", [
    (2000001, 256), (1999999, 256), (2000000, 32), (2000000, 512),
    (2000000.0, 256), (2000000, 256.0),
])
def test_cli_arguments_reject_nonpaper_values(schedule_steps, batch_size):
    with pytest.raises(ValueError):
        validate_paper_arguments(schedule_steps, batch_size)


def test_cli_validation_does_not_import_torch():
    source = (
        "import sys; sys.modules['torch'] = None; "
        "from motionbricks.data.native_mb_paper_protocol import validate_paper_arguments; "
        "validate_paper_arguments(2000000, 256); "
        "assert sys.modules['torch'] is None"
    )
    subprocess.run([sys.executable, "-c", source], cwd=REPO_ROOT / "motionbricks", check=True)


@pytest.mark.parametrize("function", [paper_config, profile_notes])
def test_invalid_stage_is_rejected(function):
    with pytest.raises(ValueError, match="Unknown native MotionBricks stage"):
        function("other")
