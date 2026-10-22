"""Paper S7.4 optimization with the official executable SOMA30 architecture."""

from functools import partial

from motionbricks.data.native_mb_models import native_config


PAPER_STEPS = 2000000
PAPER_BATCH_PER_GPU = 256
PAPER_SOURCE = "https://arxiv.org/html/2604.24833v1#S7.SS4"

_OPTIMIZER = {"_target_": "torch.optim.Adam", "lr": 5e-5, "weight_decay": 0.0}
_SCHEDULE = {"num_warmup_steps": 10000, "num_training_steps": PAPER_STEPS, "final_lr": 2e-6}


def paper_config(stage):
    """Keep release architecture/losses; apply disclosed optimization and frame range."""
    config = native_config(stage, PAPER_STEPS)
    config.model.args.min_tokens = 3
    config.model.optimizer.update(_OPTIMIZER)
    config.model.scheduler.update(_SCHEDULE)
    return config


def profile_notes(stage, *, parc_only=False):
    """Return JSON-ready scope and authority notes without dataset size assumptions."""
    if stage not in ("tokenizer", "root", "pose"):
        raise ValueError(f"Unknown native MotionBricks stage: {stage!r}")
    return {
        "stage": stage,
        "training_profile": "paper_s7_4_optimizer_release_architecture_soma30",
        "paper_reference": PAPER_SOURCE,
        "paper_disclosed_optimizer_and_schedule": {
            "optimizer": "Adam", "lr": 5e-5,
            **_SCHEDULE, "batch_per_gpu": PAPER_BATCH_PER_GPU,
            "schedule": "linear warmup followed by cosine decay",
            "motion_fps": 30, "motion_frames_min": 12, "motion_frames_max": 64,
        },
        "architecture_authority": "official executable source and release hparams",
        "architecture_ambiguity": (
            "Paper Root width 512 with 12 heads is incompatible with the executable MHA; "
            "release uses 16 heads. Paper tokenizer 1024 channels/3 residual blocks differs "
            "from release width 512/depth 4; executable parameter count is close to the "
            "reported count. Preserve source architecture and constraints without arbitrary edits."
        ),
        "explicit_deviations": [
            "SOMA30 instead of paper27",
            ("only official/generated PARC paired motion-terrain sources rather than paper full700h"
             if parc_only else "user-approved basic flat locomotion subset rather than paper full700h"),
            "fewer GPUs AppendixA permissible, not global batch equivalence",
            "architecture/constraints not claimed exact paper reimplementation",
            "Adam betas/eps standard torch defaults not specified in paper",
            "Adam weight_decay=0 standard torch default not specified in paper",
        ],
    }


def validate_paper_arguments(schedule_steps, batch_size):
    """Reject incompatible CLI values before loading Torch or constructing a model."""
    if type(schedule_steps) is not int or schedule_steps != PAPER_STEPS:
        raise ValueError(f"Paper schedule_steps must be {PAPER_STEPS}")
    if type(batch_size) is not int or batch_size != PAPER_BATCH_PER_GPU:
        raise ValueError(f"Paper batch_size per GPU must be {PAPER_BATCH_PER_GPU}")


def configure_paper_optimizer(model, config):
    """Replace only the public optimizer constructor after validating the profile."""
    expected_sections = {
        "optimizer": _OPTIMIZER,
        "scheduler": {
            "_target_": "motionbricks.motionlib.train.scheduler.WarmupCosineScheduler",
            **_SCHEDULE,
        },
    }
    for section, expected in expected_sections.items():
        actual = config.get("model", {}).get(section, {})
        for field, value in expected.items():
            if actual.get(field) != value:
                raise ValueError(f"Paper profile requires model.{section}.{field}={value!r}")

    import torch

    model.optimizer = partial(torch.optim.Adam, lr=5e-5, weight_decay=0.0)
