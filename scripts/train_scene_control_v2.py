"""Independent bounded scene_control_v2 trainer; the default is zero-update preflight."""

import argparse
import importlib
import json
from pathlib import Path

from motionbricks.scene_control_v2.curriculum import Curriculum
from motionbricks.scene_control_v2.training import (
    SceneControlTrainer, TrainingBudget, preflight_training_config, run_bounded_trial,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory, never overwritten")
    parser.add_argument("--mode", choices=("preflight", "forward", "train"), default="preflight")
    parser.add_argument("--factory", help="module:function(config) -> (SceneControlTrainer, provider(request))")
    parser.add_argument("--seed", type=int, default=2333)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = preflight_training_config(config, mode=args.mode)
    if args.output.exists():
        raise ValueError(f"Refuse to overwrite output: {args.output}")
    if args.mode == "preflight":
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "preflight.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (args.output / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    else:
        if not args.factory or args.factory.count(":") != 1:
            raise ValueError("Forward/train execution requires --factory module:function")
        module, name = args.factory.split(":")
        trainer, provider = getattr(importlib.import_module(module), name)(config)
        if not isinstance(trainer, SceneControlTrainer) or not callable(provider):
            raise TypeError("Factory must return (SceneControlTrainer, callable batch provider)")
        if trainer.lineage != config["label_lineage"] or trainer.policy_id != config["policy_id"]:
            raise ValueError("Factory model identity does not match the resolved configuration")
        summary = run_bounded_trial(trainer, provider,
                                    curriculum=Curriculum.from_dict(config["curriculum"], policy_id=config["policy_id"]),
                                    budget=TrainingBudget(**config["budget"]), output=args.output,
                                    mode=args.mode, seed=args.seed)
        (args.output / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
