"""Small command and artifact verifier for the continuous native SOMA workflow."""

import json
from pathlib import Path
import sys

from scripts.train_native_mb import sha256


FULL_SCHEDULE_STEPS = 2000001


def native_stage_command(stage, initial_contract, output, gpu, *, resume_run=None, tokenizer_run=None):
    if stage not in ("tokenizer", "root", "pose"):
        raise ValueError(f"Unsupported native stage: {stage}")
    if stage in ("tokenizer", "root") and resume_run is None:
        raise ValueError(f"{stage} requires a resume run")
    if stage == "pose" and (resume_run is not None or tokenizer_run is None):
        raise ValueError("Pose must start fresh against a completed frozen tokenizer")
    if initial_contract.get("original_schedule_steps") != FULL_SCHEDULE_STEPS:
        raise ValueError("Initial contract does not use the official full schedule")

    script = "scripts/train_native_mb.py" if stage == "tokenizer" else "scripts/train_native_mb_control.py"
    command = [sys.executable, "-u", script]
    if stage != "tokenizer":
        command += ["--stage", stage]
    command += ["--dataset", initial_contract["dataset"], "--output", str(output),
                "--steps", str(FULL_SCHEDULE_STEPS), "--schedule-steps", str(FULL_SCHEDULE_STEPS),
                "--batch-size", str(initial_contract["batch_size"]), "--seed", str(initial_contract["seed"]),
                "--workers", "2", "--gpu", str(gpu)]
    if resume_run is not None:
        command += ["--resume-run", str(resume_run)]
    if tokenizer_run is not None:
        command += ["--tokenizer-run", str(tokenizer_run)]
    return command


def verify_workflow_run(run, stage, initial_contract, previous_step, *, tokenizer_checkpoint_sha256=None,
                        target_steps=FULL_SCHEDULE_STEPS):
    run = Path(run)
    if stage not in ("tokenizer", "root", "pose"):
        raise ValueError(f"Unsupported native stage: {stage}")
    contract = json.loads((run / "run_contract.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    target = contract.get("stop_steps")
    if (contract.get("status") != "completed" or contract.get("stage") != f"native_{stage}"
            or target != target_steps or result.get("optimizer_updates") != target
            or result.get("updates_this_run") != target - previous_step):
        raise ValueError("Native stage did not complete the exact optimizer update target")
    for key in ("dataset_manifest_sha256", "config_sha256", "seed", "batch_size", "original_schedule_steps"):
        if contract.get(key) != initial_contract.get(key):
            raise ValueError(f"Native stage changed {key}")
    if sha256(run / "native_config.yaml") != contract.get("config_sha256"):
        raise ValueError("Native stage config hash mismatch")
    if sha256(run / "checkpoints" / "final.ckpt") != result.get("final_checkpoint_sha256"):
        raise ValueError("Native stage checkpoint hash mismatch")
    if stage != "tokenizer":
        if result.get("skipped_batches") != 0:
            raise ValueError("Native control stage skipped batches")
        if contract.get("original_training_step") is not True or contract.get("custom_losses") is not False:
            raise ValueError("Native control stage changed the training method")
    if stage == "pose":
        if tokenizer_checkpoint_sha256 is None:
            raise ValueError("Pose verification requires the frozen tokenizer checkpoint hash")
        binding = contract.get("tokenizer") or {}
        if (binding.get("checkpoint_sha256") != tokenizer_checkpoint_sha256
                or contract.get("tokenizer_state_sha256") != initial_contract.get("tokenizer_state_sha256")
                or result.get("tokenizer_state_sha256") != initial_contract.get("tokenizer_state_sha256")):
            raise ValueError("Pose changed or did not bind the frozen tokenizer")
    return contract, result
