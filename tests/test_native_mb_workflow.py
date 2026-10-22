import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_workflow import (
    FULL_SCHEDULE_STEPS, native_stage_command, verify_workflow_run,
)


def initial():
    return {"dataset": "/data", "original_schedule_steps": FULL_SCHEDULE_STEPS,
            "dataset_manifest_sha256": "dataset-hash",
            "config_sha256": hashlib.sha256(b"config fixture\n").hexdigest(),
            "seed": 7, "batch_size": 32, "tokenizer_state_sha256": "tensor-hash"}


@pytest.mark.parametrize("stage", ["tokenizer", "root", "pose"])
def test_native_stage_command(stage):
    command = native_stage_command(stage, initial(), "/output", 5,
        resume_run="/resume" if stage != "pose" else None,
        tokenizer_run="/tokenizer" if stage == "pose" else None)
    assert command[:3] == [sys.executable, "-u", "scripts/train_native_mb.py" if stage == "tokenizer"
                          else "scripts/train_native_mb_control.py"]
    assert "--steps" in command and command[command.index("--steps") + 1] == str(FULL_SCHEDULE_STEPS)
    assert command[command.index("--schedule-steps") + 1] == str(FULL_SCHEDULE_STEPS)
    assert command[command.index("--dataset") + 1] == "/data"
    assert command[command.index("--batch-size") + 1] == "32"
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--gpu") + 1] == "5"
    assert ("--resume-run" in command) == (stage != "pose")
    assert ("--tokenizer-run" in command) == (stage == "pose")


def test_command_requires_resume_and_fresh_pose_tokenizer():
    with pytest.raises(ValueError, match="requires a resume"):
        native_stage_command("root", initial(), "/out", 0)
    with pytest.raises(ValueError, match="start fresh"):
        native_stage_command("pose", initial(), "/out", 3, resume_run="/old", tokenizer_run="/new")
    with pytest.raises(ValueError, match="completed frozen tokenizer"):
        native_stage_command("pose", initial(), "/out", 3)


def make_run(tmp_path, stage):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "native_config.yaml").write_bytes(b"config fixture\n")
    (tmp_path / "checkpoints/final.ckpt").write_bytes(b"checkpoint fixture")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    contract = {**initial(), "stage": f"native_{stage}", "status": "completed",
                "stop_steps": FULL_SCHEDULE_STEPS}
    result = {"optimizer_updates": FULL_SCHEDULE_STEPS, "updates_this_run": FULL_SCHEDULE_STEPS - 100000,
              "final_checkpoint_sha256": digest(tmp_path / "checkpoints/final.ckpt")}
    if stage != "tokenizer":
        contract.update(original_training_step=True, custom_losses=False)
        result["skipped_batches"] = 0
    if stage == "pose":
        contract["tokenizer"] = {"checkpoint_sha256": "tokenizer-checkpoint"}
        contract["tokenizer_state_sha256"] = "tensor-hash"
        result["tokenizer_state_sha256"] = "tensor-hash"
    (tmp_path / "run_contract.json").write_text(json.dumps(contract))
    (tmp_path / "training_result.json").write_text(json.dumps(result))
    return contract, result


@pytest.mark.parametrize("stage", ["tokenizer", "root", "pose"])
def test_verifies_completed_run_and_artifact_hashes(tmp_path, stage):
    contract, result = make_run(tmp_path, stage)
    found = verify_workflow_run(tmp_path, stage, initial(), 100000,
        tokenizer_checkpoint_sha256="tokenizer-checkpoint" if stage == "pose" else None)
    assert found == (contract, result)


@pytest.mark.parametrize("mutation", ["dataset_manifest_sha256", "config_sha256", "seed", "batch_size",
                                       "original_schedule_steps"])
def test_rejects_changed_identity(tmp_path, mutation):
    contract, _ = make_run(tmp_path, "root")
    contract[mutation] = "changed"
    (tmp_path / "run_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match=mutation):
        verify_workflow_run(tmp_path, "root", initial(), 100000)


def test_rejects_incomplete_updates_and_changed_files(tmp_path):
    contract, result = make_run(tmp_path, "tokenizer")
    result["updates_this_run"] -= 1
    (tmp_path / "training_result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="exact optimizer"):
        verify_workflow_run(tmp_path, "tokenizer", initial(), 100000)
    result["updates_this_run"] += 1
    (tmp_path / "training_result.json").write_text(json.dumps(result))
    (tmp_path / "native_config.yaml").write_text("altered")
    with pytest.raises(ValueError, match="config hash"):
        verify_workflow_run(tmp_path, "tokenizer", initial(), 100000)


def test_pose_requires_checkpoint_and_matching_tensor_digest(tmp_path):
    contract, _ = make_run(tmp_path, "pose")
    with pytest.raises(ValueError, match="requires the frozen tokenizer checkpoint hash"):
        verify_workflow_run(tmp_path, "pose", initial(), 100000)
    contract["tokenizer_state_sha256"] = "changed"
    (tmp_path / "run_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="frozen tokenizer"):
        verify_workflow_run(tmp_path, "pose", initial(), 100000,
                            tokenizer_checkpoint_sha256="tokenizer-checkpoint")
