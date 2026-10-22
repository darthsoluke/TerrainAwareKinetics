"""Only orchestration changes: keep native arguments and frozen dependencies."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.continue_native_mb_control import MILESTONES, native_command, verify_completed
from scripts.train_native_mb import sha256, write_json


def initial():
    return {"dataset": "/dataset", "original_schedule_steps": 2000001,
            "batch_size": 32, "seed": 7, "config_sha256": "configuration",
            "dataset_manifest_sha256": "dataset_hash", "tokenizer_state_sha256": "tensor_digest",
            "tokenizer": {"run": "/tokenizer100k", "checkpoint_sha256": "tokenizer_hash"}}


@pytest.mark.parametrize("stage", ["root", "pose"])
def test_command_retains_native_schedule_sampling_and_tokenizer(stage):
    arguments = native_command(stage, initial(), Path("/previous"), Path(f"/{stage}-220000"), 3, 220000)
    flags = dict(zip(arguments[3::2], arguments[4::2]))
    assert arguments[2] == "scripts/train_native_mb_control.py"
    assert flags["--schedule-steps"] == "2000001"
    assert flags["--steps"] == "220000"
    assert flags["--batch-size"] == "32" and flags["--seed"] == "7"
    assert Path(flags["--resume-run"]) == Path("/previous")
    assert ("--tokenizer-run" in flags) == (stage == "pose")
    if stage == "pose":
        assert flags["--tokenizer-run"] == "/tokenizer100k"
    assert "--skip-review" not in arguments
    assert MILESTONES[-1] > 200000


@pytest.mark.parametrize("stage", ["root", "pose"])
def test_verified_milestone_requires_actual_updates_and_identical_method(tmp_path, stage):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/final.ckpt").write_bytes(b"test checkpoint")
    contract = {**initial(), "stage": "native_" + stage, "status": "completed",
                "original_training_step": True, "custom_losses": False}
    result = {"optimizer_updates": 100000, "updates_this_run": 88000, "skipped_batches": 0,
              "final_checkpoint_sha256": sha256(tmp_path / "checkpoints/final.ckpt"),
              "tokenizer_state_sha256": "tensor_digest"}
    write_json(tmp_path / "run_contract.json", contract)
    write_json(tmp_path / "training_result.json", result)
    assert verify_completed(tmp_path, stage, 100000, initial(), 12000) == result
    result["updates_this_run"] = 100000
    write_json(tmp_path / "training_result.json", result)
    with pytest.raises(ValueError, match="did not complete"):
        verify_completed(tmp_path, stage, 100000, initial(), 12000)
    result["updates_this_run"] = 88000
    write_json(tmp_path / "training_result.json", result)
    contract["custom_losses"] = True
    write_json(tmp_path / "run_contract.json", contract)
    with pytest.raises(ValueError, match="training method"):
        verify_completed(tmp_path, stage, 100000, initial(), 12000)


def test_pose_cannot_change_code_space_between_milestones(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/final.ckpt").write_bytes(b"test checkpoint")
    contract = {**initial(), "stage": "native_pose", "status": "completed",
                "original_training_step": True, "custom_losses": False}
    result = {"optimizer_updates": 220000, "updates_this_run": 120000, "skipped_batches": 0,
              "final_checkpoint_sha256": sha256(tmp_path / "checkpoints/final.ckpt"),
              "tokenizer_state_sha256": "changed_tensor_digest"}
    write_json(tmp_path / "run_contract.json", contract)
    write_json(tmp_path / "training_result.json", result)
    with pytest.raises(ValueError, match="frozen tokenizer"):
        verify_completed(tmp_path, "pose", 220000, initial(), 100000)
