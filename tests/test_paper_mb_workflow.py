import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_mb_workflow import (
    capture_stage_identity, paper_command, require_technical_checks, sha256,
)
from motionbricks.data.native_mb_paper_protocol import profile_notes


@pytest.mark.parametrize("stage", ["tokenizer", "root", "pose"])
def test_paper_commands_start_fresh_with_full_budget_and_exact_optimization(stage):
    command = paper_command(stage, "/data", "/run", 3,
                            tokenizer_run="/final-tokenizer" if stage == "pose" else None)
    for flag, expected in (("--steps", "2000000"), ("--schedule-steps", "2000000"),
                           ("--batch-size", "256"), ("--training-profile", "paper")):
        assert command[command.index(flag) + 1] == expected
    assert "--resume-run" not in command and "--skip-review" not in command
    assert ("--tokenizer-run" in command) == (stage == "pose")


def test_no_accidental_tokenizer_binding_for_root():
    with pytest.raises(ValueError, match="Only Pose"):
        paper_command("root", "/data", "/run", 0, tokenizer_run="/wrong")
    with pytest.raises(ValueError, match="Only Pose"):
        paper_command("pose", "/data", "/run", 0)


def make_contract(tmp_path, stage="pose"):
    contract = {"stage": "native_" + stage, "dataset": str(Path("/data")), "dataset_manifest_sha256": "dataset-hash",
                "batch_size": 256, "seed": 7, "stop_steps": 2000000, "original_schedule_steps": 2000000,
                "training_profile": profile_notes(stage), "tokenizer": {"checkpoint_sha256": "frozen-hash"},
                "config_sha256": "config-hash", "tokenizer_state_sha256": "frozen-tensors"}
    (tmp_path / "run_contract.json").write_text(json.dumps(contract))
    return contract


def test_running_contract_cannot_silently_use_preview_optimizer(tmp_path):
    assert capture_stage_identity(tmp_path, "pose", Path("/data"), "dataset-hash") is None
    contract = make_contract(tmp_path)
    assert capture_stage_identity(tmp_path, "pose", Path("/data"), "dataset-hash", "frozen-hash")[
        "config_sha256"] == "config-hash"
    contract["training_profile"] = {"name": "release"}
    (tmp_path / "run_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="approved paper profile"):
        capture_stage_identity(tmp_path, "pose", Path("/data"), "dataset-hash", "frozen-hash")


def test_running_pose_cannot_rebind_to_old_codebook(tmp_path):
    make_contract(tmp_path)
    with pytest.raises(ValueError, match="different tokenizer"):
        capture_stage_identity(tmp_path, "pose", Path("/data"), "dataset-hash", "old-tokenizer-hash")


def test_failed_memory_probe_does_not_start_full_training(tmp_path):
    path = tmp_path / "checks.json"
    assert require_technical_checks(path) is False
    path.write_text(json.dumps({"status": "failed", "error": "OOM at batch256"}))
    with pytest.raises(RuntimeError, match="OOM at batch256"):
        require_technical_checks(path)


def make_technical_checks(tmp_path):
    stages = {}
    for stage in ("tokenizer", "root", "pose"):
        run = tmp_path / stage
        (run / "checkpoints").mkdir(parents=True)
        checkpoint = run / "checkpoints/final.ckpt"
        checkpoint.write_bytes(b"technical-only")
        contract = {"status": "completed" if stage == "tokenizer" else "trained_not_reviewed",
                    "training_profile": profile_notes(stage), "batch_size": 256,
                    "sampling_contract": {"input_batch_size": 256}}
        result = {"optimizer_updates": 200, "updates_this_run": 200,
                  "optimizer_class": "torch.optim.adam.Adam", "skipped_batches": 0,
                  "observed_input_batch_sizes": [256]}
        (run / "run_contract.json").write_text(json.dumps(contract))
        (run / "training_result.json").write_text(json.dumps(result))
        stages[stage] = {"run": str(run), "checkpoint_sha256": sha256(checkpoint)}
    path = tmp_path / "checks.json"
    path.write_text(json.dumps({"status": "completed", "stages": stages}))
    return path


def test_real_training_checks_do_not_require_bones_specific_quality_review(tmp_path):
    assert require_technical_checks(make_technical_checks(tmp_path)) is True


@pytest.mark.parametrize("field,value", [
    ("observed_input_batch_sizes", [10]), ("observed_input_batch_sizes", [256, 10]),
    ("updates_this_run", 199), ("skipped_batches", 1), ("skipped_batches", None),
])
def test_technical_checks_reject_nominal_batches_or_skipped_updates(tmp_path, field, value):
    path = make_technical_checks(tmp_path)
    result_path = tmp_path / "pose/training_result.json"
    result = json.loads(result_path.read_text())
    result[field] = value
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="identity mismatch"):
        require_technical_checks(path)
