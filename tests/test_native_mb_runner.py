"""Execution contracts do not change native training schedules or claim quality."""

import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "native_runner", Path(__file__).parents[1] / "scripts/train_native_mb.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_gpu_uuid_selected_before_project_import(monkeypatch):
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k:
                        "0, GPU-idle, 2, 0\n1, GPU-busy, 12000, 82\n")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    result = runner.select_gpu(0)
    assert runner.os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-idle"
    assert result["utilization_percent"] == 0


@pytest.mark.parametrize("status", ["12000, 0", "2, 50"])
def test_busy_gpu_is_never_used(monkeypatch, status):
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: f"0, GPU-busy, {status}\n")
    with pytest.raises(RuntimeError, match="occupied"):
        runner.select_gpu(0)


def test_strict_json_rejects_nonfinite_training_result(tmp_path):
    with pytest.raises(ValueError):
        runner.write_json(tmp_path / "result.json", {"loss": float("nan")})
    runner.write_json(tmp_path / "result.json", {"loss": 1.0, "quality_approved": False})
    assert json.loads((tmp_path / "result.json").read_text())["quality_approved"] is False


def test_dataset_verification_rejects_missing_and_changed_artifacts(tmp_path):
    with pytest.raises(ValueError, match="normalization"):
        runner.verify_dataset(tmp_path, {})
    runner.write_json(tmp_path / "accepted.json", [{"shard": "shard.npz"}])
    (tmp_path / "stats").mkdir()
    for name in ("stats/mean.npy", "stats/std.npy", "shard.npz"):
        (tmp_path / name).write_bytes(b"fixture")
    manifest = {"artifact_sha256": {name: runner.sha256(tmp_path / name) for name in
                ("accepted.json", "stats/mean.npy", "stats/std.npy", "shard.npz")}}
    runner.verify_dataset(tmp_path, manifest)
    (tmp_path / "shard.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact mismatch"):
        runner.verify_dataset(tmp_path, manifest)


def test_resume_accepts_only_matching_native_contract_and_complete_checkpoint(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    path = tmp_path / "checkpoints/final.ckpt"
    path.write_bytes(b"checkpoint fixture")
    contract = {"stage": "native_tokenizer", "status": "completed", "seed": 7, "batch_size": 32,
                "config_sha256": "config", "dataset_manifest_sha256": "manifest"}
    runner.write_json(tmp_path / "run_contract.json", contract)
    runner.write_json(tmp_path / "training_result.json", {
        "optimizer_updates": 10, "final_checkpoint_sha256": runner.sha256(path)})
    result = runner.validate_native_resume(tmp_path, "manifest", "config", 7, 32, 20)
    assert result["saved_step"] == 10
    with pytest.raises(ValueError, match="total update target"):
        runner.validate_native_resume(tmp_path, "manifest", "config", 7, 32, 10)
    with pytest.raises(ValueError, match="mismatch"):
        runner.validate_native_resume(tmp_path, "changed", "config", 7, 32, 20)
    contract["stage"] = "tokenizer"
    runner.write_json(tmp_path / "run_contract.json", contract)
    with pytest.raises(ValueError, match="not legacy"):
        runner.validate_native_resume(tmp_path, "manifest", "config", 7, 32, 20)


def test_resume_source_guard_allows_only_line_ending_changes(tmp_path):
    names = ("motionbricks/data/native_mb_data.py", "motionbricks/data/native_mb_models.py",
             "motionbricks/vqvae/models/motion_vqvae.py", "motionbricks/vqvae/neural_modules/vqvae.py",
             "motionbricks/helper/data_training_util.py")
    recorded = {}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"VALUE = 1\n")
        recorded[name] = runner.sha256(path)
    previous = {"source_sha256": recorded}
    path.write_bytes(b"VALUE = 1\r\n")
    result = runner.verify_resume_training_sources(previous, tmp_path)
    assert result[names[-1]]["line_endings_only"]
    path.write_bytes(b"VALUE = 2\n")
    with pytest.raises(ValueError, match="training source changed"):
        runner.verify_resume_training_sources(previous, tmp_path)
    with pytest.raises(ValueError, match="lacks training source"):
        runner.verify_resume_training_sources({}, tmp_path)
