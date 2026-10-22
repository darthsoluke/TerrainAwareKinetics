"""Focused contracts for native Root/Pose training and resume handling."""

import json
from pathlib import Path
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from scripts import train_native_mb_control as runner
from test_native_mb_stage_training import (
    _attach_minimal_training_context,
    _motion_and_representation,
    _set_test_sampler_horizon,
)


@pytest.fixture(scope="module", autouse=True)
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "metadata,dataset_hash",
    [
        ({}, "dataset"),
        ({"run_status_at_review": "running", "dataset_manifest_sha256": "dataset",
          "checkpoint_sha256": "ckpt", "global_step": 10}, "dataset"),
        ({"run_status_at_review": "completed", "dataset_manifest_sha256": "other",
          "checkpoint_sha256": "ckpt", "global_step": 10}, "dataset"),
        ({"run_status_at_review": "completed", "dataset_manifest_sha256": "dataset",
          "global_step": 10}, "dataset"),
        ({"run_status_at_review": "completed", "dataset_manifest_sha256": "dataset",
          "checkpoint_sha256": "ckpt", "global_step": 0}, "dataset"),
    ],
)
def test_tokenizer_binding_rejects_incomplete_or_mismatched_identity(metadata, dataset_hash):
    with pytest.raises(ValueError):
        runner.verify_tokenizer_binding(metadata, dataset_hash)


def test_tokenizer_binding_accepts_completed_same_dataset_identity():
    metadata = {
        "run_status_at_review": "completed",
        "dataset_manifest_sha256": "dataset",
        "checkpoint_sha256": "checkpoint-sha",
        "global_step": 45_000,
    }
    assert runner.verify_tokenizer_binding(metadata, "dataset") is None


def _resume_fixture(tmp_path, code_root):
    run = tmp_path / "resume-run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "final.ckpt"
    checkpoint.write_bytes(b"native-lightning-checkpoint")
    contract = {
        "stage": "native_pose",
        "status": "completed",
        "dataset_manifest_sha256": "dataset-sha",
        "config_sha256": "config-sha",
        "seed": 7,
        "batch_size": 4,
        "tokenizer": {"checkpoint_sha256": "tokenizer-sha"},
        "source_lf_sha256": {
            name: runner.source_lf_sha256(code_root / name) for name in runner.SOURCE_FILES
        },
    }
    result = {
        "optimizer_updates": 12,
        "final_checkpoint_sha256": runner.sha256(checkpoint),
    }
    (run / "run_contract.json").write_text(json.dumps(contract))
    (run / "training_result.json").write_text(json.dumps(result))
    expected = {
        "stage": "native_pose",
        "dataset_manifest_sha256": "dataset-sha",
        "config_sha256": "config-sha",
        "seed": 7,
        "batch_size": 4,
        "tokenizer": {"checkpoint_sha256": "tokenizer-sha"},
        "stop_steps": 20,
    }
    return run, checkpoint, contract, result, expected


@pytest.mark.parametrize(
    "mismatch,expected_value,error",
    [
        ("stage", "native_root", "stage"),
        ("config_sha256", "changed-config", "config_sha256"),
        ("dataset_manifest_sha256", "changed-dataset", "dataset_manifest_sha256"),
        ("tokenizer", {"checkpoint_sha256": "changed-tokenizer"}, "tokenizer"),
        ("source", None, "source changed"),
        ("checkpoint", None, "checkpoint hash"),
        ("update", None, "update count"),
    ],
)
def test_resume_rejects_contract_source_checkpoint_and_update_mismatches(
    tmp_path, mismatch, expected_value, error
):
    run, checkpoint, contract, result, expected = _resume_fixture(tmp_path, REPO_ROOT)
    if mismatch == "source":
        contract["source_lf_sha256"][runner.SOURCE_FILES[0]] = "stale-source"
        (run / "run_contract.json").write_text(json.dumps(contract))
    elif mismatch == "checkpoint":
        checkpoint.write_bytes(b"tampered-checkpoint")
    elif mismatch == "update":
        expected["stop_steps"] = result["optimizer_updates"]
    else:
        expected[mismatch] = expected_value

    with pytest.raises(ValueError, match=error):
        runner.validate_resume(run, expected, REPO_ROOT)


@pytest.mark.parametrize("status", ["completed", "trained_not_reviewed"])
def test_resume_accepts_stable_binding_and_returns_saved_update(tmp_path, status):
    run, _, contract, _, expected = _resume_fixture(tmp_path, REPO_ROOT)
    contract["status"] = status
    (run / "run_contract.json").write_text(json.dumps(contract))

    binding = runner.validate_resume(run, expected, REPO_ROOT)
    assert binding["run"] == str(run.resolve())
    assert binding["saved_step"] == 12
    assert Path(binding["checkpoint"]).is_file()


def test_explicit_entrypoint_revision_does_not_permit_model_source_change(tmp_path):
    run, _, contract, _, expected = _resume_fixture(tmp_path, REPO_ROOT)
    contract["source_lf_sha256"]["scripts/train_native_mb_control.py"] = "old-runner"
    (run / "run_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="source changed"):
        runner.validate_resume(run, expected, REPO_ROOT)
    binding = runner.validate_resume(run, expected, REPO_ROOT,
                                     runner_change_reason="Apply and record configured matmul precision")
    assert binding["entrypoint_revision"]["previous_sha256"] == "old-runner"
    assert binding["runner_source_revisions"] == [binding["entrypoint_revision"]]
    contract["source_lf_sha256"]["scripts/train_native_mb.py"] = "old-tokenizer-runner"
    (run / "run_contract.json").write_text(json.dumps(contract))
    binding = runner.validate_resume(run, expected, REPO_ROOT,
                                     runner_change_reason="Apply and record configured matmul precision")
    assert [item["source"] for item in binding["runner_source_revisions"]] == [
        "scripts/train_native_mb_control.py", "scripts/train_native_mb.py",
    ]
    contract["source_lf_sha256"]["motionbricks/data/native_mb_models.py"] = "changed-model"
    (run / "run_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="source changed"):
        runner.validate_resume(run, expected, REPO_ROOT,
                               runner_change_reason="Apply and record configured matmul precision")


def test_state_digest_is_stable_across_devices_when_cuda_is_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU-only environment cannot exercise cross-device tensors")
    module = torch.nn.Linear(3, 2)
    cpu_digest = runner.state_digest(module)
    assert runner.state_digest(module.to("cuda")) == cpu_digest


def test_state_digest_changes_when_tensor_values_change():
    module = torch.nn.Linear(3, 2)
    before = runner.state_digest(module)
    with torch.no_grad():
        module.weight[0, 0].add_(1.0)
    assert runner.state_digest(module) != before


def test_native_pose_keeps_supporting_tokenizer_frozen_across_two_updates():
    from motionbricks.data.native_mb_models import build_native_model, native_config

    torch.manual_seed(2701)
    representation, batch = _motion_and_representation()
    tokenizer = build_native_model(
        "tokenizer", representation, native_config("tokenizer", max_steps=2_000_001)
    )
    model = build_native_model(
        "pose", representation, native_config("pose", max_steps=2_000_001), tokenizer=tokenizer
    )
    _set_test_sampler_horizon(model)
    tokenizer_net = model.supporting_nets["pose_net"]
    initial_digest = runner.state_digest(tokenizer_net)
    optimizer = torch.optim.SGD(model.backbone_net.parameters(), lr=1e-4)

    for step in range(2):
        model.train()
        _attach_minimal_training_context(model)
        model._trainer.global_step = 100_000 + step
        step_batch = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_step(step_batch, step)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in model.backbone_net.parameters()
        )
        optimizer.step()

        assert tokenizer_net.training is False
        assert all(not parameter.requires_grad and parameter.grad is None
                   for parameter in tokenizer_net.parameters())
        assert runner.state_digest(tokenizer_net) == initial_digest

    del model, tokenizer
