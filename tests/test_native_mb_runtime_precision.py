from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import train_native_mb, train_native_mb_control


def test_control_runner_reuses_shared_precision_helpers():
    assert (train_native_mb_control.apply_native_matmul_precision
            is train_native_mb.apply_native_matmul_precision)
    assert train_native_mb_control.runtime_numerics is train_native_mb.runtime_numerics
    assert train_native_mb_control.resume_numerics_transition is train_native_mb.resume_numerics_transition


class FakeTorch:
    def __init__(self, precision="highest"):
        self.precision = precision
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False),
        )

    def set_float32_matmul_precision(self, precision):
        self.precision = precision

    def get_float32_matmul_precision(self):
        return self.precision


@pytest.mark.parametrize("runner", [train_native_mb, train_native_mb_control])
def test_runner_applies_and_records_configured_matmul_precision(runner):
    fake_torch = FakeTorch()
    config = SimpleNamespace(
        get=lambda key: "high" if key == "matmul_precision" else None,
        trainer=SimpleNamespace(precision=32),
    )

    configured = runner.apply_native_matmul_precision(config, fake_torch)
    numerics = runner.runtime_numerics(config, configured, fake_torch)

    assert configured == "high"
    assert numerics["trainer_precision"] == 32
    assert numerics["configured_matmul_precision"] == "high"
    assert numerics["effective_matmul_precision"] == "high"
    assert numerics["official_precision_setting_applied"] is True
    assert numerics["bf16"] is False
    assert "not a strict IEEE FP32 claim" in numerics["precision_policy_note"]


@pytest.mark.parametrize("runner", [train_native_mb, train_native_mb_control])
def test_runner_rejects_invalid_or_ineffective_matmul_precision(runner):
    config = {"matmul_precision": "bf16"}
    with pytest.raises(ValueError, match="valid matmul_precision"):
        runner.apply_native_matmul_precision(config, FakeTorch())

    class NonApplyingTorch(FakeTorch):
        def set_float32_matmul_precision(self, precision):
            pass

    with pytest.raises(RuntimeError, match="was not applied"):
        runner.apply_native_matmul_precision({"matmul_precision": "high"}, NonApplyingTorch())


@pytest.mark.parametrize("runner", [train_native_mb, train_native_mb_control])
def test_resume_records_unknown_previous_precision_without_inference(runner):
    current = {"effective_matmul_precision": "high"}

    transition = runner.resume_numerics_transition({}, current)

    assert transition["status"] == "prior_not_recorded_current_explicit"
    assert transition["previous_effective_matmul_precision"] == "not_recorded"
    assert transition["current_effective_matmul_precision"] == "high"
    assert transition["prior_value_inferred"] is False


def test_control_resume_requires_reason_for_runner_revisions_but_keeps_model_sources_strict(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/final.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    contract = {
        "stage": "native_root",
        "status": "completed",
        "dataset_manifest_sha256": "dataset",
        "config_sha256": "config",
        "seed": 7,
        "batch_size": 256,
        "source_lf_sha256": {
            source: train_native_mb.source_lf_sha256(REPO_ROOT / source)
            for source in train_native_mb_control.SOURCE_FILES
        },
    }
    contract["source_lf_sha256"]["scripts/train_native_mb_control.py"] = "old-control-runner"
    contract["source_lf_sha256"]["scripts/train_native_mb.py"] = "old-tokenizer-runner"
    (run / "run_contract.json").write_text(json.dumps(contract))
    (run / "training_result.json").write_text(json.dumps({
        "optimizer_updates": 50_000,
        "final_checkpoint_sha256": train_native_mb.sha256(checkpoint),
    }))
    expected = {
        "stage": "native_root",
        "dataset_manifest_sha256": "dataset",
        "config_sha256": "config",
        "seed": 7,
        "batch_size": 256,
        "tokenizer": None,
        "stop_steps": 200_000,
    }

    with pytest.raises(ValueError, match="source changed"):
        train_native_mb_control.validate_resume(run, expected, REPO_ROOT)
    binding = train_native_mb_control.validate_resume(
        run, expected, REPO_ROOT,
        runner_change_reason="Apply and record configured matmul precision",
    )
    assert [item["source"] for item in binding["runner_source_revisions"]] == [
        "scripts/train_native_mb_control.py", "scripts/train_native_mb.py",
    ]

    contract["source_lf_sha256"]["motionbricks/data/native_mb_models.py"] = "changed-model"
    (run / "run_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="source changed"):
        train_native_mb_control.validate_resume(
            run, expected, REPO_ROOT,
            runner_change_reason="Apply and record configured matmul precision",
        )


def test_full_precision_evaluation_restores_high_training_matmul_state():
    torch = pytest.importorskip("torch")
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation

    prior_precision = torch.get_float32_matmul_precision()
    prior_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision("high")
        train_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

        with full_precision_evaluation():
            assert torch.get_float32_matmul_precision() == "highest"
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False

        assert torch.get_float32_matmul_precision() == "high"
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == train_flags
    finally:
        torch.set_float32_matmul_precision(prior_precision)
        torch.backends.cudnn.allow_tf32 = prior_flags[1]
