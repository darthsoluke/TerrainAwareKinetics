"""Exact-budget LAFAN reviews with small CPU checkpoints and model substitutes."""

from contextlib import nullcontext
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from scripts import evaluate_native_mb_lafan as evaluator


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _run(root, stage, steps, *, updates=None, status=None):
    run = root / stage
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/final.ckpt"
    torch.save({"global_step": steps, "state_dict": {}}, checkpoint)
    contract = {"stage": "native_" + stage, "config_sha256": "config-" + stage,
                "status": status or ("completed" if stage == "tokenizer" else "trained_not_reviewed")}
    result = {"optimizer_updates": steps if updates is None else updates,
              "final_checkpoint_sha256": evaluator._sha256(checkpoint)}
    _write(run / "run_contract.json", contract)
    _write(run / "training_result.json", result)
    return run


@pytest.mark.parametrize("stage", ["tokenizer", "root", "pose"])
@pytest.mark.parametrize("steps", [50_000, 2_000_000])
def test_budget_binds_each_actual_checkpoint_and_result(tmp_path, stage, steps):
    run = _run(tmp_path, stage, steps)
    if steps == 2_000_000:
        identity = evaluator.validate_stage_budget(run, stage)
    else:
        identity = evaluator.validate_stage_budget(run, stage, steps)
    assert identity["global_step"] == identity["optimizer_updates"] == steps
    assert identity["checkpoint_sha256"] == evaluator._sha256(run / "checkpoints/final.ckpt")


@pytest.mark.parametrize("stage", ["tokenizer", "root", "pose"])
@pytest.mark.parametrize("checkpoint_steps, updates", [(49_999, 50_000), (50_000, 49_999),
                                                        (50_001, 50_001), (True, 50_000)])
def test_budget_rejects_checkpoint_or_result_off_exact_budget(tmp_path, stage, checkpoint_steps, updates):
    run = _run(tmp_path, stage, checkpoint_steps, updates=updates)
    with pytest.raises(ValueError, match="both equal expected_stage_steps=50000"):
        evaluator.validate_stage_budget(run, stage, 50_000)


def test_budget_rejects_checkpoint_replacement(tmp_path):
    run = _run(tmp_path, "root", 50_000)
    torch.save({"global_step": 50_000, "state_dict": {"changed": True}}, run / "checkpoints/final.ckpt")
    with pytest.raises(ValueError, match="checkpoint hash"):
        evaluator.validate_stage_budget(run, "root", 50_000)


@pytest.mark.parametrize("status", ["running", "trained_not_reviewed"])
def test_tokenizer_must_be_completed(tmp_path, status):
    run = _run(tmp_path, "tokenizer", 50_000, status=status)
    with pytest.raises(ValueError, match="completed tokenizer"):
        evaluator.validate_stage_budget(run, "tokenizer", 50_000)


@pytest.mark.parametrize("budget", [0, -1, True, 50_000.0, "50000", None])
def test_api_rejects_nonpositive_or_noninteger_budget_before_model_import(budget):
    with pytest.raises(ValueError, match="positive integer"):
        evaluator.evaluate_lafan("source", "data", "tokenizer", "root", "pose", "out",
                                  expected_stage_steps=budget)


@pytest.mark.parametrize("change", ["checkpoint", "contract_digest", "result_digest"])
def test_pose_requires_the_reviewed_frozen_tokenizer(tmp_path, change):
    run = _run(tmp_path, "pose", 50_000)
    contract = json.loads((run / "run_contract.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    contract.update(tokenizer={"checkpoint_sha256": "tokenizer-hash"}, tokenizer_state_sha256="frozen")
    result["tokenizer_state_sha256"] = "frozen"
    _write(run / "run_contract.json", contract)
    _write(run / "training_result.json", result)
    info = {"checkpoint_sha256": "tokenizer-hash"}
    evaluator._validate_frozen_tokenizer(info, "frozen", run)
    if change == "checkpoint":
        contract["tokenizer"]["checkpoint_sha256"] = "other"
    elif change == "contract_digest":
        contract.pop("tokenizer_state_sha256")
    else:
        result["tokenizer_state_sha256"] = "other"
    _write(run / "run_contract.json", contract)
    _write(run / "training_result.json", result)
    with pytest.raises(ValueError, match="frozen tensor identity"):
        evaluator._validate_frozen_tokenizer(info, "frozen", run)


def _cli_args(tmp_path):
    return [value for flag in ("source", "dataset", "tokenizer-run", "root-run", "pose-run", "output")
            for value in ("--" + flag, str(tmp_path / flag))]


@pytest.mark.parametrize("flag, expected", [([], 2_000_000), (["--expected-stage-steps", "50000"], 50_000)])
def test_cli_preserves_default_and_passes_explicit_budget(tmp_path, monkeypatch, flag, expected):
    calls = []

    def fake_evaluate(*args, **kwargs):
        calls.append(kwargs["expected_stage_steps"])
        return {"status": "completed", "window_count": 2}

    monkeypatch.setattr(evaluator, "evaluate_lafan", fake_evaluate)
    evaluator.main([*_cli_args(tmp_path), *flag])
    assert calls == [expected]


@pytest.mark.parametrize("value", ["0", "-1", "50000.0", "true"])
def test_cli_rejects_invalid_budget_without_evaluation(tmp_path, monkeypatch, value):
    monkeypatch.setattr(evaluator, "evaluate_lafan", lambda *a, **k: pytest.fail("must reject before evaluation"))
    with pytest.raises(SystemExit) as error:
        evaluator.main([*_cli_args(tmp_path), "--expected-stage-steps", value])
    assert error.value.code == 2


@pytest.mark.parametrize("steps", [50_000, 2_000_000])
def test_evaluation_reports_verified_budget_with_cpu_model_substitutes(tmp_path, monkeypatch, steps):
    runs = {stage: _run(tmp_path, stage, steps) for stage in ("tokenizer", "root", "pose")}
    identities = {stage: evaluator.validate_stage_budget(run, stage, steps) for stage, run in runs.items()}
    pose_contract = json.loads((runs["pose"] / "run_contract.json").read_text())
    pose_result = json.loads((runs["pose"] / "training_result.json").read_text())
    dependency = {"checkpoint_sha256": identities["tokenizer"]["checkpoint_sha256"]}
    pose_contract.update(tokenizer=dependency, tokenizer_state_sha256="frozen")
    pose_result["tokenizer_state_sha256"] = "frozen"
    _write(runs["pose"] / "run_contract.json", pose_contract)
    _write(runs["pose"] / "training_result.json", pose_result)
    source, dataset, output = (tmp_path / name for name in ("source", "dataset", "review"))
    source.mkdir()
    dataset.mkdir()
    _write(source / "accepted.json", [])
    _write(dataset / "native_manifest.json", {})

    def fake_module(name, **members):
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)

    class Representation:
        fps = 30
        skeleton = SimpleNamespace(root_idx=0)

        def __init__(self):
            self.dual_rep = SimpleNamespace(global_to_local=lambda value, *a, **k: value)

        def change_first_heading(self, value, *args):
            return value

        def normalize(self, value):
            return value

        def inverse(self, value, *args, **kwargs):
            return {"posed_joints": torch.zeros((1, value.shape[1], 30, 3))}

    class PoseNet:
        decoder_external_cond_feature_mode = "root"

        def eval(self):
            return self

        def __call__(self, value, **kwargs):
            return {"recon_state": value}

    rep, net = Representation(), PoseNet()
    tokenizer = SimpleNamespace(pose_net=net, motion_rep=rep, local_motion_rep=rep)
    info = {**identities["tokenizer"], "run_status_at_review": "completed"}
    bundle = {stage + suffix: identities[stage][field] for stage in ("root", "pose")
              for suffix, field in (("_step", "global_step"), ("_checkpoint_sha256", "checkpoint_sha256"))}
    bundle["tokenizer"] = dependency
    infer = SimpleNamespace(global_motion_rep=rep, _vqvae_pose_model=net)
    score_names = ("mpjpe_m", "root_relative_fk_mpjpe_m", "root_error_m", "joint_velocity_error_mps",
                   "local_joint_rotation_geodesic_rad", "root_relative_joint_velocity_error_mps")
    fake_module("motionbricks.helper.data_training_util", extract_feature_from_motion_rep=lambda value, *a: value)
    fake_module("motionbricks.data.native_mb_evaluation", full_precision_evaluation=nullcontext)
    fake_module("scripts.evaluate_native_mb_checkpoint",
                load_native_checkpoint=lambda *a, **k: (tokenizer, info, {}),
                reconstruction_metrics=lambda *a: dict.fromkeys(score_names, 0.0))
    fake_module("scripts.evaluate_native_mb_inbetween", predict_window=lambda infer, raw: (raw, raw))
    fake_module("scripts.evaluate_native_mb_pipeline", load_bundle=lambda *a: (infer, dataset, bundle))
    fake_module("scripts.train_native_mb", sha256=evaluator._sha256)
    fake_module("scripts.train_native_mb_control", state_digest=lambda net: "frozen")
    windows = [{"behavior": behavior} for behavior in ("walk", "run")]
    monkeypatch.setattr(evaluator, "select_validation_windows", lambda *a: (windows, []))
    monkeypatch.setattr(evaluator, "_read_features", lambda *a: np.zeros((65, 7), dtype=np.float32))
    monkeypatch.setattr(evaluator, "_render_set", lambda *a: {"gifs": []})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == "cuda":
            args = ("cpu", *args[1:])
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    report = evaluator.evaluate_lafan(source, dataset, runs["tokenizer"], runs["root"], runs["pose"],
                                       output, expected_stage_steps=steps)
    candidate = json.loads((output / "candidate_bundle.json").read_text())
    for artifact in (report, candidate):
        assert artifact["expected_stage_steps"] == steps
        assert artifact["stage_steps"] == dict.fromkeys(runs, steps)
        assert all(row["global_step"] == row["optimizer_updates"] == steps
                   for row in artifact["stages"].values())
        assert artifact["quality_approved"] is False
    if steps == 2_000_000:
        assert candidate["paper_steps"] == steps
    else:
        assert "paper_steps" not in candidate
        assert any("no 2M completion claim" in line for line in candidate["limitations"])
