"""Endpoint-only inputs and untouched outputs for native in-between inference."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "out" / "vendor" / "kimodo"))

from scripts import evaluate_native_mb_inbetween as evaluator
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from test_native_mb_stage_training import _motion_and_representation


@pytest.fixture
def native_source():
    rep, batch = _motion_and_representation(frame_count=75)
    raw = rep.unnormalize(batch["motion"])
    raw[:, :, 0] = 2 + torch.arange(75) / 30
    return rep, raw


class FakeInference:
    EXTERNAL_POSE_FEATURE_MODE = "joint_positions_and_rotations"
    device = "cpu"
    _args = {"min_tokens": 6, "max_tokens": 16, "down_t": 2}

    def __init__(self, rep, prediction=None):
        self.motion_rep = self.global_motion_rep = rep
        self.local_motion_rep = rep.dual_rep.local_motion_rep
        self.prediction = prediction
        self.calls = []

    def eval(self):
        return self

    def predict(self, **inputs):
        self.calls.append(inputs)
        assert inputs["config"] == evaluator.PREDICT_CONFIG
        assert inputs["num_tokens"] is None
        assert inputs["allowed_pred_num_tokens"].tolist() == [[0] * 10 + [1]]
        for name in ("global_root_values", "local_root_values", "local_poses"):
            assert inputs["has_" + name].nonzero()[:, 1].tolist() == [3, 7]
            assert inputs[name].shape[:2] == (1, 8)
        assert inputs["local_root_values"][:, [0, 1, 2, 4, 5, 6]].count_nonzero() == 0
        assert inputs["local_poses"][:, [0, 1, 2, 4, 5, 6]].count_nonzero() == 0
        assert inputs["global_root_values"][:, [1, 2, 4, 5, 6]].count_nonzero() == 0
        assert inputs["global_root_values"][0, 0, 1] == 0
        assert set(inputs) == {"global_root_values", "local_root_values", "local_poses",
                               "has_global_root_values", "has_local_root_values", "has_local_poses",
                               "num_tokens", "allowed_pred_num_tokens", "config", "info"}
        return self.prediction.clone(), torch.tensor([16])


def test_constraints_are_exactly_two_unnormalized_keyframes(native_source):
    rep, raw = native_source
    infer = FakeInference(rep)
    before = raw.clone()
    inputs, canonical = evaluator.inbetween_constraints(infer, raw[:, :65])
    expected = rep.change_first_heading(raw[:, :65], 0.0, False, False)
    local = rep.dual_rep.global_to_local(expected, False, False, lengths=torch.tensor([65]))
    for name, values in (
        ("global_root_values", extract_feature_from_motion_rep(expected, rep, "root")),
        ("local_root_values", extract_feature_from_motion_rep(local, infer.local_motion_rep, "root")),
        ("local_poses", extract_feature_from_motion_rep(local, infer.local_motion_rep,
                                                       infer.EXTERNAL_POSE_FEATURE_MODE)),
    ):
        torch.testing.assert_close(inputs[name][:, [3, 7]], values[:, [3, 63]], rtol=0, atol=0)
        assert inputs["has_" + name].tolist() == [[False, False, False, True] * 2]
    torch.testing.assert_close(canonical, expected[:, :64], rtol=0, atol=0)
    torch.testing.assert_close(raw, before, rtol=0, atol=0)


def test_interior_body_root_and_full_body_velocity_cannot_leak(native_source):
    rep, raw = native_source
    infer = FakeInference(rep)
    original, _ = evaluator.inbetween_constraints(infer, raw[:, :65])
    changed = raw[:, :65].clone()
    # Preserve endpoint root derivative pairs 3->4 and 63->64, and frame-0 coordinates.
    changed[:, 5:63, :3] += 19
    unconditioned = [i for i in range(65) if i not in (3, 63)]
    changed[:, unconditioned, 5:] += 0.17
    changed[:, :, rep.indices["local_vel"]] += 37
    changed[:, 0, 1] += 8  # Frame-0 source height is not a condition.
    inputs, _ = evaluator.inbetween_constraints(infer, changed)
    for name, value in original.items():
        if value is None:
            assert inputs[name] is None
        else:
            torch.testing.assert_close(inputs[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["frames", "features", "batch", "nan"])
def test_constraints_reject_invalid_source(native_source, case):
    rep, raw = native_source
    raw = raw[:, :65].clone()
    if case == "frames":
        raw = raw[:, :64]
    elif case == "features":
        raw = raw[:, :, :-1]
    elif case == "batch":
        raw = raw.repeat(2, 1, 1)
    else:
        raw[0, 20, 0] = float("nan")
    with pytest.raises(ValueError, match="finite floating global source"):
        evaluator.inbetween_constraints(FakeInference(rep), raw)


@pytest.mark.parametrize("case", ["frames", "features", "nan", "tokens", "fractional_tokens", "batch_tokens"])
def test_predict_rejects_wrong_horizon_shape_and_nonfinite_values(native_source, case):
    rep, raw = native_source
    prediction, count = raw[:, :64].clone(), torch.tensor([16])
    if case == "frames":
        prediction = prediction[:, :-1]
    elif case == "features":
        prediction = prediction[:, :, :-1]
    elif case == "nan":
        prediction[0, 32, 0] = float("nan")
    elif case == "tokens":
        count = torch.tensor([15])
    elif case == "fractional_tokens":
        count = torch.tensor([16.5])
    else:
        count = torch.tensor([16, 16])
    infer = FakeInference(rep)
    infer.predict = lambda **kwargs: (prediction, count)
    with pytest.raises(ValueError, match="native prediction|exactly 16 tokens"):
        evaluator.predict_window(infer, raw[:, :65])


def _dataset(path, raw):
    path.mkdir()
    records = []
    for index in range(19):
        shard = f"shard-{index:02}.npz"
        np.savez(path / shard, global_features=raw[0].numpy())
        records.append({"shard": shard, "split": "validation", "behavior": f"behavior-{index % 5}"})
    evaluator.write_json(path / "accepted.json", records)
    files = ["accepted.json"] + [row["shard"] for row in records]
    evaluator.write_json(path / "native_manifest.json", {
        "artifact_sha256": {name: evaluator.sha256(path / name) for name in files}})
    return {"dataset_manifest_sha256": evaluator.sha256(path / "native_manifest.json")}


def test_evaluates_same_57_windows_without_root_or_endpoint_overwrite(tmp_path, native_source):
    rep, raw = native_source
    dataset, output = tmp_path / "dataset", tmp_path / "review"
    bundle = _dataset(dataset, raw)
    _, source = evaluator.inbetween_constraints(FakeInference(rep), raw[:, :65])
    prediction = source.clone()
    prediction[:, :, 0] += 2
    prediction[:, :4, 0] += 100
    prediction[:, 63, 0] += 100  # Must not affect either interior positions or velocities.
    original_prediction = prediction.clone()
    infer = FakeInference(rep, prediction)
    report = evaluator.evaluate_inbetween(infer, dataset, output, bundle)

    assert len(infer.calls) == report["windows_per_mode"] == 57
    expected_windows = evaluator._fixed_validation_windows(dataset)[1]
    assert [(r["shard"], r["start"]) for r in report["windows"]] == [
        (r["shard"], r["start"]) for r in expected_windows]
    assert report["scored_frames_inclusive"] == [4, 62]
    assert report["scored_velocity_interval_starts_inclusive"] == [4, 61]
    assert report["summary"]["world_fk_mpjpe_m"] == pytest.approx(2, abs=1e-5)
    assert report["summary"]["root_relative_fk_mpjpe_m"] == pytest.approx(0, abs=1e-5)
    assert report["summary"]["joint_velocity_error_mps"] == pytest.approx(0, abs=1e-4)
    assert report["root_condition_source"] == "root_model"
    assert "Root predicted intermediate root" in report["task"]
    assert "Root predicted intermediate root" in report["root_condition"]
    assert "Non-unique conditional generation" in report["interpretation"]
    assert not report["root_overwrite"] and report["postprocessing"] == "none"
    assert report["stage"] == "pose" and report["render_mode"] == "native_keyframe1_inbetween"
    assert report["conditions"]["local_root_velocity_source_intervals"] == [[3, 4], [63, 64]]
    assert report["conditions"]["source_interior_root_trajectory_supplied"] is False
    assert report["conditions"]["source_encoded_tokens"] is False
    expected_joints = rep.inverse(original_prediction, False, return_all=True)["posed_joints"][0].numpy()
    files = list(output.glob("*_reconstruction.npz"))
    assert len(files) == 5
    for path in files:
        with np.load(path, allow_pickle=False) as saved:
            assert saved["source"].shape == saved["prediction"].shape == (64, 30, 3)
            assert np.isfinite(saved["prediction"]).all()
            np.testing.assert_array_equal(saved["prediction"], expected_joints)
    torch.testing.assert_close(infer.prediction, original_prediction, rtol=0, atol=0)
    assert not list(output.glob("*.png")) and not list(output.glob("*.gif"))
    assert json.loads((output / "metrics.json").read_text())["root_condition_source"] == "root_model"


@pytest.mark.parametrize("case", ["manifest", "fk_shape", "fk_nan"])
def test_evaluation_rejects_bad_binding_or_fk_before_writing(tmp_path, native_source, monkeypatch, case):
    rep, raw = native_source
    dataset, output = tmp_path / "dataset", tmp_path / "review"
    bundle = _dataset(dataset, raw)
    if case == "manifest":
        bundle["dataset_manifest_sha256"] = "different"
    else:
        joints = torch.zeros((1, 64, 30, 3))
        if case == "fk_shape":
            joints = joints[:, :-1]
        else:
            joints[0, 10, 0, 0] = float("nan")
        monkeypatch.setattr(rep, "inverse", lambda *args, **kwargs: {"posed_joints": joints})
    with pytest.raises(ValueError, match="manifest differ|FK output"):
        evaluator.evaluate_inbetween(FakeInference(rep, raw[:, :64]), dataset, output, bundle)
    assert not output.exists()


def test_direct_cli_help_works_outside_project_without_pythonpath(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(Path(evaluator.__file__)), "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--root-run" in result.stdout and "--device" in result.stdout


@pytest.mark.parametrize("device", ["cpu", "cuda", "busy_cuda"])
def test_cli_checks_requested_gpu_before_loading_bundle(tmp_path, monkeypatch, device):
    calls = []

    def select_gpu(index):
        calls.append(("gpu", index))
        if device == "busy_cuda":
            raise RuntimeError("GPU is occupied")
        return {"uuid": "freshly-checked-gpu"}

    def load_bundle(root, pose, target):
        calls.append(("bundle", target))
        return object(), tmp_path, {}

    def evaluate(infer, dataset, output, bundle, seed):
        calls.append(("evaluate", bundle["gpu_at_launch"]))
        return {"summary": {"tested": True}}

    monkeypatch.setattr(evaluator, "select_gpu", select_gpu)
    monkeypatch.setitem(sys.modules, "scripts.evaluate_native_mb_pipeline", SimpleNamespace(load_bundle=load_bundle))
    monkeypatch.setattr(evaluator, "evaluate_inbetween", evaluate)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    args = ["--root-run", "root", "--pose-run", "pose", "--output", str(tmp_path / "output")]
    if device != "cpu":
        args += ["--device", "cuda", "--gpu", "1"]
    if device == "busy_cuda":
        with pytest.raises(RuntimeError, match="occupied"):
            evaluator.main(args)
        assert calls == [("gpu", 1)]
    else:
        evaluator.main(args)
        assert calls == ([("bundle", "cpu"), ("evaluate", None)] if device == "cpu" else
                         [("gpu", 1), ("bundle", "cuda"), ("evaluate", {"uuid": "freshly-checked-gpu"})])
