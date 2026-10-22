import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "out" / "vendor" / "kimodo"))

from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from scripts import evaluate_native_mb_checkpoint as checkpoint_review
from scripts.evaluate_native_mb_checkpoint import (
    compare_baseline,
    compare_physical_baseline,
    load_native_checkpoint,
    reconstruction_metrics,
    root_relative_joints,
    validate_run_contract,
)


def _identity_rotations(batch=1, frames=64, joints=3):
    return torch.eye(3).expand(batch, frames, joints, 3, 3).clone()


def _rotate_z(degrees, batch=1, frames=64, joints=3):
    angle = torch.tensor(degrees * torch.pi / 180)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.tensor([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rotation.expand(batch, frames, joints, 3, 3).clone()


def test_root_translation_is_removed_without_rotation_alignment():
    source = torch.randn(1, 64, 3, 3)
    moved = source + torch.tensor([3.0, -2.0, 5.0]).view(1, 1, 1, 3)
    torch.testing.assert_close(root_relative_joints(source, 0), root_relative_joints(moved, 0))

    relative = torch.zeros(1, 64, 3, 3)
    relative[..., 1, 0] = 1.0
    angle = torch.pi / 2
    rotated = relative.clone()
    rotated[..., 1, 0] = torch.cos(torch.tensor(angle))
    rotated[..., 1, 1] = torch.sin(torch.tensor(angle))
    assert not torch.allclose(root_relative_joints(relative, 0),
                              root_relative_joints(rotated, 0))


def test_reconstruction_metrics_known_rotation_and_root_translation():
    joints = torch.randn(1, 64, 3, 3)
    root_delta = torch.linspace(0, 4, 64).view(1, 64, 1, 1)
    moved_joints = joints + root_delta * torch.tensor([1.0, 0.0, 0.0])
    target = {"posed_joints": joints, "local_joint_rots": _identity_rotations()}
    prediction = {"posed_joints": moved_joints,
                  "local_joint_rots": _rotate_z(90)}

    metrics = reconstruction_metrics(prediction, target, root_index=0, fps=30.0)

    expected_world_error = float(root_delta[:, 4:64].mean())
    expected_world_velocity = float(torch.diff(root_delta[:, 4:64, 0, 0], dim=1)
                                     .mul(30.0).abs().mean())
    assert metrics["mpjpe_m"] == pytest.approx(expected_world_error)
    assert metrics["root_error_m"] == pytest.approx(expected_world_error)
    assert metrics["joint_velocity_error_mps"] == pytest.approx(expected_world_velocity)
    assert metrics["root_relative_fk_mpjpe_m"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["local_joint_rotation_geodesic_rad"] == pytest.approx(torch.pi / 2, abs=1e-6)
    assert metrics["root_relative_joint_velocity_error_mps"] == pytest.approx(0.0, abs=1e-5)


def test_reconstruction_velocity_uses_only_inclusive_scored_interval():
    target_joints = torch.zeros(1, 64, 3, 3)
    predicted_joints = target_joints.clone()
    predicted_joints[:, :4, 1, 0] = 100.0
    predicted_joints[:, 4:64, 1, 0] = torch.arange(60, dtype=torch.float32)
    target = {"posed_joints": target_joints, "local_joint_rots": _identity_rotations()}
    prediction = {"posed_joints": predicted_joints,
                  "local_joint_rots": _identity_rotations()}

    result = reconstruction_metrics(prediction, target, root_index=0, fps=30.0)

    assert result["root_relative_joint_velocity_error_mps"] == pytest.approx(10.0)


@pytest.mark.parametrize("which", ["posed_joints", "local_joint_rots"])
def test_reconstruction_rejects_nonfinite_tensors(which):
    target = {"posed_joints": torch.zeros(1, 64, 3, 3),
              "local_joint_rots": _identity_rotations()}
    prediction = {"posed_joints": torch.zeros(1, 64, 3, 3),
                  "local_joint_rots": _identity_rotations()}
    prediction[which].reshape(-1)[0] = float("nan")

    with pytest.raises(ValueError, match="Nonfinite"):
        reconstruction_metrics(prediction, target, root_index=0, fps=30.0)


def test_baseline_comparison_refuses_window_and_manifest_mismatches(tmp_path):
    metrics = {
        "mpjpe_m": 0.1,
        "root_error_m": 0.2,
        "joint_velocity_error_mps": 0.3,
        "root_relative_fk_mpjpe_m": 0.4,
        "local_joint_rotation_geodesic_rad": 0.5,
        "root_relative_joint_velocity_error_mps": 0.6,
        "normalized_smooth_l1": 0.7,
    }
    windows = [
        {"shard": "shards/a.npz", "shard_sha256": "shard-hash", "behavior": "walk",
         "start": 0, "conditioning": condition, **metrics}
        for condition in ("none", "context4")
    ]
    current = {"dataset_manifest_sha256": "manifest", "input_signature": "signature",
               "windows": windows}
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(current), encoding="utf-8")
    comparison = compare_baseline(current, baseline_path)
    assert set(comparison["deltas"]["none"]) == set(metrics)

    mismatched_window = json.loads(json.dumps(current))
    mismatched_window["windows"][0]["start"] = 1
    baseline_path.write_text(json.dumps(mismatched_window), encoding="utf-8")
    with pytest.raises(ValueError, match="windows mismatch"):
        compare_baseline(current, baseline_path)

    mismatched_manifest = json.loads(json.dumps(current))
    mismatched_manifest["dataset_manifest_sha256"] = "other-manifest"
    baseline_path.write_text(json.dumps(mismatched_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_manifest_sha256"):
        compare_baseline(current, baseline_path)


@pytest.mark.parametrize("field", ["dataset_manifest_sha256", "input_signature"])
@pytest.mark.parametrize("side", ["current", "baseline"])
def test_baseline_comparison_rejects_missing_identity(tmp_path, field, side):
    current = {"dataset_manifest_sha256": "manifest", "input_signature": "signature",
               "windows": []}
    baseline = dict(current)
    (current if side == "current" else baseline)[field] = None
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ValueError, match=f"{field} must be a nonempty string"):
        compare_baseline(current, path)


def test_baseline_comparison_rejects_nonfinite_metric(tmp_path):
    row = {"shard": "s.npz", "shard_sha256": "shard-hash", "behavior": "walk",
           "start": 0, "conditioning": "none", "mpjpe_m": float("nan")}
    current = {"dataset_manifest_sha256": "manifest", "input_signature": "signature",
               "windows": [row]}
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(current), encoding="utf-8")

    with pytest.raises(ValueError, match="nonfinite mpjpe_m"):
        compare_baseline(current, path)


def test_array_identity_handles_cpu_tensor_with_grad():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    identity = checkpoint_review._array_identity(tensor)
    expected = checkpoint_review._array_identity(tensor.detach().numpy())
    assert identity == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_array_identity_handles_cuda_tensor_with_grad():
    tensor = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3).requires_grad_()
    identity = checkpoint_review._array_identity(tensor)
    expected = checkpoint_review._array_identity(tensor.detach().cpu().numpy())
    assert identity == expected


def _physical_report():
    skeleton = {
        "joint_names": ["root", "hip"],
        "joint_parents": {"shape": [2], "dtype": "<i8", "sha256": "a" * 64},
        "neutral_geometry": {"shape": [2, 3], "dtype": "<f4", "sha256": "b" * 64},
        "root_index": 0,
    }
    rows = []
    for condition in CONDITIONS_FOR_TEST:
        rows.append({
            "shard": "heldout/clip.npz",
            "start": 4,
            "behavior": "walk",
            "conditioning": condition,
            "source_motion_sha256": "c" * 64,
            "source_motion_shape": [65, 9],
            "source_motion_dtype": "<f4",
            **{metric: 0.1 for metric in (
                "mpjpe_m", "root_error_m", "joint_velocity_error_mps",
                "root_relative_fk_mpjpe_m", "local_joint_rotation_geodesic_rad",
                "root_relative_joint_velocity_error_mps", "normalized_smooth_l1",
            )},
        })
    report = {
        "dataset_split": "train",
        "dataset_manifest_sha256": "train-manifest-v1",
        "input_signature": "train-input-signature-v1",
        "fps": 30.0,
        "frame_scoring_inclusive": [4, 63],
        "interval_scoring_endpoint_frames_inclusive": [4, 63],
        "conditions": list(CONDITIONS_FOR_TEST),
        "source_skeleton_identity": skeleton,
        "windows": rows,
    }
    report["source_motion_signature"] = checkpoint_review._source_motion_signature(
        skeleton, report["fps"], rows
    )
    return report


CONDITIONS_FOR_TEST = ("none", "context4")


def test_physical_baseline_allows_training_manifest_and_statistics_changes(tmp_path):
    current = _physical_report()
    baseline = _physical_report()
    baseline["dataset_manifest_sha256"] = "train-manifest-v2"
    baseline["input_signature"] = "different-training-statistics-bound-signature"
    baseline_path = tmp_path / "physical-baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    comparison = compare_physical_baseline(current, baseline_path)

    assert comparison["comparison_scope"] == "physical_metrics_only"
    assert set(comparison["deltas"]["none"]) == set(checkpoint_review.PHYSICAL_METRICS)
    assert comparison["excluded_metrics"] == {
        "normalized_smooth_l1": "excluded because normalization statistics can differ across training datasets",
    }


@pytest.mark.parametrize("change", ["raw_features", "skeleton", "fps", "window"])
def test_physical_baseline_refuses_changed_source_identity(tmp_path, change):
    current = _physical_report()
    baseline = _physical_report()
    if change == "raw_features":
        baseline["windows"][0]["source_motion_sha256"] = "d" * 64
    elif change == "skeleton":
        baseline["source_skeleton_identity"]["joint_names"][1] = "different_joint"
    elif change == "fps":
        baseline["fps"] = 60.0
    else:
        baseline["windows"][0]["start"] += 1
    baseline["source_motion_signature"] = checkpoint_review._source_motion_signature(
        baseline["source_skeleton_identity"], baseline["fps"], baseline["windows"]
    )
    path = tmp_path / "physical-baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ValueError, match="source_motion_signature mismatch|ordered source motion windows mismatch"):
        compare_physical_baseline(current, path)


@pytest.mark.parametrize("side", ["current", "baseline"])
def test_physical_baseline_requires_new_source_signature(tmp_path, side):
    current = _physical_report()
    baseline = _physical_report()
    (current if side == "current" else baseline).pop("source_motion_signature")
    path = tmp_path / "physical-baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ValueError, match="source_motion_signature is missing; reevaluate report"):
        compare_physical_baseline(current, path)


@pytest.mark.parametrize("field,value", [
    ("frame_scoring_inclusive", [5, 63]),
    ("interval_scoring_endpoint_frames_inclusive", [4, 62]),
    ("conditions", ["none"]),
])
def test_physical_baseline_rejects_tampered_protocol_metadata(tmp_path, field, value):
    current = _physical_report()
    baseline = _physical_report()
    baseline[field] = value
    path = tmp_path / "physical-baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ValueError, match="source signature metadata is missing or malformed"):
        compare_physical_baseline(current, path)


def test_checkpoint_loader_rejects_bad_run_contract(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_contract.json").write_text(json.dumps({
        "stage": "legacy_model",
        "status": "running",
        "dataset": str(tmp_path / "dataset"),
    }), encoding="utf-8")
    (run / "native_config.yaml").write_text("model: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stage must be native_tokenizer"):
        load_native_checkpoint(run, checkpoint=Path("intermediate.ckpt"), device="cpu")


def test_checkpoint_loader_accepts_running_contract_without_final_checkpoint(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "stats").mkdir(parents=True)
    artifacts = {
        "accepted.json": json.dumps([{"shard": "shards/train.npz", "split": "train"}]).encode(),
        "stats/mean.npy": b"mean",
        "stats/std.npy": b"std",
        "shards/train.npz": b"shard",
    }
    for relative, contents in artifacts.items():
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    manifest = {"artifact_sha256": {
        relative: hashlib.sha256(contents).hexdigest()
        for relative, contents in artifacts.items()
    }}
    manifest_path = dataset / "native_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_bytes = b"model: {}\n"
    run = tmp_path / "running-run"
    run.mkdir()
    (run / "native_config.yaml").write_bytes(config_bytes)
    (run / "run_contract.json").write_text(json.dumps({
        "stage": "native_tokenizer",
        "status": "running",
        "dataset": str(dataset),
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="intermediate.ckpt"):
        load_native_checkpoint(run, checkpoint=Path("intermediate.ckpt"), device="cpu")


def _bound_run(tmp_path, *, stop_steps=10, resume_saved_step=0):
    dataset = tmp_path / "dataset"
    (dataset / "stats").mkdir(parents=True)
    artifacts = {
        "accepted.json": json.dumps([{"shard": "shards/train.npz", "split": "train"}]).encode(),
        "stats/mean.npy": b"mean",
        "stats/std.npy": b"std",
        "shards/train.npz": b"shard",
    }
    for relative, contents in artifacts.items():
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    manifest = {"artifact_sha256": {
        relative: hashlib.sha256(contents).hexdigest()
        for relative, contents in artifacts.items()
    }}
    manifest_path = dataset / "native_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_bytes = b"model: {}\n"
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "native_config.yaml").write_bytes(config_bytes)
    (run / "run_contract.json").write_text(json.dumps({
        "stage": "native_tokenizer",
        "status": "running",
        "dataset": str(dataset),
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "stop_steps": stop_steps,
        "resume": {"saved_step": resume_saved_step},
    }), encoding="utf-8")
    return run


def test_run_contract_requires_every_accepted_shard_binding(tmp_path):
    run = _bound_run(tmp_path)
    manifest_path = tmp_path / "dataset" / "native_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifact_sha256"]["shards/train.npz"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    contract_path = run / "run_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["dataset_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="every accepted shard"):
        validate_run_contract(run)


@pytest.mark.parametrize("location,step,resume_step,expected", [
    ("run", 5, 0, "inside run/checkpoints"),
    ("checkpoints", 11, 0, "outside the run bounds"),
    ("checkpoints", 4, 5, "outside the run bounds"),
])
def test_intermediate_checkpoint_is_path_and_step_bounded(
        tmp_path, location, step, resume_step, expected):
    run = _bound_run(tmp_path, stop_steps=10, resume_saved_step=resume_step)
    directory = run if location == "run" else run / "checkpoints"
    checkpoint = directory / "step.ckpt"
    torch.save({"global_step": step, "state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match=expected):
        load_native_checkpoint(run, checkpoint=checkpoint, device="cpu")


def test_final_checkpoint_must_match_training_result_hash(tmp_path):
    run = _bound_run(tmp_path)
    checkpoint = run / "checkpoints" / "final.ckpt"
    torch.save({"global_step": 10, "state_dict": {}}, checkpoint)
    (run / "training_result.json").write_text(json.dumps({
        "final_checkpoint_sha256": "not-the-checkpoint-hash",
        "optimizer_updates": 10,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match training_result.json"):
        load_native_checkpoint(run, checkpoint=checkpoint, device="cpu")


def _tf32_flags():
    return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


@pytest.mark.parametrize("initial_flags", [(True, True), (False, True),
                                          (True, False), (False, False)])
@pytest.mark.parametrize("raises", [False, True])
def test_full_precision_context_restores_backend_flags(monkeypatch, initial_flags, raises):
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if initial_flags[0] else "highest")
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", initial_flags[1])

    def evaluate():
        with full_precision_evaluation():
            assert _tf32_flags() == (False, False)
            if raises:
                raise RuntimeError("evaluation failed")
            return "completed"

    try:
        if raises:
            with pytest.raises(RuntimeError, match="evaluation failed"):
                evaluate()
        else:
            assert evaluate() == "completed"
        assert _tf32_flags() == initial_flags
    finally:
        torch.set_float32_matmul_precision(previous_precision)


@pytest.fixture
def precision_evaluation_fixture(tmp_path, monkeypatch):
    """Small deterministic adapter to exercise the real evaluator's parity guard."""
    from motionbricks.helper import data_training_util

    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    dataset = tmp_path / "evaluation_dataset"
    dataset.mkdir()
    np.savez(dataset / "clip.npz", global_features=np.zeros((65, 9), dtype=np.float32))
    window = {"shard": "clip.npz", "shard_sha256": "fixture-shard-hash",
              "behavior": "idle", "start": 0}
    monkeypatch.setattr(checkpoint_review, "_fixed_windows", lambda _, split: ([window], []))
    monkeypatch.setattr(data_training_util, "extract_feature_from_motion_rep",
                        lambda local, rep, mode: local[..., :2])

    class Representation:
        fps = 30.0
        skeleton = SimpleNamespace(
            root_idx=0,
            bone_order_names=["root", "joint1", "joint2"],
            joint_parents=np.array([-1, 0, 1], dtype=np.int64),
            neutral_joints=np.zeros((3, 3), dtype=np.float32),
        )

        def change_first_heading(self, raw, *args):
            return raw

        def normalize(self, raw):
            return raw

        def global_to_local(self, motion, *args, **kwargs):
            return motion

        def inverse(self, state, *args, **kwargs):
            return {"posed_joints": state.reshape(1, 64, 3, 3),
                    "local_joint_rots": _identity_rotations()}

    class Tokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.quantizer = SimpleNamespace(num_heads=1, nb_code_per_head=2)
            self.decoder_external_cond_feature_mode = "fixture"
            self.context_decode_error = 0.0
            self.conditions = []
            self.condition_masks = []

        def encode_into_idx(self, local, **kwargs):
            assert _tf32_flags() == (False, False)
            return torch.zeros(1, 16, 1, dtype=torch.long)

        def forward(self, local, **kwargs):
            assert _tf32_flags() == (False, False)
            self.local = local
            return {"recon_state": local}

        def forward_decoder(self, indices, has_target_cond=None, **kwargs):
            assert _tf32_flags() == (False, False)
            has_context = has_target_cond is not None
            self.conditions.append(
                "none" if not has_context else
                "context1" if has_target_cond.sum().item() == 1 else "context4"
            )
            self.condition_masks.append(
                None if not has_context else has_target_cond.detach().cpu().clone()
            )
            offset = self.context_decode_error if has_context else 0.0
            return {"recon_state": self.local + offset}

    rep = Representation()
    rep.dual_rep = rep
    network = Tokenizer()
    model = SimpleNamespace(parameters=network.parameters, motion_rep=rep,
                            local_motion_rep=rep, pose_net=network,
                            args={"max_tokens": 16}, get_num_frames_per_code=lambda: 4)
    try:
        yield model, dataset, tmp_path / "evaluation_output"
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def test_evaluator_decorator_records_precision_and_restores_flags(precision_evaluation_fixture):
    model, dataset, output = precision_evaluation_fixture
    model.pose_net.context_decode_error = 5e-6

    result = checkpoint_review.evaluate_tokenizer(model, dataset, "fixture-manifest", output)

    assert _tf32_flags() == (True, True)
    assert model.pose_net.conditions == ["none", "context4"]
    assert result["token_decode_max_feature_error"] == pytest.approx(5e-6)
    assert result["precision_policy"] == {
        "name": "full_precision_evaluation",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert saved["precision_policy"] == result["precision_policy"]


def test_context1_uses_native_runtime_pose_slot3_and_reports_inputs(
        precision_evaluation_fixture):
    model, dataset, output = precision_evaluation_fixture

    result = checkpoint_review.evaluate_tokenizer(
        model, dataset, "fixture-manifest", output, conditions=("context1",)
    )

    assert model.pose_net.conditions == ["context1"]
    mask = model.pose_net.condition_masks[0]
    assert mask.shape == (1, 64)
    assert mask[0].nonzero().flatten().tolist() == [3]
    assert result["conditions"] == ["context1"]
    assert result["input_conditions"] == {
        "context1": {
            "pose_context_frame_indices": [3],
            "source_encoded_pose_values": True,
        }
    }
    assert result["summary"]["context1"]["windows"] == 1
    assert result["input_signature"] == checkpoint_review._window_signature(
        "fixture-manifest", [result["windows"][0]], ("context1",)
    )


def test_cli_forwards_selected_conditions(tmp_path, monkeypatch):
    captured = {}
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    monkeypatch.setattr(
        checkpoint_review, "load_native_checkpoint",
        lambda *args: (object(), {"checkpoint": "fixture"}, {
            "dataset_dir": dataset, "dataset_manifest_sha256": "manifest",
        }),
    )

    def evaluate(model, dataset_dir, manifest_hash, output, split, conditions):
        captured.update(split=split, conditions=conditions)
        output.mkdir()
        return {"summary": {}, "windows": [], "conditions": list(conditions)}

    monkeypatch.setattr(checkpoint_review, "evaluate_tokenizer", evaluate)
    checkpoint_review.main([
        "--run", str(tmp_path / "run"), "--output", str(tmp_path / "review"),
        "--conditions", "context1",
    ])

    assert captured == {"split": "validation", "conditions": ["context1"]}
    assert json.loads((tmp_path / "review" / "metrics.json").read_text())["conditions"] == [
        "context1"
    ]


@pytest.mark.parametrize("decode_error", [1.01e-5, 0.00263649])
def test_evaluator_still_rejects_parity_mismatch_and_restores_flags(
        precision_evaluation_fixture, decode_error):
    model, dataset, output = precision_evaluation_fixture
    model.pose_net.context_decode_error = decode_error

    with pytest.raises(ValueError, match="Token-index decoder parity failure"):
        checkpoint_review.evaluate_tokenizer(model, dataset, "fixture-manifest", output)

    assert _tf32_flags() == (True, True)
    assert model.pose_net.conditions == ["none", "context4"]
    assert not (output / "metrics.json").exists()


def test_fixed_training_windows_do_not_include_validation_or_test(tmp_path):
    records = []
    for split in ("train", "validation", "test"):
        shard = f"{split}.npz"
        np.savez(tmp_path / shard, global_features=np.zeros((67, 9), dtype=np.float32))
        records.append({"shard": shard, "split": split, "behavior": "idle"})
    (tmp_path / "accepted.json").write_text(json.dumps(records))
    windows, excluded = checkpoint_review._fixed_windows(tmp_path, "train")
    assert [row["start"] for row in windows] == [0, 1, 2]
    assert {row["shard"] for row in windows} == {"train.npz"}
    assert excluded == []
    validation_windows, validation_excluded = checkpoint_review._fixed_windows(tmp_path)
    assert [row["start"] for row in validation_windows] == [0, 1, 2]
    assert {row["shard"] for row in validation_windows} == {"validation.npz"}
    assert validation_excluded == []


def test_fixed_validation_windows_follow_eligible_accepted_records(tmp_path):
    records = []
    for index in range(13):
        shard = f"validation_{index}.npz"
        np.savez(tmp_path / shard,
                 global_features=np.zeros((129, 9), dtype=np.float32))
        records.append({"shard": shard, "split": "validation", "behavior": "stairs"})
    (tmp_path / "accepted.json").write_text(json.dumps(records), encoding="utf-8")

    windows, excluded = checkpoint_review._fixed_windows(tmp_path)

    assert len(windows) == 39
    assert len({row["shard"] for row in windows}) == 13
    assert excluded == []


def test_fixed_windows_requires_accepted_manifest_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="accepted.json"):
        checkpoint_review._fixed_windows(tmp_path)


def test_review_refuses_test_split_and_empty_training_split(tmp_path):
    (tmp_path / "accepted.json").write_text("[]")
    with pytest.raises(ValueError, match="test set stays untouched"):
        checkpoint_review._fixed_windows(tmp_path, "test")
    with pytest.raises(ValueError, match="No eligible train"):
        checkpoint_review._fixed_windows(tmp_path, "train")
    with pytest.raises(ValueError, match="No eligible validation"):
        checkpoint_review._fixed_windows(tmp_path, "validation")


def test_baseline_comparison_rejects_cross_split(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"dataset_split": "validation"}))
    with pytest.raises(ValueError, match="dataset split mismatch"):
        compare_baseline({"dataset_split": "train"}, baseline)
