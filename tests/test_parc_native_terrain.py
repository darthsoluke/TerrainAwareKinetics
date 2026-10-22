"""Matched offline terrain/no-terrain evaluation and provenance checks."""

import json
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

from motionbricks.data.native_mb_terrain_protocol import TERRAIN_PROTOCOL
from scripts import evaluate_parc_native_terrain as evaluator
from test_native_mb_stage_training import _motion_and_representation


def test_identical_fp32_rotations_have_zero_angular_difference():
    from scipy.spatial.transform import Rotation
    matrices = torch.as_tensor(Rotation.from_rotvec([[.13, .27, .41], [.4, .1, -.7]]).as_matrix(), dtype=torch.float32)
    assert torch.equal(evaluator._rotation_geodesic(matrices, matrices.clone()), torch.zeros(2, dtype=torch.float64))
    opposite_sign_same_matrix = matrices.clone()
    assert evaluator._rotation_geodesic(matrices, opposite_sign_same_matrix).max() == 0


def test_half_turn_geodesic_matches_pi():
    from scipy.spatial.transform import Rotation
    first = torch.eye(3)[None]
    second = torch.as_tensor(Rotation.from_rotvec([[0, np.pi, 0]]).as_matrix(), dtype=torch.float32)
    assert evaluator._rotation_geodesic(first, second).item() == pytest.approx(np.pi, abs=1e-7)


@pytest.fixture
def native_source():
    rep, batch = _motion_and_representation(frame_count=75)
    raw = rep.unnormalize(batch["motion"])
    raw[:, :, 0] = 1.25 + torch.arange(75) / 30
    return rep, raw


def _dataset(path, raw, *, world_transform=False, records_count=1):
    path.mkdir()
    (path / "shards").mkdir()
    records = []
    artifacts = {}
    acceptance_name = "acceptance.json"
    (path / acceptance_name).write_text("{}\n", encoding="utf-8")
    artifacts[acceptance_name] = evaluator.sha256(path / acceptance_name)
    terrain = np.zeros((len(raw[0]), 2, 9, 9, 3), dtype=np.float32)
    terrain[:, :, :, :, 0] = np.arange(len(terrain), dtype=np.float32)[:, None, None, None]
    valid = np.ones(terrain.shape[:-1], dtype=np.bool_)
    for index in range(records_count):
        shard = f"shards/clip-{index:02}.npz"
        values = {
            "global_features": raw[0].numpy(),
            "terrain_points": terrain,
            "terrain_valid": valid,
        }
        if world_transform:
            values.update(world_transform if isinstance(world_transform, dict) else {
                "canonical_init_heading_quat": np.asarray([[0, 0, 0, 1]], dtype=np.float32),
                "canonical_root_pos_init_xz": np.asarray([[2.5, -1.0]], dtype=np.float32),
            })
        np.savez(path / shard, **values)
        records.append({
            "shard": shard,
            "split": "validation",
            "behavior": "stairs",
            "terrain_family": "stairs",
            "source_group": f"terrain-group-{index}",
        })
        artifacts[shard] = evaluator.sha256(path / shard)
    evaluator.write_json(path / "accepted.json", records)
    artifacts["accepted.json"] = evaluator.sha256(path / "accepted.json")
    terrain_contract = dict(TERRAIN_PROTOCOL)
    terrain_contract["acceptance"] = {
        "retarget_status": "passed",
        "motion_terrain_pairing_status": "passed",
        "report": acceptance_name,
    }
    evaluator.write_json(path / "native_manifest.json", {
        "artifact_sha256": artifacts,
        "terrain_conditioning": terrain_contract,
    })
    return records, evaluator.sha256(path / "native_manifest.json")


def _builder_world_transform(rep, raw):
    from scripts import build_parc_native_dataset as builder

    decoded = rep.inverse(raw, is_normalized=False, return_all=True)
    local = decoded["local_joint_rots"][0].detach().cpu().numpy().astype(np.float32)
    positions = decoded["posed_joints"][0].detach().cpu().numpy()
    root = positions[:, rep.skeleton.root_idx].astype(np.float32)
    xz = positions[..., (0, 2)]
    minimum = xz.min(axis=(0, 1)) - 1.0
    maximum = xz.max(axis=(0, 1)) + 1.0
    dx = 0.1
    width, height = np.ceil((maximum - minimum) / dx).astype(int) + 1
    clip = {
        "heightfield_y_m": np.zeros((height, width), dtype=np.float32),
        "min_point_world_xyz_m": np.asarray([minimum[0], 0.0, minimum[1]], dtype=np.float32),
        "dx_m": dx,
    }
    encoded = builder._native_encode(local, root, rep, clip, return_frame=True)
    return {
        "canonical_init_heading_quat": encoded[-1]["init_heading_quat"],
        "canonical_root_pos_init_xz": encoded[-1]["root_pos_init_xz"],
    }


class FakeCodec:
    def __init__(self, rep):
        self.motion_rep = rep.dual_rep.local_motion_rep
        self.decoder_external_cond_feature_mode = "root_without_hip_height_without_heading"
        self.local = None

    def encode_into_idx(self, local, fetch_overall_indices=False):
        self.local = local.clone()
        return torch.zeros((local.shape[0], 16, 1), dtype=torch.long, device=local.device)

    def forward_decoder(self, tokens, **kwargs):
        return {"recon_state": self.local.clone()}


class FakeInference:
    EXTERNAL_POSE_FEATURE_MODE = "joint_positions_and_rotations"
    device = "cpu"
    _args = {"min_tokens": 6, "max_tokens": 16, "down_t": 2}

    def __init__(self, rep, raw):
        self.motion_rep = self.global_motion_rep = rep
        self.local_motion_rep = rep.dual_rep.local_motion_rep
        self._root_model = SimpleNamespace(backbone_net=SimpleNamespace(_args={"terrain_conditioning": True}))
        self._pose_model = SimpleNamespace(backbone_net=SimpleNamespace(_args={"terrain_conditioning": True}))
        self._vqvae_pose_model = FakeCodec(rep)
        self.base = rep.change_first_heading(raw[:, :65], 0.0, False, False)[:, :64].clone()
        self.calls = []

    def eval(self):
        return self

    def predict(self, **inputs):
        terrain = inputs["terrain_observation"]
        valid = terrain["valid"]
        fraction = float(valid.float().mean())
        self.calls.append({
            "seed": torch.initial_seed(),
            "inputs": {name: inputs[name].clone() for name in (
                "global_root_values", "local_root_values", "local_poses",
                "has_global_root_values", "has_local_root_values", "has_local_poses",
                "allowed_pred_num_tokens")},
            "terrain": {name: value.clone() for name, value in terrain.items()},
        })
        prediction = self.base.clone()
        root_x = self.global_motion_rep.indices["global_root_pos"][0]
        prediction[:, :, root_x] += 0.12 * fraction
        return prediction, torch.tensor([16])


def test_window_selection_is_repeatable_capped_and_group_stratified():
    records = [
        {"shard": "a", "split": "validation", "terrain_family": "stairs", "source_group": "g1"},
        {"shard": "b", "split": "validation", "terrain_family": "stairs", "source_group": "g2"},
        {"shard": "c", "split": "validation", "terrain_family": "ramp", "source_group": "g3"},
        {"shard": "d", "split": "validation", "terrain_family": "ramp", "source_group": "g4"},
    ]
    windows = [{"shard": row["shard"], "start": start, "behavior": row["terrain_family"]}
               for row in records for start in (0, 5, 10)]
    first = evaluator.select_windows(windows, records, 4, "manifest-hash")
    second = evaluator.select_windows(windows, records, 4, "manifest-hash")
    assert first == second
    assert len(first) == 4
    selected_groups = {row["source_group"] for row in records
                       if any(item["shard"] == row["shard"] for item in first)}
    selected_families = {row["terrain_family"] for row in records
                         if any(item["shard"] == row["shard"] for item in first)}
    assert len(selected_groups) == 4
    assert selected_families == {"stairs", "ramp"}
    assert len(evaluator.select_windows(windows, records, 50, "manifest-hash")) == len(windows)
    with pytest.raises(ValueError, match="positive integer"):
        evaluator.select_windows(windows, records, 0, "manifest-hash")


def test_current_terrain_uses_only_last_context_frame(tmp_path, native_source):
    rep, raw = native_source
    dataset = tmp_path / "dataset"
    _dataset(dataset, raw)
    window = {"shard": "shards/clip-00.npz", "start": 5, "behavior": "stairs"}
    terrain, info = evaluator._read_window_terrain(dataset, window, torch.device("cpu"))
    assert info["frame_index"] == 8
    assert info["valid_points"] == 2 * 9 * 9
    assert torch.all(terrain["points"][..., 0] == 8)


def test_evaluation_matches_conditions_and_seed_and_reports_nonzero_branch_delta(
        tmp_path, native_source):
    rep, raw = native_source
    dataset, output = tmp_path / "dataset", tmp_path / "review"
    _, manifest_hash = _dataset(dataset, raw)
    infer = FakeInference(rep, raw)
    report = evaluator.evaluate(
        infer, dataset, output, {"dataset_manifest_sha256": manifest_hash},
        seed=91, max_windows=2,
    )
    assert report["window_counts"]["evaluated_windows"] == 2
    assert report["window_counts"]["windows_with_any_valid_terrain"] == 2
    assert len(infer.calls) == 4
    for terrain_call, invalid_call in zip(infer.calls[::2], infer.calls[1::2]):
        assert terrain_call["seed"] == invalid_call["seed"]
        assert terrain_call["seed"] in (91, 92)
        for name in terrain_call["inputs"]:
            torch.testing.assert_close(terrain_call["inputs"][name], invalid_call["inputs"][name],
                                       rtol=0, atol=0)
        assert terrain_call["terrain"]["valid"].any()
        assert not invalid_call["terrain"]["valid"].any()
        torch.testing.assert_close(terrain_call["terrain"]["points"],
                                   invalid_call["terrain"]["points"], rtol=0, atol=0)
    assert report["terrain_branch_difference_nonzero_windows"] == 2
    assert report["summary"]["terrain_vs_all_invalid_difference"]["root_trajectory_delta_m"] > 0
    assert report["windows"][0]["canonical_world_transform"] == "unavailable_missing_canonical_transform_fields"
    assert report["world_metrics_status"] == "canonical_only_for_rows_missing_transform"
    assert "not WASD proof" in report["interpretation"]
    assert "unseen-terrain generalization claim" in report["interpretation"]
    assert report["quality_approved"] is False
    assert report["ik_or_foot_lock_or_smoothing"] == "none"
    assert report["conditions"]["future_terrain_queried"] is False
    assert set(report["windows"][0]["codec_reconstruction"]) == set(report["windows"][0]["terrain"])
    assert json.loads((output / "metrics.json").read_text())["quality_approved"] is False


def test_world_metrics_require_both_canonical_transform_fields(tmp_path, native_source):
    rep, raw = native_source
    dataset, output = tmp_path / "dataset", tmp_path / "review"
    builder_transform = _builder_world_transform(rep, raw)
    assert builder_transform["canonical_init_heading_quat"].shape in {(4,), (1, 4)}
    assert builder_transform["canonical_root_pos_init_xz"].shape in {(2,), (1, 2)}
    _, manifest_hash = _dataset(dataset, raw, world_transform=builder_transform)
    report = evaluator.evaluate(
        FakeInference(rep, raw), dataset, output,
        {"dataset_manifest_sha256": manifest_hash}, max_windows=1,
    )
    assert report["windows"][0]["canonical_world_transform"] == "available"
    assert report["windows"][0]["metric_coordinate_frame"] == "world"
    assert report["world_metrics_status"] == "available_per_window"


def test_world_transform_rejects_undocumented_shapes(tmp_path, native_source):
    rep, raw = native_source
    dataset = tmp_path / "dataset"
    records, _ = _dataset(dataset, raw, world_transform=True)
    shard_path = dataset / records[0]["shard"]
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {name: shard[name].copy() for name in shard.files}
    arrays["canonical_init_heading_quat"] = np.zeros((2, 4), dtype=np.float32)
    np.savez(shard_path, **arrays)
    manifest_path = dataset / "native_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"][records[0]["shard"]] = evaluator.sha256(shard_path)
    evaluator.write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="singleton-batched"):
        evaluator._world_transform(
            dataset, {"shard": records[0]["shard"]}, torch.device("cpu"))


def test_metrics_survive_optional_render_failure(tmp_path, native_source, monkeypatch):
    rep, raw = native_source
    dataset, output = tmp_path / "dataset", tmp_path / "review"
    _, manifest_hash = _dataset(dataset, raw)

    def fail_render(*args, **kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(evaluator, "_render_skeleton_examples", fail_render)
    report = evaluator.evaluate(
        FakeInference(rep, raw), dataset, output,
        {"dataset_manifest_sha256": manifest_hash}, max_windows=1, render=True,
    )
    persisted = json.loads((output / "metrics.json").read_text())
    assert report["summary"] == persisted["summary"]
    assert persisted["render"]["status"] == "failed"
    assert persisted["render"]["metrics_persisted"] is True
    assert persisted["window_counts"]["evaluated_windows"] == 1


def test_workflow_loader_requires_finished_three_stage_state(tmp_path):
    workflow = tmp_path / "pilot" / "state.json"
    workflow.parent.mkdir()
    dataset = tmp_path / "dataset"
    stages = {name: {"status": "completed", "run": f"runs/{name}"}
              for name in ("tokenizer", "root", "pose")}
    evaluator.write_json(workflow, {
        "status": "training_finished_behavior_review_pending",
        "dataset": str(dataset),
        "dataset_manifest_sha256": "a" * 64,
        "stages": stages,
    })
    loaded_dataset, root_run, pose_run, provenance = evaluator.load_workflow(workflow)
    assert loaded_dataset == dataset.resolve()
    assert root_run == (workflow.parent / "runs/root").resolve()
    assert pose_run == (workflow.parent / "runs/pose").resolve()
    assert provenance["state_sha256"] == evaluator.sha256(workflow)
    stages["pose"]["status"] = "running"
    evaluator.write_json(workflow, {
        "status": "training_finished_behavior_review_pending",
        "dataset": str(dataset),
        "dataset_manifest_sha256": "a" * 64,
        "stages": stages,
    })
    with pytest.raises(ValueError, match="completed tokenizer, Root, and Pose"):
        evaluator.load_workflow(workflow)


def test_cli_exposes_workflow_review_interface():
    result = subprocess.run(
        [sys.executable, str(Path(evaluator.__file__)), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    for option in ("--workflow", "--output", "--max-windows", "--render"):
        assert option in result.stdout
