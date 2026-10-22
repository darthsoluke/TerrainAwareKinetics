import json
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from motionbricks.data.native_mb_data import (
    FPS,
    NativeMBMotionDataset,
    build_native_representation,
    build_train_stats,
    collate_native_batch,
)

_builder_path = Path(__file__).parents[1] / "scripts" / "build_native_mb_dataset.py"
_builder_spec = importlib.util.spec_from_file_location("build_native_mb_dataset", _builder_path)
_builder = importlib.util.module_from_spec(_builder_spec)
_builder_spec.loader.exec_module(_builder)
build_dataset = _builder.build_dataset


def test_expansion_requires_completed_report(tmp_path):
    (tmp_path / "skeleton_contract.json").write_text(json.dumps({"expansion_note": "frozen"}))
    with pytest.raises(ValueError, match="no completed expansion report"):
        _builder._validate_expansion_source(tmp_path, [])
    (tmp_path / "expansion_report.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="completion/count mismatch"):
        _builder._validate_expansion_source(tmp_path, [])


def test_expansion_provenance_binds_frozen_records_and_bytes(tmp_path):
    source, pilot = tmp_path / "source", tmp_path / "pilot"
    source.mkdir()
    pilot.mkdir()
    records = [{"split": "train", "filename": "train"},
               {"split": "validation", "filename": "heldout"}]
    (pilot / "accepted.json").write_text(json.dumps(records))
    (source / "copied.npz").write_bytes(b"original shard")
    (source / "reserved_content_hashes.json").write_text("[]")
    report = {"status": "completed", "split_counts": {"train": 1, "validation": 1, "test": 0},
              "source_pilot": str(pilot),
              "source_accepted_sha256": _builder._sha256(pilot / "accepted.json"),
              "copied_pilot_shard_sha256": {"copied.npz": _builder._sha256(source / "copied.npz")},
              "reserved_content_hashes_sha256": _builder._sha256(source / "reserved_content_hashes.json")}
    (source / "expansion_report.json").write_text(json.dumps(report))
    assert _builder._validate_expansion_source(source, records)["report_sha256"]
    changed = [records[0], dict(records[1], filename="different")]
    with pytest.raises(ValueError, match="changed frozen held-out"):
        _builder._validate_expansion_source(source, changed)
    (source / "copied.npz").write_bytes(b"changed shard")
    with pytest.raises(ValueError, match="changed a copied pilot shard"):
        _builder._validate_expansion_source(source, records)


def _source_dataset(path, frames=141):
    path.mkdir()
    (path / "shards").mkdir()
    records = []
    for split, amplitude in (("train", 1.0), ("validation", 5.0)):
        filename = f"{split}_clip"
        shard = f"shards/{split}.npz"
        t = np.arange(frames, dtype=np.float32) / 60.0
        root = np.zeros((frames, 3), dtype=np.float32)
        root[:, 0] = 0.4 + amplitude * (0.1 * t + 0.02 * np.sin(8 * t))
        root[:, 1] = 1.0
        root[:, 2] = -0.2
        local = np.broadcast_to(np.eye(3, dtype=np.float32), (frames, 30, 3, 3)).copy()
        yaw = 0.37
        local[:, 0] = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                                [0, 1, 0],
                                [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)
        np.savez_compressed(
            path / shard,
            local_rot_mats=local,
            root_positions=root,
            global_features=np.zeros((frames, 366), dtype=np.float32),
        )
        records.append({
            "filename": filename,
            "recording_group": filename,
            "behavior": "walk",
            "split": split,
            "shard": shard,
            "source_fps": 120.0,
            "frames_60hz": frames,
        })
    (path / "accepted.json").write_text(json.dumps(records), encoding="utf-8")


def test_builder_resamples_and_recomputes_30hz_velocities(tmp_path):
    source, output = tmp_path / "source", tmp_path / "native"
    _source_dataset(source)
    build_dataset(source, output)
    rows = json.loads((output / "accepted.json").read_text(encoding="utf-8"))
    row = next(record for record in rows if record["split"] == "train")
    with np.load(output / row["shard"], allow_pickle=False) as shard:
        assert shard["fps"].item() == FPS
        assert shard["global_features"].shape[0] == 71
        assert shard["dual_features"].shape == (71, 370)
        rep = build_native_representation()
        indices = rep.indices["local_vel"]
        root_pos = shard["global_features"][:, rep.indices["global_root_pos"]]
        root_vel = shard["global_features"][:, indices[:3]]
        np.testing.assert_allclose(root_vel[:-1], np.diff(root_pos, axis=0) * FPS, atol=2e-5)
        np.testing.assert_allclose(root_vel[-1], root_vel[-2], atol=2e-5)
        assert np.max(np.abs(root_vel)) > 0.05
        assert not np.array_equal(shard["global_features"], np.zeros((71, 366), np.float32))
        regenerated = rep.dual_rep({
            "local_joint_rots": torch.from_numpy(shard["local_rot_mats"].copy()),
            "translation": torch.from_numpy(shard["root_positions"].copy()),
        }, to_normalize=False)
        regenerated_global = rep.dual_rep.get_feature_subset(regenerated, mode="global")
        torch.testing.assert_close(
            regenerated_global, torch.from_numpy(shard["global_features"].copy()),
            atol=3e-4, rtol=3e-4,
        )
        assert np.linalg.norm(shard["root_positions"][0, [0, 2]]) < 1e-5
    provenance = json.loads((output / "native_manifest.json").read_text(encoding="utf-8"))
    assert provenance["contract"]["target_fps"] == 30.0
    assert "no per-frame" in provenance["contract"]["ground"]
    sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["accepted_json_sha256"] == sha256(output / "accepted.json")
    assert provenance["stats_artifacts_sha256"] == {
        "mean.npy": sha256(output / "stats" / "mean.npy"),
        "std.npy": sha256(output / "stats" / "std.npy"),
    }
    assert provenance["artifact_sha256"]["accepted.json"] == sha256(output / "accepted.json")
    assert provenance["artifact_sha256"]["stats/mean.npy"] == sha256(output / "stats" / "mean.npy")
    assert provenance["artifact_sha256"]["stats/std.npy"] == sha256(output / "stats" / "std.npy")
    assert provenance["artifact_sha256"][row["shard"]] == sha256(output / row["shard"])
    assert "converter" in provenance["code_sha256"]
    assert {"native_mb_data", "bones_seed", "official_representation"}.issubset(
        provenance["code_sha256"]
    )
    assert row["source_shard_fps"] == 60.0
    assert row["source_original_fps"] == 120.0
    assert row["converted_shard_sha256"] == sha256(output / row["shard"])


def test_stats_are_train_only_and_use_65_frame_windows(tmp_path):
    source, output = tmp_path / "source", tmp_path / "native"
    _source_dataset(source)
    build_dataset(source, output)
    rows = json.loads((output / "accepted.json").read_text(encoding="utf-8"))
    original_mean = np.load(output / "stats" / "mean.npy")
    original_std = np.load(output / "stats" / "std.npy")
    manifest_hash = json.loads((output / "native_manifest.json").read_text(encoding="utf-8"))["source_manifest_sha256"]

    changed = tmp_path / "changed_validation"
    changed.mkdir()
    (changed / "shards").mkdir()
    changed_rows = []
    for row in rows:
        row = dict(row)
        with np.load(output / row["shard"], allow_pickle=False) as shard:
            arrays = {key: shard[key].copy() for key in shard.files}
        if row["split"] == "validation":
            arrays["global_features"][:] += 10000
        np.savez_compressed(changed / row["shard"], **arrays)
        changed_rows.append(row)
    stats = build_train_stats(changed, changed_rows, manifest_hash)
    np.testing.assert_array_equal(np.load(changed / "stats" / "mean.npy"), original_mean)
    np.testing.assert_array_equal(np.load(changed / "stats" / "std.npy"), original_std)
    assert stats["window_global_frames"] == 65
    assert stats["window_local_frames"] == 64
    assert stats["yaw_quadrature_degrees"] == [0, 90, 180, 270]


def test_dataset_minimum_and_official_batch_keys(tmp_path):
    source, output = tmp_path / "source", tmp_path / "native"
    _source_dataset(source)
    build_dataset(source, output)
    dataset = NativeMBMotionDataset(output, "train", max_frames=65, min_frames=65)
    item = dataset[0]
    assert set(item) == {"keyid", "motion"}
    rep = build_native_representation(output / "stats")
    assert item["motion"].shape == (65, len(rep.indices["all"]))
    assert item["motion"].shape[-1] == rep.motion_rep_dim == 366
    assert rep.dual_rep.motion_rep_dim == 370
    batch = collate_native_batch([item])
    assert set(batch) == {"motion", "motion_len", "motion_pad_mask", "batch_size"}
    assert batch["motion"].shape[-1] == len(rep.indices["all"])
    assert batch["motion_len"].tolist() == [65]
    assert batch["motion_pad_mask"].all()
    local = rep.dual_rep.global_to_local(
        batch["motion"], is_normalized=True, to_normalize=True,
        lengths=batch["motion_len"],
    )
    assert local.shape == (1, 65, rep.dual_rep.local_motion_rep.motion_rep_dim)
    inverse = rep.inverse(batch["motion"], is_normalized=True, return_all=True)
    assert inverse["posed_joints"].shape == (1, 65, 30, 3)
    assert inverse["local_joint_rots"].shape == (1, 65, 30, 3, 3)
    with pytest.raises(ValueError, match="No train records"):
        NativeMBMotionDataset(output, "train", max_frames=100, min_frames=72)


def test_missing_accepted_shard_fails_without_partial_output(tmp_path):
    source, output = tmp_path / "source", tmp_path / "native"
    _source_dataset(source)
    (source / "accepted.json").write_text(
        (source / "accepted.json").read_text(encoding="utf-8").replace(
            '"shard": "shards/validation.npz"', '"shard": "shards/not-present.npz"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="refusing a partial native dataset"):
        build_dataset(source, output)
    assert not output.exists()
