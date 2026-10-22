import hashlib
import json
import math

import numpy as np
import pytest
import torch

from motionbricks.data.native_mb_data import (
    NativeMBMotionDataset,
    build_native_representation,
    collate_native_batch,
)
from motionbricks.data.native_mb_online_terrain import gather_online_terrain_observation
from motionbricks.data.local_terrain_observation import extract_local_terrain_observation
from motionbricks.data.native_mb_terrain_protocol import terrain_observation_config


def test_online_only_loader_uses_same_current_scene_without_stored_patch_arrays(tmp_path):
    path = tmp_path / 'dataset'
    _make_dataset(path)
    stored = NativeMBMotionDataset(path, 'train', min_frames=5, max_frames=10,
        require_terrain=True, require_online_terrain=True)[0]
    shard_path = path / 'shards/record.npz'
    with np.load(shard_path) as shard:
        motion = shard['global_features'].copy()
    np.savez_compressed(shard_path, global_features=motion)
    online = NativeMBMotionDataset(path, 'train', min_frames=5, max_frames=10,
        require_terrain=True, require_online_terrain=True, online_only=True)[0]
    assert 'terrain_points' not in online and 'terrain_valid' not in online
    torch.testing.assert_close(online['motion'], stored['motion'], atol=0, rtol=0)
    torch.testing.assert_close(online['terrain_root_world'], stored['terrain_root_world'], atol=0, rtol=0)
    for key in stored['terrain_scene']:
        torch.testing.assert_close(online['terrain_scene'][key], stored['terrain_scene'][key], atol=0, rtol=0)
    batch = collate_native_batch([online, online])
    assert len(batch['terrain_scene']) == 2 and batch['terrain_root_world'].shape == (2, 10, 5)
    assert 'terrain_points' not in batch


def _make_dataset(path, *, frames=10, heightfield=None, missing_geometry_field=None):
    path.mkdir(parents=True)
    (path / "stats").mkdir()
    (path / "shards").mkdir()
    (path / "terrain_geometry").mkdir()
    rep = build_native_representation()
    dimension = len(rep.indices["all"])
    stats_dimension = rep.dual_rep.motion_rep_dim
    np.save(path / "stats" / "mean.npy", np.zeros(stats_dimension, dtype=np.float32))
    np.save(path / "stats" / "std.npy", np.ones(stats_dimension, dtype=np.float32))

    global_features = np.zeros((frames, dimension), dtype=np.float32)
    frame = np.arange(frames, dtype=np.float32)
    canonical_position = np.stack((0.1 * frame, 1.0 + 0.01 * frame, 0.05 * frame), axis=-1)
    canonical_yaw = 0.08 * frame
    global_features[:, rep.indices["global_root_pos"]] = canonical_position
    global_features[:, rep.indices["global_root_heading"]] = np.stack(
        (np.cos(canonical_yaw), np.sin(canonical_yaw)), axis=-1)
    terrain_points = np.zeros((frames, 2, 9, 9, 3), dtype=np.float32)
    terrain_valid = np.ones((frames, 2, 9, 9), dtype=np.bool_)
    shard_relative = "shards/record.npz"
    np.savez_compressed(
        path / shard_relative,
        global_features=global_features,
        terrain_points=terrain_points,
        terrain_valid=terrain_valid,
    )

    if heightfield is None:
        heightfield = np.zeros((101, 101), dtype=np.float32)
    geometry = {
        "heightfield_y_m": np.asarray(heightfield, dtype=np.float32),
        "min_point_world_xyz_m": np.asarray([-5.0, 0.0, -5.0], dtype=np.float32),
        "dx_m": np.asarray(0.1, dtype=np.float32),
        "canonical_init_heading_quat": np.asarray(
            [math.cos(0.35), 0.0, math.sin(0.35), 0.0], dtype=np.float32),
        "canonical_root_pos_init_xz": np.asarray([2.0, -1.0], dtype=np.float32),
    }
    if missing_geometry_field is not None:
        geometry.pop(missing_geometry_field)
    geometry_relative = "terrain_geometry/record.npz"
    geometry_path = path / geometry_relative
    np.savez_compressed(geometry_path, **geometry)
    geometry_hash = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    record = {
        "split": "train",
        "shard": shard_relative,
        "frames_30hz": frames,
        "terrain_geometry_snapshot": geometry_relative,
        "terrain_geometry_sha256": geometry_hash,
    }
    (path / "accepted.json").write_text(json.dumps([record]), encoding="utf-8")
    (path / "native_manifest.json").write_text(
        json.dumps({"artifact_sha256": {geometry_relative: geometry_hash}}), encoding="utf-8")
    return record, global_features, geometry_path


def _dataset(path, **kwargs):
    return NativeMBMotionDataset(
        path, "train", max_frames=4, min_frames=1, require_terrain=True,
        require_online_terrain=True, **kwargs)


def test_online_sidecar_restores_world_roots_and_collates_the_exact_crop(tmp_path, monkeypatch):
    path = tmp_path / "native"
    _, features, _ = _make_dataset(path)
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(2))
    dataset = _dataset(path, cache_max_bytes=1_000_000)
    item = dataset[0]

    assert item["terrain_root_world"].shape == (4, 5)
    torch.testing.assert_close(item["terrain_scene"]["heightfield_y_m"], torch.zeros(101, 101))
    torch.testing.assert_close(item["terrain_scene"]["min_point_world_xyz_m"], torch.tensor([-5., 0., -5.]))
    assert item["terrain_scene"]["dx_m"].item() == pytest.approx(0.1)

    source_pos = features[2:6, build_native_representation().indices["global_root_pos"]]
    source_yaw = 0.08 * np.arange(2, 6)
    c, s = math.cos(0.7), math.sin(0.7)
    expected = np.stack((
        c * source_pos[:, 0] + s * source_pos[:, 2] + 2.0,
        source_pos[:, 1],
        -s * source_pos[:, 0] + c * source_pos[:, 2] - 1.0,
        np.cos(source_yaw + 0.7),
        np.sin(source_yaw + 0.7),
    ), axis=-1)
    np.testing.assert_allclose(item["terrain_root_world"].numpy(), expected, atol=2e-6)

    batch = collate_native_batch([item, item])
    assert isinstance(batch["terrain_scene"], list) and len(batch["terrain_scene"]) == 2
    torch.testing.assert_close(batch["terrain_root_world"], item["terrain_root_world"].repeat(2, 1, 1))
    assert dataset._cached_bytes == dataset._cache_entry_bytes["shards/record.npz"]
    assert dataset._cached_bytes <= dataset.cache_max_bytes


def test_gather_uses_resampled_rows_current_frame_and_runtime_query_exactly(tmp_path):
    path = tmp_path / "native"
    _make_dataset(path)
    item = _dataset(path)[0]
    batch = collate_native_batch([item, item])
    chosen = torch.tensor([1, 0, 1])
    frame_indices = torch.tensor([[0, 1, 2, 1, 999], [3, 2, 1, 0, -999], [2, 0, 3, 2, 123]])
    directions = torch.tensor([[0.2, 1.0], [-0.4, 0.8], [1.0, 0.0]])
    sample_info = {"chosen_ids": chosen, "frame_indices": frame_indices}

    observed = gather_online_terrain_observation(batch, sample_info, directions)
    expected_points, expected_valid = [], []
    for row in range(len(chosen)):
        source = int(chosen[row])
        frame = int(frame_indices[row, 3])
        root = batch["terrain_root_world"][source, frame]
        scene = batch["terrain_scene"][source]
        direct = extract_local_terrain_observation(
            scene["heightfield_y_m"], scene["min_point_world_xyz_m"], scene["dx_m"],
            root[:3], torch.atan2(root[4], root[3]), directions[row],
            config=terrain_observation_config(),
        )
        expected_points.append(direct.points[0])
        expected_valid.append(direct.valid[0])
    torch.testing.assert_close(observed["points"], torch.stack(expected_points))
    torch.testing.assert_close(observed["valid"], torch.stack(expected_valid))
    assert not observed["points"].requires_grad and not observed["valid"].requires_grad


def test_command_moves_the_forward_patch_and_rotates_with_body_yaw():
    heightfield = np.zeros((81, 81), dtype=np.float32)
    heightfield[50, 40] = 0.75
    scene = {
        "heightfield_y_m": torch.from_numpy(heightfield),
        "min_point_world_xyz_m": torch.tensor([-4.0, 0.0, -4.0]),
        "dx_m": torch.tensor(0.1),
    }
    root = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0]]])
    batch = {"terrain_scene": [scene], "terrain_root_world": root}
    info = {"chosen_ids": torch.tensor([0]), "frame_indices": torch.tensor([[0, 0, 0, 0]])}
    forward = gather_online_terrain_observation(batch, info, torch.tensor([[0.0, 1.0]]))
    sideways = gather_online_terrain_observation(batch, info, torch.tensor([[1.0, 0.0]]))
    assert forward["points"][:, 1, :, :, 1].max().item() == pytest.approx(0.75)
    assert sideways["points"][:, 1, :, :, 1].max().item() == 0.0

    rotated_scene = {**scene, "heightfield_y_m": torch.from_numpy(np.rot90(heightfield).copy())}
    rotated_root = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 1.0]]])
    rotated = gather_online_terrain_observation(
        {"terrain_scene": [rotated_scene], "terrain_root_world": rotated_root},
        info, torch.tensor([[0.0, 1.0]]),
    )
    torch.testing.assert_close(rotated["points"], forward["points"])
    torch.testing.assert_close(rotated["valid"], forward["valid"])


def test_unknown_map_keeps_command_geometry_without_motion_fallback():
    scene = {
        "heightfield_y_m": torch.full((9, 9), float("nan")),
        "min_point_world_xyz_m": torch.tensor([-0.4, 0.0, -0.4]),
        "dx_m": torch.tensor(0.1),
    }
    batch = {
        "terrain_scene": [scene],
        "terrain_root_world": torch.tensor([[[0.0, 1.0, 0.0, 1.0, 0.0]]]),
    }
    info = {"chosen_ids": torch.tensor([0]), "frame_indices": torch.tensor([[0, 0, 0, 0]])}
    forward = gather_online_terrain_observation(batch, info, torch.tensor([[0.0, 1.0]]))
    right = gather_online_terrain_observation(batch, info, torch.tensor([[1.0, 0.0]]))
    assert not forward["valid"].any() and not right["valid"].any()
    assert torch.equal(forward["points"][..., 1], torch.zeros_like(forward["points"][..., 1]))
    assert not torch.equal(forward["points"][:, 1, ..., [0, 2]], right["points"][:, 1, ..., [0, 2]])


def test_online_mode_requires_offline_terrain_flag(tmp_path):
    path = tmp_path / "native"
    _make_dataset(path)
    with pytest.raises(ValueError, match="requires require_terrain=True"):
        NativeMBMotionDataset(path, "train", min_frames=1, require_online_terrain=True)


def test_geometry_hash_required_fields_and_safe_path_are_validated(tmp_path):
    path = tmp_path / "native"
    record, _, geometry_path = _make_dataset(path)
    geometry_path.write_bytes(geometry_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _dataset(path)[0]

    missing_path = tmp_path / "missing_field"
    _make_dataset(missing_path, missing_geometry_field="canonical_root_pos_init_xz")
    with pytest.raises(ValueError, match="lacks.*canonical_root_pos_init_xz"):
        _dataset(missing_path)[0]

    escape_path = tmp_path / "escape"
    escaped_record, _, _ = _make_dataset(escape_path)
    outside = tmp_path / "outside.npz"
    outside.write_bytes((escape_path / escaped_record["terrain_geometry_snapshot"]).read_bytes())
    escaped_record["terrain_geometry_snapshot"] = "../outside.npz"
    outside_hash = hashlib.sha256(outside.read_bytes()).hexdigest()
    escaped_record["terrain_geometry_sha256"] = outside_hash
    (escape_path / "accepted.json").write_text(json.dumps([escaped_record]), encoding="utf-8")
    (escape_path / "native_manifest.json").write_text(
        json.dumps({"artifact_sha256": {"../outside.npz": outside_hash}}), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes the dataset"):
        _dataset(escape_path)
