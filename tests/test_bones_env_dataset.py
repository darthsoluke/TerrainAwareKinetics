"""BONES motion-scene training set: split, contacts, scene packing, windows, native frames."""

import json

import numpy as np
import pytest

from motionbricks.data.bones_env_dataset import (BonesEnvWindows, actor_of, assign_split, contact_channels,
                                                 pack_variants, scene_variants, window_starts)

GROUND = {"center_xyz_m": [0, -0.1, 0], "half_extent_xyz_m": [20, 0.1, 20],
          "rotation_local_to_world": np.eye(3).tolist(), "top_m": 0.0, "bottom_m": -0.2, "role": "ground"}
STEP = {"center_xyz_m": [0, 0.15, 1.0], "half_extent_xyz_m": [0.5, 0.15, 0.3],
        "rotation_local_to_world": np.eye(3).tolist(), "top_m": 0.3, "bottom_m": 0.0, "role": "support"}


def test_split_is_by_actor_and_deterministic():
    assert actor_of("walk_forward_001__A123") == "A123"
    with pytest.raises(ValueError):
        actor_of("walk_forward_001")
    splits = [assign_split(f"A{i}") for i in range(4000)]
    assert splits == [assign_split(f"A{i}") for i in range(4000)]
    share = {name: splits.count(name) / len(splits) for name in ("test", "validation", "train")}
    assert abs(share["test"] - 0.08) < 0.02 and abs(share["validation"] - 0.04) < 0.015


def test_contacts_need_a_low_and_still_sole():
    frames = 30
    foot_points = np.zeros((frames, 4, 3))
    foot_points[:, 0] = [-0.1, 0.08, -0.05]      # left heel joint
    foot_points[:, 1] = [-0.1, 0.03, 0.12]       # left toe joint
    foot_points[:, 2] = [0.1, 0.08, -0.05]
    foot_points[:, 3] = [0.1, 0.03, 0.12]
    left = np.array([[-0.1, 0.005, z] for z in np.linspace(-0.08, 0.15, 10)])[None].repeat(frames, 0)
    right = left.copy()
    right[..., 0] = 0.1
    right[:, :, 1] += np.linspace(0, 0.3, frames)[:, None]            # right foot lifts off
    contacts = contact_channels(left, right, foot_points, [GROUND])
    assert contacts[:, :2].all()                                      # left heel and toe planted
    assert contacts[:3, 2:].all() and not contacts[10:, 2:].any()     # right leaves the floor
    sliding = left + np.linspace(0, 0.3, frames)[:, None, None] * np.array([0, 0, 1.0])
    assert not contact_channels(sliding, right, foot_points, [GROUND])[5:, :2].any()   # sliding at 0.31 m/s


def test_scene_packing_translates_and_selects_variants():
    scene = {"variants": [{"index": 0, "accepted": True, "geometry_accepted": True, "boxes": [GROUND, STEP]},
                          {"index": 1, "accepted": False, "geometry_accepted": True, "boxes": [GROUND]}]}
    assert [i for i, _ in scene_variants(scene)] == [0]
    assert [i for i, _ in scene_variants(scene, include_geometry_only=True)] == [0, 1]
    centers, half, rotation, mask = pack_variants([b for _, b in scene_variants(scene)], dy=-0.05, capacity=8)
    assert centers.shape == (1, 8, 3) and mask[0].sum() == 2
    assert centers[0, 1, 1] == pytest.approx(0.10) and half[0, 1, 1] == pytest.approx(0.15)


def test_windows_need_valid_commands_for_every_step():
    valid = np.ones(60, dtype=bool)
    valid[:3] = False
    valid[40] = False
    starts = window_starts(60, valid, history=16, horizon=16, stride=1)
    assert starts.min() == 0 and all(not (s + 15 <= 40 < s + 31) for s in starts)
    assert window_starts(20, np.ones(20, bool), 16, 16) .size == 0


def test_window_dataset_reads_shards(tmp_path):
    torch = pytest.importorskip("torch")
    (tmp_path / "shards").mkdir()
    frames = 48
    centers, half, rotation, mask = pack_variants([[GROUND], [GROUND, STEP]], capacity=8)
    np.savez(tmp_path / "shards/a__A1.npz", features=np.random.rand(frames, 366).astype(np.float32),
             contacts=np.zeros((frames, 4), np.float32), commands=np.zeros((frames, 4), np.float32),
             command_valid=np.ones(frames, bool), stop_event=np.zeros(frames, bool),
             box_centers=centers.astype(np.float32), box_half=half.astype(np.float32),
             box_rotation=rotation.astype(np.float32), box_mask=mask)
    rows = [{"shard": "shards/a__A1.npz", "frames": frames, "split": "train"}]
    dataset = BonesEnvWindows(tmp_path, rows, "train", stride=4)
    assert len(dataset) == len(window_starts(frames, np.ones(frames, bool)))
    item = dataset[1]
    assert item["history"].shape == (16, 366) and item["targets"].shape == (16, 366)
    assert item["commands"].shape == (16, 4) and item["box_centers"].shape == (8, 3)
    assert torch.equal(item["targets"][0], torch.from_numpy(np.load(tmp_path / "shards/a__A1.npz")["features"][item["start"] + 16]))


def test_native_frames_roundtrip_the_source_fk():
    pytest.importorskip("torch")
    pytest.importorskip("kimodo")
    from motionbricks.data.bones_env_dataset import roundtrip_error, soma30_motion, world_features
    from motionbricks.data.bones_seed import build_representation
    from motionbricks.data.pfnn_full_training_data import input_intent_labels
    rep = build_representation()
    frames = 40
    angles = np.linspace(0, 1.2, frames)
    rotations = np.tile(np.eye(3, dtype=np.float32), (frames, 77, 1, 1))
    rotations[:, 0] = [[[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]] for a in angles]
    root = np.stack([np.linspace(0, 1, frames), np.full(frames, 0.95), np.linspace(0, 2, frames)], -1)
    local30, posed30 = soma30_motion(rep, rotations, root)
    features = world_features(rep, local30, root, np.zeros((frames, 4)))
    assert features.shape == (frames, 366)
    assert roundtrip_error(rep, features, posed30) < 2e-3
    assert np.allclose(features[:, [0, 2]].numpy(), posed30[:, rep.skeleton.root_idx][:, [0, 2]].numpy(), atol=2e-3)
    intent = input_intent_labels(features.numpy())
    assert intent["command"].shape[1] == 4
