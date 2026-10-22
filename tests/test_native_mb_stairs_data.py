import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_native_mb_stairs.py"
SPEC = importlib.util.spec_from_file_location("prepare_native_mb_stairs", SCRIPT)
stairs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stairs)


@pytest.mark.parametrize("name,expected", [
    ("stairs_climbing_up__A300.bvh", (300, "up")),
    ("stairs_climbing_down__A301.bvh", (301, "down")),
    ("stairs_climbing_up_dance__A300.bvh", None),
    ("stairs_climbing_slide_down__A301.bvh", None),
    ("stairs_climbing_up__A302.bvh", None),
    ("walk_upstairs__A300.bvh", None),
    ("stairs_climbing__A300.bvh", None),
])
def test_stair_inventory_is_filename_and_actor_bounded(name, expected):
    assert stairs.stair_actor(Path(name)) == expected


def test_positive_only_weak_contacts_preserve_unknown_as_unlabeled():
    surfaces = [
        {"channel": "LeftFoot", "selected_indices": np.array([10, 11])},
        {"channel": "LeftToeBase", "selected_indices": np.array([12, 13])},
        {"channel": "RightFoot", "selected_indices": np.array([20, 21])},
        {"channel": "RightToeBase", "selected_indices": np.array([22, 23])},
    ]
    side_indices = {"left": np.array([10, 11, 12, 13]),
                    "right": np.array([20, 21, 22, 23])}
    weak = {
        "vertex_measured_left": np.array([[True, False, False, False],
                                           [False, False, True, False]]),
        "vertex_measured_right": np.zeros((2, 4), dtype=bool),
    }
    labels, mask, confidence = stairs._weak_contact_channels(
        weak, surfaces, side_indices, 3,
        ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"])
    assert labels.shape == mask.shape == confidence.shape == (3, 4)
    assert labels[:, 0].tolist() == [True, True, False]
    assert labels[:, 1].tolist() == [False, True, True]
    assert not mask[:, 2:].any()
    assert np.array_equal(labels, mask)
    assert np.array_equal(confidence, mask.astype(np.float32) * 0.5)


def test_canonical_point_transform_uses_official_heading_and_initial_root():
    import torch

    angle = np.pi / 2
    info = {
        "init_heading_quat": torch.tensor([np.cos(angle / 2), 0.0,
                                            np.sin(angle / 2), 0.0]),
        "root_pos_init_xz": torch.tensor([3.0, -2.0]),
    }
    points = np.array([[3.0, 1.5, -2.0], [4.0, 2.0, -2.0]], dtype=np.float32)
    transformed = stairs._canonicalize_points(points, info)
    np.testing.assert_allclose(transformed, [[0.0, 1.5, 0.0], [0.0, 2.0, 1.0]], atol=1e-6)


def test_flat_manifest_rows_are_copied_unchanged_and_confined(tmp_path):
    source, output = tmp_path / "reference", tmp_path / "output"
    shard = source / "shards" / "flat.npz"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"flat shard")
    row = {"filename": "flat", "split": "train", "frames_30hz": 70,
           "shard": "shards/flat.npz", "source_shard_sha256": "unchanged"}
    (source / "accepted.json").write_text(json.dumps([row]))
    output.mkdir()
    result = stairs.inherited_flat_rows(source, output)
    assert result == [row]
    assert (output / row["shard"]).read_bytes() == shard.read_bytes()
    assert stairs.sha256_file(output / row["shard"]) == stairs.sha256_file(shard)


def test_flat_manifest_rejects_shard_path_escape(tmp_path):
    source, output = tmp_path / "reference", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "accepted.json").write_text(json.dumps([{"shard": "../outside.npz"}]))
    with pytest.raises(ValueError, match="escapes reference"):
        stairs.inherited_flat_rows(source, output)


def test_skin_parity_reference_must_share_constant_ground_translation():
    import torch
    from motionbricks.data.bones_seed import official_sole_surfaces
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.data.soma_skin_reference import load_official_skinner
    from kimodo.skeleton import batch_rigid_transform

    skeleton = build_native_representation().skeleton
    surfaces, _, _, _ = official_sole_surfaces(skeleton)
    skinner = load_official_skinner(skeleton)
    local = torch.eye(3).repeat(2, 30, 1, 1)
    root = torch.tensor([[0., .02, 0.], [0., .02, 0.]])
    joints, _ = batch_rigid_transform(local, skeleton.neutral_joints[None].repeat(2, 1, 1),
                                      skeleton.joint_parents, skeleton.root_idx)
    joints = joints + root[:, None]
    before, indices, _ = stairs._skin_sole_regions(local, joints, skinner, surfaces)
    floor_shift = .017
    posed = joints.clone()
    posed[..., 1] -= floor_shift
    after = {side: points - np.array([0., floor_shift, 0.], dtype=np.float32)
             for side, points in before.items()}
    initial = {"init_heading_quat": torch.tensor([1., 0., 0., 0.]),
               "root_pos_init_xz": torch.zeros(2)}
    _, parity = stairs._skin_canonical_regions(local, posed, skinner, indices, after, initial)
    assert parity < 1e-6
    with pytest.raises(ValueError, match="parity exceeds"):
        stairs._skin_canonical_regions(local, posed, skinner, indices, before, initial)


def test_full_soma77_skinning_converts_numpy_pose_positions_to_tensors():
    import torch

    class TorchOnlySkinner:
        def __init__(self):
            self.bind_vertices = torch.zeros(200, 3)
            self.lbs_indices = torch.zeros(200, 1, dtype=torch.long)
            self.lbs_weights = torch.ones(200, 1)

        def skin(self, rotations, positions, rot_is_global=False):
            assert isinstance(rotations, torch.Tensor)
            assert isinstance(positions, torch.Tensor)
            assert rotations.dtype == positions.dtype == torch.float32
            shift = positions[:, :1].mean(dim=1)[:, None, :]
            return self.bind_vertices[None] + shift

    surfaces = [
        {"channel": "LeftFoot", "selected_indices": np.arange(0, 98)},
        {"channel": "LeftToeBase", "selected_indices": np.arange(0, 98)},
        {"channel": "RightFoot", "selected_indices": np.arange(102, 200)},
        {"channel": "RightToeBase", "selected_indices": np.arange(102, 200)},
    ]
    local = np.broadcast_to(np.eye(3, dtype=np.float32), (3, 77, 3, 3)).copy()
    posed = np.zeros((3, 77, 3), dtype=np.float32)
    soles, indices, parity = stairs._skin_sole_regions(local, posed, TorchOnlySkinner(), surfaces)
    assert parity == 0
    assert indices["left"].shape == indices["right"].shape == (98,)
    assert soles["left"].shape == soles["right"].shape == (3, 98, 3)
