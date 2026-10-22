import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from scripts import review_parc_tokenizer_terrain as exporter


def _rotation_y(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float64)


def _valid_review_arrays():
    identity = np.broadcast_to(np.eye(3, dtype=np.float32), (64, 30, 3, 3)).copy()
    roots = np.zeros((64, 3), dtype=np.float32)
    return roots, identity, roots.copy(), identity.copy(), np.zeros((7, 9), dtype=np.float32), np.zeros(3), .1


def test_world_transform_rotates_only_root_local_orientation_and_moves_roots():
    roots, local, _, _, _, _, _ = _valid_review_arrays()
    local[:, 4] = np.broadcast_to(_rotation_y(.2), (64, 3, 3))
    transform = {"rotation": _rotation_y(np.pi / 2), "translation": np.asarray([2, 1, -3])}
    expected_joints = np.broadcast_to(roots[:, None], (64, 30, 3)).copy()
    world_roots, world_local, world_joints = exporter.transform_motion_to_world(
        expected_joints, local, transform, root_index=0)
    np.testing.assert_allclose(world_roots, np.broadcast_to([2, 1, -3], (64, 3)), atol=1e-6)
    np.testing.assert_allclose(
        world_local[:, 0], np.broadcast_to(transform["rotation"], (64, 3, 3)), atol=1e-6)
    np.testing.assert_allclose(world_local[:, 4], local[:, 4], atol=1e-6)
    np.testing.assert_allclose(
        world_joints[:, 12], np.broadcast_to([2, 1, -3], (64, 3)), atol=1e-6)


def test_world_coordinate_arrays_reproduce_transformed_soma_fk():
    skeleton = SOMASkeleton30()
    rng = np.random.default_rng(7)
    local = np.broadcast_to(np.eye(3, dtype=np.float32), (64, 30, 3, 3)).copy()
    angles = rng.normal(0, .08, size=(64, 30))
    for frame in range(64):
        for joint in range(30):
            local[frame, joint] = _rotation_y(angles[frame, joint]).astype(np.float32)
    roots = np.column_stack((np.linspace(-.4, .8, 64), np.full(64, .92),
                             np.linspace(.2, 1.3, 64))).astype(np.float32)
    _, source_positions, _ = skeleton.fk(torch.from_numpy(local), torch.from_numpy(roots))
    transform = {"rotation": _rotation_y(.71), "translation": np.asarray([1.2, .3, -2.4])}
    world_roots, world_local, expected_world_positions = exporter.transform_motion_to_world(
        source_positions.numpy(), local, transform, skeleton.root_idx)
    error = exporter._verify_fk_roundtrip(
        skeleton, world_roots, world_local, expected_world_positions)
    assert error < 2e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is unavailable")
def test_fk_roundtrip_uses_device_of_loaded_checkpoint_skeleton():
    skeleton = SOMASkeleton30().to("cuda")
    roots, local, _, _, _, _, _ = _valid_review_arrays()
    _, positions, _ = skeleton.fk(torch.from_numpy(local).cuda(), torch.from_numpy(roots).cuda())
    assert exporter._verify_fk_roundtrip(skeleton, roots, local, positions.cpu().numpy()) < 2e-4


@pytest.mark.parametrize("field,value", [
    ("roots_source", np.zeros((63, 3))),
    ("local_codec", np.zeros((64, 29, 3, 3))),
    ("min_point", np.zeros((1, 3))),
])
def test_review_archive_rejects_wrong_dimensions(field, value):
    arrays = list(_valid_review_arrays())
    positions = {"roots_source": 0, "local_codec": 3, "min_point": 5}
    arrays[positions[field]] = value
    with pytest.raises(ValueError, match="shape"):
        exporter.validate_review_arrays(*arrays)


def test_review_window_selection_is_deterministic_bounded_and_family_stratified():
    records = [
        {"shard": "a", "split": "validation", "terrain_family": "stairs", "source_group": "g1"},
        {"shard": "b", "split": "validation", "terrain_family": "stairs", "source_group": "g2"},
        {"shard": "c", "split": "validation", "terrain_family": "ramp", "source_group": "g3"},
        {"shard": "d", "split": "validation", "terrain_family": "ramp", "source_group": "g4"},
    ]
    windows = [
        {"shard": row["shard"], "start": start, "behavior": row["terrain_family"]}
        for row in records for start in (0, 8, 16)
    ]
    first = exporter.select_review_windows(windows, records, 4, "manifest-sha")
    second = exporter.select_review_windows(windows, records, 4, "manifest-sha")
    assert first == second
    assert len(first) == 4
    assert {row["terrain_family"] for row in first} == {"stairs", "ramp"}
    assert len(exporter.select_review_windows(windows, records, 8, "manifest-sha")) == 8
    with pytest.raises(ValueError, match="1 to 8"):
        exporter.select_review_windows(windows, records, 9, "manifest-sha")


def test_review_archive_contract_has_native_heightfield_orientation():
    roots, local, roots_codec, local_codec, hf, min_point, dx = _valid_review_arrays()
    hf[:] = np.arange(63, dtype=np.float32).reshape(7, 9)
    normalized = exporter.validate_review_arrays(
        roots, local, roots_codec, local_codec, hf, min_point, dx)
    assert normalized["hf"].shape == (7, 9)
    assert normalized["min_point"].shape == (3,)
    assert normalized["min_point"][1] == 0
    assert normalized["fps"].item() == 30


def test_metrics_separate_root_translation_from_root_relative_pose_error():
    class IdentityRep:
        fps = 30.0
        skeleton = SimpleNamespace(root_idx=0)

        @staticmethod
        def normalize(features):
            return features

    model = SimpleNamespace(motion_rep=IdentityRep())
    source_joints = np.zeros((64, 30, 3), dtype=np.float32)
    source_joints[:, :, 1] = np.arange(30, dtype=np.float32)[None] * .02
    codec_joints = source_joints.copy()
    codec_joints[:, :, 0] += .17
    source_rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (64, 30, 3, 3)).copy()
    codec_rotations = source_rotations.copy()
    source_features = torch.zeros((1, 64, 2))
    codec_features = torch.full((1, 64, 2), .1)

    metrics = exporter._window_metrics(
        model, source_features, codec_features,
        source_joints, source_rotations, codec_joints, codec_rotations)
    assert metrics["root_error_m"] == pytest.approx(.17, abs=1e-6)
    assert metrics["root_relative_fk_mpjpe_m"] == pytest.approx(0, abs=1e-7)
    assert metrics["local_joint_rotation_geodesic_rad"] == pytest.approx(0, abs=1e-7)
    assert metrics["nonroot_local_rotation_geodesic_rad"] == pytest.approx(0, abs=1e-7)
