import io
import pickle

import numpy as np
import pytest
import torch

from motionbricks.data.local_terrain_observation import LocalTerrainObservationConfig
from motionbricks.data.parc_paired_source import (
    _ArrayUnpickler,
    _restricted_load,
    _ROOT_BASIS_QUAT_XYZW,
    _make_observation,
    _quat_multiply_xyzw,
    load_parc_paired_source,
)


@pytest.mark.parametrize('module,name', [('multiarray', '_reconstruct'),
                                        ('multiarray', 'scalar'), ('numeric', '_frombuffer')])
def test_restricted_numpy_one_and_two_names_share_only_allowed_primitives(module, name):
    reader = _ArrayUnpickler(io.BytesIO())
    assert reader.find_class('numpy.core.' + module, name) is reader.find_class('numpy._core.' + module, name)
    with pytest.raises(pickle.UnpicklingError):
        reader.find_class('numpy._core.' + module, 'load')


def test_restricted_numpy_scalar_and_array_pickle_roundtrip():
    source = {'scalar': np.float32(1.5), 'array': np.arange(6, dtype=np.float32).reshape(2, 3)}
    actual = _restricted_load(pickle.dumps(source, protocol=5))
    assert actual['scalar'] == source['scalar']
    np.testing.assert_array_equal(actual['array'], source['array'])


def _write_sample(path, *, roots=None, contacts=None, masks="default"):
    frames = 3
    roots = np.asarray(roots if roots is not None else [
        [0, 0, 1], [0.2, 0.1, 1.2], [0.4, 0.2, 1.4]
    ], dtype=np.float32)
    contacts = np.asarray(contacts if contacts is not None else np.zeros((frames, 15)), dtype=np.float32)
    if masks == "default":
        masks = [np.array([[x, y] for x in range(4) for y in range(5)], dtype=np.int64)] * frames
    motion = {
        "root_pos": roots,
        "root_rot": np.tile([0, 0, 0, 1], (frames, 1)).astype(np.float32),
        "joint_rot": np.tile([0, 0, 0, 1], (frames, 14, 1)).astype(np.float32),
        "body_contacts": contacts,
        "fps": 30,
        "loop_mode": "CLAMP",
    }
    hf = np.arange(20, dtype=np.float32).reshape(4, 5)
    terrain = {"hf": hf, "hf_maxmin": np.zeros((4, 5, 2), np.float32),
               "min_point": np.array([-1.0, -2.0], np.float32), "dx": 0.5}
    misc = {} if masks is None else {"hf_mask_inds": masks}
    container = {key: pickle.dumps(value) for key, value in (
        ("motion_data", motion), ("terrain_data", terrain), ("misc_data", misc),
    )}
    path.write_bytes(pickle.dumps(container))
    return hf


def _commands(frames=3):
    return np.tile(np.array([0.25, 1.0], np.float32), (frames, 1))


def test_z_up_conversion_is_proper_rotation_and_preserves_local_rotations(tmp_path):
    source = tmp_path / "sample.pkl"
    _write_sample(source)
    result = load_parc_paired_source(source, _commands())

    np.testing.assert_allclose(np.linalg.det([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), 1.0)
    np.testing.assert_allclose(result.root_positions_world_xyz_m[1], [0.2, 1.2, -0.1])
    expected_root = _quat_multiply_xyzw(_ROOT_BASIS_QUAT_XYZW, [0, 0, 0, 1])
    np.testing.assert_allclose(result.root_rotations_world_xyzw[0], expected_root, atol=1e-6)
    assert result.frame_observations[0].world_query_xz[0, 1, 1, 1, 0] > 0
    np.testing.assert_allclose(
        result.local_joint_rotations_xyzw,
        np.tile([0, 0, 0, 1], (3, 14, 1)),
    )


def test_heightfield_axes_reverse_source_y_and_min_point_is_first_target_cell_center(tmp_path):
    source = tmp_path / "sample.pkl"
    hf = _write_sample(source)
    result = load_parc_paired_source(source, _commands())

    np.testing.assert_array_equal(result.heightfield_y_m, hf.T[::-1, :])
    # Target row 0 is source y=+2; target row 4 is source y=-2.
    np.testing.assert_allclose(result.min_point_world_xyz_m, [-1.0, 0.0, -0.0])
    assert result.dx_m == pytest.approx(0.5)
    np.testing.assert_allclose(result.heightfield_y_m[0, 2], hf[2, 4])


def test_flat_map_relative_heights_are_invariant_to_shared_world_height_shift():
    config = LocalTerrainObservationConfig(patch_height=3, patch_width=3,
                                          near_spacing_m=0.1, forward_spacing_m=0.1,
                                          forward_start_m=0.1)
    hf = np.full((15, 15), 1.25, dtype=np.float32)
    valid = np.ones_like(hf, dtype=bool)
    origin = np.array([-0.7, 0.0, -0.7], dtype=np.float32)
    q = np.array([0, 0, 0, 1], dtype=np.float32)
    command = np.array([0, 1], dtype=np.float32)
    first = _make_observation(hf, origin, 0.1, [0, 1.25, 0], q, command, config, valid)
    shifted = _make_observation(hf + 4.0, origin, 0.1, [0, 5.25, 0], q, command, config, valid)

    torch.testing.assert_close(first.points, shifted.points)
    torch.testing.assert_close(first.points[..., 1], torch.zeros_like(first.points[..., 1]))


def test_all_15_source_contacts_survive_including_both_hands(tmp_path):
    source = tmp_path / "sample.pkl"
    contacts = np.zeros((3, 15), dtype=np.float32)
    contacts[0, 5] = 1.0
    contacts[1, 8] = 0.75
    contacts[2, [5, 8, 11, 14]] = 1.0
    _write_sample(source, contacts=contacts)
    result = load_parc_paired_source(source, _commands())

    np.testing.assert_array_equal(result.source_body_contacts, contacts)
    assert result.contact_body_names[5] == "right_hand"
    assert result.contact_body_names[8] == "left_hand"
    assert result.contact_provenance == "source_labels_not_verified_physical_truth"
    assert not result.training_ready
    assert not result.retargeted_to_soma


def test_current_observation_ignores_future_commands_and_future_motion(tmp_path):
    source = tmp_path / "sample.pkl"
    _write_sample(source)
    commands = _commands()
    baseline = load_parc_paired_source(source, commands)
    commands_changed = commands.copy()
    commands_changed[1:] = [[-100, 2], [3, -20]]
    roots_changed = np.array([[0, 0, 1], [999, 999, 999], [-999, -999, -999]], np.float32)
    _write_sample(source, roots=roots_changed)
    changed = load_parc_paired_source(source, commands_changed)

    torch.testing.assert_close(baseline.frame_observations[0].points,
                               changed.frame_observations[0].points)
    torch.testing.assert_close(baseline.frame_observations[0].valid,
                               changed.frame_observations[0].valid)


def test_hf_mask_inds_are_optional_provenance_only_not_network_validity(tmp_path):
    source = tmp_path / "sample.pkl"
    masks = [np.array([[0, 0]], dtype=np.int64) for _ in range(3)]
    _write_sample(source, masks=masks)
    with_source_masks = load_parc_paired_source(source, _commands())
    assert with_source_masks.source_hf_mask_inds is not None

    _write_sample(source, masks=None)
    without_source_masks = load_parc_paired_source(source, _commands())
    assert without_source_masks.source_hf_mask_inds is None
    for left, right in zip(with_source_masks.frame_observations,
                           without_source_masks.frame_observations):
        torch.testing.assert_close(left.points, right.points)
        torch.testing.assert_close(left.valid, right.valid)

    _write_sample(source, masks=[np.empty((0, 2), dtype=np.int64) for _ in range(3)])
    changed_source_masks = load_parc_paired_source(source, _commands())
    for left, right in zip(without_source_masks.frame_observations,
                           changed_source_masks.frame_observations):
        torch.testing.assert_close(left.points, right.points)
        torch.testing.assert_close(left.valid, right.valid)


def test_explicit_sensor_mask_is_separate_and_masks_only_supplied_cells(tmp_path):
    source = tmp_path / "sample.pkl"
    _write_sample(source, masks=None)
    all_valid = load_parc_paired_source(source, _commands())
    sensor_mask = np.zeros((5, 4), dtype=bool)
    masked = load_parc_paired_source(source, _commands(), sensor_heightfield_valid=sensor_mask)
    assert all_valid.frame_observations[0].valid.any()
    assert not masked.frame_observations[0].valid.any()


def test_loader_rejects_unapproved_pickle_globals(tmp_path):
    source = tmp_path / "unsafe.pkl"
    source.write_bytes(pickle.dumps(eval))
    with pytest.raises(pickle.UnpicklingError, match="Unapproved pickle global"):
        load_parc_paired_source(source, np.zeros((1, 2), dtype=np.float32))
