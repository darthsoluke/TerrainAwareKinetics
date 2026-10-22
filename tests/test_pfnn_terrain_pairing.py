from pathlib import Path

import numpy as np
import pytest

from motionbricks.data.pfnn_terrain_pairing import author_functions, fit_terrain, static_heightfield, terrain_kind


@pytest.fixture
def author():
    path = Path(__file__).resolve().parents[1]/'out/pfnn_mixed_20261015/source_inventory_v3/official_source'
    if not (path/'generate_database.py').is_file():
        pytest.skip('Verified PFNN author source is not staged')
    return path


def test_author_patch_sampler_preserves_xz_axis_order(author):
    sample, _ = author_functions(author)
    x, z = np.meshgrid(np.arange(128), np.arange(128), indexing='ij')
    patch = (x+3*z)[None].astype(np.float32)
    points = np.array([[0., 0.], [3.937007874, 0.], [0., 3.937007874]])
    height = sample(patch, points)[0, :, 0]
    np.testing.assert_allclose(height-height[0], [0., 3., 9.], atol=1e-8)


def test_flat_pair_is_one_static_plane_and_does_not_move_motion(author):
    points = np.zeros((10, 4, 3))
    points[..., 1] = .02
    points[:, 0, 0] = np.linspace(0, 1, 10)
    original = points.copy()
    terrain = fit_terrain(np.zeros((1, 128, 128), np.float32), points,
        np.ones((10, 4), dtype=bool), np.zeros((10, 3)), 'flat', 1., author, 1234)
    result = static_heightfield(terrain, np.zeros((10, 3)))
    np.testing.assert_array_equal(points, original)
    np.testing.assert_allclose(result['heightfield_y_m'], .02)
    np.testing.assert_allclose(terrain(np.array([[0., 0.], [10., -10.]])), .02)


def test_released_kind_exceptions_are_preserved():
    assert terrain_kind('LocomotionFlat12_000_mirror') == 'jumpy'
    assert terrain_kind('WalkingUpSteps06_000') == 'beam'
    assert terrain_kind('WalkingUpSteps09_000_mirror') == 'flat'
    assert terrain_kind('WalkingUpSteps01_000') == 'rocky'
