import numpy as np
from scipy.spatial.transform import Rotation

from motionbricks.data.pfnn_soma_retarget import source_fk, sample_source_motion


def test_source_fk_composes_parent_rotation_and_child_translation():
    local = np.tile(np.eye(3), (2, 3, 1, 1))
    local[0, 0] = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    local[0, 1] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    roots = np.array([[1., 2., 3.], [0., 0., 0.]])
    offsets = np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])
    rotations, points = source_fk(local, roots, offsets, np.array([-1, 0, 1]))
    np.testing.assert_allclose(points[0, 1], [2., 2., 3.], atol=1e-12)
    np.testing.assert_allclose(points[0, 2], [2., 3., 3.], atol=1e-12)
    np.testing.assert_allclose(points[1, 2], [1., 0., 1.], atol=1e-12)
    np.testing.assert_allclose(rotations[0, 1], local[0, 0]@local[0, 1], atol=1e-12)


def test_resampling_preserves_original_quarter_rate_samples_during_fast_rotation():
    frames = 97
    local = Rotation.from_euler('y', np.arange(frames)*np.pi/2).as_matrix()[:, None]
    roots = np.stack((np.sin(np.arange(frames)), np.arange(frames)*.01, np.zeros(frames)), -1)
    actual, positions = sample_source_motion(local, roots, 120., 30.)
    np.testing.assert_allclose(actual, local[::4], atol=1e-6)
    np.testing.assert_allclose(positions, roots[::4], atol=1e-6)
    assert np.isfinite(actual).all()
