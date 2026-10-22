"""CPU signal-analysis checks; no model, browser or physical execution."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.analyze_mb_reference_jitter import distribution, native_at, temporal_metrics


def test_constant_signal_has_no_jitter_or_spectral_peak():
    result = temporal_metrics(np.zeros((100, 29)))
    assert result['velocity_rms'] == result['acceleration_rms'] == result['jerk_rms'] == 0
    assert result['power_fraction_above_8hz'] == 0
    assert result['dominant_nonzero_frequency_hz'] is None


@pytest.mark.parametrize('frequency,high', [(1., False), (10., True)])
def test_frequency_measure_separates_gait_scale_and_fast_oscillation(frequency, high):
    time = np.arange(2048) * .02
    result = temporal_metrics(np.sin(2 * np.pi * frequency * time)[:, None])
    assert (result['power_fraction_above_8hz'] > .99) is high
    assert abs(result['dominant_nonzero_frequency_hz'] - frequency) < .1


def test_native_interpolation_uses_so3_and_removes_root_translation():
    angles = np.array([170., 180., 190.])
    rotations = Rotation.from_euler('y', angles, degrees=True).as_matrix()[:, None]
    root = np.array([[0., 1., 0.], [1., 1., 0.], [2., 1., 0.]])
    source = dict(timestamps_s=np.array([0., .1, .2]), bone_names=['Hips'],
                  local_rot_mats=rotations, world_positions_m=root[:, None])
    matrix, positions = native_at(source, .15)
    np.testing.assert_allclose(matrix[0], Rotation.from_euler('y', 185., degrees=True).as_matrix(), atol=1e-12)
    np.testing.assert_array_equal(positions, np.zeros((1, 3)))
    with pytest.raises(ValueError, match='outside'):
        native_at(source, .21)


def test_empty_distribution_is_not_a_fabricated_zero_measurement():
    assert distribution([]) is None
    assert distribution([1., 3.])['mean'] == 2.
