import numpy as np

from scripts.audit_joint_sdf_stair_treadmill import measure, phase_events


def test_phase_hysteresis_ignores_small_jitter():
    assert phase_events([0, .11, .01, -.01, .12, -.11, -.02, .02, .11]) == [1, 5, 8]


def test_slow_forward_motion_is_not_backward_seam_failure():
    roots = np.zeros((64, 3))
    roots[:, 0] = np.arange(64) * .001
    feet = np.repeat(roots[:, None], 2, axis=1)
    feet[:, 0, 0] += .3 * np.sin(np.arange(64) * np.pi / 8)
    feet[:, 1, 0] -= .3 * np.sin(np.arange(64) * np.pi / 8)
    result = measure(roots, feet, [1., 0.])
    assert result['backward_frames'] == []
    assert not result['recurrent_backward_replanning']
    assert result['cycles']
    assert all(c['net_forward_m'] > 0 for c in result['cycles'])


def test_backward_replanning_is_detected_at_original_frames():
    roots = np.zeros((64, 3))
    roots[:, 0] = np.arange(64) * .02 - (np.arange(64) // 8) * .1
    feet = np.repeat(roots[:, None], 2, axis=1)
    result = measure(roots, feet, [1., 0.])
    assert result['backward_frames'] == list(range(8, 64, 8))
    assert result['backward_seam_frames'] == result['backward_frames']
    assert result['recurrent_backward_replanning']
    assert abs(result['backward_m'] - .56) < 1e-12
    assert result['cycles'] == []
