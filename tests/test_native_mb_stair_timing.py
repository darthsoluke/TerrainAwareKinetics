import numpy as np
import pytest

from scripts.review_native_mb_stair_timing import acceleration_by_join


def test_join_diagnostic_separates_a_step_from_smooth_motion():
    joints = np.zeros((12, 2, 3), dtype=np.float64)
    joints[..., 0] = np.arange(12)[:, None] / 30
    joints[6:, :, 1] += .02
    scores = acceleration_by_join(joints, [6])
    assert scores['join_center_frames'] == 2
    assert scores['join_mps2']['mean'] == pytest.approx(.02 * 30 ** 2)
    assert scores['interior_mps2']['max'] < 1e-10


def test_no_join_is_unknown_not_zero_evidence():
    scores = acceleration_by_join(np.zeros((8, 2, 3)), [])
    assert scores['join_mps2']['count'] == 0
    assert scores['join_mps2']['mean'] is None
    assert scores['interior_mps2']['count'] == 12
