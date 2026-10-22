import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.protomotions_policy import _heading, reference_window
from motionbricks.mb_sonic.reference import G1Reference, G1_JOINT_NAMES, ReferenceCoverageError


def reference():
    times = np.arange(60)/50
    qpos = np.zeros((60, 36))
    qpos[:, 3] = 1
    rotations = np.zeros((60, 1, 4))
    rotations[..., 0] = 1
    return G1Reference(times, qpos, np.zeros((60, 35)), ("torso_link",),
        np.zeros((60, 1, 3)), rotations, np.zeros((60, 1, 3)), np.zeros((60, 1, 3)),
        "000", "test", times[-1], joint_names=G1_JOINT_NAMES)


def test_exact_future_grid_preserves_samples():
    source = reference()
    window = reference_window(source, .2, [0, .02, .04, .08, .16])
    np.testing.assert_array_equal(window.qpos, source.qpos[[10, 11, 12, 14, 18]])
    np.testing.assert_allclose(window.timestamps_s, [.2, .22, .24, .28, .36])


def test_no_tail_or_off_grid_reference():
    with pytest.raises(ReferenceCoverageError):
        reference_window(reference(), 1.1, [0, .02, .04, .08, .16])
    with pytest.raises(ReferenceCoverageError):
        reference_window(reference(), .211, [0, .02, .04, .08, .16])


def test_heading_uses_forward_yaw_not_full_body_rotation():
    rot = Rotation.from_euler("xyz", [.4, -.3, 1.2])
    heading = _heading(rot.as_quat()[[3, 0, 1, 2]])
    np.testing.assert_allclose(heading.as_rotvec(), [0, 0, 1.2], atol=1e-14)
