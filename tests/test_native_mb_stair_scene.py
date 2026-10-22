import numpy as np
import pytest

from scripts.author_native_mb_stair_scene import source_sweep_treads
from scripts.review_native_mb_stair_live_contact import scene_clearance


def source_points():
    # These are source observations, independent of any model prediction.
    return np.array([[0., 0., -.10], [0., .20, .20], [0., .40, .50],
                     [0., .60, .80]])


@pytest.mark.parametrize('heading', [0., .7, np.pi])
def test_treads_obey_source_lower_vertex_constraints_for_any_heading(heading):
    points = source_points()
    axis = np.array([np.sin(heading), np.cos(heading)])
    points[:, [0, 2]] = points[:, 2, None] * axis
    treads = source_sweep_treads(points, axis, [.20, .40, .60])
    for tread in treads:
        lower = points[:, 1] < tread['height_m'] - .01
        assert np.all(points[lower][:, [0, 2]] @ axis < tread['ascending_start_m'])
    ue = np.stack([points[:, 2], -points[:, 0], points[:, 1]], -1) * 100
    gap = scene_clearance(ue, treads)
    np.testing.assert_allclose(gap[1:], 0., atol=1e-8)
    assert np.isnan(gap[0])  # The separate ground plane is not invented here.
    assert treads[-1]['ascending_stop_m'] - treads[-1]['ascending_start_m'] >= 1.2
    reverse = source_sweep_treads(points[::-1], axis, [.60, .40, .20])
    assert reverse == treads  # Descending temporal order cannot change geometry.


@pytest.mark.parametrize('kwargs', [{'tolerance_m': np.nan}, {'width_m': np.inf},
                                   {'landing_length_m': 0.}, {'width_m': -1.}])
def test_invalid_scene_parameters_fail_closed(kwargs):
    with pytest.raises(ValueError):
        source_sweep_treads(source_points(), [0., 1.], [.2, .4, .6], **kwargs)


def test_authoring_does_not_silently_widen_or_compress_stairs():
    points = source_points()
    points[0, 0] = .56
    with pytest.raises(ValueError, match='width'):
        source_sweep_treads(points, [0., 1.], [.2, .4, .6])
    points = source_points()
    points[1, 2] = -.05
    with pytest.raises(ValueError, match='8 cm'):
        source_sweep_treads(points, [0., 1.], [.2, .4, .6])
