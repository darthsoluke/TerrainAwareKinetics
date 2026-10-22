import numpy as np
import pytest
import json

from scripts.review_pfnn_ablation_fixed_masks import motion_measures, loss_component_probe
from scripts.compare_arms import read_arms


def test_constant_position_bias_does_not_imply_constant_velocity():
    source = np.zeros((5, 4, 3))
    prediction = source.copy()
    prediction[..., 0] = .01
    result = motion_measures(prediction, source, source[0], np.ones((6, 4), bool), list(range(4)))
    assert np.median(result['stance_error']) == pytest.approx(.01)
    assert np.median(result['source_mask_speed']) == 0
    assert np.max(result['source_mask_speed']) == pytest.approx(.3)


def test_source_mask_excludes_contact_onset_and_release_intervals():
    source = np.zeros((3, 4, 3))
    prediction = source.copy()
    prediction[:, :, 0] = np.array([.01, .02, .03])[:, None]
    contacts = np.array([[False] * 4, [True] * 4, [True] * 4, [False] * 4])
    result = motion_measures(prediction, source, source[0], contacts, list(range(4)))
    assert len(result['source_mask_speed']) == 4
    np.testing.assert_allclose(result['source_mask_speed'], .3)


def test_boundary_term_is_included_in_coordinate_loss_probe():
    row = next(r for r in loss_component_probe() if r['axes'] == 3 and r['coordinate_error_m'] == .01)
    assert row['weighted_body'] == pytest.approx(.002666666666666667)
    assert row['weighted_boundary'] == pytest.approx(.006)
    assert row['weighted_sum'] == pytest.approx(.008666666666666667)


def test_wrong_clock_is_rejected():
    with pytest.raises(ValueError, match='clocks'):
        motion_measures(np.zeros((3, 4, 3)), np.zeros((3, 4, 3)), np.zeros((4, 3)),
                        np.ones((3, 4), bool), list(range(4)))


def test_compare_requires_closed_loop_result(tmp_path):
    directory = tmp_path / 'arm_C'
    directory.mkdir()
    (directory / 'tf_strat.json').write_text('{}')
    with pytest.raises(FileNotFoundError):
        read_arms(tmp_path, ['C'])


def test_compare_rejects_inconsistent_exposure(tmp_path):
    directory = tmp_path / 'arm_A'
    directory.mkdir()
    (directory / 'tf_strat.json').write_text('{}')
    (directory / 'cl_strat.json').write_text('{"summary": {}}')
    (directory / 'results.json').write_text(json.dumps(dict(updates=12, window_exposures=24575,
        plan=dict(updates=12, batch=2048))))
    with pytest.raises(ValueError, match='inconsistent'):
        read_arms(tmp_path, ['A'])
