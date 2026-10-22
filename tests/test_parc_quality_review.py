"""Regression checks for evidence interpretation, not motion-quality approval."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location('parc_quality_review', SCRIPTS / 'review_parc_pilot_quality.py')
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
finally:
    sys.path.pop(0)


def test_goal_overshoot_is_not_failure_to_ever_reach():
    positions = np.zeros((4, 2, 3))
    positions[:, :, 0] = np.array([-1., -.4, .1, .8])[:, None]
    evidence = review.path_evidence(positions, ['left_foot', 'right_foot'], [0, 0, 0], 30)
    assert evidence['ever_reached']
    assert evidence['first_reached_frame'] == 1
    assert evidence['left_goal_region_after_reaching']
    assert evidence['last_frame_distance_m'] == .8


def test_goal_radius_is_strict_and_three_dimensional():
    positions = np.zeros((3, 2, 3))
    positions[:, :, 2] = .5
    evidence = review.path_evidence(positions, ['left_foot', 'right_foot'], [0, 0, 0], 30)
    assert not evidence['ever_reached']
    assert evidence['first_reached_frame'] is None


def test_original_candidates_fail_penetration_even_without_path_penalty():
    config = {'max_contact_loss': 3., 'max_pen_loss': 8., 'max_total_loss': 30.}
    for contact, penetration in ((.1076567, 15.241385), (.0968708, 9.857176)):
        evidence = review.gate_evidence(contact, penetration, contact + penetration, config)
        assert evidence['failed'] == ['pen_loss']
        assert not evidence['passes_released_filter']


def test_released_filter_rejects_nonfinite_score_and_boundary():
    config = {'max_contact_loss': 3., 'max_pen_loss': 8., 'max_total_loss': 30.}
    assert review.gate_evidence(3., 0., 3., config)['failed'] == ['contact_loss']
    assert review.gate_evidence(0., 0., np.nan, config)['failed'] == ['total_loss']


def test_released_filter_keeps_low_scores_without_claiming_physics():
    config = {'max_contact_loss': 3., 'max_pen_loss': 8., 'max_total_loss': 30.}
    evidence = review.gate_evidence(1., 2., 3., config)
    assert evidence['passes_released_filter']
    assert 'not millimeters or physics certification' in evidence['scope']
