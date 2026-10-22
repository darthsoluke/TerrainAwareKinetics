"""Selection guards for official correction; no synthetic quality approval."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from correct_parc_candidates import selection


CONFIG = {'max_contact_loss': 3., 'max_pen_loss': 8., 'max_total_loss': 30.}


def test_released_filter_and_completion_are_both_required():
    row = {'contact_loss': 1., 'penetration_loss': 2., 'total_loss': 3., 'completed': True}
    assert selection(row, CONFIG)
    assert not selection(dict(row, completed=False), CONFIG)
    for key, limit in [('contact_loss', 3.), ('penetration_loss', 8.), ('total_loss', 30.)]:
        assert not selection(dict(row, **{key: limit}), CONFIG)


def test_nonfinite_scores_fail_closed():
    with pytest.raises(ValueError, match='nonfinite'):
        selection({'contact_loss': float('nan'), 'penetration_loss': 0., 'total_loss': 0., 'completed': True}, CONFIG)
