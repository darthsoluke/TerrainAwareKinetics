"""Trainer bookkeeping that must hold without a GPU."""

import pytest

from scripts.train_terrain_controller import parse_phases, phase_position


def test_phase_parsing():
    assert parse_phases(["teacher:5", "loop:4:7"]) == [("teacher", 0, 5), ("loop", 4, 7)]
    with pytest.raises(ValueError):
        parse_phases(["loop:4"])


def test_resume_position_follows_the_update_counter():
    phases = parse_phases(["teacher:5", "loop:4:7"])
    assert phase_position(0, phases) == (0, 0)
    assert phase_position(5, phases) == (1, 0)
    assert phase_position(10, phases) == (1, 5)          # smoke ran teacher:5 loop:4:5, resume extends loop to 7
    assert phase_position(12, phases) == (2, 0)          # everything done
    assert phase_position(10, parse_phases(["teacher:5", "loop:4:5"])) == (2, 0)
