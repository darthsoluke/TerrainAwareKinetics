"""Tests for sole-region contact geometry and material-point slip."""

import numpy as np
import pytest
import torch

from motionbricks.data.bones_seed import build_representation, official_sole_surfaces
from motionbricks.data.sole_region_contact import (FRAME_STATES, INTERVAL_STATES, SOLE_CONTACT_CLEARANCE_M,
                                                   SOLE_MIN_CONTACT_RUN_FRAMES, SOLE_SLIP_SPEED_MPS,
                                                   material_point_slip, region_gap_m, sole_region_heights,
                                                   sole_region_indices, sole_region_labels)


@pytest.fixture(scope="module")
def surfaces():
    return official_sole_surfaces(build_representation().skeleton)[0]


def rolling_fixture(frames=12, footprints=((0, 5), (4, 9), (8, 11)), heights=(0.002, 0.05)):
    """A heel-to-toe roll: each material vertex touches down in its own run.

    Height is low inside the vertex's own run and high outside it, so the identity of
    the lowest vertex changes twice while no vertex ever moves horizontally.
    """
    near, far = heights
    vertices = len(footprints)
    height = torch.full((frames, vertices), far, dtype=torch.float64)
    for index, (first, last) in enumerate(footprints):
        height[first:last + 1, index] = near
    horizontal = torch.zeros(frames, vertices, 2, dtype=torch.float64)
    horizontal[:, 0, 0] = 0.0
    horizontal[:, 1, 0] = 0.05
    horizontal[:, 2, 0] = 0.10
    return height, horizontal


def test_region_indices_are_fixed_numbered_material_vertices(surfaces):
    left = sole_region_indices(surfaces, "Left")
    right = sole_region_indices(surfaces, "Right")
    assert left.size == right.size == 98
    assert np.array_equal(left, np.unique(left)) and np.array_equal(right, np.unique(right))
    assert not set(left.tolist()) & set(right.tolist())


def test_region_gap_is_the_lowest_vertex_and_penetration_its_negative():
    height = torch.tensor([[0.01, -0.004], [0.02, 0.003]], dtype=torch.float64)
    result = region_gap_m(height)
    torch.testing.assert_close(result["region_gap_m"], torch.tensor([-0.004, 0.003], dtype=torch.float64))
    torch.testing.assert_close(result["region_penetration_m"], torch.tensor([0.004, 0.0], dtype=torch.float64))


def test_a_heel_to_toe_roll_does_not_create_slip():
    """The lowest vertex changes identity, yet no material point slides."""
    height, horizontal = rolling_fixture()
    slip = material_point_slip(height, horizontal, 60.0)
    assert int(slip["measured_vertex_count"].sum()) > 0
    assert float(slip["region_slip_mps"].max()) == 0.0
    lowest = horizontal[torch.arange(height.shape[0]), height.argmin(dim=-1)]
    naive = (lowest[1:] - lowest[:-1]).norm(dim=-1) * 60.0
    # The naive per-frame lowest point jumps when contact hands over, which would be
    # reported as metres per second of skate.
    assert float(naive.max()) > 1.0
    assert int((naive > 1.0).sum()) == 2


def test_the_same_material_points_do_report_real_sliding():
    height, horizontal = rolling_fixture()
    sliding = horizontal.clone()
    sliding[1:, :, 0] += 0.001 * torch.arange(1, height.shape[0], dtype=torch.float64)[:, None]
    slip = material_point_slip(height, sliding, 60.0)
    assert float(slip["region_slip_mps"].max()) == pytest.approx(0.06, abs=1e-9)


def test_a_single_frame_dip_during_swing_cannot_manufacture_slip():
    frames, vertices = 9, 2
    height = torch.full((frames, vertices), 0.05, dtype=torch.float64)
    height[4, 0] = 0.002
    horizontal = torch.zeros(frames, vertices, 2, dtype=torch.float64)
    horizontal[4:, 0, 0] = 0.05
    slip = material_point_slip(height, horizontal, 60.0)
    assert int(slip["measured_vertex_count"].sum()) == 0
    assert float(slip["region_slip_mps"].max()) == 0.0
    assert int((slip["contact_run_id"] >= 0).sum()) == 1


def test_contact_runs_need_a_minimum_length_before_they_carry_slip():
    frames, vertices = 6, 1
    height = torch.full((frames, vertices), 0.002, dtype=torch.float64)
    horizontal = torch.zeros(frames, vertices, 2, dtype=torch.float64)
    horizontal[:, 0, 0] = 0.01 * torch.arange(frames, dtype=torch.float64)
    short = material_point_slip(height[:SOLE_MIN_CONTACT_RUN_FRAMES - 1], horizontal[:SOLE_MIN_CONTACT_RUN_FRAMES - 1], 60.0)
    assert int(short["measured_vertex_count"].sum()) == 0
    long = material_point_slip(height, horizontal, 60.0)
    assert int(long["measured_vertex_count"].sum()) > 0


def test_a_vertex_descending_through_the_band_is_not_a_planted_point():
    """Regression: landing and take-off frames were read as metres per second of slip."""
    height = torch.tensor([[0.0095], [0.008], [0.006], [0.004], [0.002], [0.001]], dtype=torch.float64)
    horizontal = torch.zeros(6, 1, 2, dtype=torch.float64)
    horizontal[:, 0, 0] = 0.02 * torch.arange(6, dtype=torch.float64)
    gated = material_point_slip(height, horizontal, 60.0)
    assert int(gated["measured_vertex_count"].sum()) == 0
    assert float(gated["region_slip_mps"].max()) == 0.0
    ungated = material_point_slip(height, horizontal, 60.0, rest_vertical_speed_mps=10.0)
    assert float(ungated["region_slip_mps"].max()) == pytest.approx(1.2, abs=1e-9)


def test_a_resting_run_still_reports_its_displacement():
    frames, vertices = 10, 1
    height = torch.full((frames, vertices), 0.005, dtype=torch.float64)
    horizontal = torch.zeros(frames, vertices, 2, dtype=torch.float64)
    horizontal[:, 0, 0] = 0.001 * torch.arange(frames, dtype=torch.float64)
    slip = material_point_slip(height, horizontal, 60.0)
    assert int(slip["measured_vertex_count"].sum()) > 0
    assert float(slip["region_slip_mps"].max()) == pytest.approx(0.06, abs=1e-9)


def test_an_unmeasured_contact_interval_is_undecided_not_stable():
    """Regression: 282 of 835 nominally stable intervals carried no slip evidence."""
    gap = torch.zeros(4, dtype=torch.float64)
    slip = {"region_slip_mps": torch.zeros(3, dtype=torch.float64),
            "region_measured": torch.tensor([True, False, True])}
    labels = sole_region_labels(gap, slip)
    codes = [labels["interval_state_names"][int(value)] for value in labels["interval_state"]]
    assert codes == ["stable_contact", "undecided", "stable_contact"]


def test_a_descent_that_cannot_be_measured_is_not_reported_as_stable():
    height = torch.tensor([[0.0095], [0.008], [0.006], [0.004], [0.002], [0.001]], dtype=torch.float64)
    horizontal = torch.zeros(6, 1, 2, dtype=torch.float64)
    horizontal[:, 0, 0] = 0.02 * torch.arange(6, dtype=torch.float64)
    slip = material_point_slip(height, horizontal, 60.0)
    gap = height.amin(dim=-1)
    labels = sole_region_labels(gap, slip)
    assert not bool(slip["region_measured"].any())
    assert all(labels["interval_state_names"][int(value)] == "undecided"
               for value in labels["interval_state"])


def test_a_landing_does_not_hide_the_sliding_that_follows_it():
    """Regression: one descent used to veto the whole contact run."""
    frames = 12
    height = torch.zeros(frames, 1, dtype=torch.float64)
    height[:4, 0] = torch.tensor([0.009, 0.007, 0.005, 0.004], dtype=torch.float64)
    horizontal = torch.zeros(frames, 1, 2, dtype=torch.float64)
    horizontal[4:, 0, 0] = 0.01 * torch.arange(frames - 4, dtype=torch.float64)
    slip = material_point_slip(height, horizontal, 60.0)
    assert not bool(slip["region_measured"][:4].any())
    assert bool(slip["region_measured"][4:].all())
    assert float(slip["region_slip_mps"][4:].max()) == pytest.approx(0.6, abs=1e-9)


def test_labels_refuse_slip_without_a_measured_mask():
    gap = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="region_measured"):
        sole_region_labels(gap, {"region_slip_mps": torch.zeros(2, dtype=torch.float64)})


def test_frame_labels_keep_the_undecided_band_apart():
    gap = torch.tensor([0.0, 0.005, 0.02, 0.05], dtype=torch.float64)
    slip = {"region_slip_mps": torch.zeros(3, dtype=torch.float64),
            "region_measured": torch.ones(3, dtype=torch.bool)}
    labels = sole_region_labels(gap, slip)
    assert labels["frame_counts"] == {"contact": 2, "undecided": 1, "clearance": 1}
    assert labels["frame_state_names"] == FRAME_STATES
    assert int(labels["frame_state"][2]) == FRAME_STATES.index("undecided")


def test_interval_labels_separate_transitions_and_undecided_frames():
    gap = torch.tensor([0.0, 0.0, 0.02, 0.0, 0.05, 0.05], dtype=torch.float64)
    slip = {"region_slip_mps": torch.zeros(5, dtype=torch.float64),
            "region_measured": torch.ones(5, dtype=torch.bool)}
    labels = sole_region_labels(gap, slip)
    names = labels["interval_state_names"]
    codes = [names[int(value)] for value in labels["interval_state"]]
    assert codes == ["stable_contact", "undecided", "undecided", "contact_transition", "clearance"]
    assert labels["interval_state_names"] == INTERVAL_STATES


def test_slip_above_the_threshold_is_sliding_contact():
    gap = torch.zeros(4, dtype=torch.float64)
    for speed, expected in ((SOLE_SLIP_SPEED_MPS * 0.5, "stable_contact"),
                            (SOLE_SLIP_SPEED_MPS * 2.0, "sliding_contact")):
        slip = {"region_slip_mps": torch.full((3,), speed, dtype=torch.float64),
                "region_measured": torch.ones(3, dtype=torch.bool)}
        labels = sole_region_labels(gap, slip)
        assert labels["interval_state_names"][int(labels["interval_state"][0])] == expected


def test_inverted_or_unsorted_thresholds_are_rejected():
    gap = torch.zeros(3, dtype=torch.float64)
    slip = {"region_slip_mps": torch.zeros(2, dtype=torch.float64),
            "region_measured": torch.ones(2, dtype=torch.bool)}
    with pytest.raises(ValueError, match="contact clearance"):
        sole_region_labels(gap, slip, contact_clearance_m=0.05, clearance_m=0.01)
    with pytest.raises(ValueError, match="FPS"):
        material_point_slip(torch.zeros(3, 1, dtype=torch.float64), torch.zeros(3, 1, 2, dtype=torch.float64), 0.0)
    with pytest.raises(ValueError, match="at least two frames"):
        material_point_slip(torch.zeros(3, 1, dtype=torch.float64), torch.zeros(3, 1, 2, dtype=torch.float64),
                            60.0, min_run_frames=1)


def test_region_heights_reject_a_vertex_index_outside_the_mesh():
    with pytest.raises(ValueError, match="vertices that exist"):
        sole_region_heights(torch.zeros(2, 5, 3, dtype=torch.float64), np.array([7]))
