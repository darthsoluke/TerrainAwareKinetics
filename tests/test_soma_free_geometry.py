import pytest
import torch

from motionbricks.data.soma_free_geometry import observed_sole_geometry


def test_fast_near_ground_motion_is_not_selected_out():
    vertices = torch.zeros(20, 2, 3)
    vertices[:, :, 0] = torch.arange(20)[:, None] / 60 * .6
    report = observed_sole_geometry(vertices, 60)
    assert report["near_ground_material_speed_mps"]["mean"] == pytest.approx(.6)
    assert report["measured_interval_fraction"] == 1.


def test_no_near_ground_evidence_is_null_not_zero_slip():
    vertices = torch.zeros(20, 2, 3)
    vertices[..., 1] = .1
    report = observed_sole_geometry(vertices, 60)
    assert report["near_ground_material_speed_mps"] is None
    assert report["measured_intervals"] == 0 and report["clearance_frame_fraction"] == 1.


def test_penetration_and_scoring_context_are_not_hidden():
    vertices = torch.zeros(20, 2, 3)
    vertices[:4, :, 1] = -1.
    vertices[4:, :, 1] = -.02
    report = observed_sole_geometry(vertices, 60)
    assert report["penetration_m"]["max"] == pytest.approx(.02)
    assert report["scored_intervals"] == 15
