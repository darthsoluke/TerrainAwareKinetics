"""Tests for differentiable sole-region geometry supervision."""

import numpy as np
import pytest
import torch

from motionbricks.data.bones_seed import build_representation, official_sole_surfaces
from motionbricks.data.sole_geometry_losses import (frame_has_evidence, region_geometry_from_motion,
                                                    sole_geometry_supervision, stance_displacement_loss)
from motionbricks.data.sole_region_contact import FRAME_STATES, INTERVAL_STATES, sole_region_indices
from motionbricks.data.soma_skin_reference import load_official_skinner, skinned_vertices

DTYPE = torch.float64
CONTACT = FRAME_STATES.index("contact")
UNDECIDED = FRAME_STATES.index("undecided")
STABLE = INTERVAL_STATES.index("stable_contact")
SLIDING = INTERVAL_STATES.index("sliding_contact")
INTERVAL_UNDECIDED = INTERVAL_STATES.index("undecided")


@pytest.fixture(scope="module")
def rig():
    skeleton = build_representation().skeleton
    surfaces, _, _, _ = official_sole_surfaces(skeleton)
    return {"skeleton": skeleton, "skinner": load_official_skinner(skeleton),
            "indices": sole_region_indices(surfaces, "Left")}


def labels(frame_states, interval_states, measured, vertex_measured):
    return {"frame_state": torch.tensor(frame_states, dtype=torch.long),
            "interval_state": torch.tensor(interval_states, dtype=torch.long),
            "region_measured": torch.tensor(measured, dtype=torch.bool),
            "vertex_measured": torch.tensor(vertex_measured, dtype=torch.bool)}


def geometry(frames, vertices, height=0.0, speed=0.0):
    heights = torch.full((frames, vertices), height, dtype=DTYPE)
    positions = torch.zeros(frames, vertices, 2, dtype=DTYPE)
    positions[:, :, 0] = speed * torch.arange(frames, dtype=DTYPE)[:, None]
    return heights, positions


def test_frame_evidence_covers_the_frames_inside_a_measured_interval():
    measured = torch.tensor([False, True, False])
    assert torch.equal(frame_has_evidence(measured), torch.tensor([False, True, True, False]))
    with pytest.raises(ValueError, match="boolean"):
        frame_has_evidence(torch.ones(3))


def test_undecided_intervals_are_never_pinned_to_the_floor():
    frames = 6
    heights, positions = geometry(frames, 1, height=0.30)
    tags = labels([UNDECIDED] * frames, [INTERVAL_UNDECIDED] * (frames - 1), [False] * (frames - 1),
                  [[False]] * (frames - 1))
    result = sole_geometry_supervision(heights, positions, tags, 60.0)
    assert result["contact_frames_supervised"] == 0
    assert float(result["l_contact_height_m"]) == 0.0


def test_a_contact_frame_outside_any_measured_interval_is_not_supervised():
    frames = 5
    heights, positions = geometry(frames, 1, height=0.02)
    tags = labels([CONTACT] * frames, [INTERVAL_UNDECIDED] * (frames - 1), [False] * (frames - 1),
                  [[False]] * (frames - 1))
    result = sole_geometry_supervision(heights, positions, tags, 60.0)
    assert result["contact_frames_supervised"] == 0
    assert float(result["l_contact_height_m"]) == 0.0


def test_a_lifted_vertex_moving_fast_does_not_raise_the_slip_loss():
    """Regression: the loss took the max over the whole sole and penalised a lifted part."""
    frames = 6
    heights, positions = geometry(frames, 2, height=0.0, speed=0.0)
    positions[:, 1, 0] = 0.01 * torch.arange(frames, dtype=DTYPE)
    tag = labels([CONTACT] * frames, [STABLE] * (frames - 1), [True] * (frames - 1),
                 [[True, False]] * (frames - 1))
    result = sole_geometry_supervision(heights, positions, tag, 60.0)
    assert result["measured_stable_intervals_supervised"] == frames - 1
    assert float(result["l_measured_slip_mps"]) == 0.0
    assert result["measured_vertices_supervised"] == frames - 1


def test_a_measured_moving_vertex_does_raise_the_slip_loss():
    frames = 6
    heights, positions = geometry(frames, 2, height=0.0, speed=0.01)
    tag = labels([CONTACT] * frames, [STABLE] * (frames - 1), [True] * (frames - 1),
                 [[True, False]] * (frames - 1))
    result = sole_geometry_supervision(heights, positions, tag, 60.0)
    assert float(result["l_measured_slip_mps"]) == pytest.approx(0.6, abs=1e-9)


def test_unmeasured_stable_intervals_do_not_supervise_slip():
    frames = 6
    heights, positions = geometry(frames, 1, height=0.0, speed=0.02)
    tag = labels([CONTACT] * frames, [STABLE] * (frames - 1), [False] * (frames - 1),
                 [[False]] * (frames - 1))
    result = sole_geometry_supervision(heights, positions, tag, 60.0)
    assert result["measured_stable_intervals_supervised"] == 0
    assert float(result["l_measured_slip_mps"]) == 0.0
    assert result["contact_frames_supervised"] == 0


def test_sliding_intervals_do_not_get_the_stable_contact_treatment():
    frames = 4
    heights, positions = geometry(frames, 1, height=0.0, speed=0.01)
    tag = labels([CONTACT] * frames, [SLIDING] * (frames - 1), [True] * (frames - 1),
                 [[True]] * (frames - 1))
    result = sole_geometry_supervision(heights, positions, tag, 60.0)
    assert result["measured_stable_intervals_supervised"] == 0


def test_penetration_is_its_own_term_and_ignores_the_label_masks():
    frames = 3
    heights, positions = geometry(frames, 1, height=-0.02)
    tag = labels([UNDECIDED] * frames, [INTERVAL_UNDECIDED] * (frames - 1), [False] * (frames - 1),
                 [[False]] * (frames - 1))
    result = sole_geometry_supervision(heights, positions, tag, 60.0, penetration_tolerance_m=0.005)
    assert float(result["l_penetration_m"]) == pytest.approx(0.015, abs=1e-9)


def test_gradients_reach_the_prediction_and_not_the_labels():
    frames = 4
    heights, positions = geometry(frames, 1, height=0.01, speed=0.01)
    heights.requires_grad_(True)
    positions.requires_grad_(True)
    tag = labels([CONTACT] * frames, [STABLE] * (frames - 1), [True] * (frames - 1),
                 [[True]] * (frames - 1))
    result = sole_geometry_supervision(heights, positions, tag, 60.0)
    (result["l_contact_height_m"] + result["l_measured_slip_mps"] + result["l_penetration_m"]).backward()
    assert heights.grad is not None and positions.grad is not None
    assert all(not value.requires_grad for value in tag.values())


def test_the_losses_keep_separate_units():
    frames = 3
    heights, positions = geometry(frames, 1)
    tag = labels([CONTACT] * frames, [STABLE] * (frames - 1), [True] * (frames - 1),
                 [[True]] * (frames - 1))
    result = sole_geometry_supervision(heights, positions, tag, 60.0)
    assert set(result["units"]) == {"l_contact_height_m", "l_penetration_m", "l_measured_slip_mps"}


def test_missing_or_wrong_shaped_masks_are_rejected():
    heights, positions = geometry(3, 1)
    with pytest.raises(KeyError):
        sole_geometry_supervision(heights, positions, {"frame_state": torch.zeros(3, dtype=torch.long),
                                                       "interval_state": torch.zeros(2, dtype=torch.long)},
                                  60.0)
    tag = labels([CONTACT] * 3, [STABLE] * 2, [True] * 2, [[True]] * 2)
    tag["region_measured"] = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="boolean"):
        sole_geometry_supervision(heights, positions, tag, 60.0)
    tag = labels([CONTACT] * 3, [STABLE] * 2, [True] * 2, [[True]] * 2)
    with pytest.raises(ValueError, match="frames-1"):
        sole_geometry_supervision(heights, positions,
                                  dict(tag, vertex_measured=torch.ones(3, 1, dtype=torch.bool)), 60.0)


def test_a_toe_rotation_moves_the_region_by_the_official_skinned_amount(rig):
    """Regression: the whole sole used to be bound to one joint, so a toe rotation did nothing."""
    root = torch.zeros(1, 3, dtype=torch.float32)
    root[:, 1] = 0.9
    flat = torch.eye(3, dtype=torch.float32).expand(1, 30, 3, 3).contiguous()
    toe = rig["skeleton"].bone_index["LeftToeBase"]
    angle = torch.tensor(30.0 * np.pi / 180.0)
    rotated = flat.clone()
    rotated[:, toe] = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0],
                                    [torch.sin(angle), torch.cos(angle), 0.0],
                                    [0.0, 0.0, 1.0]])
    flat_height, _ = region_geometry_from_motion(flat, root, rig["skinner"], rig["indices"])
    toe_height, _ = region_geometry_from_motion(rotated, root, rig["skinner"], rig["indices"])
    assert float((toe_height - flat_height).abs().max()) > 0.01
    _, posed, _ = rig["skeleton"].fk(rotated, root)
    reference = skinned_vertices(rig["skinner"], rotated, posed)
    expected = reference.index_select(1, torch.as_tensor(rig["indices"]))[..., 1]
    torch.testing.assert_close(toe_height, expected, atol=1e-5, rtol=0)


def test_region_geometry_is_differentiable_with_respect_to_the_motion(rig):
    root = torch.zeros(1, 3, dtype=torch.float32)
    root[:, 1] = 0.9
    local = torch.eye(3, dtype=torch.float32).expand(1, 30, 3, 3).contiguous().requires_grad_(True)
    height, horizontal = region_geometry_from_motion(local, root, rig["skinner"], rig["indices"])
    (height.abs().mean() + horizontal.abs().mean()).backward()
    assert local.grad is not None and torch.isfinite(local.grad).all()
    assert float(local.grad.abs().sum()) > 0.0


def test_selected_official_lbs_matches_full_mesh_values_and_gradients(rig):
    root = torch.tensor([[0.2, 0.9, -0.1], [0.4, 1.0, 0.3]], requires_grad=True)
    local = torch.eye(3).expand(2, 30, 3, 3).clone().requires_grad_(True)
    height, horizontal = region_geometry_from_motion(local, root, rig["skinner"], rig["indices"])
    selected_grad = torch.autograd.grad(height.sum() + horizontal.sum(), (local, root))
    # The reference path invokes official skin() and its full-mesh lbs().
    posed = root[:, None].expand(-1, 30, -1)
    reference = rig["skinner"].skin(local, posed).index_select(1, torch.as_tensor(rig["indices"]))
    reference_grad = torch.autograd.grad(reference.sum(), (local, root))
    torch.testing.assert_close(height, reference[..., 1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(horizontal, reference[..., [0, 2]], atol=1e-6, rtol=1e-5)
    for selected, expected in zip(selected_grad, reference_grad):
        torch.testing.assert_close(selected, expected, atol=1e-3, rtol=1e-5)
    assert rig["skinner"].bind_vertices.shape[0] > 18000, "selection must not mutate the shared skinner"


def test_stance_displacement_measures_accumulated_drift_and_has_gradients():
    frames = 20
    _, horizontal = geometry(frames, 1, speed=0.001)
    horizontal.requires_grad_(True)
    tags = labels([CONTACT] * frames, [STABLE] * (frames - 1), [True] * (frames - 1),
                  [[True]] * (frames - 1))
    value, count = stance_displacement_loss(horizontal, tags["interval_state"], tags["region_measured"],
                                           tags["vertex_measured"], horizons=(8,))
    assert count == frames - 8
    assert float(value) == pytest.approx(0.008)
    value.backward()
    assert horizontal.grad.abs().sum() > 0


def test_stance_displacement_never_bridges_an_unmeasured_interval():
    _, horizontal = geometry(9, 2, speed=0.01)
    measured = torch.ones(8, dtype=torch.bool)
    vertices = torch.ones(8, 2, dtype=torch.bool)
    vertices[:, 1] = False
    measured[4] = False
    value, count = stance_displacement_loss(horizontal, torch.full((8,), STABLE), measured, vertices,
                                           horizons=(8,))
    assert count == 0
    assert float(value) == 0


def test_stance_displacement_allows_rolling_between_material_points():
    _, horizontal = geometry(12, 2, speed=0.0)
    horizontal[6:, 0, 0] = torch.arange(6) * 0.01
    horizontal[:6, 1, 0] = torch.arange(6) * 0.01
    vertices = torch.zeros(11, 2, dtype=torch.bool)
    vertices[:5, 0] = True
    vertices[6:, 1] = True
    value, count = stance_displacement_loss(horizontal, torch.full((11,), STABLE),
                                           torch.ones(11, dtype=torch.bool), vertices, horizons=(4,))
    assert count == 4
    assert float(value) == 0
