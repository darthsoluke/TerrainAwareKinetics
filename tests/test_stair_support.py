"""Weak support geometry must be static, bounded and never masquerade as force GT."""

import json

import numpy as np
import pytest

from motionbricks.data.stair_support import (SupportConfig, inside_polygon,
                                            polygon_overlap_area, query_support,
                                            reconstruct_support, true_runs)


def fixture(stairs=True, fps=60):
    # Three distinct stationary treads separated by moving foot intervals.
    count = int(3 * fps)
    base = np.array([[-.04, 0, -.08], [.04, 0, -.08], [.04, 0, .08], [-.04, 0, .08]])
    values = {}
    for side, x in (("left", -.15), ("right", .15)):
        foot = np.repeat(base[None], count, axis=0)
        for frame in range(count):
            phase = frame / fps
            swing_lift = 0.
            if phase < .8:
                level = 0.
            elif phase < 1.:
                level = (phase - .8) / .2
                swing_lift = .15 * np.sin(np.pi * level)
            elif phase < 1.8:
                level = 1.
            elif phase < 2.:
                level = 1. + (phase - 1.8) / .2
                swing_lift = .15 * np.sin(np.pi * (level - 1.))
            else:
                level = 2.
            foot[frame] += [x, (.2 * level if stairs else 0.) + swing_lift, .5 * level]
        values[side] = foot
    root = np.zeros((count, 3))
    root[:, 1] = 1. + np.linspace(0, .4 if stairs else .01, count)
    return values, root, float(fps)


def test_stationary_treads_create_bounded_weak_pairs():
    soles, root, fps = fixture()
    root_before = root.copy()
    sole_before = soles["left"].copy()
    arrays, report = reconstruct_support(soles, root, fps)
    assert report["summary"]["pair_status"] == "weak_pair_candidate"
    np.testing.assert_allclose(report["summary"]["usable_level_heights_m"], [0, .2, .4])
    assert report["summary"]["weak_support_patches"] == 6
    assert arrays["vertex_measured_left"].shape == (len(root) - 1, 4)
    assert np.nanmax(arrays["slip_mps_left"]) == 0
    assert (arrays["patch_id_left"] >= 0).sum() > 0
    assert (arrays["patch_id_left"] == -1).sum() > 0
    assert (arrays["support_confidence_left"] <= .8).all()
    np.testing.assert_array_equal(root, root_before)
    np.testing.assert_array_equal(soles["left"], sole_before)
    height, confidence, ids = query_support(np.array([[0., .5], [-.15, .5], [9, 9]]), report["patches"])
    assert np.isnan(height[0])  # The unvisited gap between feet is NOT completed.
    assert height[1] == pytest.approx(.2)
    assert np.isnan(height[2]) and confidence[2] == 0 and ids[2] == -1
    json.dumps(report, allow_nan=False)


def test_flat_mode_fits_single_recording_plane_without_moving_motion():
    soles, root, fps = fixture(False)
    soles["left"][..., 1] += .003
    soles["right"][..., 1] += .007
    arrays, report = reconstruct_support(soles, root, fps, SupportConfig(require_stairs=False))
    assert report["summary"]["pair_status"] == "weak_pair_candidate"
    assert report["summary"]["recording_plane_y_m"] == pytest.approx(.005)
    assert all(p["height_m"] == pytest.approx(.005) for p in report["patches"])
    assert np.nanmedian(arrays["support_gap_m_left"]) == pytest.approx(-.002)
    assert soles["left"][0, 0, 1] == .003


def test_flat_motion_cannot_be_called_stair_data():
    arrays, report = reconstruct_support(*fixture(False))
    assert report["summary"]["pair_status"] == "insufficient_geometry"
    assert "insufficient_distinct_vertical_levels" in report["summary"]["reasons"]
    assert (arrays["patch_id_left"] == -1).all()
    assert np.isnan(arrays["slip_mps_left"]).all()
    assert not arrays["vertex_measured_left"].any()


def test_a_moving_swing_does_not_become_stationary_support():
    soles, root, fps = fixture()
    for foot in soles.values():
        foot[..., 0] += np.arange(len(root))[:, None] / fps
    arrays, report = reconstruct_support(soles, root, fps)
    assert report["summary"]["candidate_events"] == 0
    assert not arrays["candidate_frames_left"].any()
    assert not arrays["vertex_measured_left"].any()


def test_vertical_translation_preserves_labels_and_shifts_patch_heights():
    soles, root, fps = fixture()
    first, report1 = reconstruct_support(soles, root, fps)
    delta = np.array([1.3, 5., -.8])
    second, report2 = reconstruct_support({s: v + delta for s, v in soles.items()}, root + delta, fps)
    np.testing.assert_array_equal(first["patch_id_left"], second["patch_id_left"])
    np.testing.assert_allclose(first["support_gap_m_left"], second["support_gap_m_left"], atol=1e-10)
    np.testing.assert_allclose(np.array(report2["summary"]["usable_level_heights_m"]),
                               np.array(report1["summary"]["usable_level_heights_m"]) + 5)


def test_yaw_preserves_geometry_and_confidence():
    soles, root, fps = fixture()
    angle = .67
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    first, report1 = reconstruct_support(soles, root, fps)
    second, report2 = reconstruct_support({s: v @ rotation.T for s, v in soles.items()}, root @ rotation.T, fps)
    assert report1["summary"]["pair_status"] == report2["summary"]["pair_status"]
    np.testing.assert_array_equal(first["patch_id_left"], second["patch_id_left"])
    np.testing.assert_allclose(first["support_confidence_left"], second["support_confidence_left"], atol=1e-10)


@pytest.mark.parametrize("fps", [30, 60, 120])
def test_thresholds_are_in_seconds_not_frames(fps):
    arrays, report = reconstruct_support(*fixture(fps=fps))
    assert report["summary"]["weak_support_patches"] == 6
    assert report["summary"]["pair_status"] == "weak_pair_candidate"


def test_free_space_violation_rejects_spatial_patch():
    soles, root, fps = fixture()
    body = root[:, None].copy()
    body[0, 0] = [-.15, -.08, 0]
    arrays, report = reconstruct_support(soles, root, fps, body_joints=body)
    bad = [p for p in report["patches"] if "source_traverses_inferred_solid_halfspace" in p["reasons"]]
    assert len(bad) == 1
    assert bad[0]["status"] == "rejected"
    assert bad[0]["free_space_violation_frames"] == 1
    assert (arrays["patch_id_left"][:40] == -1).all()


def test_unknown_queries_do_not_choose_nearest_height():
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    patches = [{"id": i, "height_m": h, "confidence": .8, "polygon_xz_m": square, "status": "weak_support"}
               for i, h in enumerate([0., .2])]
    height, confidence, ids = query_support([[.5, .5]], patches)
    assert np.isnan(height[0]) and confidence[0] == 0 and ids[0] == -1


def test_polygon_overlap_and_boundaries():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert polygon_overlap_area(square, square + [.5, .5]) == pytest.approx(.25)
    assert polygon_overlap_area(square, square + [2, 2]) == 0
    assert inside_polygon([[0, .5], [.5, .5], [-.01, .5]], square).tolist() == [True, True, False]


def test_force_and_causal_claims_are_explicitly_unavailable():
    _, report = reconstruct_support(*fixture())
    assert "unavailable" in report["protocol"]["force_labels"]
    assert "NOT contact probability" in report["protocol"]["confidence"]
    assert "future" in report["protocol"]["hindsight"]
    assert "static hovering" in report["protocol"]["unresolved"][0]


@pytest.mark.parametrize("change", ["fps", "missing_side", "nonfinite", "frame_mismatch"])
def test_invalid_inputs_fail_loudly(change):
    soles, root, fps = fixture()
    if change == "fps":
        fps = 0
    elif change == "missing_side":
        del soles["right"]
    elif change == "nonfinite":
        soles["right"][0, 0, 0] = np.nan
    else:
        root = root[:-1]
    with pytest.raises(ValueError):
        reconstruct_support(soles, root, fps)


def test_true_runs_half_open_interval_contract():
    assert true_runs([False, True, True, False, True]) == [(1, 3), (4, 5)]
