"""Generated stair metrics stay geometric, bounded and phase-explicit."""

import numpy as np
import pytest

from motionbricks.data.stair_sequence_metrics import stair_sequence_metrics


PATCH = {"id": 7, "height_m": 0., "polygon_xz_m": [[-1., -1.], [1., -1.],
                                                     [1., 1.], [-1., 1.]],
         "confidence": .5, "status": "weak_support"}


def soles(frames=3, vertices=2):
    values = np.zeros((frames, vertices, 3), dtype=np.float64)
    values[..., 0] = np.linspace(-.2, .2, vertices)
    return {"left": values.copy(), "right": values.copy()}


def score(points, **kwargs):
    frames = len(points["left"])
    return stair_sequence_metrics(
        points, 30., [PATCH], near_surface_tolerance_m=.03,
        closest_contact_tolerance_m=.03,
        scoring_frame_mask=np.ones(frames, dtype=bool),
        scoring_interval_mask=np.ones(max(0, frames - 1), dtype=bool), **kwargs)


def test_known_plane_penetration_and_unknown_coverage_are_separate():
    points = soles()
    points["left"][..., 1] = -.02
    points["right"][..., 0] += 3.
    result = score(points)
    left = result["sides"]["left"]["known_patch"]
    assert left["vertex_coverage"] == 1.
    assert left["penetration_depth_m"]["p95"] == pytest.approx(.02)
    right = result["sides"]["right"]["known_patch"]
    assert right["vertex_coverage"] == 0.
    assert right["penetration_depth_m"]["count"] == 0
    assert right["penetration_depth_m"]["p95"] is None
    assert result["sides"]["right"]["closest_contact_candidate"]["coverage_over_scored_frames"] == 0.


def test_common_world_translation_preserves_metrics():
    points = soles()
    points["left"][..., 1] = -.04
    original = score(points)["sides"]["left"]
    delta = np.array([4., 2., -3.])
    moved = {side: value + delta for side, value in points.items()}
    patch = {**PATCH, "height_m": PATCH["height_m"] + delta[1],
             "polygon_xz_m": (np.asarray(PATCH["polygon_xz_m"]) + delta[[0, 2]]).tolist()}
    frames = len(points["left"])
    shifted = stair_sequence_metrics(
        moved, 30., [patch], near_surface_tolerance_m=.03,
        closest_contact_tolerance_m=.03,
        scoring_frame_mask=np.ones(frames, bool), scoring_interval_mask=np.ones(frames - 1, bool))
    shifted = shifted["sides"]["left"]
    assert shifted["known_patch"]["penetration_depth_m"]["p95"] == pytest.approx(
        original["known_patch"]["penetration_depth_m"]["p95"])
    assert shifted["known_patch"]["vertex_coverage"] == original["known_patch"]["vertex_coverage"]


def test_source_stance_speed_is_not_mislabeled_as_generated_actual_slip():
    points = soles(frames=3, vertices=1)
    points["left"][:, 0, 0] = [0., .2, .4]
    points["left"][:, 0, 1] = [0., .2, .2]
    source = {side: np.ones((2, 1), dtype=bool) for side in ("left", "right")}
    result = stair_sequence_metrics(
        points, 30., [PATCH], near_surface_tolerance_m=.03,
        closest_contact_tolerance_m=.03, expected_source_material_masks=source,
        scoring_frame_mask=np.ones(3, bool), scoring_interval_mask=np.ones(2, bool))
    left = result["sides"]["left"]
    assert left["geometry_speed_candidate"]["tangential_speed_mps"]["count"] == 0
    assert left["source_phase_agreement"]["source_timed_tangential_speed_mps"]["mean"] == pytest.approx(6.)
    assert "not actual load-bearing slip" in left["source_phase_agreement"]["meaning"]


def test_speed_tracks_fixed_material_vertices_not_changing_lowest_vertex():
    points = soles(frames=2, vertices=2)
    points["left"][0, :, 0] = [0., .4]
    points["left"][1, :, 0] = [.1, .5]
    points["left"][0, :, 1] = [0., .02]
    points["left"][1, :, 1] = [.02, 0.]
    result = stair_sequence_metrics(
        points, 1., [PATCH], near_surface_tolerance_m=.03,
        closest_contact_tolerance_m=.03, scoring_frame_mask=np.ones(2, bool),
        scoring_interval_mask=np.ones(1, bool))
    speed = result["sides"]["left"]["geometry_speed_candidate"]["tangential_speed_mps"]
    assert speed["count"] == 2
    assert speed["mean"] == pytest.approx(.1)
    assert speed["max"] == pytest.approx(.1)


def test_frame_and_interval_masks_keep_their_own_endpoints():
    points = soles(frames=4, vertices=1)
    points["left"][:, 0, 0] = [0., .1, .3, .3]
    points["left"][:, 0, 1] = [-.1, 0., -.02, -.3]
    frame_mask = np.array([False, False, False, True])
    interval_mask = np.array([False, True, False])
    source = {side: np.array([[False], [True], [False]]) for side in ("left", "right")}
    result = stair_sequence_metrics(
        points, 10., [PATCH], near_surface_tolerance_m=.03,
        closest_contact_tolerance_m=.03, expected_source_material_masks=source,
        scoring_frame_mask=frame_mask, scoring_interval_mask=interval_mask)
    left = result["sides"]["left"]
    assert left["known_patch"]["penetration_depth_m"]["count"] == 1
    assert left["known_patch"]["penetration_depth_m"]["mean"] == pytest.approx(.3)
    assert left["geometry_speed_candidate"]["material_vertex_interval_count"] == 1
    assert left["source_phase_agreement"]["source_material_vertex_interval_count"] == 1
    assert left["source_phase_agreement"]["source_timed_tangential_speed_mps"]["mean"] == pytest.approx(2.)


@pytest.mark.parametrize("bad_fps", [0., -1., np.nan])
def test_invalid_fps_rejected(bad_fps):
    points = soles()
    with pytest.raises(ValueError, match="fps"):
        stair_sequence_metrics(points, bad_fps, [PATCH], near_surface_tolerance_m=.03,
                               closest_contact_tolerance_m=.03,
                               scoring_frame_mask=np.ones(3, bool),
                               scoring_interval_mask=np.ones(2, bool))
