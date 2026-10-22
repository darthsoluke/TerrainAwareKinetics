"""Diagnostic geometry metrics for generated stair sole sequences.

Inputs are already-skinned, world-frame XYZ sole vertices in metres (Y up),
with material-vertex indices fixed across frames. Static weak-support patches
are comparison geometry only: they are not measured terrain or force labels.
Nothing returned by this module is suitable as a prediction-dependent training
mask or as a skill-approved decision.
"""

from collections.abc import Mapping

import numpy as np

from motionbricks.data.stair_support import query_support


_SIDES = ("left", "right")


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {"count": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "p95": float(np.quantile(values, 0.95)) if values.size else None,
            "max": float(values.max()) if values.size else None}


def _mask(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return result


def stair_sequence_metrics(
    predicted_soles,
    fps,
    patches,
    *,
    near_surface_tolerance_m,
    closest_contact_tolerance_m,
    expected_source_material_masks=None,
    scoring_frame_mask,
    scoring_interval_mask,
):
    """Score arbitrary-length predicted soles against fixed weak-support patches.

    Args:
        predicted_soles: ``{"left": ..., "right": ...}``, each finite
            ``[T, V, 3]`` array of world XYZ positions in metres. Vertex index
            denotes the same material point at every frame.
        fps: Finite positive frame rate in frames/second.
        patches: Static ``stair_support`` patch dictionaries. Only
            ``status == "weak_support"`` patches count as known support.
        near_surface_tolerance_m: Positive absolute vertical-gap threshold for
            geometry-based near-surface interval candidates.
        closest_contact_tolerance_m: Positive absolute vertical-gap threshold
            for a frame's closest-contact candidate (minimum over known sole
            vertices). This is candidate coverage, not contact truth.
        expected_source_material_masks: Optional per-side boolean
            ``[T-1, V]`` source-evidence interval masks. Used only for separate
            source-phase agreement diagnostics; they do not gate geometry
            metrics and do not establish predicted stance or load.
        scoring_frame_mask: Explicit boolean ``[T]`` mask. Frame metrics use
            selected frames only.
        scoring_interval_mask: Explicit boolean ``[T-1]`` mask. Interval
            metrics use selected interval starts (interval ``i`` joins frames
            ``i`` and ``i+1``); endpoints are never inferred or extended.

    Returns JSON-friendly dictionaries with finite-sample statistics and
    coverage fractions. Empty evidence is ``None``, never zero. Penetration is
    ``max(patch_height - vertex_y, 0)`` for vertices with a known spatial
    query. Tangential speed is reported as geometry candidates only when the
    same material vertex is known and near-surface at both interval endpoints.
    """
    if not isinstance(predicted_soles, Mapping) or set(predicted_soles) != set(_SIDES):
        raise ValueError("predicted_soles must contain exactly left and right")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive, in frames/second")
    for name, value in (("near_surface_tolerance_m", near_surface_tolerance_m),
                        ("closest_contact_tolerance_m", closest_contact_tolerance_m)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, in metres")
    if not isinstance(patches, (list, tuple)):
        raise ValueError("patches must be a reusable sequence of static patch records")
    patch_ids = []
    for patch in patches:
        if not isinstance(patch, Mapping) or "status" not in patch:
            raise ValueError("Each patch must be a mapping with a status")
        if patch["status"] != "weak_support":
            continue
        try:
            height = float(patch["height_m"])
            confidence = float(patch["confidence"])
            patch_id = int(patch["id"])
            polygon = np.asarray(patch["polygon_xz_m"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Weak patches require id, height_m, confidence and polygon_xz_m") from exc
        if (patch_id < 0 or not np.isfinite([height, confidence]).all()
                or polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2
                or not np.isfinite(polygon).all()):
            raise ValueError("Weak patch geometry must be finite; height_m is metres and id is nonnegative")
        patch_ids.append(patch_id)
    if len(patch_ids) != len(set(patch_ids)):
        raise ValueError("Weak-support patch ids must be unique")

    soles = {side: np.asarray(predicted_soles[side], dtype=np.float64) for side in _SIDES}
    first = soles["left"]
    if first.ndim != 3 or first.shape[0] < 1 or first.shape[1] < 1 or first.shape[2] != 3:
        raise ValueError("Soles must have shape [T>=1, V>=1, 3]")
    frames, vertices = first.shape[:2]
    for side, values in soles.items():
        if values.shape != (frames, vertices, 3) or not np.isfinite(values).all():
            raise ValueError("Both soles must be finite [same T, same V, 3] arrays")
    frame_mask = _mask(scoring_frame_mask, (frames,), "scoring_frame_mask")
    interval_mask = _mask(scoring_interval_mask, (max(0, frames - 1),), "scoring_interval_mask")
    source_masks = None
    if expected_source_material_masks is not None:
        if not isinstance(expected_source_material_masks, Mapping) or set(expected_source_material_masks) != set(_SIDES):
            raise ValueError("expected_source_material_masks must contain exactly left and right")
        source_masks = {side: _mask(expected_source_material_masks[side], (max(0, frames - 1), vertices),
                                    f"expected_source_material_masks[{side}]")
                        for side in _SIDES}

    result = {
        "units": {"position": "m", "speed": "m/s", "time": "s", "fps": "frames/s"},
        "finite_checks": {"predicted_soles": True, "fps": True, "weak_support_patch_geometry": True},
        "thresholds": {"near_surface_tolerance_m": float(near_surface_tolerance_m),
                       "closest_contact_tolerance_m": float(closest_contact_tolerance_m)},
        "frames": int(frames), "material_vertices_per_side": int(vertices), "fps": float(fps),
        "scoring_frame_count": int(frame_mask.sum()),
        "scoring_interval_count": int(interval_mask.sum()),
        "interpretation": {
            "patches": "static weak source-derived geometry; not measured terrain or force truth",
            "contact": "closest-contact and near-surface values are geometric candidates, not stance labels",
            "geometry_speed": "fixed material-vertex tangential speed candidate; not asserted actual slip",
            "source_phase": "source-timed agreement diagnostic only; predicted swing can occur during source stance",
            "training": "diagnostic only; never use prediction-dependent masks as training supervision",
        },
        "sides": {},
    }

    for side, values in soles.items():
        height, _, patch_ids = query_support(values[..., [0, 2]], patches)
        known = (patch_ids >= 0) & np.isfinite(height)
        gap = values[..., 1] - height
        frame_known = known[frame_mask]
        sample_count = int(frame_mask.sum()) * vertices
        known_count = int(frame_known.sum())
        known_coverage = known_count / sample_count if sample_count else None
        penetration = np.maximum(-gap[frame_mask], 0)[frame_known]

        known_frame = known[frame_mask].any(axis=1)
        closest = np.zeros(int(frame_mask.sum()), dtype=bool)
        selected_gap = np.abs(gap[frame_mask])
        selected_known = known[frame_mask]
        if known_count:
            closest = (selected_known & (selected_gap <= closest_contact_tolerance_m)).any(axis=1)
        selected_frame_count = int(frame_mask.sum())
        closest_coverage = int(closest.sum()) / selected_frame_count if selected_frame_count else None
        known_frame_coverage = int(known_frame.sum()) / selected_frame_count if selected_frame_count else None

        horizontal_delta = np.diff(values[..., [0, 2]], axis=0)
        tangential_speed = np.linalg.norm(horizontal_delta, axis=-1) * fps
        near = known & (np.abs(gap) <= near_surface_tolerance_m)
        geometry_candidate = interval_mask[:, None] & near[:-1] & near[1:]
        geometry_speed = tangential_speed[geometry_candidate]
        geometry_candidate_intervals = geometry_candidate.any(axis=1)
        selected_intervals = int(interval_mask.sum())
        geometry_interval_coverage = (int((geometry_candidate_intervals & interval_mask).sum()) / selected_intervals
                                      if selected_intervals else None)

        side_result = {
            "known_patch": {
                "vertex_sample_count": known_count,
                "vertex_sample_total": sample_count,
                "vertex_coverage": known_coverage,
                "penetration_depth_m": _stats(penetration),
            },
            "closest_contact_candidate": {
                "frames_with_known_support": int(known_frame.sum()),
                "known_frame_coverage": known_frame_coverage,
                "candidate_frames": int(closest.sum()),
                "coverage_over_scored_frames": closest_coverage,
                "tolerance_m": float(closest_contact_tolerance_m),
            },
            "geometry_speed_candidate": {
                "material_vertex_interval_count": int(geometry_candidate.sum()),
                "candidate_interval_count": int(geometry_candidate_intervals.sum()),
                "coverage_over_scored_intervals": geometry_interval_coverage,
                "tangential_speed_mps": _stats(geometry_speed),
                "near_surface_tolerance_m": float(near_surface_tolerance_m),
            },
        }
        if source_masks is not None:
            source_selected = source_masks[side] & interval_mask[:, None]
            source_speed = tangential_speed[source_selected]
            source_count = int(source_selected.sum())
            overlap_count = int((source_selected & geometry_candidate).sum())
            side_result["source_phase_agreement"] = {
                "source_material_vertex_interval_count": source_count,
                "source_timed_tangential_speed_mps": _stats(source_speed),
                "source_material_vertex_intervals_also_geometry_candidates": overlap_count,
                "geometry_candidate_fraction_within_source_intervals": (
                    overlap_count / source_count if source_count else None),
                "meaning": "timing comparison only; speed can include predicted swing and is not actual load-bearing slip",
            }
        result["sides"][side] = side_result
    return result
