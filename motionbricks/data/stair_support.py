"""Offline, evidence-bounded horizontal support patches, not measured terrain.

Adapted from contact-based scene reconstruction (TERRA, arXiv:2609.38653):
stationary, locally low foot material points provide positive evidence. Static
spatial patches must not contradict traversed free space. We deliberately omit
unobserved treads, risers, scene completion, and force/load labels. The horizontal
normal is an explicit stair prior, not an orientation measured from motion.
Full-clip hindsight is allowed here; these labels are NOT causal runtime inputs.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.spatial import ConvexHull, QhullError, cKDTree


@dataclass(frozen=True)
class SupportConfig:
    # Candidate SI thresholds; intentionally distinct from flat-ground labels.
    stationary_speed_mps: float = 0.15
    min_stationary_seconds: float = 0.10
    min_event_seconds: float = 0.12
    local_low_window_seconds: float = 0.30
    local_low_band_m: float = 0.04
    sole_bottom_band_m: float = 0.015
    min_stationary_vertices: int = 3
    max_fit_residual_m: float = 0.02
    level_merge_m: float = 0.03
    min_level_span_m: float = 0.12
    min_levels: int = 3
    min_polygon_area_m2: float = 0.0001
    max_free_space_penetration_m: float = 0.03
    min_root_height_span_m: float = 0.10
    require_stairs: bool = True

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "require_stairs":
                if not isinstance(value, bool):
                    raise ValueError("require_stairs must be boolean")
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_stationary_vertices < 3 or self.min_levels < 3:
            raise ValueError("At least three vertices and three levels are required")


def true_runs(mask):
    """Half-open continuous runs; interval and frame axes stay distinct."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("Run mask must be one-dimensional")
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def inside_polygon(points, polygon, tolerance=1e-9):
    """Membership in a counterclockwise convex evidence hull, including boundary."""
    points, polygon = np.asarray(points), np.asarray(polygon)
    edge = np.roll(polygon, -1, axis=0) - polygon
    offset = points[..., None, :] - polygon
    cross = edge[:, 0] * offset[..., 1] - edge[:, 1] * offset[..., 0]
    return (cross >= -tolerance).all(axis=-1)


def polygon_area(polygon):
    polygon = np.asarray(polygon)
    other = np.roll(polygon, -1, axis=0)
    return float(abs(np.sum(polygon[:, 0] * other[:, 1] - polygon[:, 1] * other[:, 0])) / 2)


def bounded_membership(points, polygon):
    """Avoid building an [entire clip, vertices, hull edges] temporary."""
    points = np.asarray(points)
    polygon = np.asarray(polygon)
    result = np.zeros(points.shape[:-1], dtype=bool)
    box = ((points >= polygon.min(axis=0) - 1e-9) & (points <= polygon.max(axis=0) + 1e-9)).all(axis=-1)
    result[box] = inside_polygon(points[box], polygon)
    return result


def polygon_overlap_area(left, right):
    """Convex polygon clipping; no bounding-box invented support area."""
    output = list(np.asarray(left, dtype=np.float64))
    right = np.asarray(right, dtype=np.float64)
    for a, b in zip(right, np.roll(right, -1, axis=0)):
        source, output = output, []
        if not source:
            return 0.0
        edge = b - a

        def signed(p):
            offset = p - a
            return edge[0] * offset[1] - edge[1] * offset[0]

        p = source[-1]
        for q in source:
            dp, dq = signed(p), signed(q)
            if (dp >= 0) != (dq >= 0):
                output.append(p + (q - p) * (dp / (dp - dq)))
            if dq >= 0:
                output.append(q)
            p = q
    return polygon_area(output) if len(output) >= 3 else 0.0


def query_support(points_xz, patches, level_tolerance_m=0.03):
    """Static spatial query independent of foot Y or current frame.

    Unknown returns NaN height, zero confidence, and id -1. Conflicting overlapping
    heights stay unknown; we never select whichever height is closest to a foot.
    """
    points = np.asarray(points_xz, dtype=np.float64)
    if points.shape[-1] != 2 or not np.isfinite(points).all():
        raise ValueError("Query points must be finite XZ coordinates")
    shape = points.shape[:-1]
    flat = points.reshape(-1, 2)
    height = np.full(len(flat), np.nan)
    confidence = np.zeros(len(flat))
    ids = np.full(len(flat), -1, dtype=np.int64)
    low, high = np.full(len(flat), np.inf), np.full(len(flat), -np.inf)
    for patch in patches:
        if patch["status"] != "weak_support":
            continue
        mask = bounded_membership(flat, patch["polygon_xz_m"])
        low[mask] = np.minimum(low[mask], patch["height_m"])
        high[mask] = np.maximum(high[mask], patch["height_m"])
        select = mask & (patch["confidence"] > confidence)
        height[select] = patch["height_m"]
        confidence[select] = patch["confidence"]
        ids[select] = patch["id"]
    ambiguous = (ids >= 0) & (high - low > level_tolerance_m)
    height[ambiguous], confidence[ambiguous], ids[ambiguous] = np.nan, 0, -1
    return height.reshape(shape), confidence.reshape(shape), ids.reshape(shape)


def _persistent_intervals(mask, minimum):
    result = np.zeros_like(mask, dtype=bool)
    for vertex in range(mask.shape[1]):
        for first, last in true_runs(mask[:, vertex]):
            if last - first >= minimum:
                result[first:last, vertex] = True
    return result


def _extract_events(vertices, side, fps, config, first_id):
    speed = np.linalg.norm(np.diff(vertices, axis=0), axis=-1) * fps
    bottom = vertices[..., 1].min(axis=1)
    radius = int(round(config.local_low_window_seconds * fps))
    local_low = minimum_filter1d(bottom, size=2 * radius + 1, mode="nearest")
    near_bottom = vertices[..., 1] <= bottom[:, None] + config.sole_bottom_band_m
    low = bottom <= local_low + config.local_low_band_m
    stationary = (speed <= config.stationary_speed_mps)
    stationary &= near_bottom[:-1] & near_bottom[1:] & low[:-1, None] & low[1:, None]
    measured = _persistent_intervals(stationary, max(1, int(np.ceil(config.min_stationary_seconds * fps))))
    frame_vertices = np.zeros_like(near_bottom)
    frame_vertices[:-1] |= measured
    frame_vertices[1:] |= measured
    candidate = frame_vertices.sum(axis=1) >= config.min_stationary_vertices
    events, patches = [], []
    for first, last in true_runs(candidate):
        eid = first_id + len(events)
        event = {"id": eid, "side": side, "start_frame": int(first),
                 "end_frame_exclusive": int(last), "duration_seconds": float((last - first) / fps),
                 "status": "rejected", "reasons": []}
        events.append(event)
        if (last - first) / fps < config.min_event_seconds:
            event["reasons"].append("short_event")
            continue
        mask = frame_vertices[first:last]
        points = vertices[first:last][mask]
        times = np.broadcast_to(np.arange(first, last)[:, None], mask.shape)[mask]
        # Fit on alternating frames and report the other frames separately. This
        # checks temporal consistency, not independent verification of contact.
        fit_points, heldout = points[times % 2 == 0], points[times % 2 == 1]
        if len(fit_points) < 3 or len(heldout) < 3:
            event["reasons"].append("insufficient_temporal_holdout")
            continue
        height = float(np.median(fit_points[:, 1]))
        fit_p95 = float(np.quantile(abs(fit_points[:, 1] - height), 0.95))
        heldout_p95 = float(np.quantile(abs(heldout[:, 1] - height), 0.95))
        event.update(height_m=height, fit_residual_p95_m=fit_p95,
                     temporal_holdout_residual_p95_m=heldout_p95,
                     material_vertex_count=int(mask.any(axis=0).sum()),
                     evidence_point_count=int(len(points)))
        if max(fit_p95, heldout_p95) > config.max_fit_residual_m:
            event["reasons"].append("nonhorizontal_or_drifting_evidence")
            continue
        try:
            xz = points[:, [0, 2]]
            polygon = xz[ConvexHull(xz).vertices]
        except QhullError:
            event["reasons"].append("degenerate_spatial_evidence")
            continue
        area = polygon_area(polygon)
        if area < config.min_polygon_area_m2:
            event["reasons"].append("insufficient_spatial_area")
            continue
        event_speed = speed[first:last - 1][measured[first:last - 1]]
        speed_p95 = float(np.quantile(event_speed, 0.95))
        components = {
            "duration": min(1.0, event["duration_seconds"] / 0.30),
            "stationarity": max(0.0, 1 - speed_p95 / (2 * config.stationary_speed_mps)),
            "temporal_consistency": max(0.0, 1 - max(fit_p95, heldout_p95) / (2 * config.max_fit_residual_m)),
        }
        # A single source event is capped; repeated events do not become force GT.
        confidence = float(min(0.8, *components.values()))
        event.update(status="candidate", speed_p95_mps=speed_p95, confidence_components=components)
        patches.append({"id": len(patches), "height_m": height, "polygon_xz_m": polygon.tolist(),
                        "area_m2": area, "confidence": confidence, "event_ids": [eid],
                        "level_id": -1, "status": "candidate", "reasons": []})
    return candidate, measured, speed, events, patches


def reconstruct_support(sole_vertices, root_positions, fps, config=None, body_joints=None):
    """Return weak arrays and provenance; preserve source motion and its world frame.

    Input is native-rate official LBS geometry. No previous planar contact labels
    or filename classification are used to choose support height. Body checks use
    joint landmarks only and must not be described as whole-mesh collision tests.
    """
    config = config or SupportConfig()
    root = np.asarray(root_positions, dtype=np.float64)
    if not np.isfinite(fps) or fps <= 0 or root.ndim != 2 or root.shape[1] != 3 or len(root) < 3:
        raise ValueError("Require >=3 root frames and finite positive native FPS")
    if not np.isfinite(root).all() or set(sole_vertices) != {"left", "right"}:
        raise ValueError("Require finite root and both sole regions")
    soles = {side: np.asarray(value, dtype=np.float64) for side, value in sole_vertices.items()}
    for value in soles.values():
        if value.ndim != 3 or value.shape[0] != len(root) or value.shape[1] < 3 or value.shape[2] != 3:
            raise ValueError("Soles must be [root frames, >=3 fixed material vertices, xyz]")
        if not np.isfinite(value).all():
            raise ValueError("Sole coordinates must be finite")
    if body_joints is not None:
        body_joints = np.asarray(body_joints, dtype=np.float64)
        if body_joints.ndim != 3 or body_joints.shape[0] != len(root) or body_joints.shape[-1] != 3:
            raise ValueError("Body landmarks must match source frames")
        if not np.isfinite(body_joints).all():
            raise ValueError("Body landmarks must be finite")
    arrays, events, patches, measurements = {}, [], [], {}
    for side in ("left", "right"):
        candidate, measured, speed, side_events, side_patches = _extract_events(
            soles[side], side, fps, config, len(events))
        for patch in side_patches:
            patch["id"] = len(patches)
            patches.append(patch)
        events.extend(side_events)
        arrays[f"candidate_frames_{side}"] = candidate
        measurements[side] = measured, speed

    flat_plane = None
    if not config.require_stairs and patches:
        # One recording-wide robust plane, never a floor following each foot.
        # Outliers remain rejected evidence; no source coordinates are moved.
        flat_plane = float(np.median([p["height_m"] for p in patches]))
        for patch in patches:
            event = events[patch["event_ids"][0]]
            offset = abs(patch["height_m"] - flat_plane)
            patch["event_fit_height_m"] = patch["height_m"]
            patch["recording_plane_offset_m"] = float(offset)
            if offset > config.max_fit_residual_m:
                patch["reasons"].append("inconsistent_with_recording_flat_plane")
            else:
                patch["height_m"] = flat_plane
                patch["confidence"] = min(patch["confidence"], max(0.0, 1 - offset / (2 * config.max_fit_residual_m)))
            event["recording_plane_offset_m"] = float(offset)

    # Levels cluster heights without chain-merging a continuous ramp into stairs.
    levels = []
    for patch in sorted(patches, key=lambda item: item["height_m"]):
        if not levels or patch["height_m"] - min(p["height_m"] for p in levels[-1]) > config.level_merge_m:
            levels.append([])
        levels[-1].append(patch)
        patch["level_id"] = len(levels) - 1
    for i, left in enumerate(patches):
        for right in patches[i + 1:]:
            if abs(left["height_m"] - right["height_m"]) <= config.level_merge_m:
                continue
            if polygon_overlap_area(left["polygon_xz_m"], right["polygon_xz_m"]) > config.min_polygon_area_m2:
                for patch in (left, right):
                    patch["reasons"].append("contradictory_spatial_heights")
    probes = np.concatenate(list(soles.values()) + ([body_joints] if body_joints is not None else []), axis=1)
    # Only probes below some candidate plane can violate a solid halfspace.
    # A spatial index makes full-recording checks practical without cropping them.
    highest = max((p["height_m"] for p in patches), default=-np.inf)
    frame_ids, probe_ids = np.nonzero(probes[..., 1] < highest)
    low_probes = probes[frame_ids, probe_ids]
    tree = cKDTree(low_probes[:, [0, 2]]) if len(low_probes) else None
    for patch in patches:
        polygon = np.asarray(patch["polygon_xz_m"])
        center = (polygon.min(axis=0) + polygon.max(axis=0)) / 2
        radius = np.linalg.norm(polygon - center, axis=1).max() + 1e-9
        nearby = np.asarray(tree.query_ball_point(center, radius), dtype=np.int64) if tree else np.empty(0, dtype=np.int64)
        inside = inside_polygon(low_probes[nearby][:, [0, 2]], polygon)
        nearby = nearby[inside]
        depth = np.maximum(patch["height_m"] - low_probes[nearby, 1], 0)
        max_depth = float(depth.max(initial=0))
        violation_frames = np.unique(frame_ids[nearby[depth > config.max_free_space_penetration_m]])
        patch.update(free_space_max_penetration_m=max_depth,
                     free_space_violation_frames=int(len(violation_frames)))
        if len(violation_frames):
            patch["reasons"].append("source_traverses_inferred_solid_halfspace")
        patch["status"] = "rejected" if patch["reasons"] else "candidate"
    usable = [p for p in patches if p["status"] == "candidate"]
    level_heights = [float(np.median([p["height_m"] for p in usable if p["level_id"] == level]))
                     for level in sorted({p["level_id"] for p in usable})]
    sides = {events[p["event_ids"][0]]["side"] for p in usable}
    span = max(level_heights) - min(level_heights) if level_heights else 0.0
    root_span = float(np.ptp(root[:, 1]))
    pair_reasons = []
    if config.require_stairs:
        if len(level_heights) < config.min_levels or span < config.min_level_span_m:
            pair_reasons.append("insufficient_distinct_vertical_levels")
    elif len(level_heights) != 1:
        pair_reasons.append("flat_prior_requires_one_consistent_height_level")
    if len(sides) != 2:
        pair_reasons.append("missing_bilateral_support_evidence")
    if config.require_stairs and root_span < config.min_root_height_span_m:
        pair_reasons.append("insufficient_root_vertical_travel")
    pair_status = "weak_pair_candidate" if not pair_reasons else "insufficient_geometry"
    for patch in usable:
        if not pair_reasons:
            patch["status"] = "weak_support"
    for patch in patches:
        event = events[patch["event_ids"][0]]
        event["status"] = patch["status"]
        event["reasons"].extend(patch["reasons"])
    for side, vertices in soles.items():
        ids, confidence = np.full(len(root), -1, dtype=np.int64), np.zeros(len(root))
        gaps = np.full(len(root), np.nan)
        measured, speed = measurements[side]
        output_measured = np.zeros_like(measured)
        for patch in patches:
            event = events[patch["event_ids"][0]]
            if patch["status"] != "weak_support" or event["side"] != side:
                continue
            first, last = event["start_frame"], event["end_frame_exclusive"]
            ids[first:last], confidence[first:last] = patch["id"], patch["confidence"]
            gaps[first:last] = vertices[first:last, :, 1].min(axis=1) - patch["height_m"]
            within = inside_polygon(vertices[first:last][..., [0, 2]], patch["polygon_xz_m"])
            near = abs(vertices[first:last, :, 1] - patch["height_m"]) <= config.max_fit_residual_m
            output_measured[first:last - 1] |= measured[first:last - 1] & near[:-1] & near[1:] & within[:-1] & within[1:]
        # Tangential material-point speed, never displacement of changing minima.
        slip = np.linalg.norm(np.diff(vertices[..., [0, 2]], axis=0), axis=-1) * fps
        arrays.update({f"patch_id_{side}": ids, f"support_confidence_{side}": confidence,
                       f"support_gap_m_{side}": gaps, f"vertex_measured_{side}": output_measured,
                       f"slip_mps_{side}": np.where(output_measured, slip, np.nan)})
    report = {
        "schema_version": "stair-support-weak-v1", "patches": patches, "events": events,
        "summary": {"pair_status": pair_status, "reasons": pair_reasons, "frames": len(root),
                    "duration_seconds": (len(root) - 1) / fps, "native_fps": float(fps),
                    "candidate_events": len(events), "weak_support_patches": sum(p["status"] == "weak_support" for p in patches),
                    "usable_level_heights_m": level_heights, "level_span_m": span,
                    "root_vertical_span_m": root_span, "support_prior": "stairs" if config.require_stairs else "flat",
                    "recording_plane_y_m": flat_plane,
                    "evidence_frames": {s: int((arrays[f"patch_id_{s}"] >= 0).sum()) for s in soles},
                    "measured_vertex_intervals": {s: int(arrays[f"vertex_measured_{s}"].sum()) for s in soles}},
        "protocol": {"coordinate_system": "source world coordinates, Y up, metres, native FPS",
                     "thresholds": asdict(config), "threshold_status": "candidate, not calibrated or approved",
                     "source_motion_changed": False, "terrain_type": "horizontal support hulls only",
                     "confidence": "heuristic evidence quality [0,0.8], NOT contact probability or load measurement",
                     "unknown": "unobserved space, rejected patches and non-evidence frames; NaN height, id -1",
                     "force_labels": "unavailable; motion cannot prove that a foot carries load",
                     "collision_scope": "both sole regions and optional joint landmarks; NOT full body mesh",
                     "temporal_holdout": "alternating source frames; consistency only, not independent terrain GT",
                     "hindsight": "full source clip; do not feed future motion-derived features into runtime",
                     "unresolved": ["static hovering can resemble support", "normals use horizontal stair prior",
                                    "riser faces, full tread extents and unvisited terrain unknown",
                                    "requires known-terrain external validation before supervised deployment"],
                     "references": ["https://arxiv.org/html/2609.38653v1#S3.SS2",
                                    "https://arxiv.org/html/2604.24833v1#A5"]},
    }
    return arrays, report
