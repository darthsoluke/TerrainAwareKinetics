"""Support-geometry fitting for BONES-SEED clips (step 3a of terrain fitting).

The source motion is fixed. From its weak support evidence (stair_support patches,
plus palms and knees resting at a level) this module builds:

- a ground slab whose top is the lowest usable support level;
- for every higher level, a solid box from the ground up to that level whose top
  covers the evidence. Two extents are reported:
    minimal  - oriented minimum-area rectangle around the evidence, plus a margin
               on every side that stays free of the body;
    maximal  - each side pushed outward until the body (sole material points and
               landmark spheres) would be inside the box below its top, capped at
               max_growth_m.
  Any box between the two is consistent with the motion; later steps sample them.

Then it measures the result against the unchanged motion: body penetration into
the fitted solids, sole gap to the surface beneath during support, and evidence
left outside every fitted top. Coordinates stay in the source world (Y up, m).

Numpy only. to_scene_manifest() converts a fit to SceneManifest (needs torch).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SCHEMA_VERSION = "bones-terrain-fit-v1"

#: Landmark sphere radii in metres; conservative body volume for free-space checks.
LANDMARK_RADII = {
    "Hips": 0.10, "Chest": 0.12, "Neck1": 0.05, "Head": 0.10,
    "LeftHand": 0.03, "LeftHandMiddleEnd": 0.02, "RightHand": 0.03, "RightHandMiddleEnd": 0.02,
    "LeftShin": 0.05, "RightShin": 0.05, "LeftForeArm": 0.04, "RightForeArm": 0.04,
}


@dataclass(frozen=True)
class FitConfig:
    level_match_m: float = 0.03
    evidence_margin_m: float = 0.03
    max_growth_m: float = 0.60
    side_clearance_m: float = 0.01
    sole_tolerance_m: float = 0.02
    landmark_tolerance_m: float = 0.02
    palm_surface_offset_m: float = 0.03
    knee_surface_offset_m: float = 0.06
    limb_level_tolerance_m: float = 0.05
    stationary_speed_mps: float = 0.15
    min_event_seconds: float = 0.12
    ground_margin_m: float = 1.5
    ground_thickness_m: float = 0.20
    penetration_report_m: float = 0.02

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def oriented_rectangle(points_xz, minimum_side_m=0.05):
    """Minimum-area rectangle around 2D points: (center, axis_u, half_u, half_v).

    axis_u is a unit XZ vector; axis_v = (-axis_u[1], axis_u[0]). Degenerate input
    falls back to the principal axis with minimum_side_m on each thin side.
    """
    points = np.asarray(points_xz, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("Finite evidence points required")
    candidates = []
    if len(points) >= 3:
        try:
            from scipy.spatial import ConvexHull, QhullError
            hull = points[ConvexHull(points).vertices]
            edges = np.roll(hull, -1, axis=0) - hull
            candidates = [edge / np.linalg.norm(edge) for edge in edges if np.linalg.norm(edge) > 1e-9]
        except (QhullError, ValueError):
            candidates = []
    if not candidates:
        centered = points - points.mean(axis=0)
        if len(points) > 1 and np.abs(centered).max() > 1e-9:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            candidates = [vt[0]]
        else:
            candidates = [np.array([1.0, 0.0])]
    best = None
    for axis_u in candidates:
        axis_v = np.array([-axis_u[1], axis_u[0]])
        u, v = points @ axis_u, points @ axis_v
        area = max(np.ptp(u), 1e-9) * max(np.ptp(v), 1e-9)
        if best is None or area < best[0]:
            best = (area, axis_u, u, v)
    _, axis_u, u, v = best
    if axis_u[0] < 0 or (abs(axis_u[0]) < 1e-12 and axis_u[1] < 0):
        axis_u, u, v = -axis_u, -u, -v
    axis_v = np.array([-axis_u[1], axis_u[0]])
    center_u, center_v = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    center = center_u * axis_u + center_v * axis_v
    half_u = max((u.max() - u.min()) / 2, minimum_side_m / 2)
    half_v = max((v.max() - v.min()) / 2, minimum_side_m / 2)
    return center, axis_u, half_u, half_v


def _stationary_centroids(track, fps, config):
    speed = np.linalg.norm(np.diff(track, axis=0), axis=-1) * fps
    slow = speed <= config.stationary_speed_mps
    edges = np.diff(np.r_[False, slow, False].astype(np.int8))
    minimum = max(2, int(np.ceil(config.min_event_seconds * fps)))
    events = []
    for first, last in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if last + 1 - first >= minimum:
            segment = track[first:last + 1]
            events.append((int(first), int(last + 1), np.median(segment, axis=0)))
    return events


def body_probes(soles, joints):
    """[frames, probes, 3] points and [probes] radii: sole material points and landmarks."""
    names = [name for name in LANDMARK_RADII if name in joints]
    points = [soles["left"], soles["right"]] + [np.asarray(joints[name])[:, None] for name in names]
    radii = [np.zeros(soles["left"].shape[1]), np.zeros(soles["right"].shape[1])]
    radii += [np.array([LANDMARK_RADII[name]]) for name in names]
    return np.concatenate(points, axis=1).astype(np.float64), np.concatenate(radii)


def _local(points_xz, center, axis_u):
    axis_v = np.array([-axis_u[1], axis_u[0]])
    offset = points_xz - center
    return offset @ axis_u, offset @ axis_v


def _grow(rect, below_u, below_v, radius, limit, clearance):
    """Push each side outward until a below-top probe would enter; one pass per side, twice."""
    u0, u1, v0, v1 = rect
    for _ in range(2):
        for side in ("u1", "u0", "v1", "v0"):
            if side in ("u1", "u0"):
                inside = (below_v + radius > v0) & (below_v - radius < v1)
                coordinate = below_u
            else:
                inside = (below_u + radius > u0) & (below_u - radius < u1)
                coordinate = below_v
            if side.endswith("1"):
                current = {"u1": u1, "v1": v1}[side]
                ahead = inside & (coordinate - radius >= current)
                bound = min(limit[side], (coordinate - radius)[ahead].min() - clearance) if ahead.any() else limit[side]
                new = max(current, bound)
            else:
                current = {"u0": u0, "v0": v0}[side]
                ahead = inside & (coordinate + radius <= current)
                bound = max(limit[side], (coordinate + radius)[ahead].max() + clearance) if ahead.any() else limit[side]
                new = min(current, bound)
            if side == "u1":
                u1 = new
            elif side == "u0":
                u0 = new
            elif side == "v1":
                v1 = new
            else:
                v0 = new
    return u0, u1, v0, v1


def _box_record(box_id, name, role, center_xz, axis_u, rect, top, bottom):
    u0, u1, v0, v1 = rect
    axis_v = np.array([-axis_u[1], axis_u[0]])
    mid = center_xz + (u0 + u1) / 2 * axis_u + (v0 + v1) / 2 * axis_v
    # Local X = axis_u in XZ, local Y = world up, local Z = X x Y; columns are world vectors.
    x_axis = np.array([axis_u[0], 0.0, axis_u[1]])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.cross(x_axis, y_axis)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    # z_axis = x_axis cross up = (axis_v[0], 0, axis_v[1]); [x, y, z] is a proper rotation.
    return {
        "box_id": box_id, "name": name, "role": role,
        "center_xyz_m": [float(mid[0]), float((top + bottom) / 2), float(mid[1])],
        "half_extent_xyz_m": [float((u1 - u0) / 2), float((top - bottom) / 2), float((v1 - v0) / 2)],
        "rotation_local_to_world": rotation.tolist(),
        "top_m": float(top),
    }


def fit_support(soles, joints, root, fps, support_report, *, config=None):
    """Fit ground and support boxes to one clip; returns a JSON-ready dict."""
    config = config or FitConfig()
    soles = {side: np.asarray(value, dtype=np.float64) for side, value in soles.items()}
    joints = {name: np.asarray(value, dtype=np.float64) for name, value in joints.items()}
    root = np.asarray(root, dtype=np.float64)
    levels = sorted(float(level) for level in support_report["summary"]["usable_level_heights_m"])
    patches = [p for p in support_report["patches"] if p["status"] in ("candidate", "weak_support")]
    probes, radii = body_probes(soles, joints)
    sole_count = soles["left"].shape[1] + soles["right"].shape[1]
    tolerance = np.where(np.arange(len(radii)) < sole_count, config.sole_tolerance_m, config.landmark_tolerance_m)
    sole_low = np.minimum(soles["left"][..., 1].min(axis=1), soles["right"][..., 1].min(axis=1))
    ground = levels[0] if levels else float(np.percentile(sole_low, 2))

    # Evidence per level: foot patch polygons, then palms and knees resting at that level.
    evidence = {level: [] for level in levels}
    sources = {level: {"foot_patches": 0, "palm_events": 0, "knee_events": 0} for level in levels}
    for patch in patches:
        level = min(levels, key=lambda value: abs(value - patch["height_m"])) if levels else None
        if level is not None and abs(level - patch["height_m"]) <= config.level_match_m:
            evidence[level].extend(np.asarray(patch["polygon_xz_m"]).tolist())
            sources[level]["foot_patches"] += 1
    limbs = []
    for side in ("Left", "Right"):
        if f"{side}Hand" in joints and f"{side}HandMiddleEnd" in joints:
            limbs.append(("palm", 0.5 * (joints[f"{side}Hand"] + joints[f"{side}HandMiddleEnd"]),
                          config.palm_surface_offset_m))
        if f"{side}Shin" in joints:
            limbs.append(("knee", joints[f"{side}Shin"], config.knee_surface_offset_m))
    limb_events = []
    for kind, track, offset in limbs:
        for first, last, centroid in _stationary_centroids(track, fps, config):
            surface = centroid[1] - offset
            if not levels:
                continue
            level = min(levels, key=lambda value: abs(value - surface))
            if abs(level - surface) <= config.limb_level_tolerance_m:
                limb_events.append({"kind": kind, "level_m": level, "start_frame": first,
                                    "end_frame_exclusive": last, "xz_m": [float(centroid[0]), float(centroid[2])]})
                if level != ground:
                    evidence[level].append([centroid[0], centroid[2]])
                    sources[level][f"{kind}_events"] += 1

    path_low = probes[..., [0, 2]].reshape(-1, 2).min(axis=0) - config.ground_margin_m
    path_high = probes[..., [0, 2]].reshape(-1, 2).max(axis=0) + config.ground_margin_m
    ground_box = {
        "box_id": 1, "name": "ground", "role": "ground",
        "center_xyz_m": [float((path_low[0] + path_high[0]) / 2), ground - config.ground_thickness_m / 2,
                         float((path_low[1] + path_high[1]) / 2)],
        "half_extent_xyz_m": [float((path_high[0] - path_low[0]) / 2), config.ground_thickness_m / 2,
                              float((path_high[1] - path_low[1]) / 2)],
        "rotation_local_to_world": np.eye(3).tolist(), "top_m": ground,
    }
    minimal, maximal, level_reports = [ground_box], [ground_box], []
    flat_points = probes.reshape(-1, 3)
    flat_radii = np.broadcast_to(radii, probes.shape[:2]).reshape(-1)
    flat_tolerance = np.broadcast_to(tolerance, probes.shape[:2]).reshape(-1)
    for index, level in enumerate(level for level in levels if level != ground):
        points = np.asarray(evidence[level], dtype=np.float64)
        report = {"level_m": level, "height_above_ground_m": level - ground, **sources[level]}
        if len(points) == 0:
            report["status"] = "no_evidence"
            level_reports.append(report)
            continue
        center, axis_u, half_u, half_v = oriented_rectangle(points)
        below = flat_points[:, 1] - flat_radii < level - flat_tolerance
        u, v = _local(flat_points[below][:, [0, 2]], center, axis_u)
        radius = flat_radii[below]
        evidence_rect = (-half_u, half_u, -half_v, half_v)
        inside_evidence = ((u + radius > -half_u) & (u - radius < half_u) &
                           (v + radius > -half_v) & (v - radius < half_v))
        margin_limit = {"u0": -half_u - config.evidence_margin_m, "u1": half_u + config.evidence_margin_m,
                        "v0": -half_v - config.evidence_margin_m, "v1": half_v + config.evidence_margin_m}
        far_limit = {"u0": -half_u - config.max_growth_m, "u1": half_u + config.max_growth_m,
                     "v0": -half_v - config.max_growth_m, "v1": half_v + config.max_growth_m}
        outside = ~inside_evidence
        rect_min = _grow(evidence_rect, u[outside], v[outside], radius[outside], margin_limit, config.side_clearance_m)
        rect_max = _grow(rect_min, u[outside], v[outside], radius[outside], far_limit, config.side_clearance_m)
        bottom = ground - config.ground_thickness_m
        name = f"support_{index + 1}"
        minimal.append(_box_record(len(minimal) + 1, name, "support", center, axis_u, rect_min, level, bottom))
        maximal.append(_box_record(len(maximal) + 1, name, "support", center, axis_u, rect_max, level, bottom))
        report.update(status="fitted", evidence_points=int(len(points)),
                      evidence_conflict_probe_frames=int(inside_evidence.sum()),
                      minimal_size_m=[float(rect_min[1] - rect_min[0]), float(rect_min[3] - rect_min[2])],
                      maximal_size_m=[float(rect_max[1] - rect_max[0]), float(rect_max[3] - rect_max[2])])
        level_reports.append(report)

    metrics = {name: measure(soles, joints, boxes, patches, levels, config)
               for name, boxes in (("minimal", minimal), ("maximal", maximal))}
    flags = []
    if not levels:
        flags.append("no_usable_support")
    if any(r.get("evidence_conflict_probe_frames", 0) for r in level_reports):
        flags.append("evidence_conflicts_with_body")
    if any(m["penetration_frames_over_report"] for m in metrics.values()):
        flags.append("penetration_over_report")
    return {
        "schema_version": SCHEMA_VERSION, "ground_m": ground,
        "levels_m": [level - ground for level in levels], "levels": level_reports,
        "limb_events": limb_events, "boxes": {"minimal": minimal, "maximal": maximal},
        "metrics": metrics, "flags": flags, "config": asdict(config),
        "protocol": {"source_changed": False, "coordinates": "source world, Y up, metres",
                     "geometry": "ground slab plus solid boxes from the ground up to each support level",
                     "status": "weak-evidence fit; not measured terrain"},
    }


def measure(soles, joints, boxes, patches, levels, config):
    """Penetration of the unchanged body into the fitted solids and support gaps."""
    probes, radii = body_probes(soles, joints)
    sole_count = soles["left"].shape[1] + soles["right"].shape[1]
    tolerance = np.where(np.arange(len(radii)) < sole_count, config.sole_tolerance_m, config.landmark_tolerance_m)
    depth = np.full(probes.shape[:2], -np.inf)
    for box in boxes:
        center = np.asarray(box["center_xyz_m"])
        half = np.asarray(box["half_extent_xyz_m"])
        rotation = np.asarray(box["rotation_local_to_world"])
        local = (probes - center) @ rotation
        within = (np.abs(local[..., 0]) <= half[0]) & (np.abs(local[..., 2]) <= half[2])
        box_depth = np.where(within, box["top_m"] - (probes[..., 1] - radii) - tolerance, -np.inf)
        depth = np.maximum(depth, box_depth)
    frame_depth = depth.max(axis=1)
    over = frame_depth > config.penetration_report_m
    gaps = []
    for patch in patches:
        level = min(levels, key=lambda value: abs(value - patch["height_m"])) if levels else None
        if level is None or abs(level - patch["height_m"]) > config.level_match_m:
            continue
        for x, z in patch["polygon_xz_m"]:
            surface = _surface_height(np.array([x, z]), boxes)
            gaps.append(patch["height_m"] - surface if surface is not None else np.nan)
    gaps = np.asarray(gaps, dtype=np.float64)
    finite = gaps[np.isfinite(gaps)]
    return {
        "penetration_max_m": float(max(frame_depth.max(), 0.0)),
        "penetration_p99_m": float(max(np.percentile(frame_depth, 99), 0.0)),
        "penetration_frames_over_report": int(over.sum()),
        "support_evidence_points": int(len(gaps)),
        "support_evidence_unsupported": int((~np.isfinite(gaps)).sum()),
        "support_gap_abs_p95_m": float(np.percentile(np.abs(finite), 95)) if len(finite) else None,
    }


def _surface_height(point_xz, boxes):
    best = None
    for box in boxes:
        center = np.asarray(box["center_xyz_m"])
        half = np.asarray(box["half_extent_xyz_m"])
        rotation = np.asarray(box["rotation_local_to_world"])
        local = (np.array([point_xz[0], center[1], point_xz[1]]) - center) @ rotation
        if abs(local[0]) <= half[0] + 1e-6 and abs(local[2]) <= half[2] + 1e-6:
            best = box["top_m"] if best is None else max(best, box["top_m"])
    return best


def to_scene_manifest(fit, variant="minimal", name="bones_fit"):
    """SceneManifest for the fitted solids (requires torch via scene_geometry_field)."""
    from .scene_geometry_field import SceneBox, SceneManifest
    boxes = [SceneBox(box_id=b["box_id"], name=b["name"], role=b["role"],
                      center_xyz_m=tuple(b["center_xyz_m"]), half_extent_xyz_m=tuple(b["half_extent_xyz_m"]),
                      rotation_local_to_world=tuple(map(tuple, b["rotation_local_to_world"])))
             for b in fit["boxes"][variant]]
    corners = []
    for b in fit["boxes"][variant]:
        center, half = np.asarray(b["center_xyz_m"]), np.asarray(b["half_extent_xyz_m"])
        corners += [center - half.max(), center + half.max()]
    corners = np.asarray(corners)
    low, high = corners.min(axis=0) - 1.0, corners.max(axis=0) + 3.0
    return SceneManifest(name=name, known_min_xyz_m=tuple(low), known_max_xyz_m=tuple(high), boxes=tuple(boxes))


def _actor(filename):
    import re
    match = re.search(r"__(A\d+)", filename)
    return match.group(1) if match else None


def actor_scale_report(rows, reference_stems=("come_up_50cm_box", "come_down_50cm_box",
                                              "lift_crate_come_up_50cm_box", "lift_crate_come_down_50cm_box"),
                       reference_size_m=0.5):
    """Per-actor world scale of SOMA Uniform data, from the 50 cm box clips.

    Rows need filename, stem, levels_m and dimension_tokens. Uniform retargeting
    scales an actor's world translation, so a 0.5 m box appears as 0.5 * scale.
    Other sized stems are then checked against nominal size * the actor's scale.
    """
    from .bones_terrain_screen import parse_dimension

    spans = {}
    for row in rows:
        if row["stem"] in reference_stems and len(row["levels_m"]) == 2:
            spans.setdefault(_actor(row["filename"]), []).append(row["levels_m"][1])
    actors = {actor: {"clips": len(values), "scale": float(np.median(values) / reference_size_m),
                      "within_actor_std_m": float(np.std(values))}
              for actor, values in sorted(spans.items()) if actor}
    checks = {}
    for row in rows:
        actor = _actor(row["filename"])
        if row["stem"] in reference_stems or actor not in actors or not row["dimension_tokens"]:
            continue
        if len(row["levels_m"]) < 2:
            continue
        size = parse_dimension(row["dimension_tokens"][0])
        predicted = size * actors[actor]["scale"]
        observed = row["levels_m"][-1]
        checks.setdefault(row["stem"], []).append({"filename": row["filename"], "actor": actor,
                                                   "nominal_m": size, "predicted_m": predicted,
                                                   "observed_m": observed, "ratio": observed / predicted})
    scales = np.array([value["scale"] for value in actors.values()]) if actors else np.array([])
    within = np.array([value["within_actor_std_m"] for value in actors.values() if value["clips"] >= 4])
    return {
        "reference": {"stems": list(reference_stems), "size_m": reference_size_m},
        "actors": actors,
        "scale_percentiles": ({k: float(v) for k, v in zip(("p5", "p50", "p95"), np.percentile(scales, [5, 50, 95]))}
                              if len(scales) else None),
        "within_actor_std_m_p50": float(np.median(within)) if len(within) else None,
        "sized_stem_checks": {stem: {"clips": len(items),
                                     "ratio_p50": float(np.median([i["ratio"] for i in items])),
                                     "ratio_min": float(min(i["ratio"] for i in items)),
                                     "ratio_max": float(max(i["ratio"] for i in items))}
                              for stem, items in sorted(checks.items())},
        "note": "ratio = observed top level / (nominal size * actor scale); 1.0 means the nominal props are consistent",
    }
