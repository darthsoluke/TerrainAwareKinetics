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

SCHEMA_VERSION = "bones-terrain-fit-v5"

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
    min_step_m: float = 0.06
    skin_ground_percentile: float = 2.0
    ground_mismatch_m: float = 0.03
    skin_probe_stride: float = 1.0           # every skin vertex is a free-space probe (boxes are exact w.r.t. the metric)
    resting_min_points: int = 10             # skin points under a sole patch during its own contact -> foot rests on body
    slab_clearance_m: float = 0.03           # floating slab bottom above the highest skin that passes beneath it
    min_slab_thickness_m: float = 0.05
    cluster_link_m: float = 0.30
    palm_below_hips_m: float = 0.35
    kneel_shin_drop_m: float = 0.25

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


def _filter_levels(levels, min_step):
    """Drop levels within min_step of the previous kept level (toe-off and foot-pose ghosts)."""
    kept, merged = [], []
    for level in sorted(levels):
        if kept and level - kept[-1] < min_step:
            merged.append(level)
            continue
        kept.append(level)
    return kept, merged


def _limb_evidence(joints, soles, fps, levels, ground, config):
    """Palm and knee rests that plausibly carry load at a support level.

    Palms count only when well below the hips (not hanging or on a rail at waist
    height); knees only when the shin is not vertical (kneeling, not standing).
    """
    events = []
    if not levels:
        return events
    hips = joints.get("Hips")
    for side, sole in (("Left", soles["left"]), ("Right", soles["right"])):
        tracks = []
        if f"{side}Hand" in joints and f"{side}HandMiddleEnd" in joints:
            tracks.append(("palm", 0.5 * (joints[f"{side}Hand"] + joints[f"{side}HandMiddleEnd"]),
                           config.palm_surface_offset_m))
        if f"{side}Shin" in joints:
            tracks.append(("knee", joints[f"{side}Shin"], config.knee_surface_offset_m))
        sole_low = sole[..., 1].min(axis=1)
        for kind, track, offset in tracks:
            for first, last, centroid in _stationary_centroids(track, fps, config):
                surface = centroid[1] - offset
                level = min(levels, key=lambda value: abs(value - surface))
                if abs(level - surface) > config.limb_level_tolerance_m:
                    continue
                window = slice(first, last)
                if kind == "palm" and hips is not None:
                    accepted = np.median(hips[window, 1]) - centroid[1] >= config.palm_below_hips_m
                elif kind == "knee":
                    accepted = np.median(sole_low[window]) > centroid[1] - config.kneel_shin_drop_m
                else:
                    accepted = True
                events.append({"kind": kind, "side": side, "level_m": level, "start_frame": first,
                               "end_frame_exclusive": last, "xz_m": [float(centroid[0]), float(centroid[2])],
                               "accepted": bool(accepted)})
    return events


def _clusters(items, link):
    """Union items (each with an [n, 2] point array) whose point sets come within link metres."""
    parent = list(range(len(items)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i]["points"], items[j]["points"]
            if np.min(np.linalg.norm(a[:, None] - b[None], axis=-1)) <= link:
                parent[find(i)] = find(j)
    groups = {}
    for index in range(len(items)):
        groups.setdefault(find(index), []).append(items[index])
    return list(groups.values())


def _below_probes(flat_points, flat_radii, flat_tolerance, level):
    below = flat_points[:, 1] - flat_radii < level - flat_tolerance
    return flat_points[below][:, [0, 2]], flat_radii[below]


def _conflicts(points_xz, radius, rect_points, margin=0.0):
    center, axis_u, half_u, half_v = oriented_rectangle(rect_points)
    u, v = _local(points_xz, center, axis_u)
    inside = ((u + radius > -half_u - margin) & (u - radius < half_u + margin) &
              (v + radius > -half_v - margin) & (v - radius < half_v + margin))
    return int(inside.sum()), (center, axis_u, half_u, half_v), u, v, inside


def _near_polygon(xz, polygon, margin):
    """Points inside a convex polygon or within margin of its boundary."""
    polygon = np.asarray(polygon, dtype=np.float64)
    inside = np.zeros(len(xz), dtype=bool)
    near = np.zeros(len(xz), dtype=bool)
    for (x0, z0), (x1, z1) in zip(polygon, np.roll(polygon, -1, axis=0)):
        crosses = (z0 > xz[:, 1]) != (z1 > xz[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x0 + (xz[:, 1] - z0) * (x1 - x0) / (z1 - z0)
        inside ^= crosses & (xz[:, 0] < x_cross)
        edge = np.array([x1 - x0, z1 - z0])
        t = np.clip(((xz - [x0, z0]) @ edge) / max(float(edge @ edge), 1e-12), 0, 1)
        near |= np.linalg.norm(xz - ([x0, z0] + t[:, None] * edge), axis=1) <= margin
    return inside | near


def _patch_rests_on_body(patch, skin, events, frame_stride, config):
    """Skin points below a sole patch, inside its footprint, during the patch's own contact.

    If the body lies under the sole while that sole is "standing", the foot is resting on
    its top, toes or on another body part (prone, kneeling), not on a raised surface.
    Any box under such a patch would contain the body, so the patch is not support evidence.
    """
    frames = set()
    for event_id in patch.get("event_ids", []):
        event = events.get(event_id)
        if event is not None:
            frames.update(range(event["start_frame"] // frame_stride,
                                -(-event["end_frame_exclusive"] // frame_stride)))
    frames = sorted(f for f in frames if f < len(skin))
    if not frames:
        return 0
    points = skin[frames].reshape(-1, 3)
    points = points[points[:, 1] < patch["height_m"] - config.level_match_m]
    polygon = np.asarray(patch["polygon_xz_m"], dtype=np.float64)
    lo, hi = polygon.min(axis=0) - config.evidence_margin_m, polygon.max(axis=0) + config.evidence_margin_m
    xz = points[:, [0, 2]]
    xz = xz[np.all((xz >= lo) & (xz <= hi), axis=1)]
    if len(xz) == 0 or len(polygon) < 3:
        return 0
    return int(_near_polygon(xz, polygon, config.evidence_margin_m).sum())


def _inside_rect(u, v, rect, clearance):
    u0, u1, v0, v1 = rect
    return (u > u0 - clearance) & (u < u1 + clearance) & (v > v0 - clearance) & (v < v1 + clearance)


def fit_support(soles, joints, root, fps, support_report, *, config=None, skin_vertices=None,
                skin_frame_stride=1):
    """Fit ground and support boxes to one clip; returns a JSON-ready dict.

    skin_vertices, when given ([skin frames, vertices, 3], every skin_frame_stride-th
    frame of soles), is the authoritative free space and penetration metric: every box
    is grown against all skin points below its top, so no fitted box holds skin deeper
    than the sole tolerance in the measured frames. A support whose footprint has body
    beneath it at some other time becomes a floating slab above that body (ladder rungs,
    ledges); landmark spheres are reported for reference.
    """
    config = config or FitConfig()
    soles = {side: np.asarray(value, dtype=np.float64) for side, value in soles.items()}
    joints = {name: np.asarray(value, dtype=np.float64) for name, value in joints.items()}
    root = np.asarray(root, dtype=np.float64)
    raw_levels = sorted(float(level) for level in support_report["summary"]["usable_level_heights_m"])
    levels, merged_levels = _filter_levels(raw_levels, config.min_step_m)
    patches = [p for p in support_report["patches"] if p["status"] in ("candidate", "weak_support")]
    probes, radii = body_probes(soles, joints)
    sole_count = soles["left"].shape[1] + soles["right"].shape[1]
    tolerance = np.where(np.arange(len(radii)) < sole_count, config.sole_tolerance_m, config.landmark_tolerance_m)
    sole_low = np.minimum(soles["left"][..., 1].min(axis=1), soles["right"][..., 1].min(axis=1))
    ground = levels[0] if levels else float(np.percentile(sole_low, 2))
    ground_source = "lowest_foot_level" if levels else "sole_percentile"
    skin = np.asarray(skin_vertices, dtype=np.float64) if skin_vertices is not None else None
    dropped_ghost_levels = []
    if skin is not None:
        # The floor is the lowest place the body rests. When feet lie on their tops or toes
        # (prone, kneeling), the lowest foot "level" sits above the knees, hands and chest.
        skin_ground = float(np.percentile(skin[..., 1].min(axis=1), config.skin_ground_percentile))
        if not levels or levels[0] - skin_ground > config.ground_mismatch_m:
            dropped_ghost_levels = [level for level in levels if level < skin_ground + config.min_step_m]
            levels = [skin_ground] + [level for level in levels if level >= skin_ground + config.min_step_m]
            ground, ground_source = skin_ground, "skin_lowest_contacts"

    resting = []
    if skin is not None:
        events = {event["id"]: event for event in support_report.get("events", [])}
        kept = []
        for patch in patches:
            count = (_patch_rests_on_body(patch, skin, events, max(1, int(skin_frame_stride)), config)
                     if patch["height_m"] - ground >= config.min_step_m else 0)
            (resting if count >= config.resting_min_points else kept).append(patch)
        patches = kept

        def nearest(height):
            return min(levels, key=lambda level: abs(level - height))
        backed = {nearest(p["height_m"]) for p in patches}
        inverted_levels = [level for level in levels if level > ground and level not in backed
                           and any(nearest(p["height_m"]) == level for p in resting)]
        levels = [level for level in levels if level not in inverted_levels]
    else:
        inverted_levels = []

    def level_of(height):
        if not levels:
            return None
        candidates = [level for level in levels if level <= height + config.level_match_m]
        level = max(candidates) if candidates else levels[0]
        return level if height - level < config.min_step_m else None

    # Evidence items: each foot patch polygon, and each accepted palm/knee rest.
    items = {level: [] for level in levels}
    for patch in patches:
        level = level_of(patch["height_m"])
        if level is not None:
            items[level].append({"kind": "foot", "points": np.asarray(patch["polygon_xz_m"], dtype=np.float64)})
    limb_events = _limb_evidence(joints, soles, fps, levels, ground, config)
    for event in limb_events:
        if event["accepted"] and event["level_m"] != ground:
            items[event["level_m"]].append({"kind": event["kind"], "points": np.asarray([event["xz_m"]])})

    path_low = probes[..., [0, 2]].reshape(-1, 2).min(axis=0) - config.ground_margin_m
    path_high = probes[..., [0, 2]].reshape(-1, 2).max(axis=0) + config.ground_margin_m
    ground_box = {
        "box_id": 1, "name": "ground", "role": "ground",
        "center_xyz_m": [float((path_low[0] + path_high[0]) / 2), ground - config.ground_thickness_m / 2,
                         float((path_low[1] + path_high[1]) / 2)],
        "half_extent_xyz_m": [float((path_high[0] - path_low[0]) / 2), config.ground_thickness_m / 2,
                              float((path_high[1] - path_low[1]) / 2)],
        "rotation_local_to_world": np.eye(3).tolist(), "top_m": ground,
        "bottom_m": ground - config.ground_thickness_m,
    }
    minimal, maximal, level_reports = [ground_box], [ground_box], []
    if skin is not None:
        # With the full skin, free space is the skin itself (subsampled) plus the sole regions;
        # landmark spheres are coarser than the body and are not used.
        sole_points = np.concatenate([soles["left"], soles["right"]], axis=1).reshape(-1, 3)
        skin_points = skin[:, ::max(1, int(config.skin_probe_stride))].reshape(-1, 3)
        flat_points = np.concatenate([sole_points, skin_points])
        skin_flat = skin.reshape(-1, 3)
        flat_radii = np.zeros(len(flat_points))
        flat_tolerance = np.full(len(flat_points), config.sole_tolerance_m)
    else:
        flat_points = probes.reshape(-1, 3)
        flat_radii = np.broadcast_to(radii, probes.shape[:2]).reshape(-1)
        flat_tolerance = np.broadcast_to(tolerance, probes.shape[:2]).reshape(-1)
    bottom = ground - config.ground_thickness_m
    for level in levels:
        if level == ground:
            continue
        report = {"level_m": level, "height_above_ground_m": level - ground,
                  "foot_patches": sum(i["kind"] == "foot" for i in items[level]),
                  "palm_events": sum(i["kind"] == "palm" for i in items[level]),
                  "knee_events": sum(i["kind"] == "knee" for i in items[level]), "clusters": []}
        if not items[level]:
            report["status"] = "no_evidence"
            level_reports.append(report)
            continue
        level_xz, level_radius = _below_probes(flat_points, flat_radii, flat_tolerance, level)
        if skin is not None:
            level_body = skin_flat[skin_flat[:, 1] < level - config.sole_tolerance_m]
        for cluster in _clusters(items[level], config.cluster_link_m):
            # Only probes that a box grown from this cluster could reach matter.
            evidence = np.concatenate([i["points"] for i in cluster])
            reach = config.max_growth_m + config.evidence_margin_m + 0.25
            lo, hi = evidence.min(axis=0) - reach, evidence.max(axis=0) + reach
            near = np.all((level_xz >= lo) & (level_xz <= hi), axis=1)
            below_xz, below_radius = level_xz[near], level_radius[near]
            if skin is not None:
                body_below = level_body[np.all((level_body[:, [0, 2]] >= lo) & (level_body[:, [0, 2]] <= hi), axis=1)]
            attempts = [("all_evidence", cluster),
                        ("foot_only", [i for i in cluster if i["kind"] == "foot"])]
            attempts += [(f"single_item_{k}", [item]) for k, item in enumerate(cluster)]
            chosen = None
            for label, subset in attempts:
                if not subset:
                    continue
                points = np.concatenate([i["points"] for i in subset])
                count, rect, u, v, inside = _conflicts(below_xz, below_radius, points)
                if count == 0:
                    chosen = (label, subset, rect, u, v, below_radius, bottom)
                    break
                if skin is None:
                    continue
                # Body passes beneath this footprint: a floating slab above the highest such skin.
                center, axis_u, half_u, half_v = rect
                bu, bv = _local(body_below[:, [0, 2]], center, axis_u)
                under = _inside_rect(bu, bv, (-half_u, half_u, -half_v, half_v), config.evidence_margin_m)
                if not under.any():
                    continue
                slab_bottom = float(body_below[under, 1].max()) + config.slab_clearance_m
                if level - slab_bottom >= config.min_slab_thickness_m:
                    above = body_below[:, 1] >= slab_bottom
                    chosen = (f"{label}_slab", subset, rect, bu[above], bv[above],
                              np.zeros(int(above.sum())), slab_bottom)
                    break
            entry = {"items": len(cluster), "kinds": sorted({i["kind"] for i in cluster})}
            if chosen is None:
                entry["status"] = "rejected_conflicts_with_body"
                report["clusters"].append(entry)
                continue
            label, subset, (center, axis_u, half_u, half_v), u, v, probe_radius, box_bottom = chosen
            if skin is not None and not label.endswith("_slab"):
                # Exact free space: all skin below the top (sparse probes can miss a heel).
                u, v = _local(body_below[:, [0, 2]], center, axis_u)
                probe_radius = np.zeros(len(u))
            evidence_rect = (-half_u, half_u, -half_v, half_v)
            margin_limit = {"u0": -half_u - config.evidence_margin_m, "u1": half_u + config.evidence_margin_m,
                            "v0": -half_v - config.evidence_margin_m, "v1": half_v + config.evidence_margin_m}
            far_limit = {"u0": -half_u - config.max_growth_m, "u1": half_u + config.max_growth_m,
                         "v0": -half_v - config.max_growth_m, "v1": half_v + config.max_growth_m}
            rect_min = _grow(evidence_rect, u, v, probe_radius, margin_limit, config.side_clearance_m)
            rect_max = _grow(rect_min, u, v, probe_radius, far_limit, config.side_clearance_m)
            if skin is not None and _inside_rect(u, v, rect_min, -config.side_clearance_m / 2).any():
                entry = {"items": len(cluster), "kinds": sorted({i["kind"] for i in cluster}),
                         "status": "rejected_conflicts_with_body"}
                report["clusters"].append(entry)
                continue
            name = f"support_{len(minimal)}"
            frame = {"center_xz": [float(c) for c in center], "axis_u": [float(a) for a in axis_u]}
            box_min = _box_record(len(minimal) + 1, name, "support", center, axis_u, rect_min, level, box_bottom)
            box_max = _box_record(len(maximal) + 1, name, "support", center, axis_u, rect_max, level, box_bottom)
            box_min.update(bottom_m=float(box_bottom), frame=dict(frame, rect=[float(r) for r in rect_min]))
            box_max.update(bottom_m=float(box_bottom), frame=dict(frame, rect=[float(r) for r in rect_max]),
                           frame_max_rect=[float(r) for r in rect_max],
                           evidence_rect=[float(r) for r in evidence_rect])
            minimal.append(box_min)
            maximal.append(box_max)
            entry.update(status="fitted", evidence_used=label, items_used=len(subset), box=name,
                         minimal_size_m=[float(rect_min[1] - rect_min[0]), float(rect_min[3] - rect_min[2])],
                         maximal_size_m=[float(rect_max[1] - rect_max[0]), float(rect_max[3] - rect_max[2])])
            report["clusters"].append(entry)
        report["status"] = "fitted" if any(c["status"] == "fitted" for c in report["clusters"]) else "rejected"
        level_reports.append(report)

    used = [p for p in patches if level_of(p["height_m"]) is not None]
    metrics = {}
    for variant, boxes in (("minimal", minimal), ("maximal", maximal)):
        metrics[variant] = measure(soles, joints, boxes, used, levels, config)
        if skin is not None:
            metrics[variant]["skin"] = measure_points(skin, boxes, 0.0, config.penetration_report_m)
            metrics[variant]["skin_support"] = measure_points(
                skin, [b for b in boxes if b["role"] != "ground"], 0.0, config.penetration_report_m)
            metrics[variant]["skin_ground"] = measure_points(
                skin, [b for b in boxes if b["role"] == "ground"], 0.0, config.penetration_report_m)
    authority = "skin" if skin_vertices is not None else "landmark_spheres"

    def over(metric):
        if "skin_support" in metric:
            return metric["skin_support"]["frames_over_report"]
        return metric["skin"]["frames_over_report"] if "skin" in metric else metric["penetration_frames_over_report"]

    flags = []
    if not levels:
        flags.append("no_usable_support")
    if merged_levels:
        flags.append("merged_small_steps")
    if ground_source == "skin_lowest_contacts":
        flags.append("ground_from_skin")
    if inverted_levels:
        flags.append("inverted_foot_level_dropped")
    if resting:
        flags.append("foot_patches_rest_on_body")
    if any(c.get("evidence_used", "").endswith("_slab") for r in level_reports for c in r["clusters"]):
        flags.append("floating_slab")
    if any(c["status"] != "fitted" or c.get("evidence_used") != "all_evidence"
           for r in level_reports for c in r["clusters"]):
        flags.append("evidence_reduced_for_body_conflict")
    if any(c["status"] == "rejected_conflicts_with_body" for r in level_reports for c in r["clusters"]):
        flags.append("support_rejected")
    if any(over(m) for m in metrics.values()):
        flags.append("penetration_over_report")
    return {
        "schema_version": SCHEMA_VERSION, "ground_m": ground,
        "levels_m": [level - ground for level in levels], "merged_levels_m": [m - ground for m in merged_levels],
        "ground_source": ground_source, "dropped_ghost_levels_m": [g - ground for g in dropped_ghost_levels],
        "inverted_foot_levels_m": [g - ground for g in inverted_levels],
        "patches_resting_on_body": len(resting),
        "evidence_patches": [{"height_m": float(p["height_m"]), "level_m": float(level_of(p["height_m"])),
                              "polygon_xz_m": p["polygon_xz_m"]} for p in used],
        "levels": level_reports, "limb_events": limb_events, "boxes": {"minimal": minimal, "maximal": maximal},
        "metrics": metrics, "penetration_authority": authority, "flags": flags, "config": asdict(config),
        "protocol": {"source_changed": False, "coordinates": "source world, Y up, metres",
                     "geometry": "ground slab plus solid boxes from the ground up to each support level",
                     "status": "weak-evidence fit; not measured terrain"},
    }


def measure_points(points, boxes, tolerance, report_threshold):
    """Vertical depth of points inside any box: distance to the nearer of its top and bottom.

    Boxes standing on the ground reduce to depth below the top; for a floating slab a
    point passing beneath it is outside and does not count."""
    frame_depth = np.full(points.shape[0], -np.inf)
    for box in boxes:
        center = np.asarray(box["center_xyz_m"])
        half = np.asarray(box["half_extent_xyz_m"])
        rotation = np.asarray(box["rotation_local_to_world"])
        for start in range(0, points.shape[0], 64):
            chunk = points[start:start + 64]
            local = (chunk - center) @ rotation
            within = (np.abs(local[..., 0]) <= half[0]) & (np.abs(local[..., 2]) <= half[2])
            bottom = box.get("bottom_m", -np.inf)
            within &= chunk[..., 1] >= bottom
            depth = np.where(within, np.minimum(box["top_m"] - chunk[..., 1], chunk[..., 1] - bottom) - tolerance,
                             -np.inf).max(axis=1)
            frame_depth[start:start + 64] = np.maximum(frame_depth[start:start + 64], depth)
    frame_depth = np.maximum(frame_depth, 0.0)
    return {"max_m": float(frame_depth.max(initial=0.0)),
            "p99_m": float(np.percentile(frame_depth, 99)) if len(frame_depth) else 0.0,
            "frames_over_report": int((frame_depth > report_threshold).sum()),
            "frames": int(len(frame_depth))}


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
