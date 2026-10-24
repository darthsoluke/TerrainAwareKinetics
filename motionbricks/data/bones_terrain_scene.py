"""Overhead and side geometry, and scene variants, for BONES-SEED clips (steps 4-5).

Source motion is fixed. Given the full official skin of one clip (every n-th frame),
its per-frame posture labels and the support fit (bones_terrain_fit), this module

1. builds a body envelope: per XZ cell, the highest skin point at any frame;
2. fits a ceiling that explains low postures: above the envelope everywhere it
   covers, and below the top of the next-higher posture, so that the next-higher
   posture would not fit under it. Two heights are kept: tight (just above the
   body) and loose (just below the explanation bound);
3. fits walls where the still body presses a flat region against a vertical plane;
4. samples scene variants inside behaviour-neutral limits, adds high ceilings to
   upright clips and distant clutter to every clip, so geometry type does not
   predict the motion class (shortcut breaking), and measures full-skin
   penetration of every variant.

All heights are in the source world (Y up, metres; SOMA Uniform scale). Every
threshold is a candidate value recorded in the outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .bones_terrain_fit import _grow, _local, oriented_rectangle

SCHEMA_VERSION = "bones-terrain-scene-v4"
LOW_POSTURES = ("crouch", "other_low", "quadruped", "prone")


@dataclass(frozen=True)
class SceneConfig:
    cell_m: float = 0.10
    ceiling_clearance_m: float = 0.04        # tight ceiling above the body envelope
    explanation_margin_m: float = 0.05       # ceiling at least this far below the next-higher posture top
    crouch_top_ratio: float = 0.62           # crouch top / standing top (v3 tight-ceiling P5 calibration)
    quadruped_top_ratio: float = 0.40        # quadruped top / standing top (v3 tight-ceiling P5 calibration)
    in_place_blocked_fraction: float = 0.2   # low hips on blocked cells above this -> in-place transition
    min_low_seconds: float = 0.5
    ceiling_thickness_m: float = 0.20
    ceiling_growth_m: float = 0.60
    cluster_link_m: float = 0.30
    wall_still_speed_mps: float = 0.20
    wall_min_seconds: float = 0.5
    wall_contact_band_m: float = 0.015
    wall_min_contact_vertices: float = 80.0
    wall_offset_stability_m: float = 0.02
    wall_min_trunk_vertices: float = 60.0
    wall_min_hand_vertices: float = 25.0
    wall_hand_reach_m: float = 0.35
    wall_hand_min_height_m: float = 0.85     # bracing hand at least this high (hands on knees are not walls)
    wall_min_feet_distance_m: float = 0.15
    wall_max_shift_m: float = 0.05
    wall_contact_drift_m: float = 0.10       # touching skin stays at one place on the plane (P10-P90 range)
    wall_height_m: float = 2.5
    wall_thickness_m: float = 0.20
    wall_growth_m: float = 1.0
    # Behaviour-neutral limits for sampled geometry.
    neutral_side_margin_m: float = 0.30      # support box sides at least this far beyond the evidence
    neutral_headroom_m: float = 0.15         # ceilings over upright motion above envelope and standing top
    neutral_clutter_clearance_m: float = 0.30
    lookahead_seconds: float = 1.0
    upright_ceiling_probability: float = 0.5
    upright_ceiling_max_m: float = 3.0
    clutter_max: float = 4.0
    clutter_area_m: float = 3.0
    penetration_report_m: float = 0.02
    coverage_tolerance_m: float = 0.03       # surface under each foot patch within this of its level
    min_explanation_rate: float = 0.9        # accepted variants with sustained low posture

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


# ---------------------------------------------------------------- geometry helpers

def make_box(box_id, name, role, center_xz, axis_u, rect, bottom, top):
    """OBB record compatible with SceneBox: local X along axis_u, Y up."""
    u0, u1, v0, v1 = rect
    axis_u = np.asarray(axis_u, dtype=np.float64)
    axis_v = np.array([-axis_u[1], axis_u[0]])
    mid = np.asarray(center_xz) + (u0 + u1) / 2 * axis_u + (v0 + v1) / 2 * axis_v
    x_axis = np.array([axis_u[0], 0.0, axis_u[1]])
    z_axis = np.cross(x_axis, [0.0, 1.0, 0.0])
    return {"box_id": int(box_id), "name": name, "role": role,
            "center_xyz_m": [float(mid[0]), float((top + bottom) / 2), float(mid[1])],
            "half_extent_xyz_m": [float((u1 - u0) / 2), float((top - bottom) / 2), float((v1 - v0) / 2)],
            "rotation_local_to_world": np.stack([x_axis, [0.0, 1.0, 0.0], z_axis], axis=1).tolist(),
            "top_m": float(top), "bottom_m": float(bottom),
            "frame": {"center_xz": [float(c) for c in center_xz], "axis_u": [float(a) for a in axis_u],
                      "rect": [float(r) for r in rect]}}


def frame_bounds(points):
    """Per-frame axis-aligned bounds [frames, 2, 3] of points [frames, n, 3]."""
    return np.stack([points.min(axis=1), points.max(axis=1)], axis=1)


def _box_depth(points, box, bounds):
    """Per-frame deepest penetration of points into one OBB (0 when outside)."""
    center = np.asarray(box["center_xyz_m"], dtype=np.float64)
    half = np.asarray(box["half_extent_xyz_m"], dtype=np.float64)
    rotation = np.asarray(box["rotation_local_to_world"], dtype=np.float64)
    reach = np.abs(rotation) @ half                      # world AABB half extents of the box
    lo, hi = center - reach, center + reach
    depth = np.zeros(points.shape[0])
    touching = np.flatnonzero(np.all(bounds[:, 1] >= lo, axis=1) & np.all(bounds[:, 0] <= hi, axis=1))
    for start in range(0, len(touching), 32):
        frames = touching[start:start + 32]
        chunk = points[frames]
        near = np.all((chunk >= lo) & (chunk <= hi), axis=-1)
        if not near.any():
            continue
        local = np.abs((chunk[near] - center) @ rotation)
        inside = (half - local).min(axis=-1)
        frame_index = np.broadcast_to(np.arange(len(frames))[:, None], near.shape)[near]
        best = np.full(len(frames), -np.inf)
        np.maximum.at(best, frame_index, inside)
        depth[frames] = np.maximum(best, 0.0)
    return depth


def _box_key(box):
    return (tuple(np.round(box["center_xyz_m"], 9)), tuple(np.round(box["half_extent_xyz_m"], 9)),
            tuple(np.round(np.asarray(box["rotation_local_to_world"]).ravel(), 9)))


def solid_penetration(points, boxes, report_threshold, roles=None, *, bounds=None, cache=None):
    """Full-volume OBB penetration of points [frames, n, 3]: depth to the nearest face.

    Frames whose point bounds miss a box are skipped, and only points inside the box's
    world bounds are transformed. ``cache`` (a dict) reuses per-box depths across calls
    on the same points, e.g. the fitted boxes shared by every scene variant.
    """
    bounds = frame_bounds(points) if bounds is None else bounds
    frame_depth = np.zeros(points.shape[0])
    worst = {}
    for box in boxes:
        if roles and box["role"] not in roles:
            continue
        key = _box_key(box) if cache is not None else None
        if key is not None and key in cache:
            box_depth = cache[key]
        else:
            box_depth = _box_depth(points, box, bounds)
            if key is not None:
                cache[key] = box_depth
        frame_depth = np.maximum(frame_depth, box_depth)
        worst[box["name"]] = float(box_depth.max(initial=0.0))
    return {"max_m": float(frame_depth.max(initial=0.0)),
            "frames_over_report": int((frame_depth > report_threshold).sum()),
            "frames": int(points.shape[0]), "per_box_max_m": worst}


class Grid:
    """Fixed XZ grid over a clip, origin at the lower corner."""

    def __init__(self, points_xz, cell, margin):
        self.cell = float(cell)
        self.origin = points_xz.min(axis=0) - margin
        self.shape = tuple(np.ceil((points_xz.max(axis=0) + margin - self.origin) / cell).astype(int) + 1)

    def index(self, xz):
        ij = np.floor((np.asarray(xz) - self.origin) / self.cell).astype(int)
        return np.clip(ij, 0, np.array(self.shape) - 1)

    def centers(self, mask):
        ij = np.argwhere(mask)
        return self.origin + (ij + 0.5) * self.cell


def envelope(skin, grid, frame_mask=None):
    """Per-cell maximum skin height over the selected frames (-inf where never visited)."""
    top = np.full(grid.shape, -np.inf)
    points = skin if frame_mask is None else skin[frame_mask]
    if len(points) == 0:
        return top
    flat = points.reshape(-1, 3)
    ij = grid.index(flat[:, [0, 2]])
    np.maximum.at(top, (ij[:, 0], ij[:, 1]), flat[:, 1])
    return top


def floor_height(xz, support_boxes, ground):
    """Highest support top whose footprint contains xz, else the ground."""
    xz = np.atleast_2d(xz)
    result = np.full(len(xz), ground, dtype=np.float64)
    for box in support_boxes:
        if box["role"] != "support":
            continue
        center = np.asarray(box["center_xyz_m"])
        half = np.asarray(box["half_extent_xyz_m"])
        rotation = np.asarray(box["rotation_local_to_world"])
        local = (np.column_stack([xz[:, 0], np.full(len(xz), center[1]), xz[:, 1]]) - center) @ rotation
        inside = (np.abs(local[:, 0]) <= half[0]) & (np.abs(local[:, 2]) <= half[2])
        result = np.where(inside, np.maximum(result, box["top_m"]), result)
    return result


def _clusters_of_points(points, link):
    parent = list(range(len(points)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    if len(points) > 1:
        from scipy.spatial import cKDTree
        for i, j in cKDTree(points).query_pairs(link):
            parent[find(i)] = find(j)
    groups = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    return [points[indices] for indices in sorted(groups.values(), key=len, reverse=True)]


# ---------------------------------------------------------------- ceilings

def explanation_bounds(standing_top, config):
    """Ceiling height above the floor that each low posture must stay under (next-higher top - margin)."""
    crouch_top = config.crouch_top_ratio * standing_top
    quadruped_top = config.quadruped_top_ratio * standing_top
    return {"crouch": standing_top - config.explanation_margin_m,
            "other_low": standing_top - config.explanation_margin_m,
            "quadruped": crouch_top - config.explanation_margin_m,
            "prone": quadruped_top - config.explanation_margin_m}


def fit_ceiling(skin, posture, hips_xz, support_boxes, ground, standing_top, fps, config=None, box_id=100):
    """Ceiling slab over the low-posture region, or a report of why none fits."""
    config = config or SceneConfig()
    posture = np.asarray(posture).astype(str)
    low = np.isin(posture, LOW_POSTURES)
    report = {"low_frames": int(low.sum()), "frames": int(len(posture))}
    if low.sum() / fps < config.min_low_seconds:
        report["status"] = "no_sustained_low_posture"
        return None, report
    counts = {name: int((posture == name).sum()) for name in LOW_POSTURES}
    dominant = max(counts, key=counts.get)
    bounds = explanation_bounds(standing_top, config)
    grid = Grid(skin[..., [0, 2]].reshape(-1, 2), config.cell_m, config.ceiling_growth_m + 0.5)
    top_all = envelope(skin, grid)
    top_low = envelope(skin, grid, low)
    centers_all = grid.centers(np.ones(grid.shape, dtype=bool))
    floor = floor_height(centers_all, support_boxes, ground).reshape(grid.shape)
    bound = floor + bounds[dominant]
    core_mask = np.isfinite(top_low) & (top_all + config.ceiling_clearance_m <= bound)
    report.update(dominant=dominant, posture_counts=counts, bound_above_floor_m=bounds[dominant],
                  low_cells=int(np.isfinite(top_low).sum()), core_cells=int(core_mask.sum()))
    if not core_mask.any():
        report["status"] = "unexplainable_body_reaches_bound"
        return None, report
    floor_ref = float(np.median(floor[core_mask]))
    h_tight = float((top_all[core_mask]).max() + config.ceiling_clearance_m)
    h_loose = floor_ref + bounds[dominant]
    if h_tight > h_loose:
        report["status"] = "unexplainable_tight_above_bound"
        return None, report

    def build(height, name):
        blocked = np.isfinite(top_all) & (top_all + config.ceiling_clearance_m > height)
        blocked_xz = grid.centers(blocked)
        radius = np.full(len(blocked_xz), config.cell_m * 0.71)
        core_points = grid.centers(core_mask & ~blocked)
        attempts = [("all_core", core_points)] + [
            (f"cluster_{k}", points) for k, points in enumerate(_clusters_of_points(core_points, config.cluster_link_m))]
        attempts.append(("hip_path", np.asarray(hips_xz)[low]))
        for label, points in attempts:
            if len(points) == 0:
                continue
            center, axis_u, half_u, half_v = oriented_rectangle(points, minimum_side_m=config.cell_m)
            u, v = _local(blocked_xz, center, axis_u)
            inside = ((u + radius > -half_u) & (u - radius < half_u) & (v + radius > -half_v) & (v - radius < half_v))
            if inside.any():
                continue
            core_rect = (-half_u, half_u, -half_v, half_v)
            limit = {"u0": -half_u - config.ceiling_growth_m, "u1": half_u + config.ceiling_growth_m,
                     "v0": -half_v - config.ceiling_growth_m, "v1": half_v + config.ceiling_growth_m}
            grown = _grow(core_rect, u, v, radius, limit, 0.0)
            box = make_box(box_id, name, "ceiling", center, axis_u, grown, height, height + config.ceiling_thickness_m)
            box["core_rect"] = [float(r) for r in core_rect]
            return box, label
        return None, None

    tight, tight_used = build(h_tight, "ceiling_tight")
    loose, loose_used = build(h_loose, "ceiling_loose")
    if tight is None:
        # The actor stood up (or knelt down from standing) where the low posture happened:
        # no single static ceiling can explain both, so report it rather than call it a fit failure.
        blocked = np.isfinite(top_all) & (top_all + config.ceiling_clearance_m > h_tight)
        cells = grid.index(np.asarray(hips_xz)[low])
        fraction = float(blocked[cells[:, 0], cells[:, 1]].mean())
        report["low_hips_on_blocked_cells"] = fraction
        report["status"] = ("unexplainable_in_place_transition"
                            if fraction > config.in_place_blocked_fraction else "no_conflict_free_rectangle")
        return None, report
    for box, label in ((tight, tight_used), (loose, loose_used)):
        if box is not None:
            box["core_used"] = label
    report.update(status="fitted", tight_m=h_tight - floor_ref, loose_m=h_loose - floor_ref, floor_m=floor_ref)
    report["explanation"] = {name: explanation_rate(box, posture, hips_xz, floor_ref, bounds)
                             for name, box in (("tight", tight), ("loose", loose)) if box is not None}
    return {"tight": tight, "loose": loose, "floor_m": floor_ref,
            "range_above_floor_m": [h_tight - floor_ref, h_loose - floor_ref]}, report


def _inside_footprint(box, xz):
    center = np.asarray(box["center_xyz_m"])
    half = np.asarray(box["half_extent_xyz_m"])
    rotation = np.asarray(box["rotation_local_to_world"])
    xz = np.atleast_2d(xz)
    local = (np.column_stack([xz[:, 0], np.full(len(xz), center[1]), xz[:, 1]]) - center) @ rotation
    return (np.abs(local[:, 0]) <= half[0]) & (np.abs(local[:, 2]) <= half[2])


def explanation_rate(ceiling, posture, hips_xz, floor_ref, bounds):
    """Fraction of low-posture frames under the ceiling and below their own posture bound."""
    posture = np.asarray(posture).astype(str)
    low = np.isin(posture, LOW_POSTURES)
    if not low.any():
        return None
    under = _inside_footprint(ceiling, np.asarray(hips_xz)[low])
    height = ceiling["bottom_m"] - floor_ref
    below = np.array([height <= bounds[name] for name in posture[low]])
    return float((under & below).mean())


# ---------------------------------------------------------------- walls

PART_OTHER, PART_TRUNK, PART_HAND, PART_FOOT = 0, 1, 2, 3


def part_of_joint(name):
    """Body-part code of a rig joint by name: trunk (pelvis to shoulders), hand, foot or other."""
    side = name[4:] if name.startswith("Left") else name[5:] if name.startswith("Right") else name
    if "Hand" in side or "Thumb" in side or "Index" in side or "Middle" in side or "Ring" in side or "Pinky" in side:
        return PART_HAND
    if "Foot" in side or "Toe" in side or "Ankle" in side:
        return PART_FOOT
    if side.startswith(("Hips", "Spine", "Chest", "Shoulder")) or side in ("Arm", "UpperArm"):
        return PART_TRUNK
    return PART_OTHER


def vertex_part_codes(dominant_joint, joint_names):
    """Per-vertex part code from each vertex's dominant skinning joint."""
    codes = np.array([part_of_joint(name) for name in joint_names], dtype=np.int8)
    dominant_joint = np.asarray(dominant_joint)
    if dominant_joint.min() < 0 or dominant_joint.max() >= len(codes):
        raise ValueError("dominant joint index outside the joint name list")
    return codes[dominant_joint]


def _contact_drift(touching, tangent, height):
    """P10-P90 range of the per-frame centroid of touching skin on the plane (tangent, height).

    A hand or back pressed on a wall stays put on it; the outer edge of a turning or
    walking body slides along the plane. None when fewer than half the frames touch.
    """
    counts = touching.sum(axis=1)
    frames = counts > 0
    if frames.mean() < 0.5:
        return None
    t = (np.where(touching, tangent, 0).sum(axis=1)[frames] / counts[frames])
    y = (np.where(touching, height, 0).sum(axis=1)[frames] / counts[frames])
    spread = [float(np.percentile(v, 90) - np.percentile(v, 10)) for v in (t, y)]
    return max(spread)


def fit_walls(skin, hips_xz, ground, fps, config=None, box_id=200, directions=24, vertex_parts=None):
    """Vertical walls where the still body leans on, or reaches out to, one plane.

    A wall needs (a) a stable outermost plane during a still segment, (b) contact by the
    trunk (leaning) or by a hand held away from the hips (bracing), and (c) feet clearly
    away from the plane, which is what makes the plane carry load. vertex_parts gives a
    body-part code per skin vertex (PART_*); without it, two height bands of contact are
    required instead. Candidates are tried in score order; a plane that must move more
    than wall_max_shift_m to clear the body anywhere is rejected.
    """
    config = config or SceneConfig()
    hips_xz = np.asarray(hips_xz)
    speed = np.r_[0.0, np.linalg.norm(np.diff(hips_xz, axis=0), axis=-1) * fps]
    still = speed <= config.wall_still_speed_mps
    report = {"still_frames": int(still.sum()), "rejected": {}}
    if still.sum() / fps < config.wall_min_seconds:
        report["status"] = "no_still_segment"
        return [], report
    height = skin[..., 1] - ground
    band = (height > 0.2) & (height < 2.0)
    xz = skin[..., [0, 2]]
    parts = None if vertex_parts is None else np.asarray(vertex_parts)
    if parts is not None and parts.shape != (skin.shape[1],):
        raise ValueError("vertex_parts must give one code per skin vertex")
    feet = (parts == PART_FOOT) if parts is not None else None
    candidates = []
    for k in range(directions):
        angle = 2 * np.pi * k / directions
        d = np.array([np.cos(angle), np.sin(angle)])
        projection = np.where(band, xz @ d, -np.inf)
        offset = projection.max(axis=1)
        valid = still & np.isfinite(offset)
        if valid.sum() / fps < config.wall_min_seconds:
            continue
        median = float(np.median(offset[valid]))
        stable = valid & (np.abs(offset - median) <= config.wall_offset_stability_m)
        if stable.sum() / fps < config.wall_min_seconds:
            continue
        contact = (projection >= median - config.wall_contact_band_m) & band
        if feet is not None:
            foot_reach = (xz[:, feet] @ d).max(axis=1)
        else:
            low = height <= np.percentile(height, 5, axis=1, keepdims=True)
            foot_reach = np.where(low, xz @ d, -np.inf).max(axis=1)
        feet_distance = float(np.median(median - foot_reach[stable]))
        hips_distance = float(np.median(median - hips_xz[stable] @ d))
        if parts is not None:
            trunk = (contact & (parts == PART_TRUNK))[stable].sum(axis=1)
            hand = (contact & (parts == PART_HAND) & (height >= config.wall_hand_min_height_m))[stable].sum(axis=1)
            leaning = np.median(trunk) >= config.wall_min_trunk_vertices
            bracing = (np.median(hand) >= config.wall_min_hand_vertices
                       and hips_distance >= config.wall_hand_reach_m)
            kind = "lean" if leaning else "brace" if bracing else None
            score = float(np.median(trunk) + np.median(hand))
            touching = contact & ((parts == PART_TRUNK) if kind == "lean" else
                                  (parts == PART_HAND) & (height >= config.wall_hand_min_height_m))
        else:
            bands = [((height >= lo) & (height < hi) & contact)[stable].sum(axis=1) >= 30
                     for lo, hi in ((0.2, 0.8), (0.8, 1.4), (1.4, 2.0))]
            two_bands = (np.stack(bands).sum(axis=0) >= 2).mean() >= 0.5
            kind = "contact" if two_bands and np.median(contact[stable].sum(axis=1)) >= config.wall_min_contact_vertices else None
            score = float(np.median(contact[stable].sum(axis=1)))
            touching = contact
        drift = _contact_drift(touching[stable], xz[stable] @ np.array([-d[1], d[0]]), height[stable])
        reason = None
        if kind is None:
            reason = "no_trunk_or_bracing_hand_contact"
        elif feet_distance < config.wall_min_feet_distance_m:
            reason = "feet_at_the_plane"
        elif drift is None or drift > config.wall_contact_drift_m:
            reason = "contact_slides_along_plane"
        if reason:
            report["rejected"][reason] = report["rejected"].get(reason, 0) + 1
            continue
        candidates.append({"direction": d, "offset": median, "score": score, "kind": kind, "contact_drift_m": drift,
                           "stable_seconds": float(stable.sum() / fps), "feet_distance_m": feet_distance,
                           "hips_distance_m": hips_distance})
    for best in sorted(candidates, key=lambda c: -c["score"]):
        d = best["direction"]
        tangent = np.array([-d[1], d[0]])
        projection = xz @ d
        t_all = xz @ tangent
        near = band & (projection >= best["offset"] - config.wall_contact_band_m)
        t0, t1 = float(t_all[near].min()), float(t_all[near].max())
        beyond = band & (projection > best["offset"] + 0.005)
        left, right = t_all[beyond & (t_all < t0)], t_all[beyond & (t_all > t1)]
        t0_grown = max(t0 - config.wall_growth_m, float(left.max()) + 0.05) if len(left) else t0 - config.wall_growth_m
        t1_grown = min(t1 + config.wall_growth_m, float(right.min()) - 0.05) if len(right) else t1 + config.wall_growth_m
        span = band & (t_all >= t0_grown) & (t_all <= t1_grown)
        plane = max(best["offset"], float(projection[span].max()) + 0.005) if span.any() else best["offset"]
        if plane - best["offset"] > config.wall_max_shift_m:
            report["rejected"]["plane_must_move_to_clear_body"] = report["rejected"].get(
                "plane_must_move_to_clear_body", 0) + 1
            continue
        center_xz = d * (plane + config.wall_thickness_m / 2)
        rect = (-config.wall_thickness_m / 2, config.wall_thickness_m / 2, t0_grown, t1_grown)
        wall = make_box(box_id, "wall_1", "wall", center_xz, d, rect, ground - 0.2, ground + config.wall_height_m)
        report.update(status="fitted", kind=best["kind"], score=best["score"], contact_drift_m=best["contact_drift_m"],
                      stable_seconds=best["stable_seconds"], feet_distance_m=best["feet_distance_m"],
                      contact_span_m=t1 - t0, length_m=t1_grown - t0_grown, plane_shift_m=plane - best["offset"])
        return [wall], report
    report["status"] = "no_contact_plane"
    return [], report


# ---------------------------------------------------------------- variants

def lookahead_zone(hips_xz, fps, config):
    """Points the policy would look at past the clip end: continue the final velocity."""
    hips_xz = np.asarray(hips_xz)
    window = max(2, int(fps * 0.5))
    velocity = (hips_xz[-1] - hips_xz[-window]) * fps / (window - 1)
    steps = np.linspace(0, config.lookahead_seconds, 11)[:, None]
    return hips_xz[-1] + steps * velocity


def _sample_support(box, rng, config):
    frame = box["frame"]
    rect_max = box["frame_max_rect"]
    evidence = box["evidence_rect"]
    sampled = []
    for index, sign in ((0, -1), (1, 1), (2, -1), (3, 1)):
        neutral = evidence[index] + sign * config.neutral_side_margin_m
        limit = rect_max[index]
        low, high = sorted((neutral if sign * (neutral - limit) <= 0 else limit, limit))
        sampled.append(float(rng.uniform(low, high)) if high > low else float(limit))
    return sampled, frame


def sample_variant(rng, fit, ceiling, walls, top_all_grid, grid, hips_xz, upright, standing_top, fps,
                   config=None, next_id=300):
    """One scene: support boxes, fitted ceiling/walls, optional upright ceiling and clutter."""
    config = config or SceneConfig()
    boxes, params = [], {"support_sides": [], "ceiling_m": None, "upright_ceiling_m": None, "clutter": 0}
    ground = fit["ground_m"]
    for box in fit["boxes"]["maximal"]:
        if box["role"] == "ground":
            boxes.append(box)
        elif box["role"] == "support" and "frame_max_rect" in box:
            rect, frame = _sample_support(box, rng, config)
            boxes.append(make_box(box["box_id"], box["name"], "support", frame["center_xz"], frame["axis_u"],
                                  rect, box["bottom_m"], box["top_m"]))
            params["support_sides"].append(rect)
        elif box["role"] == "support":
            boxes.append(box)
    if ceiling is not None:
        low_h, high_h = ceiling["tight"]["bottom_m"], (ceiling["loose"] or ceiling["tight"])["bottom_m"]
        height = float(rng.uniform(low_h, max(low_h, high_h)))
        frame = ceiling["tight"]["frame"]
        boxes.append(make_box(100, "ceiling", "ceiling", frame["center_xz"], frame["axis_u"], frame["rect"],
                              height, height + config.ceiling_thickness_m))
        params["ceiling_m"] = height - ceiling["floor_m"]
    boxes.extend(walls)
    occupied = np.isfinite(top_all_grid)
    if upright and rng.uniform() < config.upright_ceiling_probability:
        minimum = max(float(top_all_grid[occupied].max()) + config.neutral_headroom_m,
                      ground + standing_top + config.neutral_headroom_m)
        if minimum < ground + config.upright_ceiling_max_m:
            height = float(rng.uniform(minimum, ground + config.upright_ceiling_max_m))
            points = grid.centers(occupied)
            center, axis_u, half_u, half_v = oriented_rectangle(points)
            rect = (-half_u - 1.0, half_u + 1.0, -half_v - 1.0, half_v + 1.0)
            boxes.append(make_box(101, "upright_ceiling", "ceiling", center, axis_u, rect,
                                  height, height + config.ceiling_thickness_m))
            params["upright_ceiling_m"] = height - ground
    # Clutter: boxes outside the body envelope plus a clearance, and outside the lookahead zone.
    keep_out = np.concatenate([grid.centers(occupied), lookahead_zone(hips_xz, fps, config)])
    from scipy.spatial import cKDTree
    tree = cKDTree(keep_out)
    low_corner = keep_out.min(axis=0) - config.clutter_area_m
    high_corner = keep_out.max(axis=0) + config.clutter_area_m
    count = int(rng.integers(0, int(config.clutter_max) + 1))
    placed = 0
    for attempt in range(count * 20):
        if placed >= count:
            break
        size = rng.uniform([0.3, 0.3], [1.5, 1.5])
        height = float(rng.uniform(0.3, 2.5))
        center = rng.uniform(low_corner, high_corner)
        angle = rng.uniform(0, np.pi)
        axis_u = np.array([np.cos(angle), np.sin(angle)])
        counts = np.ceil(size / 0.1).astype(int) + 1  # footprint samples no more than 0.1 m apart
        samples = np.array([[a, b] for a in np.linspace(-size[0] / 2, size[0] / 2, counts[0])
                            for b in np.linspace(-size[1] / 2, size[1] / 2, counts[1])])
        world = center + samples[:, :1] * axis_u + samples[:, 1:] * np.array([-axis_u[1], axis_u[0]])
        if tree.query(world)[0].min() < config.neutral_clutter_clearance_m + config.cell_m:
            continue
        rect = (-size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2)
        floor = float(floor_height(center, fit["boxes"]["maximal"], ground)[0])
        boxes.append(make_box(next_id + placed, f"clutter_{placed + 1}", "clutter", center, axis_u, rect,
                              floor - 0.2, floor + height))
        placed += 1
    params["clutter"] = placed
    return boxes, params


def surface_height(xz, boxes):
    """Highest top among ground/support boxes whose footprint contains each point (-inf if none)."""
    xz = np.atleast_2d(np.asarray(xz, dtype=np.float64))
    height = np.full(len(xz), -np.inf)
    for box in boxes:
        if box["role"] in ("ground", "support"):
            inside = _inside_footprint(box, xz)
            height[inside] = np.maximum(height[inside], box["top_m"])
    return height


def support_coverage(patches, boxes, tolerance):
    """Foot evidence patches whose centroid stands on a surface at the patch's support level."""
    if not patches:
        return {"patches": 0, "covered": 0, "missing": []}
    centroids = np.array([np.asarray(p["polygon_xz_m"], dtype=np.float64).mean(axis=0) for p in patches])
    levels = np.array([p.get("level_m", p["height_m"]) for p in patches])
    surface = surface_height(centroids, boxes)
    ok = np.abs(surface - levels) <= tolerance
    missing = [{"xz_m": centroids[i].round(3).tolist(), "level_m": float(levels[i]),
                "surface_m": None if not np.isfinite(surface[i]) else float(surface[i])}
               for i in np.flatnonzero(~ok)]
    return {"patches": int(len(patches)), "covered": int(ok.sum()), "missing": missing[:10]}


def scene_for_clip(skin, posture, hips_xz, fit, standing_top, fps, *, variants=4, seed=0, config=None,
                   vertex_parts=None):
    """Ceiling, walls and validated variants for one clip; JSON-ready.

    A variant is accepted only when (1) no non-ground solid enters the skin by more than the
    report threshold, (2) every foot evidence patch stands on a surface at its level, and
    (3) a clip with sustained low posture has a ceiling that explains at least
    min_explanation_rate of its low frames. A clip is training-eligible when any variant is.
    """
    config = config or SceneConfig()
    skin = np.asarray(skin, dtype=np.float64)
    hips_xz = np.asarray(hips_xz, dtype=np.float64)
    posture = np.asarray(posture).astype(str)
    support = [b for b in fit["boxes"]["minimal"] if b["role"] == "support"]
    ground = fit["ground_m"]
    ceiling, ceiling_report = fit_ceiling(skin, posture, hips_xz, support, ground, standing_top, fps, config)
    walls, wall_report = fit_walls(skin, hips_xz, ground, fps, config, vertex_parts=vertex_parts)
    bounds = frame_bounds(skin)
    depth_cache = {}
    checks = {box["name"]: solid_penetration(skin, [box], config.penetration_report_m, bounds=bounds, cache=depth_cache)
              for box in walls}
    if walls and checks[walls[0]["name"]]["frames_over_report"]:
        wall_report.update(status="dropped_penetrates_skin", dropped_max_m=checks[walls[0]["name"]]["max_m"])
        walls = []
    grid = Grid(skin[..., [0, 2]].reshape(-1, 2), config.cell_m, 0.5)
    top_all = envelope(skin, grid)
    upright = bool((posture == "standing").mean() >= 0.9)
    sustained_low = ceiling_report["status"] != "no_sustained_low_posture"
    fitted_boxes = [b for b in (ceiling["tight"], ceiling["loose"]) if b] if ceiling else []
    checks.update({box["name"]: solid_penetration(skin, [box], config.penetration_report_m, bounds=bounds,
                                                  cache=depth_cache)
                   for box in fitted_boxes})
    patches = fit.get("evidence_patches", [])
    rng = np.random.default_rng(seed)
    scenes = []
    for index in range(variants):
        boxes, params = sample_variant(rng, fit, ceiling, walls, top_all, grid, hips_xz, upright,
                                       standing_top, fps, config)
        penetration = solid_penetration(skin, [b for b in boxes if b["role"] != "ground"],
                                        config.penetration_report_m, bounds=bounds, cache=depth_cache)
        ground_box = [b for b in boxes if b["role"] == "ground"]
        ground_pen = solid_penetration(skin, ground_box, config.penetration_report_m, bounds=bounds,
                                       cache=depth_cache)
        explanation = (explanation_rate(next(b for b in boxes if b["name"] == "ceiling"), posture, hips_xz,
                                        ceiling["floor_m"], explanation_bounds(standing_top, config))
                       if ceiling is not None else None)
        coverage = support_coverage(patches, boxes, config.coverage_tolerance_m)
        reasons = []
        if penetration["frames_over_report"]:
            reasons.append("solid_penetrates_skin")
        if coverage["covered"] < coverage["patches"]:
            reasons.append("foot_evidence_not_supported")
        if sustained_low and (explanation is None or explanation < config.min_explanation_rate):
            reasons.append("low_posture_unexplained" if explanation is None else "low_posture_partly_explained")
        scenes.append({"index": index, "params": params, "boxes": boxes,
                       "penetration_m": {"solids": penetration["max_m"], "ground": ground_pen["max_m"]},
                       "frames_over_report": {"solids": penetration["frames_over_report"],
                                              "ground": ground_pen["frames_over_report"]},
                       "explanation_rate": explanation, "support_coverage": coverage,
                       "rejection_reasons": reasons, "accepted": not reasons,
                       "geometry_accepted": not [r for r in reasons if not r.startswith("low_posture")]})
    accepted = [s["index"] for s in scenes if s["accepted"]]
    # Low postures with no environmental cause (e.g. kneeling down in place) keep consistent
    # geometry but must not teach "crouch for no reason"; they are kept apart, not mixed in.
    eligibility = ("environment_explained" if accepted else
                   "geometry_only_low_posture_unexplained" if any(s["geometry_accepted"] for s in scenes) else
                   "rejected")
    envelope_cells = np.argwhere(np.isfinite(top_all))
    return {
        "schema_version": SCHEMA_VERSION, "upright": upright, "standing_top_m": standing_top,
        "ceiling": ceiling, "ceiling_report": ceiling_report, "walls": walls, "wall_report": wall_report,
        "fitted_checks": checks, "variants": scenes, "config": asdict(config),
        "sustained_low_posture": sustained_low, "accepted_variants": accepted,
        "training_eligible": bool(accepted), "eligibility": eligibility,
        "envelope": {"origin_xz": grid.origin.tolist(), "cell_m": grid.cell, "shape": [int(n) for n in grid.shape],
                     "cells": envelope_cells.tolist(),
                     "top_m": np.round(top_all[np.isfinite(top_all)], 3).tolist()},
        "protocol": {"source_changed": False, "penetration": "full official skin, every n-th frame",
                     "neutral_limits": "support sides >= evidence + neutral_side_margin; upright ceilings >= "
                                       "envelope and standing top + neutral_headroom; clutter >= "
                                       "neutral_clutter_clearance from the body envelope and the lookahead zone",
                     "explanation": "ceiling bottom below the next-higher posture top minus explanation_margin",
                     "acceptance": "no solid > report threshold in skin; every foot patch on a surface at its "
                                   "level; sustained low posture explained >= min_explanation_rate"},
    }
