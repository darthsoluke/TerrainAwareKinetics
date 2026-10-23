"""Geometric screening of BONES-SEED clips before terrain fitting.

Input is one clip's source-world motion at a fixed rate: sole vertices from the
official skin and named joint positions. Nothing here moves or edits motion. The
screen answers, per clip, what the motion itself shows:

- support levels: horizontal foot-support heights from stair_support's weak
  evidence (stationary, locally low sole material points that the source never
  passes through), clustered into levels;
- hand and knee support: stationary hand/knee landmarks resting at a foot level;
- posture: per-frame standing / crouch / quadruped / prone / other-low, from the
  hip height above the support level beneath it, torso pitch and contacts;
- head clearance above support, per posture, which bounds any later ceiling;
- known dimensions: support-level span and lowest-body clearance, for stems whose
  file name states a size (50cm box, 0.75m obstacle).

All thresholds are candidates, recorded in every report. A screen result selects
clips for fitting; it is not fitted geometry.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from .stair_support import SupportConfig, reconstruct_support, true_runs

SCHEMA_VERSION = "bones-terrain-screen-v1"
POSTURES = ("standing", "crouch", "quadruped", "prone", "other_low")
LANDMARKS = ("Hips", "Chest", "Neck1", "Head",
             "LeftHand", "LeftHandMiddleEnd", "RightHand", "RightHandMiddleEnd",
             "LeftShin", "RightShin", "LeftForeArm", "RightForeArm")


@dataclass(frozen=True)
class ScreenConfig:
    stationary_speed_mps: float = 0.15
    min_event_seconds: float = 0.12
    hand_level_tolerance_m: float = 0.08
    knee_level_tolerance_m: float = 0.12
    forearm_level_tolerance_m: float = 0.10
    standing_hip_ratio: float = 0.80
    crouch_hip_ratio: float = 0.45
    prone_chest_height_m: float = 0.40
    prone_pitch_deg: float = 60.0
    multi_level_span_m: float = 0.10
    level_merge_m: float = 0.03

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.crouch_hip_ratio >= self.standing_hip_ratio:
            raise ValueError("crouch_hip_ratio must be below standing_hip_ratio")


def parse_dimension(token):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(cm|mm|m)", token)
    if not match:
        raise ValueError(f"Not a dimension token: {token}")
    value, unit = float(match.group(1)), match.group(2)
    return value * {"m": 1.0, "cm": 0.01, "mm": 0.001}[unit]


def _stationary_events(track, fps, config):
    """Runs where one landmark moves slower than the stationary speed."""
    speed = np.linalg.norm(np.diff(track, axis=0), axis=-1) * fps
    slow = speed <= config.stationary_speed_mps  # one value per frame interval
    minimum = max(2, int(np.ceil(config.min_event_seconds * fps)))
    events = []
    for first, last in true_runs(slow):
        # Intervals [first, last) span frames [first, last]; the event is half-open.
        if last + 1 - first >= minimum:
            events.append((int(first), int(last + 1), float(np.median(track[first:last + 1, 1]))))
    return events


def _nearest_level(height, levels):
    if not levels:
        return None, None
    levels = np.asarray(levels)
    index = int(np.argmin(np.abs(levels - height)))
    return float(levels[index]), float(height - levels[index])


def _support_beneath(sole_low, levels, fallback, snap_m=0.05):
    """Per-frame support level under the body, from the lower foot.

    A frame takes the level nearest to its lowest sole point when that point is
    within snap_m of it; otherwise (swing, airborne, hands-only) the last assigned
    level is held. Leading unassigned frames take the first assigned level.
    """
    result = np.full(len(sole_low), np.nan)
    if levels:
        candidates = np.asarray(levels)
        nearest = candidates[np.argmin(np.abs(sole_low[:, None] - candidates[None]), axis=1)]
        close = np.abs(sole_low - nearest) <= snap_m
        result[close] = nearest[close]
    known = np.flatnonzero(np.isfinite(result))
    if not len(known):
        return np.full(len(sole_low), fallback, dtype=np.float64)
    result[:known[0]] = result[known[0]]
    for index in range(known[0] + 1, len(result)):
        if not np.isfinite(result[index]):
            result[index] = result[index - 1]
    return result


def _limb_support(name, track, fps, config, levels, tolerance):
    events = []
    for first, last, height in _stationary_events(track, fps, config):
        level, offset = _nearest_level(height, levels)
        events.append({"landmark": name, "start_frame": first, "end_frame_exclusive": last,
                       "duration_seconds": round((last - first) / fps, 4),
                       "height_m": round(height, 4),
                       "nearest_level_m": None if level is None else round(level, 4),
                       "offset_from_level_m": None if offset is None else round(offset, 4),
                       "at_level": offset is not None and -0.05 <= offset <= tolerance})
    return events


def _longest_run_seconds(mask, fps):
    runs = true_runs(mask)
    return round(max((last - first for first, last in runs), default=0) / fps, 4)


def _percentiles(values):
    if len(values) == 0:
        return None
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return {"p5": round(float(p5), 4), "p50": round(float(p50), 4), "p95": round(float(p95), 4)}


def screen_clip(soles, joints, root, fps, standing_hip_height_m, *,
                dimension_tokens=(), config=None, support_config=None):
    """Screen one clip.

    soles: {"left", "right"} -> [frames, vertices, 3] official sole material points.
    joints: name -> [frames, 3] for every name in LANDMARKS (source world, Y up, m).
    root: [frames, 3]. Returns (row, support_report) without changing any input.
    """
    config = config or ScreenConfig()
    root = np.asarray(root, dtype=np.float64)
    frames = len(root)
    missing = [name for name in LANDMARKS if name not in joints]
    if missing:
        raise ValueError(f"Missing landmarks: {missing}")
    joints = {name: np.asarray(joints[name], dtype=np.float64) for name in LANDMARKS}
    if any(value.shape != (frames, 3) for value in joints.values()):
        raise ValueError("Every landmark must be [frames, 3] and match root frames")
    if not standing_hip_height_m > 0:
        raise ValueError("A positive standing hip height is required")

    # Foot support: reuse the weak-support reconstruction. Its pair-level stair
    # requirements are ignored here; per-patch free-space checks still apply.
    support_config = support_config or SupportConfig(require_stairs=True)
    probes = np.stack([joints[name] for name in LANDMARKS], axis=1)
    _, support = reconstruct_support(soles, root, fps, support_config, probes)
    patches = [p for p in support["patches"] if p["status"] in ("candidate", "weak_support")]
    levels = sorted({round(float(level), 4) for level in support["summary"]["usable_level_heights_m"]})
    rejected = {}
    for patch in support["patches"]:
        for reason in patch["reasons"]:
            rejected[reason] = rejected.get(reason, 0) + 1
    sole_low = np.minimum(soles["left"][..., 1].min(axis=1), soles["right"][..., 1].min(axis=1))
    ground = levels[0] if levels else float(np.percentile(sole_low, 2))
    span = (levels[-1] - levels[0]) if levels else 0.0

    # Hands use the palm midpoint; knees the shin (knee) joint; forearms the elbow.
    palms = {side: 0.5 * (joints[f"{side}Hand"] + joints[f"{side}HandMiddleEnd"])
             for side in ("Left", "Right")}
    limb_events = []
    for side in ("Left", "Right"):
        limb_events += _limb_support(f"{side}Palm", palms[side], fps, config, levels or [ground],
                                     config.hand_level_tolerance_m)
        limb_events += _limb_support(f"{side}Knee", joints[f"{side}Shin"], fps, config, levels or [ground],
                                     config.knee_level_tolerance_m)
        limb_events += _limb_support(f"{side}Elbow", joints[f"{side}ForeArm"], fps, config, levels or [ground],
                                     config.forearm_level_tolerance_m)

    # Posture per frame, relative to the support level the feet stand on.
    beneath = _support_beneath(sole_low, levels, ground)
    hip_ratio = (joints["Hips"][:, 1] - beneath) / standing_hip_height_m
    torso = joints["Neck1"] - joints["Hips"]
    pitch = np.degrees(np.arccos(np.clip(torso[:, 1] / np.maximum(np.linalg.norm(torso, axis=1), 1e-9), -1, 1)))
    chest_height = joints["Chest"][:, 1] - beneath
    hand_low = np.zeros(frames, dtype=bool)
    knee_low = np.zeros(frames, dtype=bool)
    for side in ("Left", "Right"):
        hand_low |= palms[side][:, 1] - beneath <= config.hand_level_tolerance_m
        knee_low |= joints[f"{side}Shin"][:, 1] - beneath <= config.knee_level_tolerance_m
    posture = np.full(frames, "other_low", dtype=object)
    standing = hip_ratio >= config.standing_hip_ratio
    prone = ~standing & (chest_height <= config.prone_chest_height_m) & (pitch >= config.prone_pitch_deg)
    quadruped = ~standing & ~prone & hand_low & (pitch >= 30.0)
    crouch = ~standing & ~prone & ~quadruped & (hip_ratio >= config.crouch_hip_ratio)
    posture[standing] = "standing"
    posture[crouch] = "crouch"
    posture[quadruped] = "quadruped"
    posture[prone] = "prone"
    head = joints["Head"][:, 1] - beneath

    lowest_body = np.minimum(sole_low, probes[..., 1].min(axis=1)) - ground
    row = {
        "schema_version": SCHEMA_VERSION,
        "frames": frames, "fps": float(fps), "duration_seconds": round((frames - 1) / fps, 4),
        "support": {
            "levels_m": [round(level - ground, 4) for level in levels],
            "level_count": len(levels), "level_span_m": round(span, 4),
            "multi_level": span >= config.multi_level_span_m,
            "candidate_patches": len(patches), "rejected_patch_reasons": rejected,
            "root_vertical_span_m": round(float(np.ptp(root[:, 1])), 4),
            "ground_source": "lowest usable level" if levels else "2nd percentile sole height (no usable level)",
        },
        "limb_support": {
            "events": limb_events,
            "palm_seconds_at_level": round(sum(e["duration_seconds"] for e in limb_events
                                               if e["landmark"].endswith("Palm") and e["at_level"]), 4),
            "knee_seconds_at_level": round(sum(e["duration_seconds"] for e in limb_events
                                               if e["landmark"].endswith("Knee") and e["at_level"]), 4),
            "elbow_seconds_at_level": round(sum(e["duration_seconds"] for e in limb_events
                                                if e["landmark"].endswith("Elbow") and e["at_level"]), 4),
            "palm_above_ground_at_level_seconds": round(sum(
                e["duration_seconds"] for e in limb_events if e["landmark"].endswith("Palm")
                and e["at_level"] and e["nearest_level_m"] is not None and e["nearest_level_m"] - ground > 0.05), 4),
        },
        "posture": {
            "fraction": {name: round(float((posture == name).mean()), 4) for name in POSTURES},
            "longest_run_seconds": {name: _longest_run_seconds(posture == name, fps) for name in POSTURES},
            "hip_ratio": _percentiles(hip_ratio),
            "torso_pitch_deg": _percentiles(pitch),
        },
        "head_clearance_m": {name: _percentiles(head[posture == name]) for name in POSTURES
                             if (posture == name).any()},
        "lowest_body_clearance_max_m": round(float(lowest_body.max()), 4),
        "dimensions": [],
    }
    for token in dimension_tokens:
        size = parse_dimension(token)
        row["dimensions"].append({
            "token": token, "size_m": size,
            "level_span_error_m": round(span - size, 4) if levels else None,
            "lowest_body_clearance_minus_size_m": round(float(lowest_body.max()) - size, 4),
        })
    row["flags"] = sorted(flag for flag, active in (
        ("multi_level", row["support"]["multi_level"]),
        ("no_usable_support", not levels),
        ("palm_support", row["limb_support"]["palm_seconds_at_level"] > 0),
        ("palm_support_above_ground", row["limb_support"]["palm_above_ground_at_level_seconds"] > 0),
        ("knee_support", row["limb_support"]["knee_seconds_at_level"] > 0),
        ("low_posture", row["posture"]["fraction"]["standing"] < 0.9),
        ("sustained_crouch", row["posture"]["longest_run_seconds"]["crouch"] >= 0.5),
        ("sustained_quadruped", row["posture"]["longest_run_seconds"]["quadruped"] >= 0.5),
        ("sustained_prone", row["posture"]["longest_run_seconds"]["prone"] >= 0.5),
    ) if active)
    return row, support


def summarize(rows):
    """Aggregate screen rows (each with 'stem' and 'group') per stem and per group."""
    def aggregate(items):
        ok = [item for item in items if "error" not in item]
        result = {"clips": len(items), "screened": len(ok), "failed": len(items) - len(ok)}
        if not ok:
            return result
        flags = {}
        for item in ok:
            for flag in item["flags"]:
                flags[flag] = flags.get(flag, 0) + 1
        result["flag_fraction"] = {flag: round(count / len(ok), 4) for flag, count in sorted(flags.items())}
        result["level_span_m"] = _percentiles(np.array([item["support"]["level_span_m"] for item in ok]))
        result["level_count"] = _percentiles(np.array([item["support"]["level_count"] for item in ok]))
        result["posture_fraction_mean"] = {name: round(float(np.mean(
            [item["posture"]["fraction"][name] for item in ok])), 4) for name in POSTURES}
        errors = [d["level_span_error_m"] for item in ok for d in item["dimensions"]
                  if d["level_span_error_m"] is not None]
        clearance = [d["lowest_body_clearance_minus_size_m"] for item in ok for d in item["dimensions"]]
        if errors:
            result["dimension_level_span_error_m"] = _percentiles(np.array(errors))
        if clearance:
            result["dimension_clearance_minus_size_m"] = _percentiles(np.array(clearance))
        result["duration_seconds"] = round(sum(item["duration_seconds"] for item in ok), 3)
        return result

    by_stem, by_group = {}, {}
    for row in rows:
        by_stem.setdefault(row["stem"], []).append(row)
        by_group.setdefault(row["group"], []).append(row)
    return ({stem: dict(group=items[0]["group"], **aggregate(items)) for stem, items in sorted(by_stem.items())},
            {group: aggregate(items) for group, items in sorted(by_group.items())})


def select_clips(bvh_paths, config, *, groups=None, limit_per_stem=None):
    """Map BVH files to configured groups by action stem; deterministic per-stem caps.

    Returns (selected rows, selection report). Unmatched configured stems are reported
    so a misspelled stem cannot silently drop a group.
    """
    import hashlib
    from pathlib import Path

    from .bones_terrain_survey import action_stem, dimension_tokens, is_mirror_name

    stem_group = {}
    for name, group in config["groups"].items():
        if groups and name not in groups:
            continue
        for stem in group["stems"]:
            if stem in stem_group:
                raise ValueError(f"Stem {stem} is listed in both {stem_group[stem]} and {name}")
            stem_group[stem] = name
    by_stem = {}
    for path in bvh_paths:
        path = Path(path)
        if config.get("originals_only", True) and is_mirror_name(path.stem):
            continue
        stem = action_stem(path.stem)
        if stem in stem_group:
            by_stem.setdefault(stem, []).append(path)
    seed = config.get("sample_seed", "")
    selected, per_stem = [], {}
    for stem, paths in sorted(by_stem.items()):
        group = stem_group[stem]
        cap = limit_per_stem or config["groups"][group].get("max_clips_per_stem")
        ordered = sorted(paths, key=lambda p: hashlib.sha256(f"{seed}:{p.stem}".encode()).hexdigest())
        chosen = ordered[:cap] if cap else ordered
        per_stem[stem] = {"group": group, "available": len(paths), "selected": len(chosen)}
        selected += [{"path": str(p), "filename": p.stem, "stem": stem, "group": group,
                      "dimension_tokens": dimension_tokens(p.stem)} for p in chosen]
    report = {
        "selected": len(selected),
        "per_group": {name: sum(1 for row in selected if row["group"] == name)
                      for name in sorted(set(stem_group.values()))},
        "per_stem": per_stem,
        "configured_stems_without_files": sorted(set(stem_group) - set(by_stem)),
    }
    return selected, report
