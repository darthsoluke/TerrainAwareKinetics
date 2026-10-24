"""Ceilings, walls and variants: explain low postures, never enter the unchanged body."""

import json

import numpy as np
import pytest

from motionbricks.data.bones_terrain_scene import (PART_FOOT, PART_HAND, PART_OTHER, PART_TRUNK, SceneConfig,
                                                   explanation_bounds, fit_ceiling, fit_walls, scene_for_clip,
                                                   solid_penetration, vertex_part_codes)

FPS = 15.0
STANDING_TOP = 1.8


def body(center_xz, top, radius=0.2):
    """A column of skin points from the floor to `top` around center_xz."""
    heights = np.linspace(0.02, top, 12)
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    points = [[center_xz[0] + radius * np.cos(a), h, center_xz[1] + radius * np.sin(a)]
              for h in heights for a in angles]
    points.append([center_xz[0], top, center_xz[1]])
    return np.array(points)


def clip(segments):
    """segments: list of (frames, x0, x1, top, posture). Returns skin, posture, hips_xz."""
    skin, posture, hips = [], [], []
    for frames, x0, x1, top, label in segments:
        for x in np.linspace(x0, x1, frames):
            skin.append(body((x, 0.0), top))
            posture.append(label)
            hips.append([x, 0.0])
    return np.stack(skin), np.array(posture), np.array(hips)


def ground_fit():
    ground = {"box_id": 1, "name": "ground", "role": "ground", "center_xyz_m": [0, -0.1, 0],
              "half_extent_xyz_m": [20, 0.1, 20], "rotation_local_to_world": np.eye(3).tolist(),
              "top_m": 0.0, "bottom_m": -0.2}
    return {"ground_m": 0.0, "boxes": {"minimal": [ground], "maximal": [ground]}}


def test_crouch_ceiling_starts_where_the_body_lowers_and_explains_the_crouch():
    skin, posture, hips = clip([(15, 0, 0, 1.8, "standing"), (45, 0.6, 3.0, 1.2, "crouch")])
    ceiling, report = fit_ceiling(skin, posture, hips, [], 0.0, STANDING_TOP, FPS)
    assert report["status"] == "fitted" and report["dominant"] == "crouch"
    assert report["tight_m"] == pytest.approx(1.24, abs=1e-6)
    assert report["loose_m"] == pytest.approx(STANDING_TOP - 0.05)
    tight = ceiling["tight"]
    assert solid_penetration(skin, [tight], .02)["frames_over_report"] == 0
    assert solid_penetration(skin, [ceiling["loose"]], .02)["frames_over_report"] == 0
    # The standing column (x=0, top 1.8) stays outside the tight ceiling's footprint.
    from motionbricks.data.bones_terrain_scene import _inside_footprint
    assert not _inside_footprint(tight, [[0.0, 0.0]])[0]
    assert report["explanation"]["tight"] > .9 and report["explanation"]["loose"] > .9


def test_prone_is_explained_by_a_ceiling_below_the_quadruped_top():
    skin, posture, hips = clip([(45, 0, 2.0, 0.45, "prone")])
    ceiling, report = fit_ceiling(skin, posture, hips, [], 0.0, STANDING_TOP, FPS)
    bounds = explanation_bounds(STANDING_TOP, SceneConfig())
    assert report["status"] == "fitted"
    assert report["loose_m"] == pytest.approx(bounds["prone"])
    assert ceiling["loose"]["bottom_m"] < SceneConfig().quadruped_top_ratio * STANDING_TOP


def test_low_posture_in_place_of_a_standing_body_is_unexplainable():
    skin, posture, hips = clip([(30, 0, 0, 1.8, "standing"), (30, 0, 0, 1.2, "crouch")])
    _, report = fit_ceiling(skin, posture, hips, [], 0.0, STANDING_TOP, FPS)
    assert report["status"].startswith("unexplainable")
    # Kneeling down where the actor stood: the low body is wider than the standing one, so
    # some cells are only ever under the low body, but the hips stay over standing cells.
    skin = np.stack([body((0.0, 0.0), 1.8)] * 30 + [body((0.05, 0.0), 1.0, radius=0.35)] * 30)
    posture = np.array(["standing"] * 30 + ["crouch"] * 30)
    hips = np.array([[0.0, 0.0]] * 30 + [[0.05, 0.0]] * 30)
    _, report = fit_ceiling(skin, posture, hips, [], 0.0, STANDING_TOP, FPS)
    assert report["status"] == "unexplainable_in_place_transition"
    assert report["low_hips_on_blocked_cells"] == pytest.approx(1.0)


def leaning_body(contact_x=0.3, contact_part=PART_TRUNK, feet_radius=0.1, frames=30, contact_heights=(0.8, 1.4)):
    """Feet near the origin, a body column, and a flat patch of one body part at x = contact_x."""
    feet = np.array([[feet_radius * np.cos(a), h, feet_radius * np.sin(a)]
                     for h in (0.02, 0.06, 0.1) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
    column = body((0.0, 0.0), 1.7)
    column = column[column[:, 1] > 0.15]
    flat = np.array([[contact_x, y, z] for y in np.linspace(*contact_heights, 20) for z in np.linspace(-0.2, 0.2, 9)])
    points = np.concatenate([feet, column, flat])
    parts = np.concatenate([np.full(len(feet), PART_FOOT), np.full(len(column), PART_OTHER),
                            np.full(len(flat), contact_part)])
    return np.repeat(points[None], frames, axis=0), np.zeros((frames, 2)), parts


def test_wall_is_placed_against_a_leaning_trunk_and_clears_the_body():
    skin, hips, parts = leaning_body()
    walls, report = fit_walls(skin, hips, 0.0, FPS, vertex_parts=parts)
    assert report["status"] == "fitted" and len(walls) == 1 and report["kind"] == "lean"
    wall = walls[0]
    rotation = np.asarray(wall["rotation_local_to_world"])
    assert abs(rotation[0, 0]) == pytest.approx(1.0, abs=1e-6)       # normal along x
    assert wall["center_xyz_m"][0] - wall["half_extent_xyz_m"][0] == pytest.approx(0.305, abs=1e-3)
    assert solid_penetration(skin, walls, .02)["frames_over_report"] == 0


def test_outermost_skin_without_trunk_or_bracing_hand_is_not_a_wall():
    skin, hips, parts = leaning_body(contact_part=PART_OTHER)
    walls, report = fit_walls(skin, hips, 0.0, FPS, vertex_parts=parts)
    assert walls == [] and report["status"] == "no_contact_plane"
    assert report["rejected"]["no_trunk_or_bracing_hand_contact"] > 0


def test_hand_near_the_hips_or_feet_at_the_plane_do_not_make_a_wall():
    skin, hips, parts = leaning_body(contact_x=0.3, contact_part=PART_HAND)
    assert fit_walls(skin, hips, 0.0, FPS, vertex_parts=parts)[0] == []      # hand only 0.3 m from hips
    skin, hips, parts = leaning_body(contact_x=0.5, contact_part=PART_HAND)
    walls, report = fit_walls(skin, hips, 0.0, FPS, vertex_parts=parts)
    assert len(walls) == 1 and report["kind"] == "brace"
    skin, hips, parts = leaning_body(contact_x=0.5, contact_part=PART_HAND, contact_heights=(0.3, 0.6))
    assert fit_walls(skin, hips, 0.0, FPS, vertex_parts=parts)[0] == []      # hands resting low, e.g. on knees
    skin, hips, parts = leaning_body(contact_x=0.3, feet_radius=0.25)
    walls, report = fit_walls(skin, hips, 0.0, FPS, vertex_parts=parts)
    assert walls == [] and report["rejected"].get("feet_at_the_plane", 0) > 0


def test_contact_sliding_along_the_plane_is_not_a_wall():
    skin, hips, parts = leaning_body()
    flat = parts == PART_TRUNK
    slide = np.linspace(-0.3, 0.3, len(skin))
    skin = skin.copy()
    skin[:, flat, 2] += slide[:, None]          # the touching trunk skin moves along the plane
    walls, report = fit_walls(skin, hips, 0.0, FPS, vertex_parts=parts)
    assert walls == [] and report["rejected"].get("contact_slides_along_plane", 0) > 0
    skin, hips, parts = leaning_body()
    walls, report = fit_walls(skin, hips, 0.0, FPS, vertex_parts=parts)
    assert report["contact_drift_m"] == pytest.approx(0.0, abs=1e-9)


def test_part_codes_follow_rig_joint_names():
    names = ["Hips", "Spine1", "Chest", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
             "LeftHandIndex1", "RightFoot", "RightToeBase", "Head", "LeftLeg"]
    codes = vertex_part_codes(np.arange(len(names)), names).tolist()
    assert codes == [PART_TRUNK] * 5 + [PART_OTHER, PART_HAND, PART_HAND, PART_FOOT, PART_FOOT,
                                        PART_OTHER, PART_OTHER]


def test_variant_rejected_when_foot_evidence_has_no_surface():
    skin, posture, hips = clip([(45, 0, 3.0, 1.8, "standing")])
    fit = ground_fit()
    fit["evidence_patches"] = [{"height_m": 0.5, "level_m": 0.5,
                                "polygon_xz_m": [[1, -0.1], [1.2, -0.1], [1.2, 0.1], [1, 0.1]]}]
    scene = scene_for_clip(skin, posture, hips, fit, STANDING_TOP, FPS, variants=3, seed=1)
    assert not scene["training_eligible"] and scene["eligibility"] == "rejected"
    assert all("foot_evidence_not_supported" in v["rejection_reasons"] for v in scene["variants"])


def test_in_place_kneel_keeps_geometry_but_is_not_environment_explained():
    skin, posture, hips = clip([(30, 0, 0, 1.8, "standing"), (30, 0, 0, 1.0, "crouch")])
    scene = scene_for_clip(skin, posture, hips, ground_fit(), STANDING_TOP, FPS, variants=2, seed=0)
    assert scene["ceiling"] is None and scene["sustained_low_posture"]
    assert scene["eligibility"] == "geometry_only_low_posture_unexplained"
    assert all(v["geometry_accepted"] and not v["accepted"] for v in scene["variants"])


def test_upright_variants_add_high_ceilings_and_distant_clutter_only():
    skin, posture, hips = clip([(45, 0, 3.0, 1.8, "standing")])
    scene = scene_for_clip(skin, posture, hips, ground_fit(), STANDING_TOP, FPS, variants=12, seed=3)
    assert scene["upright"] and scene["ceiling"] is None
    assert all(v["accepted"] for v in scene["variants"])
    heights = [v["params"]["upright_ceiling_m"] for v in scene["variants"] if v["params"]["upright_ceiling_m"]]
    assert heights and min(heights) >= 1.8 + 0.15 - 1e-9
    config = SceneConfig()
    body_xz = skin[..., [0, 2]].reshape(-1, 2)
    for variant in scene["variants"]:
        for box in (b for b in variant["boxes"] if b["role"] == "clutter"):
            center = np.asarray(box["center_xyz_m"])
            rotation = np.asarray(box["rotation_local_to_world"])
            local = (np.column_stack([body_xz[:, 0], np.full(len(body_xz), center[1]), body_xz[:, 1]]) - center) @ rotation
            outside = np.maximum(np.abs(local[:, [0, 2]]) - np.asarray(box["half_extent_xyz_m"])[[0, 2]], 0)
            # Clearance is checked on 0.1 m footprint samples against 0.1 m envelope cells.
            assert np.linalg.norm(outside, axis=1).min() >= config.neutral_clutter_clearance_m - .1
    assert any(v["params"]["clutter"] > 0 for v in scene["variants"])
    json.dumps(scene, allow_nan=False)


def test_crouch_variants_keep_ceiling_inside_the_explained_range():
    skin, posture, hips = clip([(15, 0, 0, 1.8, "standing"), (45, 0.6, 3.0, 1.2, "crouch")])
    scene = scene_for_clip(skin, posture, hips, ground_fit(), STANDING_TOP, FPS, variants=8, seed=1)
    assert not scene["upright"] and scene["ceiling"] is not None
    low, high = scene["ceiling"]["range_above_floor_m"]
    for variant in scene["variants"]:
        assert low - 1e-9 <= variant["params"]["ceiling_m"] <= high + 1e-9
        assert variant["accepted"] and variant["explanation_rate"] > .9


def test_support_sides_respect_neutral_margin_unless_the_motion_limits_them():
    from motionbricks.data.bones_terrain_scene import _sample_support
    box = {"frame": {"center_xz": [0, 0], "axis_u": [1, 0], "rect": [-.5, .25, -.6, .6]},
           "frame_max_rect": [-.5, .25, -.6, .6], "evidence_rect": [-.2, .2, -.1, .1]}
    rng = np.random.default_rng(0)
    for _ in range(50):
        rect, _ = _sample_support(box, rng, SceneConfig())
        assert rect[0] == pytest.approx(-.5)            # neutral and motion limit coincide
        assert rect[1] == pytest.approx(.25)            # motion limit is tighter than neutral (.5)
        assert -.6 <= rect[2] <= -.4 and .4 <= rect[3] <= .6
