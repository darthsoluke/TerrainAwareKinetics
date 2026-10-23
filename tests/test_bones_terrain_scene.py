"""Ceilings, walls and variants: explain low postures, never enter the unchanged body."""

import json

import numpy as np
import pytest

from motionbricks.data.bones_terrain_scene import (SceneConfig, explanation_bounds, fit_ceiling, fit_walls,
                                                   scene_for_clip, solid_penetration)

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


def test_wall_is_placed_against_a_flat_contact_and_clears_the_body():
    frames = 30
    skin = []
    for _ in range(frames):
        points = body((0.0, 0.0), 1.7)
        flat = np.array([[0.3, y, z] for y in np.linspace(0.5, 1.6, 20) for z in np.linspace(-0.2, 0.2, 9)])
        skin.append(np.concatenate([points, flat]))
    skin = np.stack(skin)
    hips = np.zeros((frames, 2))
    walls, report = fit_walls(skin, hips, 0.0, FPS)
    assert report["status"] == "fitted" and len(walls) == 1
    wall = walls[0]
    rotation = np.asarray(wall["rotation_local_to_world"])
    assert abs(rotation[0, 0]) == pytest.approx(1.0, abs=1e-6)       # normal along x
    assert wall["center_xyz_m"][0] - wall["half_extent_xyz_m"][0] == pytest.approx(0.305, abs=1e-3)
    assert solid_penetration(skin, walls, .02)["frames_over_report"] == 0


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
