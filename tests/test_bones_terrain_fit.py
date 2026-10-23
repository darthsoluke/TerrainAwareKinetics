"""Support fitting: boxes explain the support evidence without entering the body."""

import json

import numpy as np
import pytest

from motionbricks.data.bones_terrain_fit import FitConfig, fit_support, measure, oriented_rectangle
from motionbricks.data.stair_support import SupportConfig, reconstruct_support

FPS = 30.0
FOOT = np.array([[-.04, 0, -.08], [.04, 0, -.08], [.04, 0, .08], [-.04, 0, .08]])


def step_up(top=.5, frames=90):
    """Stand on the ground at z=0, step onto a box top at z=0.6, stand there."""
    soles, joints = {}, {}
    for side, x, delay in (("left", -.12, 0), ("right", .12, 6)):
        foot = np.repeat(FOOT[None], frames, axis=0).copy()
        for frame in range(frames):
            phase = np.clip((frame - 36 - delay) / 12, 0, 1)
            lift = .12 * np.sin(np.pi * phase)
            rise, forward = min(1.0, 2 * phase), max(0.0, (phase - .3) / .7)
            foot[frame] += [x, top * rise + lift, .6 * forward]
        soles[side] = foot
    progress = np.clip((np.arange(frames) - 36) / 18, 0, 1)
    hips = np.stack([np.zeros(frames), .95 + top * progress, .6 * progress], -1)
    up = np.array([0, 1, 0])
    joints = {"Hips": hips, "Chest": hips + .30 * up, "Neck1": hips + .50 * up, "Head": hips + .62 * up}
    for side, x in (("Left", -.2), ("Right", .2)):
        joints[f"{side}Hand"] = hips + [x, -.05, 0]
        joints[f"{side}HandMiddleEnd"] = hips + [x, -.13, 0]
        joints[f"{side}ForeArm"] = hips + [x, .2, 0]
        joints[f"{side}Shin"] = hips + [x * .5, -.48, .03]
    return soles, joints, hips.copy()


def support_of(soles, joints, root):
    probes = np.stack([joints[name] for name in joints], axis=1)
    return reconstruct_support(soles, root, FPS, SupportConfig(require_stairs=True), probes)[1]


def test_oriented_rectangle_recovers_rotated_box():
    angle = np.radians(30)
    axis = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-axis[1], axis[0]])
    points = np.array([a * axis * .4 + b * normal * .1 for a in (-1, 1) for b in (-1, 1)]) + [2, 3]
    center, axis_u, half_u, half_v = oriented_rectangle(points)
    np.testing.assert_allclose(center, [2, 3], atol=1e-9)
    assert abs(abs(axis_u @ axis) - 1) < 1e-9 or abs(abs(axis_u @ normal) - 1) < 1e-9
    low, high = sorted([half_u, half_v])
    assert low == pytest.approx(.1) and high == pytest.approx(.4)
    _, _, half_u, half_v = oriented_rectangle([[0, 0]])
    assert half_u == half_v == pytest.approx(.025)


def test_step_up_box_is_bounded_by_ground_stance_and_free_of_body():
    soles, joints, root = step_up()
    before = soles["left"].copy()
    support = support_of(soles, joints, root)
    fit = fit_support(soles, joints, root, FPS, support)
    np.testing.assert_allclose(fit["levels_m"], [0, .5], atol=1e-6)
    minimal, maximal = fit["boxes"]["minimal"], fit["boxes"]["maximal"]
    assert [b["role"] for b in minimal] == ["ground", "support"]
    box = maximal[1]
    assert box["top_m"] == pytest.approx(.5, abs=1e-6)
    # The near edge (toward z=0) stops before the ground-stance feet (z up to 0.08).
    rotation = np.asarray(box["rotation_local_to_world"])
    corners_z = [box["center_xyz_m"][2] + s * box["half_extent_xyz_m"][i] * rotation[2, i]
                 for i in (0, 2) for s in (-1, 1)]
    assert min(corners_z) >= .08
    assert min(corners_z) < .6 - .08  # but it did grow toward the stance feet
    for variant in ("minimal", "maximal"):
        metrics = fit["metrics"][variant]
        assert metrics["penetration_frames_over_report"] == 0
        assert metrics["support_evidence_unsupported"] == 0
        assert metrics["support_gap_abs_p95_m"] == pytest.approx(0, abs=1e-6)
    assert fit["flags"] == []
    np.testing.assert_array_equal(soles["left"], before)
    json.dumps(fit, allow_nan=False)


def test_measure_reports_body_inside_a_wrong_box():
    soles, joints, root = step_up()
    support = support_of(soles, joints, root)
    fit = fit_support(soles, joints, root, FPS, support)
    wrong = [dict(box) for box in fit["boxes"]["minimal"]]
    wrong[1] = dict(wrong[1], center_xyz_m=[0, wrong[1]["center_xyz_m"][1], 0],
                    half_extent_xyz_m=[1, wrong[1]["half_extent_xyz_m"][1], 1])
    patches = [p for p in support["patches"] if p["status"] in ("candidate", "weak_support")]
    levels = sorted(support["summary"]["usable_level_heights_m"])
    metrics = measure(soles, joints, wrong, patches, levels, FitConfig())
    assert metrics["penetration_frames_over_report"] > 0
    assert metrics["penetration_max_m"] > .3


def test_flat_clip_fits_ground_only():
    soles, joints, root = step_up(top=0.0)
    for side in soles:
        soles[side][..., 1] = np.maximum(soles[side][..., 1], 0)
    fit = fit_support(soles, joints, root, FPS, support_of(soles, joints, root))
    assert fit["levels_m"] == [0.0]
    assert [b["role"] for b in fit["boxes"]["maximal"]] == ["ground"]
    assert fit["metrics"]["maximal"]["penetration_frames_over_report"] == 0


def test_actor_scale_from_box_clips_predicts_other_sizes():
    from motionbricks.data.bones_terrain_fit import actor_scale_report
    rows = []
    for actor, scale in (("A1", 1.2), ("A2", 0.9)):
        for take in range(4):
            rows.append({"filename": f"come_up_50cm_box_R_00{take}__{actor}", "stem": "come_up_50cm_box",
                         "levels_m": [0.0, 0.5 * scale + 0.001 * take], "dimension_tokens": ["50cm"]})
        rows.append({"filename": f"jump_on_1m_001__{actor}", "stem": "jump_on_1m",
                     "levels_m": [0.0, 1.0 * scale], "dimension_tokens": ["1m"]})
    report = actor_scale_report(rows)
    assert report["actors"]["A1"]["scale"] == pytest.approx(1.2 + .003, abs=.005)
    assert report["within_actor_std_m_p50"] < .002
    check = report["sized_stem_checks"]["jump_on_1m"]
    assert check["clips"] == 2 and check["ratio_p50"] == pytest.approx(1.0, abs=.01)


def test_fit_script_in_process_with_stubbed_conversion(tmp_path):
    import scripts.fit_bones_terrain_support as fit_script
    from scripts import screen_bones_terrain_geometry as screen
    soles, joints, root = step_up()
    original = (screen._init_worker, screen.convert)
    screen._init_worker = lambda: screen._STATE.update(standing_hip=.95)
    screen.convert = lambda path, extra_names=(), return_motion=False: (soles, joints, root, 120.0, 0.0)
    try:
        run = tmp_path / "screen"
        run.mkdir()
        rows = [{"filename": f"come_up_50cm_box_R_00{i}__A1", "stem": "come_up_50cm_box",
                 "group": "known_dimension_support", "path": f"/x/{i}.bvh", "dimension_tokens": ["50cm"]}
                for i in range(3)]
        (run / "selection.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (run / "clips.jsonl").write_text("\n".join(json.dumps({**r, "flags": []}) for r in rows[:2]) + "\n" +
                                         json.dumps({**rows[2], "error": "ValueError: skipped"}))
        output = tmp_path / "fit"
        assert fit_script.main(["--screen-run", str(run), "--output", str(output), "--workers", "0",
                                "--viz-per-stem", "1", "--skin-stride", "0"]) == 0
    finally:
        screen._init_worker, screen.convert = original
    report = json.loads((output / "report.json").read_text())
    assert report["clips"] == 2 and report["fitted"] == 2 and report["rendered"]["ok"] == 1
    assert len(list((output / "fits").glob("*.json"))) == 2
    assert len(list((output / "viz" / "known_dimension_support").glob("*.png"))) == 1
    index = json.loads((output / "viewer" / "index.json").read_text())
    payload = json.loads((output / "viewer" / index["files"][0]).read_text())
    assert payload["meta"]["stem"] == "come_up_50cm_box" and payload["frames_mm"]
    lines = (output / "fits_compact.tsv").read_text().splitlines()
    assert len(lines) == 3 and all(len(l.split("\t")) == len(fit_script.COMPACT) for l in lines)
    group = json.loads((output / "by_group.json").read_text())["known_dimension_support"]
    assert group["fitted"] == 2 and group["minimal"]["clips_with_frames_over_2cm"] == 0


def test_hanging_palms_at_box_height_are_not_support_evidence():
    soles, joints, root = step_up(top=.69)
    hips = joints["Hips"]
    for side, x in (("Left", -.2), ("Right", .2)):
        joints[f"{side}Hand"] = hips + [x, -.19, 0]
        joints[f"{side}HandMiddleEnd"] = hips + [x, -.27, 0]
    fit = fit_support(soles, joints, root, FPS, support_of(soles, joints, root))
    palms = [e for e in fit["limb_events"] if e["kind"] == "palm" and abs(e["level_m"] - .69) < .01]
    assert palms and not any(e["accepted"] for e in palms)
    assert fit["levels"][0]["palm_events"] == 0
    assert fit["metrics"]["minimal"]["penetration_frames_over_report"] == 0


def test_small_steps_merge_and_far_patches_split():
    from motionbricks.data.bones_terrain_fit import _clusters, _filter_levels
    kept, merged = _filter_levels([0.0, 0.023, 0.17, 0.2, 0.35], 0.06)
    assert kept == [0.0, 0.17, 0.35] and merged == [0.023, 0.2]
    items = [{"kind": "foot", "points": np.array([[0, 0], [.1, .2]])},
             {"kind": "foot", "points": np.array([[.3, .2], [.4, .4]])},
             {"kind": "foot", "points": np.array([[2.0, 0], [2.1, .2]])}]
    groups = sorted(len(group) for group in _clusters(items, .3))
    assert groups == [1, 2]


def test_skin_points_inside_a_box_are_measured():
    from motionbricks.data.bones_terrain_fit import measure_points
    box = {"center_xyz_m": [0, .25, 0], "half_extent_xyz_m": [.5, .25, .5],
           "rotation_local_to_world": np.eye(3).tolist(), "top_m": .5}
    points = np.zeros((4, 2, 3))
    points[:, 0] = [0, .55, 0]     # above the top
    points[:, 1] = [0, .60, 0]
    points[2, 1] = [0, .40, 0]     # 10 cm inside
    points[3, 1] = [2, .10, 0]     # outside the footprint
    result = measure_points(points, [box], 0.0, .02)
    assert result["max_m"] == pytest.approx(.10) and result["frames_over_report"] == 1


def test_fit_reports_skin_authority_when_skin_is_given():
    soles, joints, root = step_up()
    skin = np.concatenate([soles["left"], soles["right"]], axis=1)
    fit = fit_support(soles, joints, root, FPS, support_of(soles, joints, root), skin_vertices=skin)
    assert fit["penetration_authority"] == "skin"
    assert fit["metrics"]["minimal"]["skin"]["frames_over_report"] == 0


def test_prone_ground_comes_from_skin_and_upside_down_feet_are_not_a_platform():
    soles, joints, root = step_up(top=0.0)
    for side in soles:
        soles[side][..., 1] = np.maximum(soles[side][..., 1], 0) + 0.10      # soles face up, 10 cm high
    support = support_of(soles, joints, root)
    assert min(support["summary"]["usable_level_heights_m"]) > 0.05
    # The skin of the foot tops (and the rest of the body) lies on the floor at y = 0.
    feet_tops = np.concatenate([soles["left"], soles["right"]], axis=1).copy()
    feet_tops[..., 1] = 0.01
    skin = np.concatenate([soles["left"], soles["right"], feet_tops], axis=1)
    config = FitConfig(skin_probe_stride=1)
    fit = fit_support(soles, joints, root, FPS, support, config=config, skin_vertices=skin)
    assert fit["ground_source"] == "skin_lowest_contacts" and fit["ground_m"] == pytest.approx(0.01, abs=1e-6)
    assert fit["levels_m"] == [0.0] and fit["inverted_foot_levels_m"]
    assert [b["role"] for b in fit["boxes"]["minimal"]] == ["ground"]
    assert {"ground_from_skin", "inverted_foot_level_dropped"} <= set(fit["flags"])
    assert fit["evidence_patches"] == []


def test_raised_soles_on_a_real_box_are_kept_with_skin():
    soles, joints, root = step_up()
    skin = np.concatenate([soles["left"], soles["right"]], axis=1)
    fit = fit_support(soles, joints, root, FPS, support_of(soles, joints, root),
                      config=FitConfig(skin_probe_stride=1), skin_vertices=skin)
    assert fit["ground_source"] == "lowest_foot_level" and not fit["inverted_foot_levels_m"]
    assert any(b["role"] == "support" for b in fit["boxes"]["minimal"])
    assert all(abs(p["level_m"] - fit["ground_m"]) < 1e-9 or abs(p["level_m"] - fit["ground_m"] - fit["levels_m"][-1]) < 1e-9
               for p in fit["evidence_patches"])


def test_skin_vertices_uses_official_skinner_interface():
    torch = pytest.importorskip("torch")
    import scripts.fit_bones_terrain_support as fit_script
    from scripts import screen_bones_terrain_geometry as screen

    class FakeSkinner:
        def skin(self, local, posed, rot_is_global=False):
            assert rot_is_global is False and local.shape[0] == posed.shape[0]
            return posed[:, :2] + 0.0

    saved = dict(screen._STATE)
    screen._STATE.update(torch=torch, skinner=FakeSkinner())
    try:
        rotations = np.repeat(np.eye(3)[None, None], 10, axis=0).repeat(3, axis=1)
        posed = np.arange(10 * 3 * 3, dtype=np.float32).reshape(10, 3, 3)
        vertices = fit_script.skin_vertices(rotations, posed, 2)
    finally:
        screen._STATE.clear()
        screen._STATE.update(saved)
    assert vertices.shape == (5, 2, 3)
    np.testing.assert_allclose(vertices[1], posed[2, :2])


def test_fit_script_builds_scenes_when_skin_is_available(tmp_path):
    import scripts.fit_bones_terrain_support as fit_script
    from scripts import screen_bones_terrain_geometry as screen
    soles, joints, root = step_up()
    skin = np.concatenate([soles["left"], soles["right"]] + [joints[n][:, None] for n in joints], axis=1)
    original = (screen._init_worker, screen.convert, fit_script.skin_vertices)
    screen._init_worker = lambda: screen._STATE.update(standing_hip=.95, standing_top=1.8)
    screen.convert = lambda path, extra_names=(), return_motion=False: (
        (soles, joints, root, 120.0, 0.0, (None, None)) if return_motion else (soles, joints, root, 120.0, 0.0))
    fit_script.skin_vertices = lambda rotations, posed, stride: skin[::stride]
    try:
        run = tmp_path / "screen"
        run.mkdir()
        row = {"filename": "come_up_50cm_box_R_001__A1", "stem": "come_up_50cm_box",
               "group": "known_dimension_support", "path": "/x/1.bvh", "dimension_tokens": ["50cm"]}
        (run / "selection.jsonl").write_text(json.dumps(row))
        (run / "clips.jsonl").write_text(json.dumps({**row, "flags": []}))
        output = tmp_path / "fit"
        assert fit_script.main(["--screen-run", str(run), "--output", str(output), "--workers", "0",
                                "--viz-per-stem", "1", "--variants", "3"]) == 0
    finally:
        screen._init_worker, screen.convert, fit_script.skin_vertices = original
        screen._STATE.pop("standing_top", None)
    fit = json.loads((output / "fits" / "come_up_50cm_box_R_001__A1.json").read_text())
    scene = fit["scene"]
    assert scene["upright"] and scene["ceiling_report"]["status"] == "no_sustained_low_posture"
    assert len(scene["variants"]) == 3 and all(v["accepted"] for v in scene["variants"])
    assert any(b["role"] == "support" for b in scene["variants"][0]["boxes"])
    group = json.loads((output / "by_group.json").read_text())["known_dimension_support"]
    assert group["scene"]["variants"] == 3 and group["scene"]["variants_accepted"] == 3
    assert scene["eligibility"] == "environment_explained" and scene["training_eligible"]
    assert group["scene"]["eligibility"]["environment_explained"]["clips"] == 1
    header = (output / "fits_compact.tsv").read_text().splitlines()[0].split("\t")
    assert "ceiling_status" in header and "variants_accepted" in header
    payload = json.loads(next((output / "viewer").glob("come_up*.json")).read_text())
    assert len(payload["variants"]) == 3 and payload["scene_report"]["ceiling"]["status"]


def test_skin_vertex_parts_checks_labels_on_the_bind_pose():
    import scripts.fit_bones_terrain_support as fit_script
    from motionbricks.data.bones_terrain_scene import PART_FOOT, PART_HAND, PART_TRUNK
    names = ["Hips", "Chest", "LeftHand", "LeftFoot", "Head"]
    heights = {0: 1.0, 1: 1.3, 2: 1.4, 3: 0.05, 4: 1.7}
    dominant = np.repeat(np.arange(5), 120)
    bind = np.zeros((len(dominant), 3))
    bind[:, 1] = [heights[j] for j in dominant]

    class Skinner:
        lbs_indices = np.stack([dominant, (dominant + 1) % 5], axis=1)
        lbs_weights = np.tile([0.8, 0.2], (len(dominant), 1))

    parts = fit_script.skin_vertex_parts(Skinner(), names, bind)
    assert (parts[dominant == 2] == PART_HAND).all() and (parts[dominant == 3] == PART_FOOT).all()
    assert (parts[dominant <= 1] == PART_TRUNK).all()
    with pytest.raises(RuntimeError):
        fit_script.skin_vertex_parts(Skinner(), ["Hips", "Chest", "LeftFoot", "LeftHand", "Head"], bind)
