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
    screen.convert = lambda path, extra_names=(): (soles, joints, root, 120.0, 0.0)
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
                                "--viz-per-stem", "1"]) == 0
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
