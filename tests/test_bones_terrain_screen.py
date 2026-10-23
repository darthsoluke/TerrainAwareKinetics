"""Geometric screen: levels, limb support and posture from unchanged source motion."""

import json

import numpy as np
import pytest

from motionbricks.data.bones_terrain_screen import (LANDMARKS, ScreenConfig, parse_dimension,
                                                    screen_clip, summarize)

FPS = 30.0
STANDING_HIP = 0.95


def soles_fixture(stairs, fps=FPS):
    """Two feet, three stationary treads; stairs rise 0.2 m per level."""
    count = int(3 * fps)
    base = np.array([[-.04, 0, -.08], [.04, 0, -.08], [.04, 0, .08], [-.04, 0, .08]])
    values = {}
    for side, x in (("left", -.15), ("right", .15)):
        foot = np.repeat(base[None], count, axis=0)
        for frame in range(count):
            phase = frame / fps
            lift = 0.
            if phase < .8:
                level = 0.
            elif phase < 1.:
                level = (phase - .8) / .2
                lift = .15 * np.sin(np.pi * level)
            elif phase < 1.8:
                level = 1.
            elif phase < 2.:
                level = 1. + (phase - 1.8) / .2
                lift = .15 * np.sin(np.pi * (level - 1.))
            else:
                level = 2.
            foot[frame] += [x, (.2 * level if stairs else 0.) + lift, .5 * level]
        values[side] = foot
    return values


def body(frames, hip, pitch_deg, palm=None, knee=None, z=None):
    """Static landmarks above x=0; torso tilted forward by pitch_deg about the hips."""
    z = np.zeros(frames) if z is None else z
    pitch = np.radians(pitch_deg)
    up = np.array([0, np.cos(pitch), np.sin(pitch)])
    hips = np.stack([np.zeros(frames), np.full(frames, hip), z], -1)
    joints = {"Hips": hips, "Chest": hips + .30 * up, "Neck1": hips + .50 * up, "Head": hips + .62 * up}
    for side, x in (("Left", -.2), ("Right", .2)):
        if palm is None:
            hand = hips + .30 * up + [x, -.35, 0]
        else:
            hand = np.stack([np.full(frames, x), np.full(frames, palm + .04), z + .45], -1)
        joints[f"{side}Hand"] = hand
        joints[f"{side}HandMiddleEnd"] = hand - [0, .04 if palm is not None else -.08, 0]
        joints[f"{side}ForeArm"] = hand + [0, .25, -.05]
        knee_y = hip * .5 if knee is None else knee
        joints[f"{side}Shin"] = np.stack([np.full(frames, x * .5), np.full(frames, knee_y), z + .05], -1)
    return joints


def test_stairs_report_three_levels_and_known_dimension():
    soles = soles_fixture(True)
    frames = len(soles["left"])
    rise = np.linspace(0, .4, frames)
    joints = body(frames, STANDING_HIP, 5)
    for name in joints:
        joints[name] = joints[name] + np.stack([np.zeros(frames), rise, np.linspace(0, 1, frames)], -1)
    root = joints["Hips"].copy()
    before = {k: v.copy() for k, v in soles.items()}
    row, support = screen_clip(soles, joints, root, FPS, STANDING_HIP, dimension_tokens=["0.4m"])
    np.testing.assert_allclose(row["support"]["levels_m"], [0, .2, .4], atol=1e-6)
    assert row["support"]["multi_level"] and "multi_level" in row["flags"]
    assert row["dimensions"][0]["level_span_error_m"] == pytest.approx(0, abs=1e-6)
    assert row["posture"]["fraction"]["standing"] == 1.0
    assert "palm_support" not in row["flags"]
    np.testing.assert_array_equal(soles["left"], before["left"])
    json.dumps(row, allow_nan=False)


def test_quadruped_uses_palms_and_knees_at_ground():
    soles = soles_fixture(False)
    frames = len(soles["left"])
    joints = body(frames, .55, 80, palm=0.0, knee=.06)
    row, _ = screen_clip(soles, joints, joints["Hips"], FPS, STANDING_HIP)
    assert row["support"]["levels_m"] == [0.0]
    assert row["posture"]["fraction"]["quadruped"] == 1.0
    assert {"palm_support", "knee_support", "sustained_quadruped", "low_posture"} <= set(row["flags"])
    assert "palm_support_above_ground" not in row["flags"]


def test_crouch_and_prone_are_separated_by_chest_height_and_pitch():
    soles = soles_fixture(False)
    frames = len(soles["left"])
    crouch, _ = screen_clip(soles, body(frames, .60, 15), np.zeros((frames, 3)), FPS, STANDING_HIP)
    assert crouch["posture"]["fraction"]["crouch"] == 1.0
    assert crouch["head_clearance_m"]["crouch"]["p50"] == pytest.approx(.60 + .62 * np.cos(np.radians(15)), abs=1e-3)
    prone, _ = screen_clip(soles, body(frames, .15, 88), np.zeros((frames, 3)), FPS, STANDING_HIP)
    assert prone["posture"]["fraction"]["prone"] == 1.0


def test_missing_landmark_and_bad_dimension_are_explicit():
    soles = soles_fixture(False)
    frames = len(soles["left"])
    joints = body(frames, STANDING_HIP, 0)
    joints.pop("Head")
    with pytest.raises(ValueError, match="Missing landmarks"):
        screen_clip(soles, joints, np.zeros((frames, 3)), FPS, STANDING_HIP)
    assert parse_dimension("50cm") == pytest.approx(.5)
    assert parse_dimension("0.75m") == pytest.approx(.75)
    with pytest.raises(ValueError):
        parse_dimension("box")
    with pytest.raises(ValueError):
        ScreenConfig(crouch_hip_ratio=.9)
    assert set(LANDMARKS) >= {"Hips", "Head", "LeftShin", "RightHandMiddleEnd"}


def test_summary_groups_by_stem_and_keeps_failures():
    soles = soles_fixture(True)
    frames = len(soles["left"])
    joints = body(frames, STANDING_HIP, 5)
    row, _ = screen_clip(soles, joints, joints["Hips"], FPS, STANDING_HIP, dimension_tokens=["40cm"])
    rows = [dict(row, stem="stairs_climbing_up", group="stairs"),
            {"stem": "stairs_climbing_up", "group": "stairs", "error": "ValueError: bad"}]
    by_stem, by_group = summarize(rows)
    assert by_stem["stairs_climbing_up"]["clips"] == 2 and by_stem["stairs_climbing_up"]["failed"] == 1
    assert by_group["stairs"]["flag_fraction"]["multi_level"] == 1.0
    assert "dimension_level_span_error_m" in by_group["stairs"]


def test_selection_maps_stems_skips_mirrors_and_caps_deterministically():
    from motionbricks.data.bones_terrain_screen import select_clips
    config = {"originals_only": True, "sample_seed": "s",
              "groups": {"known": {"stems": ["come_up_50cm_box"]},
                         "flat": {"stems": ["walk", "typo_stem"], "max_clips_per_stem": 2}}}
    paths = [f"/d/20250101/come_up_50cm_box_R_00{i}__A1.bvh" for i in range(3)]
    paths += ["/d/20250101/come_up_50cm_box_R_001__A1_M.bvh"]
    paths += [f"/d/20250102/walk_ff_loop_00{i}__A{i}.bvh" for i in range(5)]
    selected, report = select_clips(paths, config)
    assert report["per_group"] == {"flat": 2, "known": 3}
    assert report["configured_stems_without_files"] == ["typo_stem"]
    assert all(not row["filename"].endswith("_M") for row in selected)
    assert {row["dimension_tokens"][0] for row in selected if row["group"] == "known"} == {"50cm"}
    again, _ = select_clips(list(reversed(paths)), config)
    assert [r["filename"] for r in again] == [r["filename"] for r in selected]
    only, _ = select_clips(paths, config, groups={"flat"}, limit_per_stem=1)
    assert len(only) == 1
    with pytest.raises(ValueError, match="both"):
        select_clips(paths, {"groups": {"a": {"stems": ["walk"]}, "b": {"stems": ["walk"]}}})


def test_shipped_config_stems_are_unique():
    from pathlib import Path
    from motionbricks.data.bones_terrain_screen import select_clips
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/bones_terrain_screen_v1.json").read_text())
    selected, report = select_clips([], config)
    assert selected == [] and len(report["configured_stems_without_files"]) > 50


def test_dry_run_writes_selection_without_kimodo(tmp_path):
    from scripts.screen_bones_terrain_geometry import main
    folder = tmp_path / "bvh" / "20250101"
    folder.mkdir(parents=True)
    for name in ("crawl_ff_loop_001__A1", "crawl_ff_loop_001__A1_M", "come_up_50cm_box_R_001__A2", "dance_x_001__A3"):
        (folder / f"{name}.bvh").write_text("HIERARCHY\n")
    output = tmp_path / "out"
    assert main(["--bvh-root", str(tmp_path / "bvh"), "--output", str(output), "--dry-run"]) == 0
    rows = [json.loads(line) for line in (output / "selection.jsonl").read_text().splitlines()]
    assert {(row["stem"], row["group"]) for row in rows} == {
        ("crawl", "prone"), ("come_up_50cm_box", "known_dimension_support")}
    report = json.loads((output / "report.json").read_text())
    assert report["bvh_files_found"] == 4 and report["protocol"]["source_changed"] is False
    with pytest.raises(FileExistsError):
        main(["--bvh-root", str(tmp_path / "bvh"), "--output", str(output), "--dry-run"])


def test_compact_table_has_one_line_per_clip():
    from scripts.screen_bones_terrain_geometry import COMPACT_COLUMNS, write_compact
    import tempfile
    from pathlib import Path
    soles = soles_fixture(False)
    frames = len(soles["left"])
    joints = body(frames, .55, 80, palm=0.0, knee=.06)
    row, _ = screen_clip(soles, joints, joints["Hips"], FPS, STANDING_HIP)
    rows = [dict(row, filename="mohak_001__A1", stem="mohak", group="quadruped"),
            {"filename": "bad__A2", "stem": "crawl", "group": "prone", "error": "ValueError: x\ty"}]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "c.tsv"
        write_compact(rows, path)
        lines = path.read_text().splitlines()
    assert lines[0].split("\t") == list(COMPACT_COLUMNS)
    assert all(len(line.split("\t")) == len(COMPACT_COLUMNS) for line in lines)
    assert "failed" in lines[1] and "sustained_quadruped" in lines[2]
