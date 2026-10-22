import json

import numpy as np
import pytest

from scripts.review_native_mb_stairs import (
    root_scores, select_windows, skill_gate, sole_scores, stair_constraints, stats, transform_support,
)


def patch(height=.2):
    return {"id": 0, "level_id": 0, "status": "weak_support", "height_m": height,
            "polygon_xz_m": [[-1, -1], [1, -1], [1, 1], [-1, 1]], "confidence": .8}


def arrays():
    source = {}
    points = {}
    for side in ("left", "right"):
        value = np.zeros((64, 98, 3))
        value[..., 1] = .2
        points[side] = value
        source[f"patch_id_{side}"] = np.zeros(64, dtype=int)
        source[f"vertex_measured_{side}"] = np.ones((63, 98), dtype=bool)
    return source, points


def test_stats_empty_is_not_zero():
    assert stats([np.nan])["mean"] is None
    assert stats([])["count"] == 0


def test_contact_region_scores_exclude_conditioned_frames():
    source, points = arrays()
    for value in points.values():
        value[:4, :, 1] = -99
        value[63:, :, 1] = -99
    result = sole_scores(points, source, [patch()], 0)
    assert result["left"]["expected_support_abs_gap_m"]["max"] == 0
    assert result["left"]["known_patch_penetration_m"]["max"] == 0
    assert result["left"]["source_measured_intervals"] == 58


def test_source_vertex_mask_only_and_tangential_speed():
    source, points = arrays()
    source["vertex_measured_left"][:] = False
    source["vertex_measured_left"][:, 0] = True
    points["left"][:, 1:, 0] = np.arange(64)[:, None] * .6 / 30
    points["left"][:, :, 1] += np.arange(64)[:, None] * .1 / 30
    result = sole_scores(points, source, [patch()], 0)
    assert result["left"]["source_mask_material_slip_mps"]["max"] == 0
    points["left"][:, 0, 0] = np.arange(64) * .6 / 30
    assert sole_scores(points, source, [patch()], 0)["left"]["source_mask_material_slip_mps"]["mean"] == pytest.approx(.6)


def test_unknown_space_is_not_zero_penetration_evidence():
    source, points = arrays()
    points["left"][..., 0] = 10
    result = sole_scores(points, source, [patch()], 0)["left"]
    assert result["known_patch_penetration_m"]["count"] == 0
    assert result["known_patch_penetration_m"]["mean"] is None
    assert result["spatial_query_unknown_vertex_fraction"] == 1


def test_wrong_length_evidence_rejected():
    source, points = arrays()
    source["vertex_measured_left"] = source["vertex_measured_left"][:60]
    with pytest.raises(ValueError, match="not aligned"):
        sole_scores(points, source, [patch()], 0)


def test_rigid_transform_preserves_patch_height_and_does_not_mutate():
    p = patch()
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    actual = transform_support([p], rotation, np.array([1, 0, 2]))[0]
    assert actual["height_m"] == pytest.approx(.2)
    assert p["polygon_xz_m"][0] == [-1, -1]
    assert actual["polygon_xz_m"][0] == [-3, 2]


def test_endpoint_height_is_not_stair_progress():
    source = np.zeros((64, 30, 3))
    source[:, :, 1] = np.linspace(0, .6, 64)[:, None]
    pred = np.zeros_like(source)
    pred[63] = source[63]
    score = root_scores(pred, source, 0)
    assert score["endpoint_error_m"] == 0
    assert not score["height_direction_matches"]
    assert score["height_change_error_m"] > .5


def test_skill_gate_rejects_empty_support_evidence():
    source, points = arrays()
    scores = sole_scores(points, source, [patch()], 0)
    root = {"height_direction_matches": True, "height_change_error_m": 0., "horizontal_travel_error_m": 0.}
    assert skill_gate(scores, scores, root)["passed"]
    scores["left"]["expected_support_abs_gap_m"]["p95"] = None
    assert not skill_gate(scores, scores, root)["passed"]


def test_window_selection_uses_only_validation_and_both_directions(tmp_path):
    rows = []
    for name, direction, split in (("up", 1, "validation"), ("down", -1, "validation"), ("leak", 1, "train")):
        features = np.zeros((160, 10), np.float32)
        features[:, 1] = direction * np.arange(160) / 100
        ids = np.arange(160) // 20
        np.savez(tmp_path / (name + ".npz"), global_features=features,
                 patch_id_left=ids, patch_id_right=ids)
        report = {"summary": {"pair_status": "weak_pair_candidate"},
                  "patches": [{"id": i, "level_id": i} for i in range(8)]}
        (tmp_path / (name + ".json")).write_text(json.dumps(report))
        rows.append({"shard": name + ".npz", "split": split, "support_report": name + ".json"})
    (tmp_path / "accepted.json").write_text(json.dumps(rows))
    selected = select_windows(tmp_path)
    assert {row["direction"] for row in selected} == {"up", "down"}
    assert {row["split"] for row in selected} == {"validation"}
    for left in selected:
        for right in selected:
            if left is not right and left["shard"] == right["shard"]:
                assert abs(left["start"] - right["start"]) >= 64


def test_four_context_frames_and_no_future_motion_leak():
    import torch
    from test_native_mb_inbetween import FakeInference
    from test_native_mb_stage_training import _motion_and_representation

    rep, batch = _motion_and_representation(frame_count=75)
    raw = rep.unnormalize(batch["motion"])[:, :65]
    infer = FakeInference(rep)
    original, _ = stair_constraints(infer, raw)
    assert original["has_global_root_values"].tolist() == [[True, True, True, True, False, False, False, True]]
    assert original["has_local_poses"].tolist() == [[True, True, True, True, False, False, False, True]]
    assert original["has_local_root_values"].tolist() == [[True, True, True, False, False, False, False, True]]
    assert original["local_root_values"][:, 3].count_nonzero() == 0
    changed = raw.clone()
    changed[:, 4:63] += .17
    inputs, _ = stair_constraints(infer, changed)
    for key, value in original.items():
        if value is None:
            assert inputs[key] is None
        else:
            torch.testing.assert_close(inputs[key], value, rtol=0, atol=0)


def test_renderer_produces_moving_frames(tmp_path):
    from PIL import Image, ImageChops
    from scripts.review_native_mb_stairs import render_example

    points = np.zeros((64, 30, 3))
    points[:, :, 0] = np.arange(30) / 30
    points[:, :, 1] = np.arange(30) / 20 + np.arange(64)[:, None] / 200
    points[:, :, 2] = np.arange(64)[:, None] / 60
    render_example(tmp_path / "up", {"Source": points, "Generated": points}, [patch()],
                   np.array([-1] + list(range(29))), "up")
    with Image.open(tmp_path / "up.gif") as gif:
        assert gif.n_frames > 1
        first = gif.convert("RGB")
        gif.seek(gif.n_frames - 1)
        assert ImageChops.difference(first, gif.convert("RGB")).getbbox() is not None
