"""Integration tests for the offline contact review entry point.

These build synthetic review windows from the real SOMA30 skeleton and the official
sole proxies, so the report structure, the status separation and the input manifest
are exercised end to end without touching the recorded review windows.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from motionbricks.data.bones_seed import build_representation, official_sole_offsets, sole_positions

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_contact_mechanics.py"
FRAMES = 32
FPS = 60.0


@pytest.fixture(scope="module")
def review():
    spec = importlib.util.spec_from_file_location("review_contact_mechanics_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def skeleton():
    return build_representation().skeleton


def write_window(folder, name, skeleton, offsets, *, source_x_step=0.0, pred_x_step=0.0, contacts=None,
                 confidence=None, break_fk=False, clearance=0.005, frames=FRAMES):
    """One SOMA30 window with soles just above the floor, plus optional defects.

    source_x_step and pred_x_step let the source and the reconstruction drift apart
    while both keep identity local rotations.
    """
    identity = torch.eye(3, dtype=torch.float32).expand(frames, skeleton.nbjoints, 3, 3).contiguous()
    steps = torch.arange(frames, dtype=torch.float32)

    def pose(step):
        root = torch.zeros(frames, 3, dtype=torch.float32)
        root[:, 0] = step * steps
        rotations, posed, _ = skeleton.fk(identity, root)
        return rotations, posed, root

    rotations, posed, root = pose(source_x_step)
    shift = clearance - float(sole_positions(posed, rotations, skeleton, offsets)[..., 1].min())
    root[:, 1] += shift
    rotations, posed, _ = skeleton.fk(identity, root)
    pred_rotations, pred_posed, _ = pose(pred_x_step)
    pred_posed = pred_posed + torch.tensor([0.0, shift, 0.0], dtype=pred_posed.dtype)
    labels = torch.ones(frames, 4, dtype=torch.bool) if contacts is None else torch.as_tensor(contacts)
    confident = torch.ones(frames, 4, dtype=torch.bool) if confidence is None else torch.as_tensor(confidence)
    saved_rotations = identity.clone()
    if break_fk:
        saved_rotations[:, skeleton.bone_index["LeftShin"]] = torch.eye(3, dtype=torch.float32) * 0.5
    np.savez_compressed(folder / f"{name}_review_window.npz",
                        source_positions_m=posed.numpy(), pred_positions_m=pred_posed.numpy(),
                        source_local_rot_mats=saved_rotations.numpy(),
                        pred_local_rot_mats=saved_rotations.numpy(),
                        source_contacts=labels.numpy(), pred_contacts=labels.numpy(),
                        source_confidence=confident.numpy(), fps=np.array(FPS))


@pytest.fixture(scope="module")
def offsets(skeleton):
    return official_sole_offsets(skeleton)[0]


def report_for(review, folder, expected, **kwargs):
    return review.build_report(folder, folder / "out", expected, 4, 0.0, [0.6], **kwargs)


def test_a_processed_window_records_three_separate_statuses(review, skeleton, offsets, tmp_path):
    write_window(tmp_path, "walk", skeleton, offsets)
    result = report_for(review, tmp_path, ["walk"])
    record = result["behaviors"]["walk"]
    assert record["execution_status"] == "processed"
    assert record["motion_quality_status"] in ("pass", "fail", "not_assessed")
    assert record["physics_check_status"] == "assessed_resultant_only"
    assert result["input_status"] == "complete"
    # A static window is its own reconstruction, so the screening criteria pass.
    assert record["motion_quality_status"] == "pass"
    assert record["resultant_force_check"]["frame_state"]["source"]["stable_support"] == FRAMES - 4
    assert record["resultant_force_check"]["per_friction_coefficient"]["0.6"]["source"]["violations"] == {
        "ground_pull_frames": 0, "friction_capacity_frames": 0, "airborne_force_frames": 0}


def test_a_missing_expected_window_is_reported_not_ignored(review, skeleton, offsets, tmp_path):
    write_window(tmp_path, "walk", skeleton, offsets)
    result = report_for(review, tmp_path, ["walk", "run"])
    assert result["input_status"] == "incomplete"
    assert result["missing_behaviors"] == ["run"]
    assert result["valid_behavior_count"] == 1
    assert any("run" in note for note in result["notes"])


def test_an_unreadable_window_is_a_failed_execution_not_a_silent_skip(review, skeleton, offsets, tmp_path):
    write_window(tmp_path, "walk", skeleton, offsets)
    (tmp_path / "run_review_window.npz").write_bytes(b"not an npz archive")
    result = report_for(review, tmp_path, ["walk", "run"])
    assert set(result["behaviors"]) == {"walk"}
    assert result["failed_behaviors"]["run"]["execution_status"] == "failed"
    assert "could not be read" in result["failed_behaviors"]["run"]["execution_detail"]
    assert result["input_status"] == "incomplete"
    assert result["missing_behaviors"] == []


def test_an_fk_rebuild_mismatch_is_rejected(review, skeleton, offsets, tmp_path):
    write_window(tmp_path, "walk", skeleton, offsets, break_fk=True)
    result = report_for(review, tmp_path, ["walk"])
    record = result["failed_behaviors"]["walk"]
    assert record["execution_status"] == "failed"
    assert "did not reproduce" in record["execution_detail"]
    assert record["motion_quality"] is None
    assert record["resultant_force_check"] is None


def test_a_window_without_valid_support_reports_nulls_and_zero_counts(review, skeleton, offsets, tmp_path):
    write_window(tmp_path, "idle", skeleton, offsets, contacts=np.zeros((FRAMES, 4), dtype=bool))
    result = report_for(review, tmp_path, ["idle"])
    record = result["behaviors"]["idle"]
    coverage = record["coverage"]
    # No support label, but every sole is on the floor, so this is contact and never flight.
    assert coverage["whole_body_frames"] == {"stable_support": 0, "near_ground_undecided": FRAMES - 4,
                                             "airborne": 0, "unknown": 0}
    assert coverage["channel_intervals"]["sustained_support"] == 0
    summary = record["groups"]["all_channels"]["source"]["support_tangential_speed_mps"]
    assert summary == {"samples": 0, "mean": None, "minimum": None, "p95": None, "peak": None}
    violations = record["resultant_force_check"]["per_friction_coefficient"]["0.6"]["source"]["violations"]
    assert violations["airborne_force_frames"] == 0
    assert violations["ground_pull_frames"] == 0


def test_a_sole_in_the_clearance_band_is_unknown_not_airborne(review, skeleton, offsets, tmp_path):
    write_window(tmp_path, "turn", skeleton, offsets, confidence=np.zeros((FRAMES, 4), dtype=bool),
                 clearance=0.05)
    result = report_for(review, tmp_path, ["turn"])
    coverage = result["behaviors"]["turn"]["coverage"]
    assert coverage["whole_body_frames"]["airborne"] == 0
    assert coverage["whole_body_frames"]["unknown"] == FRAMES - 4
    assert coverage["whole_body_frame_known_fraction"] == 0.0


def test_coverage_denominators_keep_channel_and_whole_body_rows_apart(review, skeleton, offsets, tmp_path):
    write_window(tmp_path, "walk", skeleton, offsets)
    coverage = report_for(review, tmp_path, ["walk"])["behaviors"]["walk"]["coverage"]
    assert coverage["channel_frames"]["scored"] == 4 * (FRAMES - 4)
    assert coverage["channel_intervals"]["scored"] == 4 * (FRAMES - 5)
    assert 0.0 <= coverage["channel_interval_known_fraction"] <= 1.0
    assert 0.0 <= coverage["whole_body_interval_known_fraction"] <= 1.0
    assert coverage["whole_body_intervals"]["scored"] == FRAMES - 5


def test_a_drifting_reconstruction_fails_the_quality_screen(review, skeleton, offsets, tmp_path):
    write_window(tmp_path, "backward", skeleton, offsets, pred_x_step=0.01)
    result = report_for(review, tmp_path, ["backward"])
    record = result["behaviors"]["backward"]
    assert record["execution_status"] == "processed"
    assert record["motion_quality_status"] == "fail"
    assert "sustained_support_segment_net_drift_peak_m" in record["motion_quality"]["failed_criteria"]


def test_the_report_is_json_serialisable_and_keeps_separate_units(review, skeleton, offsets, tmp_path):
    write_window(tmp_path, "walk", skeleton, offsets)
    result = report_for(review, tmp_path, ["walk"])
    text = json.dumps(result, allow_nan=False)
    assert "ground_pull_mps2" in text and "penetration_m" in text
    assert result["behaviors"]["walk"]["resultant_force_check"]["not_a_measurement"]
    assert result["protocol"]["scoring"]["quantiles"].startswith("P95 is computed over pooled samples")
