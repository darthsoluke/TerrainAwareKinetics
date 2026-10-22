import copy
import json

import pytest

from scripts.summarize_soma_joint_rollout import MATCHED_KEYS, summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(folder, empty=False):
    contract = {k: "same" for k in MATCHED_KEYS}
    contract.update(initial_bundle_sha256="Original", initial_root_state="root",
        coupled_root_trial=True, history_policy="scheduled_generated", decoded_motion_weight=1.,
        commit_frames=8, motion_fps=60, frozen_state={"tokenizer": "codec"})
    for arm in ("A", "B"):
        write(folder / f"full_v2_{arm}" / "contract.json", {**contract, "arm": arm, "train_root": arm == "B"})
        write(folder / f"full_v2_{arm}" / "result.json", {"steps": 10000, "frozen_unchanged": True,
            "pose_changed": True, "root_changed": arm == "B", "sampling_sha256": "same",
            "history_counts": {"generated": 9}, "bundle_sha256": arm})
    for arm in ("Original", "A", "B"):
        contact = {"region_penetration_m": {"mean": .001, "peak": .01}, "valid_samples": {"frames": 10},
            "contact_height_abs_m": None if empty else {"median": .003},
            "measured_slip_mps": None if empty else {"mean": .3 if arm != "B" else .2}}
        score = {"source_contact": {"identity": "source"}, "predicted_contact": contact,
            "root_relative_mpjpe_mm": {"mean": 20.}, "root_position_error_m": {"mean": .02},
            "angular_step_deg": {"max": 30.}, "boundary_joint_step_mm": {"p95": 30.}}
        aggregate = {"native_root": {"mean_window_mpjpe_mm": 20., "mean_window_root_position_error_m": .02,
            "worst_joint_step_deg": 30., "mean_window_boundary_joint_step_mm": 30.,
            "mean_window_penetration_m": .001, "mean_window_measured_slip_mps": contact["measured_slip_mps"]["mean"] if not empty else None},
            "teacher_source_root": {"mean_window_mpjpe_mm": 20.}}
        long = {"initial_filename": "idle", "initial_frame": 0, "boundary_joint_step_mm": {"p95": 30.},
            "boundary_angular_step_deg": {"p95": 15. if arm != "B" else 14.},
            "angular_step_deg": {"max": 30. if arm != "B" else 28.}, "joint_acceleration_mps2": {"p95": 100.},
            "root_command_velocity_error_mps": {"mean": .3},
            "per_command": {"idle": {"velocity_error_mps": {"mean": .3}},
                            "stop": {"velocity_error_mps": {"mean": .3}}}}
        report = {"bundle_sha256": arm, "evaluation_sha256": "same", "dataset_sha256": "same",
            "commit_frames": 8, "tokenizer_state": "codec", "test_used": False, "post_processing": None,
            "frozen_state": {"root": "root" if arm != "B" else "trained", "pose": arm,
                             "encoder": "same", "quantizer": "same", "decoder": "same"},
            "aggregate": aggregate, "windows": [{"filename": str(i), "start": 0, "input_sha256": "same",
                "behavior": "walk", "native_root": copy.deepcopy(score), "teacher_source_root": copy.deepcopy(score)} for i in range(57)],
            "rollouts": {str(i): copy.deepcopy(long) for i in range(3)}}
        report["rollouts"]["delay_12"] = {"angular_step_deg": {"max": 30.}}
        write(folder / f"review_{arm}" / "metrics.json", report)


def test_matched_gain_allows_only_ue_review_not_promotion(tmp_path):
    fixture(tmp_path)
    report = summarize(tmp_path)
    assert report["candidates"]["B"]["eligible_for_ue_review"]
    assert not report["quality_approved"] and not report["default_bundle_changed"]
    assert report["paired_windows"]["B_vs_A"]["summary"]["slip_mps"]["improved"] == 57


def test_better_average_cannot_hide_stop_regression(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "review_B/metrics.json"
    report = json.loads(path.read_text())
    for index in ("0", "1", "2"):
        report["rollouts"][index]["root_command_velocity_error_mps"]["mean"] = .2
        report["rollouts"][index]["per_command"]["stop"]["velocity_error_mps"]["mean"] = .5
    write(path, report)
    result = summarize(tmp_path)
    assert not result["candidates"]["B"]["eligible_for_ue_review"]


@pytest.mark.parametrize("defect", ["sampling", "evidence", "root", "decoder"])
def test_unmatched_trials_are_rejected(tmp_path, defect):
    fixture(tmp_path)
    path = tmp_path / ("full_v2_B/result.json" if defect == "sampling" else "review_B/metrics.json")
    report = json.loads(path.read_text())
    if defect == "sampling":
        report["sampling_sha256"] = "different"
    elif defect == "evidence":
        report["windows"][0]["native_root"]["predicted_contact"]["valid_samples"]["frames"] = 9
    else:
        report["frozen_state"][defect] = "root" if defect == "root" else "changed"
    write(path, report)
    with pytest.raises(ValueError):
        summarize(tmp_path)


def test_empty_contact_evidence_cannot_qualify(tmp_path):
    fixture(tmp_path, empty=True)
    result = summarize(tmp_path)
    assert result["arms"]["B"]["mean_window_contact_height_m"] is None
    assert not result["candidates"]["B"]["eligible_for_ue_review"]
