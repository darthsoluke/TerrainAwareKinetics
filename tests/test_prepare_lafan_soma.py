import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation


MOTIONBRICKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOTIONBRICKS))
from motionbricks.data.lafan_soma_retarget import JOINT_MAP  # noqa: E402
from scripts import prepare_lafan_soma as preparation  # noqa: E402


def assessment_inputs(frames=50):
    joint_count = len(JOINT_MAP)
    source_positions = np.zeros((frames, joint_count, 3), dtype=np.float64)
    target_positions = np.zeros_like(source_positions)
    desired_positions = np.zeros_like(source_positions)
    target_rotations = np.tile(np.eye(3), (frames, joint_count, 1, 1))
    desired_rotations = target_rotations.copy()
    sole_heights = np.tile([.20, .21, .22, .23], (frames, 1))
    indices = np.arange(joint_count)
    return (source_positions, target_positions, desired_positions, target_rotations,
            desired_rotations, sole_heights, indices, indices.copy(), 1.1)


def assess(*values):
    return preparation.assess_motion(*values)


def test_assessment_uses_source_stationary_anchors_and_one_fixed_floor_without_mutation():
    values = list(assessment_inputs())
    source_positions = values[0]
    source_before = source_positions.copy()
    values[5] += np.linspace(-.002, .002, len(values[5]))[:, None]

    metrics, failures, floor = assess(*values)

    assert failures == []
    assert metrics["stationary_anchor_intervals"] >= preparation.BOUNDS["stationary_anchor_min_intervals"]
    assert isinstance(floor, float)
    assert metrics["fixed_plane_source_y_m"] == floor
    assert .20 <= floor <= .23
    np.testing.assert_array_equal(source_positions, source_before)
    assert values[5].shape[1] == 4


def test_stationarity_is_measured_from_source_even_when_target_feet_slide():
    values = list(assessment_inputs())
    feet = [index for index, name in enumerate(JOINT_MAP)
            if name.endswith(("Foot", "ToeBase"))]
    target_indices = values[7]
    values[1][:, target_indices[feet], 0] = np.arange(len(values[1]))[:, None] * .01

    metrics, failures, _ = assess(*values)

    assert metrics["stationary_anchor_intervals"] >= preparation.BOUNDS["stationary_anchor_min_intervals"]
    assert "stationary_anchor_added_speed_p95_mps" in failures


@pytest.mark.parametrize(("speed_mps", "should_fail"), [(.049, False), (.051, True)])
def test_added_stationary_anchor_speed_uses_the_five_centimetre_bound(speed_mps, should_fail):
    values = list(assessment_inputs())
    feet = [index for index, name in enumerate(JOINT_MAP)
            if name.endswith(("Foot", "ToeBase"))]
    target_indices = values[7]
    values[1][:, target_indices[feet], 0] = (
        np.arange(len(values[1]))[:, None] * speed_mps / 60.
    )

    metrics, failures, _ = assess(*values)

    assert preparation.BOUNDS["stationary_anchor_added_speed_p95_mps"] == pytest.approx(.05)
    assert metrics["stationary_anchor_added_speed_p95_mps"] == pytest.approx(speed_mps)
    assert ("stationary_anchor_added_speed_p95_mps" in failures) is should_fail


def test_missing_source_foot_anchor_fails_instead_of_claiming_assessment():
    values = list(assessment_inputs())
    values[6] = values[6][:-1]
    with pytest.raises((IndexError, ValueError)):
        assess(*values)


def test_root_tracking_error_is_reported_as_a_failure():
    values = list(assessment_inputs())
    values[1][:, 0, 0] = .06

    metrics, failures, _ = assess(*values)

    assert metrics["root_tracking_max_m"] == pytest.approx(.06)
    assert "root_tracking_max_m" in failures


def test_single_frame_rotation_jump_is_caught_by_angular_step_bound():
    values = list(assessment_inputs())
    values[3][25, 0] = Rotation.from_euler("x", 40., degrees=True).as_matrix()

    metrics, failures, _ = assess(*values)

    assert metrics["added_angular_step_max_deg"] == pytest.approx(40.)
    assert "added_angular_step_max_deg" in failures


def test_changed_staged_bvh_fails_before_any_retarget_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "walk1_subject1.bvh").write_bytes(b"changed source")
    output = tmp_path / "out"
    row = {"filename": "walk1_subject1.bvh", "sha256": "0" * 64}
    args = {"source": str(source), "output": str(output), "bind": "unused", "gmr": "unused"}

    with pytest.raises(ValueError, match="identity changed"):
        preparation.convert_clip((row, args))
    assert not output.exists()


def _write_manifest(source, bind, *, bind_digest=None):
    (source / "source_manifest.json").write_text(json.dumps({
        "source": {"bind_bvh_sha256": bind_digest or preparation.sha256(bind)},
        "candidate_files": [{
            "filename": "walk1_subject1.bvh", "sha256": "a" * 64,
            "subject_id": "subject1",
        }],
    }), encoding="utf-8")


def _set_candidates(source, filenames):
    path = source / "source_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["candidate_files"] = [
        {"filename": filename, "sha256": hashlib.sha256(filename.encode()).hexdigest(),
         "subject_id": "subject" + Path(filename).stem.rsplit("subject", 1)[1]}
        for filename in filenames
    ]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_changed_bind_hash_fails_before_creating_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    bind = tmp_path / "Geno_bind.bvh"
    bind.write_bytes(b"bind")
    _write_manifest(source, bind, bind_digest="f" * 64)
    output = tmp_path / "retarget-v1"
    monkeypatch.setattr(sys, "argv", [
        "prepare_lafan_soma.py", "--source", str(source), "--bind", str(bind),
        "--gmr", str(tmp_path / "gmr"), "--output", str(output), "--max-frames", "4",
    ])

    with pytest.raises(ValueError, match="reference BVH identity changed"):
        preparation.main()
    assert not output.exists()


@pytest.mark.parametrize(
    ("cost_args", "expected_cost"),
    [([], 50.), (["--endpoint-position-cost", "500"], 500.)],
)
def test_partial_preflight_never_publishes_training_eligibility(
    tmp_path, monkeypatch, cost_args, expected_cost
):
    source = tmp_path / "source"
    source.mkdir()
    bind = tmp_path / "Geno_bind.bvh"
    bind.write_bytes(b"bind")
    _write_manifest(source, bind)
    output = tmp_path / "preflight-v1"

    class StubExecutor:
        def __init__(self, max_workers):
            assert max_workers == 1

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def map(self, function, tasks):
            results = []
            for _, config in tasks:
                assert config["endpoint_position_cost"] == expected_cost
                results.append({
                    "filename": "walk1_subject1", "shard": "shards/walk1_subject1.npz",
                    "shard_sha256": "b" * 64, "frames_60hz": 4, "split": "train",
                    "behavior": "walk",
                    "retarget_assessment": {"status": "passed", "failures": []},
                })
            return iter(results)

    monkeypatch.setattr(preparation, "ProcessPoolExecutor", StubExecutor)
    monkeypatch.setattr(sys, "argv", [
        "prepare_lafan_soma.py", "--source", str(source), "--bind", str(bind),
        "--gmr", str(tmp_path / "gmr"), "--output", str(output), "--exclude-failed-clips",
        "--max-frames", "4", "--workers", "1", *cost_args,
    ])

    preparation.main()

    report = json.loads((output / "retarget_review.json").read_text(encoding="utf-8"))
    assert report["status"] == "preflight_only"
    assert report["training_eligible"] is False
    assert report["clips"] == 1
    assert not (output / "preparation_failure.json").exists()


def test_excluded_failed_raw_clip_is_archived_but_never_accepted(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    bind = tmp_path / "Geno_bind.bvh"
    bind.write_bytes(b"bind")
    filenames = [
        f"{behavior}1_subject{subject}.bvh"
        for subject in (1, 2, 3)
        for behavior in ("walk", "run")
    ]
    filenames += [f"{behavior}{index}_subject{subject}.bvh"
                  for subject in (4, 5)
                  for behavior, index in (("walk", 1), ("run", 1), ("walk", 2),
                                          ("run", 2), ("walk", 3))]
    assert len(filenames) == 16
    _write_manifest(source, bind)
    _set_candidates(source, filenames)
    failed_filename = "walk1_subject1.bvh"
    output = tmp_path / "full-v3"

    class StubExecutor:
        def __init__(self, max_workers):
            assert max_workers == 1

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def map(self, _function, tasks):
            results = []
            for row, config in tasks:
                filename = row["filename"]
                stem = Path(filename).stem
                shard = "shards/" + stem + ".npz"
                shard_path = Path(config["output"]) / shard
                shard_path.write_bytes(("raw shard for " + filename).encode())
                subject = int(stem.rsplit("subject", 1)[1])
                split = "validation" if subject == 4 else "test" if subject == 5 else "train"
                failed = filename == failed_filename
                results.append({
                    "filename": stem, "shard": shard,
                    "shard_sha256": preparation.sha256(shard_path),
                    "frames_60hz": 120, "split": split,
                    "behavior": "run" if filename.startswith("run") else "walk",
                    "retarget_assessment": {
                        "status": "failed" if failed else "passed",
                        "failures": ["stationary_anchor_added_speed_p95_mps"] if failed else [],
                    },
                })
            return iter(results)

    monkeypatch.setattr(preparation, "ProcessPoolExecutor", StubExecutor)
    monkeypatch.setattr(sys, "argv", [
        "prepare_lafan_soma.py", "--source", str(source), "--bind", str(bind),
        "--gmr", str(tmp_path / "gmr"), "--output", str(output),
        "--workers", "1", "--exclude-failed-clips",
    ])

    preparation.main()

    all_results = json.loads((output / "conversion_results.json").read_text(encoding="utf-8"))
    accepted = json.loads((output / "accepted.json").read_text(encoding="utf-8"))
    rejected = json.loads((output / "rejected.json").read_text(encoding="utf-8"))
    report = json.loads((output / "retarget_review.json").read_text(encoding="utf-8"))
    rejected_row = next(row for row in rejected if row["filename"] == Path(failed_filename).stem)

    assert len(all_results) == len(filenames) == 16
    assert len(rejected) == 1
    assert rejected_row["retarget_assessment"]["failures"] == [
        "stationary_anchor_added_speed_p95_mps"
    ]
    raw_shard = output / rejected_row["shard"]
    assert raw_shard.is_file()
    assert raw_shard.read_bytes() == ("raw shard for " + failed_filename).encode()
    assert all(row["filename"] != Path(failed_filename).stem for row in accepted)
    assert rejected_row["shard"] not in report["accepted_shard_sha256"]
    assert report["source_inventory_clips"] == 16
    assert report["source_inventory_quality_status"] == "partial_rejection"
    assert report["accepted_contains_failed_clips"] is False
    assert report["training_eligible"] is True
    assert report["status"] == "passed"


def test_missing_run_in_a_split_blocks_full_collection_eligibility(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    bind = tmp_path / "Geno_bind.bvh"
    bind.write_bytes(b"bind")
    filenames = [
        "walk1_subject1.bvh", "run1_subject1.bvh",
        "walk1_subject4.bvh", "run1_subject4.bvh",
        "walk1_subject5.bvh",
    ]
    _write_manifest(source, bind)
    _set_candidates(source, filenames)
    output = tmp_path / "missing-test-run"

    class StubExecutor:
        def __init__(self, max_workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def map(self, _function, tasks):
            results = []
            for row, _config in tasks:
                filename = row["filename"]
                stem = Path(filename).stem
                subject = int(stem.rsplit("subject", 1)[1])
                results.append({
                    "filename": stem, "shard": "shards/" + stem + ".npz",
                    "shard_sha256": "c" * 64, "frames_60hz": 120,
                    "split": "validation" if subject == 4 else "test" if subject == 5 else "train",
                    "behavior": "run" if filename.startswith("run") else "walk",
                    "retarget_assessment": {"status": "passed", "failures": []},
                })
            return iter(results)

    monkeypatch.setattr(preparation, "ProcessPoolExecutor", StubExecutor)
    monkeypatch.setattr(sys, "argv", [
        "prepare_lafan_soma.py", "--source", str(source), "--bind", str(bind),
        "--gmr", str(tmp_path / "gmr"), "--output", str(output), "--workers", "1",
    ])

    preparation.main()

    report = json.loads((output / "retarget_review.json").read_text(encoding="utf-8"))
    assert report["source_inventory_clips"] == len(filenames)
    assert report["training_eligible"] is False
    assert report["status"] == "preflight_only"
    assert report["source_inventory_quality_status"] == "all_passed"


def test_missing_anatomical_anchor_cannot_publish_retarget_eligibility(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    bind = tmp_path / "Geno_bind.bvh"
    bind.write_bytes(b"bind")
    _write_manifest(source, bind)
    output = tmp_path / "missing-anchor-v1"

    class MissingAnchorExecutor:
        def __init__(self, max_workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def map(self, _function, _tasks):
            raise ValueError("Geno hierarchy is missing anatomical mapping anchor HeadEnd")

    monkeypatch.setattr(preparation, "ProcessPoolExecutor", MissingAnchorExecutor)
    monkeypatch.setattr(sys, "argv", [
        "prepare_lafan_soma.py", "--source", str(source), "--bind", str(bind),
        "--gmr", str(tmp_path / "gmr"), "--output", str(output), "--workers", "1",
    ])

    with pytest.raises(ValueError, match="missing anatomical mapping anchor"):
        preparation.main()

    failure = json.loads((output / "preparation_failure.json").read_text(encoding="utf-8"))
    assert failure["training_eligible"] is False
    assert not (output / "retarget_review.json").exists()
