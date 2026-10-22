"""Tiny-fixture contracts for the curated PARC correction/physics handoff."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_parc_curated_pipeline as curated
import correct_parc_candidates as correction
from run_parc_physics_record import validate_bundle


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _old_bundle(root):
    bundle = root / "old_bundle"
    files = {
        "PARC/scripts/convert_legacy_motions.py": b"official converter fixture\n",
        "PARC/data/assets/humanoid.xml": b"<mujoco/>\n",
        "tracker/model.pt": b"tracker fixture\n",
        "tracker/dm_env.yaml": b"env: {}\n",
        "tracker/agent_config.yaml": b"agent_name: fixture\n",
        "motions/raw_candidate.pkl": b"immutable raw candidate\n",
    }
    for relative, content in files.items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return bundle, files


def _correction_report(root, old_bundle, *, status="correction_finished_physics_pending",
                       tamper_after_hash=False):
    raw = old_bundle / "motions/raw_candidate.pkl"
    final = root / "corrected/final.pkl"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"corrected motion fixture\n")
    final_hash = _sha256(final)
    if tamper_after_hash:
        final.write_bytes(b"changed after report hash\n")
    intent = root / "corrected/intent.npz"
    commands = np.column_stack((np.linspace(0.0, 1.0, 12), np.zeros(12))).astype(np.float32)
    np.savez_compressed(intent, command_world_xy=commands,
                        source_frame_indices=np.arange(12, dtype=np.int64), fps=30)
    case = {
        "case": "terraces",
        "status": "kinematic_correction_complete_physics_pending",
        "source_sha256": _sha256(raw),
        "source_group": "course/run_01",
        "selected_candidate": {"candidate_key": "course/run_01/candidate_01"},
        "intent": str(intent),
        "intent_sha256": _sha256(intent),
        "stages": {"final": {"motion_file": str(final), "sha256": final_hash,
                               "metrics": {"frames": 12}}},
    }
    report_path = root / "corrections/correction_report.json"
    _write_json(report_path, {"status": status, "cases": [case]})
    return report_path.parent, raw, final


def test_reference_jobs_preserve_candidate_hash_scores_and_group(tmp_path):
    old_bundle, _ = _old_bundle(tmp_path)
    candidates = []
    showcase = []
    for index, name in enumerate(("terraces", "platform_2", "three_stair_flights")):
        key = "course/run_%02d/candidate_%02d" % (index, index)
        candidates.append({
            "candidate_key": key,
            "reported_source_sha256": "%064x" % (index + 1),
            "report_entry": {"weighted_contact_score": 1.0 + index,
                             "weighted_penetration_score": 2.0 + index,
                             "selection_score": 3.0 + index,
                             "endpoint_reached": True},
        })
        showcase.append({"name": name, "candidate_key": key})
    showcase.append({"name": "not_selected", "candidate_key": candidates[0]["candidate_key"]})
    catalog = tmp_path / "catalog.json"
    summary = tmp_path / "summary.json"
    output = tmp_path / "jobs/jobs.json"
    _write_json(catalog, {"candidates": candidates})
    _write_json(summary, {"showcase": showcase})

    curated.prepare_reference_jobs(catalog, summary, old_bundle, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["old_bundle"] == str(old_bundle)
    assert [row["name"] for row in payload["jobs"]] == [
        "terraces", "platform_2", "three_stair_flights"]
    for index, (row, candidate) in enumerate(zip(payload["jobs"], candidates)):
        assert row["candidate_key"] == candidate["candidate_key"]
        assert row["sha256"] == candidate["reported_source_sha256"]
        assert row["source_group"] == "course/run_%02d" % index
        assert row["contact_loss"] == candidate["report_entry"]["weighted_contact_score"]
        assert row["completed"] is True


def test_prepare_reference_jobs_rejects_missing_fixed_case_before_output(tmp_path):
    catalog = tmp_path / "catalog.json"
    summary = tmp_path / "summary.json"
    output = tmp_path / "jobs.json"
    _write_json(catalog, {"candidates": []})
    _write_json(summary, {"showcase": []})

    with pytest.raises(ValueError, match="three fixed regression cases"):
        curated.prepare_reference_jobs(catalog, summary, tmp_path / "bundle", output)
    assert not output.exists()


def test_mixed_generation_selection_keeps_source_intent_and_generation_provenance(tmp_path):
    generation = tmp_path / "generation.json"
    rows = []
    for candidate_id, score, completed in ((8, 2.0, True), (3, 1.0, True), (5, 0.1, False)):
        rows.append({
            "candidate_id": candidate_id,
            "motion": "candidate_%03d/generated.pkl" % candidate_id,
            "sha256": "%064x" % (candidate_id + 1),
            "endpoint_reached": completed,
            "weighted_contact_score": 0.5,
            "weighted_penetration_score": 0.5,
            "selection_score": score,
            "geometric_route_covered": True,
            "intent": "candidate_%03d/intent.npz" % candidate_id,
            "intent_sha256": "%064x" % (candidate_id + 101),
        })
    _write_json(generation, {"status": "kinematic_candidates_generated_physics_pending",
                             "candidates": rows})
    config = {"max_contact_loss": 3.0, "max_pen_loss": 8.0, "max_total_loss": 30.0,
              "mdm_path": {"top_k": 2}}

    selected = curated.select_generated_jobs(generation, "mixed_v0", config)

    assert [row["name"] for row in selected] == ["mixed_v0_c003", "mixed_v0_c008"]
    for row in selected:
        candidate_id = int(row["name"][-3:])
        source = next(source for source in rows if source["candidate_id"] == candidate_id)
        assert row["sha256"] == source["sha256"]
        assert row["intent"] == source["intent"]
        assert row["intent_sha256"] == source["intent_sha256"]
        assert row["generation_sha256"] == _sha256(generation)


def test_mixed_generation_selection_rejects_incomplete_report_or_too_few_candidates(tmp_path):
    generation = tmp_path / "generation.json"
    config = {"max_contact_loss": 3.0, "max_pen_loss": 8.0, "max_total_loss": 30.0,
              "mdm_path": {"top_k": 2}}
    _write_json(generation, {"status": "failed", "candidates": []})
    with pytest.raises(ValueError, match="Generation not complete"):
        curated.select_generated_jobs(generation, "mixed_v0", config)

    _write_json(generation, {"status": "kinematic_candidates_generated_physics_pending",
                             "candidates": [{"candidate_id": 1, "motion": "candidate.pkl",
                                 "sha256": "a" * 64, "endpoint_reached": True,
                                 "weighted_contact_score": 0.5,
                                 "weighted_penetration_score": 0.5,
                                 "selection_score": 1.0,
                                 "geometric_route_covered": True,
                                 "intent": "intent.npz", "intent_sha256": "b" * 64}]})
    with pytest.raises(ValueError, match="Insufficient candidates"):
        curated.select_generated_jobs(generation, "mixed_v0", config)


def test_mixed_generation_selection_rejects_endpoint_shortcut_without_route_coverage(tmp_path):
    generation = tmp_path / "generation.json"
    rows = [
        {"candidate_id": 1, "motion": "shortcut.pkl", "sha256": "a" * 64,
         "endpoint_reached": True, "weighted_contact_score": 0.1,
         "weighted_penetration_score": 0.1, "selection_score": 0.1,
         "geometric_route_covered": False, "intent": "shortcut.npz",
         "intent_sha256": "b" * 64},
        {"candidate_id": 2, "motion": "route.pkl", "sha256": "c" * 64,
         "endpoint_reached": True, "weighted_contact_score": 1.0,
         "weighted_penetration_score": 1.0, "selection_score": 2.0,
         "geometric_route_covered": True, "intent": "route.npz",
         "intent_sha256": "d" * 64},
    ]
    _write_json(generation, {"status": "kinematic_candidates_generated_physics_pending",
                             "candidates": rows})
    config = {"max_contact_loss": 3.0, "max_pen_loss": 8.0, "max_total_loss": 30.0,
              "mdm_path": {"top_k": 1}}

    selected = curated.select_generated_jobs(generation, "mixed_v0", config)

    assert [row["name"] for row in selected] == ["mixed_v0_c002"]
    assert selected[0]["raw_route_covered"] is True

    config["mdm_path"]["top_k"] = 2
    with pytest.raises(ValueError, match="Insufficient candidates"):
        curated.select_generated_jobs(generation, "mixed_v0", config)


def test_corrected_bundle_is_physics_compatible_and_preserves_raw_bundle(tmp_path):
    old_bundle, old_files = _old_bundle(tmp_path)
    corrections, raw_candidate, final_motion = _correction_report(tmp_path, old_bundle)
    before = {relative: (old_bundle / relative).read_bytes() for relative in old_files}
    output = tmp_path / "curated_bundle"

    manifest = curated.corrected_bundle(old_bundle, corrections, output)

    assert validate_bundle(output) == manifest
    assert (output / "PARC/scripts/convert_legacy_motions.py").read_bytes() == old_files[
        "PARC/scripts/convert_legacy_motions.py"]
    assert (output / "tracker/model.pt").read_bytes() == old_files["tracker/model.pt"]
    motion = output / "motions/terraces.pkl"
    assert motion.read_bytes() == final_motion.read_bytes()
    metadata = manifest["motions"]["motions/terraces.pkl"]
    assert metadata["sha256"] == _sha256(motion)
    assert metadata["source_frames"] == 12
    assert metadata["original_candidate_sha256"] == _sha256(raw_candidate)
    intent_relative = metadata["intent_path"]
    bundled_intent = output / intent_relative
    assert bundled_intent.is_file()
    assert metadata["intent_sha256"] == _sha256(bundled_intent)
    with np.load(bundled_intent, allow_pickle=False) as data:
        assert data["command_world_xy"].shape == (12, 2)
        assert np.array_equal(data["source_frame_indices"], np.arange(12))
        assert float(data["fps"]) == 30
    assert all((old_bundle / relative).read_bytes() == content
               for relative, content in before.items())


def test_export_intent_preserves_bundle_commands_and_hash(tmp_path):
    old_bundle, _ = _old_bundle(tmp_path)
    corrections, _, _ = _correction_report(tmp_path, old_bundle)
    bundle = tmp_path / "curated_bundle"
    curated.corrected_bundle(old_bundle, corrections, bundle)
    export = tmp_path / "export"
    (export / "recordings").mkdir(parents=True)
    _write_json(export / "manifest.json", {"recordings": [{
        "candidate_key": "terraces", "whole_motion": True, "frames": 12, "fps": 30.0,
    }]})

    curated.attach_export_intent(bundle, export)

    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    row = manifest["recordings"][0]
    output_intent = export / row["intent_path"]
    source_intent = bundle / "motions/terraces_intent.npz"
    assert row["intent_status"] == "captured_before_generation_frame_aligned"
    assert row["intent_sha256"] == _sha256(output_intent) == _sha256(source_intent)
    assert row["original_candidate_sha256"] == _sha256(old_bundle / "motions/raw_candidate.pkl")
    with np.load(output_intent, allow_pickle=False) as actual, np.load(source_intent, allow_pickle=False) as expected:
        assert np.array_equal(actual["command_world_xy"], expected["command_world_xy"])
        assert np.array_equal(actual["source_frame_indices"], expected["source_frame_indices"])


def test_partial_physics_recording_does_not_receive_full_motion_intent(tmp_path):
    old_bundle, _ = _old_bundle(tmp_path)
    corrections, _, _ = _correction_report(tmp_path, old_bundle)
    bundle = tmp_path / "curated_bundle"
    curated.corrected_bundle(old_bundle, corrections, bundle)
    export = tmp_path / "export"
    (export / "recordings").mkdir(parents=True)
    _write_json(export / "manifest.json", {"recordings": [{
        "candidate_key": "terraces", "whole_motion": False, "frames": 6, "fps": 30.0,
    }]})

    curated.attach_export_intent(bundle, export)

    row = json.loads((export / "manifest.json").read_text(encoding="utf-8"))["recordings"][0]
    assert row["intent_status"] == "unavailable_no_training_admission"
    assert "intent_path" not in row
    assert not list((export / "recordings").glob("*_intent.npz"))


def test_validate_intent_rejects_changed_hash_and_wrong_frame_contract(tmp_path):
    intent = tmp_path / "intent.npz"
    commands = np.zeros((12, 2), dtype=np.float32)
    np.savez_compressed(intent, command_world_xy=commands, fps=30)
    row = {"intent": str(intent), "intent_sha256": _sha256(intent)}

    assert np.array_equal(correction.validate_intent(row, frame_count=12), commands)
    with pytest.raises(ValueError, match="Intent frame/axis contract mismatch"):
        correction.validate_intent(row, frame_count=13)

    row["intent_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Source intent changed"):
        correction.validate_intent(row)


def test_corrected_bundle_rejects_unfinished_report_without_touching_source(tmp_path):
    old_bundle, old_files = _old_bundle(tmp_path)
    corrections, raw_candidate, _ = _correction_report(
        tmp_path, old_bundle, status="failed")
    source_before = raw_candidate.read_bytes()
    output = tmp_path / "curated_bundle"

    with pytest.raises(ValueError, match="No completed corrections"):
        curated.corrected_bundle(old_bundle, corrections, output)

    assert not output.exists()
    assert raw_candidate.read_bytes() == source_before
    assert all((old_bundle / relative).read_bytes() == content
               for relative, content in old_files.items())


def test_corrected_bundle_rejects_tampered_final_motion(tmp_path):
    old_bundle, _ = _old_bundle(tmp_path)
    corrections, raw_candidate, _ = _correction_report(
        tmp_path, old_bundle, tamper_after_hash=True)
    source_before = raw_candidate.read_bytes()

    with pytest.raises(ValueError, match="Correction source changed"):
        curated.corrected_bundle(old_bundle, corrections, tmp_path / "curated_bundle")

    assert raw_candidate.read_bytes() == source_before


def test_corrected_bundle_refuses_to_overwrite_existing_output(tmp_path):
    old_bundle, _ = _old_bundle(tmp_path)
    corrections, _, _ = _correction_report(tmp_path, old_bundle)
    output = tmp_path / "curated_bundle"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        curated.corrected_bundle(old_bundle, corrections, output)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_physics_bundle_validation_rejects_changed_motion(tmp_path):
    old_bundle, _ = _old_bundle(tmp_path)
    corrections, _, _ = _correction_report(tmp_path, old_bundle)
    output = tmp_path / "curated_bundle"
    curated.corrected_bundle(old_bundle, corrections, output)
    (output / "motions/terraces.pkl").write_bytes(b"tampered after manifest\n")

    with pytest.raises(ValueError, match="Bundle input changed"):
        validate_bundle(output)
