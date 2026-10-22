import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_parc_correction.py"
SPEC = importlib.util.spec_from_file_location("parc_correction_render", SCRIPT)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def test_selected_frames_are_deterministic_and_cover_endpoints():
    assert renderer.selected_frame_indices(5) == [0, 2, 4]
    assert renderer.selected_frame_indices(2) == [0, 1]
    assert renderer.selected_frame_indices(1) == [0]


def test_gif_durations_align_all_frames_at_half_speed():
    durations = renderer.gif_durations(30)
    assert len(durations) == 30
    assert set(durations) == {60, 70}
    assert sum(durations) == 2000


def test_missing_stages_are_recorded_without_claiming_success(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "physics_corrected": False,
        "cases": [{"case": "stairs_up", "status": "partial", "stages": {}}],
    }))

    result = renderer.run(report, tmp_path / "render", tmp_path / "humanoid.xml", 1440)

    row = result["results"][0]
    assert row["status"] == "skipped_missing_required_stages"
    assert "stages.raw" in row["missing"]
    assert "stages.optimized" in row["missing"]
    assert "stages.final" in row["missing"]
    assert (tmp_path / "render" / "render_manifest.json").is_file()


def _stage_files(tmp_path, motion_fps, diagnostic_fps=None, nested_motion_bytes=True):
    motion = tmp_path / "motion.pkl"
    motion_data = {"fps": motion_fps}
    if nested_motion_bytes:
        motion_data = pickle.dumps(motion_data)
    motion.write_bytes(pickle.dumps({"motion_data": motion_data}))
    diagnostics = tmp_path / "diagnostics.npz"
    values = {
        "body_names": np.asarray(["pelvis"]),
        "body_world_pos": np.zeros((1, 1, 3), dtype=np.float32),
        "body_world_rot": np.asarray([[[0, 0, 0, 1]]], dtype=np.float32),
        "body_contacts": np.zeros((1, 1), dtype=np.float32),
    }
    if diagnostic_fps is not None:
        values["fps"] = np.asarray(diagnostic_fps)
    np.savez_compressed(diagnostics, **values)
    return {"motion_file": motion, "diagnostics_file": diagnostics}


def test_motion_file_fps_is_authoritative_when_diagnostics_omit_fps(tmp_path):
    stage = _stage_files(tmp_path, 30)

    saved = renderer.load_saved_fk(stage)

    assert saved["fps"] == 30


def test_motion_data_dictionary_fps_is_supported(tmp_path):
    stage = _stage_files(tmp_path, 30, nested_motion_bytes=False)

    saved = renderer.load_saved_fk(stage)

    assert saved["fps"] == 30


def test_non_30_motion_fps_is_rejected(tmp_path):
    stage = _stage_files(tmp_path, 60)

    with pytest.raises(ValueError, match="authoritative motion fps"):
        renderer.load_saved_fk(stage)


def test_diagnostic_fps_must_match_authoritative_motion_fps(tmp_path):
    stage = _stage_files(tmp_path, 30, diagnostic_fps=60)

    with pytest.raises(ValueError, match="disagrees with motion fps"):
        renderer.load_saved_fk(stage)


def test_report_accepts_and_explicitly_skips_unrequested_variants(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "physics_corrected": False,
        "cases": [
            {"case": "stairs_up", "status": "partial", "stages": {}},
            {"case": "stairs_up_variant_1", "status": "complete", "stages": {}},
            {"case": "stairs_down", "status": "partial", "stages": {}},
            {"case": "stairs_down_variant_1", "status": "complete", "stages": {}},
            {"case": "hand_contact_continuation", "status": "partial", "stages": {}},
            {"case": "hand_contact_continuation_variant_1", "status": "complete", "stages": {}},
        ],
    }))

    result = renderer.run(report, tmp_path / "render", tmp_path / "humanoid.xml", 1440)

    assert result["results"][0]["status"] == "skipped_missing_required_stages"
    assert result["results"][1]["status"] == "skipped_unrequested_variant"
    assert result["results"][3]["status"] == "skipped_unrequested_variant"
    assert result["results"][5]["status"] == "skipped_unrequested_variant"


@pytest.mark.parametrize('failure_status', [
    'render_failed', 'skipped_missing_required_stages', 'rendered_with_missing_stages'])
def test_cli_does_not_hide_one_failed_case_behind_one_rendered_case(monkeypatch, failure_status):
    monkeypatch.setattr(renderer.sys, 'argv', ['render', '--report', 'report.json',
                                             '--output', 'renders'])
    monkeypatch.setattr(renderer, 'run', lambda *args: {
        'results': [{'status': 'rendered'}, {'status': failure_status}]})
    with pytest.raises(SystemExit, match='Incomplete rendering'):
        renderer.main()
