import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_parc_source_selection import build_manifest


def _write_inspected_source(audit_dir: Path, *, filename="sample.pkl", member="sample/member.pkl"):
    stairs_dir = audit_dir / "stairs"
    stairs_dir.mkdir(parents=True)
    frames = 4
    motion = {
        "root_pos": np.array([[0, 0, 0.5], [0, 0, 0.7], [0, 0, 0.8], [0, 0, 1.0]], np.float32),
        "root_rot": np.tile([0, 0, 0, 1], (frames, 1)).astype(np.float32),
        "joint_rot": np.tile([0, 0, 0, 1], (frames, 14, 1)).astype(np.float32),
        "body_contacts": np.zeros((frames, 15), np.float32),
        "fps": 30,
    }
    motion["body_contacts"][:, 5] = [0, 1, 1, 0]
    motion["body_contacts"][2:, 8] = 1
    terrain = {"hf": np.zeros((3, 4), np.float32), "min_point": np.zeros(2, np.float32), "dx": 0.4}
    payload = pickle.dumps({
        "motion_data": pickle.dumps(motion),
        "terrain_data": pickle.dumps(terrain),
        "misc_data": pickle.dumps({}),
    })
    local = stairs_dir / filename
    local.write_bytes(payload)
    inspection = {
        "revision": "release-test-revision",
        "archive": {"path": "dec_release.zip", "lfs": {"oid": "a" * 64}, "size": 1000},
        "samples": [{
            "member": member,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "local_file": str(local.resolve()),
            "size": len(payload),
            "motion_data": {"root_pos": {"shape": [frames, 3]}},
        }],
    }
    (stairs_dir / "inspection.json").write_text(json.dumps(inspection), encoding="utf-8")
    samples_dir = audit_dir / "samples"
    samples_dir.mkdir()
    (samples_dir / "inspection.json").write_text(json.dumps({
        "revision": "release-test-revision", "archive": inspection["archive"], "samples": []
    }), encoding="utf-8")
    return local, payload, member


def _patch_specs(monkeypatch, filename, member):
    import build_parc_source_selection as builder

    monkeypatch.setattr(builder, "SOURCE_SPECS", ({
        "behavior": "test_behavior", "member": member, "filename": filename,
        "source_group": "test_group", "description": "test source",
    },))


def test_manifest_binds_hash_metrics_contacts_and_unassigned_split(tmp_path, monkeypatch):
    local, _, member = _write_inspected_source(tmp_path)
    _patch_specs(monkeypatch, local.name, member)

    manifest = build_manifest(tmp_path)
    item = manifest["selected_items"][0]

    assert item["source"]["local_file"] == str(local.resolve())
    assert item["source"]["sha256"] == item["source"]["archive_member_sha256"]
    assert item["metrics"]["frame_count"] == 4
    assert item["metrics"]["fps"] == 30
    assert item["metrics"]["root_z_displacement_m_source_coordinates"] == pytest.approx(0.5)
    counts = item["metrics"]["contact_labels"]["positive_frame_count_by_body"]
    assert counts["right_hand"] == 2
    assert counts["left_hand"] == 2
    assert item["metrics"]["contact_labels"]["bodies_with_any_positive_frame"] == 2
    assert item["split"] == "unassigned_not_training_data"
    assert item["training_approved"] is False
    assert item["retargeted"] is False
    assert manifest["selection_policy"]["no_train_validation_leakage"] is True


def test_rejects_local_source_hash_mismatch(tmp_path, monkeypatch):
    local, _, member = _write_inspected_source(tmp_path)
    payload = bytearray(local.read_bytes())
    payload[-1] ^= 1
    local.write_bytes(payload)
    _patch_specs(monkeypatch, local.name, member)

    with pytest.raises(ValueError, match="SHA-256"):
        build_manifest(tmp_path)


def test_rejects_missing_inspected_member(tmp_path, monkeypatch):
    local, _, member = _write_inspected_source(tmp_path)
    _patch_specs(monkeypatch, local.name, member + ".missing")

    with pytest.raises(ValueError, match="exactly one inspected archive member"):
        build_manifest(tmp_path)


def test_rejects_malformed_motion_data(tmp_path, monkeypatch):
    local, _, member = _write_inspected_source(tmp_path)
    outer = pickle.loads(local.read_bytes())
    motion = pickle.loads(outer["motion_data"])
    motion["body_contacts"] = np.zeros((4, 14), np.float32)
    outer["motion_data"] = pickle.dumps(motion)
    payload = pickle.dumps(outer)
    local.write_bytes(payload)
    inspection_path = tmp_path / "stairs" / "inspection.json"
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    inspection["samples"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    inspection["samples"][0]["size"] = len(payload)
    inspection_path.write_text(json.dumps(inspection), encoding="utf-8")
    _patch_specs(monkeypatch, local.name, member)

    with pytest.raises(ValueError, match="body_contacts"):
        build_manifest(tmp_path)
