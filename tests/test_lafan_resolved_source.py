import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path

import pytest


MOTIONBRICKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOTIONBRICKS))
from scripts.prepare_lafan_resolved_source import prepare_source  # noqa: E402


def bvh(offset="0 10 0", frames=2, frame_time="0.016667"):
    return f"""HIERARCHY
ROOT Hips
{{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
 JOINT Knee
 {{
  OFFSET {offset}
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  End Site
  {{
   OFFSET 0 0 0
  }}
 }}
}}
MOTION
Frames: {frames}
Frame Time: {frame_time}
""".encode("utf-8")


def inputs(tmp_path, members=None, bind=b"auto"):
    if bind == b"auto":
        bind = bvh()
    bind_path = tmp_path / "Geno_bind.bvh"
    bind_path.write_bytes(bind)
    archive_path = tmp_path / "bvh.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in (members or [("walk1_subject1.bvh", bvh())]):
            archive.writestr(name, content)
    return archive_path, bind_path


def test_selects_only_matching_bvhs_and_records_subject_provenance(tmp_path):
    archive, bind = inputs(tmp_path, [
        ("walk1_subject1.bvh", bvh()),
        ("nested/run2_subject5.bvh", bvh(frames=3)),
        ("walk5_subject1.bvh", b"not a BVH"),
        ("Geno_bind.bvh", b"not a candidate"),
    ])
    output = tmp_path / "prepared"
    manifest = prepare_source(archive, bind, output)
    records = manifest["candidate_files"]
    assert [item["filename"] for item in records] == ["run2_subject5.bvh", "walk1_subject1.bvh"]
    assert records[0]["recording_id"] == "run2_subject5"
    assert records[0]["heldout_split_key"] == "subject5"
    assert records[0]["motion_frame_count"] == 3
    assert records[0]["source_fps"] == pytest.approx(1 / 0.016667)
    assert (output / "run2_subject5.bvh").is_file()
    assert not (output / "walk5_subject1.bvh").exists()
    assert manifest["source"]["crc_validated"] is True
    assert manifest["source"]["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert manifest["source"]["bind_bvh_sha256"] == hashlib.sha256(bind.read_bytes()).hexdigest()
    assert manifest["source"]["archive_bvh_count"] == 4
    assert records[0]["sha256"] == hashlib.sha256((output / records[0]["staged_path"]).read_bytes()).hexdigest()
    assert manifest["skeleton"]["joint_names"] == ["Hips", "Knee"]
    assert manifest["skeleton"]["parents"] == [None, "Hips"]
    assert manifest["training_eligible"] is False
    assert manifest["SOMA_retarget_verified"] is False
    assert manifest["license"]["identifier"] == "CC BY-NC-ND 4.0"
    assert manifest["license"]["commercial_use_approved"] is False
    assert json.loads((output / "source_manifest.json").read_text()) == manifest


def test_nonfinite_frame_time_is_rejected(tmp_path):
    archive, bind = inputs(tmp_path, [("walk1_subject1.bvh", bvh(frame_time="nan"))])
    with pytest.raises(ValueError, match="frame time and frame count"):
        prepare_source(archive, bind, tmp_path / "prepared")


@pytest.mark.parametrize("name", ["../walk1_subject1.bvh", r"..\walk1_subject1.bvh", "/walk1_subject1.bvh"])
def test_rejects_traversal_paths_without_creating_output(tmp_path, name):
    archive, bind = inputs(tmp_path, [(name, bvh())])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Unsafe ZIP member path"):
        prepare_source(archive, bind, output)
    assert not output.exists()


def test_rejects_duplicate_archive_member(tmp_path):
    archive, bind = inputs(tmp_path, [
        ("walk1_subject1.bvh", bvh()),
        ("walk1_subject1.bvh", bvh()),
    ])
    with pytest.raises(ValueError, match="Duplicate ZIP member"):
        prepare_source(archive, bind, tmp_path / "prepared")


def test_rejects_flattened_candidate_name_collision(tmp_path):
    archive, bind = inputs(tmp_path, [
        ("one/walk1_subject1.bvh", bvh()),
        ("two/walk1_subject1.bvh", bvh()),
    ])
    with pytest.raises(ValueError, match="Duplicate candidate output filename"):
        prepare_source(archive, bind, tmp_path / "prepared")


def test_crc_failure_and_missing_sources_fail_closed(tmp_path):
    archive, bind = inputs(tmp_path)
    raw = bytearray(archive.read_bytes())
    with zipfile.ZipFile(archive) as source_zip:
        info = source_zip.getinfo("walk1_subject1.bvh")
    name_length, extra_length = struct.unpack_from("<HH", raw, info.header_offset + 26)
    payload_offset = info.header_offset + 30 + name_length + extra_length
    raw[payload_offset + 20] ^= 0x40
    archive.write_bytes(raw)
    with pytest.raises(ValueError, match="CRC validation failed"):
        prepare_source(archive, bind, tmp_path / "crc-output")

    with pytest.raises(FileNotFoundError, match="Archive does not exist"):
        prepare_source(tmp_path / "missing.zip", bind, tmp_path / "missing-output")
    with pytest.raises(FileNotFoundError, match="Bind BVH does not exist"):
        prepare_source(archive, tmp_path / "missing-bind.bvh", tmp_path / "bind-output")


def test_rejects_archive_without_candidate_sources(tmp_path):
    archive, bind = inputs(tmp_path, [("Geno_stance.bvh", bvh())])
    with pytest.raises(ValueError, match="no supported walk/run candidate"):
        prepare_source(archive, bind, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_records_first_frame_offsets_without_assuming_bind_geometry(tmp_path):
    archive, bind = inputs(tmp_path, [("walk1_subject1.bvh", bvh(offset="0 11 0"))])
    manifest = prepare_source(archive, bind, tmp_path / "prepared")
    candidate = manifest["candidate_files"][0]
    assert candidate["offset_comparison_to_bind"]["max_nonroot_offset_difference_raw_units"] == 1.0
    assert candidate["offset_comparison_to_bind"]["offsets_identical_to_bind"] is False
    assert candidate["header_offsets_raw_units"][1] == [0.0, 11.0, 0.0]
    assert manifest["skeleton"]["constant_bind_offsets_assumed"] is False
    assert manifest["skeleton"]["nonroot_translation_channels_present"] is True
    assert manifest["skeleton"]["full_animation_channel_import_verified"] is False


def test_rejects_hierarchy_or_channels_mismatch_and_existing_output(tmp_path):
    archive, bind = inputs(tmp_path, [("walk1_subject1.bvh", bvh().replace(b"JOINT Knee", b"JOINT Ankle"))])
    with pytest.raises(ValueError, match="hierarchy differs"):
        prepare_source(archive, bind, tmp_path / "prepared")
    archive, bind = inputs(tmp_path, [("walk1_subject1.bvh", bvh().replace(
        b"Zrotation Yrotation Xrotation", b"Xrotation Yrotation Zrotation"))])
    with pytest.raises(ValueError, match="channels differ"):
        prepare_source(archive, bind, tmp_path / "prepared")
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError, match="must be new"):
        prepare_source(archive, bind, output)
