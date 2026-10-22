import importlib.util
import hashlib
import json
from pathlib import Path
import random
import sys
import zipfile

import pytest


_PROJECT_ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "prepare_parc_release_subset_tested",
    _PROJECT_ROOT / "scripts" / "prepare_parc_release_subset.py",
)
_SUBSET = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SUBSET
_SPEC.loader.exec_module(_SUBSET)


def _entry(name, payload=b"payload"):
    import zlib

    return {
        "name": name,
        "size": len(payload),
        "compressed_size": len(payload),
        "crc32": zlib.crc32(payload) & 0xFFFFFFFF,
    }


def _selection_fixture():
    entries = [_entry("dec_release/initial/legacy.pkl")]
    for family in _SUBSET.OFFICIAL_FAMILIES:
        if family == "initial_aug":
            entries.extend([
                _entry("dec_release/initial_aug/stairs/keyauthorstairs_001_aug001_dm_aug0.pkl"),
                _entry("dec_release/initial_aug/stairs/keyauthorstairs_001_flipped_aug001_dm_aug0.pkl"),
                _entry("dec_release/initial_aug/stairs/keyauthorstairs_001_aug002_dm_aug1.pkl"),
                _entry("dec_release/initial_aug/climbing/climb_001_aug001_dm_aug0.pkl"),
            ])
            continue
        prefix = "BOXES" if family.endswith("_boxes") else "teaser"
        chunk = "boxes_0_99" if family.endswith("_boxes") else "teaser_2000_2049"
        entries.extend([
            _entry(f"dec_release/{family}/{chunk}/{prefix}_41_0_opt_dm.pkl"),
            _entry(f"dec_release/{family}/{chunk}/{prefix}_41_0_opt_flipped_dm.pkl"),
            _entry(f"dec_release/{family}/{chunk}/{prefix}_41_1_opt_dm.pkl"),
            _entry(f"dec_release/{family}/{chunk}/{prefix}_42_0_opt_dm.pkl"),
            _entry(f"dec_release/{family}/{chunk}/{prefix}_42_1_opt_dm.pkl"),
            _entry(f"dec_release/{family}/{chunk}/{prefix}_43_0_opt_dm.pkl"),
        ])
    return entries


def test_selection_covers_all_official_families_evenly_and_excludes_legacy_initial():
    result = _SUBSET.select_members(_selection_fixture(), groups_per_family=2)
    counts = result["selection"]["families"]

    assert tuple(counts) == _SUBSET.OFFICIAL_FAMILIES
    assert {row["family"] for row in result["selected"]} == set(_SUBSET.OFFICIAL_FAMILIES)
    assert {item["selected_source_group_count"] for item in counts.values()} == {2}
    assert result["selection"]["source_group_count"] == 20
    assert "dec_release/initial/legacy.pkl" not in {row["name"] for row in result["selected"]}
    assert result["excluded_by_member"]["dec_release/initial/legacy.pkl"] == "legacy_flat_initial"
    assert result["selection"]["selected_member_count"] == len(result["selected"])
    assert result["selection"]["excluded_member_count"] == (
        result["selection"]["archive_pickle_member_count"] - len(result["selected"])
    )


def test_initial_aug_collapses_mirror_and_all_augmentation_variants_without_losing_stairs():
    entries = [
        _entry(f"dec_release/initial_aug/stairs/keyauthorstairs_001_aug{index:03d}_dm_aug{index}.pkl")
        for index in range(50)
    ]
    entries.extend([
        _entry("dec_release/initial_aug/stairs/keyauthorstairs_001_flipped_aug001_dm_aug0.pkl"),
        _entry("dec_release/initial_aug/stairs/keyauthorstairs_001_before_opt_dm.pkl"),
        _entry("dec_release/initial_aug/stairs/keyauthorstairs_001_wip_dm.pkl"),
    ])
    result = _SUBSET.select_members(
        entries,
        groups_per_family=1,
        families=("initial_aug",),
    )

    assert len(result["selected"]) == 1
    sample = result["selected"][0]
    assert sample["terrain_family"] == "stairs"
    assert sample["base_id"] == "official:initial_aug:base_motion:stairs:keyauthorstairs_001"
    assert sample["source_group"] == sample["base_id"]
    assert sample["base_derivation"]["augmentation_tokens"]
    assert result["selection"]["families"]["initial_aug"]["source_group_count"] == 1
    assert all(
        result["excluded_by_member"][row["name"]] == "normalized_variant_not_selected"
        for row in entries if row["name"] != sample["name"]
    )


def test_iteration_source_group_uses_terrain_id_and_base_id_keeps_candidate():
    family = "rerun_april272025_iter_1_boxes"
    entries = [
        _entry(f"dec_release/{family}/boxes_0_99/boxes_77_0_opt_dm.pkl"),
        _entry(f"dec_release/{family}/boxes_0_99/boxes_77_0_opt_flipped_dm.pkl"),
        _entry(f"dec_release/{family}/boxes_0_99/boxes_77_1_opt_dm.pkl"),
    ]
    result = _SUBSET.select_members(entries, groups_per_family=1, families=(family,))

    assert len(result["selected"]) == 2
    assert {row["source_group"] for row in result["selected"]} == {
        f"official:{family}:terrain:boxes_77"
    }
    assert {row["base_id"] for row in result["selected"]} == {
        f"official:{family}:terrain:boxes_77:candidate:0",
        f"official:{family}:terrain:boxes_77:candidate:1",
    }
    assert {row["terrain_family"] for row in result["selected"]} == {"boxes"}
    # The original spelling is preferred when a mirrored pair is present.
    assert all("flipped" not in row["name"] for row in result["selected"])


def test_selection_is_deterministic_under_index_reordering():
    entries = _selection_fixture()
    expected = _SUBSET.select_members(entries, groups_per_family=3)
    random.Random(17).shuffle(entries)
    actual = _SUBSET.select_members(entries, groups_per_family=3)

    assert [(row["name"], row["source_group"], row["base_id"])
            for row in actual["selected"]] == [
        (row["name"], row["source_group"], row["base_id"])
        for row in expected["selected"]
    ]
    assert actual["selection"] == expected["selection"]


def test_iteration_group_cap_applies_to_members_without_exceeding_limit():
    family = "dec2024_iter_2_boxes"
    entries = [
        _entry(f"dec_release/{family}/mdm_batch_000/BOXES_{terrain}_{candidate}_opt_dm.pkl")
        for terrain in range(10)
        for candidate in range(2)
    ]
    result = _SUBSET.select_members(
        entries,
        groups_per_family=8,
        max_members_per_family=5,
        families=(family,),
    )

    assert len(result["selected"]) <= 5
    assert result["selection"]["families"][family]["selected_source_group_count"] <= 8
    selected_terrain_groups = {row["source_group"] for row in result["selected"]}
    assert all(row["source_group"] in selected_terrain_groups for row in result["selected"])


def test_archive_member_names_reject_path_traversal_and_windows_paths():
    for name in (
        "dec_release/initial_aug/../outside.pkl",
        "../outside.pkl",
        "C:/outside.pkl",
        "dec_release\\initial_aug\\motion.pkl",
    ):
        with pytest.raises(ValueError):
            _SUBSET.validate_member_name(name)


def test_index_validation_rejects_zip_traversal_before_extraction(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    member = "dec_release/initial_aug/../../outside.pkl"
    archive_path.write_bytes(b"")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, b"not extracted")
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.infolist()[0]
        rows = [{
            "name": info.filename,
            "size": info.file_size,
            "compressed_size": info.compress_size,
            "crc32": info.CRC,
        }]
        with pytest.raises(ValueError, match="Traversal"):
            _SUBSET.validate_archive_index(archive, rows)


def test_archive_hash_mismatch_fails_before_extraction(tmp_path):
    archive_path = tmp_path / "small.zip"
    archive_path.write_bytes(b"complete but not official")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _SUBSET.verify_archive(
            archive_path,
            expected_sha256="0" * 64,
            expected_size=archive_path.stat().st_size,
        )


def test_extraction_uses_sha256_basename_and_refuses_existing_output(tmp_path):
    archive_path = tmp_path / "members.zip"
    name = "dec_release/initial_aug/stairs/keyauthorstairs_001_dm.pkl"
    payload = b"opaque paired member payload"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)
    selected = [{"name": name, "size": len(payload), "family": "initial_aug"}]
    output = tmp_path / "subset"
    with zipfile.ZipFile(archive_path) as archive:
        samples = _SUBSET.extract_selected(archive, selected, output)

    sha = hashlib.sha256(payload).hexdigest()
    assert Path(samples[0]["local_file"]).name == f"{sha}.pkl"
    assert Path(samples[0]["local_file"]).read_bytes() == payload
    assert samples[0]["sha256"] == sha

    marker = output / "keep.txt"
    marker.write_text("leave intact", encoding="utf-8")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            _SUBSET.extract_selected(archive, selected, output)
    assert marker.read_text(encoding="utf-8") == "leave intact"
    assert not (output / "samples" / "keyauthorstairs_001_dm.pkl").exists()


def test_inspection_sample_fields_match_catalog_handoff_contract():
    family = "dec2024_iter_3_siggraph_terrain"
    entries = [_entry(f"dec_release/{family}/TEASER_TERRAIN_020/dec2024_teaser_2020_0_opt_dm.pkl")]
    result = _SUBSET.select_members(entries, groups_per_family=1, families=(family,))
    sample = result["selected"][0]

    assert {"source_group", "base_id", "terrain_family"} <= sample.keys()
    assert sample["source_group"] == f"official:{family}:terrain:dec2024_teaser_2020"
    assert sample["base_id"] == f"official:{family}:terrain:dec2024_teaser_2020:candidate:0"
    assert sample["terrain_family"] == "siggraph_terrain"


def test_prepare_subset_writes_catalog_compatible_inspection_and_manifest(tmp_path, monkeypatch):
    archive_path = tmp_path / "release.zip"
    payloads = {}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for family in _SUBSET.OFFICIAL_FAMILIES:
            if family == "initial_aug":
                name = "dec_release/initial_aug/stairs/keyauthorstairs_001_before_opt_dm.pkl"
            else:
                prefix = "BOXES" if family.endswith("_boxes") else "teaser"
                name = f"dec_release/{family}/bucket/{prefix}_100_0_opt_dm.pkl"
            payload = f"opaque paired payload for {family}".encode("utf-8")
            archive.writestr(name, payload)
            payloads[name] = payload

    with zipfile.ZipFile(archive_path) as archive:
        index_rows = [{
            "name": info.filename,
            "size": info.file_size,
            "compressed_size": info.compress_size,
            "crc32": info.CRC,
        } for info in archive.infolist()]
    index_path = tmp_path / "archive_index.json"
    index_path.write_text(
        json.dumps(index_rows),
        encoding="utf-8",
    )
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_size = archive_path.stat().st_size
    monkeypatch.setattr(_SUBSET, "OFFICIAL_ARCHIVE_SHA256", archive_sha)
    monkeypatch.setattr(_SUBSET, "OFFICIAL_ARCHIVE_SIZE", archive_size)

    audit_path = tmp_path / "release_audit.json"
    audit_path.write_text(
        json.dumps({
            "revision": _SUBSET.OFFICIAL_REVISION,
            "archive": {
                "path": "dec_release.zip",
                "size": archive_size,
                "lfs": {"oid": archive_sha, "size": archive_size},
            },
        }),
        encoding="utf-8",
    )
    output = tmp_path / "subset"
    result = _SUBSET.prepare_subset(
        archive_path,
        index_path,
        audit_path,
        output,
        groups_per_family=1,
        max_members_per_family=4,
    )

    inspection = json.loads(Path(result["inspection"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert len(inspection["samples"]) == len(_SUBSET.OFFICIAL_FAMILIES)
    assert manifest["selected_source_group_count"] == len(_SUBSET.OFFICIAL_FAMILIES)
    assert manifest["excluded_member_count"] == 0
    for sample in inspection["samples"]:
        assert {"member", "local_file", "sha256", "source_group", "base_id", "terrain_family"} <= sample.keys()
        assert Path(sample["local_file"]).read_bytes() == payloads[sample["member"]]
        assert Path(sample["local_file"]).name == f"{sample['sha256']}.pkl"
