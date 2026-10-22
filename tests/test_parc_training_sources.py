import hashlib
import json

import pytest

from motionbricks.data.parc_training_sources import (
    ParcTrainingSourceValidationError,
    SCHEMA,
    SCHEMA_VERSION,
    validate_parc_training_sources,
    verify_parc_dataset_sources,
    require_parc_training_origins,
)


def _row(
    source_id,
    origin,
    *,
    split="train",
    disposition="admitted",
    identity=None,
    artifact_hash=None,
    manifest_hash=None,
    base_id=None,
    source_group=None,
    terrain_content_sha256=None,
    geometry_component=None,
    exclusion_reason=None,
):
    identity = identity or f"release-or-run/{source_id}"
    source_hash = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
    row = {
        "source_id": source_id,
        "origin": origin,
        "upstream_identity": identity,
        "upstream_sha256": artifact_hash or source_hash,
        "upstream_manifest_sha256": manifest_hash or ("c" if origin == "parc_generated" else "d") * 64,
        "base_id": base_id or f"base-{source_id}",
        "source_group": source_group or f"group-{source_id}",
        "disposition": disposition,
    }
    if disposition == "admitted":
        row["split"] = split
    else:
        row["exclusion_reason"] = exclusion_reason or "not approved for training"
    if terrain_content_sha256 is not None:
        row["terrain_content_sha256"] = terrain_content_sha256
    if geometry_component is not None:
        row["geometry_component"] = geometry_component
    return row


def _manifest(rows):
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_row_count": len(rows),
        "source_rows": rows,
    }


def _valid_rows():
    return [
        _row("generated-train", "parc_generated"),
        _row("official-train", "parc_official"),
        _row("official-validation", "parc_official", split="validation"),
        _row("generated-excluded", "parc_generated", disposition="excluded"),
    ]


def _dataset(tmp_path):
    rows = _valid_rows()
    source_path = tmp_path / 'parc_sources.json'
    source_path.write_text(json.dumps(_manifest(rows)), encoding='utf-8')
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {'parc_sources': {'snapshot': 'parc_sources.json', 'sha256': digest},
                'artifact_sha256': {'parc_sources.json': digest}}
    records = [{key: row[key] for key in ('source_id', 'origin', 'base_id', 'source_group', 'split')}
               | {'source_original_sha256': row['upstream_sha256']}
               for row in rows if row['disposition'] == 'admitted']
    return manifest, records


def _write_dataset_snapshot(tmp_path, rows, records):
    source_path = tmp_path / "parc_sources.json"
    source_path.write_text(json.dumps(_manifest(rows)), encoding="utf-8")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {
        "parc_sources": {"snapshot": "parc_sources.json", "sha256": digest},
        "artifact_sha256": {"parc_sources.json": digest},
    }
    return manifest, records


def test_dataset_binds_only_exact_admitted_rows(tmp_path):
    manifest, records = _dataset(tmp_path)
    summary = verify_parc_dataset_sources(tmp_path, manifest, records)
    assert summary['admitted_source_count'] == 3
    assert require_parc_training_origins([r for r in records if r['split'] == 'train']) == {
        'parc_generated': 1, 'parc_official': 1}


@pytest.mark.parametrize('change', ['extra', 'missing', 'origin', 'split', 'hash', 'snapshot'])
def test_dataset_rejects_unbound_or_modified_inputs(tmp_path, change):
    manifest, records = _dataset(tmp_path)
    if change == 'extra':
        records.append(dict(records[0]))
    elif change == 'missing':
        records.pop()
    elif change == 'snapshot':
        (tmp_path / 'parc_sources.json').write_text('{}', encoding='utf-8')
    elif change == 'hash':
        records[0]['source_original_sha256'] = 'e' * 64
    else:
        records[0][change] = 'lafan' if change == 'origin' else 'test'
    with pytest.raises(ParcTrainingSourceValidationError):
        verify_parc_dataset_sources(tmp_path, manifest, records)


def test_filtering_cannot_remove_official_training_source():
    with pytest.raises(ParcTrainingSourceValidationError, match='both PARC origins'):
        require_parc_training_origins([_row('gen', 'parc_generated')])


def test_training_policy_is_enforced_even_without_cli_flag(tmp_path):
    from scripts.train_native_mb import verify_training_source_policy
    manifest, records = _dataset(tmp_path)
    (tmp_path / 'accepted.json').write_text(json.dumps(records), encoding='utf-8')
    assert verify_training_source_policy(tmp_path, manifest)['admitted_source_count'] == 3
    with pytest.raises(ParcTrainingSourceValidationError):
        verify_training_source_policy(tmp_path, {}, parc_only=True)
    assert verify_training_source_policy(tmp_path, {}) is None


def test_validates_both_origins_and_accounts_for_excluded_sources():
    summary = validate_parc_training_sources(_manifest(_valid_rows()))

    assert summary == {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_row_count": 4,
        "admitted_source_count": 3,
        "excluded_source_count": 1,
        "split_counts": {"train": 2, "validation": 1, "test": 0},
        "train_origin_counts": {"parc_generated": 1, "parc_official": 1},
        "admitted_source_ids": ["generated-train", "official-train", "official-validation"],
        "excluded_source_ids": ["generated-excluded"],
    }
    assert "status" not in summary


def test_legacy_snapshot_without_geometry_fields_remains_compatible(tmp_path):
    manifest, records = _dataset(tmp_path)

    assert verify_parc_dataset_sources(tmp_path, manifest, records)["admitted_source_count"] == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("terrain_content_sha256", "not-a-hash"),
        ("terrain_content_sha256", "A" * 64),
        ("geometry_component", "  component"),
        ("geometry_component", ["component"]),
    ],
)
def test_rejects_malformed_optional_geometry_identity(field, value):
    rows = _valid_rows()
    rows[0][field] = value

    with pytest.raises(ParcTrainingSourceValidationError, match=field):
        validate_parc_training_sources(_manifest(rows))


def test_admitted_geometry_component_must_match_source_group():
    rows = _valid_rows()
    rows[0]["geometry_component"] = "different-component"

    with pytest.raises(ParcTrainingSourceValidationError, match="must equal source_group"):
        validate_parc_training_sources(_manifest(rows))


@pytest.mark.parametrize("identity_field", ["terrain_content_sha256", "geometry_component"])
def test_rejects_geometry_identity_split_leakage(identity_field):
    shared = "a" * 64 if identity_field == "terrain_content_sha256" else "shared-component"
    first = _row(
        "train", "parc_generated", split="train",
        **{identity_field: shared, **({"source_group": shared} if identity_field == "geometry_component" else {})},
    )
    second = _row(
        "validation", "parc_generated", split="validation",
        **{identity_field: shared, **({"source_group": shared} if identity_field == "geometry_component" else {})},
    )
    rows = [first, second, _row("official", "parc_official")]

    with pytest.raises(ParcTrainingSourceValidationError, match=f"{identity_field} .*leaks"):
        validate_parc_training_sources(_manifest(rows))


@pytest.mark.parametrize("field", ["terrain_content_sha256", "geometry_component"])
def test_dataset_accepted_record_must_match_optional_snapshot_identity(tmp_path, field):
    rows = _valid_rows()
    value = "a" * 64 if field == "terrain_content_sha256" else "group-generated-train"
    rows[0][field] = value
    if field == "geometry_component":
        rows[0]["source_group"] = value
    records = []
    for row in rows:
        if row["disposition"] != "admitted":
            continue
        record = {
            key: row[key]
            for key in ("source_id", "origin", "base_id", "source_group", "split")
        }
        record["source_original_sha256"] = row["upstream_sha256"]
        if field in row:
            record[field] = row[field]
        records.append(record)
    manifest, records = _write_dataset_snapshot(tmp_path, rows, records)

    assert verify_parc_dataset_sources(tmp_path, manifest, records)["admitted_source_count"] == 3
    records[0][field] = "b" * 64 if field == "terrain_content_sha256" else "tampered-component"
    with pytest.raises(ParcTrainingSourceValidationError, match=f"record mismatch: {field}"):
        verify_parc_dataset_sources(tmp_path, manifest, records)


def test_loads_manifest_from_json_path(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(_manifest(_valid_rows())), encoding="utf-8")

    assert validate_parc_training_sources(path)["source_row_count"] == 4


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_rejects_unsupported_schema_version(version):
    manifest = _manifest(_valid_rows())
    manifest["schema_version"] = version

    with pytest.raises(ParcTrainingSourceValidationError, match="schema_version"):
        validate_parc_training_sources(manifest)


@pytest.mark.parametrize("origin", ["unknown", "mixed", "parc_generated+parc_official", ["parc_generated"]])
def test_rejects_unknown_or_mixed_origins(origin):
    rows = _valid_rows()
    rows[0]["origin"] = origin

    with pytest.raises(ParcTrainingSourceValidationError, match="origin"):
        validate_parc_training_sources(_manifest(rows))


@pytest.mark.parametrize("field", ["upstream_identity", "upstream_sha256", "upstream_manifest_sha256"])
def test_requires_explicit_immutable_upstream_identity_and_hashes(field):
    rows = _valid_rows()
    rows[0].pop(field)

    with pytest.raises(ParcTrainingSourceValidationError, match=field):
        validate_parc_training_sources(_manifest(rows))


def test_rejects_malformed_upstream_hash():
    rows = _valid_rows()
    rows[0]["upstream_sha256"] = "not-a-sha256"

    with pytest.raises(ParcTrainingSourceValidationError, match="SHA-256"):
        validate_parc_training_sources(_manifest(rows))


def test_rejects_conflicting_hashes_for_reused_upstream_identity():
    first = _row("same-a", "parc_generated", identity="pinned-run/candidate")
    second = _row(
        "same-b", "parc_generated", identity="pinned-run/candidate", artifact_hash="e" * 64
    )
    rows = [first, second, _row("official", "parc_official")]

    with pytest.raises(ParcTrainingSourceValidationError, match="conflicting origin or hashes"):
        validate_parc_training_sources(_manifest(rows))


def test_rejects_same_upstream_hash_claimed_by_mixed_origins():
    shared_hash = "e" * 64
    rows = [
        _row("generated", "parc_generated", artifact_hash=shared_hash),
        _row("official", "parc_official", artifact_hash=shared_hash),
    ]

    with pytest.raises(ParcTrainingSourceValidationError, match="mixed origins"):
        validate_parc_training_sources(_manifest(rows))


@pytest.mark.parametrize("field", ["base_id", "source_group"])
def test_rejects_base_or_source_group_split_leakage(field):
    first = _row("train", "parc_generated", split="train", **{field: "shared"})
    second = _row("validation", "parc_generated", split="validation", **{field: "shared"})
    rows = [first, second, _row("official", "parc_official")]

    with pytest.raises(ParcTrainingSourceValidationError, match=f"{field} .*leaks"):
        validate_parc_training_sources(_manifest(rows))


def test_rejects_upstream_identity_split_leakage():
    identity = "pinned-release/reused-asset"
    shared_hash = hashlib.sha256(b"same upstream asset").hexdigest()
    rows = [
        _row("train", "parc_generated", identity=identity, split="train", artifact_hash=shared_hash),
        _row("validation", "parc_generated", identity=identity, split="validation", artifact_hash=shared_hash),
        _row("official", "parc_official"),
    ]

    with pytest.raises(ParcTrainingSourceValidationError, match="upstream_identity .*leaks"):
        validate_parc_training_sources(_manifest(rows))


def test_requires_both_source_origins_in_train():
    rows = [
        _row("generated", "parc_generated"),
        _row("official-validation", "parc_official", split="validation"),
    ]

    with pytest.raises(ParcTrainingSourceValidationError, match="both PARC source origins"):
        validate_parc_training_sources(_manifest(rows))


def test_rejects_missing_disposition_instead_of_treating_pass_as_admission():
    rows = _valid_rows()
    rows[0].pop("disposition")
    rows[0]["status"] = "passed"

    with pytest.raises(ParcTrainingSourceValidationError, match="disposition"):
        validate_parc_training_sources(_manifest(rows))


def test_rejects_row_count_mismatch_and_duplicate_source_ids():
    manifest = _manifest(_valid_rows())
    manifest["source_row_count"] -= 1
    with pytest.raises(ParcTrainingSourceValidationError, match="source_row_count"):
        validate_parc_training_sources(manifest)

    rows = _valid_rows()
    rows[1]["source_id"] = rows[0]["source_id"]
    with pytest.raises(ParcTrainingSourceValidationError, match="Duplicate source_id"):
        validate_parc_training_sources(_manifest(rows))


def test_excluded_rows_need_reasons_and_cannot_be_assigned_a_split():
    rows = _valid_rows()
    rows[-1]["exclusion_reason"] = "  "
    with pytest.raises(ParcTrainingSourceValidationError, match="exclusion_reason"):
        validate_parc_training_sources(_manifest(rows))

    rows = _valid_rows()
    rows[-1]["split"] = "train"
    with pytest.raises(ParcTrainingSourceValidationError, match="cannot be assigned"):
        validate_parc_training_sources(_manifest(rows))


@pytest.mark.parametrize("disposition", ["passed", ["admitted"]])
def test_rejects_unsupported_disposition_values(disposition):
    rows = _valid_rows()
    rows[0]["disposition"] = disposition

    with pytest.raises(ParcTrainingSourceValidationError, match="disposition"):
        validate_parc_training_sources(_manifest(rows))
