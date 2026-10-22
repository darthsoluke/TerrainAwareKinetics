import hashlib
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


_PACKAGE_ROOT = Path(__file__).parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))
_KIMODO_ROOT = _PACKAGE_ROOT / "out/vendor/kimodo"
if str(_KIMODO_ROOT) not in sys.path:
    sys.path.insert(0, str(_KIMODO_ROOT))

from motionbricks.data.native_mb_data import build_native_representation  # noqa: E402
from motionbricks.data.native_mb_terrain_protocol import validate_terrain_dataset  # noqa: E402
from motionbricks.data.parc_training_sources import (  # noqa: E402
    ParcTrainingSourceValidationError,
    verify_parc_dataset_sources,
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_BUILDER = _load_module(
    "build_parc_only_dataset_tested",
    _PACKAGE_ROOT / "scripts" / "build_parc_only_dataset.py",
)
_NATIVE_TEST_FIXTURES = _load_module(
    "parc_native_dataset_fixture_helpers",
    Path(__file__).with_name("test_parc_native_dataset.py"),
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _catalog_fixture(tmp_path):
    admissions, splits_path, source_items = _NATIVE_TEST_FIXTURES._fixture(tmp_path)
    split_rows = json.loads(splits_path.read_text(encoding="utf-8"))["items"]
    splits = {item["source_id"]: item for item in split_rows}
    rows = []
    for index, item in enumerate(source_items):
        source_id = item["source_id"]
        source_path = Path(item["source_path"]).resolve()
        retarget_path = Path(item["retarget_path"]).resolve()
        # These source groups describe different physical terrain instances.
        with np.load(retarget_path, allow_pickle=False) as shard:
            arrays = {key: shard[key].copy() for key in shard.files}
        arrays["hf"] += np.float32(index * .001)
        np.savez_compressed(retarget_path, **arrays)
        report_path = retarget_path.parent / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["shard"] = str(retarget_path)
        report["source"]["local_file"] = str(source_path)
        report["shard_sha256"] = _hash(retarget_path)
        report_path.write_text(json.dumps(report), encoding="utf-8")

        origin = "parc_generated" if source_id == "course-b" else "parc_official"
        revision = "a" * 40
        member_artifact = f"release-r1/{source_id}.pkl"
        identity = f"mxucg/PARC@{revision}:{member_artifact}"
        upstream_path = retarget_path.parent / "upstream_manifest.json"
        if origin == "parc_official":
            upstream = {
                "revision": revision,
                "archive": {"lfs": {"oid": "b" * 64}},
                "samples": [{
                    "member": member_artifact,
                    "sha256": _hash(source_path),
                    "local_file": str(source_path),
                }],
            }
        else:
            upstream = {
                "schema": "parc.generated_run_manifest.v1",
                "upstream_identity": identity,
                "source_sha256": _hash(source_path),
            }
        upstream_path.write_text(json.dumps(upstream), encoding="utf-8")
        split = "train" if source_id in {"course-a", "course-b"} else "validation"
        row = {
            "source_id": source_id,
            "origin": origin,
            "source_group": splits[source_id]["source_group"],
            "base_id": f"base-{source_id}",
            "terrain_family": splits[source_id]["terrain_family"],
            "terrain_seed": splits[source_id]["terrain_seed"],
            "split": split,
            "upstream_identity": identity,
            "upstream_manifest_path": str(upstream_path),
            "upstream_manifest_sha256": _hash(upstream_path),
            "source_path": str(source_path),
            "source_sha256": _hash(source_path),
            "official_revision": revision if origin == "parc_official" else None,
            "official_archive_sha256": "b" * 64 if origin == "parc_official" else None,
            "official_member": member_artifact if origin == "parc_official" else None,
            "retarget_path": str(retarget_path),
            "retarget_sha256": _hash(retarget_path),
            "report_path": str(report_path),
            "report_sha256": _hash(report_path),
            "disposition": "admitted",
        }
        if origin == "parc_generated":
            row.update({
                "physics_report_path": item["physics_report_path"],
                "physics_report_sha256": item["physics_report_sha256"],
                "export_manifest_path": item["export_manifest_path"],
                "export_manifest_sha256": item["export_manifest_sha256"],
                "physics_candidate_key": item["physics_candidate_key"],
            })
        rows.append(row)
    catalog = {
        "schema": "parc.only_teacher_catalog.v1",
        "purpose": "paired geometric teacher for native MB training; not a dynamics certificate",
        "quality_approved": False,
        "rows": rows,
    }
    catalog_path = tmp_path / "input" / "parc_only_catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return catalog_path, catalog


def test_builds_native_parc_only_dataset_with_causal_proxy_and_train_only_stats(tmp_path):
    catalog_path, catalog = _catalog_fixture(tmp_path)
    output = tmp_path / "parc-only-v1"

    manifest = _BUILDER.build_dataset(catalog_path, output)
    accepted = json.loads((output / "accepted.json").read_text(encoding="utf-8"))

    assert manifest["quality_approved"] is False
    assert manifest["training_usable"] is True
    assert manifest["train_origin_counts"] == {"parc_generated": 1, "parc_official": 1}
    assert manifest["split_counts"] == {"train": 2, "validation": 1, "test": 0}
    assert manifest["parc_sources"] == {
        "snapshot": "parc_sources.json",
        "sha256": _hash(output / "parc_sources.json"),
    }
    assert manifest["artifact_sha256"]["parc_sources.json"] == manifest["parc_sources"]["sha256"]
    source_manifest = json.loads((output / "parc_sources.json").read_text(encoding="utf-8"))
    assert source_manifest["source_row_count"] == 3
    assert len(source_manifest["source_rows"]) == 3
    verify_parc_dataset_sources(output, manifest, accepted)
    assert manifest["stats_contract"]["split"] == "train"
    assert manifest["stats_contract"]["windows_augmented"] == 8
    assert manifest["stats_contract"]["frames_augmented"] == 8 * 65
    assert "captured_before_generation" not in json.dumps(manifest)

    source_rows = {row["source_id"]: row for row in catalog["rows"]}
    for record in accepted:
        source = source_rows[record["source_id"]]
        assert record["origin"] == source["origin"]
        assert record["base_id"] == source["base_id"]
        assert record["source_group"] == source["source_group"]
        assert record["split"] == source["split"]
        assert record["source_original_sha256"] == source["source_sha256"]
        assert record["source_intent"]["kind"] == "causal_expert_velocity_proxy"
        assert record["source_intent"]["is_captured_instruction"] is False
        assert record["source_intent"]["uses_future_frames"] is False
        assert record["source_intent"]["predictive_supervision"] is False
        with np.load(output / record["source_contact_sidecar"], allow_pickle=False) as contacts:
            assert contacts["source_contacts"].shape == (65, 15)
        with np.load(output / record["terrain_geometry_snapshot"], allow_pickle=False) as geometry:
            assert geometry["heightfield_y_m"].ndim == 2
            assert geometry["min_point_world_xyz_m"].shape == (3,)
            assert geometry["scene_scale"].item() > 0
        with np.load(output / record["shard"], allow_pickle=False) as shard:
            assert shard["global_features"].shape == (65, 366)
            assert shard["dual_features"].shape == (65, 370)
            assert shard["terrain_points"].shape == (65, 2, 9, 9, 3)
            assert shard["mb_foot_contacts"].shape == (65, 4)
            assert shard["feature_fk_roundtrip_error_m"].item() <= 0.002
            frame = {'init_heading_quat': torch.from_numpy(shard['canonical_init_heading_quat'].copy()),
                     'root_pos_init_xz': torch.from_numpy(shard['canonical_root_pos_init_xz'].copy())}
            posed = build_native_representation().inverse(
                torch.from_numpy(shard['global_features'].copy()), False,
                init_heading_info=frame, return_all=True)['posed_joints']
            source_retarget = next(item for item in catalog['rows'] if item['source_id'] == record['source_id'])
            with np.load(source_retarget['retarget_path']) as original:
                torch.testing.assert_close(posed[:, 0], torch.from_numpy(original['root_positions'].copy()),
                                           atol=2e-5, rtol=0)
    official = next(row for row in accepted if row["origin"] == "parc_official")
    generated = next(row for row in accepted if row["origin"] == "parc_generated")
    assert "not locally physically retested" in official["physics_evidence"]
    assert generated["physics_report_snapshot"]
    assert generated["physics_export_manifest_snapshot"]

    representation = build_native_representation(output / "stats")
    assert representation.motion_rep_dim == 366
    validate_terrain_dataset(manifest, accepted)
    for relative, expected in manifest["artifact_sha256"].items():
        assert _hash(output / relative) == expected


def test_rejects_missing_catalog_hash_before_creating_output(tmp_path):
    catalog_path, catalog = _catalog_fixture(tmp_path)
    catalog["rows"][0]["source_sha256"] = "0" * 64
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    output = tmp_path / "should-not-exist"

    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        _BUILDER.build_dataset(catalog_path, output)
    assert not output.exists()


def test_same_terrain_category_is_allowed_with_distinct_instances(tmp_path):
    path, catalog = _catalog_fixture(tmp_path)
    for row in catalog["rows"]:
        row["terrain_family"] = "jumping"
    path.write_text(json.dumps(catalog))
    _, rows, accounting = _BUILDER._load_catalog(path)
    assert accounting["split_counts"]["validation"] == 1
    assert len({row["terrain_content_sha256"] for row in rows}) == 3


def test_identical_terrain_rejected_even_when_instance_labels_differ(tmp_path):
    path, catalog = _catalog_fixture(tmp_path)
    train, heldout = catalog["rows"][0], catalog["rows"][-1]
    with np.load(train["retarget_path"]) as shard:
        hf = shard["hf"].copy()
    with np.load(heldout["retarget_path"]) as shard:
        arrays = {key: shard[key].copy() for key in shard.files}
    arrays["hf"] = hf
    np.savez_compressed(heldout["retarget_path"], **arrays)
    heldout["retarget_sha256"] = _hash(heldout["retarget_path"])
    report_path = Path(heldout["report_path"])
    report = json.loads(report_path.read_text())
    report["shard_sha256"] = heldout["retarget_sha256"]
    report_path.write_text(json.dumps(report))
    heldout["report_sha256"] = _hash(report_path)
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="terrain_content_sha256.*leaks"):
        _BUILDER._load_catalog(path)


def test_declared_terrain_digest_must_bind_actual_retarget(tmp_path):
    path, catalog = _catalog_fixture(tmp_path)
    catalog["rows"][0]["terrain_content_sha256"] = "0" * 64
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="does not bind retarget geometry"):
        _BUILDER._load_catalog(path)


def test_rejects_admitted_failed_prepare_screen(tmp_path):
    catalog_path, catalog = _catalog_fixture(tmp_path)
    row = catalog["rows"][0]
    report_path = Path(row["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["screen_status"] = "fail"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    row["report_sha256"] = _hash(report_path)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="screen pass"):
        _BUILDER.build_dataset(catalog_path, tmp_path / "failed-screen")


def test_requires_both_origins_in_training_and_accounts_exclusions(tmp_path):
    catalog_path, catalog = _catalog_fixture(tmp_path)
    generated = next(row for row in catalog["rows"] if row["origin"] == "parc_generated")
    generated["split"] = "validation"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ParcTrainingSourceValidationError, match="both PARC source origins"):
        _BUILDER.build_dataset(catalog_path, tmp_path / "missing-train-origin")

    generated["split"] = "train"
    excluded = catalog["rows"][-1]
    excluded["disposition"] = "excluded"
    excluded["split"] = None
    excluded["exclusion_reason"] = "held out by catalog policy"
    for key in ("retarget_path", "retarget_sha256", "report_path", "report_sha256"):
        excluded.pop(key, None)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    loaded, _, accounting = _BUILDER._load_catalog(catalog_path)
    assert loaded["catalog_row_count"] == 3
    assert accounting["excluded_source_count"] == 1
    assert excluded["source_id"] in accounting["excluded_source_ids"]


def test_causal_proxy_is_future_invariant_and_uses_zero_initial_frame():
    roots = np.zeros((6, 3), dtype=np.float32)
    roots[:, 0] = np.arange(6, dtype=np.float32) * 0.1
    proxy = _BUILDER.causal_expert_velocity_proxy(roots)
    changed_future = roots.copy()
    changed_future[4:, 0] += np.asarray([8.0, -4.0])
    changed = _BUILDER.causal_expert_velocity_proxy(changed_future)

    np.testing.assert_array_equal(proxy[0], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(proxy[:4], changed[:4])
    assert np.allclose(proxy[1:, 0], 3.0)


def test_excluded_partial_physics_recording_is_audited_not_admitted(tmp_path):
    catalog_path, catalog = _catalog_fixture(tmp_path)
    original = next(row for row in catalog['rows'] if row['origin'] == 'parc_generated')
    excluded = copy.deepcopy(original)
    excluded.update(source_id='partial-physics', upstream_identity='PARC-generation:fixture:partial',
                    disposition='excluded', split=None, exclusion_reason='partial physics recording')
    report = json.loads(Path(original['physics_report_path']).read_text())
    report['physics_recording_completed'] = False
    report_path = tmp_path / 'partial_physics.json'
    report_path.write_text(json.dumps(report))
    excluded.update(physics_report_path=str(report_path), physics_report_sha256=_hash(report_path))
    catalog['rows'].append(excluded)
    catalog_path.write_text(json.dumps(catalog))
    _, validated, accounting = _BUILDER._load_catalog(catalog_path)
    assert accounting['excluded_source_count'] == 1
    row = next(item for item in validated if item['source_id'] == 'partial-physics')
    assert row['_physics_report'] is None
    # A changed disposition cannot turn the same incomplete evidence into training truth.
    excluded.update(disposition='admitted', split='train')
    excluded.pop('exclusion_reason')
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError):
        _BUILDER._load_catalog(catalog_path)


def test_expansion_preserves_checkpoint_statistics_and_rejects_tampering(tmp_path):
    catalog_path,_=_catalog_fixture(tmp_path)
    original=tmp_path/'original'
    _BUILDER.build_dataset(catalog_path,original)
    expanded=tmp_path/'expanded'
    manifest=_BUILDER.build_dataset(catalog_path,expanded,reuse_stats_from=original)
    for name in ('mean.npy','std.npy','contract.json'):
        assert (expanded/'stats'/name).read_bytes()==(original/'stats'/name).read_bytes()
    assert manifest['statistics_source_manifest_sha256']==_hash(original/'native_manifest.json')
    with (original/'stats/mean.npy').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        _BUILDER.build_dataset(catalog_path,tmp_path/'tampered',reuse_stats_from=original)


def test_incremental_conversion_needs_no_old_raw_files_and_preserves_native_shards(tmp_path):
    old_path,old_catalog=_catalog_fixture(tmp_path/'old_source')
    base=tmp_path/'base'
    _BUILDER.build_dataset(old_path,base)
    base_rows=json.loads((base/'accepted.json').read_text())
    new_path,new_catalog=_catalog_fixture(tmp_path/'new_source')
    row=next(row for row in new_catalog['rows'] if row['origin']=='parc_generated')
    row['source_id']='new-generated'
    row['source_group']='new-group'
    row['base_id']='new-base'
    row['upstream_identity']='PARC-generation:new-case'
    source=Path(row['source_path'])
    source.write_bytes(source.read_bytes()+b'new measured recording')
    row['source_sha256']=_hash(source)
    report=Path(row['report_path']); value=json.loads(report.read_text())
    value['source']['sha256']=row['source_sha256']; report.write_text(json.dumps(value))
    row['report_sha256']=_hash(report)
    export=Path(row['export_manifest_path']); value=json.loads(export.read_text())
    value['recordings'][0]['converted_sha256']=row['source_sha256']; export.write_text(json.dumps(value))
    row['export_manifest_sha256']=_hash(export)
    new_catalog['rows']=[row]; new_path.write_text(json.dumps(new_catalog))
    old_path.parent.rename(old_path.parent.with_name('raw-unavailable'))
    output=tmp_path/'expanded'
    manifest=_BUILDER.build_dataset(new_path,output,incremental_base=base)
    rows=json.loads((output/'accepted.json').read_text())
    assert rows[:len(base_rows)]==base_rows and len(rows)==len(base_rows)+1
    for original in base_rows:
        assert (base/original['shard']).read_bytes()==(output/original['shard']).read_bytes()
    assert manifest['incremental_preserved_records']==len(base_rows)
    assert (base/'stats/mean.npy').read_bytes()==(output/'stats/mean.npy').read_bytes()
