"""Build a PARC-only native MotionBricks dataset from an explicit teacher catalog.

Catalog paths are relative to the catalog JSON unless absolute. The catalog has
``schema='parc.only_teacher_catalog.v1'`` and a ``rows`` list. Admitted rows
bind source, upstream manifest, prepare_parc_soma retarget shard and report;
excluded rows may lack unavailable downstream artifacts. Generated rows also
bind a complete physics report and official export manifest. Official provenance
is bound to release inspection metadata, revision, archive hash, member path and
member artifact hash.

Run with ``--catalog catalog.json --output parc-only-v1``. Existing outputs are
never overwritten. This is a dataset converter, not training or quality approval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))
_KIMODO_ROOT = _PACKAGE_ROOT / "out/vendor/kimodo"
if str(_KIMODO_ROOT) not in sys.path:
    sys.path.insert(0, str(_KIMODO_ROOT))

from motionbricks.data.native_mb_data import (  # noqa: E402
    CONTRACT_VERSION,
    FPS,
    REPRESENTATION_NAME,
    build_native_representation,
    build_train_stats,
)
from motionbricks.data.native_mb_terrain_protocol import (  # noqa: E402
    TERRAIN_PROTOCOL,
)
from motionbricks.data.parc_training_sources import (  # noqa: E402
    validate_parc_training_sources,
)


_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_parc_native_dataset_helpers", Path(__file__).with_name("build_parc_native_dataset.py")
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError("Could not load shared PARC native dataset helpers")
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPERS
_HELPER_SPEC.loader.exec_module(_HELPERS)


CATALOG_SCHEMA = "parc.only_teacher_catalog.v1"
OUTPUT_SCHEMA = "parc-only-native-dataset.v1"
MIN_TRAINABLE_FRAMES = _HELPERS.MIN_TRAINABLE_FRAMES
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_ROW_FIELDS = {
    "source_id", "origin", "source_group", "base_id", "terrain_family",
    "terrain_seed", "upstream_identity", "upstream_manifest_path",
    "upstream_manifest_sha256", "source_path", "source_sha256", "disposition",
}
_GENERATED_FIELDS = {
    "physics_report_path", "physics_report_sha256", "export_manifest_path",
    "export_manifest_sha256", "physics_candidate_key",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _nonempty(value: Any, field: str, source_id: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a nonempty trimmed string for {source_id}")
    return value


def _resolve(base: Path, value: Any, field: str, source_id: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty path for {source_id}")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _verify_file(path: Path, expected_hash: Any, field: str, source_id: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {field} for {source_id}: {path}")
    if not isinstance(expected_hash, str) or _HASH_RE.fullmatch(expected_hash) is None:
        raise ValueError(f"{field} SHA256 must be lowercase 64-character hex for {source_id}")
    actual = _sha256(path)
    if actual != expected_hash:
        raise ValueError(f"{field} SHA256 mismatch for {source_id}")
    return actual


def _validate_official_manifest(path: Path, row: dict[str, Any],
                                source_path: Path) -> dict[str, Any]:
    payload = _json(path)
    source_id = row["source_id"]
    revision = _nonempty(row.get("official_revision"), "official_revision", source_id)
    member_name = _nonempty(row.get("official_member"), "official_member", source_id)
    archive_hash = row.get("official_archive_sha256")
    if not isinstance(archive_hash, str) or _HASH_RE.fullmatch(archive_hash) is None:
        raise ValueError(f"official_archive_sha256 must be lowercase SHA256 for {source_id}")
    if not isinstance(payload, dict) or payload.get("revision") != revision:
        raise ValueError(f"Official source manifest revision mismatch for {source_id}")
    archive = payload.get("archive")
    lfs = archive.get("lfs") if isinstance(archive, dict) else None
    if not isinstance(lfs, dict) or lfs.get("oid") != archive_hash:
        raise ValueError(f"Official source manifest archive hash mismatch for {source_id}")
    identity_fragment = f"@{revision}:{member_name}"
    if identity_fragment not in row["upstream_identity"]:
        raise ValueError(f"upstream_identity does not bind official revision/member for {source_id}")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Official source manifest needs a samples list for {source_id}")
    matches = [sample for sample in samples if isinstance(sample, dict) and
               sample.get("member") == member_name]
    if len(matches) != 1:
        raise ValueError(f"Official source manifest must bind exactly one member for {source_id}")
    sample = matches[0]
    if sample.get("sha256") != row["source_sha256"]:
        raise ValueError(f"Official member artifact hash does not bind source {source_id}")
    recorded_path = _resolve(path.parent, sample.get("local_file"),
                              "official sample.local_file", source_id)
    if recorded_path != source_path:
        raise ValueError(f"Official member artifact path does not bind source {source_id}")
    return payload


def _validate_prepare_report(row: dict[str, Any], paths: dict[str, Path], report: Any) -> None:
    source_id = row["source_id"]
    if not isinstance(report, dict):
        raise ValueError(f"prepare_parc_soma report must be an object for {source_id}")
    report_source = report.get("source")
    if (not isinstance(report_source, dict) or
            report_source.get("sha256") != row["source_sha256"]):
        raise ValueError(f"prepare_parc_soma report does not bind the source hash for {source_id}")
    source_path = _resolve(paths["report"].parent, report_source.get("local_file"),
                           "report.source.local_file", source_id)
    if source_path != paths["source"]:
        raise ValueError(f"prepare_parc_soma report source path mismatch for {source_id}")
    if report.get("shard_sha256") != row["retarget_sha256"]:
        raise ValueError(f"prepare_parc_soma report does not bind the retarget shard for {source_id}")
    shard_path = _resolve(paths["report"].parent, report.get("shard"), "report.shard", source_id)
    if shard_path != paths["retarget"]:
        raise ValueError(f"prepare_parc_soma report shard path mismatch for {source_id}")
    if row["disposition"] == "admitted":
        if report.get("execution_status") != "processed" or report.get("screen_status") != "pass":
            raise ValueError(
                f"Admitted source {source_id} lacks a real processed prepare_parc_soma screen pass"
            )
        if report.get("training_approved") is True:
            raise ValueError(f"prepare_parc_soma report must not imply quality approval for {source_id}")


def _catalog_to_training_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_rows = []
    for row in rows:
        item = {
            "source_id": row["source_id"],
            "origin": row["origin"],
            "upstream_identity": row["upstream_identity"],
            "upstream_sha256": row["source_sha256"],
            "upstream_manifest_sha256": row["upstream_manifest_sha256"],
            "base_id": str(row["base_id"]),
            "source_group": str(row["source_group"]),
            "disposition": row["disposition"],
        }
        if row["disposition"] == "admitted":
            item["split"] = row["split"]
            for field in ("terrain_content_sha256", "geometry_component"):
                if row.get(field) is not None:
                    item[field] = row[field]
        else:
            item["exclusion_reason"] = row["exclusion_reason"]
        source_rows.append(item)
    return {
        "schema": "motionbricks.parc_training_sources",
        "schema_version": 1,
        "source_row_count": len(source_rows),
        "source_rows": source_rows,
    }


def _load_catalog(catalog_path: Path, *, prior_source_rows=None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    catalog_path = Path(catalog_path).resolve()
    catalog = _json(catalog_path)
    if not isinstance(catalog, dict) or catalog.get("schema") != CATALOG_SCHEMA:
        raise ValueError(f"Catalog schema must be {CATALOG_SCHEMA!r}")
    rows = catalog.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Catalog needs a nonempty rows list")
    if catalog.get("quality_approved") is True:
        raise ValueError("Catalog must not claim general physical or skill quality approval")

    identities: set[str] = set()
    validated: list[dict[str, Any]] = []
    # A family is a category, not an instance: held-out stairs may share the
    # stairs category, but must not share the same source or actual heightfield.
    terrain_splits: dict[str, dict[str, str]] = {"terrain_seed": {}, "terrain_content_sha256": {}}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("Catalog source_rows entries must be objects")
        missing = _ROW_FIELDS - raw.keys()
        if missing:
            raise ValueError(f"Catalog row is missing fields: {sorted(missing)}")
        row = dict(raw)
        source_id = _nonempty(row["source_id"], "source_id", "catalog row")
        if source_id in identities:
            raise ValueError(f"Duplicate source_id {source_id!r}")
        identities.add(source_id)
        if row["origin"] not in {"parc_official", "parc_generated"}:
            raise ValueError(f"Unknown source origin for {source_id}: {row['origin']!r}")
        for field in ("source_group", "base_id", "upstream_identity"):
            _nonempty(row[field], field, source_id)
        for field in ("terrain_family", "terrain_seed"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
                raise ValueError(f"{field} must be a nonempty string or integer for {source_id}")
            row[field] = str(value)
        disposition = row["disposition"]
        if disposition == "admitted":
            if row.get("split") not in {"train", "validation", "test"}:
                raise ValueError(f"Admitted source {source_id} needs train/validation/test split")
            if row.get("exclusion_reason") is not None:
                raise ValueError(f"Admitted source {source_id} cannot have an exclusion_reason")
            for field, value in (("terrain_seed", row["terrain_seed"]),):
                previous = terrain_splits[field].get(value)
                if previous is not None and previous != row["split"]:
                    raise ValueError(f"{field} {value!r} leaks across splits")
                terrain_splits[field][value] = row["split"]
        elif disposition == "excluded":
            if row.get("split") is not None:
                raise ValueError(f"Excluded source {source_id} must have split=null")
            _nonempty(row.get("exclusion_reason"), "exclusion_reason", source_id)
        else:
            raise ValueError(f"Unsupported disposition for {source_id}: {disposition!r}")

        required_hash_fields = ("upstream_manifest_sha256", "source_sha256")
        for field in required_hash_fields:
            if not isinstance(row[field], str) or _HASH_RE.fullmatch(row[field]) is None:
                raise ValueError(f"{field} must be lowercase SHA256 for {source_id}")
        if row["origin"] == "parc_generated":
            missing_generated = _GENERATED_FIELDS - row.keys()
            if missing_generated:
                raise ValueError(f"Generated source {source_id} lacks fields: {sorted(missing_generated)}")
            for field in ("physics_report_sha256", "export_manifest_sha256"):
                if not isinstance(row[field], str) or _HASH_RE.fullmatch(row[field]) is None:
                    raise ValueError(f"{field} must be lowercase SHA256 for {source_id}")
            _nonempty(row["physics_candidate_key"], "physics_candidate_key", source_id)

        path_fields = {
            "upstream_manifest": ("upstream_manifest_path", "upstream_manifest_sha256"),
            "source": ("source_path", "source_sha256"),
        }
        paths = {}
        for name, (path_field, hash_field) in path_fields.items():
            paths[name] = _resolve(catalog_path.parent, row[path_field], path_field, source_id)
            _verify_file(paths[name], row[hash_field], name, source_id)

        for name, path_field, hash_field in (
            ("retarget", "retarget_path", "retarget_sha256"),
            ("report", "report_path", "report_sha256"),
            ("intent", "intent_path", "intent_sha256"),
        ):
            raw_path, raw_hash = row.get(path_field), row.get(hash_field)
            if raw_path is None and raw_hash is None:
                if disposition == "admitted" and name in {"retarget", "report"}:
                    raise ValueError(f"Admitted source {source_id} lacks {path_field}/{hash_field}")
                continue
            if raw_path is None or raw_hash is None:
                raise ValueError(f"{path_field} and {hash_field} must both be present for {source_id}")
            paths[name] = _resolve(catalog_path.parent, raw_path, path_field, source_id)
            _verify_file(paths[name], raw_hash, name, source_id)

        if row["origin"] == "parc_generated":
            for name, path_field, hash_field in (
                ("physics_report", "physics_report_path", "physics_report_sha256"),
                ("export_manifest", "export_manifest_path", "export_manifest_sha256"),
            ):
                paths[name] = _resolve(catalog_path.parent, row[path_field], path_field, source_id)
                _verify_file(paths[name], row[hash_field], name, source_id)

        upstream = _json(paths["upstream_manifest"])
        if not isinstance(upstream, dict):
            raise ValueError(f"Upstream source manifest must be a JSON object for {source_id}")
        if row["origin"] == "parc_official":
            _validate_official_manifest(paths["upstream_manifest"], row, paths["source"])

        report = None
        if "report" in paths:
            report = _json(paths["report"])
            _validate_prepare_report(row, paths, report)
        if disposition == "admitted":
            with np.load(paths["retarget"], allow_pickle=False) as shard:
                hf = np.asarray(shard["hf"], dtype="<f4")
                dx = np.asarray(shard["dx"], dtype="<f8")
            if (hf.ndim != 2 or dx.size != 1 or not np.isfinite(hf).all()
                    or not np.isfinite(dx).all() or float(dx.item()) <= 0):
                raise ValueError(f"Invalid retarget terrain geometry for {source_id}")
            geometry_hash = hashlib.sha256()
            for array in (np.asarray(hf.shape, dtype="<i8"), hf, dx):
                geometry_hash.update(array.tobytes(order="C"))
            identity = geometry_hash.hexdigest()
            if row.get("terrain_content_sha256", identity) != identity:
                raise ValueError(f"terrain_content_sha256 does not bind retarget geometry for {source_id}")
            component = row.get("geometry_component")
            if component is not None and (component != row["source_group"] or component != row["terrain_seed"]):
                raise ValueError(f"geometry_component must bind source_group and terrain_seed for {source_id}")
            previous = terrain_splits["terrain_content_sha256"].get(identity)
            if previous is not None and previous != row["split"]:
                raise ValueError(f"terrain_content_sha256 {identity!r} leaks across splits")
            terrain_splits["terrain_content_sha256"][identity] = row["split"]
            row["terrain_content_sha256"] = identity
        physics_report = export_manifest = None
        if row["origin"] == "parc_generated" and disposition == "admitted":
            physics_report, export_manifest = _HELPERS._validate_physics_retest(
                paths["physics_report"], row["physics_report_sha256"],
                paths["export_manifest"], row["export_manifest_sha256"],
                row["physics_candidate_key"], paths["source"], row["source_sha256"],
            )
        row["_paths"] = paths
        row["_report"] = report
        row["_physics_report"] = physics_report
        row["_export_manifest"] = export_manifest
        validated.append(row)

    training_manifest = _catalog_to_training_manifest(validated)
    if prior_source_rows:
        training_manifest['source_rows'] = prior_source_rows + training_manifest['source_rows']
        training_manifest['source_row_count'] = len(training_manifest['source_rows'])
    accounting = validate_parc_training_sources(training_manifest)
    return {
        **catalog,
        "catalog_path": str(catalog_path),
        "catalog_sha256": _sha256(catalog_path),
        "catalog_row_count": len(rows),
    }, validated, accounting


def causal_expert_velocity_proxy(root_positions: np.ndarray) -> np.ndarray:
    """Return causal PARC-plane root velocity in scaled m/s; frame zero is zero."""
    roots = np.asarray(root_positions, dtype=np.float64)
    if roots.ndim != 2 or roots.shape[1] != 3 or len(roots) < 1 or not np.isfinite(roots).all():
        raise ValueError("root_positions must be finite [T,3]")
    commands = np.zeros((len(roots), 2), dtype=np.float32)
    if len(roots) > 1:
        native_delta = np.diff(roots, axis=0)
        source_delta = native_delta @ _HELPERS.Y_TO_Z.T
        commands[1:] = (source_delta[:, :2] * FPS).astype(np.float32)
    return commands


def _slug(source_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_id).strip("._-")[:72] or "clip"
    return f"{stem}-{hashlib.sha256(source_id.encode()).hexdigest()[:10]}"


def _proxy_clip(source_id: str, row: dict[str, Any], proxy_dir: Path) -> dict[str, Any]:
    with np.load(row["_paths"]["retarget"], allow_pickle=False) as shard:
        if "root_positions" not in shard.files:
            raise ValueError(f"Retarget shard lacks root_positions for {source_id}")
        root = np.asarray(shard["root_positions"], dtype=np.float32).copy()
    intent_path = row["_paths"].get("intent")
    if intent_path is None:
        commands = causal_expert_velocity_proxy(root)
        frame_indices = np.arange(len(root), dtype=np.int64)
        intent_path = proxy_dir / f"{_slug(source_id)}_causal_velocity.npz"
        np.savez_compressed(intent_path, command_world_xy=commands,
                            source_frame_indices=frame_indices, fps=np.asarray(FPS, dtype=np.float32))
        metadata = {
            "kind": "causal_expert_velocity_proxy",
            "source": "current_and_previous_root_positions_only",
            "frame_zero": "zero_vector",
            "uses_future_frames": False,
            "is_captured_instruction": False,
            "purpose": "selects bounded current-frame terrain observation direction only",
            "predictive_supervision": False,
        }
    else:
        metadata = {
            "kind": "recorded_intent_sidecar",
            "source": "hash-verified catalog intent_path",
            "capture_timing": "not asserted by this builder",
            "purpose": "selects bounded current-frame terrain observation direction only",
            "predictive_supervision": False,
        }
    helper_record = {
        "paths": {"retarget": row["_paths"]["retarget"], "intent": intent_path},
        "report": row["_report"],
    }
    clip = _HELPERS._load_clip(source_id, helper_record, row["split"])
    clip["source_intent_metadata"] = metadata
    return clip


def _copy_snapshot(source: Path, destination: Path, relative: Path,
                   artifact_hashes: dict[str, str]) -> tuple[str, str]:
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    digest = _sha256(target)
    artifact_hashes[relative.as_posix()] = digest
    return relative.as_posix(), digest


def _validate_output_separation(output: Path, rows: list[dict[str, Any]]) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    for row in rows:
        for path in row["_paths"].values():
            if output == path or output in path.parents or path in output.parents:
                raise ValueError(f"Output must be separate from all input files: {path}")


def build_dataset(catalog_path: Path, output: Path, *, reuse_stats_from=None, incremental_base=None) -> dict[str, Any]:
    catalog_path = Path(catalog_path).resolve()
    base_manifest = base_sources = None
    base_records = []
    if incremental_base is not None:
        from scripts.train_native_mb import verify_dataset
        from motionbricks.data.parc_training_sources import verify_parc_dataset_sources
        incremental_base = Path(incremental_base).resolve()
        base_manifest = _json(incremental_base/'native_manifest.json')
        verify_dataset(incremental_base, base_manifest)
        base_records = _json(incremental_base/'accepted.json')
        verify_parc_dataset_sources(incremental_base,base_manifest,base_records)
        base_sources = _json(incremental_base/'parc_sources.json')
        if reuse_stats_from is not None and Path(reuse_stats_from).resolve()!=incremental_base:
            raise ValueError('Incremental conversion must preserve the base statistics')
        reuse_stats_from = incremental_base
    catalog, rows, accounting = _load_catalog(catalog_path,
        prior_source_rows=base_sources['source_rows'] if base_sources else None)
    if base_sources:
        previous_hashes={row['upstream_sha256'] for row in base_sources['source_rows'] if row['disposition']=='admitted'}
        if any(row['source_sha256'] in previous_hashes for row in rows if row['disposition']=='admitted'):
            raise ValueError('Incremental conversion duplicates an existing source motion')
    output = Path(output).resolve()
    _validate_output_separation(output, rows)
    if incremental_base is not None and (output==incremental_base or incremental_base in output.parents
                                         or output in incremental_base.parents):
        raise ValueError('Incremental output must be separate from the base dataset')
    admitted = [row for row in rows if row["disposition"] == "admitted"]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        artifact_hashes: dict[str, str] = {}
        acceptance_items: list[dict[str, Any]] = []
        if base_manifest:
            replacement={'accepted.json','parc_sources.json',base_manifest['terrain_conditioning']['acceptance']['report']}
            for relative,digest in base_manifest['artifact_sha256'].items():
                if relative in replacement or relative.startswith('stats/'):
                    continue
                target=staging/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(incremental_base/relative,target)
                artifact_hashes[relative]=digest
            acceptance_items=_json(incremental_base/base_manifest['terrain_conditioning']['acceptance']['report'])['items']
        clips: dict[str, dict[str, Any]] = {}
        with tempfile.TemporaryDirectory(prefix="parc-only-causal-proxy-") as proxy_temp:
            proxy_dir = Path(proxy_temp)
            for row in admitted:
                clips[row["source_id"]] = _proxy_clip(row["source_id"], row, proxy_dir)

        for name in ('shards','source_contacts','source_intent','terrain_geometry','prepare_reports','upstream_manifests','physics_reports'):
            (staging/name).mkdir(exist_ok=True)

        representation = build_native_representation()
        accepted: list[dict[str, Any]] = list(base_records)
        for row in admitted:
            source_id = row["source_id"]
            clip = clips[source_id]
            (global_features, dual_features, saved_local, saved_root, roundtrip_error,
             mb_foot_contacts, canonical_frame) = _HELPERS._native_encode(
                clip["local_rot_mats"], clip["root_positions"], representation, clip, return_frame=True
            )
            terrain_points, terrain_valid = _HELPERS._build_terrain_observations(clip)
            stem = _slug(source_id)
            shard_rel = Path("shards") / f"{stem}_native30.npz"
            if (staging/shard_rel).exists():
                raise ValueError('Incremental conversion would overwrite an existing shard')
            contacts_rel = Path("source_contacts") / f"{stem}_parc15.npz"
            intent_rel = Path("source_intent") / f"{stem}_causal_velocity_proxy.npz"
            geometry_rel = Path("terrain_geometry") / f"{stem}_paired_heightfield.npz"
            prepare_rel = Path("prepare_reports") / f"{stem}_prepare_parc_soma.json"
            upstream_rel = Path("upstream_manifests") / f"{stem}_upstream.json"

            np.savez_compressed(
                staging / shard_rel,
                global_features=global_features,
                dual_features=dual_features,
                local_rot_mats=saved_local,
                root_positions=saved_root,
                terrain_points=terrain_points,
                terrain_valid=terrain_valid,
                mb_foot_contacts=mb_foot_contacts,
                fps=np.asarray(FPS, dtype=np.float32),
                feature_fk_roundtrip_error_m=np.asarray(roundtrip_error, dtype=np.float32),
                canonical_init_heading_quat=canonical_frame['init_heading_quat'],
                canonical_root_pos_init_xz=canonical_frame['root_pos_init_xz'],
            )
            np.savez_compressed(
                staging / contacts_rel,
                source_contacts=clip["source_contacts"].astype(np.float32),
                source_contact_names=clip["source_contact_names"].astype(str),
                source_frame_indices=clip["source_frame_indices"].astype(np.int64),
                fps=np.asarray(FPS, dtype=np.float32),
            )
            np.savez_compressed(
                staging / intent_rel,
                command_world_xy=clip["command_world_xy"].astype(np.float32),
                source_frame_indices=clip["source_frame_indices"].astype(np.int64),
                fps=np.asarray(FPS, dtype=np.float32),
                command_coordinates=np.asarray(
                    "scaled PARC z-up world XY causal velocity proxy"
                    if clip["source_intent_metadata"]["kind"] == "causal_expert_velocity_proxy"
                    else "recorded PARC z-up world XY command"
                ),
            )
            np.savez_compressed(
                staging / geometry_rel,
                heightfield_y_m=clip["heightfield_y_m"].astype(np.float32),
                min_point_world_xyz_m=clip["min_point_world_xyz_m"].astype(np.float32),
                dx_m=np.asarray(clip["dx_m"], dtype=np.float32),
                scene_scale=np.asarray(clip["scene_scale"], dtype=np.float32),
                canonical_init_heading_quat=canonical_frame['init_heading_quat'],
                canonical_root_pos_init_xz=canonical_frame['root_pos_init_xz'],
            )

            prepare_snapshot, prepare_hash = _copy_snapshot(
                row["_paths"]["report"], staging, prepare_rel, artifact_hashes)
            upstream_snapshot, upstream_hash = _copy_snapshot(
                row["_paths"]["upstream_manifest"], staging, upstream_rel, artifact_hashes)
            shard_hash = _sha256(staging / shard_rel)
            contact_hash = _sha256(staging / contacts_rel)
            intent_hash = _sha256(staging / intent_rel)
            geometry_hash = _sha256(staging / geometry_rel)
            for relative, digest in ((shard_rel, shard_hash), (contacts_rel, contact_hash),
                                     (intent_rel, intent_hash), (geometry_rel, geometry_hash)):
                artifact_hashes[relative.as_posix()] = digest

            physics_snapshot = export_snapshot = None
            physics_hash = export_hash = None
            if row["origin"] == "parc_generated":
                physics_snapshot, physics_hash = _copy_snapshot(
                    row["_paths"]["physics_report"], staging,
                    Path("physics_reports") / f"{stem}_full_retest.json", artifact_hashes)
                export_snapshot, export_hash = _copy_snapshot(
                    row["_paths"]["export_manifest"], staging,
                    Path("physics_reports") / f"{stem}_export_manifest.json", artifact_hashes)

            record = {
                "source_id": source_id,
                "origin": row["origin"],
                "base_id": row["base_id"],
                "source_group": row["source_group"],
                "split": row["split"],
                "source_original_sha256": row["source_sha256"],
                "source_original_path": str(row["_paths"]["source"]),
                "source_shard": str(row["_paths"]["retarget"]),
                "source_shard_sha256": row["retarget_sha256"],
                "filename": row["_paths"]["source"].name,
                "recording_group": row["source_group"],
                "terrain_family": row["terrain_family"],
                "terrain_seed": row["terrain_seed"],
                "terrain_content_sha256": row.get("terrain_content_sha256"),
                "geometry_component": row.get("geometry_component"),
                "source_manifest_identity": row["upstream_identity"],
                "source_manifest_snapshot": upstream_snapshot,
                "source_manifest_snapshot_sha256": upstream_hash,
                "prepare_report_snapshot": prepare_snapshot,
                "prepare_report_sha256": prepare_hash,
                "source_contact_sidecar": contacts_rel.as_posix(),
                "source_contact_sidecar_sha256": contact_hash,
                "source_contact_channels": 15,
                "source_contact_names": clip["source_contact_names"].tolist(),
                "source_contact_semantics": "PARC source labels retained verbatim; provenance only, not certified physical truth",
                "source_intent_sidecar": intent_rel.as_posix(),
                "source_intent_sidecar_sha256": intent_hash,
                "source_intent": clip["source_intent_metadata"],
                "terrain_geometry_snapshot": geometry_rel.as_posix(),
                "terrain_geometry_sha256": geometry_hash,
                "terrain_geometry_coordinates": "same scaled SOMA Y-up world coordinates as retarget root_positions",
                "shard": shard_rel.as_posix(),
                "converted_shard_sha256": shard_hash,
                "source_shard_fps": FPS,
                "source_original_fps": FPS,
                "frames_30hz": clip["frames"],
                "duration_seconds": (clip["frames"] - 1) / FPS,
                "scene_scale": clip["scene_scale"],
                "terrain_observation_valid_fraction": float(terrain_valid.mean()),
                "feature_fk_roundtrip_error_m": roundtrip_error,
            }
            if row["origin"] == "parc_generated":
                record.update({
                    "physics_evidence": "full measured-origin physics retest and official export validated",
                    "physics_report_snapshot": physics_snapshot,
                    "physics_report_sha256": physics_hash,
                    "physics_export_manifest_snapshot": export_snapshot,
                    "physics_export_manifest_sha256": export_hash,
                    "physics_candidate_key": row["physics_candidate_key"],
                })
            else:
                record["physics_evidence"] = "official author-released paired teacher; not locally physically retested"
            accepted.append(record)

            report = row["_report"]
            acceptance_items.append({
                "source_id": source_id,
                "origin": row["origin"],
                "execution_status": report["execution_status"],
                "screen_status": report["screen_status"],
                "prepare_report_snapshot": prepare_snapshot,
                "prepare_report_sha256": prepare_hash,
                "retarget_sha256": row["retarget_sha256"],
                "source_sha256": row["source_sha256"],
                "paired_geometry_evidence": "prepare_parc_soma report with exact source/shard hash and path binding",
                "physics_evidence": record["physics_evidence"],
                "reported_not_certified": report.get("not_certified", []),
                "reported_limitations": report.get("limitations", []),
                "reported_source_contact_semantics": report.get("source_contact_semantics"),
            })

        accepted_path = staging / "accepted.json"
        _write_json(accepted_path, accepted)
        artifact_hashes["accepted.json"] = _sha256(accepted_path)

        source_rows = (base_sources['source_rows'] if base_sources else []) + _catalog_to_training_manifest(rows)['source_rows']
        catalog_rows = (base_sources['catalog_rows'] if base_sources else []) + catalog['rows']
        source_snapshot = {
            **_catalog_to_training_manifest(rows), 'source_rows':source_rows,'source_row_count':len(source_rows),
            "catalog_schema": catalog["schema"],
            "catalog_path": catalog["catalog_path"],
            "catalog_sha256": catalog["catalog_sha256"],
            "catalog_row_count": len(catalog_rows),
            "catalog_rows": catalog_rows,
            "training_source_validation": accounting,
            "source_accounting": {
                "catalog_rows": accounting["source_row_count"],
                "admitted_rows": accounting["admitted_source_count"],
                "excluded_rows": accounting["excluded_source_count"],
                "admitted_source_ids": accounting["admitted_source_ids"],
                "excluded_source_ids": accounting["excluded_source_ids"],
                "split_counts": accounting["split_counts"],
                "train_origin_counts": accounting["train_origin_counts"],
            },
        }
        source_snapshot_path = staging / "parc_sources.json"
        _write_json(source_snapshot_path, source_snapshot)
        source_snapshot_hash = _sha256(source_snapshot_path)
        artifact_hashes["parc_sources.json"] = source_snapshot_hash

        if reuse_stats_from is None:
            stats_contract = build_train_stats(staging, accepted, source_snapshot_hash)
        else:
            source = Path(reuse_stats_from).resolve()
            source_manifest = _json(source/'native_manifest.json')
            source_contract = source_manifest['contract']
            if (source_contract['version'] != CONTRACT_VERSION
                    or source_contract['representation'] != REPRESENTATION_NAME
                    or source_contract['target_fps'] != FPS):
                raise ValueError('Statistics source uses a different representation')
            stats_contract = _json(source/'stats/contract.json')
            if stats_contract.get('split') != 'train':
                raise ValueError('Statistics source must be explicitly train-only')
            for name in ('mean.npy', 'std.npy', 'contract.json'):
                relative = 'stats/'+name
                _verify_file(source/relative, source_manifest['artifact_sha256'][relative], relative, 'reuse-stats')
            build_native_representation(source/'stats')
            shutil.copytree(source/'stats', staging/'stats')
        for name in ("mean.npy", "std.npy", "contract.json"):
            artifact_hashes[f"stats/{name}"] = _sha256(staging / "stats" / name)

        acceptance_report = {
            "schema": "parc-only-paired-geometry-acceptance.v1",
            "status": "validated_for_native_dataset_conversion",
            "stage_specific_geometric_teacher_approved": True,
            "quality_approved": False,
            "physics_approved": False,
            "skill_approved": False,
            "training_usable": True,
            "source_accounting": source_snapshot["source_accounting"],
            "pairing_basis": "each admitted retarget is bound to its actual processed, screen-pass prepare_parc_soma report and exact source/shard hashes and paths",
            "items": acceptance_items,
            "limitations_preserved_from_prepare_reports": True,
            "official_physics_scope": "official sources are author-released paired teachers and are not represented as locally physically retested",
        }
        acceptance_path = staging / "terrain_conditioning" / "acceptance_report.json"
        _write_json(acceptance_path, acceptance_report)
        artifact_hashes["terrain_conditioning/acceptance_report.json"] = _sha256(acceptance_path)

        code_hashes = _HELPERS._code_hashes()
        code_hashes["parc_only_dataset_builder"] = _sha256(Path(__file__))
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "stage_specific_geometric_teacher_approved": True,
            "quality_approved": False,
            "physics_approved": False,
            "skill_approved": False,
            "training_usable": True,
            "approval_scope": "stage-specific paired geometric teacher only; not physics or skill approval",
            "contract": {
                "version": CONTRACT_VERSION,
                "representation": REPRESENTATION_NAME,
                "source_catalog_sha256": catalog["catalog_sha256"],
                "source_manifest_sha256": source_snapshot_hash,
                "source_shard_fps": FPS,
                "target_fps": FPS,
                "resampling": "none; full untrimmed 30 Hz prepare_parc_soma frames are encoded in place",
                "features": "official MotionBricks SOMA30 dual_root_global_joints encoding",
                "fk_validation": "official SOMASkeleton30 FK over every retarget frame plus full MotionBricks feature inverse/FK roundtrip",
                "statistics": "native MB train-only 65-frame windows with four-way yaw augmentation",
                "terrain_geometry_snapshot": "paired retarget heightfield, origin, dx, and scene scale retained in scaled SOMA world coordinates",
                "source_contact_channels": "all 15 PARC channels preserved verbatim in sidecars; official MB four foot contacts remain in model features",
                "source_intent": "causal root-velocity proxy selects terrain observation direction only; it adds no predictive supervision",
                "code_sha256": code_hashes,
            },
            "parc_sources": {"snapshot": "parc_sources.json", "sha256": source_snapshot_hash},
            "artifact_sha256": artifact_hashes,
            "terrain_conditioning": {
                **TERRAIN_PROTOCOL,
                "acceptance": {
                    "retarget_status": "passed",
                    "motion_terrain_pairing_status": "passed",
                    "report": "terrain_conditioning/acceptance_report.json",
                },
                "intent_provenance": {
                    "kind": "causal_expert_velocity_proxy",
                    "captured_instruction": False,
                    "input_frames": "current and immediately previous root positions only",
                    "frame_zero": "zero vector",
                    "uses_future_frames": False,
                    "training_target": False,
                },
            },
            "source_rows": accounting["source_row_count"],
            "converted_rows": len(accepted),
            "excluded_rows": accounting["excluded_source_count"],
            "split_counts": {split: sum(item["split"] == split for item in accepted)
                             for split in ("train", "validation", "test")},
            "train_origin_counts": accounting["train_origin_counts"],
            "stats_contract": stats_contract,
            "limitations": [
                "Training readiness does not certify physical validity, skill quality, safety, or task success.",
                "Official PARC rows are author-released paired teachers; no local physics retest is claimed for them.",
                "Inferred causal velocity is not a captured WASD or other generation-time instruction.",
                "Terrain observations use current-frame bounded heightfield queries and cannot represent overhangs, vertical faces, ceilings, or handholds.",
                "PARC source contacts are preserved for provenance and are not native MB contact-loss targets.",
            ],
        }
        if reuse_stats_from is not None:
            manifest['contract']['statistics'] = 'unchanged verified source train-only scales for compatible warmstart'
            manifest['statistics_source_manifest_sha256'] = _sha256(Path(reuse_stats_from)/'native_manifest.json')
        if incremental_base is not None:
            manifest['incremental_base_manifest_sha256'] = _sha256(incremental_base/'native_manifest.json')
            manifest['incremental_preserved_records'] = len(base_records)
        _write_json(staging / "native_manifest.json", manifest)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _json(output / "native_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True,
                        help="parc.only_teacher_catalog.v1 source catalog")
    parser.add_argument("--output", type=Path, required=True,
                        help="New versioned output directory; existing paths are never overwritten")
    parser.add_argument('--reuse-stats-from', type=Path)
    parser.add_argument('--incremental-base', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(build_dataset(args.catalog, args.output, reuse_stats_from=args.reuse_stats_from,
                                   incremental_base=args.incremental_base), indent=2))


if __name__ == "__main__":
    main()
