"""Validate provenance and split isolation for PARC-only MotionBricks data.

The input manifest has ``schema='motionbricks.parc_training_sources'``,
``schema_version=1``, an exact ``source_row_count``, and a nonempty
``source_rows`` list. Each row supplies ``source_id``, ``origin`` (exactly
``parc_generated`` or ``parc_official``), ``upstream_identity``,
``upstream_sha256``, ``upstream_manifest_sha256``, ``base_id``,
``source_group``, and ``disposition``. Admitted rows additionally supply a
``split`` of train/validation/test. Excluded rows supply a nonempty
``exclusion_reason`` and must not have a training split. Rows may additionally
include ``terrain_content_sha256`` and ``geometry_component``; these optional
geometry identities are validated and kept within one split when present.

The identity should be a pinned release/revision plus artifact identifier, or
a pinned PARC generation run plus candidate identifier. Hashes must bind the
upstream artifact and its source manifest. This module checks that those
identifiers and hashes are explicit, well-formed, and internally consistent;
it does not fetch or re-hash external artifacts.
"""

from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA = "motionbricks.parc_training_sources"
SCHEMA_VERSION = 1
ALLOWED_ORIGINS = frozenset({"parc_generated", "parc_official"})
SPLITS = ("train", "validation", "test")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ParcTrainingSourceValidationError(ValueError):
    """Raised when source provenance or split isolation is incomplete/unsafe."""


def _nonempty_string(value: Any, field: str, source_id: str | None = None) -> str:
    where = f" for {source_id!r}" if source_id is not None else ""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ParcTrainingSourceValidationError(
            f"{field} must be a nonempty, trimmed string{where}"
        )
    return value


def _sha256(value: Any, field: str, source_id: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ParcTrainingSourceValidationError(
            f"{field} must be a lowercase 64-character SHA-256 hex digest for {source_id!r}"
        )
    return value


def _load_manifest(manifest: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ParcTrainingSourceValidationError(
                f"Cannot read PARC training source manifest {path}: {error}"
            ) from error
    else:
        value = manifest
    if not isinstance(value, Mapping):
        raise ParcTrainingSourceValidationError("Source manifest must be a JSON object")
    return value


def validate_parc_training_sources(
    manifest: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate a versioned PARC source manifest and return a JSON-safe summary.

    A successful return is the validation result; no synthetic pass/status field
    is added. Rows marked excluded are counted separately and never contribute
    to split or training-origin counts.
    """
    payload = _load_manifest(manifest)
    if payload.get("schema") != SCHEMA:
        raise ParcTrainingSourceValidationError(f"Expected schema {SCHEMA!r}")
    version = payload.get("schema_version")
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise ParcTrainingSourceValidationError(
            f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION}"
        )

    rows = payload.get("source_rows")
    if not isinstance(rows, list) or not rows:
        raise ParcTrainingSourceValidationError("source_rows must be a nonempty list")
    declared_count = payload.get("source_row_count")
    if (isinstance(declared_count, bool) or not isinstance(declared_count, int)
            or declared_count != len(rows)):
        raise ParcTrainingSourceValidationError(
            "source_row_count must exactly match every supplied source row"
        )

    seen_ids: set[str] = set()
    identity_bindings: dict[str, tuple[str, str, str]] = {}
    artifact_origins: dict[str, str] = {}
    manifest_origins: dict[str, str] = {}
    split_owners: dict[str, dict[str, str]] = {
        "base_id": {}, "source_group": {},
        "upstream_identity": {}, "upstream_sha256": {},
        "terrain_content_sha256": {}, "geometry_component": {},
    }
    admitted_ids: list[str] = []
    excluded_ids: list[str] = []
    split_counts: Counter[str] = Counter()
    train_origin_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ParcTrainingSourceValidationError(f"source_rows[{index}] must be an object")
        source_id = _nonempty_string(row.get("source_id"), "source_id")
        if source_id in seen_ids:
            raise ParcTrainingSourceValidationError(f"Duplicate source_id {source_id!r}")
        seen_ids.add(source_id)

        origin = row.get("origin")
        if not isinstance(origin, str) or origin not in ALLOWED_ORIGINS:
            raise ParcTrainingSourceValidationError(
                f"Unknown or mixed origin for {source_id!r}: {origin!r}"
            )
        upstream_identity = _nonempty_string(
            row.get("upstream_identity"), "upstream_identity", source_id
        )
        upstream_sha256 = _sha256(row.get("upstream_sha256"), "upstream_sha256", source_id)
        manifest_sha256 = _sha256(
            row.get("upstream_manifest_sha256"), "upstream_manifest_sha256", source_id
        )
        base_id = _nonempty_string(row.get("base_id"), "base_id", source_id)
        source_group = _nonempty_string(row.get("source_group"), "source_group", source_id)
        terrain_content_sha256 = (
            _sha256(row["terrain_content_sha256"], "terrain_content_sha256", source_id)
            if "terrain_content_sha256" in row else None
        )
        geometry_component = (
            _nonempty_string(row["geometry_component"], "geometry_component", source_id)
            if "geometry_component" in row else None
        )

        old_binding = identity_bindings.get(upstream_identity)
        binding = (origin, upstream_sha256, manifest_sha256)
        if old_binding is not None and old_binding != binding:
            raise ParcTrainingSourceValidationError(
                f"upstream_identity {upstream_identity!r} has conflicting origin or hashes"
            )
        identity_bindings[upstream_identity] = binding
        for digest, owner, label in (
            (upstream_sha256, artifact_origins, "upstream artifact hash"),
            (manifest_sha256, manifest_origins, "upstream manifest hash"),
        ):
            previous_origin = owner.get(digest)
            if previous_origin is not None and previous_origin != origin:
                raise ParcTrainingSourceValidationError(
                    f"{label} {digest} is assigned mixed origins"
                )
            owner[digest] = origin

        disposition = row.get("disposition")
        if not isinstance(disposition, str) or disposition not in {"admitted", "excluded"}:
            raise ParcTrainingSourceValidationError(
                f"{source_id!r} needs disposition 'admitted' or 'excluded'"
            )
        if disposition == "excluded":
            _nonempty_string(row.get("exclusion_reason"), "exclusion_reason", source_id)
            if row.get("split") is not None:
                raise ParcTrainingSourceValidationError(
                    f"Excluded source {source_id!r} cannot be assigned to a split"
                )
            excluded_ids.append(source_id)
            continue

        if row.get("exclusion_reason") is not None:
            raise ParcTrainingSourceValidationError(
                f"Admitted source {source_id!r} cannot carry an exclusion_reason"
            )
        split = row.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            raise ParcTrainingSourceValidationError(
                f"Admitted source {source_id!r} needs split train, validation, or test"
            )
        if geometry_component is not None and geometry_component != source_group:
            raise ParcTrainingSourceValidationError(
                f"geometry_component must equal source_group for {source_id!r}"
            )

        split_identities = [("base_id", base_id)]
        if geometry_component is not None:
            split_identities.append(("geometry_component", geometry_component))
        split_identities.extend((
            ("source_group", source_group),
            ("upstream_identity", upstream_identity),
            ("upstream_sha256", upstream_sha256),
        ))
        if terrain_content_sha256 is not None:
            split_identities.append(("terrain_content_sha256", terrain_content_sha256))
        for key, identity in split_identities:
            owner = split_owners[key]
            previous_split = owner.get(identity)
            if previous_split is not None and previous_split != split:
                raise ParcTrainingSourceValidationError(
                    f"{key} {identity!r} leaks across splits"
                )
            owner[identity] = split

        admitted_ids.append(source_id)
        split_counts[split] += 1
        if split == "train":
            train_origin_counts[origin] += 1

    missing_train_origins = ALLOWED_ORIGINS - train_origin_counts.keys()
    if missing_train_origins:
        raise ParcTrainingSourceValidationError(
            "Training split must contain both PARC source origins; missing "
            + ", ".join(sorted(missing_train_origins))
        )

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_row_count": len(rows),
        "admitted_source_count": len(admitted_ids),
        "excluded_source_count": len(excluded_ids),
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "train_origin_counts": {
            origin: train_origin_counts[origin] for origin in sorted(ALLOWED_ORIGINS)
        },
        "admitted_source_ids": sorted(admitted_ids),
        "excluded_source_ids": sorted(excluded_ids),
    }


def verify_parc_dataset_sources(dataset_dir, manifest, records):
    """Bind the whitelist to the exact immutable records consumed by training."""
    root = Path(dataset_dir).resolve()
    policy = manifest.get("parc_sources", {})
    snapshot = policy.get("snapshot")
    if not isinstance(snapshot, str) or not snapshot:
        raise ParcTrainingSourceValidationError("Dataset needs a PARC source snapshot")
    path = (root / snapshot).resolve()
    if root not in path.parents or not path.is_file():
        raise ParcTrainingSourceValidationError("PARC source snapshot must be inside the dataset")
    expected = manifest.get("artifact_sha256", {}).get(snapshot)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != actual or policy.get("sha256") != actual:
        raise ParcTrainingSourceValidationError("PARC source snapshot SHA256 mismatch")
    sources = _load_manifest(path)
    summary = validate_parc_training_sources(sources)
    admitted = {row["source_id"]: row for row in sources["source_rows"]
                if row["disposition"] == "admitted"}
    if not isinstance(records, list) or len(records) != len(admitted):
        raise ParcTrainingSourceValidationError("Accepted records differ from admitted PARC sources")
    seen = set()
    for row in records:
        source_id = row.get("source_id")
        if source_id not in admitted or source_id in seen:
            raise ParcTrainingSourceValidationError("Unexpected/duplicate PARC training record")
        seen.add(source_id)
        source = admitted[source_id]
        for key in ("origin", "source_group", "split", "base_id"):
            if row.get(key) != source[key]:
                raise ParcTrainingSourceValidationError(f"PARC record mismatch: {key}")
        for key in ("geometry_component", "terrain_content_sha256"):
            if key in source and row.get(key) != source[key]:
                raise ParcTrainingSourceValidationError(f"PARC record mismatch: {key}")
        if row.get("source_original_sha256") != source["upstream_sha256"]:
            raise ParcTrainingSourceValidationError("PARC record source hash mismatch")
    return summary


def require_parc_training_origins(records):
    """Filtering short clips must not silently remove one of the two sources."""
    counts = Counter()
    for row in records:
        if row.get("origin") not in ALLOWED_ORIGINS or row.get("split") != "train":
            raise ParcTrainingSourceValidationError("Training loader contains a non-PARC/non-train row")
        counts[row["origin"]] += 1
    if ALLOWED_ORIGINS - counts.keys():
        raise ParcTrainingSourceValidationError("Training loader must contain both PARC origins")
    return {key: counts[key] for key in sorted(ALLOWED_ORIGINS)}
