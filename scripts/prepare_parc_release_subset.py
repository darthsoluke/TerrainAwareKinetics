"""Select and safely extract a balanced subset of the pinned official PARC release.

This is archive staging only. It never unpickles a member and does not certify
contacts, dynamics, or control semantics.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REVISION = "e8f119532d2f03955031ce629d08243403443e58"
OFFICIAL_ARCHIVE_SHA256 = "d82f4605d25b005d0a365b754c6af6ec218ea3e97b292f82a7281ff19854b82a"
OFFICIAL_ARCHIVE_SIZE = 3907916685
DEFAULT_SELECTION_SEED = "parc-release-e8f119532d2f03955031ce629d08243403443e58-v1"
DEFAULT_AUDIT_DIR = ROOT / "out" / "parc_release_audit_20261008"

# These are the ten directories listed in out/vendor/PARC/data/configs/
# parc_0_setup_iter_config.yaml. The separate legacy "initial/" directory is
# deliberately absent.
OFFICIAL_FAMILIES = (
    "initial_aug",
    "rerun_april272025_iter_1_boxes",
    "rerun_april272025_iter_2_boxes",
    "rerun_april272025_iter_3_siggraph_terrain",
    "rerun_april272025_iter_4_siggraph_terrain",
    "rerun_april272025_iter_5_siggraph_terrain",
    "dec2024_iter_1_boxes",
    "dec2024_iter_2_boxes",
    "dec2024_iter_3_siggraph_terrain",
    "dec2024_iter_4_siggraph_terrain",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_archive_stream(stream, expected_sha256, expected_size):
    if len(expected_sha256) != 64 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Expected archive SHA256 must be 64 lowercase hex characters")
    size = os.fstat(stream.fileno()).st_size
    if size != expected_size:
        raise ValueError(f"Archive size mismatch: expected {expected_size}, got {size}")
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Archive SHA256 mismatch: expected {expected_sha256}, got {actual}")
    stream.seek(0)
    return {"sha256": actual, "size": size}


def verify_archive(path, *, expected_sha256=OFFICIAL_ARCHIVE_SHA256,
                   expected_size=OFFICIAL_ARCHIVE_SIZE):
    """Verify the complete archive before a caller opens or extracts its ZIP."""
    with Path(path).open("rb") as stream:
        return _verify_archive_stream(stream, expected_sha256, expected_size)


def validate_member_name(name):
    """Reject non-portable and traversal ZIP names, even though outputs are hashed."""
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise ValueError(f"Unsafe archive member name: {name!r}")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"Absolute archive member name: {name!r}")
    raw_parts = name.rstrip("/").split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"Traversal or empty path component in archive member: {name!r}")
    if ":" in raw_parts[0]:
        raise ValueError(f"Drive-like archive member name: {name!r}")
    return PurePosixPath(name)


def _validate_index_rows(rows):
    if not isinstance(rows, list):
        raise ValueError("Archive index must be a JSON list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("Archive index rows must contain a string name")
        name = row["name"]
        validate_member_name(name)
        if name in result:
            raise ValueError(f"Duplicate archive index member: {name}")
        for field in ("size", "compressed_size", "crc32"):
            if field in row and (not isinstance(row[field], int) or row[field] < 0):
                raise ValueError(f"Invalid {field} for archive member {name}")
        result[name] = row
    return result


def validate_archive_index(archive, index_rows):
    """Require the ZIP central directory to match the locally audited index."""
    expected = _validate_index_rows(index_rows)
    actual = {}
    for info in archive.infolist():
        validate_member_name(info.filename)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"Symbolic-link ZIP member is not accepted: {info.filename}")
        if info.filename in actual:
            raise ValueError(f"Duplicate ZIP member: {info.filename}")
        actual[info.filename] = info
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ValueError(f"Archive index member mismatch; missing={missing[:3]}, extra={extra[:3]}")
    for name, info in actual.items():
        row = expected[name]
        observed = (info.file_size, info.compress_size, info.CRC)
        indexed = (row.get("size"), row.get("compressed_size"), row.get("crc32"))
        if observed != indexed:
            raise ValueError(f"Archive index metadata mismatch for {name}")
    return actual


def _strip_variant_tokens(stem):
    tokens = stem.split("_")
    kept = []
    removed = []
    augmentation_tokens = []
    status = {"mirrored": False, "wip": False, "before_opt": False}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        lowered = token.casefold()
        if lowered == "before" and index + 1 < len(tokens) and tokens[index + 1].casefold() == "opt":
            status["before_opt"] = True
            removed.append({"tokens": tokens[index:index + 2], "rule": "status:before_opt"})
            index += 2
            continue
        if lowered == "flipped":
            status["mirrored"] = True
            removed.append({"tokens": [token], "rule": "mirror:flipped"})
        elif lowered == "wip":
            status["wip"] = True
            removed.append({"tokens": [token], "rule": "status:wip"})
        elif re.fullmatch(r"aug\d+", lowered):
            augmentation_tokens.append(token)
            removed.append({"tokens": [token], "rule": "augmentation:augN"})
        elif lowered in {"dm", "opt"}:
            removed.append({"tokens": [token], "rule": f"release_suffix:{lowered}"})
        else:
            kept.append(token)
        index += 1
    return kept, removed, augmentation_tokens, status


def derive_member_identity(name, families=OFFICIAL_FAMILIES):
    """Return the source grouping and evidence supported by the official filename."""
    path = validate_member_name(name)
    parts = path.parts
    if len(parts) < 3 or parts[0] != "dec_release" or parts[1] not in families:
        return None
    if not name.casefold().endswith(".pkl"):
        return None

    family = parts[1]
    original_stem = path.stem
    tokens, removed, augmentation_tokens, status = _strip_variant_tokens(original_stem)
    normalized_stem = "_".join(tokens).casefold()
    if not normalized_stem:
        raise ValueError(f"Filename has no source identity after variant normalization: {name}")

    child = parts[2] if len(parts) > 3 else "unclassified"
    if family == "initial_aug":
        terrain_family = child.casefold()
        base_id = f"official:{family}:base_motion:{terrain_family}:{normalized_stem}"
        source_group = base_id
        derivation = {
            "rule": "initial_aug filename base motion; remove recognized mirror, augN, dm/opt, and status tokens",
            "family_directory": family,
            "terrain_class_from_path": child,
            "original_stem": original_stem,
            "normalized_base_stem": normalized_stem,
            "removed_tokens": removed,
            "augmentation_tokens": augmentation_tokens,
        }
    else:
        terrain_family = "siggraph_terrain" if family.endswith("_siggraph_terrain") else "boxes"
        if len(tokens) < 3 or not tokens[-1].isdigit() or not tokens[-2].isdigit():
            raise ValueError(f"Iteration member does not match <terrainId>_<candidate> naming: {name}")
        terrain_id = "_".join(tokens[:-1]).casefold()
        candidate = tokens[-1]
        base_id = f"official:{family}:terrain:{terrain_id}:candidate:{candidate}"
        source_group = f"official:{family}:terrain:{terrain_id}"
        derivation = {
            "rule": "iteration filename trailing numeric terrainId_candidate; source_group drops candidate only",
            "family_directory": family,
            "original_stem": original_stem,
            "normalized_stem": normalized_stem,
            "terrain_id": terrain_id,
            "candidate_id": candidate,
            "removed_tokens": removed,
            "augmentation_tokens": augmentation_tokens,
        }
    return {
        "name": name,
        "family": family,
        "terrain_family": terrain_family,
        "source_group": source_group,
        "base_id": base_id,
        "stratum": child.casefold(),
        "base_derivation": derivation,
        "variant_status": status,
    }


def _rank(seed, *parts):
    material = "\0".join((seed, *[str(part) for part in parts])).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _ordered_groups(groups, family, seed):
    by_stratum = defaultdict(list)
    for group in groups.values():
        by_stratum[group["stratum"]].append(group)
    for stratum, values in by_stratum.items():
        values.sort(key=lambda item: (_rank(seed, "group", family, item["source_group"]), item["source_group"]))

    strata = sorted(by_stratum, key=lambda value: (_rank(seed, "stratum", family, value), value))
    ordered = []
    seen = set()

    # Keep the official initial stairs source set in the subset. Its released
    # names are marked wip/before_opt; those labels are evidence, not exclusions.
    if family == "initial_aug" and "stairs" in by_stratum:
        stairs = by_stratum["stairs"]
        ordered.extend(stairs)
        seen.update(item["source_group"] for item in stairs)

    # One group from each available path stratum before taking a second from any.
    coverage_strata = list(strata)
    if family == "initial_aug" and "stairs" in coverage_strata:
        coverage_strata.remove("stairs")
        coverage_strata.insert(0, "stairs")
    for stratum in coverage_strata:
        values = by_stratum[stratum]
        for item in values:
            if item["source_group"] not in seen:
                ordered.append(item)
                seen.add(item["source_group"])
                break

    while True:
        added = False
        for stratum in strata:
            for item in by_stratum[stratum]:
                if item["source_group"] not in seen:
                    ordered.append(item)
                    seen.add(item["source_group"])
                    added = True
                    break
        if not added:
            break
    return ordered


def select_members(index_rows, *, groups_per_family=32, max_members_per_family=256,
                   seed=DEFAULT_SELECTION_SEED, families=OFFICIAL_FAMILIES):
    """Choose source groups evenly by family, then one stable variant per base id."""
    if groups_per_family < 1 or max_members_per_family < 1:
        raise ValueError("Per-family group and member limits must be positive")
    if not seed:
        raise ValueError("Selection seed must not be empty")
    if len(set(families)) != len(families):
        raise ValueError("Official family names must be unique")
    indexed = _validate_index_rows(index_rows)
    pickle_rows = {name: row for name, row in indexed.items() if name.casefold().endswith(".pkl")}
    grouped = {family: {} for family in families}
    member_identities = {}

    for name, row in pickle_rows.items():
        identity = derive_member_identity(name, families)
        if identity is None:
            continue
        member_identities[name] = identity
        family_groups = grouped[identity["family"]]
        group = family_groups.setdefault(identity["source_group"], {
            "source_group": identity["source_group"],
            "terrain_family": identity["terrain_family"],
            "strata": set(),
            "bases": defaultdict(list),
        })
        if group["terrain_family"] != identity["terrain_family"]:
            raise ValueError(f"Inconsistent terrain_family for source group {identity['source_group']}")
        group["strata"].add(identity["stratum"])
        group["bases"][identity["base_id"]].append((identity, row))

    selected = []
    selected_names = set()
    selected_groups = set()
    selected_bases = set()
    family_counts = {}

    for family in families:
        groups = grouped[family]
        # A terrain ID can span storage buckets; the lexical stratum choice keeps
        # ranking invariant to ZIP central-directory order.
        for group in groups.values():
            group["stratum"] = min(group["strata"])
            group["representatives"] = {}
            for base_id, candidates in group["bases"].items():
                candidates.sort(key=lambda pair: (
                    pair[0]["variant_status"]["mirrored"],
                    _rank(seed, "variant", family, base_id, pair[0]["name"]),
                    pair[0]["name"],
                ))
                group["representatives"][base_id] = candidates[0]

        ordered = _ordered_groups(groups, family, seed)
        member_limit = max_members_per_family
        chosen_group_count = 0
        family_selected = []
        for group in ordered:
            if chosen_group_count >= groups_per_family:
                break
            bases = sorted(group["representatives"])
            cost = len(bases)
            if len(family_selected) + cost > member_limit:
                continue
            group_rank = _rank(seed, "group", family, group["source_group"])
            for base_id in bases:
                identity, row = group["representatives"][base_id]
                selected_record = {
                    **identity,
                    "size": row.get("size"),
                    "compressed_size": row.get("compressed_size"),
                    "crc32": row.get("crc32"),
                    "selection_hash": _rank(seed, "member", family, identity["name"]),
                    "group_selection_hash": group_rank,
                }
                family_selected.append(selected_record)
                selected.append(selected_record)
                selected_names.add(identity["name"])
                selected_bases.add((family, base_id))
            chosen_group_count += 1
            selected_groups.add((family, group["source_group"]))

        family_counts[family] = {
            "archive_pickle_member_count": sum(1 for name in pickle_rows if name in member_identities and member_identities[name]["family"] == family),
            "source_group_count": len(groups),
            "base_id_count": sum(len(group["bases"]) for group in groups.values()),
            "selected_source_group_count": chosen_group_count,
            "selected_base_id_count": len(family_selected),
            "selected_member_count": len(family_selected),
            "excluded_member_count": 0,
        }

    excluded_by_member = {}
    for name in pickle_rows:
        if name in selected_names:
            continue
        identity = member_identities.get(name)
        if identity is None:
            parts = PurePosixPath(name).parts
            reason = "legacy_flat_initial" if len(parts) > 1 and parts[:2] == ("dec_release", "initial") else "outside_official_training_families"
        elif (identity["family"], identity["source_group"]) not in selected_groups:
            reason = "source_group_not_selected"
        elif (identity["family"], identity["base_id"]) not in selected_bases:
            reason = "member_cap_group_not_admitted"
        else:
            reason = "normalized_variant_not_selected"
        excluded_by_member[name] = reason

    for family, counts in family_counts.items():
        reasons = Counter(
            reason for name, reason in excluded_by_member.items()
            if name in member_identities and member_identities[name]["family"] == family
        )
        counts["excluded_member_count"] = sum(reasons.values())
        counts["excluded_by_reason"] = dict(sorted(reasons.items()))

    excluded_by_reason = Counter(excluded_by_member.values())
    selection = {
        "seed": seed,
        "hash_algorithm": "sha256",
        "groups_per_family_limit": groups_per_family,
        "max_members_per_family": max_members_per_family,
        "source_group_count": len(selected_groups),
        "selected_member_count": len(selected),
        "archive_pickle_member_count": len(pickle_rows),
        "excluded_member_count": len(excluded_by_member),
        "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
        "families": family_counts,
    }
    selected.sort(key=lambda item: (families.index(item["family"]), item["source_group"], item["base_id"]))
    return {"selected": selected, "selection": selection, "excluded_by_member": excluded_by_member}


def _extract_records(archive, selected, sample_directory, logical_sample_directory):
    sample_directory = Path(sample_directory)
    logical_sample_directory = Path(logical_sample_directory)
    sample_directory.mkdir(parents=True, exist_ok=False)
    extracted = []
    for record in selected:
        name = record["name"]
        validate_member_name(name)
        temp_path = None
        digest = hashlib.sha256()
        byte_count = 0
        try:
            fd, temp_name = tempfile.mkstemp(prefix=".partial-", dir=sample_directory)
            temp_path = Path(temp_name)
            with os.fdopen(fd, "wb") as destination, archive.open(name, "r") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(block)
                    digest.update(block)
                    byte_count += len(block)
                destination.flush()
                os.fsync(destination.fileno())
            if record.get("size") is not None and byte_count != record["size"]:
                raise ValueError(f"Extracted size mismatch for {name}: {byte_count} != {record['size']}")
            sha256 = digest.hexdigest()
            target = sample_directory / f"{sha256}.pkl"
            if target.exists():
                if target.stat().st_size != byte_count or sha256_file(target) != sha256:
                    raise FileExistsError(f"Content-addressed output already contains different data: {target}")
                temp_path.unlink()
            else:
                try:
                    with temp_path.open("rb") as source, target.open("xb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                        destination.flush()
                        os.fsync(destination.fileno())
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
                temp_path.unlink()
            temp_path = None
            extracted.append({
                **record,
                "sha256": sha256,
                "size": byte_count,
                "local_file": str((logical_sample_directory / target.name).resolve()),
            })
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    return extracted


def extract_selected(archive, selected, output_directory):
    """Extract selected members to hashed basenames into a new directory only."""
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=False)
    return _extract_records(archive, selected, output_directory / "samples",
                           output_directory / "samples")


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _validate_release_audit(audit):
    archive = audit.get("archive", {})
    lfs = archive.get("lfs", {})
    if audit.get("revision") != OFFICIAL_REVISION:
        raise ValueError("Release audit revision does not match the pinned official revision")
    if lfs.get("oid") != OFFICIAL_ARCHIVE_SHA256 or lfs.get("size") != OFFICIAL_ARCHIVE_SIZE:
        raise ValueError("Release audit LFS identity does not match the pinned official archive")
    if archive.get("size") != OFFICIAL_ARCHIVE_SIZE:
        raise ValueError("Release audit archive size does not match the pinned official archive")
    return archive


def prepare_subset(archive_path, archive_index_path, release_audit_path, output_directory, *,
                   groups_per_family=32, max_members_per_family=256,
                   seed=DEFAULT_SELECTION_SEED):
    """Verify, select, extract, and write catalog-compatible audit artifacts."""
    archive_path = Path(archive_path).resolve()
    archive_index_path = Path(archive_index_path).resolve()
    release_audit_path = Path(release_audit_path).resolve()
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_directory}")

    audit = _load_json(release_audit_path)
    archive_metadata = _validate_release_audit(audit)
    index_raw = archive_index_path.read_bytes()
    index_rows = json.loads(index_raw.decode("utf-8"))
    _validate_index_rows(index_rows)

    # Hash and size-check the complete file before parsing the ZIP or extracting.
    with archive_path.open("rb") as stream:
        archive_identity = _verify_archive_stream(stream, OFFICIAL_ARCHIVE_SHA256, OFFICIAL_ARCHIVE_SIZE)
        with zipfile.ZipFile(stream, "r") as archive:
            validate_archive_index(archive, index_rows)
            selection_result = select_members(
                index_rows,
                groups_per_family=groups_per_family,
                max_members_per_family=max_members_per_family,
                seed=seed,
            )
            for family, counts in selection_result["selection"]["families"].items():
                if counts["selected_source_group_count"] == 0:
                    raise ValueError(f"No complete source group selected for official family {family}")

            output_directory.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.partial-",
                                            dir=output_directory.parent))
            try:
                samples = _extract_records(
                    archive,
                    selection_result["selected"],
                    staging / "samples",
                    output_directory / "samples",
                )
                for sample in samples:
                    sample["member"] = sample.pop("name")
                    sample.pop("selection_hash", None)
                    sample.pop("group_selection_hash", None)
                    sample.pop("compressed_size", None)
                    sample.pop("crc32", None)

                excluded_by_reason = selection_result["selection"]["excluded_by_reason"]
                manifest = {
                    "schema": "parc.official_release_subset.v1",
                    "purpose": "Deterministic official paired motion-terrain archive-member subset for later PARC-only terrain training",
                    "archive": {
                        "revision": audit["revision"],
                        "path": archive_metadata.get("path", "dec_release.zip"),
                        "lfs_sha256": archive_identity["sha256"],
                        "size": archive_identity["size"],
                        "archive_index_sha256": hashlib.sha256(index_raw).hexdigest(),
                        "full_archive_sha256_verified_before_extraction": True,
                    },
                    "source_evidence": {
                        "official_family_config": "out/vendor/PARC/data/configs/parc_0_setup_iter_config.yaml: input_dataset_folder_paths",
                        "official_archive_index": str(archive_index_path),
                        "legacy_flat_initial_selected": False,
                        "filename_grouping_rules": [
                            "initial_aug: remove flipped, augN, dm/opt and status tokens; group all variants by path class plus base motion",
                            "iteration families: remove mirror/augmentation/release suffix tokens; group by family plus numeric terrain ID and retain candidate ID in base_id",
                        ],
                        "status_marker_policy": "wip and before_opt are recorded but not blanket-excluded; official setup config includes the full initial_aug folder",
                    },
                    "selection": selection_result["selection"],
                    "selected_member_count": len(samples),
                    "excluded_member_count": selection_result["selection"]["excluded_member_count"],
                    "selected_source_group_count": selection_result["selection"]["source_group_count"],
                    "excluded_member_counts_by_reason": excluded_by_reason,
                    "selected_members": [
                        {key: sample[key] for key in (
                            "member", "sha256", "size", "local_file", "family",
                            "source_group", "base_id", "terrain_family", "base_derivation",
                            "variant_status",
                        )}
                        for sample in samples
                    ],
                    "validation_scope": {
                        "pickle_members_unpickled": False,
                        "contact_dynamics_certified": False,
                        "wasd_labels_created": False,
                    },
                }
                # Keep the per-member audit schema consumed by prepare_parc_only_catalog.py.
                inspection = {
                    "revision": audit["revision"],
                    "archive": archive_metadata,
                    "download_mode": "verified local ZIP; no network access",
                    "entries": len(index_rows),
                    "selected_member_count": len(samples),
                    "selection_manifest": "manifest.json",
                    "samples": samples,
                }
                _write_json(staging / "manifest.json", manifest)
                _write_json(staging / "inspection.json", inspection)
                if output_directory.exists():
                    raise FileExistsError(f"Refusing to overwrite existing output: {output_directory}")
                os.rename(staging, output_directory)
            except Exception:
                if staging.exists():
                    shutil.rmtree(staging)
                raise

    return {
        "output": str(output_directory),
        "selected_member_count": len(samples),
        "selected_source_group_count": selection_result["selection"]["source_group_count"],
        "excluded_member_count": selection_result["selection"]["excluded_member_count"],
        "inspection": str(output_directory / "inspection.json"),
        "manifest": str(output_directory / "manifest.json"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Local complete dec_release.zip")
    parser.add_argument("--archive-index", type=Path,
                        default=DEFAULT_AUDIT_DIR / "archive_index.json")
    parser.add_argument("--release-audit", type=Path,
                        default=DEFAULT_AUDIT_DIR / "inspection.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-family", type=int, default=32)
    parser.add_argument("--max-members-per-family", type=int, default=256)
    parser.add_argument("--seed", default=DEFAULT_SELECTION_SEED)
    args = parser.parse_args(argv)
    result = prepare_subset(
        args.archive,
        args.archive_index,
        args.release_audit,
        args.output,
        groups_per_family=args.groups_per_family,
        max_members_per_family=args.max_members_per_family,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
