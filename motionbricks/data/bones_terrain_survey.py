"""Candidate survey for BONES-SEED terrain fitting, from annotations only.

This module never reads or edits motion. It sorts clips into candidate buckets
for the terrain-fitting stage using the official timeline annotations and, when
available, the official clip metadata table:

- support_height_change: the source may stand on more than one support height
  (stairs, ramps, platforms, curbs, steps onto/off objects).
- hand_or_obstacle_support: hands or the body may carry load on geometry
  (vaulting, climbing, mantling, ladders).
- low_posture: crouch, crawl, kneel, prone and related postures that a ceiling
  or overhang could explain.

Buckets are evidence for later geometric screening, not geometry. A keyword hit
does not prove a support change, and a missing hit does not prove flat ground.
Every row records which text or metadata field produced each hit so the rules
can be audited and revised.

Standard library only; pandas is required just for parquet/pickle metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA_VERSION = "bones-terrain-survey-v2"
SOURCE_FPS = 120.0

#: Columns of the official BONES-SEED metadata table that this survey reads.
METADATA_COLUMNS = (
    "filename", "move_name", "move_duration_frames", "package", "category",
    "is_mirror", "take_actor", "actor_uid", "content_name",
    "content_type_of_movement", "content_body_position", "content_vertical_move",
    "content_horizontal_move", "content_props", "content_uniform_style",
    "content_natural_desc_1", "content_natural_desc_2", "content_natural_desc_3",
    "content_natural_desc_4", "content_technical_description",
    "content_short_description", "content_short_description_2",
)
METADATA_TEXT_COLUMNS = (
    "content_name", "content_type_of_movement", "content_natural_desc_1",
    "content_natural_desc_2", "content_natural_desc_3", "content_natural_desc_4",
    "content_technical_description", "content_short_description",
    "content_short_description_2",
)

#: Candidate vocabulary. Phrases match on word boundaries after lowercasing and
#: replacing "_" / "-" with spaces. Revise by version, never silently.
BUCKET_TERMS = {
    "support_height_change": (
        "stair", "stairs", "staircase", "upstairs", "downstairs", "step up",
        "steps up", "step down", "steps down", "step onto", "steps onto",
        "step off", "steps off", "ramp", "slope", "sloped", "incline", "inclined",
        "hill", "uphill", "downhill", "platform", "ledge", "curb", "kerb",
        "onto a box", "onto the box", "onto box", "off the box", "crate",
        "jump onto", "jumps onto", "jump off", "jumps off", "jump down",
        "jumps down", "ascend", "ascends", "ascending",
        "descend", "descends", "descending", "elevated", "uneven", "obstacle",
        "obstacles",
    ),
    "hand_or_obstacle_support": (
        "vault", "vaults", "vaulting", "climb", "climbs", "climbing", "mantle",
        "mantles", "mantling", "ladder", "scramble", "scrambles", "hurdle",
        "parkour", "hands on the", "hand on the", "pull up onto", "pulls up onto",
        "pull himself", "pull herself", "push off the",
    ),
    "low_posture": (
        "crouch", "crouches", "crouched", "crouching", "crawl", "crawls",
        "crawling", "kneel", "kneels", "kneeling", "on knees", "on one knee",
        "on their knees", "on his knees", "on her knees", "prone", "squat",
        "squats", "squatting", "duck", "ducks", "ducking", "sneak", "sneaks",
        "sneaking", "stealth", "all fours", "hands and knees", "army crawl",
        "commando", "belly", "lying", "lie down", "lies down", "lying down",
        "low stance", "hunched", "bent over", "underneath",
    ),
}

#: Context reported per row, never used to remove a clip in this survey.
CONTEXT_TERMS = {
    "seating": ("sit", "sits", "sitting", "seated", "chair", "bench", "sofa", "couch", "bed"),
    "dance_or_sport": ("dance", "dancing", "fight", "fighting", "punch", "kick",
                       "martial", "acrobat", "somersault", "handstand", "cartwheel"),
    "prop_handling": ("carry", "carries", "carrying", "hold", "holds", "holding",
                      "pick up", "picks up", "lift", "lifts", "push", "pushes",
                      "pull", "pulls", "rifle", "pistol", "gun", "weapon"),
    "fall": ("fall", "falls", "falling", "trip", "trips", "stumble", "stumbles"),
}

#: Official body-position values that indicate a low posture. Unknown values
#: are reported verbatim so new categories become visible.
LOW_BODY_POSITIONS = ("crouch", "crawl", "kneel", "prone", "lying", "squat", "on floor", "ground")


def _phrase_pattern(terms):
    escaped = sorted((re.escape(term) for term in terms), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(escaped) + r")(?![a-z0-9])")


BUCKET_PATTERNS = {name: _phrase_pattern(terms) for name, terms in BUCKET_TERMS.items()}
CONTEXT_PATTERNS = {name: _phrase_pattern(terms) for name, terms in CONTEXT_TERMS.items()}
BUCKETS = tuple(BUCKET_TERMS)


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[_\-/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_terms(pattern, text):
    return sorted(set(pattern.findall(normalize_text(text))))


def recording_group(filename):
    """Same grouping as motionbricks.data.bones_seed.recording_group.

    Duplicated here so the survey runs without Kimodo, PyTorch or SciPy; a test
    checks that both definitions agree when the full stack is installed.
    """
    name = re.sub(r"_M$", "", Path(filename).stem)
    name = re.sub(r"__A\d+$", "", name, flags=re.IGNORECASE)
    return re.sub(r"_\d+$", "", name).lower()


#: BONES file names follow <action>_<direction/phase/angle tokens>_<take>__A<actor>[_M].
#: These tokens describe the take, not the action, and are dropped from the stem.
STEM_DROP_TOKENS = frozenset({"ff", "r", "l", "start", "stop", "loop", "cw", "ccw", "neutral"})
STEM_ANGLES = frozenset({"0", "45", "90", "135", "180", "225", "270", "315", "360"})
DIMENSION = re.compile(r"^\d+(?:\.\d+)?(?:cm|mm|m)$")


def action_stem(filename):
    """Action family name with take, actor, mirror, direction and angle removed.

    Examples: crouch_ff_loop_225_101__A128_M -> crouch;
    jump_over_obstacle_0_75m_R_003__A415 -> jump_over_obstacle_0.75m.
    Decimal dimensions written with "_" are joined first so "0_75m" is kept.
    """
    name = re.sub(r"__A\d+(?:_M)?$", "", Path(filename).stem, flags=re.IGNORECASE).lower()
    name = re.sub(r"(?<![a-z0-9])(\d+)_(\d+)(cm|mm|m)(?![a-z0-9])", r"\1.\2\3", name)
    tokens = [token for token in re.split(r"[_\-]+", name) if token]
    while tokens and tokens[-1].isdigit() and tokens[-1] not in STEM_ANGLES:
        tokens.pop()  # take number
    kept = [token for token in tokens if token not in STEM_DROP_TOKENS and token not in STEM_ANGLES]
    return "_".join(kept) or name


def dimension_tokens(filename):
    """Explicit sizes in the file name, e.g. 50cm or 0.75m: candidate known geometry."""
    return [token for token in action_stem(filename).split("_") if DIMENSION.match(token)]


def actor_of(filename):
    match = re.search(r"__(A\d+)(?:_M)?$", Path(filename).stem, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def is_mirror_name(filename):
    return bool(re.search(r"_M$", Path(filename).stem))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_timelines(path):
    annotations = {}
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            name = entry.get("filename")
            if not name:
                raise ValueError(f"{path}:{number}: timeline entry without filename")
            if name in annotations:
                raise ValueError(f"{path}:{number}: duplicate timeline filename {name}")
            annotations[name] = entry
    return annotations


def _rows_from_object(value):
    """Turn a loaded table-like object into a list of dicts."""
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        rows = list(value.values())
        if rows and all(isinstance(row, dict) for row in rows):
            keyed = []
            for key, row in value.items():
                row = dict(row)
                row.setdefault("filename", key)
                keyed.append(row)
            return keyed
    if isinstance(value, (list, tuple)) and all(isinstance(row, dict) for row in value):
        return [dict(row) for row in value]
    raise ValueError(f"Unsupported metadata object of type {type(value).__name__}")


def read_metadata_rows(path):
    """Read .parquet/.csv/.json/.jsonl/.pkl metadata into a list of dicts.

    Pickle executes code while loading; use it only for files you produced or trust.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if suffix == ".json":
        return _rows_from_object(json.loads(path.read_text(encoding="utf-8")))
    import pandas as pd
    if suffix == ".parquet":
        return _rows_from_object(pd.read_parquet(path))
    if suffix in (".pkl", ".pickle"):
        return _rows_from_object(pd.read_pickle(path))
    raise ValueError(f"Unsupported metadata format: {path.suffix}")


def describe_metadata(rows):
    """Column inventory for an unfamiliar metadata file."""
    columns = Counter()
    examples = {}
    for row in rows:
        for key, value in row.items():
            columns[key] += 1
            if key not in examples and value not in (None, ""):
                examples[key] = str(value)[:120]
    present = [name for name in METADATA_COLUMNS if name in columns]
    return {"rows": len(rows), "columns": dict(sorted(columns.items())),
            "examples": dict(sorted(examples.items())),
            "official_columns_present": present,
            "official_columns_missing": [name for name in METADATA_COLUMNS if name not in columns]}


def index_metadata(rows):
    index = {}
    for row in rows:
        name = row.get("filename") or row.get("move_name")
        if not name:
            raise ValueError("Metadata rows need a 'filename' or 'move_name' column; "
                             "run with --describe-metadata to inspect the file")
        name = Path(str(name)).stem
        if name in index:
            raise ValueError(f"Duplicate metadata filename {name}")
        index[name] = row
    return index


def _flag(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "1.0", "true", "yes"):
        return True
    if text in ("0", "0.0", "false", "no", ""):
        return False
    return None


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def classify_clip(filename, annotation=None, metadata=None):
    """Return one survey row; evidence keeps the field each hit came from."""
    annotation = annotation or {}
    events = annotation.get("events") or []
    sources = {"filename": filename, "overview": annotation.get("overview_description", "")}
    if metadata:
        sources["metadata_text"] = " | ".join(
            str(metadata[column]) for column in METADATA_TEXT_COLUMNS
            if metadata.get(column) not in (None, ""))
    evidence = {bucket: {} for bucket in BUCKETS}
    for bucket, pattern in BUCKET_PATTERNS.items():
        for source, text in sources.items():
            hits = find_terms(pattern, text)
            if hits:
                evidence[bucket][source] = hits
    event_seconds = {bucket: 0.0 for bucket in BUCKETS}
    event_hits = {bucket: 0 for bucket in BUCKETS}
    for event in events:
        start, end = _number(event.get("start_time")), _number(event.get("end_time"))
        if start is None or end is None or end < start:
            continue
        for bucket, pattern in BUCKET_PATTERNS.items():
            hits = find_terms(pattern, event.get("description", ""))
            if hits:
                event_seconds[bucket] += end - start
                event_hits[bucket] += 1
                merged = set(evidence[bucket].get("events", [])) | set(hits)
                evidence[bucket]["events"] = sorted(merged)
    metadata_fields = {}
    if metadata:
        vertical = _flag(metadata.get("content_vertical_move"))
        if vertical:
            evidence["support_height_change"]["metadata:content_vertical_move"] = ["1"]
        position = normalize_text(metadata.get("content_body_position"))
        if position and any(term in position for term in LOW_BODY_POSITIONS):
            evidence["low_posture"]["metadata:content_body_position"] = [position]
        metadata_fields = {key: metadata.get(key) for key in (
            "package", "category", "content_type_of_movement", "content_body_position",
            "content_vertical_move", "content_horizontal_move", "content_props")
            if key in metadata}
    context = {name: hits for name, pattern in CONTEXT_PATTERNS.items()
               if (hits := find_terms(pattern, " ".join(
                   [filename, sources["overview"]] + [e.get("description", "") for e in events])))}
    metadata_mirror = _flag(metadata.get("is_mirror")) if metadata else None
    mirror = metadata_mirror if metadata_mirror is not None else is_mirror_name(filename)
    frames = _number(metadata.get("move_duration_frames")) if metadata else None
    if frames is not None:
        duration, duration_source = frames / SOURCE_FPS, "metadata_frames_at_120fps"
    else:
        ends = [_number(e.get("end_time")) for e in events]
        ends = [value for value in ends if value is not None]
        duration = max(ends) if ends else None
        duration_source = "last_timeline_event_end" if ends else "unknown"
    if "propagated_from_filename" in annotation:
        annotation_source = "propagated" if annotation["propagated_from_filename"] else "human"
    else:
        annotation_source = "unknown" if annotation else "missing"
    return {
        "filename": filename,
        "buckets": [bucket for bucket in BUCKETS if evidence[bucket]],
        "evidence": {bucket: hits for bucket, hits in evidence.items() if hits},
        "event_seconds": {bucket: round(value, 4) for bucket, value in event_seconds.items() if value},
        "event_count": {bucket: value for bucket, value in event_hits.items() if value},
        "context": context,
        "mirror": bool(mirror),
        "mirror_source": "metadata" if metadata_mirror is not None else "filename",
        "actor": (str(metadata.get("take_actor")) if metadata and metadata.get("take_actor")
                  else actor_of(filename)),
        "recording_group": recording_group(filename),
        "stem": action_stem(filename),
        "dimension_tokens": dimension_tokens(filename),
        "duration_seconds": duration,
        "duration_source": duration_source,
        "annotation_source": annotation_source,
        "propagated_from": annotation.get("propagated_from_filename"),
        "overview": annotation.get("overview_description"),
        "metadata": metadata_fields,
    }


def assign_families(rows):
    """Union recording groups linked by annotation propagation, as in data prep."""
    parents = {}

    def find(group):
        parents.setdefault(group, group)
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    for row in rows:
        find(row["recording_group"])
        if row.get("propagated_from"):
            left, right = find(row["recording_group"]), find(recording_group(row["propagated_from"]))
            if left != right:
                parents[max(left, right)] = min(left, right)
    for row in rows:
        row["family"] = find(row["recording_group"])
    return rows


def _seconds(rows):
    values = [row["duration_seconds"] for row in rows if row["duration_seconds"] is not None]
    return {"seconds": round(sum(values), 3), "rows_without_duration": len(rows) - len(values)}


def summarize_bucket(rows, bucket=None):
    originals = [row for row in rows if not row["mirror"]]
    mirrors = [row for row in rows if row["mirror"]]
    terms = Counter()
    for row in rows:
        for source, hits in row["evidence"].get(bucket, {}).items() if bucket else ():
            for hit in hits:
                terms[f"{source}:{hit}"] += 1
    summary = {
        "clips": len(rows), "original_clips": len(originals), "mirror_clips": len(mirrors),
        "original_duration": _seconds(originals),
        "families": len({row["family"] for row in rows}),
        "actors": len({row["actor"] for row in rows if row["actor"]}),
        "annotation_source": dict(Counter(row["annotation_source"] for row in rows)),
        "with_bvh": sum(1 for row in rows if row.get("bvh_present")),
    }
    if bucket:
        summary["original_matched_event_seconds"] = round(
            sum(row["event_seconds"].get(bucket, 0.0) for row in originals), 3)
        summary["evidence_only_from_metadata"] = sum(
            1 for row in rows if all(source.startswith("metadata:")
                                     for source in row["evidence"].get(bucket, {})))
        summary["top_terms"] = dict(terms.most_common(25))
        summary["by_category"] = dict(Counter(
            str(row["metadata"].get("category")) for row in rows if "category" in row["metadata"]
        ).most_common(20))
        summary["by_body_position"] = dict(Counter(
            str(row["metadata"].get("content_body_position")) for row in rows
            if "content_body_position" in row["metadata"]).most_common(20))
    return summary


def stem_table(rows):
    """One row per action stem over all clips, including those without any hit.

    hit_fraction shows whether a keyword hit is systematic for the family or
    comes from a few descriptions; mixed stems deserve review either way.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row["stem"]].append(row)
    table = []
    for stem, members in groups.items():
        originals = [row for row in members if not row["mirror"]] or members
        hits = Counter(bucket for row in originals for bucket in row["buckets"])
        example = sorted(originals, key=lambda row: row["filename"])[0]
        table.append({
            "stem": stem,
            "buckets": "+".join(bucket for bucket in BUCKETS if hits[bucket]) or "none",
            "original_clips": sum(1 for row in members if not row["mirror"]),
            "mirror_clips": sum(1 for row in members if row["mirror"]),
            "original_seconds": round(sum(row["duration_seconds"] or 0.0 for row in members
                                          if not row["mirror"]), 3),
            "hit_fraction": {bucket: round(hits[bucket] / len(originals), 3)
                             for bucket in BUCKETS if hits[bucket]},
            "families": len({row["family"] for row in members}),
            "actors": len({row["actor"] for row in members if row["actor"]}),
            "dimension_tokens": example["dimension_tokens"],
            "example_filename": example["filename"],
            "example_overview": (example["overview"] or "")[:200],
        })
    table.sort(key=lambda item: (item["buckets"] == "none", item["buckets"], -item["original_clips"], item["stem"]))
    return table


def survey(timeline_rows, metadata_rows=None, bvh_stems=None):
    """Classify every clip named by the timelines or metadata; return (rows, report)."""
    metadata_index = index_metadata(metadata_rows) if metadata_rows is not None else {}
    names = sorted(set(timeline_rows) | set(metadata_index))
    rows = [classify_clip(name, timeline_rows.get(name), metadata_index.get(name)) for name in names]
    assign_families(rows)
    if bvh_stems is not None:
        for row in rows:
            row["bvh_present"] = row["filename"] in bvh_stems
    candidates = [row for row in rows if row["buckets"]]
    overlaps = Counter(" + ".join(row["buckets"]) for row in candidates)
    report = {
        "schema_version": SCHEMA_VERSION,
        "clips": {
            "total": len(rows),
            "timeline_only": sum(1 for n in names if n in timeline_rows and n not in metadata_index),
            "metadata_only": sum(1 for n in names if n in metadata_index and n not in timeline_rows),
            "both": sum(1 for n in names if n in timeline_rows and n in metadata_index),
        },
        "all_clips": summarize_bucket(rows),
        "any_candidate": summarize_bucket(candidates),
        "buckets": {bucket: summarize_bucket([row for row in rows if bucket in row["buckets"]], bucket)
                    for bucket in BUCKETS},
        "bucket_combinations": dict(overlaps.most_common()),
        "flat_remainder": summarize_bucket([row for row in rows if not row["buckets"]]),
        "stems": {
            "unique": len({row["stem"] for row in rows}),
            "with_any_hit": len({row["stem"] for row in candidates}),
            "with_dimension_tokens": len({row["stem"] for row in rows if row["dimension_tokens"]}),
        },
        "rules": {"bucket_terms": BUCKET_TERMS, "context_terms": CONTEXT_TERMS,
                  "low_body_positions": LOW_BODY_POSITIONS,
                  "metadata_rules": {"support_height_change": "content_vertical_move == 1",
                                     "low_posture": "content_body_position contains a low_body_positions term"}},
        "protocol": {
            "motion_read": False, "source_changed": False,
            "status": "candidate selection only; geometric screening required before terrain fitting",
            "family": "recording_group plus annotation-propagation union, as in data preparation",
            "duration": "original (non-mirror) clips only; metadata frames at 120 fps, else last timeline event end",
            "event_seconds": "summed durations of timeline events whose own description matched the bucket",
            "flat_remainder": "no keyword or metadata hit; not evidence of flat support",
            "stem": "file name without take, actor, mirror, direction, phase and angle tokens; see stems.tsv",
        },
    }
    if bvh_stems is not None:
        report["bvh"] = {"files_indexed": len(bvh_stems),
                         "named_clips_without_bvh": sum(1 for row in rows if not row["bvh_present"]),
                         "bvh_without_annotation": len(set(bvh_stems) - set(names))}
    return rows, report
