"""Terrain-candidate survey: annotation-only, auditable, mirror- and family-aware."""

import csv
import json

import pytest

from motionbricks.data.bones_terrain_survey import (
    classify_clip, describe_metadata, find_terms, BUCKET_PATTERNS, read_metadata_rows,
    recording_group, survey,
)
from scripts.survey_bones_terrain_candidates import main


def timeline(name, overview, events=(), propagated=False, with_field=True):
    entry = {"filename": name, "overview_description": overview,
             "events": [{"start_time": s, "end_time": e, "description": d} for s, e, d in events]}
    if with_field:
        entry["propagated_from_filename"] = propagated or None
    return entry


TIMELINES = {
    "stairs_climbing_up_002__A300": timeline(
        "stairs_climbing_up_002__A300", "A person walks up the stairs.",
        [(0.0, 1.0, "A person stands idle."), (1.0, 4.0, "A person ascends the stairs.")]),
    "stairs_climbing_up_002__A300_M": timeline(
        "stairs_climbing_up_002__A300_M", "A person walks up the stairs.",
        [(0.0, 4.0, "A person ascends the stairs.")], propagated="stairs_climbing_up_002__A300"),
    "crawl_forward_001__A010": timeline(
        "crawl_forward_001__A010", "A person moves forward.",
        [(0.0, 2.0, "A person kneels down."), (2.0, 6.5, "A person crawls on hands and knees.")]),
    "walk_forward_001__A006": timeline(
        "walk_forward_001__A006", "A person walks forward.", [(0.0, 3.0, "A person walks forward.")]),
    "duckling_dance_001__A007": timeline(
        "duckling_dance_001__A007", "A person dances like a duckling.", [(0.0, 2.0, "dancing")]),
}


def test_word_boundaries_and_separators():
    assert find_terms(BUCKET_PATTERNS["low_posture"], "duckling dance") == []
    assert find_terms(BUCKET_PATTERNS["low_posture"], "crawl_forward") == ["crawl"]
    assert find_terms(BUCKET_PATTERNS["support_height_change"], "Walks UP-STAIRS") == ["stairs"]
    assert find_terms(BUCKET_PATTERNS["support_height_change"], "stairwell") == []
    assert find_terms(BUCKET_PATTERNS["support_height_change"], "goes upstairs") == ["upstairs"]
    assert find_terms(BUCKET_PATTERNS["low_posture"], "hands and knees") == ["hands and knees"]


def test_recording_group_matches_data_preparation():
    bones_seed = pytest.importorskip("motionbricks.data.bones_seed")
    for name in ("stairs_climbing_up_002__A300_M", "walk_forward_001__a006", "idle_loop__A1"):
        assert recording_group(name) == bones_seed.recording_group(name)


def test_event_seconds_count_only_matching_events():
    row = classify_clip("stairs_climbing_up_002__A300", TIMELINES["stairs_climbing_up_002__A300"])
    assert row["buckets"] == ["support_height_change", "hand_or_obstacle_support"]
    assert row["event_seconds"]["support_height_change"] == pytest.approx(3.0)
    assert row["duration_seconds"] == pytest.approx(4.0)
    assert row["duration_source"] == "last_timeline_event_end"
    assert row["annotation_source"] == "human" and not row["mirror"] and row["actor"] == "A300"
    crawl = classify_clip("crawl_forward_001__A010", TIMELINES["crawl_forward_001__A010"])
    assert crawl["buckets"] == ["low_posture"]
    assert crawl["event_seconds"]["low_posture"] == pytest.approx(6.5)
    assert set(crawl["evidence"]["low_posture"]) == {"filename", "events"}


def test_survey_separates_mirrors_and_joins_propagated_families():
    rows, report = survey(TIMELINES)
    by_name = {row["filename"]: row for row in rows}
    mirror = by_name["stairs_climbing_up_002__A300_M"]
    assert mirror["mirror"] and mirror["annotation_source"] == "propagated"
    assert mirror["family"] == by_name["stairs_climbing_up_002__A300"]["family"]
    support = report["buckets"]["support_height_change"]
    assert (support["clips"], support["original_clips"], support["mirror_clips"]) == (2, 1, 1)
    assert support["original_duration"]["seconds"] == pytest.approx(4.0)
    assert support["original_matched_event_seconds"] == pytest.approx(3.0)
    assert support["families"] == 1
    assert report["flat_remainder"]["clips"] == 2  # walk and duckling dance
    assert by_name["duckling_dance_001__A007"]["context"] == {"dance_or_sport": ["dance", "dancing"]}
    assert report["protocol"]["motion_read"] is False


def test_metadata_flags_override_name_and_add_evidence(tmp_path):
    path = tmp_path / "meta.csv"
    fields = ["filename", "move_duration_frames", "is_mirror", "category",
              "content_body_position", "content_vertical_move", "take_actor"]
    rows = [
        {"filename": "walk_forward_001__A006", "move_duration_frames": "600", "is_mirror": "False",
         "category": "Environments", "content_body_position": "standing",
         "content_vertical_move": "1", "take_actor": "A006"},
        {"filename": "sneak_loop_004__A020", "move_duration_frames": "240", "is_mirror": "True",
         "category": "Basic Locomotion Styles", "content_body_position": "crouching",
         "content_vertical_move": "0", "take_actor": "A020"},
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = read_metadata_rows(path)
    assert "content_vertical_move" in describe_metadata(metadata)["official_columns_present"]
    result, report = survey(TIMELINES, metadata)
    by_name = {row["filename"]: row for row in result}
    walk = by_name["walk_forward_001__A006"]
    assert walk["buckets"] == ["support_height_change"]
    assert walk["evidence"]["support_height_change"] == {"metadata:content_vertical_move": ["1"]}
    assert walk["duration_seconds"] == pytest.approx(5.0)
    assert walk["duration_source"] == "metadata_frames_at_120fps"
    sneak = by_name["sneak_loop_004__A020"]
    assert sneak["mirror"] and sneak["mirror_source"] == "metadata"
    assert sneak["annotation_source"] == "missing"
    assert "metadata:content_body_position" in sneak["evidence"]["low_posture"]
    assert report["clips"] == {"total": 6, "timeline_only": 4, "metadata_only": 1, "both": 1}
    assert report["buckets"]["support_height_change"]["evidence_only_from_metadata"] == 1
    assert report["buckets"]["support_height_change"]["by_category"] == {"Environments": 1}


def test_pickle_dict_of_rows_is_keyed_by_filename(tmp_path):
    pd = pytest.importorskip("pandas")
    path = tmp_path / "meta.pkl"
    pd.to_pickle({"crawl_forward_001__A010": {"content_body_position": "crawling"}}, path)
    rows = read_metadata_rows(path)
    assert rows == [{"content_body_position": "crawling", "filename": "crawl_forward_001__A010"}]


def test_missing_filename_column_is_explicit():
    with pytest.raises(ValueError, match="describe-metadata"):
        survey({}, [{"name": "x"}])


def test_cli_writes_report_and_refuses_existing_output(tmp_path, capsys):
    timelines = tmp_path / "timelines.jsonl"
    timelines.write_text("\n".join(json.dumps(entry) for entry in TIMELINES.values()) + "\n")
    bvh = tmp_path / "bvh" / "20250101"
    bvh.mkdir(parents=True)
    (bvh / "stairs_climbing_up_002__A300.bvh").write_text("HIERARCHY\n")
    (bvh / "unannotated_clip__A1.bvh").write_text("HIERARCHY\n")
    output = tmp_path / "out"
    assert main(["--timelines", str(timelines), "--bvh-root", str(tmp_path / "bvh"),
                 "--output", str(output)]) == 0
    report = json.loads((output / "survey_report.json").read_text())
    assert report["inputs"]["timelines"]["entries"] == len(TIMELINES)
    assert report["bvh"] == {"files_indexed": 2, "named_clips_without_bvh": 4, "bvh_without_annotation": 1}
    assert report["buckets"]["support_height_change"]["with_bvh"] == 1
    candidates = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert {row["filename"] for row in candidates} == {
        "stairs_climbing_up_002__A300", "stairs_climbing_up_002__A300_M", "crawl_forward_001__A010"}
    assert "support_height_change" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        main(["--timelines", str(timelines), "--output", str(output)])
