"""Local tests for PARC collection summary verification and selection."""

import hashlib
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_parc_collection as summarizer


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_specs():
    return [
        {
            "candidate_key": "g3/run/results/extension_v0_s20262008/candidate_000",
            "course_kind": "extension",
            "course_variant": 0,
            "frames": 30,
            "duration_s": 1.0,
            "decoded_frames": 30,
            "endpoint_reached": True,
            "geometric_route_covered": False,
            "covered_segments": [True, True, True, True, True],
            "selection_score": -100.0,
        },
        {
            "candidate_key": "g3/run/results/extension_v0_s20262008/candidate_001",
            "course_kind": "extension",
            "course_variant": 0,
            "frames": 60,
            "duration_s": 2.0,
            "decoded_frames": 60,
            "endpoint_reached": False,
            "geometric_route_covered": True,
            "covered_segments": [True],
            "selection_score": -10.0,
        },
        {
            "candidate_key": "g3/run/results/extension_v0_s20262008/candidate_002",
            "course_kind": "extension",
            "course_variant": 0,
            "frames": 90,
            "duration_s": 3.0,
            "decoded_frames": 90,
            "endpoint_reached": True,
            "geometric_route_covered": True,
            "covered_segments": [True, True, True],
            "selection_score": 10.0,
        },
        {
            "candidate_key": "g3/run/results/extension_v0_s20262008/candidate_003",
            "course_kind": "extension",
            "course_variant": 0,
            "frames": 120,
            "duration_s": 4.0,
            "decoded_frames": 120,
            "endpoint_reached": True,
            "geometric_route_covered": True,
            "covered_segments": [True, True, True],
            "selection_score": 2.0,
        },
        {
            "candidate_key": "g3/run/results/extension_v1_s20262009/candidate_000",
            "course_kind": "extension",
            "course_variant": 1,
            "frames": 150,
            "duration_s": 5.0,
            "decoded_frames": 150,
            "endpoint_reached": False,
            "geometric_route_covered": False,
            "covered_segments": [],
            "selection_score": 1.0,
        },
    ]


def _make_collection(tmp_path, monkeypatch):
    output = tmp_path / "collection"
    gallery = output / "gallery"
    specs = _candidate_specs()
    candidates, outcomes = [], []
    decoded = {}

    for index, spec in enumerate(specs):
        key = spec["candidate_key"]
        source_path = f"{key}/generated.pkl"
        source = output / "downloads" / source_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source_bytes = f"source motion {index}".encode("ascii")
        source.write_bytes(source_bytes)
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()

        render_dir = f"clips/candidate_{index:03d}"
        video_path = f"{render_dir}/generated_kinematic_candidate_1x.mp4"
        video = gallery / video_path
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video {index}".encode("ascii"))
        manifest = {
            "frames": spec["frames"],
            "source_fps": 30.0,
            "source_motion_sha256": source_sha256,
            "pose_modification": "none",
            "mj_step_called": False,
            "physics_verified": False,
        }
        _write_json(gallery / render_dir / "manifest.json", manifest)
        decoded[str(video)] = (spec["decoded_frames"], spec["decoded_frames"] / 30.0)

        candidates.append({
            "candidate_key": key,
            "course_kind": spec["course_kind"],
            "course_variant": spec["course_variant"],
            "frames": spec["frames"],
            "duration_s": spec["duration_s"],
            "endpoint_reached": spec["endpoint_reached"],
            "geometric_route_covered": spec["geometric_route_covered"],
            "reported_source_sha256": source_sha256,
            "source_path": source_path,
            "report_entry": {
                "passage": {"segments": [
                    {"covered_in_order": covered}
                    for covered in spec["covered_segments"]
                ]},
                "selection_score": spec["selection_score"],
            },
        })
        outcomes.append({
            "candidate_key": key,
            "status": "cached" if index % 2 == 0 else "rendered",
            "render_dir": render_dir,
            "video_path": video_path,
        })

    _write_json(output / "completion_progress.json", {
        "state": "generated_and_rendered",
        "total_candidates": len(candidates),
    })
    _write_json(gallery / "catalog.json", {
        "candidate_count": len(candidates),
        "physics_verified": False,
        "training_approved": False,
        "candidates": candidates,
    })
    _write_json(gallery / "render_outcomes.json", {
        "physics_verified": False,
        "training_approved": False,
        "outcomes": outcomes,
    })
    monkeypatch.setattr(
        summarizer.imageio_ffmpeg,
        "count_frames_and_secs",
        lambda video_path: decoded[str(Path(video_path))],
    )
    return output, specs, candidates, outcomes, decoded


def test_summarize_writes_exact_complete_catalog_and_outcome_summary(tmp_path, monkeypatch):
    output, specs, _, _, _ = _make_collection(tmp_path, monkeypatch)

    summary = summarizer.summarize(output)

    expected_verified = []
    for index, spec in enumerate(specs):
        key = spec["candidate_key"]
        source = output / "downloads" / f"{key}/generated.pkl"
        video = output / "gallery" / f"clips/candidate_{index:03d}/generated_kinematic_candidate_1x.mp4"
        expected_verified.append({
            "candidate_key": key,
            "frames": spec["frames"],
            "fps": 30.0,
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "video": str(video),
            "duration_s": spec["frames"] / 30.0,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })

    expected = {
        "status": "raw_generation_and_full_video_delivery_complete",
        "candidates": 5,
        "videos_decoded_and_frame_count_verified": 5,
        "source_files_sha256_unique": 5,
        "motion_time_span_seconds": 15.0,
        "video_stream_duration_seconds": 15.0,
        "groups": [
            {
                "kind": "extension",
                "variant": 0,
                "candidates": 4,
                "endpoint_reached": 3,
                "geometric_route_covered": 3,
                "motion_time_span_seconds": 10.0,
            },
            {
                "kind": "extension",
                "variant": 1,
                "candidates": 1,
                "endpoint_reached": 0,
                "geometric_route_covered": 0,
                "motion_time_span_seconds": 5.0,
            },
        ],
        "showcase": [
            {
                "name": "terraces",
                "candidate_key": specs[3]["candidate_key"],
                "video": str(output / "showcase" / "terraces.mp4"),
                "video_sha256": expected_verified[3]["video_sha256"],
                "geometric_route_covered": True,
                "physics_verified": False,
                "training_approved": False,
            },
            {
                "name": "bridges",
                "candidate_key": specs[4]["candidate_key"],
                "video": str(output / "showcase" / "bridges.mp4"),
                "video_sha256": expected_verified[4]["video_sha256"],
                "geometric_route_covered": False,
                "physics_verified": False,
                "training_approved": False,
            },
        ],
        "gallery": str(output / "gallery" / "index.html"),
        "showcase_selection": (
            "Prefer full geometric route coverage, then segment coverage, then official "
            "selection score; all candidates remain in the gallery"
        ),
        "physics_verified": False,
        "training_approved": False,
        "not_done": ["Official physics tracking", "SOMA retarget admission", "MB training"],
        "verified_videos": expected_verified,
    }
    assert summary == expected
    assert _read_json(output / "summary.json") == expected


def test_representative_selection_keeps_every_candidate_in_verified_summary(tmp_path, monkeypatch):
    output, specs, _, outcomes, _ = _make_collection(tmp_path, monkeypatch)

    summary = summarizer.summarize(output)

    assert summary["showcase"][0]["candidate_key"] == specs[3]["candidate_key"]
    assert [row["candidate_key"] for row in summary["verified_videos"]] == [
        spec["candidate_key"] for spec in specs
    ]
    assert summary["candidates"] == len(specs)
    assert all((output / "gallery" / row["video_path"]).is_file() for row in outcomes)


def test_rejects_collection_not_marked_generated_and_rendered(tmp_path, monkeypatch):
    output, _, _, _, _ = _make_collection(tmp_path, monkeypatch)
    progress_path = output / "completion_progress.json"
    progress = _read_json(progress_path)
    progress["state"] = "rendering"
    _write_json(progress_path, progress)

    with pytest.raises(ValueError, match="Generation and rendering have not finished"):
        summarizer.summarize(output)


def test_rejects_catalog_or_outcome_count_short_of_completed_batch(tmp_path, monkeypatch):
    output, _, candidates, _, _ = _make_collection(tmp_path, monkeypatch)
    catalog_path = output / "gallery" / "catalog.json"
    catalog = _read_json(catalog_path)
    catalog["candidates"] = candidates[:-1]
    _write_json(catalog_path, catalog)

    with pytest.raises(ValueError, match="Catalog/outcome counts"):
        summarizer.summarize(output)


def test_rejects_video_decode_frame_count_mismatch(tmp_path, monkeypatch):
    output, specs, _, outcomes, decoded = _make_collection(tmp_path, monkeypatch)
    video = output / "gallery" / outcomes[0]["video_path"]
    decoded[str(video)] = (specs[0]["frames"] - 1, (specs[0]["frames"] - 1) / 30.0)

    with pytest.raises(ValueError, match="full source timing"):
        summarizer.summarize(output)


@pytest.mark.parametrize("mismatch", ["manifest", "source_file"])
def test_rejects_source_hash_mismatch(tmp_path, monkeypatch, mismatch):
    output, _, candidates, outcomes, _ = _make_collection(tmp_path, monkeypatch)
    if mismatch == "manifest":
        manifest_path = output / "gallery" / outcomes[0]["render_dir"] / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest["source_motion_sha256"] = "0" * 64
        _write_json(manifest_path, manifest)
        error = "Render/source hash mismatch"
    else:
        source = output / "downloads" / candidates[0]["source_path"]
        source.write_bytes(b"changed after render")
        error = "Source file changed after rendering"

    with pytest.raises(ValueError, match=error):
        summarizer.summarize(output)


@pytest.mark.parametrize(("field", "value"), [
    ("pose_modification", "smoothed"),
    ("mj_step_called", True),
    ("physics_verified", True),
])
def test_rejects_pose_changes_or_physics_claims(tmp_path, monkeypatch, field, value):
    output, _, _, outcomes, _ = _make_collection(tmp_path, monkeypatch)
    manifest_path = output / "gallery" / outcomes[0]["render_dir"] / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest[field] = value
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="Unexpected motion alteration or physics claim"):
        summarizer.summarize(output)


def test_duplicate_catalog_key_cannot_stand_in_for_a_missing_candidate(tmp_path, monkeypatch):
    output, _, _, _, _ = _make_collection(tmp_path, monkeypatch)
    catalog_path = output / "gallery" / "catalog.json"
    catalog = _read_json(catalog_path)
    catalog["candidates"][-1] = catalog["candidates"][0]
    _write_json(catalog_path, catalog)

    with pytest.raises(ValueError, match="candidate"):
        summarizer.summarize(output)


def test_decoded_video_must_match_catalog_source_frame_count(tmp_path, monkeypatch):
    output, _, _, _, _ = _make_collection(tmp_path, monkeypatch)
    catalog_path = output / "gallery" / "catalog.json"
    catalog = _read_json(catalog_path)
    catalog["candidates"][0]["frames"] += 1
    catalog["candidates"][0]["duration_s"] += 1 / 30.0
    _write_json(catalog_path, catalog)

    with pytest.raises(ValueError, match="full source timing"):
        summarizer.summarize(output)


@pytest.mark.parametrize("mismatch", ["duplicate", "unexpected"])
def test_outcome_candidate_identities_must_match_exactly(tmp_path, monkeypatch, mismatch):
    output, _, _, _, _ = _make_collection(tmp_path, monkeypatch)
    path = output / "gallery" / "render_outcomes.json"
    data = _read_json(path)
    rows = data["outcomes"]
    rows[-1]["candidate_key"] = rows[0]["candidate_key"] if mismatch == "duplicate" else "unexpected"
    _write_json(path, data)
    with pytest.raises(ValueError, match="candidate identities"):
        summarizer.summarize(output)
