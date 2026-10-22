"""Tests for local PARC batch candidate cataloging and gallery sidecars."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_parc_batch_gallery as gallery


def _write(path: Path, value: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else value.encode("utf-8"))


def _collection(tmp_path: Path, *, include_unreported=True):
    root = tmp_path / "download collection"
    job = root / "g2" / "run 01" / "results" / "platform_v1_s20262010"
    rows = []
    for rank, covered in ((0, True), (1, False)):
        name = f"candidate_{rank:03d}"
        source = job / name / "generated.pkl"
        _write(source, f"motion-{rank}")
        rows.append({
            "rank": rank,
            "candidate_id": 500 + rank,
            "motion": f"${PRIVATE_STORAGE}/results/platform_v1_s20262010/{name}/generated.pkl",
            "sha256": hashlib.sha256(f"motion-{rank}".encode()).hexdigest(),
            "frames": 60 + rank,
            "duration_s": 2.0,
            "endpoint_reached": covered,
            "geometric_route_covered": covered,
            "weighted_contact_score": 0.25,
        })
    if include_unreported:
        _write(job / "candidate_002" / "generated.pkl", b"unreported candidate")
    report = {
        "status": "kinematic_candidates_generated_physics_pending",
        "course_kind": "platform",
        "course_variant": 1,
        "training_approved": False,
        "candidates": rows,
    }
    _write(job / "generation.json", json.dumps(report))
    return root, job


def _fake_render(candidate, source, source_hash, output_dir, renderer_hash, repo,
                 panel_width, panel_height, force):
    if candidate["candidate"] == "candidate_001":
        raise RuntimeError("fixture render failure")
    base = (output_dir / "clips" / Path(*PurePosixPath(candidate["candidate_key"]).parts)
            / f"{source_hash[:16]}-{renderer_hash[:12]}-{panel_width}x{panel_height}")
    render_dir = gallery._next_render_dir(base) if force else base
    render_dir.mkdir(parents=True)
    video = render_dir / gallery.VIDEO_NAME
    _write(video, b"fake mp4 bytes")
    strict = {
        "source_motion_sha256": source_hash,
        "source_sha256_unchanged_after_render": True,
        "renderer_sha256": renderer_hash,
        "physics_verified": False,
        "pose_modification": "none",
        "ik": "not run",
        "smoothing_or_interpolation": "none",
        "playback_speed": 1.0,
        "source_fps": 30.0,
        "video": str(video),
    }
    gallery._write_json(render_dir / "manifest.json", strict)
    gallery._write_json(render_dir / "batch_render_manifest.json", {
        "schema_version": 1,
        "verified": True,
        "source_path": candidate["source_path"],
        "source_sha256": source_hash,
        "renderer_sha256": renderer_hash,
        "panel_width": panel_width,
        "panel_height": panel_height,
        "source_fps": 30.0,
        "playback_speed": 1.0,
        "pose_modification": "none",
        "ik": "not run",
        "smoothing_or_interpolation": "none",
        "physics_verified": False,
        "training_approved": False,
        "video_path": gallery.VIDEO_NAME,
        "video_size_bytes": video.stat().st_size,
    })
    return {
        "status": "rendered",
        "render_dir": render_dir.relative_to(output_dir).as_posix(),
        "video_path": video.relative_to(output_dir).as_posix(),
        "source_sha256": source_hash,
        "renderer_sha256": renderer_hash,
    }


def test_catalog_maps_remote_paths_and_includes_unreported_candidate(tmp_path):
    root, _ = _collection(tmp_path)

    catalog = gallery.build_catalog(root)

    assert catalog["candidate_count"] == 3
    assert catalog["physics_verified"] is False
    assert catalog["training_approved"] is False
    first = catalog["candidates"][0]
    assert first["source_path"] == (
        "g2/run 01/results/platform_v1_s20262010/candidate_000/generated.pkl"
    )
    assert first["source_exists"] is True
    assert first["geometric_route_covered"] is True
    assert catalog["candidates"][1]["geometric_route_covered"] is False
    unreported = catalog["candidates"][2]
    assert unreported["candidate"] == "candidate_002"
    assert unreported["report_entry"] is None
    assert unreported["coverage_state"] == "unknown"
    summary = catalog["summary_by_kind_variant"][0]
    assert (summary["course_kind"], summary["course_variant"]) == ("platform", 1)
    assert summary["candidates"] == 3
    assert summary["geometric_route_covered"] == 1
    assert summary["geometric_route_not_covered"] == 1
    assert summary["geometric_route_unknown"] == 1
    assert summary["physics_verified"] == summary["training_approved"] == 0


def test_report_rows_without_motion_and_disk_candidates_are_retained(tmp_path):
    root, job = _collection(tmp_path, include_unreported=False)
    report_path = job / "generation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["candidates"].append({
        "rank": 2,
        "endpoint_reached": False,
        "geometric_route_covered": False,
    })
    _write(report_path, json.dumps(report))

    catalog = gallery.build_catalog(root)

    assert catalog["candidate_count"] == 3
    missing = catalog["candidates"][2]
    assert missing["candidate"] == "candidate_002"
    assert missing["source_exists"] is False
    assert missing["geometric_route_covered"] is False
    assert "missing" in missing["source_error"]


def test_catalog_only_writes_static_gallery_with_caveat_labels(tmp_path):
    root, _ = _collection(tmp_path)

    result = gallery.run_batch(root, mode="catalog-only")

    page = Path(result["gallery_path"]).read_text(encoding="utf-8")
    outcomes = json.loads(Path(result["outcomes_path"]).read_text(encoding="utf-8"))
    assert result["candidate_count"] == 3
    assert all(row["status"] == "not_rendered" for row in outcomes["outcomes"])
    assert "RAW KINEMATIC" in page
    assert "PHYSICS UNVERIFIED" in page
    assert "NOT TRAINING APPROVED" in page
    assert "Geometric route coverage" in page
    assert "not covered 1" in page and "unknown 1" in page
    assert "https://" not in page and "http://" not in page


def test_gallery_uses_midframe_poster_and_pauses_hidden_clips(tmp_path):
    root, _ = _collection(tmp_path)
    catalog = gallery.build_catalog(root)
    page = gallery._gallery_html(catalog, [{
        "candidate_key": catalog["candidates"][0]["candidate_key"],
        "status": "rendered", "video_path": "clips/test.mp4",
        "poster_path": "clips/middle frame.png",
    }])
    assert 'poster="clips/middle%20frame.png"' in page
    assert "querySelector('video')?.pause()" in page


def test_render_limit_keeps_full_catalog_and_continues_after_clip_failure(tmp_path, monkeypatch):
    root, _ = _collection(tmp_path)
    monkeypatch.setattr(gallery, "_renderer_sha256", lambda: "a" * 64)
    monkeypatch.setattr(gallery, "_render_candidate", _fake_render)

    result = gallery.run_batch(root, mode="render-existing", limit=2)

    catalog = json.loads(Path(result["catalog_path"]).read_text(encoding="utf-8"))
    outcomes = json.loads(Path(result["outcomes_path"]).read_text(encoding="utf-8"))["outcomes"]
    assert catalog["candidate_count"] == 3
    assert [row["status"] for row in outcomes] == [
        "rendered", "failed", "not_selected_by_limit"
    ]
    assert result["rendered"] == 1 and result["failed"] == 1
    page = Path(result["gallery_path"]).read_text(encoding="utf-8")
    assert "run%2001" in page
    assert gallery.VIDEO_NAME in page
    assert "fixture render failure" in page


def test_existing_cache_requires_matching_source_renderer_and_strict_manifest(tmp_path):
    root, _ = _collection(tmp_path)
    candidate = gallery.build_catalog(root)["candidates"][0]
    source = root.joinpath(*PurePosixPath(candidate["source_path"]).parts)
    source_hash = gallery._sha256_file(source)
    renderer_hash = "b" * 64
    render_dir = tmp_path / "cache"
    render_dir.mkdir()
    video = render_dir / gallery.VIDEO_NAME
    _write(video, b"cached video")
    gallery._write_json(render_dir / "manifest.json", {
        "source_motion_sha256": source_hash,
        "source_sha256_unchanged_after_render": True,
        "renderer_sha256": renderer_hash,
        "physics_verified": False,
        "pose_modification": "none",
        "ik": "not run",
        "smoothing_or_interpolation": "none",
        "playback_speed": 1.0,
        "source_fps": 30.0,
        "video": str(video),
    })
    gallery._write_json(render_dir / "batch_render_manifest.json", {
        "verified": True,
        "source_sha256": source_hash,
        "renderer_sha256": renderer_hash,
        "panel_width": 480,
        "panel_height": 360,
        "source_fps": 30.0,
        "playback_speed": 1.0,
        "pose_modification": "none",
        "ik": "not run",
        "smoothing_or_interpolation": "none",
        "physics_verified": False,
        "training_approved": False,
        "video_path": gallery.VIDEO_NAME,
        "video_size_bytes": video.stat().st_size,
    })

    assert gallery._verified_cache(render_dir, source_hash, renderer_hash, 480, 360) == video
    assert gallery._verified_cache(render_dir, "0" * 64, renderer_hash, 480, 360) is None
    assert gallery._verified_cache(render_dir, source_hash, "0" * 64, 480, 360) is None
    gallery._write_json(render_dir / "manifest.json", {"physics_verified": True})
    assert gallery._verified_cache(render_dir, source_hash, renderer_hash, 480, 360) is None


def test_render_existing_reuses_verified_cache_without_loading_renderer(tmp_path, monkeypatch):
    root, _ = _collection(tmp_path)
    output = tmp_path / "gallery output"
    candidate = gallery.build_catalog(root)["candidates"][0]
    source = root.joinpath(*PurePosixPath(candidate["source_path"]).parts)
    source_hash = gallery._sha256_file(source)
    renderer_hash = "c" * 64
    _fake_render(candidate, source, source_hash, output, renderer_hash, None, 480, 360, False)
    monkeypatch.setattr(gallery, "_renderer_sha256", lambda: renderer_hash)
    monkeypatch.setattr(gallery, "_load_renderer", lambda: pytest.fail("cache miss"))

    result = gallery.run_batch(root, output, mode="render-existing", limit=1)

    outcomes = json.loads(Path(result["outcomes_path"]).read_text(encoding="utf-8"))["outcomes"]
    assert outcomes[0]["status"] == "cached"
    assert outcomes[0]["video_path"].endswith(gallery.VIDEO_NAME)
    assert "Open MP4" in Path(result["gallery_path"]).read_text(encoding="utf-8")


def test_render_all_bypasses_a_valid_cache(tmp_path, monkeypatch):
    root, _ = _collection(tmp_path)
    output = tmp_path / "gallery output"
    candidate = gallery.build_catalog(root)["candidates"][0]
    source = root.joinpath(*PurePosixPath(candidate["source_path"]).parts)
    source_hash = gallery._sha256_file(source)
    renderer_hash = "d" * 64
    _fake_render(candidate, source, source_hash, output, renderer_hash, None, 480, 360, False)
    monkeypatch.setattr(gallery, "_renderer_sha256", lambda: renderer_hash)

    class FakeRenderer:
        @staticmethod
        def render_long_course(motion, render_dir, **kwargs):
            render_dir.mkdir(parents=True)
            video = render_dir / gallery.VIDEO_NAME
            _write(video, b"fresh fake mp4")
            manifest = {
                "source_motion_sha256": source_hash,
                "source_sha256_unchanged_after_render": True,
                "renderer_sha256": renderer_hash,
                "physics_verified": False,
                "pose_modification": "none",
                "ik": "not run",
                "smoothing_or_interpolation": "none",
                "playback_speed": 1.0,
                "source_fps": 30.0,
                "video": str(video),
            }
            gallery._write_json(render_dir / "manifest.json", manifest)
            return manifest

    monkeypatch.setattr(gallery, "_load_renderer", lambda: FakeRenderer)

    result = gallery.run_batch(root, output, mode="render-all", limit=1)

    outcomes = json.loads(Path(result["outcomes_path"]).read_text(encoding="utf-8"))["outcomes"]
    assert outcomes[0]["status"] == "rendered"
    assert "-run-0002/" in outcomes[0]["video_path"]
    assert Path(output, outcomes[0]["video_path"]).read_bytes() == b"fresh fake mp4"


def test_nonexistent_and_zero_candidate_collections_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        gallery.build_catalog(tmp_path / "absent")

    empty = tmp_path / "empty"
    (empty / "g2" / "run" / "results" / "job").mkdir(parents=True)
    _write(empty / "g2" / "run" / "results" / "job" / "generation.json",
           json.dumps({"candidates": []}))
    with pytest.raises(ValueError, match="No PARC candidates"):
        gallery.build_catalog(empty)


@pytest.mark.parametrize("limit", [0, -1])
def test_limit_must_be_positive(tmp_path, limit):
    root, _ = _collection(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        gallery.run_batch(root, mode="catalog-only", limit=limit)
