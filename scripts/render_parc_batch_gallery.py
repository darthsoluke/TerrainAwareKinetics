"""Catalog and optionally render a local collection of PARC batch candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any
from urllib.parse import quote


SCRIPT_DIR = Path(__file__).resolve().parent
RENDERER_PATH = SCRIPT_DIR / "render_parc_long_course.py"
DEFAULT_PANEL_WIDTH = 480
DEFAULT_PANEL_HEIGHT = 360
VIDEO_NAME = "generated_kinematic_candidate_1x.mp4"
_CANDIDATE_RE = re.compile(r"candidate_(\d+)$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not (float("-inf") < value < float("inf")):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _as_rank(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        rank = int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return rank if rank >= 0 else fallback


def _reported_candidate_relative(motion_path: Any) -> PurePosixPath:
    """Keep only the candidate-relative suffix of an original remote path."""
    if not isinstance(motion_path, str) or not motion_path.strip():
        raise ValueError("report has no motion path")
    remote = PurePosixPath(motion_path.replace("\\", "/"))
    matches = [index for index, part in enumerate(remote.parts)
               if _CANDIDATE_RE.fullmatch(part)]
    if not matches:
        raise ValueError("reported motion path has no candidate_### component")
    relative = PurePosixPath(*remote.parts[matches[-1]:])
    if len(relative.parts) != 2 or relative.parts[1] != "generated.pkl":
        raise ValueError("reported motion path must end in candidate_###/generated.pkl")
    return relative


def _report_paths(collection: Path) -> list[Path]:
    reports = []
    for path in collection.rglob("generation.json"):
        relative = path.relative_to(collection)
        if len(relative.parts) >= 5 and relative.parts[-3] == "results":
            reports.append(path)
    return sorted(reports, key=lambda path: path.relative_to(collection).as_posix())


def _coverage(value: Any) -> str:
    if value is True:
        return "covered"
    if value is False:
        return "not_covered"
    return "unknown"


def _candidate_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        kind = str(candidate.get("course_kind") or "unknown")
        variant = candidate.get("course_variant")
        variant_key = "unknown" if variant is None else str(variant)
        key = (kind, variant_key)
        summary = grouped.setdefault(key, {
            "course_kind": kind,
            "course_variant": variant,
            "candidates": 0,
            "geometric_route_covered": 0,
            "geometric_route_not_covered": 0,
            "geometric_route_unknown": 0,
            "physics_verified": 0,
            "training_approved": 0,
        })
        summary["candidates"] += 1
        summary[f"geometric_route_{_coverage(candidate.get('geometric_route_covered'))}"] += 1
    return [grouped[key] for key in sorted(grouped)]


def build_catalog(collection_dir: Path) -> dict[str, Any]:
    collection = Path(collection_dir).expanduser().resolve()
    if not collection.exists() or not collection.is_dir():
        raise FileNotFoundError(f"Collection directory does not exist: {collection}")

    jobs = []
    candidates: dict[str, dict[str, Any]] = {}
    for report_path in _report_paths(collection):
        report_relative = report_path.relative_to(collection)
        job_dir = report_path.parent
        job_relative = job_dir.relative_to(collection)
        host = job_relative.parts[0]
        run = job_relative.parts[1]
        job_id = job_dir.name
        report: dict[str, Any] = {}
        report_error = None
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("generation.json root must be an object")
            report = loaded
        except (OSError, json.JSONDecodeError, ValueError) as error:
            report_error = f"{type(error).__name__}: {error}"

        job_info = {
            "host": host,
            "run": run,
            "job_id": job_id,
            "job_key": job_relative.as_posix(),
            "report_path": report_relative.as_posix(),
            "report_status": report.get("status"),
            "course_kind": report.get("course_kind"),
            "course_variant": report.get("course_variant"),
            "report_error": report_error,
        }
        jobs.append(job_info)

        rows = report.get("candidates", [])
        if not isinstance(rows, list):
            rows = []
            job_info["report_error"] = job_info["report_error"] or \
                "ValueError: candidates must be a list"
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                row = {"report_row": _json_safe(row)}
            rank = _as_rank(row.get("rank"), row_index)
            path_error = None
            try:
                candidate_relative = _reported_candidate_relative(row.get("motion"))
            except ValueError as error:
                path_error = str(error)
                candidate_relative = PurePosixPath(f"candidate_{rank:03d}", "generated.pkl")
            candidate_name = candidate_relative.parts[0]
            candidate_dir_name = job_relative / candidate_name
            key = candidate_dir_name.as_posix()
            item = candidates.setdefault(key, {
                "candidate_key": key,
                "host": host,
                "run": run,
                "job_id": job_id,
                "job_key": job_relative.as_posix(),
                "candidate": candidate_name,
                "rank": rank,
                "course_kind": report.get("course_kind"),
                "course_variant": report.get("course_variant"),
                "report_status": report.get("status"),
                "report_path": report_relative.as_posix(),
                "report_motion_path": row.get("motion"),
                "source_path": (job_relative / candidate_relative).as_posix(),
                "source_mapping_error": path_error,
                "endpoint_reached": row.get("endpoint_reached"),
                "geometric_route_covered": (row.get("geometric_route_covered")
                                               if isinstance(row.get("geometric_route_covered"), bool)
                                               else None),
                "reported_source_sha256": row.get("sha256"),
                "frames": row.get("frames"),
                "duration_s": row.get("duration_s"),
                "report_entry": _json_safe(row),
            })
            if item.get("source_mapping_error") is None and path_error:
                item["source_mapping_error"] = path_error

        # The filesystem is authoritative for candidates saved just before an
        # interrupted report update.
        for candidate_dir in sorted(job_dir.iterdir(), key=lambda path: path.name):
            if not candidate_dir.is_dir() or not _CANDIDATE_RE.fullmatch(candidate_dir.name):
                continue
            key = (job_relative / candidate_dir.name).as_posix()
            candidates.setdefault(key, {
                "candidate_key": key,
                "host": host,
                "run": run,
                "job_id": job_id,
                "job_key": job_relative.as_posix(),
                "candidate": candidate_dir.name,
                "rank": int(_CANDIDATE_RE.fullmatch(candidate_dir.name).group(1)),
                "course_kind": report.get("course_kind"),
                "course_variant": report.get("course_variant"),
                "report_status": report.get("status"),
                "report_path": report_relative.as_posix(),
                "report_motion_path": None,
                "source_path": (job_relative / candidate_dir.name / "generated.pkl").as_posix(),
                "source_mapping_error": None,
                "endpoint_reached": None,
                "geometric_route_covered": None,
                "reported_source_sha256": None,
                "frames": None,
                "duration_s": None,
                "report_entry": None,
            })

    ordered = [candidates[key] for key in sorted(candidates)]
    if not ordered:
        raise ValueError(f"No PARC candidates found under: {collection}")

    for candidate in ordered:
        source = collection.joinpath(*PurePosixPath(candidate["source_path"]).parts)
        candidate["source_exists"] = source.is_file() and not source.is_symlink()
        if source.is_symlink():
            candidate["source_error"] = "candidate source is a symlink"
        elif not source.is_file():
            candidate["source_error"] = "generated.pkl is missing"
        else:
            try:
                source.resolve().relative_to(collection)
                candidate["source_error"] = None
            except ValueError:
                candidate["source_exists"] = False
                candidate["source_error"] = "candidate source resolves outside the collection"
        candidate["coverage_state"] = _coverage(candidate.get("geometric_route_covered"))

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_dir": str(collection),
        "candidate_count": len(ordered),
        "job_count": len(jobs),
        "physics_verified": False,
        "training_approved": False,
        "jobs": jobs,
        "summary_by_kind_variant": _candidate_summary(ordered),
        "candidates": ordered,
    }


def _renderer_sha256() -> str:
    if not RENDERER_PATH.is_file():
        raise FileNotFoundError(f"Strict PARC course renderer is missing: {RENDERER_PATH}")
    return _sha256_file(RENDERER_PATH)


def _safe_manifest_video(render_dir: Path, manifest: dict[str, Any]) -> Path | None:
    relative = manifest.get("video_path")
    if not isinstance(relative, str):
        return None
    parsed = PurePosixPath(relative.replace("\\", "/"))
    if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
        return None
    path = render_dir.joinpath(*parsed.parts)
    try:
        path.resolve().relative_to(render_dir.resolve())
    except (OSError, ValueError):
        return None
    return path


def _verified_cache(render_dir: Path, source_hash: str, renderer_hash: str,
                    panel_width: int, panel_height: int) -> Path | None:
    sidecar_path = render_dir / "batch_render_manifest.json"
    strict_path = render_dir / "manifest.json"
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(sidecar, dict) or not isinstance(strict, dict):
        return None
    video = _safe_manifest_video(render_dir, sidecar)
    try:
        if video is None or not video.is_file() or video.stat().st_size <= 0:
            return None
        strict_video = strict.get("video")
        strict_video_path = Path(strict_video).resolve() if isinstance(strict_video, str) else None
    except (OSError, TypeError, ValueError):
        return None
    checks = (
        sidecar.get("verified") is True,
        sidecar.get("source_sha256") == source_hash,
        sidecar.get("renderer_sha256") == renderer_hash,
        sidecar.get("panel_width") == panel_width,
        sidecar.get("panel_height") == panel_height,
        sidecar.get("video_size_bytes") == video.stat().st_size,
        sidecar.get("physics_verified") is False,
        sidecar.get("training_approved") is False,
        sidecar.get("playback_speed") == 1.0,
        sidecar.get("pose_modification") == "none",
        sidecar.get("ik") == "not run",
        sidecar.get("smoothing_or_interpolation") == "none",
        strict.get("source_motion_sha256") == source_hash,
        strict.get("source_sha256_unchanged_after_render") is True,
        strict.get("renderer_sha256") == renderer_hash,
        strict.get("physics_verified") is False,
        strict.get("training_approved", False) is False,
        strict.get("pose_modification") == "none",
        strict.get("ik") == "not run",
        strict.get("smoothing_or_interpolation") == "none",
        strict.get("playback_speed") == 1.0,
        strict.get("source_fps") == sidecar.get("source_fps"),
        strict_video_path == video.resolve(),
    )
    if not all(checks):
        return None
    return video


def _load_renderer():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import render_parc_long_course
    return render_parc_long_course


def _next_render_dir(base: Path) -> Path:
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = base.with_name(f"{base.name}-run-{index:04d}")
        if not candidate.exists():
            return candidate
        index += 1


def _render_candidate(candidate: dict[str, Any], source: Path, source_hash: str,
                      output_dir: Path, renderer_hash: str, repo: Path | None,
                      panel_width: int, panel_height: int, force: bool) -> dict[str, Any]:
    candidate_key = PurePosixPath(candidate["candidate_key"])
    base = output_dir / "clips" / Path(*candidate_key.parts) / \
        f"{source_hash[:16]}-{renderer_hash[:12]}-{panel_width}x{panel_height}"

    if not force:
        for render_dir in sorted(base.parent.glob(base.name + "*"), reverse=True):
            if not render_dir.is_dir():
                continue
            cached_video = _verified_cache(render_dir, source_hash, renderer_hash,
                                           panel_width, panel_height)
            if cached_video is not None:
                return {
                    "status": "cached",
                    "render_dir": render_dir.relative_to(output_dir).as_posix(),
                    "video_path": cached_video.relative_to(output_dir).as_posix(),
                    "source_sha256": source_hash,
                    "renderer_sha256": renderer_hash,
                }

    render_dir = _next_render_dir(base)
    renderer = _load_renderer()
    render_options = {
        "panel_width": panel_width,
        "panel_height": panel_height,
        "gif_preview": False,
    }
    if repo is not None:
        render_options["repo"] = repo
    strict_manifest = renderer.render_long_course(source, render_dir, **render_options)
    if _sha256_file(source) != source_hash:
        raise RuntimeError("source generated.pkl changed during rendering")
    video = render_dir / VIDEO_NAME
    if not video.is_file() or video.stat().st_size <= 0:
        raise RuntimeError("strict renderer returned without a nonempty MP4")
    if strict_manifest.get("source_motion_sha256") != source_hash:
        raise RuntimeError("strict renderer manifest source hash does not match")
    if strict_manifest.get("renderer_sha256") != renderer_hash:
        raise RuntimeError("strict renderer manifest hash does not match renderer source")
    sidecar = {
        "schema_version": 1,
        "verified": True,
        "source_path": candidate["source_path"],
        "source_sha256": source_hash,
        "renderer_path": RENDERER_PATH.name,
        "renderer_sha256": renderer_hash,
        "panel_width": panel_width,
        "panel_height": panel_height,
        "source_fps": strict_manifest.get("source_fps"),
        "playback_speed": 1.0,
        "pose_modification": "none",
        "ik": "not run",
        "smoothing_or_interpolation": "none",
        "physics_verified": False,
        "training_approved": False,
        "video_path": VIDEO_NAME,
        "video_size_bytes": video.stat().st_size,
    }
    _write_json(render_dir / "batch_render_manifest.json", sidecar)
    verified_video = _verified_cache(render_dir, source_hash, renderer_hash,
                                     panel_width, panel_height)
    if verified_video is None:
        raise RuntimeError("render completed but did not pass cache verification")
    return {
        "status": "rendered",
        "render_dir": render_dir.relative_to(output_dir).as_posix(),
        "video_path": verified_video.relative_to(output_dir).as_posix(),
        "source_sha256": source_hash,
        "renderer_sha256": renderer_hash,
    }


def _outcome(candidate: dict[str, Any], status: str, **fields) -> dict[str, Any]:
    return {
        "candidate_key": candidate["candidate_key"],
        "status": status,
        "geometric_route_covered": candidate.get("geometric_route_covered"),
        "physics_verified": False,
        "training_approved": False,
        **fields,
    }


def _gallery_html(catalog: dict[str, Any], outcomes: list[dict[str, Any]]) -> str:
    outcome_by_key = {row["candidate_key"]: row for row in outcomes}
    candidates = catalog["candidates"]
    course_options = sorted({
        (str(item.get("course_kind") or "unknown"),
         "unknown" if item.get("course_variant") is None else str(item["course_variant"]))
        for item in candidates
    })
    options = ['<option value="">All courses</option>']
    for kind, variant in course_options:
        value = f"{kind}|{variant}"
        options.append(f'<option value="{html.escape(value, quote=True)}">'
                       f'{html.escape(kind)} v{html.escape(variant)}</option>')

    articles = []
    for candidate in candidates:
        result = outcome_by_key.get(candidate["candidate_key"], {})
        video_path = result.get("video_path") if result.get("status") in ("rendered", "cached") else None
        if video_path:
            relative = Path(video_path).as_posix()
            video_url = quote(relative, safe="/")
            poster = result.get("poster_path")
            poster_attribute = (f' poster="{html.escape(quote(poster, safe="/"), quote=True)}"'
                                if poster else "")
            video = (f'<video controls preload="none" playsinline '
                     f'{poster_attribute} '
                     f'aria-label="{html.escape(candidate["candidate"], quote=True)} raw kinematic candidate">'
                     f'<source src="{html.escape(video_url, quote=True)}" type="video/mp4">'
                     "Your browser does not support MP4 playback.</video>")
            mp4_link = (f'<a class="mp4-link" href="{html.escape(video_url, quote=True)}">'
                        "Open MP4</a>")
        else:
            message = "Not rendered" if not result else result.get("error", result.get("status", "Not rendered"))
            video = f'<div class="missing">{html.escape(str(message))}</div>'
            mp4_link = ""
        kind = str(candidate.get("course_kind") or "unknown")
        variant = "unknown" if candidate.get("course_variant") is None else str(candidate["course_variant"])
        course = f"{kind}|{variant}"
        coverage = candidate.get("coverage_state", "unknown")
        coverage_label = {
            "covered": "GEOMETRIC ROUTE COVERED",
            "not_covered": "GEOMETRIC ROUTE NOT COVERED",
            "unknown": "GEOMETRIC ROUTE UNKNOWN",
        }.get(coverage, "GEOMETRIC ROUTE UNKNOWN")
        display_key = f'{candidate["job_id"]} / {candidate["candidate"]}'
        search = f'{candidate["host"]} {candidate["run"]} {candidate["job_id"]} {candidate["candidate"]}'
        seconds = candidate.get("duration_s")
        duration_label = f"{seconds:.2f}" if isinstance(seconds, (int, float)) else "?"
        articles.append(
            f'<article class="clip" data-course="{html.escape(course, quote=True)}" '
            f'data-coverage="{html.escape(coverage, quote=True)}" '
            f'data-search="{html.escape(search.casefold(), quote=True)}">'
            f'<div class="clip-head"><strong title="{html.escape(candidate["candidate_key"], quote=True)}">'
            f'{html.escape(display_key)}</strong><span>{html.escape(kind)} v{html.escape(variant)}</span></div>'
            f'{video}{mp4_link}<div class="labels"><span>RAW KINEMATIC</span>'
            f'<span>PHYSICS UNVERIFIED</span><span>NOT TRAINING APPROVED</span></div>'
            f'<div class="coverage {coverage}">{coverage_label}</div>'
            f'<div class="meta">{html.escape(str(candidate.get("frames") or "?"))} frames'
            f' &middot; {duration_label} s'
            f' &middot; {html.escape(str(result.get("status", "not_rendered")))}</div>'
            "</article>"
        )

    body = "\n".join(articles)
    summary_rows = []
    for summary in catalog["summary_by_kind_variant"]:
        kind = html.escape(str(summary["course_kind"]))
        variant = html.escape(str(summary["course_variant"]))
        summary_rows.append(
            f'<div class="summary-row"><strong>{kind} v{variant}</strong>'
            f'<span>{summary["candidates"]} candidates</span>'
            f'<span>covered {summary["geometric_route_covered"]}</span>'
            f'<span>not covered {summary["geometric_route_not_covered"]}</span>'
            f'<span>unknown {summary["geometric_route_unknown"]}</span></div>'
        )
    summary_html = "".join(summary_rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PARC Candidate Gallery</title>
<style>
:root {{ color-scheme: light; font: 14px/1.4 system-ui, sans-serif; color: #20282d; background: #f4f6f6; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; }}
header {{ background: #20282d; color: #fff; padding: 14px 18px; }}
h1 {{ font-size: 20px; line-height: 1.2; margin: 0 0 7px; }}
.notice {{ color: #f1cf61; font-weight: 700; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: end; padding: 12px 18px; border-bottom: 1px solid #cbd2d3; background: #fff; }}
.toolbar label {{ display: grid; gap: 3px; font-size: 12px; font-weight: 650; }}
select, input {{ min-height: 34px; min-width: 175px; padding: 5px 8px; border: 1px solid #929c9f; border-radius: 4px; background: #fff; color: inherit; font: inherit; }}
#count {{ margin-left: auto; align-self: center; font-variant-numeric: tabular-nums; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr)); gap: 12px; padding: 12px 18px 24px; }}
.clip {{ min-width: 0; border: 1px solid #c8d0d1; border-radius: 5px; background: #fff; overflow: hidden; }}
.clip-head {{ display: flex; justify-content: space-between; gap: 10px; padding: 8px 9px 6px; }}
.clip-head strong {{ overflow-wrap: anywhere; }}
.clip-head span {{ flex: 0 0 auto; color: #526166; }}
video {{ display: block; width: 100%; aspect-ratio: 960 / 472; background: #20282d; object-fit: contain; }}
.mp4-link {{ display: inline-block; padding: 5px 9px 0; color: #125a72; }}
.missing {{ display: grid; place-items: center; min-height: 120px; padding: 12px; color: #6a3030; background: #f5e8e6; overflow-wrap: anywhere; }}
.labels {{ display: flex; flex-wrap: wrap; gap: 4px; padding: 7px 8px 3px; }}
.labels span {{ border: 1px solid #a76f14; border-radius: 3px; padding: 2px 5px; color: #633f08; background: #fff5d8; font-size: 10px; font-weight: 750; }}
.coverage {{ padding: 4px 9px 2px; font-size: 11px; font-weight: 750; }}
.coverage.covered {{ color: #17623b; }} .coverage.not_covered {{ color: #8b3028; }} .coverage.unknown {{ color: #535e61; }}
.meta {{ padding: 2px 9px 8px; color: #5a6669; font-size: 11px; font-variant-numeric: tabular-nums; }}
.summary {{ padding: 8px 18px; border-bottom: 1px solid #cbd2d3; background: #edf1f1; }}
.summary-row {{ display: flex; flex-wrap: wrap; gap: 5px 14px; padding: 3px 0; font-size: 12px; }}
.summary-row strong {{ min-width: 130px; }}
@media (max-width: 520px) {{ header, .toolbar {{ padding-left: 12px; padding-right: 12px; }} .grid {{ padding-left: 10px; padding-right: 10px; }} #count {{ margin-left: 0; width: 100%; }} select, input {{ min-width: min(100%, 200px); }} }}
</style>
</head>
<body>
<header><h1>PARC Generated Candidates</h1><div class="notice">RAW KINEMATIC · PHYSICS UNVERIFIED · NOT TRAINING APPROVED</div></header>
<section class="summary" aria-label="Geometric route summary">{summary_html}</section>
<section class="toolbar" aria-label="Gallery filters">
<label>Course<select id="course">{"".join(options)}</select></label>
<label>Geometric route coverage<select id="coverage"><option value="">All coverage</option><option value="covered">Covered</option><option value="not_covered">Not covered</option><option value="unknown">Unknown</option></select></label>
<label>Search<input id="search" type="search" placeholder="Host, run, or job"></label>
<output id="count" aria-live="polite"></output>
</section>
<main class="grid">{body}</main>
<script>
const courseFilter = document.getElementById('course');
const coverageFilter = document.getElementById('coverage');
const searchFilter = document.getElementById('search');
const clips = [...document.querySelectorAll('.clip')];
function updateFilters() {{
  const query = searchFilter.value.trim().toLocaleLowerCase();
  let visible = 0;
  for (const clip of clips) {{
    const show = (!courseFilter.value || clip.dataset.course === courseFilter.value)
      && (!coverageFilter.value || clip.dataset.coverage === coverageFilter.value)
      && (!query || clip.dataset.search.includes(query));
    clip.hidden = !show;
    if (!show) clip.querySelector('video')?.pause();
    if (show) visible += 1;
  }}
  document.getElementById('count').textContent = `${{visible}} / ${{clips.length}} candidates`;
}}
courseFilter.addEventListener('change', updateFilters);
coverageFilter.addEventListener('change', updateFilters);
searchFilter.addEventListener('input', updateFilters);
updateFilters();
</script>
</body>
</html>
"""


def run_batch(collection_dir: Path, output_dir: Path | None = None, *, mode: str,
              limit: int | None = None, repo: Path | None = None,
              panel_width: int = DEFAULT_PANEL_WIDTH,
              panel_height: int = DEFAULT_PANEL_HEIGHT) -> dict[str, Any]:
    if mode not in ("catalog-only", "render-existing", "render-all"):
        raise ValueError("mode must be catalog-only, render-existing, or render-all")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if panel_width < 320 or panel_height < 240 or panel_height % 2:
        raise ValueError("panel dimensions must be at least 320 x 240 with an even height")

    collection = Path(collection_dir).expanduser().resolve()
    catalog = build_catalog(collection)
    destination = (Path(output_dir).expanduser().resolve() if output_dir else collection / "gallery")
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "catalog.json", catalog)

    renderer_hash = _renderer_sha256() if mode != "catalog-only" else None
    render_candidates = catalog["candidates"] if limit is None else catalog["candidates"][:limit]
    selected = {candidate["candidate_key"] for candidate in render_candidates}
    outcomes = []
    for candidate in catalog["candidates"]:
        if mode == "catalog-only":
            outcomes.append(_outcome(candidate, "not_rendered"))
        elif candidate["candidate_key"] not in selected:
            outcomes.append(_outcome(candidate, "not_selected_by_limit"))
        else:
            outcomes.append(_outcome(candidate, "pending"))
    outcomes_path = destination / "render_outcomes.json"
    _write_json(outcomes_path, {
        "schema_version": 1,
        "mode": mode,
        "physics_verified": False,
        "training_approved": False,
        "outcomes": outcomes,
    })

    if mode != "catalog-only":
        repo_path = Path(repo).expanduser().resolve() if repo is not None else None
        index_by_key = {row["candidate_key"]: index for index, row in enumerate(outcomes)}
        for candidate in render_candidates:
            index = index_by_key[candidate["candidate_key"]]
            source = collection.joinpath(*PurePosixPath(candidate["source_path"]).parts)
            if candidate.get("source_error") or not candidate.get("source_exists"):
                outcomes[index] = _outcome(candidate, "missing_or_invalid_source",
                                           error=candidate.get("source_error"))
            else:
                try:
                    source_hash = _sha256_file(source)
                    expected_hash = candidate.get("reported_source_sha256")
                    if isinstance(expected_hash, str) and expected_hash and source_hash != expected_hash:
                        outcomes[index] = _outcome(
                            candidate, "source_hash_mismatch", source_sha256=source_hash,
                            expected_source_sha256=expected_hash,
                            error="downloaded generated.pkl does not match generation.json sha256",
                        )
                    else:
                        result = _render_candidate(
                            candidate, source, source_hash, destination, renderer_hash,
                            repo_path, panel_width, panel_height, mode == "render-all",
                        )
                        previews = sorted((destination / result["render_dir"]).glob("frame_*.png"))
                        if previews:
                            result["poster_path"] = previews[len(previews) // 2].relative_to(destination).as_posix()
                        outcomes[index] = _outcome(candidate, result.pop("status"), **result)
                except Exception as error:  # A failed clip must not stop the batch.
                    outcomes[index] = _outcome(
                        candidate, "failed", error=f"{type(error).__name__}: {error}"
                    )
            _write_json(outcomes_path, {
                "schema_version": 1,
                "mode": mode,
                "physics_verified": False,
                "training_approved": False,
                "outcomes": outcomes,
            })
            print(json.dumps({
                "candidate": candidate["candidate_key"],
                "status": outcomes[index]["status"],
            }), flush=True)

    _write_json(outcomes_path, {
        "schema_version": 1,
        "mode": mode,
        "physics_verified": False,
        "training_approved": False,
        "outcomes": outcomes,
    })
    gallery_path = destination / "index.html"
    gallery_path.write_text(_gallery_html(catalog, outcomes), encoding="utf-8")
    return {
        "collection_dir": str(collection),
        "output_dir": str(destination),
        "catalog_path": str(destination / "catalog.json"),
        "outcomes_path": str(outcomes_path),
        "gallery_path": str(gallery_path),
        "candidate_count": catalog["candidate_count"],
        "rendered": sum(row["status"] == "rendered" for row in outcomes),
        "cached": sum(row["status"] == "cached" for row in outcomes),
        "failed": sum(row["status"] in ("failed", "source_hash_mismatch",
                                          "missing_or_invalid_source") for row in outcomes),
        "mode": mode,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True,
                        help="local root containing host/run/results/job_id/generation.json")
    parser.add_argument("--output-dir", type=Path,
                        help="gallery output directory (default: COLLECTION_DIR/gallery)")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--catalog-only", action="store_true",
                       help="write the complete catalog and static gallery without rendering")
    modes.add_argument("--render-existing", action="store_true",
                       help="render available generated.pkl files and reuse verified cache entries")
    modes.add_argument("--render-all", action="store_true",
                       help="force a fresh render for every available generated.pkl file")
    parser.add_argument("--limit", type=int,
                        help="render only the first N candidates; catalog still includes all")
    parser.add_argument("--repo", type=Path,
                        help="official PARC checkout (defaults to renderer's configured checkout)")
    parser.add_argument("--panel-width", type=int, default=DEFAULT_PANEL_WIDTH)
    parser.add_argument("--panel-height", type=int, default=DEFAULT_PANEL_HEIGHT)
    args = parser.parse_args(argv)
    mode = "catalog-only" if args.catalog_only else (
        "render-all" if args.render_all else "render-existing"
    )
    try:
        result = run_batch(args.collection_dir, args.output_dir, mode=mode, limit=args.limit,
                           repo=args.repo, panel_width=args.panel_width,
                           panel_height=args.panel_height)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
