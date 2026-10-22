"""Verify full-length videos and expose a small, traceable PARC showcase."""

import argparse
import collections
import json
from pathlib import Path
import shutil

import imageio_ffmpeg

import deploy_parc_remote_batch as deploy


def summarize(output):
    output = output.resolve()
    progress = json.loads((output / "completion_progress.json").read_text())
    if progress.get("state") != "generated_and_rendered":
        raise ValueError("Generation and rendering have not finished")
    gallery = output / "gallery"
    catalog = json.loads((gallery / "catalog.json").read_text())
    outcomes = json.loads((gallery / "render_outcomes.json").read_text())["outcomes"]
    expected = progress["total_candidates"]
    if len(catalog["candidates"]) != expected or len(outcomes) != expected:
        raise ValueError("Catalog/outcome counts do not match the completed batch")
    candidate_keys = [row["candidate_key"] for row in catalog["candidates"]]
    outcome_keys = [row["candidate_key"] for row in outcomes]
    if (len(set(candidate_keys)) != expected or len(set(outcome_keys)) != expected
            or set(candidate_keys) != set(outcome_keys)):
        raise ValueError("Catalog/outcome candidate identities must be unique and identical")
    outcomes = {row["candidate_key"]: row for row in outcomes}
    groups = collections.defaultdict(list)
    verified = []
    for row in catalog["candidates"]:
        outcome = outcomes[row["candidate_key"]]
        if outcome["status"] not in ("rendered", "cached"):
            raise ValueError(f"Candidate was not rendered: {row['candidate_key']}")
        directory = gallery / outcome["render_dir"]
        manifest = json.loads((directory / "manifest.json").read_text())
        video = gallery / outcome["video_path"]
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
        if (frames != manifest["frames"] or frames != row["frames"]
                or abs(seconds - frames / manifest["source_fps"]) > 1 / manifest["source_fps"]):
            raise ValueError(f"Video does not preserve full source timing: {video}")
        if manifest["source_motion_sha256"] != row["reported_source_sha256"]:
            raise ValueError("Render/source hash mismatch")
        source = output / "downloads" / row["source_path"]
        if deploy.digest(source) != row["reported_source_sha256"]:
            raise ValueError("Source file changed after rendering")
        if manifest["pose_modification"] != "none" or manifest["mj_step_called"] or manifest["physics_verified"]:
            raise ValueError("Unexpected motion alteration or physics claim")
        record = {"candidate_key": row["candidate_key"], "frames": frames, "fps": manifest["source_fps"],
                  "video_sha256": deploy.digest(video), "video": str(video),
                  "duration_s": frames / manifest["source_fps"],
                  "source_sha256": row["reported_source_sha256"]}
        verified.append(record)
        groups[(row["course_kind"], row["course_variant"])].append((row, outcome, record))
    names = {("extension", 0): "terraces", ("extension", 1): "bridges",
             ("extension", 2): "wide_s", ("stairs", 0): "three_stair_flights"}
    showcase = output / "showcase"
    showcase.mkdir(exist_ok=True)
    summary_groups, selections = [], []
    for (kind, variant), rows in sorted(groups.items()):
        summary_groups.append({"kind": kind, "variant": variant, "candidates": len(rows),
            "endpoint_reached": sum(bool(r[0]["endpoint_reached"]) for r in rows),
            "geometric_route_covered": sum(bool(r[0]["geometric_route_covered"]) for r in rows),
            "motion_time_span_seconds": sum(r[0]["duration_s"] for r in rows)})
        # This is a representative gallery shortcut, never an acceptance gate.
        def selection_key(row):
            candidate = row[0]
            entry = candidate["report_entry"]
            segments = entry.get("passage", {}).get("segments", [])
            return (not candidate["geometric_route_covered"],
                    -sum(bool(s["covered_in_order"]) for s in segments), entry["selection_score"])
        row, outcome, record = min(rows, key=selection_key)
        short_name = names.get((kind, variant), f"{kind}_{variant}")
        target = showcase / (short_name + ".mp4")
        if target.exists() and deploy.digest(target) != record["video_sha256"]:
            raise ValueError(f"Refusing to replace a different showcase video: {target}")
        if not target.exists():
            shutil.copyfile(record["video"], target)
        selection = {"name": short_name, "candidate_key": row["candidate_key"],
                     "video": str(target), "video_sha256": record["video_sha256"],
                     "geometric_route_covered": row["geometric_route_covered"],
                     "physics_verified": False, "training_approved": False}
        if outcome.get("poster_path"):
            poster = showcase / (short_name + ".png")
            shutil.copyfile(gallery / outcome["poster_path"], poster)
            selection["poster"] = str(poster)
        selections.append(selection)
    summary = {"status": "raw_generation_and_full_video_delivery_complete", "candidates": expected,
        "videos_decoded_and_frame_count_verified": len(verified),
        "source_files_sha256_unique": len({r["source_sha256"] for r in verified}),
        "motion_time_span_seconds": sum(r["duration_s"] for r in catalog["candidates"]),
        "video_stream_duration_seconds": sum(r["duration_s"] for r in verified),
        "groups": summary_groups, "showcase": selections, "gallery": str(gallery / "index.html"),
        "showcase_selection": "Prefer full geometric route coverage, then segment coverage, then official selection score; all candidates remain in the gallery",
        "physics_verified": False, "training_approved": False,
        "not_done": ["Official physics tracking", "SOMA retarget admission", "MB training"],
        "verified_videos": verified}
    deploy.write_json(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "verified_videos"}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=deploy.ROOT / "out/parc_complete_20261008")
    summarize(parser.parse_args().output)
