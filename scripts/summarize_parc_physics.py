"""Count saved official physics outputs, never the recorder's cumulative flags."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(report, catalog):
    if (report.get("status") != "official_recording_finished" or
            not report.get("inputs_unchanged_after_run")):
        raise ValueError("An incomplete or mutated-input run cannot be summarized as complete")
    terrain = report.get("packed_terrain_validation", {})
    if (not terrain.get("grid_spacing_matches") or not terrain.get("all_tile_cells_checked")
            or terrain.get("maximum_height_error_m", float("inf")) > 1e-5):
        raise ValueError("Terrain/collision pairing has not been validated")
    sources = {row["candidate_key"]: row for row in catalog["candidates"]}
    inputs = {row["candidate_key"]: row for row in report["inputs"]}
    if len(inputs) != len(report["inputs"]) or set(inputs) != set(sources):
        raise ValueError("Full batch candidate identities disagree")
    saved = {row["candidate_key"]: row for row in report["saved"]}
    if len(saved) != len(report["saved"]) or not set(saved) <= set(inputs):
        raise ValueError("Duplicate or unknown physical recording")
    if set(report["failed_candidate_ids"]) != set(inputs) - set(saved):
        raise ValueError("Saved and failed candidates do not partition inputs")
    groups = defaultdict(lambda: {"candidates": 0, "full_start_complete": 0,
                                 "partial_only": 0, "failed": 0,
                                 "full_motion_seconds": 0., "partial_motion_seconds": 0.,
                                 "full_start_and_raw_route_covered": 0})
    for key, source in sources.items():
        if inputs[key]["sha256"] != source["reported_source_sha256"]:
            raise ValueError("Raw provenance hash mismatch")
        group = groups[(source["course_kind"], source["course_variant"])]
        group["candidates"] += 1
        if key not in saved:
            group["failed"] += 1
            continue
        recording = saved[key]
        if recording.get("recorded_terrain_error_m", float("inf")) > 1e-5:
            raise ValueError("Recorded terrain has not been validated in world coordinates")
        if recording["source_sha256"] != inputs[key]["sha256"]:
            raise ValueError("Physical recording/raw provenance mismatch")
        full = recording["whole_motion"]
        if full and (recording["start_fraction"] != 0 or recording["frames"] != source["frames"]):
            raise ValueError("Invalid full-motion success claim")
        seconds = (recording["frames"] - 1) / recording["fps"]
        group["full_start_complete" if full else "partial_only"] += 1
        group["full_motion_seconds" if full else "partial_motion_seconds"] += seconds
        group["full_start_and_raw_route_covered"] += int(full and source["geometric_route_covered"])
    result = {"status": "official_physics_recording_completed_not_training_approved",
              "groups": [dict(kind=key[0], variant=key[1], **value)
                         for key, value in sorted(groups.items())],
              "elapsed_seconds": report["elapsed_seconds"], "training_approved": False,
              "physical_route_coverage_recomputed": False,
              "note": "The raw route-coverage column describes the reference only, not the simulated trajectory.",
              "tracker_sha256": report["tracker_sha256"]}
    for field in next(iter(groups.values())):
        result[field] = sum(group[field] for group in groups.values())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text()), json.loads(args.catalog.read_text()))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
