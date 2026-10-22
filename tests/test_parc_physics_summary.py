"""Full clips, partial salvage, failures and route coverage remain distinct."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("parc_physics_summary_under_test",
    Path(__file__).resolve().parents[1] / "scripts/summarize_parc_physics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample():
    catalog = {"candidates": [dict(candidate_key=str(i), reported_source_sha256=f"hash{i}",
        course_kind="stairs", course_variant=0, frames=61, geometric_route_covered=i != 0)
        for i in range(3)]}
    report = dict(status="official_recording_finished", inputs_unchanged_after_run=True,
        inputs=[dict(candidate_key=str(i), sha256=f"hash{i}") for i in range(3)],
        saved=[dict(candidate_key=str(i), source_sha256=f"hash{i}", whole_motion=i == 0,
                    recorded_terrain_error_m=0.,
                    start_fraction=i * .5, frames=61 - i * 30, fps=30) for i in range(2)],
        failed_candidate_ids=["2"], elapsed_seconds=10, tracker_sha256="tracker",
        packed_terrain_validation=dict(grid_spacing_matches=True, all_tile_cells_checked=100,
                                       maximum_height_error_m=0.))
    return report, catalog


def test_partition_and_duration():
    result = module.summarize(*sample())
    assert (result["full_start_complete"], result["partial_only"], result["failed"]) == (1, 1, 1)
    assert result["full_motion_seconds"] == 2
    assert result["partial_motion_seconds"] == 1
    assert result["full_start_and_raw_route_covered"] == 0
    assert not result["physical_route_coverage_recomputed"]
    assert not result["training_approved"]


@pytest.mark.parametrize("change", ["incomplete", "missing", "duplicate", "hash", "partial_claim",
                                    "packed_terrain_missing", "recorded_terrain_wrong"])
def test_invalid_evidence_is_rejected(change):
    report, catalog = deepcopy(sample())
    if change == "incomplete":
        report["status"] = "initializing"
    elif change == "missing":
        report["failed_candidate_ids"] = []
    elif change == "duplicate":
        report["saved"].append(report["saved"][0])
    elif change == "hash":
        report["inputs"][0]["sha256"] = "different"
    elif change == "partial_claim":
        report["saved"][1]["whole_motion"] = True
    elif change == "packed_terrain_missing":
        del report["packed_terrain_validation"]
    else:
        report["saved"][0]["recorded_terrain_error_m"] = .2
    with pytest.raises(ValueError):
        module.summarize(report, catalog)
