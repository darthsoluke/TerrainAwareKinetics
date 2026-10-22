"""Structural tests for deterministic static PARC platform extensions."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_parc_platform_extensions import (  # noqa: E402
    EXTENSION_VARIANTS,
    build_extension_course,
    save_extension_course,
)


EXPECTED_LEVELS = (
    (0.0, 0.12, 0.24, 0.36, 0.54, 0.72, 0.9, 1.05, 1.2),
    (0.0, 0.15, 0.3, 0.45, 0.6),
    (0.0, 0.18, 0.36, 0.54, 0.72),
)


@pytest.fixture(scope="module", params=range(len(EXTENSION_VARIANTS)))
def extension_course(request):
    return build_extension_course(device="cpu", variant=request.param)


def _path_grid(terrain, path):
    grid_float = (path[:, :2] - terrain.min_point) / terrain.dxdy
    grid = grid_float.round().to(torch.int64)
    torch.testing.assert_close(grid_float, grid.to(grid_float.dtype), rtol=0.0, atol=1e-5)
    return grid


def test_returns_official_subterrain_and_path_samples_actual_heightfield(extension_course):
    terrain, path, manifest = extension_course
    from parc.util.terrain_util import SubTerrain

    assert isinstance(terrain, SubTerrain)
    assert path.device.type == "cpu" and path.dtype == torch.float32
    assert terrain.hf.shape == (160, 140)
    assert manifest["terrain"]["grid_spacing_xy_m"] == [0.2, 0.2]
    grid = _path_grid(terrain, path)
    assert bool(((grid >= 0) & (grid < terrain.dims)).all())
    torch.testing.assert_close(path, terrain.get_xyz_point(grid), rtol=0.0, atol=0.0)
    torch.testing.assert_close(terrain.hf_maxmin[..., 0], terrain.hf, rtol=0.0, atol=0.0)
    torch.testing.assert_close(terrain.hf_maxmin[..., 1], terrain.hf, rtol=0.0, atol=0.0)


def test_route_is_four_connected_unique_and_within_length_limits(extension_course):
    terrain, path, manifest = extension_course
    grid = _path_grid(terrain, path).cpu().numpy()
    edges = np.abs(np.diff(grid, axis=0)).sum(axis=1)
    planar = manifest["course"]["planar_length_m"]
    surface = manifest["course"]["surface_length_m"]

    assert np.all(edges == 1)
    assert len({tuple(point) for point in grid.tolist()}) == len(grid)
    assert 25.0 <= planar <= 50.0
    assert planar <= surface <= 50.0
    assert manifest["course"]["route_node_count"] == len(path)
    np.testing.assert_allclose(np.diff(path[:, 2].numpy()), np.diff(terrain.hf[grid[:, 0], grid[:, 1]].numpy()))


def test_exact_height_levels_and_bounded_vertical_steps(extension_course):
    terrain, path, manifest = extension_course
    expected = EXPECTED_LEVELS[manifest["variant"]]
    actual = np.unique(terrain.hf.detach().cpu().numpy())
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    dz = np.diff(path[:, 2].detach().cpu().numpy())
    nonzero = np.abs(dz[np.abs(dz) > 1e-7])
    assert len(nonzero) > 0
    assert np.all(nonzero >= 0.12 - 1e-6)
    assert np.all(nonzero <= 0.20 + 1e-6)
    for segment in manifest["route_segments"]:
        segment_dz = dz[segment["path_node_start"]:segment["path_node_end"]]
        if segment["stage"] not in {"ascent", "descent"}:
            assert np.allclose(segment_dz, 0.0, rtol=0.0, atol=1e-6)


def test_stair_metadata_matches_route_profiles_and_minimum_width(extension_course):
    _, path, manifest = extension_course
    stairs = {segment["flight_id"]: segment for segment in manifest["route_segments"] if segment["flight_id"]}
    assert len(stairs) == manifest["course"]["stair_flight_count"]
    for flight in manifest["stair_flights"]:
        segment = stairs[flight["flight_id"]]
        assert flight["width_m"] >= 2.4
        assert flight["riser_height_m"] == pytest.approx(segment["riser_height_m"])
        assert flight["path_segments"][segment["stage"]] == {
            "path_node_start": segment["path_node_start"],
            "path_node_end": segment["path_node_end"],
        }
        heights = path[segment["path_node_start"]:segment["path_node_end"] + 1, 2].numpy()
        changes = np.diff(heights)
        riser_changes = np.flatnonzero(np.abs(changes) > 1e-7)
        assert len(riser_changes) == flight["riser_count"]
        assert np.all(np.diff(riser_changes) == flight["tread_cells"])
        np.testing.assert_allclose(
            changes[riser_changes],
            np.full(flight["riser_count"], flight["vertical_direction"] * flight["riser_height_m"]),
            rtol=0.0,
            atol=1e-6,
        )
        assert flight["start_xyz_m"] == pytest.approx(path[segment["path_node_start"]].tolist())
        assert flight["end_xyz_m"] == pytest.approx(path[segment["path_node_end"]].tolist())


def test_segments_waypoints_and_rasterization_invariants(extension_course):
    _, path, manifest = extension_course
    segments = manifest["route_segments"]
    assert segments[0]["path_node_start"] == 0
    assert segments[-1]["path_node_end"] == len(path) - 1
    for left, right in zip(segments, segments[1:]):
        assert left["path_node_end"] == right["path_node_start"]
    for segment in segments:
        assert segment["path_node_end"] > segment["path_node_start"]
        assert segment["width_m"] == pytest.approx(segment["width_cells"] * 0.2)
        assert segment["start_xyz_m"] == pytest.approx(path[segment["path_node_start"]].tolist())
        assert segment["end_xyz_m"] == pytest.approx(path[segment["path_node_end"]].tolist())

    waypoints = manifest["task_waypoints"]
    indices = [point["path_node_index"] for point in waypoints]
    assert indices == sorted(set(indices))
    assert indices[0] == 0 and indices[-1] == len(path) - 1
    assert [point["order"] for point in waypoints] == list(range(len(waypoints)))
    for waypoint in waypoints:
        assert waypoint["xyz_m"] == pytest.approx(path[waypoint["path_node_index"]].tolist())

    raster = manifest["surface_rasterization"]
    assert raster["surface_cell_count"] > 0
    assert raster["same_height_join_cell_writes"] > 0
    assert raster["height_conflict_count"] == 0
    assert manifest["geometry_only"] is True
    assert manifest["motion_generation"] == "not performed"
    assert manifest["physical_validity"] == "not assessed"


def test_s_route_keeps_parallel_legs_separated(extension_course):
    _, _, manifest = extension_course
    if manifest["variant"] != 2:
        pytest.skip("only the S-route variant has three parallel legs")
    legs = [segment for segment in manifest["route_segments"] if segment["name"].startswith("s_leg_")]
    spacing = manifest["layout_constraints"]["parallel_leg_center_spacing_m"]
    assert len(legs) == 3
    assert len(spacing) == 2 and min(spacing) >= 3.0
    assert manifest["layout_constraints"]["minimum_parallel_leg_clearance_m"] > 0.0
    assert len({leg["width_m"] for leg in legs}) == 3


def test_terrace_endpoint_has_same_height_support_beyond_route(extension_course):
    terrain, path, manifest = extension_course
    if manifest["variant"] != 0:
        pytest.skip("only the terrace variant has an elevated terminal support extension")

    support = manifest["terminal_support"]
    assert support["beyond_route_m"] >= 1.0
    assert support["height_m"] == pytest.approx(path[-1, 2].item())
    assert support["start_grid_ij"] == manifest["course"]["route_end_grid_ij"]
    assert support["end_grid_ij"][0] > support["start_grid_ij"][0]
    assert torch.equal(
        path[-1],
        terrain.get_xyz_point(torch.tensor([support["start_grid_ij"]], dtype=torch.int64))[0],
    )

    i0, j0 = support["start_grid_ij"]
    i1, _ = support["end_grid_ij"]
    width_cells = round(support["width_m"] / 0.2)
    lateral_start = j0 - width_cells // 2
    patch = terrain.hf[i0:i1 + 1, lateral_start:lateral_start + width_cells]
    assert patch.shape == (round(support["beyond_route_m"] / 0.2) + 1, width_cells)
    assert torch.allclose(patch, torch.full_like(patch, support["height_m"]))


def test_extension_uses_existing_static_save_and_preview_conventions(extension_course, tmp_path):
    terrain, path, manifest = extension_course
    outputs = save_extension_course(tmp_path, terrain, path, manifest)

    assert set(outputs) == {"terrain_and_route", "manifest", "preview"}
    assert all(file.is_file() for file in outputs.values())
    assert outputs["preview"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads(outputs["manifest"].read_text(encoding="utf-8")) == manifest
    with np.load(outputs["terrain_and_route"]) as data:
        np.testing.assert_allclose(data["path_nodes"], path.numpy(), rtol=0.0, atol=0.0)
        assert len(data["segment_names"]) == len(manifest["route_segments"])


def test_variants_are_reproducible_and_distinct(extension_course):
    terrain, path, manifest = extension_course
    repeat_terrain, repeat_path, repeat_manifest = build_extension_course(variant=manifest["variant"])
    torch.testing.assert_close(terrain.hf, repeat_terrain.hf, rtol=0.0, atol=0.0)
    torch.testing.assert_close(path, repeat_path, rtol=0.0, atol=0.0)
    assert manifest == repeat_manifest
    assert manifest["scenario_id"] == EXTENSION_VARIANTS[manifest["variant"]]["scenario_id"]
