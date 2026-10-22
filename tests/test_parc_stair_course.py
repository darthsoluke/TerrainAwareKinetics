"""Tests for deterministic static PARC stair-course geometry."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_parc_stair_course as builder
from build_parc_stair_course import build_course, build_platform_course, save_course


@pytest.fixture(scope="module")
def course():
    return build_course(device="cpu")


def test_uses_official_subterrain_and_verified_xyz_path_contract(course):
    terrain, path, manifest = course

    from parc.util.terrain_util import SubTerrain

    assert isinstance(terrain, SubTerrain)
    assert path.dtype == torch.float32
    assert path.device.type == "cpu"
    assert path.ndim == 2 and path.shape[1] == 3
    assert path.shape[0] == manifest["course"]["route_node_count"]
    assert terrain.hf.shape == (160, 140)

    route_grid = torch.round((path[:, :2] - terrain.min_point) / terrain.dxdy).to(torch.int64)
    reconstructed = terrain.get_xyz_point(route_grid)
    torch.testing.assert_close(path, reconstructed, rtol=0.0, atol=1e-6)
    assert bool((route_grid[:, 0] >= 0).all() and (route_grid[:, 0] < terrain.hf.shape[0]).all())
    assert bool((route_grid[:, 1] >= 0).all() and (route_grid[:, 1] < terrain.hf.shape[1]).all())


def test_course_is_four_connected_non_crossing_and_starts_ends_flat(course):
    terrain, path, manifest = course
    grid = torch.round((path[:, :2] - terrain.min_point) / terrain.dxdy).to(torch.int64).numpy()
    edges = np.abs(np.diff(grid, axis=0)).sum(axis=1)

    assert np.all(edges == 1)
    assert len({tuple(point) for point in grid.tolist()}) == len(grid)
    assert path[0, 2].item() == pytest.approx(0.0)
    assert path[-1, 2].item() == pytest.approx(0.0)
    assert path[1, 2].item() == pytest.approx(0.0)
    assert path[-2, 2].item() == pytest.approx(0.0)
    assert terrain.hf[grid[0, 0], grid[0, 1]].item() == pytest.approx(0.0)
    assert terrain.hf[grid[-1, 0], grid[-1, 1]].item() == pytest.approx(0.0)
    assert 30.0 <= manifest["course"]["planar_length_m"] <= 50.0
    assert manifest["course"]["surface_length_m"] <= 50.0


def test_three_spaced_flights_have_exact_risers_treads_and_flat_landings(course):
    _, path, manifest = course
    flights = manifest["stair_flights"]
    assert [flight["riser_count"] for flight in flights] == [3, 4, 5]
    assert [flight["riser_height_m"] for flight in flights] == pytest.approx([0.12, 0.18, 0.24])
    assert [flight["tread_depth_m"] for flight in flights] == pytest.approx([0.4, 0.4, 0.6])
    assert all(flight["width_m"] >= 2.0 for flight in flights)
    assert [flight["high_platform_length_m"] for flight in flights] == pytest.approx([1.2] * 3)
    assert len({tuple(flight["start_grid_ij"]) for flight in flights}) == 3
    centers = np.asarray([flight["start_grid_ij"] for flight in flights], dtype=np.float64)
    assert np.min(np.linalg.norm(np.diff(centers, axis=0), axis=1)) * 0.2 >= 5.0

    for flight in flights:
        ascent = flight["path_segments"]["ascent"]
        platform = flight["path_segments"]["platform"]
        descent = flight["path_segments"]["descent"]
        up_z = path[ascent["path_node_start"]:ascent["path_node_end"] + 1, 2].numpy()
        flat_z = path[platform["path_node_start"]:platform["path_node_end"] + 1, 2].numpy()
        down_z = path[descent["path_node_start"]:descent["path_node_end"] + 1, 2].numpy()
        up_dz, down_dz = np.diff(up_z), np.diff(down_z)
        riser = flight["riser_height_m"]

        assert np.count_nonzero(up_dz > 0.0) == flight["riser_count"]
        assert np.allclose(up_dz[up_dz > 0.0], riser, atol=1e-6)
        assert np.all(np.diff(np.flatnonzero(up_dz > 0.0)) == flight["tread_cells"])
        assert np.allclose(flat_z, flight["total_rise_m"], atol=1e-6)
        assert platform["path_node_end"] - platform["path_node_start"] == 6
        assert np.count_nonzero(down_dz < 0.0) == flight["riser_count"]
        assert np.allclose(-down_dz[down_dz < 0.0], riser, atol=1e-6)
        assert np.all(np.diff(np.flatnonzero(down_dz < 0.0)) == flight["tread_cells"])
        assert flight["width_m"] == pytest.approx(2.4)


def test_every_vertical_route_change_is_a_designated_stair_riser(course):
    _, path, manifest = course
    z = path[:, 2].numpy()
    dz = np.diff(z)
    allowed = {0.0, 0.12, -0.12, 0.18, -0.18, 0.24, -0.24}

    assert all(any(value == pytest.approx(candidate, abs=1e-6) for candidate in allowed) for value in dz)
    for segment in manifest["route_segments"]:
        segment_dz = dz[segment["path_node_start"]:segment["path_node_end"]]
        if segment["stage"] not in {"ascent", "descent"}:
            assert np.allclose(segment_dz, 0.0, atol=1e-6)
    assert manifest["motion_generation"] == "not performed"
    assert manifest["geometry_only"] is True


def test_course_is_reproducible(course):
    terrain_a, path_a, manifest_a = course
    terrain_b, path_b, manifest_b = build_course(device="cpu")

    torch.testing.assert_close(terrain_a.hf, terrain_b.hf, rtol=0.0, atol=0.0)
    torch.testing.assert_close(path_a, path_b, rtol=0.0, atol=0.0)
    assert manifest_a == manifest_b


def test_saves_npz_manifest_and_static_preview(course, tmp_path):
    terrain, path, manifest = course
    outputs = save_course(tmp_path, terrain, path, manifest)

    assert outputs["terrain_and_route"].name == "terrain_and_route.npz"
    assert outputs["manifest"].name == "course_manifest.json"
    assert outputs["preview"].name == "course_preview.png"
    assert all(output.is_file() for output in outputs.values())
    assert outputs["preview"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    saved_manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert saved_manifest == manifest
    with np.load(outputs["terrain_and_route"]) as saved:
        np.testing.assert_allclose(saved["hf"], terrain.hf.numpy())
        np.testing.assert_allclose(saved["path_nodes"], path.numpy())
        np.testing.assert_array_equal(saved["path_grid_ij"].shape, (len(path), 2))
        assert len(saved["segment_names"]) == len(manifest["route_segments"])


@pytest.fixture(scope="module", params=range(3))
def platform_course(request):
    return build_platform_course(device="cpu", variant=request.param)


def test_platform_route_is_on_discrete_top_and_inside_wide_rectangle(platform_course):
    terrain, path, manifest = platform_course
    from parc.util.terrain_util import SubTerrain

    assert isinstance(terrain, SubTerrain)
    assert path.dtype == torch.float32 and path.device.type == "cpu"
    grid_float = (path[:, :2] - terrain.min_point) / terrain.dxdy
    grid = grid_float.round().to(torch.int64)
    torch.testing.assert_close(grid_float, grid.float(), rtol=0, atol=1e-5)
    assert bool(((grid >= 0) & (grid < terrain.dims)).all())
    torch.testing.assert_close(path, terrain.get_xyz_point(grid), rtol=0, atol=0)
    assert torch.all(torch.abs(torch.diff(grid, dim=0)).sum(dim=1) == 1)
    assert len(torch.unique(grid, dim=0)) == len(grid)

    variant = manifest["variant"]
    platform = manifest["platforms"][0]
    expected_height = (0.48, 0.72, 0.96)[variant]
    expected_width = (4.8, 6.4, 8.0)[variant]
    assert platform["height_m"] == pytest.approx(expected_height)
    assert platform["width_m"] == pytest.approx(expected_width)
    i0, j0 = platform["grid_min_ij"]
    i1, j1 = platform["grid_max_ij_inclusive"]
    assert (j1 - j0 + 1) * 0.2 == pytest.approx(expected_width)
    assert (i1 - i0 + 1) * 0.2 == pytest.approx(platform["length_m"])
    assert bool(torch.allclose(terrain.hf[i0:i1 + 1, j0:j1 + 1], torch.tensor(expected_height)))

    start, end = platform["path_node_start"], platform["path_node_end"]
    top_path, top_grid = path[start:end + 1], grid[start:end + 1]
    assert torch.allclose(top_path[:, 2], torch.tensor(expected_height))
    assert bool(((top_grid >= torch.tensor([i0, j0])) & (top_grid <= torch.tensor([i1, j1]))).all())
    top_distance = torch.linalg.vector_norm(torch.diff(top_path[:, :2], dim=0), dim=1).sum().item()
    assert top_distance >= 12.0
    assert top_distance == pytest.approx(manifest["course"]["platform_route_length_m"])
    assert torch.count_nonzero(path[:6, 2]) == 0 and torch.count_nonzero(path[-6:, 2]) == 0
    assert 25.0 <= manifest["course"]["planar_length_m"] <= 45.0
    assert manifest["course"]["surface_length_m"] <= 45.0
    torch.testing.assert_close(terrain.hf_maxmin[..., 0], terrain.hf)
    torch.testing.assert_close(terrain.hf_maxmin[..., 1], terrain.hf)


def test_platform_stairs_and_summit_metadata_match_passage_consumer(platform_course):
    terrain, path, manifest = platform_course
    from generate_parc_long_stairs import passage_evidence

    assert len(manifest["stair_flights"]) == 1
    flight = manifest["stair_flights"][0]
    assert flight["orientation"] == "x" and flight["axis"] == 0 and flight["direction"] == 1
    assert flight["ascent_end_xyz_m"][0] < flight["platform_end_xyz_m"][0]
    for stage, direction in (("ascent", 1), ("descent", -1)):
        span = flight["path_segments"][stage]
        points = path[span["path_node_start"]:span["path_node_end"] + 1]
        delta = torch.diff(points, dim=0)
        assert torch.all(delta[:, 0] > 0) and torch.count_nonzero(delta[:, 1]) == 0
        dz = delta[:, 2]
        changes = dz[torch.abs(dz) > 1e-6]
        assert len(changes) == 4
        torch.testing.assert_close(changes, torch.full_like(changes, direction * flight["riser_height_m"]))
    for segment in manifest["route_segments"]:
        if segment["stage"] not in {"ascent", "descent"}:
            heights = path[segment["path_node_start"]:segment["path_node_end"] + 1, 2]
            assert torch.count_nonzero(torch.diff(heights)) == 0

    feet = path.clone()
    feet[:, 2] += 0.08
    assert passage_evidence(feet, terrain, manifest)["all_summits_visited_in_order"]
    assert not passage_evidence(feet[[0, -1]], terrain, manifest)["all_summits_visited_in_order"]


def test_platform_waypoints_capture_ordered_corners_and_wide_u_turns(platform_course):
    terrain, path, manifest = platform_course
    points = manifest["task_waypoints"]
    indices = [point["path_node_index"] for point in points]
    assert indices == sorted(set(indices))
    assert len({point["waypoint_id"] for point in points}) == len(points)
    assert len({tuple(point["grid_ij"]) for point in points}) == len(points)
    assert [point["order"] for point in points] == list(range(len(points)))
    for point in points:
        xyz = path[point["path_node_index"]].numpy()
        np.testing.assert_array_equal(point["xyz_m"], xyz)
        np.testing.assert_array_equal(point["xy_m"], xyz[:2])
        assert point["height_m"] == xyz[2]

    grid = torch.round((path[:, :2] - terrain.min_point) / terrain.dxdy).numpy().astype(int)
    headings = np.diff(grid, axis=0)
    corners = np.flatnonzero(np.any(headings[1:] != headings[:-1], axis=1)) + 1
    declared = [point for point in points if point["stage"] == "platform_turn"]
    assert [point["path_node_index"] for point in declared] == corners.tolist()
    assert len(declared) == 4
    assert all(abs(point["signed_turn_degrees"]) == 90 for point in declared)
    assert not np.any(np.sum(headings[1:] * headings[:-1], axis=1) == -1)
    assert np.any(headings[:, 0] < 0) and np.any(headings[:, 0] > 0)
    by_id = {point["waypoint_id"]: point for point in points}
    assert len(manifest["platform_turns"]) == 2
    for turn in manifest["platform_turns"]:
        pair = [by_id[key] for key in turn["waypoint_ids"]]
        assert sum(point["signed_turn_degrees"] for point in pair) == turn["signed_turn_degrees"]
        assert abs(turn["signed_turn_degrees"]) == 180
        assert pair[1]["path_node_index"] - pair[0]["path_node_index"] >= 6


def test_platform_variants_are_reproducible_and_identifiably_distinct(platform_course):
    terrain, path, manifest = platform_course
    second_terrain, second_path, second_manifest = build_platform_course(variant=manifest["variant"])
    assert torch.equal(terrain.hf, second_terrain.hf)
    assert torch.equal(path, second_path)
    assert manifest == second_manifest
    assert len({spec["scenario_id"] for spec in builder.PLATFORM_VARIANTS}) == 3
    assert len({spec["width_m"] for spec in builder.PLATFORM_VARIANTS}) == 3
    assert manifest["scenario_id"] == terrain.terrain_name
    assert manifest["course_kind"] == "platform" and manifest["geometry_only"]
    assert manifest["motion_generation"] == "not performed"


def test_platform_cli_serializes_route_tasks_and_preview(platform_course, tmp_path, capsys):
    terrain, path, manifest = platform_course
    builder.main(["--kind", "platform", "--variant", str(manifest["variant"]), "--output-dir", str(tmp_path)])
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "geometry_only"
    assert json.loads((tmp_path / "course_manifest.json").read_text()) == manifest
    with np.load(tmp_path / "terrain_and_route.npz") as data:
        np.testing.assert_array_equal(data["hf"], terrain.hf.numpy())
        np.testing.assert_array_equal(data["path_nodes"], path.numpy())
        for point in manifest["task_waypoints"]:
            np.testing.assert_array_equal(data["path_nodes"][point["path_node_index"]], point["xyz_m"])
    import matplotlib.image as mpimg

    pixels = mpimg.imread(tmp_path / "course_preview.png")
    assert pixels.shape[0] >= 800 and pixels.shape[1] >= 1500
    assert pixels[..., :3].std() > 0.03


@pytest.mark.parametrize("variant", [-1, 3, 0.5, "0", True])
def test_platform_rejects_invalid_variants(variant):
    with pytest.raises(ValueError, match="variant"):
        build_platform_course(variant=variant)


def test_cli_defaults_keep_stairs_and_platform_outputs_separate(monkeypatch, capsys):
    destinations = []

    def capture_save(output_dir, terrain, path, manifest):
        destinations.append((output_dir, manifest))
        return {"manifest": output_dir / "course_manifest.json"}

    monkeypatch.setattr(builder, "save_course", capture_save)
    builder.main([])
    builder.main(["--kind", "platform", "--variant", "2"])
    assert destinations[0][0] == builder.DEFAULT_OUTPUT
    assert destinations[0][1]["artifact_kind"] == "static_parc_stair_course_geometry"
    assert destinations[1][0] == builder.DEFAULT_PLATFORM_OUTPUT / destinations[1][1]["scenario_id"]
    assert destinations[0][0] != destinations[1][0]
    for args in (["--variant", "3"], ["--kind", "stairs", "--variant", "1"]):
        with pytest.raises(SystemExit) as error:
            builder.main(args)
        assert error.value.code == 2
