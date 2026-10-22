"""Geometry, route, and determinism tests for the PARC mixed course builder."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_parc_mixed_course import (  # noqa: E402
    PARC_ASTAR_JUMP_LIMITS,
    VARIANT_PROFILES,
    OBSTACLE_BOUNDS_IJ,
    ROUTE_WIDTH_CELLS,
    build_mixed_course,
    save_mixed_course,
)


@pytest.fixture(scope="module", params=tuple(VARIANT_PROFILES))
def mixed_course(request):
    return build_mixed_course(device="cpu", variant=request.param, seed=47)


def _grid_for(terrain, path):
    grid_float = (path[:, :2] - terrain.min_point) / terrain.dxdy
    grid = grid_float.round().to(torch.int64)
    torch.testing.assert_close(grid_float, grid.to(grid_float.dtype), rtol=0.0, atol=2e-5)
    return grid


def test_all_route_nodes_are_surface_nodes_and_jump_is_a_discrete_edge(mixed_course):
    terrain, path, manifest = mixed_course
    from parc.util.terrain_util import SubTerrain

    assert isinstance(terrain, SubTerrain)
    assert terrain.hf.shape == (160, 100)
    assert path.dtype == torch.float32 and path.device.type == "cpu"
    assert manifest["terrain"]["grid_spacing_xy_m"] == [0.2, 0.2]
    grid = _grid_for(terrain, path)
    assert bool(((grid >= 0) & (grid < terrain.dims)).all())

    hf = terrain.hf[grid[:, 0], grid[:, 1]]
    torch.testing.assert_close(path[:, 2], hf, rtol=0.0, atol=1e-6)
    jump = next(s for s in manifest["route_segments"]
                if s.get("edge_type") == "official_parc_astar_jump_edge_candidate")
    start, end = jump["path_node_start"], jump["path_node_end"]
    assert end == start + 1
    assert jump["intermediate_route_target_count"] == 0
    assert jump["takeoff_node_role"] == jump["landing_node_role"] == "terrain_surface_support_candidate"
    assert path[start, 2].item() == pytest.approx(terrain.hf[grid[start, 0], grid[start, 1]].item())
    assert path[end, 2].item() == pytest.approx(terrain.hf[grid[end, 0], grid[end, 1]].item())

    start_ij, end_ij = grid[start].tolist(), grid[end].tolist()
    assert start_ij[1] == end_ij[1]
    direction = 1 if end_ij[0] > start_ij[0] else -1
    gap_cells = [(i, start_ij[1]) for i in range(start_ij[0] + direction, end_ij[0], direction)]
    assert gap_cells
    gap_heights = torch.stack([terrain.hf[i, j] for i, j in gap_cells])
    assert torch.all(gap_heights < min(path[start, 2], path[end, 2]) - 0.01)
    xy_distance = torch.linalg.vector_norm(path[end, :2] - path[start, :2]).item()
    z_difference = (path[end, 2] - path[start, 2]).item()
    assert 0.0 < xy_distance <= PARC_ASTAR_JUMP_LIMITS["max_jump_xy_dist_m"]
    assert PARC_ASTAR_JUMP_LIMITS["min_jump_z_diff_m"] <= z_difference <= PARC_ASTAR_JUMP_LIMITS["max_jump_z_diff_m"]
    assert jump["astar_jump_limits"] == PARC_ASTAR_JUMP_LIMITS
    assert torch.equal(terrain.hf_maxmin[..., 0], terrain.hf)
    assert torch.equal(terrain.hf_maxmin[..., 1], terrain.hf)
    assert manifest["validation"]["all_route_nodes_on_heightfield"]
    assert manifest["validation"]["jump_gap_between_support_nodes_checked"]


def test_route_is_bounded_unique_and_only_jump_edge_skips_grid_cells(mixed_course):
    terrain, path, manifest = mixed_course
    grid = _grid_for(terrain, path).numpy()
    edge_lengths = np.abs(np.diff(grid, axis=0)).sum(axis=1)
    jump = next(s for s in manifest["route_segments"]
                if s.get("edge_type") == "official_parc_astar_jump_edge_candidate")
    jump_edge_index = jump["path_node_start"]
    assert edge_lengths[jump_edge_index] > 1
    assert np.all(np.delete(edge_lengths, jump_edge_index) == 1)
    assert len({tuple(point) for point in grid.tolist()}) == len(grid)
    assert 25.0 <= manifest["course"]["planar_length_m"] <= 35.0
    assert manifest["course"]["route_node_count"] == len(path)
    segments = manifest["route_segments"]
    assert segments[0]["path_node_start"] == 0
    assert segments[-1]["path_node_end"] == len(path) - 1
    for left, right in zip(segments, segments[1:]):
        assert left["path_node_end"] == right["path_node_start"]
    for segment in segments:
        assert segment["path_node_end"] >= segment["path_node_start"] + 1
        assert segment["start_xyz_m"] == pytest.approx(path[segment["path_node_start"]].tolist())
        assert segment["end_xyz_m"] == pytest.approx(path[segment["path_node_end"]].tolist())


def test_obstacle_blocks_direct_lane_and_route_uses_clear_detour(mixed_course):
    terrain, path, manifest = mixed_course
    grid = _grid_for(terrain, path).numpy()
    i0, i1, j0, j1 = OBSTACLE_BOUNDS_IJ
    assert torch.all(terrain.hf[i0:i1 + 1, j0:j1 + 1] > 0)

    route_width_radius = ROUTE_WIDTH_CELLS // 2
    for i, j in grid.tolist():
        route_i0, route_i1 = i - route_width_radius, i + route_width_radius - 1
        route_j0, route_j1 = j - route_width_radius, j + route_width_radius - 1
        assert route_i1 < i0 or route_i0 > i1 or route_j1 < j0 or route_j0 > j1
    assert manifest["features"][0]["geometry_limit"].startswith("single-valued heightfield")
    assert "obstacle_detour" in [segment["stage"] for segment in manifest["route_segments"]]


def test_course_contains_all_requested_terrain_families_without_claiming_success(mixed_course):
    _, _, manifest = mixed_course
    stages = {segment["stage"] for segment in manifest["route_segments"]}
    assert {"walk", "obstacle_detour", "jump_candidate", "stairs_up", "stairs_down",
            "platform_walk", "rough_rock"} <= stages
    assert ("climb_candidate" if manifest["variant"] == 2 else "stepped_ledge") in stages
    assert manifest["geometry_only"] is True
    assert manifest["motion_generation"] == "not performed"
    assert manifest["physical_validity"] == "not assessed"
    assert manifest["behavioral_success"] == "not assessed"
    assert any("not an unsupported open void" in text for text in manifest["limitations"])
    assert manifest["jump_edge_contract"]["intermediate_route_or_pose_targets"] == 0
    if manifest["variant"] == 2:
        assert "low_ledge_ascent_candidate" not in manifest["course"]["requested_behavior_families"]
        assert "high_ledge_climb_candidate" in manifest["course"]["requested_behavior_families"]


def test_official_parc_astar_creates_course_jump_edge_between_gap_support_nodes(mixed_course):
    import yaml

    from build_parc_stair_course import _load_official_terrain_util
    terrain, path, manifest = mixed_course
    config_path = ROOT / "out" / "vendor" / "PARC" / "data" / "configs" / "parc_2_kin_gen_default.yaml"
    astar_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["astar"]
    limits = {
        "max_jump_xy_dist_m": astar_config["max_jump_xy_dist"],
        "max_jump_z_diff_m": astar_config["max_jump_z_diff"],
        "min_jump_z_diff_m": astar_config["min_jump_z_diff"],
        "source": PARC_ASTAR_JUMP_LIMITS["source"],
    }
    assert limits == PARC_ASTAR_JUMP_LIMITS

    terrain_util = _load_official_terrain_util()
    from parc.motion_synthesis.procgen.astar import construct_navigation_graph

    jump = next(segment for segment in manifest["route_segments"]
                if segment.get("edge_type") == "official_parc_astar_jump_edge_candidate")
    start_ij = np.asarray(jump["start_grid_ij"], dtype=np.int64)
    end_ij = np.asarray(jump["end_grid_ij"], dtype=np.int64)
    crop_i0, crop_i1 = int(start_ij[0] - 4), int(end_ij[0] + 5)
    crop_j0, crop_j1 = int(start_ij[1] - 8), int(start_ij[1] + 9)
    crop_hf = terrain.hf[crop_i0:crop_i1, crop_j0:crop_j1]
    local_start = tuple((start_ij - (crop_i0, crop_j0)).tolist())
    local_end = tuple((end_ij - (crop_i0, crop_j0)).tolist())
    cropped_terrain = terrain_util.SubTerrain(
        "astar_course_jump_edge_test", x_dim=crop_hf.shape[0], y_dim=crop_hf.shape[1], dx=0.2, dy=0.2,
        min_x=0.0, min_y=0.0, device="cpu",
    )
    cropped_terrain.hf.copy_(crop_hf)
    cropped_terrain.hf_maxmin[..., 0] = cropped_terrain.hf
    cropped_terrain.hf_maxmin[..., 1] = cropped_terrain.hf
    navigation = construct_navigation_graph(
        cropped_terrain,
        max_z_diff=astar_config["max_z_diff"],
        max_jump_xy_dist=limits["max_jump_xy_dist_m"],
        max_jump_z_diff=limits["max_jump_z_diff_m"],
        min_jump_z_diff=limits["min_jump_z_diff_m"],
    )
    assert navigation[local_start[0]][local_start[1]].pos[2] == pytest.approx(path[jump["path_node_start"], 2].item())
    assert navigation[local_end[0]][local_end[1]].pos[2] == pytest.approx(path[jump["path_node_end"], 2].item())
    assert (local_end[0], local_end[1]) in navigation[local_start[0]][local_start[1]].edges
    gap_heights = crop_hf[local_start[0] + 1:local_end[0], local_start[1]]
    support_floor = min(
        navigation[local_start[0]][local_start[1]].pos[2],
        navigation[local_end[0]][local_end[1]].pos[2],
    )
    assert gap_heights.max().item() < support_floor


def test_stair_and_ledge_profiles_are_bounded_and_join_exactly(mixed_course):
    _, path, manifest = mixed_course
    ledge_name = "high_ledge_climb_candidate" if manifest["variant"] == 2 else "stepped_ledge_ascent"
    for name in ("stairs_up", ledge_name, "stairs_down"):
        segment = next(item for item in manifest["route_segments"] if item["name"] == name)
        section = path[segment["path_node_start"]:segment["path_node_end"] + 1, 2].numpy()
        changes = np.diff(section)
        nonzero = changes[np.abs(changes) > 1e-7]
        assert len(nonzero) == segment["riser_count"]
        assert np.max(np.abs(nonzero)) <= 0.70 + 1e-6
        assert np.all(nonzero > 0) if name != "stairs_down" else np.all(nonzero < 0)
    ledge = next(feature for feature in manifest["features"] if feature["feature_id"] == "ledge_1")
    assert ledge["overhang"] is False


def test_climb_challenge_uses_high_ledge_and_narrow_gap_as_unverified_candidates():
    _, path, manifest = build_mixed_course(variant=2, seed=20261009)
    jump = next(segment for segment in manifest["route_segments"]
                if segment.get("edge_type") == "official_parc_astar_jump_edge_candidate")
    ledge = next(segment for segment in manifest["route_segments"]
                 if segment["stage"] == "climb_candidate")
    feature = next(item for item in manifest["features"] if item["feature_id"] == "ledge_1")

    assert jump["planar_length_m"] == pytest.approx(2.8)
    assert jump["planar_length_m"] <= PARC_ASTAR_JUMP_LIMITS["max_jump_xy_dist_m"]
    assert jump["gap_terrain_sample_count"] == 13
    assert ledge["riser_count"] == 1
    assert ledge["riser_height_m"] == pytest.approx(0.65)
    assert ledge["requested_behavior"] == "high_ledge_climb_candidate"
    assert feature["kind"] == "high_ledge_climb_candidate"
    assert feature["behavior_candidate_only"] is True
    assert feature["hand_contact_evidence"] == "not observed by geometry builder"
    assert "force-derived hand-contact evidence" in feature["promotion_requirement"]
    assert manifest["behavioral_success"] == "not assessed"
    assert 32.0 <= manifest["course"]["planar_length_m"] <= 35.0
    assert torch.isfinite(path).all()


def test_easy_and_medium_keep_low_ledge_curriculum():
    assert VARIANT_PROFILES[0]["ledge_riser_m"] == pytest.approx(0.14)
    assert VARIANT_PROFILES[1]["ledge_riser_m"] == pytest.approx(0.18)
    assert VARIANT_PROFILES[0]["jump_span_cells"] == VARIANT_PROFILES[1]["jump_span_cells"] == 12
    for variant in (0, 1):
        _, _, manifest = build_mixed_course(variant=variant, seed=20261009)
        rock_field = next(feature for feature in manifest["features"]
                          if feature["feature_id"] == "rocks_1")
        assert rock_field["bounds_grid_ij_inclusive"] == [92, 112, 12, 28]
        assert "low_ledge_ascent_candidate" in manifest["course"]["requested_behavior_families"]


def test_seed_reproducibility_and_terrain_diversity():
    first = build_mixed_course(variant=1, seed=123)
    repeat = build_mixed_course(variant=1, seed=123)
    varied = build_mixed_course(variant=1, seed=124)
    torch.testing.assert_close(first[0].hf, repeat[0].hf, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first[1], repeat[1], rtol=0.0, atol=0.0)
    assert first[2] == repeat[2]
    assert not torch.equal(first[0].hf, varied[0].hf)
    assert first[2]["scenario_id"] != varied[2]["scenario_id"]


def test_difficulty_is_bounded_and_changes_geometry():
    easy = build_mixed_course(variant=0, seed=9)
    medium = build_mixed_course(variant=1, seed=9)
    assert easy[2]["course"]["planar_length_m"] == pytest.approx(medium[2]["course"]["planar_length_m"])
    assert easy[2]["terrain"]["height_range_m"][1] < medium[2]["terrain"]["height_range_m"][1]
    assert easy[2]["features"][0]["top_height_m"] < medium[2]["features"][0]["top_height_m"]
    assert easy[2]["features"][3]["height_range_m"] != medium[2]["features"][3]["height_range_m"]


def test_existing_static_save_contract_round_trips(tmp_path, mixed_course):
    terrain, path, manifest = mixed_course
    outputs = save_mixed_course(tmp_path, terrain, path, manifest)
    assert set(outputs) == {"terrain_and_route", "manifest", "preview"}
    assert all(file.is_file() for file in outputs.values())
    assert outputs["preview"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads(outputs["manifest"].read_text(encoding="utf-8")) == manifest
    with np.load(outputs["terrain_and_route"]) as data:
        np.testing.assert_allclose(data["path_nodes"], path.numpy(), rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(data["path_grid_ij"], _grid_for(terrain, path).numpy())


@pytest.mark.parametrize("variant", (True, -1, 3, "easy"))
def test_invalid_variant_is_rejected(variant):
    with pytest.raises(ValueError, match="variant"):
        build_mixed_course(variant=variant)


@pytest.mark.parametrize("seed", (True, -1, 2**32, 1.5))
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        build_mixed_course(seed=seed)
