"""Build deterministic PARC mixed-terrain courses (geometry and route only)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "out" / "parc_mixed_courses"
GRID_SPACING_M = 0.2
TERRAIN_SHAPE_IJ = (160, 100)
ROUTE_WIDTH_CELLS = 16
PARC_ASTAR_JUMP_LIMITS = {
    "max_jump_xy_dist_m": 3.0,
    "max_jump_z_diff_m": 0.5,
    "min_jump_z_diff_m": -1.0,
    "source": "PARC/data/configs/parc_2_kin_gen_default.yaml:astar",
}

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_parc_stair_course as stair_course_builder


VARIANT_PROFILES = {
    0: {
        "name": "easy",
        "stair_riser_m": 0.12,
        "ledge_riser_m": 0.14,
        "ledge_riser_count": 2,
        "ledge_tread_cells": 4,
        "ledge_behavior": "low_ledge_ascent_candidate",
        "trench_depth_m": 0.36,
        "jump_span_cells": 12,
        "downstream_shift_cells": 0,
        "obstacle_height_m": 0.65,
        "rock_step_m": 0.05,
        "rock_tile_cells": (4, 4),
    },
    1: {
        "name": "medium",
        "stair_riser_m": 0.15,
        "ledge_riser_m": 0.18,
        "ledge_riser_count": 2,
        "ledge_tread_cells": 4,
        "ledge_behavior": "low_ledge_ascent_candidate",
        "trench_depth_m": 0.48,
        "jump_span_cells": 12,
        "downstream_shift_cells": 0,
        "obstacle_height_m": 0.9,
        "rock_step_m": 0.08,
        "rock_tile_cells": (3, 3),
    },
    2: {
        "name": "climb_challenge",
        "stair_riser_m": 0.15,
        "ledge_riser_m": 0.65,
        "ledge_riser_count": 1,
        "ledge_tread_cells": 8,
        "ledge_behavior": "high_ledge_climb_candidate",
        "trench_depth_m": 0.60,
        "jump_span_cells": 14,
        "downstream_shift_cells": 2,
        "obstacle_height_m": 0.9,
        "rock_step_m": 0.08,
        "rock_tile_cells": (3, 3),
    },
}

ROUTE_START_IJ = (8, 20)
ROUTE_END_IJ = (136, 20)
OBSTACLE_BOUNDS_IJ = (27, 34, 12, 28)  # inclusive i0, i1, j0, j1
ROCK_BOUNDS_IJ = (92, 112, 12, 28)
def _stair_profile(start_m: float, riser_count: int, tread_cells: int,
                   riser_height_m: float, ascending: bool) -> list[float]:
    direction = 1.0 if ascending else -1.0
    intervals = riser_count * tread_cells
    return [
        start_m + direction * min(index // tread_cells, riser_count) * riser_height_m
        for index in range(intervals + 1)
    ]


def _axis_points(start_ij: tuple[int, int], end_ij: tuple[int, int]) -> list[tuple[int, int]]:
    changed = [axis for axis in range(2) if start_ij[axis] != end_ij[axis]]
    if len(changed) != 1:
        raise ValueError("Every route segment must be a nonzero axis-aligned grid segment")
    axis = changed[0]
    direction = 1 if end_ij[axis] > start_ij[axis] else -1
    result = []
    for value in range(start_ij[axis] + direction, end_ij[axis] + direction, direction):
        point = list(start_ij)
        point[axis] = value
        result.append(tuple(point))
    return result


def _paint_cells(terrain, points, heights, width_cells: int, owner: str,
                 written_cells: dict[tuple[int, int], tuple[float, str]]) -> int:
    nx, ny = TERRAIN_SHAPE_IJ
    if width_cells < 2 or width_cells % 2:
        raise ValueError("Painted route widths must be positive even numbers of cells")
    if len(points) != len(heights):
        raise ValueError("Surface heights must align one-to-one with painted grid points")
    axis = 0 if points[0][0] != points[-1][0] else 1
    lateral_axis = 1 - axis
    overlaps = 0
    for (i, j), height in zip(points, heights):
        center = (i, j)[lateral_axis]
        lo, hi = center - width_cells // 2, center + width_cells // 2
        for lateral in range(lo, hi):
            cell = (i, lateral) if axis == 0 else (lateral, j)
            if not 0 <= cell[0] < nx or not 0 <= cell[1] < ny:
                raise ValueError(f"Surface {owner!r} leaves the terrain grid at {cell}")
            previous = written_cells.get(cell)
            if previous is not None:
                if not math.isclose(previous[0], float(height), rel_tol=0.0, abs_tol=1e-7):
                    raise ValueError(
                        f"Conflicting surface heights at {cell}: {previous[0]} m from "
                        f"{previous[1]!r} versus {height} m from {owner!r}"
                    )
                overlaps += 1
            written_cells[cell] = (float(height), owner)
            terrain.hf[cell[0], cell[1]] = float(height)
    return overlaps


def _paint_rectangle(terrain, bounds_ij, height_m: float, owner: str,
                     written_cells: dict[tuple[int, int], tuple[float, str]]) -> None:
    i0, i1, j0, j1 = bounds_ij
    nx, ny = TERRAIN_SHAPE_IJ
    if not (0 <= i0 <= i1 < nx and 0 <= j0 <= j1 < ny):
        raise ValueError(f"Rectangle {owner!r} is outside the terrain grid")
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            cell = (i, j)
            previous = written_cells.get(cell)
            if previous is not None and not math.isclose(
                previous[0], height_m, rel_tol=0.0, abs_tol=1e-7
            ):
                raise ValueError(
                    f"Conflicting surface heights at {cell}: {previous[0]} m from "
                    f"{previous[1]!r} versus {height_m} m from {owner!r}"
                )
            written_cells[cell] = (float(height_m), owner)
            terrain.hf[i, j] = float(height_m)


def _paint_rock_field(terrain, bounds_ij, base_height_m: float, tile_cells: tuple[int, int],
                      step_m: float, rng: random.Random,
                      written_cells: dict[tuple[int, int], tuple[float, str]]) -> tuple[float, float]:
    i0, i1, j0, j1 = bounds_ij
    tile_i, tile_j = tile_cells
    levels = (-1, 0, 1)
    min_height = max_height = base_height_m
    for ii0 in range(i0, i1 + 1, tile_i):
        ii1 = min(i1, ii0 + tile_i - 1)
        for jj0 in range(j0, j1 + 1, tile_j):
            jj1 = min(j1, jj0 + tile_j - 1)
            # Pin both route ends to the adjoining level platform and stairs.
            level = 0 if ii0 == i0 or ii1 == i1 else rng.choice(levels)
            height = base_height_m + level * step_m
            min_height = min(min_height, height)
            max_height = max(max_height, height)
            for i in range(ii0, ii1 + 1):
                for j in range(jj0, jj1 + 1):
                    cell = (i, j)
                    previous = written_cells.get(cell)
                    if previous is not None and not math.isclose(
                        previous[0], height, rel_tol=0.0, abs_tol=1e-7
                    ):
                        raise ValueError(
                            f"Conflicting rock height at {cell}: {previous[0]} m from "
                            f"{previous[1]!r} versus {height} m from 'rock_field'"
                        )
                    written_cells[cell] = (float(height), "rock_field")
                    terrain.hf[i, j] = float(height)
    return min_height, max_height


def _validate_path_and_terrain(terrain, path_grid, path_nodes, segments) -> dict:
    import numpy as np
    import torch

    grid_np = path_grid.detach().cpu().numpy()
    nodes_np = path_nodes.detach().cpu().numpy()
    hf = terrain.hf.detach().cpu().numpy()
    nx, ny = hf.shape
    if not ((grid_np[:, 0] >= 0).all() and (grid_np[:, 0] < nx).all()
            and (grid_np[:, 1] >= 0).all() and (grid_np[:, 1] < ny).all()):
        raise RuntimeError("Route leaves the terrain grid")
    if len({tuple(point) for point in grid_np.tolist()}) != len(grid_np):
        raise RuntimeError("Authored route intersects or retraces itself")
    sampled = hf[grid_np[:, 0], grid_np[:, 1]]
    surface_error = np.abs(nodes_np[:, 2] - sampled)
    if not np.allclose(surface_error, 0.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("Every route node must lie on its sampled heightfield surface")

    jump_segments = [segment for segment in segments
                     if segment.get("edge_type") == "official_parc_astar_jump_edge_candidate"]
    if len(jump_segments) != 1:
        raise RuntimeError("The course must contain exactly one declared PARC A* jump edge")
    jump = jump_segments[0]
    jump_start = int(jump["path_node_start"])
    jump_end = int(jump["path_node_end"])
    if jump_end != jump_start + 1:
        raise RuntimeError("A jump edge must have exactly two route endpoints and no intermediate targets")

    edge_lengths_cells = np.abs(np.diff(grid_np, axis=0)).sum(axis=1)
    jump_edge_mask = np.zeros(len(grid_np) - 1, dtype=bool)
    jump_edge_mask[jump_start] = True
    if np.any((edge_lengths_cells != 1) & ~jump_edge_mask):
        raise RuntimeError("Only the declared jump edge may skip intermediate terrain cells")
    if edge_lengths_cells[jump_start] <= 1:
        raise RuntimeError("The declared jump edge must cross intervening terrain cells")

    start_grid = grid_np[jump_start]
    end_grid = grid_np[jump_end]
    terrain_dxdy = terrain.dxdy.detach().cpu().numpy()
    edge_xy_m = (end_grid - start_grid) * terrain_dxdy
    jump_xy_dist = float(np.linalg.norm(edge_xy_m))
    jump_z_diff = float(nodes_np[jump_end, 2] - nodes_np[jump_start, 2])
    if jump_xy_dist > PARC_ASTAR_JUMP_LIMITS["max_jump_xy_dist_m"] + 1e-6:
        raise RuntimeError("Jump edge exceeds the official PARC max_jump_xy_dist")
    if not (PARC_ASTAR_JUMP_LIMITS["min_jump_z_diff_m"] - 1e-6
            <= jump_z_diff <= PARC_ASTAR_JUMP_LIMITS["max_jump_z_diff_m"] + 1e-6):
        raise RuntimeError("Jump endpoint height difference is outside the official PARC limits")

    terrain_line_grid = np.asarray(
        [tuple(start_grid.tolist()), *_axis_points(tuple(start_grid), tuple(end_grid))],
        dtype=np.int64,
    )
    terrain_line_heights = hf[terrain_line_grid[:, 0], terrain_line_grid[:, 1]]
    if len(terrain_line_heights) < 3:
        raise RuntimeError("Jump candidate must have terrain samples between its support endpoints")
    endpoint_floor = min(nodes_np[jump_start, 2], nodes_np[jump_end, 2])
    gap_heights = terrain_line_heights[1:-1]
    if np.any(gap_heights >= endpoint_floor - 0.01):
        raise RuntimeError("The declared jump edge has no depressed terrain between its support endpoints")
    ray_limit = nodes_np[jump_start, 2] + PARC_ASTAR_JUMP_LIMITS["max_jump_z_diff_m"] + 1e-3
    if np.any(terrain_line_heights >= ray_limit):
        raise RuntimeError("Terrain along the candidate edge violates PARC A* jump ray clearance")

    deltas = np.diff(nodes_np, axis=0)
    return {
        "route_node_count": int(len(grid_np)),
        "planar_length_m": float(np.linalg.norm(deltas[:, :2], axis=1).sum()),
        "route_length_3d_m": float(np.linalg.norm(deltas, axis=1).sum()),
        "ground_following_node_height_error_max_m": float(surface_error.max(initial=0.0)),
        "jump_edge_count": len(jump_segments),
        "jump_edge_xy_distance_m": jump_xy_dist,
        "jump_edge_z_difference_m": jump_z_diff,
        "jump_gap_min_depth_below_support_m": float(endpoint_floor - gap_heights.max()),
        "bounds_checked": True,
        "ordinary_route_edges_four_connected": True,
        "all_route_edges_cardinal_or_declared_jump": True,
        "self_intersection_free": True,
        "all_route_nodes_on_heightfield": True,
        "jump_gap_between_support_nodes_checked": True,
        "torch_tensor_contract": isinstance(path_nodes, torch.Tensor),
    }


def build_mixed_course(device: str = "cpu", variant: int = 0, seed: int = 20261009, *, profile_overrides=None):
    """Return (official PARC SubTerrain, XYZ route tensor, geometry manifest).

    Route nodes lie on the heightfield. The trench is crossed by one discrete
    PARC A* jump-edge candidate between two terrain nodes; no intermediate body
    targets or motion samples are authored here.
    """
    import numpy as np
    import torch

    if isinstance(variant, bool) or not isinstance(variant, int) or variant not in VARIANT_PROFILES:
        raise ValueError(f"variant must be one of {tuple(VARIANT_PROFILES)} (0=easy, 1=medium)")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    profile = dict(VARIANT_PROFILES[variant])
    if profile_overrides:
        allowed = {'stair_riser_m', 'obstacle_height_m', 'rock_step_m', 'trench_depth_m', 'ledge_riser_m'}
        if set(profile_overrides) - allowed or any(not math.isfinite(v) or v <= 0 for v in profile_overrides.values()):
            raise ValueError('Invalid declared mixed-terrain geometry overrides')
        profile.update(profile_overrides)
    difficulty = profile["name"]
    rng = random.Random(seed)
    nx, ny = TERRAIN_SHAPE_IJ
    terrain_util = stair_course_builder._load_official_terrain_util()
    terrain = terrain_util.SubTerrain(
        f"parc_mixed_{difficulty}_{seed}", x_dim=nx, y_dim=ny,
        dx=GRID_SPACING_M, dy=GRID_SPACING_M, min_x=0.0, min_y=0.0, device=device,
    )
    terrain.hf.zero_()

    route = [ROUTE_START_IJ]
    route_surface_heights = [0.0]
    route_command_heights = [0.0]
    route_segments = []
    written_cells: dict[tuple[int, int], tuple[float, str]] = {}
    overlap_writes = 0

    # A continuous solid berm blocks the direct lane; the route goes around it.
    obstacle_height = profile["obstacle_height_m"]
    _paint_rectangle(terrain, OBSTACLE_BOUNDS_IJ, obstacle_height,
                     "solid_detour_berm", written_cells)

    def append_segment(name, stage, end_ij, surface_profile, *, feature_id,
                       requested_behavior, route_height_mode="terrain_surface",
                       command_profile=None, metadata=None, paint_surface=True):
        nonlocal overlap_writes
        start = route[-1]
        end = tuple(int(value) for value in end_ij)
        new_points = _axis_points(start, end)
        all_points = [start, *new_points]
        if len(surface_profile) != len(all_points):
            raise ValueError(f"Surface profile for {name!r} has the wrong number of samples")
        if not math.isclose(surface_profile[0], route_surface_heights[-1], abs_tol=1e-7):
            raise ValueError(f"Surface segment {name!r} does not join the incoming terrain height")
        if command_profile is None:
            command_profile = surface_profile
        if len(command_profile) != len(all_points):
            raise ValueError(f"Command profile for {name!r} has the wrong number of samples")
        if not math.isclose(command_profile[0], route_command_heights[-1], abs_tol=1e-7):
            raise ValueError(f"Route segment {name!r} does not join the incoming path height")
        width_cells = ROUTE_WIDTH_CELLS
        if paint_surface:
            overlap_writes += _paint_cells(
                terrain, all_points, surface_profile, width_cells, name, written_cells
            )
        start_node = len(route) - 1
        route.extend(new_points)
        route_surface_heights.extend(float(value) for value in surface_profile[1:])
        route_command_heights.extend(float(value) for value in command_profile[1:])
        end_node = len(route) - 1
        changed_axis = 0 if start[0] != end[0] else 1
        segment = {
            "name": name,
            "stage": stage,
            "feature_id": feature_id,
            "requested_behavior": requested_behavior,
            "path_node_start": start_node,
            "path_node_end": end_node,
            "start_grid_ij": list(start),
            "end_grid_ij": list(end),
            "axis": changed_axis,
            "direction": 1 if end[changed_axis] > start[changed_axis] else -1,
            "width_cells": width_cells,
            "width_m": width_cells * GRID_SPACING_M,
            "planar_length_m": len(new_points) * GRID_SPACING_M,
            "surface_start_height_m": float(surface_profile[0]),
            "surface_end_height_m": float(surface_profile[-1]),
            "route_height_mode": route_height_mode,
        }
        if metadata:
            segment.update(metadata)
        route_segments.append(segment)
        return segment

    def append_jump_edge(name, end_ij, terrain_profile, *, feature_id,
                         requested_behavior, metadata=None):
        nonlocal overlap_writes
        start = route[-1]
        end = tuple(int(value) for value in end_ij)
        terrain_points = [start, *_axis_points(start, end)]
        if len(terrain_profile) != len(terrain_points):
            raise ValueError(f"Terrain profile for jump edge {name!r} has the wrong sample count")
        if not math.isclose(terrain_profile[0], route_surface_heights[-1], abs_tol=1e-7):
            raise ValueError(f"Jump edge {name!r} does not begin at the incoming support surface")
        if not math.isfinite(float(terrain_profile[-1])):
            raise ValueError(f"Jump edge {name!r} must end on a finite terrain height")
        overlap_writes += _paint_cells(
            terrain, terrain_points, terrain_profile, ROUTE_WIDTH_CELLS, name, written_cells
        )

        start_node = len(route) - 1
        route.append(end)
        route_surface_heights.append(float(terrain_profile[-1]))
        route_command_heights.append(float(terrain_profile[-1]))
        end_node = len(route) - 1
        changed_axis = 0 if start[0] != end[0] else 1
        distance_cells = abs(end[changed_axis] - start[changed_axis])
        jump = {
            "name": name,
            "stage": "jump_candidate",
            "feature_id": feature_id,
            "requested_behavior": requested_behavior,
            "path_node_start": start_node,
            "path_node_end": end_node,
            "start_grid_ij": list(start),
            "end_grid_ij": list(end),
            "takeoff_node_role": "terrain_surface_support_candidate",
            "landing_node_role": "terrain_surface_support_candidate",
            "edge_type": "official_parc_astar_jump_edge_candidate",
            "route_height_mode": "terrain_surface_endpoints",
            "intermediate_route_target_count": 0,
            "gap_terrain_sample_count": len(terrain_points) - 2,
            "axis": changed_axis,
            "direction": 1 if end[changed_axis] > start[changed_axis] else -1,
            "width_cells": ROUTE_WIDTH_CELLS,
            "width_m": ROUTE_WIDTH_CELLS * GRID_SPACING_M,
            "planar_length_m": distance_cells * GRID_SPACING_M,
            "surface_start_height_m": float(terrain_profile[0]),
            "surface_end_height_m": float(terrain_profile[-1]),
            "astar_jump_limits": dict(PARC_ASTAR_JUMP_LIMITS),
        }
        if metadata:
            jump.update(metadata)
        route_segments.append(jump)
        return jump

    flat = lambda count, height=0.0: [float(height)] * (count + 1)

    append_segment("start_walk", "walk", (18, 20), flat(10), feature_id="flat_ground",
                   requested_behavior="walk")
    append_segment("obstacle_detour_out", "obstacle_detour", (18, 38), flat(18),
                   feature_id="berm_1", requested_behavior="detour_around_obstacle")
    append_segment("obstacle_detour_bypass", "obstacle_detour", (44, 38), flat(26),
                   feature_id="berm_1", requested_behavior="detour_around_obstacle")
    append_segment("obstacle_detour_return", "obstacle_detour", (44, 20), flat(18),
                   feature_id="berm_1", requested_behavior="detour_around_obstacle")

    append_segment("jump_takeoff_approach", "walk", (52, 20), flat(8),
                   feature_id="trench_1", requested_behavior="walk_to_trench")
    depth = profile["trench_depth_m"]
    jump_end_i = 52 + profile["jump_span_cells"]
    trench_points = [(52, 20), *_axis_points((52, 20), (jump_end_i, 20))]
    trench_surface = [0.0, *([-depth] * (len(trench_points) - 2)), 0.0]
    trench_bounds = (53, jump_end_i - 1, 12, 27)
    jump_segment = append_jump_edge(
        "jump_over_trench_candidate", (jump_end_i, 20), trench_surface,
        feature_id="trench_1", requested_behavior="jump_over_depression_candidate",
        metadata={"gap_depth_m": depth},
    )
    downstream_shift = profile["downstream_shift_cells"]
    stair_start_i = 66 + downstream_shift
    append_segment("jump_landing_walk", "walk", (stair_start_i, 20), flat(2),
                   feature_id="trench_1", requested_behavior="walk_after_landing")

    stair_height = profile["stair_riser_m"]
    stair_up = _stair_profile(0.0, 6, 2, stair_height, ascending=True)
    up_end = stair_up[-1]
    up_segment = append_segment(
        "stairs_up", "stairs_up", (stair_start_i + 12, 20), stair_up, feature_id="stairs_up_1",
        requested_behavior="stair_ascent_candidate",
        metadata={"riser_count": 6, "riser_height_m": stair_height,
                  "tread_depth_m": 2 * GRID_SPACING_M},
    )

    ledge_height = profile["ledge_riser_m"]
    ledge_riser_count = profile["ledge_riser_count"]
    ledge_tread_cells = profile["ledge_tread_cells"]
    ledge_up = _stair_profile(
        up_end, ledge_riser_count, ledge_tread_cells, ledge_height, ascending=True
    )
    ledge_end = ledge_up[-1]
    climb_challenge = variant == 2
    ledge_name = "high_ledge_climb_candidate" if climb_challenge else "stepped_ledge_ascent"
    ledge_stage = "climb_candidate" if climb_challenge else "stepped_ledge"
    append_segment(
        ledge_name, ledge_stage, (stair_start_i + 20, 20), ledge_up,
        feature_id="ledge_1", requested_behavior=profile["ledge_behavior"],
        metadata={"riser_count": ledge_riser_count, "riser_height_m": ledge_height,
                  "tread_depth_m": ledge_tread_cells * GRID_SPACING_M,
                  "overhang": False, "handholds": "not represented",
                  "behavior_candidate_only": True,
                  "hand_contact_evidence": "not observed by geometry builder",
                  "promotion_requirement": "force-derived hand-contact evidence in simulated motion"},
    )
    platform_end_i = stair_start_i + 26
    append_segment("elevated_walk", "platform_walk", (platform_end_i, 20), flat(6, ledge_end),
                   feature_id="upper_terrace_1", requested_behavior="walk_on_raised_surface")

    rock_bounds = (platform_end_i, platform_end_i + 20, ROCK_BOUNDS_IJ[2], ROCK_BOUNDS_IJ[3])
    rock_min, rock_max = _paint_rock_field(
        terrain, rock_bounds, ledge_end, profile["rock_tile_cells"],
        profile["rock_step_m"], rng, written_cells,
    )
    rock_end_i = rock_bounds[1]
    rock_heights = [float(terrain.hf[i, 20].item()) for i in range(platform_end_i, rock_end_i + 1)]
    rock_segment = append_segment(
        "rough_rock_traverse", "rough_rock", (rock_end_i, 20), rock_heights,
        feature_id="rocks_1", requested_behavior="uneven_stepping_candidate",
        metadata={"tile_cells_ij": list(profile["rock_tile_cells"]),
                  "height_quantum_m": profile["rock_step_m"]},
        paint_surface=False,
    )

    descent_riser = ledge_end / 8.0
    stair_down = _stair_profile(ledge_end, 8, 2, descent_riser, ascending=False)
    stairs_down_end_i = 128 + downstream_shift
    down_segment = append_segment(
        "stairs_down", "stairs_down", (stairs_down_end_i, 20), stair_down, feature_id="stairs_down_1",
        requested_behavior="stair_descent_candidate",
        metadata={"riser_count": 8, "riser_height_m": descent_riser,
                  "tread_depth_m": 2 * GRID_SPACING_M},
    )
    append_segment("finish_walk", "walk", ROUTE_END_IJ,
                   flat(ROUTE_END_IJ[0] - route[-1][0]),
                   feature_id="flat_ground", requested_behavior="walk_to_finish")

    terrain.hf_maxmin[..., 0] = terrain.hf
    terrain.hf_maxmin[..., 1] = terrain.hf

    path_grid = torch.tensor(route, dtype=torch.int64, device=device)
    sampled_nodes = terrain.get_xyz_point(path_grid)
    path_nodes = sampled_nodes.clone()
    path_nodes[:, 2] = torch.tensor(route_command_heights, dtype=path_nodes.dtype, device=device)
    validation = _validate_path_and_terrain(terrain, path_grid, path_nodes, route_segments)
    path_np = path_nodes.detach().cpu().numpy()
    surface_np = sampled_nodes.detach().cpu().numpy()
    grid_np = path_grid.detach().cpu().numpy()
    for segment in route_segments:
        start, end = segment["path_node_start"], segment["path_node_end"]
        segment["start_xyz_m"] = [float(value) for value in path_np[start]]
        segment["end_xyz_m"] = [float(value) for value in path_np[end]]
        segment["surface_start_xyz_m"] = [float(value) for value in surface_np[start]]
        segment["surface_end_xyz_m"] = [float(value) for value in surface_np[end]]

    waypoints = []
    waypoint_nodes = [(0, "course_start", "walk")]
    waypoint_nodes.extend(
        (segment["path_node_end"], segment["name"], segment["stage"])
        for segment in route_segments
    )
    for order, (node_index, name, stage) in enumerate(waypoint_nodes):
        incoming = None if node_index == 0 else (
            grid_np[node_index] - grid_np[node_index - 1]
        ).tolist()
        outgoing = None if node_index == len(route) - 1 else (
            grid_np[node_index + 1] - grid_np[node_index]
        ).tolist()
        xyz = [float(value) for value in path_np[node_index]]
        waypoints.append({
            "waypoint_id": f"parc_mixed_{difficulty}_{seed}:{name}",
            "order": order,
            "stage": stage,
            "path_node_index": node_index,
            "grid_ij": grid_np[node_index].tolist(),
            "xy_m": xyz[:2],
            "height_m": xyz[2],
            "xyz_m": xyz,
            "incoming_heading_ij": incoming,
            "outgoing_heading_ij": outgoing,
        })

    min_height = float(terrain.hf.min().item())
    max_height = float(terrain.hf.max().item())
    stair_flight = {
        "flight_id": "stairs_up_1",
        "orientation": "x",
        "axis": 0,
        "direction": 1,
        "riser_count": 6,
        "riser_height_m": stair_height,
        "total_rise_m": up_end,
        "tread_depth_m": 2 * GRID_SPACING_M,
        "tread_cells": 2,
        "width_cells": ROUTE_WIDTH_CELLS,
        "width_m": ROUTE_WIDTH_CELLS * GRID_SPACING_M,
        "start_grid_ij": [stair_start_i, 20],
        "end_grid_ij": [stair_start_i + 12, 20],
        "start_xyz_m": up_segment["start_xyz_m"],
        "ascent_end_xyz_m": up_segment["end_xyz_m"],
        "end_xyz_m": up_segment["end_xyz_m"],
        "platform_end_xyz_m": up_segment["end_xyz_m"],
        "path_segments": {"ascent": {
            "path_node_start": up_segment["path_node_start"],
            "path_node_end": up_segment["path_node_end"],
        }},
    }
    feature_records = [
        {
            "feature_id": "berm_1", "kind": "solid_heightfield_obstacle",
            "bounds_grid_ij_inclusive": list(OBSTACLE_BOUNDS_IJ),
            "top_height_m": obstacle_height,
            "route_segments": ["obstacle_detour_out", "obstacle_detour_bypass",
                               "obstacle_detour_return"],
            "geometry_limit": "single-valued heightfield berm; no overhang or separate mesh",
        },
        {
            "feature_id": "trench_1", "kind": "depression_under_discrete_jump_edge",
            "bounds_grid_ij_inclusive": list(trench_bounds),
            "depth_m": depth,
            "route_segment": "jump_over_trench_candidate",
            "takeoff_grid_ij": jump_segment["start_grid_ij"],
            "landing_grid_ij": jump_segment["end_grid_ij"],
            "geometry_limit": "depressed solid heightfield under a discrete A* edge; not an open void",
        },
        {
            "feature_id": "ledge_1",
            "kind": "high_ledge_climb_candidate" if climb_challenge else "broad_stepped_ledge",
            "bounds_grid_ij_inclusive": [stair_start_i + 12, stair_start_i + 20, 12, 27],
            "riser_height_m": ledge_height,
            "riser_count": ledge_riser_count,
            "overhang": False,
            "behavior_candidate_only": True,
            "hand_contact_evidence": "not observed by geometry builder",
            "promotion_requirement": "force-derived hand-contact evidence in simulated motion",
            "route_segment": ledge_name,
            "geometry_limit": "single-valued solid heightfield; no overhang or handhold geometry",
        },
        {
            "feature_id": "rocks_1", "kind": "quantized_rough_stepping_surface",
            "bounds_grid_ij_inclusive": list(rock_bounds),
            "height_range_m": [rock_min, rock_max],
            "tile_cells_ij": list(profile["rock_tile_cells"]),
            "route_segment": "rough_rock_traverse",
            "geometry_limit": "blocky single-valued heightfield; not individually simulated rocks",
        },
    ]

    segments_for_stairs = [up_segment, down_segment]
    deltas = np.diff(path_np, axis=0)
    ledge_limitation = (
        "The high ledge has no handholds and is only a climb candidate pending force-derived hand-contact evidence."
        if climb_challenge else
        "The low ledge is a broad stair-like heightfield; vertical handholds and overhangs are not represented."
    )
    manifest = {
        "schema_version": 1,
        "scenario_id": f"parc_mixed_{difficulty}_{seed}",
        "course_kind": "mixed_terrain",
        "variant": variant,
        "difficulty": difficulty,
        "seed": seed,
        "artifact_kind": "static_parc_mixed_course_geometry",
        "geometry_only": True,
        "motion_generation": "not performed",
        "physical_validity": "not assessed",
        "behavioral_success": "not assessed",
        "randomized": True,
        "deterministic_given_seed": True,
        "terrain": {
            "terrain_name": terrain.terrain_name,
            "grid_shape_ij": [nx, ny],
            "grid_spacing_xy_m": [GRID_SPACING_M, GRID_SPACING_M],
            "min_point_xy_m": terrain.min_point.detach().cpu().tolist(),
            "max_sample_point_xy_m": terrain.get_max_point().detach().cpu().tolist(),
            "grid_convention": "world_xy = min_point + (i, j) * dxdy; hf[i, j] is sampled top height",
            "height_units": "meters",
            "height_range_m": [min_height, max_height],
            "collision_representation": "PARC SubTerrain single-valued heightfield; no overhangs or separate mesh affordances",
            "hf_maxmin_frozen_to_authored_height": True,
        },
        "course": {
            "route_construction": "deterministic grid route plus one discrete PARC A* jump edge; rough rock tiles vary by seed",
            "route_start_grid_ij": list(ROUTE_START_IJ),
            "route_end_grid_ij": list(ROUTE_END_IJ),
            "route_start_xyz_m": [float(value) for value in path_np[0]],
            "route_end_xyz_m": [float(value) for value in path_np[-1]],
            "route_node_count": len(route),
            "planar_length_m": validation["planar_length_m"],
            "route_length_3d_m": validation["route_length_3d_m"],
            "flat_start_and_end": math.isclose(path_np[0, 2], surface_np[0, 2], abs_tol=1e-7)
            and math.isclose(path_np[-1, 2], surface_np[-1, 2], abs_tol=1e-7),
            "stair_flight_count": 1,
            "stair_segment_count": len(segments_for_stairs),
            "requested_behavior_families": ["walk", "stairs_up", "stairs_down",
                profile["ledge_behavior"], "jump_over_depression_candidate",
                "obstacle_detour", "uneven_rock_stepping"],
            "path_height_semantics": "every route node is a terrain-surface node; the jump is one edge between two support candidates",
        },
        "jump_edge_contract": {
            **PARC_ASTAR_JUMP_LIMITS,
            "representation": "single directed navigation edge between takeoff and landing terrain nodes",
            "intermediate_route_or_pose_targets": 0,
            "inferred_body_motion": "left to the official PARC generator; not authored by this course builder",
        },
        "layout_constraints": {
            "route_width_m": ROUTE_WIDTH_CELLS * GRID_SPACING_M,
            "obstacle_route_surface_gap_cells": 1,
            "terrain_bounds_checked": True,
            "surface_height_conflicts_rejected": True,
        },
        "surface_rasterization": {
            "mode": "PARC axis-aligned solid heightfield strips and rectangles",
            "surface_cell_count": len(written_cells),
            "same_height_join_cell_writes": overlap_writes,
            "height_conflict_count": 0,
        },
        "validation": validation,
        "features": feature_records,
        "stair_flights": [stair_flight],
        "route_segments": route_segments,
        "task_waypoints": waypoints,
        "task_waypoint_contract": "ordered terrain-surface route nodes; a jump is represented by one discrete edge, not a trajectory",
        "limitations": [
            "The trench is a depressed solid heightfield, not an unsupported open void.",
            "The jump edge is a PARC A* navigation candidate; it does not specify an apex or guarantee a generated jump.",
            "The high ledge remains a climb candidate until force-derived hand-contact evidence is observed in simulation.",
            "Jump and climb labels are route/terrain intent candidates, not achieved motion labels.",
            ledge_limitation,
            "No poses are synthesized and no physical or motion-quality validation is performed here.",
        ],
    }
    return terrain, path_nodes, manifest


def save_mixed_course(output_dir: Path, terrain, path_nodes, manifest: dict) -> dict:
    """Save terrain, ordered route, manifest, and a static mixed-course preview."""
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    terrain_path = output_dir / "terrain_and_route.npz"
    manifest_path = output_dir / "course_manifest.json"
    preview_path = output_dir / "course_preview.png"

    path_np = path_nodes.detach().cpu().numpy()
    min_xy = terrain.min_point.detach().cpu().numpy()
    dxdy = terrain.dxdy.detach().cpu().numpy()
    path_grid = np.rint((path_np[:, :2] - min_xy) / dxdy).astype(np.int64)
    segments = manifest["route_segments"]
    np.savez_compressed(
        terrain_path,
        hf=terrain.hf.detach().cpu().numpy(),
        hf_maxmin=terrain.hf_maxmin.detach().cpu().numpy(),
        min_point=min_xy,
        dxdy=dxdy,
        path_nodes=path_np,
        path_grid_ij=path_grid,
        segment_names=np.asarray([item["name"] for item in segments]),
        segment_stages=np.asarray([item["stage"] for item in segments]),
        segment_path_start=np.asarray([item["path_node_start"] for item in segments], dtype=np.int64),
        segment_path_end=np.asarray([item["path_node_end"] for item in segments], dtype=np.int64),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _save_preview(preview_path, terrain, path_nodes, manifest)
    return {"terrain_and_route": terrain_path, "manifest": manifest_path, "preview": preview_path}


def _save_preview(path: Path, terrain, path_nodes, manifest: dict) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    heights = terrain.hf.detach().cpu().numpy()
    nodes = path_nodes.detach().cpu().numpy()
    min_xy = terrain.min_point.detach().cpu().numpy()
    dxdy = terrain.dxdy.detach().cpu().numpy()
    extent = [min_xy[0], min_xy[0] + (heights.shape[0] - 1) * dxdy[0],
              min_xy[1], min_xy[1] + (heights.shape[1] - 1) * dxdy[1]]
    grid = np.rint((nodes[:, :2] - min_xy) / dxdy).astype(np.int64)
    distance = np.concatenate((
        np.zeros(1), np.cumsum(np.linalg.norm(np.diff(nodes[:, :2], axis=0), axis=1))
    ))
    figure, (map_axis, profile_axis) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    image = map_axis.imshow(heights.T, origin="lower", extent=extent, cmap="terrain",
                            interpolation="nearest", aspect="equal")
    map_axis.plot(nodes[:, 0], nodes[:, 1], color="#17212b", linewidth=1.8, label="route")
    map_axis.scatter(*nodes[0, :2], marker="o", s=42, color="#17864a", label="start")
    map_axis.scatter(*nodes[-1, :2], marker="X", s=52, color="#bd3f35", label="end")
    for feature in manifest["features"]:
        i0, i1, j0, j1 = feature["bounds_grid_ij_inclusive"]
        rect = Rectangle(
            (min_xy[0] + i0 * dxdy[0], min_xy[1] + j0 * dxdy[1]),
            (i1 - i0 + 1) * dxdy[0], (j1 - j0 + 1) * dxdy[1],
            fill=False, edgecolor="#202a35", linewidth=1.0, linestyle="--",
        )
        map_axis.add_patch(rect)
        map_axis.text(rect.get_x(), rect.get_y() + rect.get_height(), feature["feature_id"],
                      fontsize=8, color="#111820", va="bottom")
    map_axis.set_title("Terrain and Route")
    map_axis.set_xlabel("X (m)")
    map_axis.set_ylabel("Y (m)")
    map_axis.legend(loc="upper left", frameon=True)
    colorbar = figure.colorbar(image, ax=map_axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Heightfield top (m)")

    jump = next(segment for segment in manifest["route_segments"]
                if segment.get("edge_type") == "official_parc_astar_jump_edge_candidate")
    jump_start = int(jump["path_node_start"])
    jump_end = int(jump["path_node_end"])
    profile_axis.plot(distance[:jump_start + 1], nodes[:jump_start + 1, 2],
                      color="#235b8e", linewidth=1.8, label="route nodes on terrain")
    profile_axis.plot(distance[jump_end:], nodes[jump_end:, 2],
                      color="#235b8e", linewidth=1.8)
    profile_axis.plot(distance[jump_start:jump_end + 1], nodes[jump_start:jump_end + 1, 2],
                      color="#7744a3", linewidth=1.6, linestyle=":", marker="o",
                      label="PARC A* jump edge (endpoints only)")
    start_grid = grid[jump_start]
    end_grid = grid[jump_end]
    gap_grid = np.asarray([start_grid, *_axis_points(tuple(start_grid), tuple(end_grid))])
    gap_xy = min_xy[:2] + gap_grid * dxdy
    gap_distance = distance[jump_start] + np.linalg.norm(gap_xy - gap_xy[0], axis=1)
    gap_heights = heights[gap_grid[:, 0], gap_grid[:, 1]]
    profile_axis.plot(gap_distance, gap_heights, color="#b45c30", linewidth=1.4,
                      linestyle="--", label="heightfield through gap")
    profile_axis.set_title("Terrain Nodes and Discrete Jump Edge")
    profile_axis.set_xlabel("Planar route distance (m)")
    profile_axis.set_ylabel("Height (m)")
    profile_axis.grid(True, linewidth=0.5, alpha=0.45)
    profile_axis.legend(loc="best")
    profile_axis.set_xlim(0.0, distance[-1])
    figure.suptitle(
        f"PARC mixed terrain: {manifest['difficulty']} / seed {manifest['seed']} / "
        f"{manifest['course']['planar_length_m']:.1f} m\nGeometry only; motion and physical success not assessed"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu", help="Torch device for static terrain tensors")
    parser.add_argument("--variant", type=int, choices=tuple(VARIANT_PROFILES), default=0,
                        help="0=easy, 1=medium")
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args(argv)
    terrain, path_nodes, manifest = build_mixed_course(
        device=args.device, variant=args.variant, seed=args.seed
    )
    output_dir = args.output_dir or DEFAULT_OUTPUT / f"variant_{args.variant}_seed_{args.seed}"
    outputs = save_mixed_course(output_dir, terrain, path_nodes, manifest)
    print(json.dumps({
        "status": "geometry_only",
        "motion_generation": "not performed",
        "behavioral_success": "not assessed",
        "scenario_id": manifest["scenario_id"],
        "variant": args.variant,
        "difficulty": manifest["difficulty"],
        "seed": args.seed,
        "output_files": {name: str(path.resolve()) for name, path in outputs.items()},
        "planar_length_m": manifest["course"]["planar_length_m"],
        "route_length_3d_m": manifest["course"]["route_length_3d_m"],
        "segment_count": len(manifest["route_segments"]),
    }, indent=2))


if __name__ == "__main__":
    main()
