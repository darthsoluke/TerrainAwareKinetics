"""Build deterministic, static PARC platform-extension courses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "out" / "parc_platform_extensions"
GRID_SPACING_M = 0.2
TERRAIN_SHAPE_IJ = (160, 140)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_parc_stair_course as stair_course_builder


EXTENSION_VARIANTS = (
    {
        "scenario_id": "parc_extension_terrace_stairs",
        "description": "Three exact-height terraces connected by broad straight stair flights.",
        "start_ij": (5, 70),
        "segments": (
            {"name": "start_approach", "stage": "flat_approach", "end_ij": (20, 70), "width_m": 4.0, "height_m": 0.0},
            {"name": "flight_1_ascent", "stage": "ascent", "end_ij": (26, 70), "width_m": 2.4,
             "stair": {"riser_count": 3, "riser_height_m": 0.12, "tread_depth_m": 0.4, "ascending": True}},
            {"name": "terrace_1", "stage": "platform", "end_ij": (56, 70), "width_m": 4.8, "height_m": 0.36},
            {"name": "flight_2_ascent", "stage": "ascent", "end_ij": (62, 70), "width_m": 2.4,
             "stair": {"riser_count": 3, "riser_height_m": 0.18, "tread_depth_m": 0.4, "ascending": True}},
            {"name": "terrace_2", "stage": "platform", "end_ij": (87, 70), "width_m": 3.6, "height_m": 0.9},
            {"name": "flight_3_ascent", "stage": "ascent", "end_ij": (91, 70), "width_m": 2.4,
             "stair": {"riser_count": 2, "riser_height_m": 0.15, "tread_depth_m": 0.4, "ascending": True}},
            {"name": "terrace_3", "stage": "platform", "end_ij": (131, 70), "width_m": 5.2, "height_m": 1.2},
        ),
        "layout_constraints": {"layout": "ascending_terraces"},
        "terminal_support": {"end_ij": (137, 70), "width_m": 5.2, "height_m": 1.2},
    },
    {
        "scenario_id": "parc_extension_elevated_bridges",
        "description": "Two elevated platforms joined by a broad-turn, level bridge.",
        "start_ij": (8, 20),
        "segments": (
            {"name": "start_approach", "stage": "flat_approach", "end_ij": (18, 20), "width_m": 3.2, "height_m": 0.0},
            {"name": "flight_1_ascent", "stage": "ascent", "end_ij": (30, 20), "width_m": 2.4,
             "stair": {"riser_count": 4, "riser_height_m": 0.15, "tread_depth_m": 0.6, "ascending": True}},
            {"name": "elevated_platform_1", "stage": "platform", "end_ij": (75, 20), "width_m": 5.2, "height_m": 0.6},
            {"name": "level_bridge_1", "stage": "bridge", "end_ij": (75, 50), "width_m": 2.4, "height_m": 0.6},
            {"name": "elevated_platform_2", "stage": "platform", "end_ij": (30, 50), "width_m": 5.2, "height_m": 0.6},
            {"name": "flight_2_descent", "stage": "descent", "end_ij": (18, 50), "width_m": 2.4,
             "stair": {"riser_count": 4, "riser_height_m": 0.15, "tread_depth_m": 0.6, "ascending": False}},
            {"name": "end_exit", "stage": "flat_exit", "end_ij": (8, 50), "width_m": 3.2, "height_m": 0.0},
        ),
        "layout_constraints": {
            "layout": "elevated_platform_bridge_turn",
            "parallel_leg_center_spacing_m": [6.0],
            "minimum_parallel_leg_clearance_m": 0.8,
            "bridge_width_m": 2.4,
        },
    },
    {
        "scenario_id": "parc_extension_broad_s_route",
        "description": "Three broad parallel platform legs joined by level bridges with generous spacing.",
        "start_ij": (10, 18),
        "segments": (
            {"name": "start_approach", "stage": "flat_approach", "end_ij": (17, 18), "width_m": 3.2, "height_m": 0.0},
            {"name": "flight_1_ascent", "stage": "ascent", "end_ij": (29, 18), "width_m": 2.4,
             "stair": {"riser_count": 4, "riser_height_m": 0.18, "tread_depth_m": 0.6, "ascending": True}},
            {"name": "s_leg_1", "stage": "platform", "end_ij": (67, 18), "width_m": 4.0, "height_m": 0.72},
            {"name": "level_bridge_1", "stage": "bridge", "end_ij": (67, 44), "width_m": 2.4, "height_m": 0.72},
            {"name": "s_leg_2", "stage": "platform", "end_ij": (29, 44), "width_m": 5.2, "height_m": 0.72},
            {"name": "level_bridge_2", "stage": "bridge", "end_ij": (29, 70), "width_m": 2.4, "height_m": 0.72},
            {"name": "s_leg_3", "stage": "platform", "end_ij": (67, 70), "width_m": 4.4, "height_m": 0.72},
            {"name": "flight_2_descent", "stage": "descent", "end_ij": (79, 70), "width_m": 2.4,
             "stair": {"riser_count": 4, "riser_height_m": 0.18, "tread_depth_m": 0.6, "ascending": False}},
            {"name": "end_exit", "stage": "flat_exit", "end_ij": (110, 70), "width_m": 3.2, "height_m": 0.0},
        ),
        "layout_constraints": {
            "layout": "three_leg_s_route",
            "parallel_leg_center_spacing_m": [5.2, 5.2],
            "minimum_parallel_leg_clearance_m": 0.4,
        },
    },
)


def _raster_cells(distance_m: float, label: str) -> int:
    cells = int(round(distance_m / GRID_SPACING_M))
    if cells < 1 or not math.isclose(cells * GRID_SPACING_M, distance_m, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{label} must be a positive multiple of {GRID_SPACING_M} m")
    return cells


def _segment_grid_points(start_ij: tuple[int, int], end_ij: tuple[int, int]) -> list[tuple[int, int]]:
    changed_axes = [axis for axis in range(2) if start_ij[axis] != end_ij[axis]]
    if len(changed_axes) != 1:
        raise ValueError("Every route segment must be a nonzero, axis-aligned grid segment")
    axis = changed_axes[0]
    direction = 1 if end_ij[axis] > start_ij[axis] else -1
    points = []
    for value in range(start_ij[axis] + direction, end_ij[axis] + direction, direction):
        point = list(start_ij)
        point[axis] = value
        points.append(tuple(point))
    return points


def _segment_profile(start_height: float, spec: dict, cell_count: int) -> tuple[list[float], dict | None]:
    if "stair" not in spec:
        height = float(spec["height_m"])
        if not math.isclose(height, start_height, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError(f"Flat segment {spec['name']!r} does not continue at the incoming height")
        return [height] * (cell_count + 1), None

    stair = spec["stair"]
    riser_count = stair["riser_count"]
    riser_height = float(stair["riser_height_m"])
    tread_cells = _raster_cells(float(stair["tread_depth_m"]), "stair tread depth")
    if riser_count < 1 or not 0.12 - 1e-8 <= riser_height <= 0.20 + 1e-8:
        raise ValueError("Stair flights need positive riser counts and 0.12-0.20 m risers")
    if cell_count != riser_count * tread_cells:
        raise ValueError(f"Stair segment {spec['name']!r} length must equal riser_count * tread_depth")
    sign = 1 if stair["ascending"] else -1
    profile = []
    for offset in range(cell_count + 1):
        level = min(offset // tread_cells, riser_count)
        profile.append(start_height + sign * level * riser_height)
    flight = {
        "riser_count": int(riser_count),
        "riser_height_m": riser_height,
        "tread_cells": tread_cells,
        "tread_depth_m": tread_cells * GRID_SPACING_M,
        "ascending": bool(stair["ascending"]),
        "start_height_m": float(profile[0]),
        "end_height_m": float(profile[-1]),
    }
    return profile, flight


def _paint_segment(terrain, points: list[tuple[int, int]], heights: list[float], width_cells: int,
                   segment_name: str, written_cells: dict[tuple[int, int], tuple[float, str]]) -> int:
    nx, ny = TERRAIN_SHAPE_IJ
    axis = 0 if points[0][0] != points[-1][0] else 1
    center_axis = 1 - axis
    overlap_writes = 0
    for (i, j), height in zip(points, heights):
        center = (i, j)[center_axis]
        lateral_start = center - width_cells // 2
        lateral_end = lateral_start + width_cells
        for lateral in range(lateral_start, lateral_end):
            cell = (i, lateral) if axis == 0 else (lateral, j)
            if not (0 <= cell[0] < nx and 0 <= cell[1] < ny):
                raise ValueError(f"Segment {segment_name!r} surface leaves the terrain grid at {cell}")
            previous = written_cells.get(cell)
            if previous is not None:
                if not math.isclose(previous[0], height, rel_tol=0.0, abs_tol=1e-7):
                    raise ValueError(
                        f"Conflicting surface heights at {cell}: {previous[0]} m from "
                        f"{previous[1]!r} versus {height} m from {segment_name!r}"
                    )
                overlap_writes += 1
            written_cells[cell] = (float(height), segment_name)
            terrain.hf[cell[0], cell[1]] = float(height)
    return overlap_writes


def _waypoint_records(scenario_id: str, route: list[tuple[int, int]], path_np,
                      route_segments: list[dict]) -> list[dict]:
    waypoints = []
    waypoint_nodes = [(0, "course_start", "course_start")]
    waypoint_nodes.extend(
        (segment["path_node_end"], segment["name"], segment["stage"])
        for segment in route_segments
    )
    for order, (index, name, stage) in enumerate(waypoint_nodes):
        grid_ij = route[index]
        incoming = None if index == 0 else [
            route[index][axis] - route[index - 1][axis] for axis in range(2)
        ]
        outgoing = None if index == len(route) - 1 else [
            route[index + 1][axis] - route[index][axis] for axis in range(2)
        ]
        turn = 0.0
        if incoming is not None and outgoing is not None:
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
            turn = math.degrees(math.atan2(cross, dot))
        xyz = [float(value) for value in path_np[index]]
        waypoints.append({
            "waypoint_id": f"{scenario_id}:{name}",
            "order": order,
            "stage": stage,
            "path_node_index": index,
            "grid_ij": list(grid_ij),
            "xy_m": xyz[:2],
            "height_m": xyz[2],
            "xyz_m": xyz,
            "signed_turn_degrees": turn,
            "incoming_heading_xy": incoming,
            "outgoing_heading_xy": outgoing,
        })
    return waypoints


def build_extension_course(device: str = "cpu", variant: int = 0):
    """Return an official SubTerrain, sampled XYZ route, and static geometry manifest."""
    import numpy as np
    import torch

    if isinstance(variant, bool) or not isinstance(variant, int) or variant not in range(len(EXTENSION_VARIANTS)):
        raise ValueError(f"variant must be one of 0, 1, {len(EXTENSION_VARIANTS) - 1}")
    spec = EXTENSION_VARIANTS[variant]
    nx, ny = TERRAIN_SHAPE_IJ
    terrain = stair_course_builder._load_official_terrain_util().SubTerrain(
        spec["scenario_id"], x_dim=nx, y_dim=ny,
        dx=GRID_SPACING_M, dy=GRID_SPACING_M, min_x=0.0, min_y=0.0, device=device,
    )
    terrain.hf.zero_()

    route = [tuple(spec["start_ij"])]
    route_heights = [0.0]
    route_segments = []
    stair_flights = []
    platforms = []
    written_cells: dict[tuple[int, int], tuple[float, str]] = {}
    overlap_writes = 0

    for segment_spec in spec["segments"]:
        start = route[-1]
        end = tuple(segment_spec["end_ij"])
        new_points = _segment_grid_points(start, end)
        all_points = [start, *new_points]
        cell_count = len(new_points)
        profile, flight_profile = _segment_profile(route_heights[-1], segment_spec, cell_count)
        width_cells = _raster_cells(float(segment_spec["width_m"]), "route width")
        if width_cells % 2:
            raise ValueError(f"Segment {segment_spec['name']!r} needs an even number of width cells")
        if flight_profile is not None and width_cells * GRID_SPACING_M < 2.4 - 1e-8:
            raise ValueError("Straight stair flights must be at least 2.4 m wide")

        overlap_writes += _paint_segment(
            terrain, all_points, profile, width_cells, segment_spec["name"], written_cells
        )
        start_node = len(route) - 1
        route.extend(new_points)
        route_heights.extend(profile[1:])
        end_node = len(route) - 1
        changed_axis = 0 if start[0] != end[0] else 1
        segment = {
            "name": segment_spec["name"],
            "stage": segment_spec["stage"],
            "flight_id": None,
            "path_node_start": start_node,
            "path_node_end": end_node,
            "start_grid_ij": list(start),
            "end_grid_ij": list(end),
            "axis": changed_axis,
            "direction": 1 if end[changed_axis] > start[changed_axis] else -1,
            "width_cells": width_cells,
            "width_m": width_cells * GRID_SPACING_M,
            "planar_length_m": cell_count * GRID_SPACING_M,
            "start_height_m": float(profile[0]),
            "end_height_m": float(profile[-1]),
        }
        route_segments.append(segment)

        if flight_profile is not None:
            flight_id = f"flight_{len(stair_flights) + 1}"
            segment["flight_id"] = flight_id
            segment["tread_cells"] = flight_profile["tread_cells"]
            segment["riser_count"] = flight_profile["riser_count"]
            segment["riser_height_m"] = flight_profile["riser_height_m"]
            segment["tread_depth_m"] = flight_profile["tread_depth_m"]
            stair_flights.append({
                "flight_id": flight_id,
                "orientation": "x" if changed_axis == 0 else "y",
                "axis": changed_axis,
                "direction": segment["direction"],
                "vertical_direction": 1 if flight_profile["ascending"] else -1,
                "riser_count": flight_profile["riser_count"],
                "riser_height_m": flight_profile["riser_height_m"],
                "total_rise_m": flight_profile["riser_count"] * flight_profile["riser_height_m"],
                "tread_depth_m": flight_profile["tread_depth_m"],
                "tread_cells": flight_profile["tread_cells"],
                "width_cells": width_cells,
                "width_m": width_cells * GRID_SPACING_M,
                "start_grid_ij": list(start),
                "end_grid_ij": list(end),
                "path_segments": {
                    segment_spec["stage"]: {
                        "path_node_start": start_node,
                        "path_node_end": end_node,
                    }
                },
            })
        elif segment_spec["stage"] == "platform":
            platforms.append({
                "platform_id": segment_spec["name"],
                "height_m": float(profile[-1]),
                "width_m": width_cells * GRID_SPACING_M,
                "length_m": cell_count * GRID_SPACING_M,
                "path_node_start": start_node,
                "path_node_end": end_node,
                "start_grid_ij": list(start),
                "end_grid_ij": list(end),
            })

    terminal_support = None
    support_spec = spec.get("terminal_support")
    if support_spec is not None:
        support_start = route[-1]
        support_end = tuple(support_spec["end_ij"])
        support_points = [support_start, *_segment_grid_points(support_start, support_end)]
        support_width_cells = _raster_cells(float(support_spec["width_m"]), "terminal support width")
        support_height = float(support_spec["height_m"])
        if support_width_cells % 2 or not math.isclose(
            support_height, route_heights[-1], rel_tol=0.0, abs_tol=1e-7
        ):
            raise ValueError("Terminal support must be an even-cell strip at the route endpoint height")
        overlap_writes += _paint_segment(
            terrain,
            support_points,
            [support_height] * len(support_points),
            support_width_cells,
            "terminal_support",
            written_cells,
        )
        support_length = (len(support_points) - 1) * GRID_SPACING_M
        if support_length < 1.0:
            raise ValueError("Terminal support must extend at least 1.0 m beyond the route endpoint")
        terminal_support = {
            "start_grid_ij": list(support_start),
            "end_grid_ij": list(support_end),
            "height_m": support_height,
            "width_m": support_width_cells * GRID_SPACING_M,
            "beyond_route_m": support_length,
            "purpose": "same-height terminal platform beyond the final route node; not part of the routed trajectory",
        }

    terrain.hf_maxmin[..., 0] = terrain.hf
    terrain.hf_maxmin[..., 1] = terrain.hf

    path_grid = torch.tensor(route, dtype=torch.int64, device=device)
    if len(set(route)) != len(route):
        raise RuntimeError("Authored extension route intersects or retraces itself")
    if not bool(((path_grid >= 0) & (path_grid < terrain.dims)).all()):
        raise RuntimeError("Authored extension route leaves the terrain grid")
    path_nodes = terrain.get_xyz_point(path_grid)
    path_np = path_nodes.detach().cpu().numpy()
    grid_np = path_grid.detach().cpu().numpy()
    if not np.array_equal(np.abs(np.diff(grid_np, axis=0)).sum(axis=1), np.ones(len(route) - 1, dtype=np.int64)):
        raise RuntimeError("Authored extension route is not four-connected")
    if not np.allclose(path_np[:, 2], np.asarray(route_heights), rtol=0.0, atol=1e-6):
        raise RuntimeError("Route samples do not match the authored height profile")

    for segment in route_segments:
        start_node, end_node = segment["path_node_start"], segment["path_node_end"]
        segment["start_xyz_m"] = [float(value) for value in path_np[start_node]]
        segment["end_xyz_m"] = [float(value) for value in path_np[end_node]]
    for flight, segment in zip(stair_flights, [item for item in route_segments if item["flight_id"]]):
        flight["start_xyz_m"] = segment["start_xyz_m"]
        flight["end_xyz_m"] = segment["end_xyz_m"]
    waypoints = _waypoint_records(spec["scenario_id"], route, path_np, route_segments)

    deltas = np.diff(path_np, axis=0)
    planar_length = float(np.linalg.norm(deltas[:, :2], axis=1).sum())
    surface_length = float(np.linalg.norm(deltas, axis=1).sum())
    if not 25.0 <= planar_length <= 50.0 or surface_length > 50.0 + 1e-6:
        raise RuntimeError("Extension course route must have a 25-50 m planar and surface length")

    manifest = {
        "schema_version": 1,
        "scenario_id": spec["scenario_id"],
        "variant": variant,
        "course_kind": "platform_extension",
        "description": spec["description"],
        "artifact_kind": "static_parc_platform_extension_geometry",
        "geometry_only": True,
        "motion_generation": "not performed",
        "physical_validity": "not assessed",
        "randomized": False,
        "terrain": {
            "terrain_name": terrain.terrain_name,
            "grid_shape_ij": [nx, ny],
            "grid_spacing_xy_m": [GRID_SPACING_M, GRID_SPACING_M],
            "min_point_xy_m": terrain.min_point.detach().cpu().tolist(),
            "max_sample_point_xy_m": terrain.get_max_point().detach().cpu().tolist(),
            "grid_convention": "world_xy = min_point + (i, j) * dxdy; hf[i, j] is the sampled top height",
            "height_units": "meters",
        },
        "course": {
            "route_construction": "deterministic four-connected axis-aligned grid route; no shortcutting",
            "route_node_count": len(route),
            "route_start_grid_ij": list(route[0]),
            "route_end_grid_ij": list(route[-1]),
            "route_start_xyz_m": [float(value) for value in path_np[0]],
            "route_end_xyz_m": [float(value) for value in path_np[-1]],
            "planar_length_m": planar_length,
            "surface_length_m": surface_length,
            "target_length_m": [25.0, 50.0],
            "stair_flight_count": len(stair_flights),
            "flat_start_and_end": math.isclose(path_np[0, 2], 0.0) and math.isclose(path_np[-1, 2], 0.0),
        },
        "layout_constraints": dict(spec["layout_constraints"]),
        "surface_rasterization": {
            "mode": "axis-aligned solid heightfield strips; conflicting height writes are rejected",
            "surface_cell_count": len(written_cells),
            "same_height_join_cell_writes": overlap_writes,
            "height_conflict_count": 0,
        },
        "platforms": platforms,
        "terminal_support": terminal_support,
        "stair_flights": stair_flights,
        "route_segments": route_segments,
        "task_waypoints": waypoints,
        "task_waypoint_contract": "ordered route positions; xyz_m includes the sampled terrain top height",
    }
    return terrain, path_nodes, manifest


def save_extension_course(output_dir: Path, terrain, path_nodes, manifest: dict) -> dict:
    """Save with the established PARC terrain, manifest, and preview conventions."""
    return stair_course_builder.save_course(output_dir, terrain, path_nodes, manifest)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu", help="Torch device for static terrain tensors")
    parser.add_argument("--variant", type=int, choices=range(len(EXTENSION_VARIANTS)), default=0)
    args = parser.parse_args(argv)
    terrain, path_nodes, manifest = build_extension_course(device=args.device, variant=args.variant)
    output_dir = args.output_dir or (DEFAULT_OUTPUT / manifest["scenario_id"])
    outputs = save_extension_course(output_dir, terrain, path_nodes, manifest)
    print(json.dumps({
        "status": "geometry_only",
        "motion_generation": "not performed",
        "scenario_id": manifest["scenario_id"],
        "output_files": {name: str(path.resolve()) for name, path in outputs.items()},
        "planar_length_m": manifest["course"]["planar_length_m"],
        "surface_length_m": manifest["course"]["surface_length_m"],
        "stair_flight_count": len(manifest["stair_flights"]),
    }, indent=2))


if __name__ == "__main__":
    main()
