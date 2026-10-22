"""Build a deterministic PARC heightfield course without generating motion."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PARC_REPO = ROOT / "out" / "vendor" / "PARC"
DEFAULT_OUTPUT = ROOT / "out" / "parc_stair_course"
DEFAULT_PLATFORM_OUTPUT = ROOT / "out" / "parc_platform_course"

PLATFORM_VARIANTS = (
    {"scenario_id": "parc_platform_low_048", "height_m": 0.48, "riser_height_m": 0.12,
     "width_m": 4.8, "tread_depth_m": 0.4, "lateral_direction": 1},
    {"scenario_id": "parc_platform_mid_072", "height_m": 0.72, "riser_height_m": 0.18,
     "width_m": 6.4, "tread_depth_m": 0.6, "lateral_direction": -1},
    {"scenario_id": "parc_platform_high_096", "height_m": 0.96, "riser_height_m": 0.24,
     "width_m": 8.0, "tread_depth_m": 0.6, "lateral_direction": 1},
)

GRID_SPACING_M = 0.2
TERRAIN_SHAPE_IJ = (160, 140)
STAIR_WIDTH_CELLS = 12
PLATFORM_LENGTH_M = 1.2
START_IJ = (10, 50)
END_IJ = (150, 115)

FLIGHT_SPECS = (
    {
        "flight_id": "flight_1",
        "orientation": "x",
        "start_ij": (25, 50),
        "riser_count": 3,
        "riser_height_m": 0.12,
        "tread_depth_m": 0.4,
    },
    {
        "flight_id": "flight_2",
        "orientation": "y",
        "start_ij": (60, 85),
        "riser_count": 4,
        "riser_height_m": 0.18,
        "tread_depth_m": 0.4,
    },
    {
        "flight_id": "flight_3",
        "orientation": "x",
        "start_ij": (100, 115),
        "riser_count": 5,
        "riser_height_m": 0.24,
        "tread_depth_m": 0.6,
    },
)


def _load_official_terrain_util():
    if not PARC_REPO.is_dir():
        raise FileNotFoundError(f"Official PARC checkout not found: {PARC_REPO}")
    repo_path = str(PARC_REPO)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    if (PARC_REPO / "PARC").is_dir() and "parc" not in sys.modules:
        sys.modules["parc"] = importlib.import_module("PARC")
    from parc.util import terrain_util

    return terrain_util


def _flight_grid_points(spec: dict, platform_cells: int, tread_cells: int) -> dict:
    axis = 0 if spec["orientation"] == "x" else 1
    start = list(spec["start_ij"])
    rise_cells = spec["riser_count"] * tread_cells
    top = start.copy()
    top[axis] += rise_cells
    platform_end = top.copy()
    platform_end[axis] += platform_cells
    end = platform_end.copy()
    end[axis] += rise_cells
    return {
        "start_ij": start,
        "ascent_end_ij": top,
        "platform_end_ij": platform_end,
        "end_ij": end,
        "axis": axis,
        "rise_cells": rise_cells,
    }


def _raster_cells(distance_m: float, label: str) -> int:
    cells = int(round(distance_m / GRID_SPACING_M))
    if cells < 1 or not math.isclose(
        cells * GRID_SPACING_M, distance_m, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError(f"{label} must be a positive multiple of the terrain spacing")
    return cells


def _apply_flight(terrain, spec: dict, points: dict, platform_cells: int, tread_cells: int):
    axis = points["axis"]
    lateral_axis = 1 - axis
    center = spec["start_ij"][lateral_axis]
    lateral_start = center - STAIR_WIDTH_CELLS // 2
    lateral_end = lateral_start + STAIR_WIDTH_CELLS
    start_axis = spec["start_ij"][axis]
    platform_end_offset = points["rise_cells"] + platform_cells
    final_offset = platform_end_offset + points["rise_cells"]

    for offset in range(final_offset + 1):
        if offset <= points["rise_cells"]:
            level = min(offset // tread_cells, spec["riser_count"])
        elif offset <= platform_end_offset:
            level = spec["riser_count"]
        else:
            down_steps = (offset - platform_end_offset) // tread_cells
            level = max(0, spec["riser_count"] - down_steps)

        height = level * spec["riser_height_m"]
        grid_index = start_axis + offset
        if axis == 0:
            terrain.hf[grid_index, lateral_start:lateral_end] = height
        else:
            terrain.hf[lateral_start:lateral_end, grid_index] = height


def _append_axis_segment(route: list, segments: list, target_ij, name: str, stage: str,
                         flight_id: str | None = None) -> dict:
    start = route[-1]
    target = tuple(int(value) for value in target_ij)
    changed_axes = [axis for axis in range(2) if start[axis] != target[axis]]
    if len(changed_axes) != 1:
        raise ValueError(f"Segment {name!r} must be a nonzero, axis-aligned grid segment")

    axis = changed_axes[0]
    direction = 1 if target[axis] > start[axis] else -1
    start_node = len(route) - 1
    for value in range(start[axis] + direction, target[axis] + direction, direction):
        point = list(route[-1])
        point[axis] = value
        route.append(tuple(point))

    segment = {
        "name": name,
        "stage": stage,
        "flight_id": flight_id,
        "path_node_start": start_node,
        "path_node_end": len(route) - 1,
        "start_grid_ij": list(start),
        "end_grid_ij": list(target),
    }
    segments.append(segment)
    return segment


def _segment_reference(segment: dict) -> dict:
    return {
        "path_node_start": segment["path_node_start"],
        "path_node_end": segment["path_node_end"],
    }


def build_course(device: str = "cpu"):
    """Return the official PARC SubTerrain, XYZ path tensor, and course manifest."""
    import numpy as np
    import torch

    terrain_util = _load_official_terrain_util()
    nx, ny = TERRAIN_SHAPE_IJ
    terrain = terrain_util.SubTerrain(
        "parc_deterministic_stair_course",
        x_dim=nx,
        y_dim=ny,
        dx=GRID_SPACING_M,
        dy=GRID_SPACING_M,
        min_x=0.0,
        min_y=0.0,
        device=device,
    )
    terrain.hf.zero_()

    platform_cells = _raster_cells(PLATFORM_LENGTH_M, "platform length")
    prepared_flights = []
    for source_spec in FLIGHT_SPECS:
        spec = dict(source_spec)
        tread_cells = _raster_cells(spec["tread_depth_m"], "tread depth")
        points = _flight_grid_points(spec, platform_cells, tread_cells)
        _apply_flight(terrain, spec, points, platform_cells, tread_cells)
        spec.update({
            "tread_cells": tread_cells,
            "platform_cells": platform_cells,
            "points": points,
        })
        prepared_flights.append(spec)

    # Freeze the raster at its authored heights; downstream randomization must
    # not turn these exact course elevations into ranges.
    terrain.hf_maxmin[..., 0] = terrain.hf
    terrain.hf_maxmin[..., 1] = terrain.hf

    route = [START_IJ]
    segments = []
    flight_records = []

    _append_axis_segment(
        route, segments, prepared_flights[0]["points"]["start_ij"],
        "start_approach", "flat_approach",
    )

    for spec in prepared_flights:
        points = spec["points"]
        flight_segments = {}
        for phase, target_key in (
            ("ascent", "ascent_end_ij"),
            ("platform", "platform_end_ij"),
            ("descent", "end_ij"),
        ):
            segment = _append_axis_segment(
                route,
                segments,
                points[target_key],
                f"{spec['flight_id']}_{phase}",
                phase,
                spec["flight_id"],
            )
            flight_segments[phase] = _segment_reference(segment)

        flight_records.append({
            "flight_id": spec["flight_id"],
            "orientation": spec["orientation"],
            "riser_count": spec["riser_count"],
            "riser_height_m": spec["riser_height_m"],
            "total_rise_m": spec["riser_count"] * spec["riser_height_m"],
            "tread_depth_m": spec["tread_depth_m"],
            "tread_cells": spec["tread_cells"],
            "width_cells": STAIR_WIDTH_CELLS,
            "width_m": STAIR_WIDTH_CELLS * GRID_SPACING_M,
            "high_platform_length_m": PLATFORM_LENGTH_M,
            "start_grid_ij": points["start_ij"],
            "ascent_end_grid_ij": points["ascent_end_ij"],
            "platform_end_grid_ij": points["platform_end_ij"],
            "end_grid_ij": points["end_ij"],
            "path_segments": flight_segments,
        })

        if spec["flight_id"] == "flight_1":
            _append_axis_segment(route, segments, (50, 50), "connector_1a", "flat_connector")
            _append_axis_segment(route, segments, (50, 85), "connector_1b", "flat_connector")
            _append_axis_segment(route, segments, (60, 85), "connector_1c", "flat_connector")
        elif spec["flight_id"] == "flight_2":
            _append_axis_segment(route, segments, (60, 115), "connector_2a", "flat_connector")
            _append_axis_segment(route, segments, (100, 115), "connector_2b", "flat_connector")

    _append_axis_segment(route, segments, END_IJ, "end_exit", "flat_exit")

    path_grid_ij = torch.tensor(route, dtype=torch.int64, device=device)
    path_nodes = terrain.get_xyz_point(path_grid_ij)

    route_np = path_grid_ij.detach().cpu().numpy()
    if len({tuple(point) for point in route}) != len(route):
        raise RuntimeError("Authored course route intersects itself")
    if not ((route_np >= 0).all() and (route_np[:, 0] < nx).all() and (route_np[:, 1] < ny).all()):
        raise RuntimeError("Authored course route leaves the terrain grid")

    path_np = path_nodes.detach().cpu().numpy()
    min_xy = terrain.min_point.detach().cpu().numpy()
    dxdy = terrain.dxdy.detach().cpu().numpy()
    for segment in segments:
        segment["start_xyz_m"] = [float(value) for value in path_np[segment["path_node_start"]]]
        segment["end_xyz_m"] = [float(value) for value in path_np[segment["path_node_end"]]]
        start = segment["path_node_start"]
        end = segment["path_node_end"]
        segment["planar_length_m"] = float(
            (end - start) * GRID_SPACING_M
        )

    for flight in flight_records:
        for key, grid_key in (
            ("start_xyz_m", "start_grid_ij"),
            ("ascent_end_xyz_m", "ascent_end_grid_ij"),
            ("platform_end_xyz_m", "platform_end_grid_ij"),
            ("end_xyz_m", "end_grid_ij"),
        ):
            ij = flight[grid_key]
            xyz = [
                float(min_xy[0] + ij[0] * dxdy[0]),
                float(min_xy[1] + ij[1] * dxdy[1]),
                float(terrain.hf[ij[0], ij[1]].item()),
            ]
            flight[key] = xyz

    deltas = path_np[1:] - path_np[:-1]
    planar_length = float(np.linalg.norm(deltas[:, :2], axis=1).sum())
    surface_length = float(np.linalg.norm(deltas, axis=1).sum())
    manifest = {
        "schema_version": 1,
        "artifact_kind": "static_parc_stair_course_geometry",
        "geometry_only": True,
        "motion_generation": "not performed",
        "randomized": False,
        "terrain": {
            "terrain_name": terrain.terrain_name,
            "grid_shape_ij": [nx, ny],
            "grid_spacing_xy_m": [GRID_SPACING_M, GRID_SPACING_M],
            "min_point_xy_m": [float(value) for value in min_xy],
            "max_sample_point_xy_m": [
                float(min_xy[0] + (nx - 1) * dxdy[0]),
                float(min_xy[1] + (ny - 1) * dxdy[1]),
            ],
            "grid_convention": "world_xy = min_point + (i, j) * dxdy; hf[i, j] is the sampled top height",
            "height_units": "meters",
        },
        "course": {
            "route_construction": "deterministic four-connected grid path; no A* shortcuts",
            "route_node_count": len(route),
            "route_start_grid_ij": list(START_IJ),
            "route_end_grid_ij": list(END_IJ),
            "route_start_xyz_m": [float(value) for value in path_np[0]],
            "route_end_xyz_m": [float(value) for value in path_np[-1]],
            "planar_length_m": planar_length,
            "surface_length_m": surface_length,
            "flat_start_and_end": True,
            "target_length_m": [30.0, 50.0],
            "stair_flight_count": len(flight_records),
        },
        "stair_flights": flight_records,
        "route_segments": segments,
    }
    return terrain, path_nodes, manifest


def build_platform_course(device: str = "cpu", variant: int = 0):
    """Return (official SubTerrain, XYZ path tensor, manifest) for a platform route.

    Variants 0/1/2 have heights 0.48/0.72/0.96 m and widths 4.8/6.4/8.0 m.
    task_waypoints are ordered route positions, including every top corner.
    Each 180-degree turn uses two spaced 90-degree corners, without retracing.
    """
    import numpy as np
    import torch

    if isinstance(variant, bool) or not isinstance(variant, int) or variant not in range(3):
        raise ValueError("variant must be one of 0, 1, 2")
    spec = PLATFORM_VARIANTS[variant]
    scenario_id = spec["scenario_id"]
    width_cells = _raster_cells(spec["width_m"], "platform width")
    tread_cells = _raster_cells(spec["tread_depth_m"], "tread depth")
    riser_count = 4
    stair_run = riser_count * tread_cells
    x0, x1, y0 = 40, 79, 20
    y1 = y0 + width_cells - 1
    entry_y, exit_y = y0 + 6, y1 - 5
    if spec["lateral_direction"] < 0:
        entry_y, exit_y = exit_y, entry_y
    middle_y = (entry_y + exit_y) // 2
    top_points = [
        (x0, entry_y), (x1 - 6, entry_y), (x1 - 6, middle_y),
        (x0 + 6, middle_y), (x0 + 6, exit_y), (x1, exit_y),
    ]
    ascent_start = (x0 - stair_run, entry_y)
    descent_end = (x1 + stair_run, exit_y)
    route_start = (ascent_start[0] - 15, entry_y)
    route_end = (descent_end[0] + 15, exit_y)
    terrain = _load_official_terrain_util().SubTerrain(
        scenario_id, x_dim=120, y_dim=80, dx=GRID_SPACING_M, dy=GRID_SPACING_M,
        min_x=0.0, min_y=0.0, device=device,
    )
    terrain.hf[x0:x1 + 1, y0:y1 + 1] = spec["height_m"]
    for stair_start, ascending in ((ascent_start, True), (top_points[-1], False)):
        for offset in range(stair_run + 1):
            level = offset // tread_cells
            if not ascending:
                level = riser_count - level
            x, y = stair_start[0] + offset, stair_start[1]
            terrain.hf[x, y - 6:y + 6] = level * spec["riser_height_m"]
    terrain.hf_maxmin[..., 0] = terrain.hf
    terrain.hf_maxmin[..., 1] = terrain.hf

    route, segments = [route_start], []
    waypoint_refs = [("course_start", "flat_approach", 0)]
    approach = _append_axis_segment(route, segments, ascent_start, "start_approach", "flat_approach")
    waypoint_refs.append(("ascent_start", "ascent", approach["path_node_end"]))
    ascent = _append_axis_segment(route, segments, top_points[0], "flight_1_ascent", "ascent", "flight_1")
    top_start_index = ascent["path_node_end"]
    waypoint_refs.append(("platform_entry", "platform_entry", top_start_index))
    for index, point in enumerate(top_points[1:], 1):
        segment = _append_axis_segment(
            route, segments, point, f"flight_1_platform_{index}", "platform", "flight_1",
        )
        name = "platform_exit" if index == len(top_points) - 1 else f"platform_turn_{index}"
        stage = "platform_exit" if index == len(top_points) - 1 else "platform_turn"
        waypoint_refs.append((name, stage, segment["path_node_end"]))
    top_end_index = len(route) - 1
    descent = _append_axis_segment(route, segments, descent_end, "flight_1_descent", "descent", "flight_1")
    waypoint_refs.append(("descent_end", "descent", descent["path_node_end"]))
    _append_axis_segment(route, segments, route_end, "end_exit", "flat_exit")
    waypoint_refs.append(("course_end", "flat_exit", len(route) - 1))

    grid = torch.tensor(route, dtype=torch.int64, device=device)
    if len(set(route)) != len(route) or not bool(((grid >= 0) & (grid < terrain.dims)).all()):
        raise RuntimeError("Platform route must stay in bounds without intersecting itself")
    path_nodes = terrain.get_xyz_point(grid)
    path_np = path_nodes.detach().cpu().numpy()
    for segment in segments:
        start, end = segment["path_node_start"], segment["path_node_end"]
        segment.update({
            "start_xyz_m": path_np[start].tolist(), "end_xyz_m": path_np[end].tolist(),
            "planar_length_m": (end - start) * GRID_SPACING_M,
        })

    waypoints = []
    for order, (name, stage, node_index) in enumerate(waypoint_refs):
        incoming = [route[node_index][i] - route[node_index - 1][i] for i in range(2)] if node_index else None
        outgoing = ([route[node_index + 1][i] - route[node_index][i] for i in range(2)]
                    if node_index < len(route) - 1 else None)
        turn = 0.0
        if incoming is not None and outgoing is not None:
            turn = math.degrees(math.atan2(
                incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
                sum(a * b for a, b in zip(incoming, outgoing)),
            ))
        waypoints.append({
            "waypoint_id": f"{scenario_id}:{name}", "order": order, "stage": stage,
            "path_node_index": node_index, "grid_ij": list(route[node_index]),
            "xy_m": path_np[node_index, :2].tolist(), "height_m": float(path_np[node_index, 2]),
            "xyz_m": path_np[node_index].tolist(), "signed_turn_degrees": turn,
            "incoming_heading_xy": incoming, "outgoing_heading_xy": outgoing,
        })
    corners = [point for point in waypoints if point["stage"] == "platform_turn"]
    top_length = (top_end_index - top_start_index) * GRID_SPACING_M
    flight = {
        "flight_id": "flight_1", "orientation": "x", "axis": 0, "direction": 1,
        "riser_count": riser_count, "riser_height_m": spec["riser_height_m"],
        "total_rise_m": spec["height_m"], "tread_depth_m": spec["tread_depth_m"],
        "tread_cells": tread_cells, "width_cells": STAIR_WIDTH_CELLS,
        "width_m": STAIR_WIDTH_CELLS * GRID_SPACING_M,
        "platform_width_m": spec["width_m"], "high_platform_length_m": (x1 - x0) * GRID_SPACING_M,
        "platform_route_length_m": top_length,
        "start_grid_ij": list(ascent_start), "ascent_end_grid_ij": list(top_points[0]),
        "platform_end_grid_ij": list(top_points[-1]), "end_grid_ij": list(descent_end),
        "start_xyz_m": ascent["start_xyz_m"], "ascent_end_xyz_m": ascent["end_xyz_m"],
        "platform_end_xyz_m": descent["start_xyz_m"], "end_xyz_m": descent["end_xyz_m"],
        "path_segments": {
            "ascent": _segment_reference(ascent),
            "platform": {"path_node_start": top_start_index, "path_node_end": top_end_index},
            "descent": _segment_reference(descent),
        },
    }
    deltas = np.diff(path_np, axis=0)
    manifest = {
        "schema_version": 1, "scenario_id": scenario_id, "variant": variant, "course_kind": "platform",
        "artifact_kind": "static_parc_platform_course_geometry", "geometry_only": True,
        "motion_generation": "not performed", "randomized": False,
        "terrain": {
            "terrain_name": terrain.terrain_name, "grid_shape_ij": [120, 80],
            "grid_spacing_xy_m": [GRID_SPACING_M, GRID_SPACING_M], "min_point_xy_m": [0.0, 0.0],
            "max_sample_point_xy_m": terrain.get_max_point().detach().cpu().tolist(),
            "grid_convention": "world_xy = min_point + (i, j) * dxdy; hf[i, j] is the sampled top height",
            "height_units": "meters",
        },
        "course": {
            "route_construction": "deterministic four-connected grid path with two wide platform U-turns",
            "route_node_count": len(route), "route_start_grid_ij": list(route_start),
            "route_end_grid_ij": list(route_end), "route_start_xyz_m": path_np[0].tolist(),
            "route_end_xyz_m": path_np[-1].tolist(),
            "planar_length_m": float(np.linalg.norm(deltas[:, :2], axis=1).sum()),
            "surface_length_m": float(np.linalg.norm(deltas, axis=1).sum()),
            "platform_route_length_m": top_length, "flat_start_and_end": True,
            "target_length_m": [25.0, 45.0], "stair_flight_count": 1,
        },
        "platforms": [{
            "platform_id": "main_platform", "height_m": spec["height_m"],
            "width_m": spec["width_m"], "length_m": (x1 - x0 + 1) * GRID_SPACING_M,
            "grid_min_ij": [x0, y0], "grid_max_ij_inclusive": [x1, y1],
            "path_node_start": top_start_index, "path_node_end": top_end_index,
        }],
        "stair_flights": [flight], "route_segments": segments, "task_waypoints": waypoints,
        "task_waypoint_contract": "ordered route positions; xyz_m includes the sampled terrain top height",
        "platform_turns": [{
            "turn_id": f"{scenario_id}:u_turn_{index + 1}", "kind": "two_spaced_right_angle_corners",
            "waypoint_ids": [point["waypoint_id"] for point in pair],
            "signed_turn_degrees": sum(point["signed_turn_degrees"] for point in pair),
        } for index, pair in enumerate((corners[:2], corners[2:]))],
    }
    return terrain, path_nodes, manifest


def save_course(output_dir: Path, terrain, path_nodes, manifest: dict) -> dict:
    """Write the heightfield, route, manifest, and static course preview."""
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
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_preview(preview_path, terrain, path_nodes, manifest)
    return {
        "terrain_and_route": terrain_path,
        "manifest": manifest_path,
        "preview": preview_path,
    }


def save_preview(path: Path, terrain, path_nodes, manifest: dict) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    height = terrain.hf.detach().cpu().numpy()
    nodes = path_nodes.detach().cpu().numpy()
    min_xy = terrain.min_point.detach().cpu().numpy()
    dxdy = terrain.dxdy.detach().cpu().numpy()
    extent = [
        min_xy[0], min_xy[0] + (height.shape[0] - 1) * dxdy[0],
        min_xy[1], min_xy[1] + (height.shape[1] - 1) * dxdy[1],
    ]
    planar_distance = np.concatenate((
        np.zeros(1), np.cumsum(np.linalg.norm(np.diff(nodes[:, :2], axis=0), axis=1))
    ))

    figure, (map_axis, elevation_axis) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    image = map_axis.imshow(
        height.T,
        origin="lower",
        extent=extent,
        cmap="terrain",
        interpolation="nearest",
        aspect="equal",
    )
    map_axis.plot(nodes[:, 0], nodes[:, 1], color="#17212b", linewidth=1.8, label="route")
    map_axis.scatter(*nodes[0, :2], marker="o", s=42, color="#17864a", label="start")
    map_axis.scatter(*nodes[-1, :2], marker="X", s=52, color="#bd3f35", label="end")
    for flight in manifest["stair_flights"]:
        center = np.mean(
            [flight["start_xyz_m"][:2], flight["end_xyz_m"][:2]], axis=0
        )
        map_axis.annotate(flight["flight_id"], center, fontsize=9, ha="center", va="bottom")
    for waypoint in manifest.get("task_waypoints", []):
        if not waypoint["stage"].startswith("platform"):
            continue
        xy = waypoint["xy_m"]
        map_axis.scatter(*xy, s=18, color="#bd3f35", zorder=4)
        map_axis.annotate(str(waypoint["order"]), xy, xytext=(4, 5), textcoords="offset points", fontsize=9)
    if manifest.get("course_kind") == "platform":
        for segment in manifest["route_segments"]:
            if segment["stage"] == "platform":
                start, end = segment["path_node_start"], segment["path_node_end"]
                middle = (start + end) // 2
                map_axis.annotate("", nodes[middle + 2, :2], xytext=nodes[middle - 2, :2],
                                  arrowprops={"arrowstyle": "->", "color": "#17212b", "lw": 1.8})
    map_axis.set_title("Course Top View")
    map_axis.set_xlabel("X (m)")
    map_axis.set_ylabel("Y (m)")
    map_axis.legend(loc="upper left", frameon=True)
    colorbar = figure.colorbar(image, ax=map_axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Top height (m)")

    elevation_axis.plot(planar_distance, nodes[:, 2], color="#235b8e", linewidth=1.8)
    elevation_axis.fill_between(planar_distance, 0.0, nodes[:, 2], color="#81b1c9", alpha=0.35)
    elevation_axis.set_title("Route Elevation")
    elevation_axis.set_xlabel("Planar route distance (m)")
    elevation_axis.set_ylabel("Top height (m)")
    elevation_axis.grid(True, linewidth=0.5, alpha=0.45)
    elevation_axis.set_xlim(0.0, planar_distance[-1])
    if manifest.get("course_kind") == "platform":
        figure.suptitle(
            f"{manifest['scenario_id']}: {manifest['course']['planar_length_m']:.1f} m route, "
            f"{manifest['course']['platform_route_length_m']:.1f} m on top\n"
            "Numbers identify ordered platform task waypoints"
        )
    elif manifest.get('course_kind') == 'diverse':
        figure.suptitle(f"{manifest['scenario_id']}: {manifest['course']['planar_length_m']:.1f} m route")
    else:
        figure.suptitle(
            f"PARC stair course: {manifest['course']['planar_length_m']:.1f} m planar, "
            f"{len(manifest['stair_flights'])} stair flights"
        )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu", help="Torch device for the terrain tensors")
    parser.add_argument("--kind", choices=("stairs", "platform"), default="stairs")
    parser.add_argument("--variant", type=int, choices=range(3), default=0, help="Platform variant: 0, 1, or 2")
    args = parser.parse_args(argv)

    if args.kind == "platform":
        terrain, path_nodes, manifest = build_platform_course(device=args.device, variant=args.variant)
        default_output = DEFAULT_PLATFORM_OUTPUT / manifest["scenario_id"]
    else:
        if args.variant != 0:
            parser.error("--variant requires --kind platform")
        terrain, path_nodes, manifest = build_course(device=args.device)
        default_output = DEFAULT_OUTPUT
    outputs = save_course(args.output_dir or default_output, terrain, path_nodes, manifest)
    print(json.dumps({
        "status": "geometry_only",
        "motion_generation": "not performed",
        "output_files": {name: str(path.resolve()) for name, path in outputs.items()},
        "planar_length_m": manifest["course"]["planar_length_m"],
        "surface_length_m": manifest["course"]["surface_length_m"],
        "stair_flight_count": len(manifest["stair_flights"]),
    }, indent=2))


if __name__ == "__main__":
    main()
