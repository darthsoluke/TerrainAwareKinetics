"""Bounded read-only closed-loop probe for the SOMA Terrain Preview websocket."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import numbers
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import numpy as np


PUBLISH_FRAMES = 8
IDLE_BLOCKS = 6
DEFAULT_BLOCKS = 24
MAX_BLOCKS = 64
MAX_SPEED_MPS = 3.0


class ProbeError(RuntimeError):
    pass


def finite_number(value):
    """Return a finite Python float, or None for missing/invalid values."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def finite_xyz(value):
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None
    return vector


def phase_for_block(block_index):
    if isinstance(block_index, bool) or not isinstance(block_index, int) or block_index < 0:
        raise ValueError("block_index must be a nonnegative integer")
    return "idle" if block_index < IDLE_BLOCKS else "straight"


def command_for_phase(phase, speed_mps):
    speed = finite_number(speed_mps)
    if speed is None or speed < 0 or speed > MAX_SPEED_MPS:
        raise ValueError(f"speed_mps must be finite and between 0 and {MAX_SPEED_MPS:g}")
    if phase == "idle":
        return {"velocity_xz": [0.0, 0.0], "heading": 0.0}
    if phase == "straight":
        return {"velocity_xz": [0.0, speed], "heading": 0.0}
    raise ValueError(f"Unknown phase label: {phase}")


def decode_published_block(metadata, payload, expected_vertex_count=None):
    """Validate the viewer's complete float32-le block and return its raw rows."""
    if not isinstance(metadata, dict):
        raise ValueError("Plan metadata must be an object")
    frame_count = metadata.get("frame_count")
    vertex_count = metadata.get("vertex_count")
    stride = metadata.get("frame_stride_floats")
    if (isinstance(frame_count, bool) or not isinstance(frame_count, int)
            or frame_count != PUBLISH_FRAMES):
        raise ValueError(f"Terrain plans must publish exactly {PUBLISH_FRAMES} frames")
    if (isinstance(vertex_count, bool) or not isinstance(vertex_count, int)
            or vertex_count < 1):
        raise ValueError("Plan vertex_count must be a positive integer")
    if expected_vertex_count is not None and vertex_count != expected_vertex_count:
        raise ValueError("Plan vertex_count differs from the terrain ready message")
    expected_stride = 3 + 3 * vertex_count
    if isinstance(stride, bool) or not isinstance(stride, int) or stride != expected_stride:
        raise ValueError("Plan frame_stride_floats does not match the SOMA geometry")
    if metadata.get("encoding") != "float32-le":
        raise ValueError("Plan encoding must be float32-le")
    expected_bytes = frame_count * stride * np.dtype("<f4").itemsize
    try:
        view = memoryview(payload)
    except TypeError as error:
        raise ValueError("Plan binary payload is not bytes-like") from error
    if view.nbytes != expected_bytes:
        raise ValueError(
            f"Plan binary payload has {view.nbytes} bytes; expected {expected_bytes}")
    values = np.frombuffer(view, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("Plan binary payload contains non-finite float32 values")
    return values.reshape(frame_count, stride).copy()


def debug_xyz_trace(debug, key, frame_count=PUBLISH_FRAMES):
    """Read the first published frames from a potentially longer generation trace."""
    if not isinstance(debug, dict) or key not in debug or debug[key] is None:
        return {"status": "missing", "available_frames": None, "values": None}
    try:
        trace = np.asarray(debug[key], dtype=np.float64)
    except (TypeError, ValueError):
        return {"status": "invalid", "available_frames": None, "values": None}
    if trace.ndim != 2 or trace.shape[1] != 3:
        return {"status": "invalid", "available_frames": None, "values": None}
    available = int(trace.shape[0])
    if available < frame_count:
        return {"status": "short", "available_frames": available, "values": None}
    published_count = debug.get("published_frame_count")
    if (published_count is not None and
            (isinstance(published_count, bool) or not isinstance(published_count, int)
             or published_count != frame_count)):
        return {"status": "published_frame_count_mismatch",
                "available_frames": available, "values": None}
    published = trace[:frame_count]
    if not np.isfinite(published).all():
        return {"status": "nonfinite", "available_frames": available, "values": None}
    return {"status": "ok", "available_frames": available, "values": published.tolist()}


def _spacing_xz(value):
    if isinstance(value, dict):
        x, z = value.get("x"), value.get("z", value.get("y"))
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = np.asarray(value, dtype=object).reshape(-1).tolist()
        if len(values) not in (1, 2):
            return None
        x, z = values[0], values[-1]
    else:
        x = z = value
    x, z = finite_number(x), finite_number(z)
    if x is None or z is None or x <= 0 or z <= 0:
        return None
    return np.asarray([x, z], dtype=np.float64)


def _heightfield_array(value):
    shape = None
    if isinstance(value, dict):
        shape = value.get("shape")
        value = value.get("data", value.get("values"))
    if value is None:
        raise ValueError("terrain heightfield is missing")
    array = np.asarray(value, dtype=np.float64)
    if shape is not None:
        dimensions = np.asarray(shape, dtype=np.int64).reshape(-1)
        if dimensions.size != 2 or np.any(dimensions <= 0):
            raise ValueError("terrain heightfield shape must contain two positive dimensions")
        if array.size != int(dimensions[0] * dimensions[1]):
            raise ValueError("terrain heightfield data does not match its shape")
        array = array.reshape(tuple(int(item) for item in dimensions))
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError("terrain heightfield must be a nonempty 2D grid")
    return array


class TerrainHeightfield:
    """Nearest-cell world-Y sampler that rejects unknown and out-of-map queries."""

    def __init__(self, heightfield, origin, spacing, valid):
        self.heightfield = heightfield
        self.origin = origin
        self.spacing = spacing
        self.valid = valid

    @classmethod
    def from_config(cls, config):
        scene = config.get("terrain_scene") if isinstance(config, dict) else None
        if not isinstance(scene, dict):
            return None, {"status": "missing", "reason": "terrain_scene is absent"}
        try:
            heightfield = _heightfield_array(
                scene.get("hf", scene.get("heightfield_y_m")))
            origin = finite_xyz(scene.get("min_point", scene.get("min_point_world_xyz_m")))
            spacing = _spacing_xz(scene.get("dx", scene.get("dx_m")))
            if origin is None:
                raise ValueError("terrain origin must contain three finite world coordinates")
            if spacing is None:
                raise ValueError("terrain spacing must be finite and positive")
            known = np.isfinite(heightfield)
            valid_source = scene.get("heightfield_valid")
            if valid_source is not None:
                valid = np.asarray(valid_source)
                if valid.shape != heightfield.shape or valid.dtype != np.bool_:
                    raise ValueError("heightfield_valid must be a boolean grid matching the heightfield")
                known &= valid
            if not known.any():
                return None, {
                    "status": "invalid", "reason": "heightfield has no known cells",
                    "known_points": 0, "total_points": int(heightfield.size),
                    "shape": [int(size) for size in heightfield.shape],
                }
            return cls(heightfield, origin, spacing, known), {
                "status": "valid", "known_points": int(known.sum()),
                "total_points": int(known.size),
                "known_fraction": float(known.mean()),
                "shape": [int(size) for size in heightfield.shape],
                "validity_source": ("heightfield_valid_and_finite" if valid_source is not None
                                    else "finite_heightfield_cells"),
            }
        except (TypeError, ValueError, OverflowError) as error:
            return None, {"status": "invalid", "reason": str(error)}

    def sample(self, x, z):
        x, z = finite_number(x), finite_number(z)
        if x is None or z is None:
            return None, "invalid_query"
        column = math.floor((x - float(self.origin[0])) / float(self.spacing[0]) + 0.5)
        row = math.floor((z - float(self.origin[2])) / float(self.spacing[1]) + 0.5)
        if row < 0 or row >= self.heightfield.shape[0] or column < 0 or column >= self.heightfield.shape[1]:
            return None, "outside_heightfield"
        if not self.valid[row, column]:
            return None, "unknown_heightfield_cell"
        height = finite_number(self.heightfield[row, column])
        if height is None:
            return None, "unknown_heightfield_cell"
        return height, "valid"


def numeric_summary(values):
    numbers_ = [number for value in values
                if (number := finite_number(value)) is not None]
    if not numbers_:
        return {"count": 0, "mean": None, "median": None, "p95": None,
                "min": None, "max": None}
    return {
        "count": len(numbers_),
        "mean": float(np.mean(numbers_)),
        "median": float(np.median(numbers_)),
        "p95": float(np.percentile(numbers_, 95)),
        "min": float(min(numbers_)),
        "max": float(max(numbers_)),
    }


def summarize_timings(rows):
    fields = ("response_ms", "inference_ms", "skin_ms")

    def group_summary(selected):
        result = {"blocks": len(selected)}
        for field in fields:
            result[field] = numeric_summary(row.get(field) for row in selected)
        return result

    phases = {phase: group_summary([row for row in rows if row.get("phase") == phase])
              for phase in ("idle", "straight")}
    generations = {
        label: group_summary([row for row in rows if row.get("generation_status") == label])
        for label in ("neural", "cached", "unknown")
    }
    return {"by_phase": phases, "by_generation": generations}


def summarize_displacements(rows):
    internal_by_phase = {"idle": [], "straight": []}
    boundary_by_phase = {"idle": [], "straight": []}
    boundaries = []
    previous_root = None
    previous_request_id = None
    for row in rows:
        phase = row.get("phase")
        roots = [finite_xyz(value) for value in row.get("published_root_xyz_m", [])]
        roots = [value for value in roots if value is not None]
        if phase not in internal_by_phase or not roots:
            continue
        internal = [float(np.linalg.norm(right - left))
                    for left, right in zip(roots, roots[1:])]
        internal_by_phase[phase].extend(internal)
        if previous_root is not None:
            delta = roots[0] - previous_root
            boundary_by_phase[phase].append(float(np.linalg.norm(delta)))
            boundaries.append({
                "from_request_id": previous_request_id,
                "to_request_id": row.get("request_id"),
                "phase": phase,
                "previous_acknowledged_root_xyz_m": previous_root.tolist(),
                "current_obs_root_xyz_m": row.get("current_obs_root_xyz_m"),
                "debug_observation_root_xyz_m": row.get("debug_observation_root_xyz_m"),
                "debug_observation_gap_vs_played_state_xyz_m": row.get("debug_observation_gap_vs_played_state_xyz_m"),
                "first_published_root_delta_xyz_m": delta.tolist(),
                "first_published_root_displacement_m": float(np.linalg.norm(delta)),
            })
        previous_root = roots[-1]
        previous_request_id = row.get("request_id")
    return {
        "internal_frame_displacement_m": {
            phase: numeric_summary(values) for phase, values in internal_by_phase.items()},
        "block_boundary_displacement_m": {
            phase: numeric_summary(values) for phase, values in boundary_by_phase.items()},
        "state_boundaries": boundaries,
        "interpretation": "Root displacement only; this is not a motion-quality verdict.",
    }


def _endpoint_urls(raw_url):
    parsed = urlsplit(raw_url)
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.hostname:
        raise ValueError("--url must be an http(s) or ws(s) server URL")
    secure = parsed.scheme in {"https", "wss"}
    http_scheme, ws_scheme = (("https", "wss") if secure else ("http", "ws"))
    path = parsed.path.rstrip("/")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host + (f":{parsed.port}" if parsed.port is not None else "")
    config_url = urlunsplit((http_scheme, netloc, f"{path}/api/config", "", ""))
    websocket_url = urlunsplit((ws_scheme, netloc, f"{path}/ws", "", ""))
    safe_url = urlunsplit((http_scheme, netloc, path, "", ""))
    return config_url, websocket_url, safe_url


async def _receive_json(socket, description):
    message = await socket.receive()
    if message.type != aiohttp.WSMsgType.TEXT:
        raise ProbeError(f"Expected JSON {description}, received websocket message {message.type}")
    try:
        result = json.loads(message.data)
    except (TypeError, ValueError) as error:
        raise ProbeError(f"Invalid JSON {description}") from error
    if not isinstance(result, dict):
        raise ProbeError(f"JSON {description} must be an object")
    if result.get("type") == "error":
        raise ProbeError(str(result.get("message", "server error")))
    return result


def _trace_vector(debug, key):
    value = finite_xyz(debug.get(key)) if isinstance(debug, dict) else None
    return value


def _generation_status(metadata):
    generation = metadata.get("generation")
    neural = generation.get("neural_generation") if isinstance(generation, dict) else None
    if not isinstance(neural, bool):
        return None, "unknown", None
    label = "neural" if neural else "cached"
    reason = generation.get("reason") if isinstance(generation, dict) else None
    return neural, label, reason if isinstance(reason, str) else None


def _make_frame_rows(roots, root_trace, decoded_trace, terrain):
    frames = []
    for index, root in enumerate(roots[:, :3]):
        ground_y, ground_status = terrain.sample(root[0], root[2]) if terrain else (None, "heightfield_unavailable")
        gap = float(root[1] - ground_y) if ground_y is not None else None
        stage_xyz = root_trace["values"][index] if root_trace["status"] == "ok" else None
        decoded_xyz = decoded_trace["values"][index] if decoded_trace["status"] == "ok" else None
        frames.append({
            "frame_index": index,
            "published_root_xyz_m": root.tolist(),
            "predicted_root_xyz_m": stage_xyz,
            "predicted_root_y_m": stage_xyz[1] if stage_xyz is not None else None,
            "final_decoded_root_xyz_m": decoded_xyz,
            "final_decoded_y_m": decoded_xyz[1] if decoded_xyz is not None else None,
            "ground_y_m": ground_y,
            "ground_sample_status": ground_status,
            "ground_root_gap_y_m": gap,
        })
    return frames


def _new_report(url, blocks, speed_mps, paced):
    _, _, safe_url = _endpoint_urls(url)
    return {
        "schema_version": "soma_online_probe_v1",
        "status": "running",
        "url": safe_url,
        "execution_mode": "paced_websocket" if paced else "offline_accelerated_websocket",
        "offline_accelerated": not paced,
        "read_only": True,
        "actuating": False,
        "probe": {
            "total_blocks": blocks,
            "idle_blocks": min(IDLE_BLOCKS, blocks),
            "straight_blocks": max(0, blocks - IDLE_BLOCKS),
            "straight_speed_mps": speed_mps,
            "publish_frames_per_block": PUBLISH_FRAMES,
            "paced_playback": paced,
            "playback_note": ("Wait one published block duration before each ACK."
                              if paced else
                              "No wall-clock playback pacing; this run is offline accelerated."),
            "max_blocks": MAX_BLOCKS,
        },
        "terrain_knownpoints": {"status": "not_read", "known_points": None,
                                "total_points": None, "known_fraction": None},
        "rows": [],
        "motion_quality_assessed": False,
        "interpretation_note": (
            "Root-to-heightfield gaps and root displacements are descriptive diagnostics only; "
            "they do not establish contact quality or overall motion quality."),
        "summary": {},
        "error": None,
    }


async def run_probe(url="http://127.0.0.1:28780", *, blocks=DEFAULT_BLOCKS,
                    speed_mps=1.0, paced=True):
    if isinstance(blocks, bool) or not isinstance(blocks, int) or not 1 <= blocks <= MAX_BLOCKS:
        raise ValueError(f"blocks must be between 1 and {MAX_BLOCKS}")
    speed = finite_number(speed_mps)
    if speed is None or speed < 0 or speed > MAX_SPEED_MPS:
        raise ValueError(f"speed_mps must be finite and between 0 and {MAX_SPEED_MPS:g}")
    if not isinstance(paced, bool):
        raise ValueError("paced must be boolean")
    config_url, websocket_url, _ = _endpoint_urls(url)
    report = _new_report(url, blocks, speed, paced)
    timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=180)
    terrain = None
    timing_rows = report["rows"]

    try:
        async with aiohttp.ClientSession(timeout=timeout) as client:
            async with client.ws_connect(
                    websocket_url, heartbeat=20,
                    timeout=aiohttp.ClientWSTimeout(ws_receive=180),
                    max_msg_size=8 * 1024 * 1024) as socket:
                initial_ready = await _receive_json(socket, "initial ready message")
                if initial_ready.get("type") != "ready":
                    raise ProbeError("The websocket did not begin with a ready message")

                try:
                    async with client.get(config_url) as response:
                        if response.status != 200:
                            report["terrain_knownpoints"] = {
                                "status": "unavailable", "known_points": None,
                                "total_points": None, "known_fraction": None,
                                "reason": f"/api/config returned HTTP {response.status}",
                            }
                        else:
                            config = await response.json()
                            terrain, knownpoints = TerrainHeightfield.from_config(config)
                            report["terrain_knownpoints"] = knownpoints
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
                    report["terrain_knownpoints"] = {
                        "status": "unavailable", "known_points": None,
                        "total_points": None, "known_fraction": None,
                        "reason": str(error),
                    }

                await socket.send_json({"type": "reset", "mode": "terrain"})
                ready = await _receive_json(socket, "terrain reset ready message")
                if ready.get("type") != "ready" or ready.get("mode") != "terrain":
                    raise ProbeError("Terrain reset did not return a terrain ready message")
                initial_root = finite_xyz(ready.get("root"))
                if initial_root is None:
                    raise ProbeError("Terrain ready message omitted a finite root XYZ")
                fps = finite_number(ready.get("fps"))
                if fps is None or fps <= 0:
                    raise ProbeError("Terrain ready message omitted a positive finite fps")
                vertex_count = ready.get("vertex_count")
                if (isinstance(vertex_count, bool) or not isinstance(vertex_count, int)
                        or vertex_count < 1):
                    raise ProbeError("Terrain ready message omitted a positive vertex_count")
                current_observation_root = initial_root.copy()
                previous_ack_root = None

                for index in range(blocks):
                    phase = phase_for_block(index)
                    command = command_for_phase(phase, speed)
                    request_id = index + 1
                    plan = {"type": "plan", "mode": "terrain", "request_id": request_id,
                            **command}
                    started = time.perf_counter()
                    await socket.send_json(plan)
                    metadata = await _receive_json(socket, f"plan {request_id} metadata")
                    response_id = metadata.get("request_id")
                    if (metadata.get("type") != "plan" or metadata.get("mode") != "terrain"
                            or isinstance(response_id, bool) or not isinstance(response_id, int)
                            or response_id != request_id):
                        raise ProbeError(f"Plan {request_id} metadata has an unexpected type or request_id")
                    binary = await socket.receive()
                    response_ms = (time.perf_counter() - started) * 1000.0
                    if binary.type != aiohttp.WSMsgType.BINARY:
                        raise ProbeError(f"Plan {request_id} did not return a binary frame block")
                    raw = decode_published_block(metadata, binary.data, vertex_count)
                    roots = raw[:, :3]
                    debug = metadata.get("debug") if isinstance(metadata.get("debug"), dict) else {}
                    root_trace = debug_xyz_trace(debug, "root_stage_world_xyz_m")
                    decoded_trace = debug_xyz_trace(debug, "decoded_world_xyz_m")
                    debug_observation = _trace_vector(debug, "observation_root_world_xyz_m")
                    observation_gap = (debug_observation - current_observation_root
                                       if debug_observation is not None else None)
                    neural, generation_status, generation_reason = _generation_status(metadata)
                    frame_rows = _make_frame_rows(roots, root_trace, decoded_trace, terrain)
                    boundary_delta = (roots[0] - previous_ack_root
                                      if previous_ack_root is not None else None)
                    row = {
                        "request_id": request_id,
                        "phase": phase,
                        "command": command,
                        "frame_count": int(metadata["frame_count"]),
                        "vertex_count": int(metadata["vertex_count"]),
                        "frame_stride_floats": int(metadata["frame_stride_floats"]),
                        "response_ms": finite_number(response_ms),
                        "inference_ms": finite_number(metadata.get("inference_ms")),
                        "skin_ms": finite_number(metadata.get("skin_ms")),
                        "neural_generation": neural,
                        "generation_status": generation_status,
                        "generation_reason": generation_reason,
                        "current_obs_root_xyz_m": current_observation_root.tolist(),
                        "current_obs_root_y_m": float(current_observation_root[1]),
                        "debug_observation_root_xyz_m": (
                            debug_observation.tolist() if debug_observation is not None else None),
                        "debug_observation_status": "ok" if debug_observation is not None else "missing_or_invalid",
                        "debug_observation_gap_vs_played_state_xyz_m": (
                            observation_gap.tolist() if observation_gap is not None else None),
                        "debug_observation_gap_vs_played_state_y_m": (
                            float(observation_gap[1]) if observation_gap is not None else None),
                        "root_stage_trace_status": root_trace["status"],
                        "root_stage_trace_available_frames": root_trace["available_frames"],
                        "decoded_trace_status": decoded_trace["status"],
                        "decoded_trace_available_frames": decoded_trace["available_frames"],
                        "published_root_xyz_m": roots.tolist(),
                        "predicted_root_y_m": [frame["predicted_root_y_m"] for frame in frame_rows],
                        "final_decoded_y_m": [frame["final_decoded_y_m"] for frame in frame_rows],
                        "frames": frame_rows,
                        "boundary_displacement_xyz_m": (
                            boundary_delta.tolist() if boundary_delta is not None else None),
                        "boundary_displacement_m": (
                            float(np.linalg.norm(boundary_delta)) if boundary_delta is not None else None),
                        "internal_frame_displacement_m": [
                            float(np.linalg.norm(right - left))
                            for left, right in zip(roots, roots[1:])],
                        "terrain_known_points": report["terrain_knownpoints"].get("known_points"),
                        "acknowledged": False,
                    }
                    timing_rows.append(row)

                    if paced:
                        await asyncio.sleep(PUBLISH_FRAMES / fps)
                    await socket.send_json({"type": "played", "request_id": request_id,
                                            "frame_count": PUBLISH_FRAMES})
                    acknowledgement = await _receive_json(socket, f"plan {request_id} playback acknowledgement")
                    acknowledged_id = acknowledgement.get("request_id")
                    acknowledged_frames = acknowledgement.get("frame_count")
                    if (acknowledgement.get("type") != "played"
                            or isinstance(acknowledged_id, bool)
                            or not isinstance(acknowledged_id, int)
                            or acknowledged_id != request_id
                            or isinstance(acknowledged_frames, bool)
                            or not isinstance(acknowledged_frames, int)
                            or acknowledged_frames != PUBLISH_FRAMES):
                        raise ProbeError(f"Plan {request_id} was not acknowledged as a full {PUBLISH_FRAMES}-frame block")
                    row["acknowledged"] = True
                    previous_ack_root = roots[-1].copy()
                    current_observation_root = previous_ack_root.copy()

        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"

    report["summary"] = {
        "received_blocks": len(timing_rows),
        "acknowledged_blocks": sum(bool(row.get("acknowledged")) for row in timing_rows),
        "timings": summarize_timings(timing_rows),
        "displacements": summarize_displacements(
            [row for row in timing_rows if row.get("acknowledged") is True]),
        "ground_root_gap_y_m": numeric_summary(
            frame.get("ground_root_gap_y_m") for row in timing_rows for frame in row["frames"]),
        "ground_root_gap_status": (
            "available" if any(frame.get("ground_root_gap_y_m") is not None
                               for row in timing_rows for frame in row["frames"])
            else "unavailable_or_no_valid_root_samples"),
    }
    report["state_boundaries"] = report["summary"]["displacements"]["state_boundaries"]
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:28780",
                        help="SOMA terrain viewer base URL")
    parser.add_argument("--output", type=Path, default=Path("-"),
                        help="JSON output path, or '-' for stdout")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS,
                        help=f"total block count, including the first {IDLE_BLOCKS} idle blocks (1-{MAX_BLOCKS})")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="straight-phase +Z command speed in m/s (0-3)")
    parser.add_argument("--no-pace", action="store_true",
                        help="ACK received blocks immediately; label the run offline accelerated")
    return parser


def _write_json(report, destination):
    encoded = json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    if destination == Path("-"):
        sys.stdout.write(encoded)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.blocks <= MAX_BLOCKS:
        parser.error(f"--blocks must be between 1 and {MAX_BLOCKS}")
    speed = finite_number(args.speed)
    if speed is None or speed < 0 or speed > MAX_SPEED_MPS:
        parser.error(f"--speed must be finite and between 0 and {MAX_SPEED_MPS:g}")
    try:
        report = asyncio.run(run_probe(args.url, blocks=args.blocks, speed_mps=speed,
                                       paced=not args.no_pace))
        _write_json(report, args.output)
    except Exception as error:
        fallback = {
            "schema_version": "soma_online_probe_v1", "status": "failed",
            "read_only": True, "actuating": False,
            "error": f"{type(error).__name__}: {error}", "rows": [],
            "motion_quality_assessed": False,
        }
        try:
            _write_json(fallback, args.output)
        except Exception as write_error:
            sys.stderr.write(f"Could not write probe JSON: {write_error}\n")
            sys.stdout.write(json.dumps(fallback, ensure_ascii=True, allow_nan=False) + "\n")
        return 1
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
