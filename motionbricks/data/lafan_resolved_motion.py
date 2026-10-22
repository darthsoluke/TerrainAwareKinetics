"""Bounded decoder for LAFAN1-Resolved's full-channel BVH exports."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


_MOTIONBRICKS_ROOT = Path(__file__).resolve().parents[2]
_KIMODO_ROOT = _MOTIONBRICKS_ROOT / "out" / "vendor" / "kimodo"
_EXPECTED_CHANNELS = (
    "Xposition", "Yposition", "Zposition", "Zrotation", "Yrotation", "Xrotation"
)
_FK_BATCH_SIZE = 512


def _load_bvh_class():
    parser_path = _KIMODO_ROOT / "kimodo" / "skeleton" / "bvh.py"
    if not parser_path.is_file():
        package_spec = importlib.util.find_spec("kimodo")
        if package_spec is None or package_spec.origin is None:
            raise ImportError("The official Kimodo BVH parser is required")
        parser_path = Path(package_spec.origin).parent / "skeleton" / "bvh.py"
    spec = importlib.util.spec_from_file_location("_lafan_resolved_kimodo_bvh", parser_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official Kimodo BVH parser at {parser_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Bvh


Bvh = _load_bvh_class()


def _distance_scale(distance_unit):
    scales = {"cm": 0.01, "m": 1.0}
    if distance_unit not in scales:
        raise ValueError("distance_unit must be explicitly set to 'cm' or 'm'")
    return scales[distance_unit]


def _read_header(stream):
    lines = []
    for raw_line in stream:
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("BVH header is not UTF-8") from error
        lines.append(line)
        if line.strip().startswith("Frame Time:"):
            break
    else:
        raise ValueError("BVH is missing its Frame Time header")

    try:
        parser = Bvh("".join(lines), backend="graph")
        frame_count = parser.nframes
        frame_time = parser.frame_time
    except (IndexError, LookupError, StopIteration, ValueError) as error:
        raise ValueError("BVH hierarchy or motion header is malformed") from error
    if frame_count <= 0 or not math.isfinite(frame_time) or frame_time <= 0:
        raise ValueError("BVH frame count and frame time must be positive and finite")
    source_fps = 1.0 / frame_time
    if not math.isfinite(source_fps):
        raise ValueError("BVH source frame rate must be finite")
    return parser, frame_count, source_fps


def _read_skeleton(parser, scale):
    if len(list(parser.root.filter("ROOT"))) != 1:
        raise ValueError("BVH hierarchy must contain exactly one ROOT")
    joints = parser.get_joints()
    names = [joint.name for joint in joints]
    if not names:
        raise ValueError("BVH contains no root joint")
    if len(names) != len(set(names)):
        raise ValueError("BVH skeleton contains duplicate joint names")

    parents = []
    parent_names = []
    offsets = []
    raw_offsets = []
    channels = []
    for index, joint in enumerate(joints):
        parent = None if joint.parent == parser.root else joint.parent
        if index == 0 and parent is not None:
            raise ValueError("BVH hierarchy is not rooted at its first joint")
        if index > 0 and (parent is None or parent.name not in names[:index]):
            raise ValueError(f"BVH joint {joint.name} has an invalid parent")
        parent_names.append(None if parent is None else parent.name)
        parents.append(-1 if parent is None else names.index(parent.name))

        try:
            offset = np.asarray([float(value) for value in joint["OFFSET"][:3]], dtype=np.float64)
            joint_channels = tuple(parser.joint_channels(joint.name))
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid offset or channels for joint {joint.name}") from error
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError(f"Invalid OFFSET for joint {joint.name}")
        if joint_channels != _EXPECTED_CHANNELS:
            raise ValueError(
                f"Unsupported channel order for joint {joint.name}: {joint_channels}; "
                f"expected {_EXPECTED_CHANNELS}"
            )
        raw_offsets.append(offset)
        offsets.append(offset * scale)
        channels.append(joint_channels)

    if sum(parent == -1 for parent in parents) != 1:
        raise ValueError("BVH hierarchy must contain exactly one root")
    return {
        "names": names,
        "parents": parents,
        "parent_names": parent_names,
        "header_offsets": np.asarray(offsets),
        "header_offsets_raw": np.asarray(raw_offsets),
        "channels": channels,
    }


def _read_motion(stream, frame_count, joint_count):
    channel_count = joint_count * len(_EXPECTED_CHANNELS)
    try:
        values = np.loadtxt(stream, dtype=np.float64, ndmin=2)
    except (ValueError, OSError) as error:
        raise ValueError("BVH motion rows must contain only numeric channel values") from error
    if values.shape != (frame_count, channel_count):
        raise ValueError(
            f"BVH motion data has shape {values.shape}; expected "
            f"({frame_count}, {channel_count})"
        )
    if not np.isfinite(values).all():
        raise ValueError("BVH motion contains non-finite channel values")
    return values.reshape(frame_count, joint_count, len(_EXPECTED_CHANNELS))


def _kimodo_fk(local_rot_mats, local_positions, parents):
    if str(_KIMODO_ROOT) not in sys.path:
        sys.path.insert(0, str(_KIMODO_ROOT))
    try:
        import torch
        from kimodo.skeleton.kinematics import batch_rigid_transform
    except ImportError as error:
        raise ImportError("Kimodo FK and PyTorch are required to decode BVH motion") from error

    parent_tensor = torch.as_tensor(parents, dtype=torch.long)
    global_rots = np.empty_like(local_rot_mats)
    posed_joints = np.empty_like(local_positions)
    for start in range(0, len(local_positions), _FK_BATCH_SIZE):
        stop = min(start + _FK_BATCH_SIZE, len(local_positions))
        rotations = torch.from_numpy(local_rot_mats[start:stop])
        positions = torch.from_numpy(local_positions[start:stop])
        posed, global_rotation = batch_rigid_transform(
            rotations, positions, parent_tensor, root_idx=0
        )
        global_rots[start:stop] = global_rotation.cpu().numpy()
        posed_joints[start:stop] = posed.cpu().numpy()
    return global_rots, posed_joints


def _accumulate_source_positions(local_positions, parents):
    """Build unrotated absolute coordinates expected by Kimodo rigid-transform FK."""
    absolute_positions = np.empty_like(local_positions)
    for joint_index, parent_index in enumerate(parents):
        if parent_index == -1:
            absolute_positions[:, joint_index] = local_positions[:, joint_index]
        else:
            absolute_positions[:, joint_index] = (
                absolute_positions[:, parent_index] + local_positions[:, joint_index]
            )
    return absolute_positions


def decode_lafan_resolved_bvh(path, *, distance_unit):
    """Decode all joint channels and Kimodo FK results from one BVH file.

    ``distance_unit`` is mandatory; supported source units are ``"cm"`` and
    ``"m"``. BVH OFFSET values describe the first frame's local positions in
    this exporter and are metadata only; they are not added to motion channels.
    """
    scale = _distance_scale(distance_unit)
    with Path(path).open("rb") as stream:
        parser, frame_count, source_fps = _read_header(stream)
        source_skeleton = _read_skeleton(parser, scale)
        data = _read_motion(stream, frame_count, len(source_skeleton["names"]))

    local_positions = data[..., :3] * scale
    # BVH declares Z, Y, X rotations; uppercase SciPy order is intrinsic.
    local_rot_mats = Rotation.from_euler(
        "ZYX", data[..., 3:].reshape(-1, 3), degrees=True
    ).as_matrix().reshape(frame_count, len(source_skeleton["names"]), 3, 3)
    global_rot_mats, posed_joints = _kimodo_fk(
        local_rot_mats,
        _accumulate_source_positions(local_positions, source_skeleton["parents"]),
        source_skeleton["parents"],
    )
    return {
        "source_skeleton": source_skeleton,
        "source_fps": source_fps,
        "distance_unit": distance_unit,
        "local_rot_mats": local_rot_mats,
        "local_positions": local_positions,
        "global_rot_mats": global_rot_mats,
        "posed_joints": posed_joints,
    }


def decode_geno_bind_bvh(path, *, distance_unit):
    """Decode Geno_bind with its actual channel rotations, not identity pose."""
    return decode_lafan_resolved_bvh(path, distance_unit=distance_unit)
