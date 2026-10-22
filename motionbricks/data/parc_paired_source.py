"""Restricted bridge from PARC motion-terrain samples to causal MB inputs."""

from dataclasses import dataclass
import io
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

try:
    from numpy._core import multiarray as _numpy_multiarray, numeric as _numpy_numeric
except ImportError:
    from numpy.core import multiarray as _numpy_multiarray, numeric as _numpy_numeric

from motionbricks.data.local_terrain_observation import (
    LocalTerrainObservation,
    LocalTerrainObservationConfig,
    extract_local_terrain_observation,
)


_MAX_PICKLE_BYTES = 128 * 1024 * 1024
_MAX_FRAMES = 1_000_000
_ROOT_BASIS_QUAT_XYZW = np.array(
    [-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float64
)
_CONTACT_BODY_NAMES = (
    "pelvis", "torso", "head", "right_upper_arm", "right_lower_arm",
    "right_hand", "left_upper_arm", "left_lower_arm", "left_hand",
    "right_thigh", "right_shin", "right_foot", "left_thigh", "left_shin",
    "left_foot",
)


class _ArrayUnpickler(pickle.Unpickler):
    """Load primitive containers and NumPy arrays without importing source code."""

    _ALLOWED = {
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("numpy.core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
        ("numpy.core.multiarray", "scalar"): _numpy_multiarray.scalar,
        ("numpy._core.multiarray", "scalar"): _numpy_multiarray.scalar,
        ("numpy.core.numeric", "_frombuffer"): _numpy_numeric._frombuffer,
        ("numpy._core.numeric", "_frombuffer"): _numpy_numeric._frombuffer,
    }

    def find_class(self, module, name):
        try:
            return self._ALLOWED[module, name]
        except KeyError as error:
            raise pickle.UnpicklingError(f"Unapproved pickle global: {module}.{name}") from error


@dataclass(frozen=True)
class PARCPairedSource:
    """PARC-body-frame motion; this is not a SOMA/generic-root retarget.

    The root quaternion's world basis is converted to Y-up, while local joint
    quaternions and PARC's local +X forward convention remain source-native.
    """

    source_path: str
    fps: float
    root_positions_world_xyz_m: np.ndarray
    root_rotations_world_xyzw: np.ndarray
    local_joint_rotations_xyzw: np.ndarray
    source_body_contacts: np.ndarray
    contact_body_names: tuple[str, ...]
    heightfield_y_m: np.ndarray
    min_point_world_xyz_m: np.ndarray
    dx_m: float
    frame_observations: tuple[LocalTerrainObservation, ...]
    source_hf_mask_inds: tuple[np.ndarray, ...] | None = None
    training_ready: bool = False
    contact_provenance: str = "source_labels_not_verified_physical_truth"
    retargeted_to_soma: bool = False


def _restricted_load(payload: bytes):
    if len(payload) > _MAX_PICKLE_BYTES:
        raise ValueError("Pickle payload exceeds the bounded inspection limit")
    return _ArrayUnpickler(io.BytesIO(payload)).load()


def _decode_section(container: dict, key: str) -> dict:
    value = container.get(key)
    if isinstance(value, bytes):
        value = _restricted_load(value)
    if not isinstance(value, dict):
        raise ValueError(f"PARC sample is missing dictionary section {key!r}")
    return value


def _finite_array(value, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape [frames, {shape_tail}]")
    if not 0 < len(array) <= _MAX_FRAMES or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain a bounded, finite frame sequence")
    return np.ascontiguousarray(array)


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack((lw * rx + lx * rw + ly * rz - lz * ry,
                     lw * ry - lx * rz + ly * rw + lz * rx,
                     lw * rz + lx * ry - ly * rx + lz * rw,
                     lw * rw - lx * rx - ly * ry - lz * rz), axis=-1)


def _normalize_quaternions(quaternions: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms < 1e-8):
        raise ValueError(f"{name} contains a zero or invalid quaternion")
    return np.ascontiguousarray(quaternions / norms)


def _yaw_from_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = quaternion
    # PARC's root heading is local +X; MB local terrain commands face +Z at yaw 0.
    forward_x = 1.0 - 2.0 * (y * y + z * z)
    forward_z = 2.0 * (x * z - w * y)
    if forward_x * forward_x + forward_z * forward_z < 1e-10:
        raise ValueError("Root orientation has no stable projected forward heading")
    return float(np.arctan2(forward_x, forward_z))


def _make_observation(
    heightfield_y_m: np.ndarray,
    min_point_world_xyz_m: np.ndarray,
    dx_m: float,
    root_position_world_xyz_m: np.ndarray,
    root_rotation_world_xyzw: np.ndarray,
    command_direction_local_xz: np.ndarray,
    config: LocalTerrainObservationConfig | None,
    sensor_heightfield_valid: np.ndarray | None = None,
) -> LocalTerrainObservation:
    return extract_local_terrain_observation(
        heightfield_y_m,
        min_point_world_xyz_m,
        dx_m,
        root_position_world_xyz_m,
        _yaw_from_xyzw(root_rotation_world_xyzw),
        command_direction_local_xz,
        heightfield_valid=(None if sensor_heightfield_valid is None else
                           torch.as_tensor(sensor_heightfield_valid, dtype=torch.bool)),
        config=config,
    )


def load_parc_paired_source(
    path: str | Path,
    command_direction_local_xz: Sequence[Sequence[float]] | np.ndarray,
    *,
    observation_config: LocalTerrainObservationConfig | None = None,
    sensor_heightfield_valid: np.ndarray | torch.Tensor | None = None,
) -> PARCPairedSource:
    """Load one PARC sample and make current-frame-only local map observations.

    Commands are required explicitly as [T, 2] root-local XZ directions. Frame t's
    observation uses only frame t's root, heading, command, and optional
    caller-supplied sensor validity mask; it never reads future commands or motion.
    PARC hf_mask_inds is preserved only as offline augmentation provenance and is
    never interpreted as sensor validity. Source contact values are carried
    verbatim and are not promoted to verified physical-contact truth.
    """
    source_path = Path(path)
    size = source_path.stat().st_size
    if size > _MAX_PICKLE_BYTES:
        raise ValueError("PARC sample exceeds the bounded inspection limit")
    with source_path.open("rb") as handle:
        container = _restricted_load(handle.read(_MAX_PICKLE_BYTES + 1))
    if not isinstance(container, dict):
        raise ValueError("PARC pickle root must be a dictionary")

    motion = _decode_section(container, "motion_data")
    terrain = _decode_section(container, "terrain_data")
    misc = container.get("misc_data")
    if isinstance(misc, bytes):
        misc = _restricted_load(misc)
    if misc is None:
        misc = {}
    if not isinstance(misc, dict):
        raise ValueError("PARC misc_data must be a dictionary when present")

    source_root = _finite_array(motion.get("root_pos"), (3,), "root_pos")
    source_root_q = _finite_array(motion.get("root_rot"), (4,), "root_rot")
    local_q = _finite_array(motion.get("joint_rot"), (14, 4), "joint_rot")
    contacts = _finite_array(motion.get("body_contacts"), (15,), "body_contacts")
    frames = len(source_root)
    if any(len(value) != frames for value in (source_root_q, local_q, contacts)):
        raise ValueError("PARC motion arrays must have a shared frame count")
    fps = float(motion.get("fps", 0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("PARC fps must be finite and positive")

    commands = np.asarray(command_direction_local_xz, dtype=np.float32)
    if commands.shape != (frames, 2) or not np.isfinite(commands).all():
        raise ValueError(f"command_direction_local_xz must be finite with shape [{frames}, 2]")

    hf_source = np.asarray(terrain.get("hf"), dtype=np.float32)
    if hf_source.ndim != 2 or min(hf_source.shape) <= 0 or not np.isfinite(hf_source).all():
        raise ValueError("terrain hf must be a finite [source_x, source_y] heightfield")
    nx, ny = hf_source.shape
    dx_m = float(terrain.get("dx", 0))
    min_source = np.asarray(terrain.get("min_point"), dtype=np.float64)
    if min_source.shape != (2,) or not np.isfinite(min_source).all():
        raise ValueError("terrain min_point must be finite source [x, y]")
    if not np.isfinite(dx_m) or dx_m <= 0:
        raise ValueError("terrain dx must be finite and positive")

    source_mask_indices = misc.get("hf_mask_inds")
    if source_mask_indices is not None:
        if not isinstance(source_mask_indices, (list, tuple)) or len(source_mask_indices) != frames:
            raise ValueError("Optional misc_data.hf_mask_inds must have one entry per frame")
        source_mask_indices = tuple(np.asarray(indices).copy() for indices in source_mask_indices)

    sensor_valid = None
    if sensor_heightfield_valid is not None:
        sensor_valid = (sensor_heightfield_valid.detach().cpu().numpy()
                        if isinstance(sensor_heightfield_valid, torch.Tensor)
                        else np.asarray(sensor_heightfield_valid))
        if sensor_valid.dtype != np.bool_ or sensor_valid.shape not in ((ny, nx), (frames, ny, nx)):
            raise ValueError(
                f"sensor_heightfield_valid must be boolean with shape [{ny}, {nx}] "
                f"or [{frames}, {ny}, {nx}]"
            )

    # Source (x,y,z) -> target (x,z,-y), a proper +1 determinant rotation.
    root_positions = np.stack((source_root[:, 0], source_root[:, 2], -source_root[:, 1]), axis=-1)
    source_root_q = _normalize_quaternions(source_root_q, "root_rot")
    root_q = _normalize_quaternions(
        _quat_multiply_xyzw(_ROOT_BASIS_QUAT_XYZW, source_root_q), "converted root_rot"
    )
    local_q = _normalize_quaternions(local_q, "joint_rot")

    # Source [x,y] -> target [z,x]; reverse source-y so target z increases by row.
    heightfield = np.ascontiguousarray(hf_source.T[::-1, :])
    min_point_target = np.array(
        [min_source[0], 0.0, -(min_source[1] + (ny - 1) * dx_m)], dtype=np.float32
    )
    observations = []
    for index in range(frames):
        frame_sensor_valid = sensor_valid
        if sensor_valid is not None and sensor_valid.ndim == 3:
            frame_sensor_valid = sensor_valid[index]
        observations.append(_make_observation(
            heightfield,
            min_point_target,
            dx_m,
            root_positions[index],
            root_q[index],
            commands[index],
            observation_config,
            frame_sensor_valid,
        ))

    return PARCPairedSource(
        source_path=str(source_path.resolve()),
        fps=fps,
        root_positions_world_xyz_m=np.ascontiguousarray(root_positions, dtype=np.float32),
        root_rotations_world_xyzw=root_q.astype(np.float32),
        local_joint_rotations_xyzw=local_q.astype(np.float32),
        source_body_contacts=contacts.copy(),
        contact_body_names=_CONTACT_BODY_NAMES,
        heightfield_y_m=heightfield,
        min_point_world_xyz_m=min_point_target,
        dx_m=dx_m,
        frame_observations=tuple(observations),
        source_hf_mask_inds=source_mask_indices,
    )
