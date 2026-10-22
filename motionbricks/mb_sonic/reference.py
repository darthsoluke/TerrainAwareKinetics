"""Timestamped reference data; independent of simulation state and model loading."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np


SONIC_FPS = 50
SONIC_FUTURE_OFFSETS = np.arange(10, dtype=np.int64) * 5
# These are gather indexes: q_mujoco = q_isaaclab[MUJOCO_FROM_ISAACLAB].
MUJOCO_FROM_ISAACLAB = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
])
ISAACLAB_FROM_MUJOCO = np.argsort(MUJOCO_FROM_ISAACLAB)
G1_JOINT_NAMES = tuple(
    f"{side}_{name}_joint" for side in ("left", "right")
    for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
) + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") + tuple(
    f"{side}_{name}_joint" for side in ("left", "right")
    for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")
)
OFFICIAL_BODY_INDEXES = (0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29)
OFFICIAL_BODY_NAMES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


class ReferenceCoverageError(ValueError):
    """Requested timestamps are not covered by real reference samples."""


class EmbodimentConversionError(ValueError):
    """A measured G1 state does not provide a complete native SOMA observation."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(value, shape):
    array = np.array(value, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected finite array {shape}, got {array.shape}")
    array.setflags(write=False)
    return array


def _times(value):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all() or (np.diff(result) <= 0).any():
        raise ValueError("Timestamps must be finite, nonempty and strictly increasing")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SomaReference:
    timestamps_s: np.ndarray
    raw_features: np.ndarray
    bone_names: tuple[str, ...]
    root_positions_m: np.ndarray
    local_rot_mats: np.ndarray
    global_rot_mats: np.ndarray
    world_positions_m: np.ndarray
    scene_id: str
    command_id: str
    trusted_until_s: float
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        time = _times(self.timestamps_s)
        object.__setattr__(self, "timestamps_s", time)
        count = len(time)
        if len(self.bone_names) != 30 or len(set(self.bone_names)) != 30 or self.bone_names[0] != "Hips":
            raise ValueError("The source must name the actual SOMA30 bones")
        for name, shape in (
            ("raw_features", (count, 366)), ("root_positions_m", (count, 3)),
            ("local_rot_mats", (count, 30, 3, 3)), ("global_rot_mats", (count, 30, 3, 3)),
            ("world_positions_m", (count, 30, 3)),
        ):
            object.__setattr__(self, name, _snapshot(getattr(self, name), shape))
        if not time[0] <= self.trusted_until_s <= time[-1]:
            raise ValueError("Trusted coverage must lie inside the raw generation interval")


@dataclass(frozen=True)
class NativeSomaHistory:
    timestamps_s: np.ndarray
    features: np.ndarray
    scene_id: str
    provenance: Mapping

    def __post_init__(self):
        times = _times(self.timestamps_s)
        if times.shape != (4,) or not np.allclose(np.diff(times), 1 / 30, atol=1e-9, rtol=0):
            raise ValueError("Native MB history requires four real-time samples at 30 Hz")
        if self.provenance.get("kind") not in (
            "source_seed", "calibrated_executed_g1", "open_loop_generated_context", "legacy_reference_history", "actual_g1_error_state",
        ):
            raise ValueError("Native history must distinguish seed, estimated execution and generated proposal context")
        object.__setattr__(self, "timestamps_s", times)
        object.__setattr__(self, "features", _snapshot(self.features, (4, 366)))


@dataclass(frozen=True)
class RawMBBlock:
    """Raw native output and actual conditioning input; FK is an offline operation."""

    timestamps_s: np.ndarray
    raw_features: np.ndarray
    history_features: np.ndarray
    scene_id: str
    command_id: str
    trusted_until_s: float
    provenance: Mapping

    def __post_init__(self):
        times = _times(self.timestamps_s)
        if times.shape != (32,) or not times[0] <= self.trusted_until_s <= times[-1]:
            raise ValueError("Raw MB block requires its actual 32-frame finite interval")
        object.__setattr__(self, "timestamps_s", times)
        object.__setattr__(self, "raw_features", _snapshot(self.raw_features, (32, 366)))
        object.__setattr__(self, "history_features", _snapshot(self.history_features, (4, 366)))


def save_raw_mb_block(path: Path, block: RawMBBlock):
    metadata = dict(schema="motionbricks.mb_sonic.raw_global366.v1", scene_id=block.scene_id,
                    command_id=block.command_id, trusted_until_s=block.trusted_until_s,
                    provenance=dict(block.provenance))
    with Path(path).open("xb") as stream:
        np.savez(stream, timestamps_s=block.timestamps_s, raw_features=block.raw_features,
                 history_features=block.history_features, metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))


def load_raw_mb_block(path: Path) -> RawMBBlock:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata.pop("schema") != "motionbricks.mb_sonic.raw_global366.v1":
            raise ValueError("Unknown raw native MB block schema")
        return RawMBBlock(archive["timestamps_s"], archive["raw_features"], archive["history_features"], **metadata)


def save_soma_reference(path: Path, reference: SomaReference):
    """Lossless generated artifact; never overwrite an earlier experiment."""
    arrays = {name: getattr(reference, name) for name in (
        "timestamps_s", "raw_features", "root_positions_m", "local_rot_mats",
        "global_rot_mats", "world_positions_m",
    )}
    metadata = dict(schema="motionbricks.mb_sonic.soma_reference.v1", scene_id=reference.scene_id,
                    command_id=reference.command_id, trusted_until_s=reference.trusted_until_s,
                    provenance=dict(reference.provenance))
    with Path(path).open("xb") as stream:
        np.savez(stream, **arrays, bone_names=np.asarray(reference.bone_names),
                 metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))


def load_soma_reference(path: Path) -> SomaReference:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata.pop("schema") != "motionbricks.mb_sonic.soma_reference.v1":
            raise ValueError("Unknown SOMA reference artifact schema")
        return SomaReference(**{name: archive[name] for name in (
            "timestamps_s", "raw_features", "root_positions_m", "local_rot_mats",
            "global_rot_mats", "world_positions_m",
        )}, bone_names=tuple(archive["bone_names"].tolist()), **metadata)


@dataclass(frozen=True)
class G1Reference:
    """SI, Z-up world; MuJoCo joint order; all quaternions are wxyz.

    qvel uses MuJoCo's tangent convention (world root linear, root-local angular).
    Body velocities are world-frame quantities. A reference is never execution.
    """

    timestamps_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    body_names: tuple[str, ...]
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    scene_id: str
    command_id: str
    trusted_until_s: float
    provenance: Mapping = field(default_factory=dict)
    adaptation: Mapping = field(default_factory=dict)
    joint_names: tuple[str, ...] = G1_JOINT_NAMES

    def __post_init__(self):
        time = _times(self.timestamps_s)
        object.__setattr__(self, "timestamps_s", time)
        count, bodies = len(time), len(self.body_names)
        if self.joint_names != G1_JOINT_NAMES or len(set(self.body_names)) != bodies:
            raise ValueError("G1 reference names/order differ from the 29-DOF contract")
        for name, shape in (
            ("qpos", (count, 36)), ("qvel", (count, 35)),
            ("body_pos_w", (count, bodies, 3)), ("body_quat_w", (count, bodies, 4)),
            ("body_lin_vel_w", (count, bodies, 3)), ("body_ang_vel_w", (count, bodies, 3)),
        ):
            object.__setattr__(self, name, _snapshot(getattr(self, name), shape))
        if not np.allclose(np.linalg.norm(self.qpos[:, 3:7], axis=-1), 1, atol=2e-5):
            raise ValueError("Reference root quaternions must be unit wxyz")
        if not np.allclose(np.linalg.norm(self.body_quat_w, axis=-1), 1, atol=2e-5):
            raise ValueError("Reference body quaternions must be unit wxyz")
        if not time[0] <= self.trusted_until_s <= time[-1]:
            raise ValueError("Trusted coverage must lie inside the sampled reference interval")

    @property
    def joint_pos(self):
        return self.qpos[:, 7:]

    @property
    def joint_vel(self):
        return self.qvel[:, 6:]

    @property
    def joint_pos_isaaclab(self):
        return self.joint_pos[:, ISAACLAB_FROM_MUJOCO]

    @property
    def joint_vel_isaaclab(self):
        return self.joint_vel[:, ISAACLAB_FROM_MUJOCO]

    @property
    def latest_window_start_s(self):
        return min(self.timestamps_s[-1], self.trusted_until_s) - 0.9

    def sonic_window(self, current_time_s: float) -> G1Reference:
        """The released G1 encoder's 10 samples at offsets 0..0.9 seconds."""
        if not np.isfinite(current_time_s):
            raise ValueError("Current reference time must be finite")
        requested = current_time_s + SONIC_FUTURE_OFFSETS / SONIC_FPS
        end = min(self.timestamps_s[-1], self.trusted_until_s)
        if requested[0] < self.timestamps_s[0] - 1e-9 or requested[-1] > end + 1e-9:
            raise ReferenceCoverageError(f"SONIC requires [{requested[0]:.6f}, {requested[-1]:.6f}]s; "
                                         f"trusted coverage is [{self.timestamps_s[0]:.6f}, {end:.6f}]s")
        indexes = np.searchsorted(self.timestamps_s, requested - 1e-9)
        if not np.allclose(self.timestamps_s[indexes], requested, atol=1e-9, rtol=0):
            raise ReferenceCoverageError("SONIC timestamps must lie on the existing 50 Hz sample grid")
        fields = ("qpos", "qvel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        return replace(self, timestamps_s=self.timestamps_s[indexes],
                       trusted_until_s=float(self.timestamps_s[indexes[-1]]),
                       **{name: getattr(self, name)[indexes] for name in fields})


G1_ARRAY_FIELDS = ('timestamps_s', 'qpos', 'qvel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w')


def save_g1_reference(path, reference):
    metadata = dict(schema='motionbricks.g1_fixed_reference.v1', body_names=reference.body_names,
        scene_id=reference.scene_id, command_id=reference.command_id, trusted_until_s=reference.trusted_until_s,
        provenance=dict(reference.provenance), adaptation=dict(reference.adaptation), joint_names=reference.joint_names)
    with Path(path).open('xb') as stream:
        np.savez(stream, **{name: getattr(reference, name) for name in G1_ARRAY_FIELDS},
            metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))


def load_g1_reference(path, *, expected_sha256, scene_id):
    if file_sha256(path) != expected_sha256:
        raise ValueError('Fixed G1 reference differs from its explicit SHA256')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        if metadata.pop('schema') != 'motionbricks.g1_fixed_reference.v1' or metadata['scene_id'] != scene_id:
            raise ValueError('Fixed G1 reference schema or canonical scene mismatch')
        metadata['body_names'], metadata['joint_names'] = tuple(metadata['body_names']), tuple(metadata['joint_names'])
        return G1Reference(**{name: archive[name] for name in G1_ARRAY_FIELDS}, **metadata)


def load_official_reference(directory: Path, *, scene_id: str, command_id="official_reference") -> G1Reference:
    """Load the released 50 Hz CSV schema without recomputing/smoothing signals."""
    from scipy.spatial.transform import Rotation

    directory = Path(directory)
    metadata_path = directory / "metadata.txt"
    metadata = metadata_path.read_text(encoding="utf-8")
    match = re.search(r"Body part indexes:\s*\[([^]]+)\]", metadata)
    indexes = tuple(np.fromstring(match.group(1), sep=" ", dtype=int)) if match else ()
    if indexes != OFFICIAL_BODY_INDEXES:
        raise ValueError("Official body metadata does not match the released 14-body layout")
    arrays, hashes = {}, {"metadata.txt": file_sha256(metadata_path)}
    for name, width in (("joint_pos", 29), ("joint_vel", 29), ("body_pos", 42),
                        ("body_quat", 56), ("body_lin_vel", 42), ("body_ang_vel", 42)):
        path = directory / f"{name}.csv"
        value = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if value.shape[1] != width:
            raise ValueError(f"{name}.csv has {value.shape[1]} columns, expected {width}")
        arrays[name], hashes[path.name] = value, file_sha256(path)
    count = len(arrays["joint_pos"])
    frames = re.search(r"Total timesteps:\s*(\d+)", metadata)
    if frames is None or int(frames.group(1)) != count or any(len(value) != count for value in arrays.values()):
        raise ValueError("CSV frame counts disagree with the official metadata")
    positions = arrays["body_pos"].reshape(count, 14, 3)
    quats = arrays["body_quat"].reshape(count, 14, 4)
    lin = arrays["body_lin_vel"].reshape(count, 14, 3)
    ang = arrays["body_ang_vel"].reshape(count, 14, 3)
    root_rot = Rotation.from_quat(quats[:, 0, [1, 2, 3, 0]])
    qpos = np.concatenate((positions[:, 0], quats[:, 0], arrays["joint_pos"][:, MUJOCO_FROM_ISAACLAB]), axis=-1)
    qvel = np.concatenate((lin[:, 0], root_rot.inv().apply(ang[:, 0]),
                           arrays["joint_vel"][:, MUJOCO_FROM_ISAACLAB]), axis=-1)
    timestamps = np.arange(count, dtype=np.float64) / SONIC_FPS
    return G1Reference(timestamps, qpos, qvel, OFFICIAL_BODY_NAMES, positions, quats, lin, ang,
                       scene_id, command_id, float(timestamps[-1]), provenance=dict(
                           kind="official_csv_reference", source=str(directory.resolve()),
                           fps=50, source_joint_order="isaaclab", source_quaternion_order="wxyz",
                           source_files_sha256=hashes, velocity_source="released_csv"))


@dataclass
class ExecutedHistory:
    """Snapshots from the physical MjData, kept outside all reference buffers."""

    scene_id: str
    samples: list = field(default_factory=list)
    body_names: tuple[str, ...] = ()

    def append(self, model, data):
        import mujoco

        if self.samples and data.time <= self.samples[-1]["timestamp_s"]:
            raise ValueError("Executed simulation time must advance; automatic reset is unsupported")
        self.body_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                                for i in range(model.nbody))
        contacts = []
        for index in range(data.ncon):
            contact = data.contact[index]
            wrench = np.empty(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            contacts.append(dict(body_ids=tuple(int(model.geom_bodyid[i]) for i in contact.geom),
                                 position_world=contact.pos.copy(), wrench_contact=wrench))
        self.samples.append(dict(timestamp_s=float(data.time), qpos=data.qpos.copy(),
                                 qvel=data.qvel.copy(), body_pos_w=data.xpos.copy(),
                                 body_quat_w=data.xquat.copy(), contacts=tuple(contacts)))

    def native_soma_history(self, converter=None, *, end_time_s=None):
        if converter is None:
            raise EmbodimentConversionError(
                "G1 executed history needs an explicit calibrated embodiment converter, "
                "including four SOMA contact channels from measured contacts.")
        return converter.estimate_history(self, end_time_s=end_time_s)
