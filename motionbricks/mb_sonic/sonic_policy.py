"""Frozen SONIC release inference, ported from the official G1 deployer."""

from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation
import yaml

from .reference import G1_JOINT_NAMES, ISAACLAB_FROM_MUJOCO, MUJOCO_FROM_ISAACLAB


POLICY_DT = 0.02
HISTORY_FRAMES = 10
BODY_JOINT_NAMES = G1_JOINT_NAMES
# Index a MuJoCo vector with MJ_TO_ISAAC to obtain the policy's IsaacLab order.
MJ_TO_ISAAC = ISAACLAB_FROM_MUJOCO
ISAAC_TO_MJ = MUJOCO_FROM_ISAACLAB
DEFAULT_ANGLES = np.array(
    [-0.312, 0, 0, 0.669, -0.363, 0] * 2 + [0, 0, 0]
    + [0.2, 0.2, 0, 0.6, 0, 0, 0] + [0.2, -0.2, 0, 0.6, 0, 0, 0], dtype=float
)
_ARMATURE = np.array(
    [0.025101925, 0.025101925, 0.010177520, 0.025101925, 0.003609725, 0.003609725] * 2
    + [0.010177520, 0.003609725, 0.003609725]
    + [0.003609725] * 5 + [0.00425] * 2 + [0.003609725] * 5 + [0.00425] * 2
)
_OMEGA = 10 * 2.0 * 3.1415926535
_GAIN_MULTIPLIER = np.ones(29)
_GAIN_MULTIPLIER[[4, 5, 10, 11, 13, 14]] = 2.0
KP = _ARMATURE * _OMEGA ** 2 * _GAIN_MULTIPLIER
KD = 4.0 * _ARMATURE * _OMEGA * _GAIN_MULTIPLIER
_SCALE_EFFORT = np.array(
    [139, 139, 88, 139, 25, 25] * 2 + [88, 25, 25]
    + [25] * 5 + [5, 5] + [25] * 5 + [5, 5], dtype=float
)
# The official scale uses single-motor stiffness, including for doubled PD gains.
ACTION_SCALE = 0.25 * _SCALE_EFFORT / (_ARMATURE * _OMEGA ** 2)
DECODER_FIELDS = (
    ("token_state", 64), ("his_base_angular_velocity_10frame_step1", 30),
    ("his_body_joint_positions_10frame_step1", 290),
    ("his_body_joint_velocities_10frame_step1", 290),
    ("his_last_actions_10frame_step1", 290), ("his_gravity_dir_10frame_step1", 30),
)
ENCODER_FIELDS = (
    ("encoder_mode_4", 4), ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_root_z_position_10frame_step5", 10), ("motion_root_z_position", 1),
    ("motion_anchor_orientation", 6), ("motion_anchor_orientation_10frame_step5", 60),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9), ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720), ("smpl_anchor_orientation_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
)
G1_REQUIRED_FIELDS = ("encoder_mode_4", "motion_joint_positions_10frame_step5",
                      "motion_joint_velocities_10frame_step5", "motion_anchor_orientation_10frame_step5")


def _array(value, shape, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    result.flags.writeable = False
    return result


def _quaternions(value, shape, name):
    result = _array(value, shape, name)
    if not np.allclose(np.linalg.norm(result, axis=-1), 1.0, atol=1e-5, rtol=0):
        raise ValueError(f"{name} must contain unit wxyz quaternions")
    return result


def _rotation(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[..., [1, 2, 3, 0]])


def _heading(wxyz):
    forward = _rotation(wxyz).apply([1.0, 0, 0])
    return Rotation.from_rotvec([0, 0, np.arctan2(forward[1], forward[0])])


@dataclass(frozen=True)
class Proprioception:
    time_s: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    base_quat_wxyz: np.ndarray
    base_angular_velocity: np.ndarray
    last_action_isaac: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.time_s):
            raise ValueError("Proprioception time must be finite")
        for name, shape in (("joint_positions", (29,)), ("joint_velocities", (29,)),
                            ("base_angular_velocity", (3,)), ("last_action_isaac", (29,))):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        object.__setattr__(self, "base_quat_wxyz", _quaternions(
            self.base_quat_wxyz, (4,), "base_quat_wxyz"))


class ObservationHistory:
    """Actual measurements only, oldest first; no startup frame padding."""

    def __init__(self):
        self.entries = deque(maxlen=HISTORY_FRAMES)

    def append(self, observation):
        if self.entries and not np.isclose(
            observation.time_s - self.entries[-1].time_s, POLICY_DT, atol=1e-8, rtol=0
        ):
            raise ValueError("Proprioception must advance at exactly 50Hz")
        self.entries.append(observation)

    def decoder_input(self, token):
        if len(self.entries) != HISTORY_FRAMES:
            raise RuntimeError("SONIC requires ten actual 50Hz proprioception samples")
        entries = tuple(self.entries)
        q = np.stack([entry.joint_positions - DEFAULT_ANGLES for entry in entries])[:, MJ_TO_ISAAC]
        dq = np.stack([entry.joint_velocities for entry in entries])[:, MJ_TO_ISAAC]
        gyro = np.stack([entry.base_angular_velocity for entry in entries])
        actions = np.stack([entry.last_action_isaac for entry in entries])
        gravity = np.stack([_rotation(entry.base_quat_wxyz).inv().apply([0, 0, -1]) for entry in entries])
        token = _array(token, (64,), "token")
        return np.concatenate([token, gyro.ravel(), q.ravel(), dq.ravel(), actions.ravel(),
                               gravity.ravel()]).astype(np.float32)[None]


class SonicObservationBuilder:
    def __init__(self, observation_config):
        with Path(observation_config).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        enabled = lambda fields: tuple(field["name"] for field in fields if field["enabled"])
        if enabled(config["observations"]) != tuple(name for name, _ in DECODER_FIELDS):
            raise ValueError("Expected the original SONIC release decoder observation configuration")
        encoder = config["encoder"]
        if (encoder["dimension"] != 64 or encoder["use_fp16"]
                or enabled(encoder["encoder_observations"]) != tuple(name for name, _ in ENCODER_FIELDS)):
            raise ValueError("Expected the original SONIC release encoder observation configuration")
        g1_mode = next(mode for mode in encoder["encoder_modes"] if mode["name"] == "g1")
        if g1_mode["mode_id"] != 0 or tuple(g1_mode["required_observations"]) != G1_REQUIRED_FIELDS:
            raise ValueError("Expected SONIC G1 mode 0 observation mask")
        self.encoder_slices = {}
        offset = 0
        for name, dimension in ENCODER_FIELDS:
            self.encoder_slices[name] = slice(offset, offset + dimension)
            offset += dimension
        self.encoder_dimension = offset
        self.heading_alignment = None

    def reset_heading(self, initial_base_quat, initial_reference_quat):
        self.heading_alignment = _heading(initial_base_quat) * _heading(initial_reference_quat).inv()

    def encoder_input(self, reference, base_quat_wxyz):
        if self.heading_alignment is None:
            raise RuntimeError("Initialize reference heading explicitly before inference")
        result = np.zeros((1, self.encoder_dimension), dtype=np.float32)
        result[0, self.encoder_slices["motion_joint_positions_10frame_step5"]] = (
            reference.joint_pos_isaaclab.ravel())
        result[0, self.encoder_slices["motion_joint_velocities_10frame_step5"]] = (
            reference.joint_vel_isaaclab.ravel())
        relative = (_rotation(base_quat_wxyz).inv() * self.heading_alignment
                    * _rotation(reference.qpos[:, 3:7]))
        result[0, self.encoder_slices["motion_anchor_orientation_10frame_step5"]] = (
            relative.as_matrix()[:, :, :2].reshape(-1))
        # Mode 0 is [0,0,0,0]; all other-mode fields are masked by the official config.
        return result


@dataclass(frozen=True)
class PolicyCommand:
    action_isaac: np.ndarray
    joint_target: np.ndarray
    token: np.ndarray
    encoder_input: np.ndarray
    decoder_input: np.ndarray
    timing_ns: dict


class FrozenSonicPolicy:
    """Original matched ONNX encoder/decoder pair, CPU execution only."""

    def __init__(self, encoder_path, decoder_path, observation_config, *,
                 encoder_sha256, decoder_sha256, threads=1):
        self.observations = SonicObservationBuilder(observation_config)
        self.asset_metadata = {}
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sessions = []
        for label, path, expected_hash, input_dim, output_dim in (
            ("encoder", encoder_path, encoder_sha256, self.observations.encoder_dimension, 64),
            ("decoder", decoder_path, decoder_sha256, sum(size for _, size in DECODER_FIELDS), 29),
        ):
            path = Path(path).resolve(strict=True)
            with path.open("rb") as stream:
                digest = hashlib.sha256(stream.read()).hexdigest()
            if digest != expected_hash:
                raise ValueError(f"{label} SHA256 mismatch: {digest}")
            session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            session.disable_fallback()
            if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
                raise ValueError(f"{label} must have one input and one output")
            source, target = session.get_inputs()[0], session.get_outputs()[0]
            for item, dimension in ((source, input_dim), (target, output_dim)):
                if item.type != "tensor(float)" or len(item.shape) != 2 or item.shape[-1] != dimension:
                    raise ValueError(f"Unexpected {label} tensor: {item.name} {item.type} {item.shape}")
            self.asset_metadata[label] = dict(path=str(path), sha256=digest,
                                               input_name=source.name, output_name=target.name,
                                               input_shape=source.shape, output_shape=target.shape)
            sessions.append(session)
        self.encoder, self.decoder = sessions

    def reset_heading(self, base_quat_wxyz, reference_quat_wxyz):
        self.observations.reset_heading(base_quat_wxyz, reference_quat_wxyz)

    def infer(self, reference, history):
        if len(history.entries) != HISTORY_FRAMES:
            raise RuntimeError("Collect ten actual history frames before SONIC inference")
        current = history.entries[-1]
        if not np.isclose(reference.timestamps_s[0], current.time_s, atol=1e-8, rtol=0):
            raise ValueError("Reference window must begin at the current simulation time")
        start = perf_counter_ns()
        encoder_obs = self.observations.encoder_input(reference, current.base_quat_wxyz)
        built = perf_counter_ns()
        token = self.encoder.run(None, {self.encoder.get_inputs()[0].name: encoder_obs})[0].reshape(64)
        encoded = perf_counter_ns()
        decoder_obs = history.decoder_input(token)
        gathered = perf_counter_ns()
        action = self.decoder.run(None, {self.decoder.get_inputs()[0].name: decoder_obs})[0].reshape(29)
        decoded = perf_counter_ns()
        if not np.isfinite(action).all():
            raise FloatingPointError("SONIC decoder returned non-finite action")
        target = DEFAULT_ANGLES + action[ISAAC_TO_MJ].astype(float) * ACTION_SCALE
        return PolicyCommand(action.copy(), target, token.copy(), encoder_obs, decoder_obs,
                             dict(encoder_observation=built-start, encoder=encoded-built,
                                  decoder_observation=gathered-encoded, decoder=decoded-gathered))
