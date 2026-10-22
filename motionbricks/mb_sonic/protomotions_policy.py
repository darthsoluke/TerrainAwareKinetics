"""Frozen ProtoMotions G1 deployment ONNX, with its published sensor contract."""

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation
import yaml

from .reference import G1_JOINT_NAMES, ReferenceCoverageError, file_sha256


PROTOMOTIONS_REVISION = "f69fdcaabff31846b4d2247207417b57dc788828"
PROTOMOTIONS_ONNX_SHA256 = "a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c"
PROTOMOTIONS_METADATA_SHA256 = "57d81b60dda9aeb30c57d3428a822f203a83bca399c5abe82ffea7a9aaf8b139"


@dataclass(frozen=True)
class ProtoMotionsCommand:
    joint_target: np.ndarray
    raw_action: np.ndarray
    processed_action_mujoco: np.ndarray
    inputs: dict
    timing_ns: dict
    execution_task: dict | None = None


def reference_window(reference, time_s, offsets):
    requested = time_s + np.asarray(offsets, dtype=np.float64)
    end = min(reference.trusted_until_s, reference.timestamps_s[-1])
    if requested[0] < reference.timestamps_s[0]-1e-9 or requested[-1] > end+1e-9:
        raise ReferenceCoverageError("ProtoMotions future samples exceed the real trusted reference")
    indexes = np.searchsorted(reference.timestamps_s, requested-1e-9)
    if not np.allclose(reference.timestamps_s[indexes], requested, atol=1e-9, rtol=0):
        raise ReferenceCoverageError("ProtoMotions requires existing 50Hz reference samples")
    fields = ("qpos", "qvel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    return replace(reference, timestamps_s=reference.timestamps_s[indexes],
                   trusted_until_s=float(reference.timestamps_s[indexes[-1]]),
                   **{name: getattr(reference, name)[indexes] for name in fields})


def _rotation(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[..., [1, 2, 3, 0]])


def _heading(wxyz):
    forward = _rotation(wxyz).apply([1., 0., 0.])
    return Rotation.from_rotvec([0., 0., np.arctan2(forward[1], forward[0])])


class FrozenProtoMotionsPolicy:
    """CPU-only inference; PD targets already contain the trained offset and scale."""

    control_mode = "implicit_pd"
    anchor_body_name = "torso_link"
    minimum_history_frames = 1

    def __init__(self, onnx_path, metadata_path, *, onnx_sha256, metadata_sha256, threads=1):
        onnx_path, metadata_path = Path(onnx_path).resolve(strict=True), Path(metadata_path).resolve(strict=True)
        if (onnx_sha256 != PROTOMOTIONS_ONNX_SHA256 or metadata_sha256 != PROTOMOTIONS_METADATA_SHA256
                or file_sha256(onnx_path) != onnx_sha256 or file_sha256(metadata_path) != metadata_sha256):
            raise ValueError("ProtoMotions model or deployment metadata hash mismatch")
        self.metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        robot, control, timing = (self.metadata[name] for name in ("robot", "control", "timing"))
        if (tuple(robot["joint_names"]) != G1_JOINT_NAMES or robot["num_dofs"] != 29
                or robot["anchor_body_name"] != self.anchor_body_name
                or robot["root_body_name"] != "pelvis" or timing["control_dt"] != .02
                or control["action_ema_alpha"] != 1. or control["pd_target_max_accel"] is not None):
            raise ValueError("Expected the unfiltered 29-joint G1 BONES-SEED deployment contract")
        self.physics_dt = float(timing["physics_dt"])
        self.kp, self.kd = (np.asarray(control[name], dtype=float) for name in ("stiffness", "damping"))
        if (self.kp.shape != (29,) or self.kd.shape != (29,) or not (self.kp > 0).all()
                or not (self.kd > 0).all() or self.physics_dt != .001):
            raise ValueError("Invalid published G1 PD parameters or physics timestep")
        self.reference_offsets_s = np.r_[0., self.metadata["motion"]["future_dt_seconds"]]
        if not np.array_equal(self.reference_offsets_s, [0., .02, .04, .08, .16]):
            raise ValueError("Expected the published four-horizon G1 tracker")
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.session.disable_fallback()
        expected = {row["name"]: row["shape"] for row in self.metadata["policy_inputs"]}
        actual = {row.name: [1, *row.shape[1:]] for row in self.session.get_inputs()
                  if row.type == "tensor(float)" and (row.shape[0] == 1 or isinstance(row.shape[0], str))}
        if actual != expected:
            raise ValueError(f"ProtoMotions ONNX input contract differs: {actual}")
        self.output_names = self.metadata["_runtime"]["onnx_out_names"]
        if [row.name for row in self.session.get_outputs()] != self.output_names:
            raise ValueError("ProtoMotions ONNX output order differs")
        self.heading_alignment = None
        self.previous_target = None
        self.asset_metadata = dict(tracker="protomotions_g1_bones_deploy", source_revision=PROTOMOTIONS_REVISION,
            onnx=dict(path=str(onnx_path), sha256=onnx_sha256),
            deployment_metadata=dict(path=str(metadata_path), sha256=metadata_sha256),
            physics_dt_s=self.physics_dt, control_dt_s=.02, reference_offsets_s=self.reference_offsets_s.tolist(),
            anchor_body_name=self.anchor_body_name, root_angular_velocity_frame="pelvis_local",
            control_mode=self.control_mode, action_ema_alpha=1., pd_target_max_accel=None)

    def reference_window(self, reference, time_s):
        return reference_window(reference, time_s, self.reference_offsets_s)

    def reset_heading(self, actual_anchor_wxyz, reference_anchor_wxyz):
        self.heading_alignment = _heading(actual_anchor_wxyz) * _heading(reference_anchor_wxyz).inv()
        self.previous_target = None

    def infer(self, reference, history, *, anchor_quat_wxyz):
        started = perf_counter_ns()
        current = history.entries[-1]
        if self.heading_alignment is None or not np.isclose(reference.timestamps_s[0], current.time_s, atol=1e-8, rtol=0):
            raise ValueError("Initialize the heading and use the current physical timestamp")
        anchor = reference.body_names.index(self.anchor_body_name)
        future_rot = (self.heading_alignment * _rotation(reference.body_quat_w[1:, anchor])).as_quat()
        inputs = {
            "current_anchor_rot": _rotation(anchor_quat_wxyz).as_quat()[None],
            "current_dof_pos": current.joint_positions[None],
            "current_dof_vel": current.joint_velocities[None],
            "current_root_local_ang_vel": current.base_angular_velocity[None],
            "historical_processed_actions": (np.zeros(29) if self.previous_target is None else self.previous_target)[None, None],
            "mimic_future_anchor_rot": future_rot[None],
            "mimic_future_dof_pos": reference.joint_pos[None, 1:],
            "mimic_future_dof_vel": reference.joint_vel[None, 1:],
        }
        inputs = {name: np.asarray(value, dtype=np.float32) for name, value in inputs.items()}
        gathered = perf_counter_ns()
        action, targets, kp, kd = self.session.run(self.output_names, inputs)
        finished = perf_counter_ns()
        if not all(np.isfinite(value).all() for value in (action, targets, kp, kd)):
            raise FloatingPointError("ProtoMotions returned nonfinite targets")
        if not np.allclose(kp[0], self.kp, atol=1e-5, rtol=1e-6) or not np.allclose(kd[0], self.kd, atol=1e-5, rtol=1e-6):
            raise ValueError("ONNX gains differ from the pinned deployment metadata")
        target = targets[0].astype(float)
        self.previous_target = target.copy()
        return ProtoMotionsCommand(target, action[0].copy(), target.copy(), inputs,
                                  dict(observation=gathered-started, tracker=finished-gathered))
