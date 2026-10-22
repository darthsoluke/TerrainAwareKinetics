"""Static, paired-data SOMA30/G1 kinematic calibration, separate from physics.

No motion model, optimizer, MjData state writes, contact-distance labels or
unobserved angles disguised as measurements are used by this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .reference import (G1_JOINT_NAMES, OFFICIAL_BODY_NAMES, G1Reference,
                        NativeSomaHistory, ReferenceCoverageError, SomaReference,
                        SONIC_FPS, SONIC_FUTURE_OFFSETS)
from .scene import SOMA_TO_MUJOCO, sha256


SCHEMA = "motionbricks.mb_sonic.retarget.v1"
SEGMENTS = ("Hips", "Chest", "LeftLeg", "LeftShin", "LeftFoot", "RightLeg", "RightShin", "RightFoot",
            "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand")
BODY_MAP = ("pelvis", "torso_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link",
            "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link", "left_shoulder_yaw_link",
            "left_elbow_link", "left_wrist_yaw_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link")
OBSERVED_RANK = dict(zip(SEGMENTS, (3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 3, 3, 1, 3)))
SONIC_REQUIRED_FUTURE_S = float(SONIC_FUTURE_OFFSETS[-1] / SONIC_FPS)
REPLAN_GUARD_S = .1
CONSUMER_HORIZON_S = SONIC_REQUIRED_FUTURE_S + REPLAN_GUARD_S
PACKET_CONTRACT = "motionbricks.mb_sonic.fixed_1s_consumer.v1"
EXPLICIT_PACKET_CONTRACT = "motionbricks.mb_sonic.explicit_consumer_horizon.v2"
SOURCE_GAUGE = "motionbricks.mb_sonic.anatomical_fk_markers.v2"


def _matrix(quat_wxyz):
    q = np.asarray(quat_wxyz)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(*q.shape[:-1], 3, 3)


def _quat(matrix):
    r = np.asarray(matrix)
    q = Rotation.from_matrix(r.reshape(-1, 3, 3)).as_quat().reshape(*r.shape[:-2], 4)[..., [3, 0, 1, 2]]
    if q.ndim >= 2 and len(q) > 1:
        signs = np.where((q[1:] * q[:-1]).sum(-1) < 0, -1., 1.)
        q[1:] *= np.cumprod(signs, axis=0)[..., None]
    return q


def _frame(primary, secondary, name):
    x = primary / np.linalg.norm(primary, axis=-1, keepdims=True)
    y = secondary - (secondary * x).sum(-1, keepdims=True) * x
    length = np.linalg.norm(y, axis=-1, keepdims=True)
    if (length < 1e-8).any() or not np.isfinite(x).all():
        raise ValueError(f"Anatomical frame {name} is unobservable from collinear/missing markers")
    y = y / length
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def anatomical_frames(positions, names, *, limb_frames=None):
    """Named marker frames with optional transported limb and foot frames."""
    positions = np.asarray(positions, float)
    names = tuple(names)
    points = {name: positions[:, names.index(name)] for name in names}
    frames = [_frame(points["LeftLeg"] - points["RightLeg"], points["Spine1"] - points["Hips"], "Hips"),
              _frame(points["LeftArm"] - points["RightArm"], points["Neck1"] - points["Chest"], "Chest")]
    for side in ("Left", "Right"):
        hip, knee, ankle, toe = (points[side + name] for name in ("Leg", "Shin", "Foot", "ToeBase"))
        frames.extend(((limb_frames[side + "Leg"] if limb_frames is not None else
                        _frame(knee - hip, ankle - knee, side + "Leg")),
                       (limb_frames[side + "Shin"] if limb_frames is not None else
                        _frame(ankle - knee, hip - knee, side + "Shin")),
                       (limb_frames[side + "Foot"] if limb_frames is not None else
                        _frame(toe - ankle, knee - ankle, side + "Foot"))))
    for side in ("Left", "Right"):
        shoulder, elbow, hand = (points[side + name] for name in ("Arm", "ForeArm", "Hand"))
        thumb_name = side + ("HandThumbEnd" if side + "HandThumbEnd" in points else "HandThumb1")
        middle_name = side + ("HandMiddleEnd" if side + "HandMiddleEnd" in points else "HandMiddle1")
        frames.extend(((limb_frames[side + "Arm"] if limb_frames is not None else
                        _frame(elbow - shoulder, hand - elbow, side + "Arm")),
                       (limb_frames[side + "ForeArm"] if limb_frames is not None else
                        _frame(hand - elbow, shoulder - elbow, side + "ForeArm")),
                       _frame(points[middle_name] - hand, points[thumb_name] - hand, side + "Hand")))
    return np.stack(frames, axis=1)


def native_limb_material_frames(global_rotations, neutral, names):
    """Transport native limb and foot axes, including at straight knees/elbows."""
    names = tuple(names)
    neutral, global_rotations = np.asarray(neutral, float), np.asarray(global_rotations)
    result = {}
    for side, elbow_sign in (("Left", -1.), ("Right", 1.)):
        for segment, child, hinge in (("Leg", "Shin", [1., 0., 0.]),
                ("Shin", "Foot", [1., 0., 0.]),
                ("Arm", "ForeArm", [0., elbow_sign, 0.]),
                ("ForeArm", "Hand", [0., elbow_sign, 0.])):
            index, child_index = names.index(side+segment), names.index(side+child)
            direction = neutral[child_index]-neutral[index]
            # The native knee flexes about +X; elbows about -Y/+Y. This is the
            # non-collinear limit of the calibrated bending-plane frame.
            material = _frame(direction[None], np.cross(hinge, direction)[None], side+segment)[0]
            result[side+segment] = global_rotations[:, index]@material
        foot, toe, knee = (names.index(side+name) for name in ("Foot", "ToeBase", "Shin"))
        material = _frame((neutral[toe]-neutral[foot])[None],
                          (neutral[knee]-neutral[foot])[None], side+"Foot")[0]
        result[side+"Foot"] = global_rotations[:, foot]@material
    return result


def native_rest_frames(neutral, names):
    """Official neutral bone directions and explicit identity-rig twist axes."""
    neutral = np.asarray(neutral, float)[None]
    points = {name: neutral[:, tuple(names).index(name)] for name in names}
    frames = [_frame(points["LeftLeg"] - points["RightLeg"], points["Spine1"] - points["Hips"], "Hips"),
              _frame(points["LeftArm"] - points["RightArm"], points["Neck1"] - points["Chest"], "Chest")]
    for side in ("Left", "Right"):
        hip, knee, ankle, toe = (points[side + name] for name in ("Leg", "Shin", "Foot", "ToeBase"))
        frames.extend((_frame(knee - hip, np.array([[1., 0., 0.]]), side + "Leg_rest"),
                       _frame(ankle - knee, np.array([[1., 0., 0.]]), side + "Shin_rest"),
                       _frame(toe - ankle, knee - ankle, side + "Foot_rest")))
    for side in ("Left", "Right"):
        shoulder, elbow, hand = (points[side + name] for name in ("Arm", "ForeArm", "Hand"))
        frames.extend((_frame(elbow - shoulder, np.array([[0., 1., 0.]]), side + "Arm_rest"),
                       _frame(hand - elbow, np.array([[0., 1., 0.]]), side + "ForeArm_rest"),
                       _frame(points[side + "HandMiddleEnd"] - hand, points[side + "HandThumbEnd"] - hand, side + "Hand_rest")))
    return np.stack(frames, axis=1)[0]


class G1Kinematics:
    """Batched FK from the actual XML constants. Never allocates/changes MjData."""

    def __init__(self, robot_model_path):
        self.path = Path(robot_model_path).resolve(strict=True)
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        m = self.model
        self.body_names = tuple(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody))
        self.pelvis = self.body_names.index("pelvis")
        self.joint_ids = {name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name) for name in G1_JOINT_NAMES}
        if min(self.joint_ids.values()) < 0:
            raise ValueError("Robot must contain all 29 original named G1 joints")
        self.column = {joint: 7 + index for index, joint in enumerate(self.joint_ids.values())}
        self.rest = _matrix(m.body_quat)
        self.limits = np.asarray([m.jnt_range[self.joint_ids[name]] for name in G1_JOINT_NAMES])
        urdf_path = self.path.with_suffix(".urdf").resolve(strict=True)
        urdf_joints = {joint.attrib["name"]: joint for joint in ET.parse(urdf_path).getroot().findall("joint")}
        self.velocity_limits = np.array([float(urdf_joints[name].find("limit").attrib["velocity"])
                                         for name in G1_JOINT_NAMES])
        if not np.isfinite(self.velocity_limits).all() or (self.velocity_limits <= 0).any():
            raise ValueError("The original robot URDF must declare positive finite joint velocity limits")
        self.velocity_limit_source = dict(path=str(urdf_path), sha256=sha256(urdf_path), units="rad/s",
                                         kind="original_URDF_limit_velocity; declared_not_measured")
        self.body_ids = np.array([self.body_names.index(name) for name in BODY_MAP])
        self.output_ids = np.array([self.body_names.index(name) for name in OFFICIAL_BODY_NAMES])

    def fk(self, qpos):
        qpos = np.asarray(qpos, float)
        n, m = len(qpos), self.model
        pos = np.empty((n, m.nbody, 3))
        rot = np.empty((n, m.nbody, 3, 3))
        pos[:, 0], rot[:, 0] = 0., np.eye(3)
        hinge_bodies = [body for body in range(1, m.nbody) if body != self.pelvis and m.body_jntnum[body]]
        rotvecs = []
        for body in hinge_bodies:
            if m.body_jntnum[body] != 1:
                raise ValueError("G1 FK requires one hinge per articulated non-root body")
            joint = int(m.body_jntadr[body])
            angle = qpos[:, self.column[joint]] if joint in self.column else np.full(n, m.qpos0[m.jnt_qposadr[joint]])
            rotvecs.append(angle[:, None] * m.jnt_axis[joint])
        hinges = Rotation.from_rotvec(np.concatenate(rotvecs)).as_matrix().reshape(len(hinge_bodies), n, 3, 3)
        hinge_by_body = dict(zip(hinge_bodies, hinges))
        for body in range(1, m.nbody):
            if body == self.pelvis:
                pos[:, body], rot[:, body] = qpos[:, :3], _matrix(qpos[:, 3:7])
                continue
            parent = m.body_parentid[body]
            local_r = np.broadcast_to(self.rest[body], (n, 3, 3))
            local_p = np.broadcast_to(m.body_pos[body], (n, 3))
            if m.body_jntnum[body]:
                joint = int(m.body_jntadr[body])
                hinge = hinge_by_body[body]
                local_r = self.rest[body] @ hinge
                local_p = local_p + np.einsum("ij,nj->ni", self.rest[body], m.jnt_pos[joint] - hinge @ m.jnt_pos[joint])
            pos[:, body] = pos[:, parent] + np.einsum("nij,nj->ni", rot[:, parent], local_p)
            rot[:, body] = rot[:, parent] @ local_r
        return pos, rot


def _load_pair(soma_path, robot_path):
    import joblib

    human, robot = joblib.load(soma_path), joblib.load(robot_path)
    if len(human) != 1 or set(human) != set(robot):
        raise ValueError("Calibration requires one explicitly matched SOMA/G1 clip")
    name = next(iter(human))
    h, r = human[name], robot[name]
    if h["fps"] != r["fps"] or len(h["soma_joints"]) != len(r["dof"]):
        raise ValueError("Paired clip frame counts and original sampling rates must match")
    q = np.concatenate((r["root_trans_offset"], r["root_rot"][:, [3, 0, 1, 2]], r["dof"]), axis=-1)
    if q.shape[1] != 36 or not np.isfinite(q).all():
        raise ValueError("Official robot sample must have 29 DOFs and finite root XYZ/xyzw")
    return name, h, q


@dataclass(frozen=True)
class SomaEstimate:
    root_positions_m: np.ndarray
    local_rot_mats: np.ndarray
    global_rot_mats: np.ndarray
    world_positions_m: np.ndarray


class CalibratedSomaG1Adapter:
    def __init__(self, robot_model_path, paired_soma_path, paired_robot_path, *, representation, force_threshold_n=1.):
        self._initialize(robot_model_path, representation, force_threshold_n)
        name, human, qpos = _load_pair(paired_soma_path, paired_robot_path)
        frames = anatomical_frames(np.asarray(human["soma_joints"], float), human["joint_names"])
        human_r = frames @ self.rest_frames.swapaxes(-1, -2)
        _, robot_r = self.kinematics.fk(qpos)
        relative = human_r.swapaxes(-1, -2) @ robot_r[:, self.kinematics.body_ids]
        self.offset = np.stack([Rotation.from_matrix(relative[:, i]).mean().as_matrix() for i in range(len(SEGMENTS))])
        self.offset.setflags(write=False)
        delta = relative @ self.offset.swapaxes(-1, -2)
        residual = Rotation.from_matrix(delta.reshape(-1, 3, 3)).magnitude().reshape(len(qpos), -1)
        self.root_z_offset_m = float(np.median(qpos[:, 2] - np.asarray(human["soma_transl"])[:, 1]))
        self.calibration = dict(schema=SCHEMA, clip=name, source_soma_sha256=sha256(paired_soma_path),
                                source_robot_sha256=sha256(paired_robot_path), robot_xml_sha256=sha256(robot_model_path),
                                frame_count=len(qpos), fps=float(human["fps"]), optimizer_updates=0,
                                paired_coordinate_contract="soma_joints_preconverted_Z_up_root_relative; soma_transl_Y_up_world",
                                rotation_method="SO3_Markley_static_frame_mean; inferred_anatomical_twist",
                                segment_rotation_rmse_rad=dict(zip(SEGMENTS, np.sqrt((residual ** 2).mean(0)).tolist())),
                                root_z_offset_m=self.root_z_offset_m, root_xy_mapping="identity_metric_world",
                                heel_toe_split_x_m=self.heel_toe_split_x_m, force_threshold_n=self.force_threshold_n,
                                bone_names=list(self.names), neutral_positions_m=self.neutral.tolist(), offsets=self.offset.tolist(),
                                quality_gate="NOT_RUN", quality_gate_reason="paired_and_held_out_mirror_residual_review_required")
        payload = json.dumps(self.calibration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.calibration["identity"] = hashlib.sha256(payload).hexdigest()

    def _initialize(self, robot_model_path, representation, force_threshold_n):
        if representation.fps != 30 or representation.skeleton.neutral_joints.device.type != "cpu":
            raise ValueError("Retargeting requires the unnormalized native 30Hz CPU representation, without model loading")
        self.rep = representation
        self.skeleton = representation.skeleton
        self.names = tuple(self.skeleton.bone_order_names)
        self.parents = np.asarray(self.skeleton.joint_parents.cpu(), int)
        self.neutral = np.asarray(self.skeleton.neutral_joints.cpu(), float)
        self.segment_ids = np.array([self.names.index(name) for name in SEGMENTS])
        self.kinematics = G1Kinematics(robot_model_path)
        self.rest_frames = native_rest_frames(self.neutral, self.names)
        self.force_threshold_n = float(force_threshold_n)
        if not 0 < self.force_threshold_n < np.inf:
            raise ValueError("A positive measured-contact normal-force threshold is required")
        self.heel_toe_split_x_m = {}
        m = self.kinematics.model
        for side in ("left", "right"):
            bid = self.kinematics.body_names.index(side + "_ankle_roll_link")
            ids = (m.geom_bodyid == bid) & ((m.geom_contype != 0) | (m.geom_conaffinity != 0))
            x = m.geom_pos[ids, 0]
            if len(x) < 2 or np.ptp(x) <= 0:
                raise ValueError("Foot collision anchors do not identify a heel/toe split")
            self.heel_toe_split_x_m[side] = float((x.min() + x.max()) / 2)
        self.groups = self._groups()
        self.limb_rotation_source = "markers"

    def save(self, path):
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(self.calibration, stream, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path, *, robot_model_path, representation, limb_rotation_source="markers"):
        calibration = json.loads(Path(path).read_text(encoding="utf-8"))
        identity = calibration.pop("identity")
        payload = json.dumps(calibration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if calibration["schema"] != SCHEMA or hashlib.sha256(payload).hexdigest() != identity:
            raise ValueError("Calibration schema/identity mismatch")
        if calibration["robot_xml_sha256"] != sha256(robot_model_path):
            raise ValueError("Calibration belongs to a different original robot XML")
        result = cls.__new__(cls)
        result._initialize(robot_model_path, representation, calibration["force_threshold_n"])
        if calibration["bone_names"] != list(result.names) or calibration["neutral_positions_m"] != result.neutral.tolist():
            raise ValueError("Calibration belongs to a different native skeleton")
        result.offset = np.asarray(calibration["offsets"], float)
        result.offset.setflags(write=False)
        result.root_z_offset_m = calibration["root_z_offset_m"]
        calibration["identity"] = identity
        result.calibration = calibration
        if limb_rotation_source not in ("markers", "native_material_axes"):
            raise ValueError("Unknown native limb rotation source")
        result.limb_rotation_source = limb_rotation_source
        return result

    def _groups(self):
        groups = [("Chest", ("waist_yaw", "waist_roll", "waist_pitch"), "ZXY", None)]
        for side in ("left", "right"):
            label = side.title()
            groups.extend(((label + "Leg", tuple(side + "_" + name for name in ("hip_pitch", "hip_roll", "hip_yaw")), "YXZ", 0),
                           (label + "Shin", (side + "_knee",), "Y", None),
                           (label + "Foot", tuple(side + "_" + name for name in ("ankle_pitch", "ankle_roll")), "YXZ", None),
                           (label + "Arm", tuple(side + "_" + name for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw")), "YXZ", 1),
                           (label + "ForeArm", (side + "_elbow",), "Y", None),
                           (label + "Hand", tuple(side + "_" + name for name in ("wrist_roll", "wrist_pitch", "wrist_yaw")), "XYZ", None)))
        return groups

    def _map_rotations(self, root_soma, global_soma):
        """Legacy raw-rotation-gauge diagnostic, not the runtime source mapping."""
        return self._map_segments(root_soma, global_soma[:, self.segment_ids])[0]

    def _map_anatomical(self, roots, positions, names, *, root_rotations=None, global_rotations=None):
        limbs = None
        if self.limb_rotation_source == "native_material_axes":
            if global_rotations is None:
                raise ValueError("Native material-axis mapping requires the actual native bone rotations")
            limbs = native_limb_material_frames(global_rotations, self.neutral, names)
        rotations = anatomical_frames(positions, names, limb_frames=limbs) @ self.rest_frames.swapaxes(-1, -2)
        if root_rotations is not None:
            # The root is observed. Preserve its native orientation, not marker roundoff.
            rotations[:, 0] = root_rotations
        return self._map_segments(roots, rotations)

    def _map_segments(self, root_soma, segment_rotations):
        n, kin, m = len(root_soma), self.kinematics, self.kinematics.model
        targets = SOMA_TO_MUJOCO @ segment_rotations @ self.offset
        result = np.empty((n, 36))
        result[:, :3] = root_soma @ SOMA_TO_MUJOCO.T
        result[:, 2] += self.root_z_offset_m
        result[:, 3:7] = _quat(targets[:, 0])
        world_r = {kin.pelvis: targets[:, 0]}
        for segment, prefixes, sequence, offset_axis in self.groups:
            joints = [kin.joint_ids[name + "_joint"] for name in prefixes]
            bodies = [int(m.jnt_bodyid[joint]) for joint in joints]
            parent = int(m.body_parentid[bodies[0]])
            relative = world_r[parent].swapaxes(-1, -2) @ targets[:, SEGMENTS.index(segment)]
            desired = kin.rest[bodies[0]].T @ relative
            if len(joints) == 1:
                angle = np.arctan2(desired[:, 0, 2] - desired[:, 2, 0], desired[:, 0, 0] + desired[:, 2, 2])[:, None]
            else:
                angle = Rotation.from_matrix(desired).as_euler(sequence)[:, :len(joints)]
                if offset_axis is not None:
                    rv = Rotation.from_matrix(kin.rest[bodies[1]]).as_rotvec()
                    expected_axis = np.eye(3)["XYZ".index(sequence[offset_axis])]
                    if np.linalg.norm(np.cross(rv, expected_axis)) > 1e-7:
                        raise ValueError("Unexpected noncommuting G1 joint rest frame")
                    angle[:, offset_axis] -= rv @ expected_axis
            hinges = Rotation.from_rotvec(np.concatenate([
                angle[:, i, None] * m.jnt_axis[joint] for i, joint in enumerate(joints)
            ])).as_matrix().reshape(len(joints), n, 3, 3)
            actual = world_r[parent]
            for i, (joint, body) in enumerate(zip(joints, bodies)):
                result[:, kin.column[joint]] = angle[:, i]
                actual = actual @ kin.rest[body] @ hinges[i]
                world_r[body] = actual
        actual_segments = np.stack([world_r[body] for body in kin.body_ids], axis=1)
        error = Rotation.from_matrix((actual_segments @ targets.swapaxes(-1, -2)).reshape(-1, 3, 3)).magnitude().reshape(n, -1)
        residual = dict(scope="every_raw_source_segment_orientation", sample_count=n,
                        rms_rad=float(np.sqrt((error ** 2).mean())), max_rad=float(error.max()),
                        rms_rad_by_segment=dict(zip(SEGMENTS, np.sqrt((error ** 2).mean(0)).tolist())))
        return result, residual

    def inverse(self, root_robot, body_rotations):
        """Robot FK -> constrained native pose, including 58 explicitly inferred rotation DOFs."""
        roots = (np.asarray(root_robot) - [0., 0., self.root_z_offset_m]) @ SOMA_TO_MUJOCO
        mapped = SOMA_TO_MUJOCO.T @ body_rotations[:, self.kinematics.body_ids] @ self.offset.swapaxes(-1, -2)
        return self._native_pose(roots, mapped)

    def _native_pose(self, roots, mapped):
        import torch

        globals_, locals_ = np.empty((len(roots), 30, 3, 3)), np.empty((len(roots), 30, 3, 3))
        lookup = {self.names.index(name): mapped[:, i] for i, name in enumerate(SEGMENTS)}
        hips, chest = mapped[:, 0], mapped[:, 1]
        relative = Rotation.from_matrix(hips.swapaxes(-1, -2) @ chest).as_rotvec()
        spine_length = np.linalg.norm(np.diff(self.neutral[:4], axis=0), axis=-1)
        weights = np.cumsum(spine_length) / spine_length.sum()
        for name, weight in zip(("Spine1", "Spine2"), weights[:2]):
            lookup[self.names.index(name)] = hips @ Rotation.from_rotvec(relative * weight).as_matrix()
        for index, parent in enumerate(self.parents):
            parent_r = np.eye(3) if parent < 0 else globals_[:, parent]
            globals_[:, index] = lookup[index] if index in lookup else parent_r
            locals_[:, index] = parent_r.swapaxes(-1, -2) @ globals_[:, index]
        global_fk, positions, _ = self.skeleton.fk(torch.as_tensor(locals_, dtype=torch.float32), torch.as_tensor(roots, dtype=torch.float32))
        return SomaEstimate(roots, locals_, global_fk.numpy(), positions.numpy())

    def diagnose_pair(self, soma_path, robot_path):
        """Paired residuals and separate G1/native/G1 roundtrip, without quality override."""
        name, human, actual_q = _load_pair(soma_path, robot_path)
        started = perf_counter_ns()
        mapped_q, _ = self._map_anatomical(np.asarray(human["soma_transl"], float),
                                           np.asarray(human["soma_joints"], float) @ SOMA_TO_MUJOCO,
                                           human["joint_names"])
        predicted_pos, predicted_rot = self.kinematics.fk(mapped_q)
        forward_ns = perf_counter_ns() - started
        actual_pos, actual_rot = self.kinematics.fk(actual_q)
        joint_error = (mapped_q[:, 7:] - actual_q[:, 7:] + np.pi) % (2 * np.pi) - np.pi
        ids = self.kinematics.output_ids
        positions = np.linalg.norm(predicted_pos[:, ids] - actual_pos[:, ids], axis=-1)
        body_relative = ((predicted_pos[:, ids] - mapped_q[:, None, :3])
                         - (actual_pos[:, ids] - actual_q[:, None, :3]))
        relative_position_error = np.linalg.norm(body_relative, axis=-1)
        source_displacement = mapped_q[:, :2] - mapped_q[0, :2]
        robot_displacement = actual_q[:, :2] - actual_q[0, :2]
        gain = float((source_displacement * robot_displacement).sum() / (source_displacement ** 2).sum())
        trajectory_gain_error = np.linalg.norm(robot_displacement - gain * source_displacement, axis=-1)
        relative = predicted_rot[:, ids] @ actual_rot[:, ids].swapaxes(-1, -2)
        orientation = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(len(mapped_q), -1)
        inverse_started = perf_counter_ns()
        recovered = self.inverse(actual_q[:, :3], actual_rot)
        roundtrip_q = self._map_rotations(recovered.root_positions_m, recovered.global_rot_mats)
        roundtrip_error = (roundtrip_q[:, 7:] - actual_q[:, 7:] + np.pi) % (2 * np.pi) - np.pi
        roundtrip_ns = perf_counter_ns() - inverse_started
        low, high = self.kinematics.limits.T
        violations = (mapped_q[:, 7:] < low - 1e-6) | (mapped_q[:, 7:] > high + 1e-6)
        shared = [n for n in human["joint_names"] if n in self.names]
        canonical_landmarks = (recovered.world_positions_m - recovered.root_positions_m[:, None]) @ SOMA_TO_MUJOCO.T
        original_landmarks = np.asarray(human["soma_joints"], float)[:, [human["joint_names"].index(n) for n in shared]]
        native_landmarks = canonical_landmarks[:, [self.names.index(n) for n in shared]]
        landmark_error = np.linalg.norm(native_landmarks - original_landmarks, axis=-1)
        return dict(schema=SCHEMA, status="REHEARSAL_ONLY", kind="paired_kinematic_diagnostic_not_quality_approval", clip=name,
                    source_soma_sha256=sha256(soma_path), source_robot_sha256=sha256(robot_path),
                    held_out=sha256(soma_path) != self.calibration["source_soma_sha256"], frames=len(actual_q),
                    forward_input="actual_paired_markers; same_anatomical_gauge_as_native_FK_markers",
                    inverse_roundtrip_scope="legacy_rotation_gauge_algebra; not_native_pose_forward_validation",
                    joint_rmse_rad=float(np.sqrt((joint_error ** 2).mean())), joint_max_abs_rad=float(np.abs(joint_error).max()),
                    joint_rmse_by_name=dict(zip(G1_JOINT_NAMES, np.sqrt((joint_error ** 2).mean(0)).tolist())),
                    body_position_rms_m=float(np.sqrt((positions ** 2).mean())), body_position_max_m=float(positions.max()),
                    root_relative_body_position_rms_m=float(np.sqrt((relative_position_error ** 2).mean())),
                    root_relative_body_position_max_m=float(relative_position_error.max()),
                    paired_root_xy_displacement_gain_diagnostic_only=gain,
                    paired_root_xy_gain_fit_rms_m=float(np.sqrt((trajectory_gain_error ** 2).mean())),
                    runtime_root_xy_scaling=False,
                    body_rotation_rms_rad=float(np.sqrt((orientation ** 2).mean())), body_rotation_max_rad=float(orientation.max()),
                    reconstructed_native_landmark_rms_m=float(np.sqrt((landmark_error ** 2).mean())),
                    native_landmark_names=shared, source_and_native_proportions="not_rescaled; canonical_native_bone_lengths",
                    root_position_rms_m=float(np.sqrt(((mapped_q[:, :3] - actual_q[:, :3]) ** 2).sum(-1).mean())),
                    joint_limit_violation_count=int(violations.sum()),
                    roundtrip_joint_max_abs_rad=float(np.abs(roundtrip_error).max()),
                    roundtrip_root_max_abs_m=float(np.abs(roundtrip_q[:, :3] - actual_q[:, :3]).max()),
                    forward_full_clip_ns=forward_ns, inverse_roundtrip_full_clip_ns=roundtrip_ns,
                    optimizer_updates=0, physical_state_writes=0)

    def retarget(self, source: SomaReference, *, start_time_s=None, constraints=None, boundary=None):
        started = perf_counter_ns()
        if tuple(source.bone_names) != self.names:
            raise ValueError("Source bone order differs from the calibrated native SOMA skeleton")
        start = float(source.timestamps_s[0] if start_time_s is None else start_time_s)
        end = float(source.trusted_until_s)
        if start < source.timestamps_s[0] - 1e-9 or start > end:
            raise ReferenceCoverageError("Requested G1 start is outside actual trusted SOMA samples")
        count=int(np.floor((end-start)*50+1e-8))+1
        start_tick=round(start*150)
        phase=start-start_tick/150
        times=phase+(start_tick+3*np.arange(count))/150
        if len(times) < 2:
            raise ReferenceCoverageError("At least two covered 50Hz samples are required for reference velocities")
        # Represent the same boundary instant exactly as it occurs in the source.
        sample_times=np.clip(times,source.timestamps_s[0],source.timestamps_s[-1])
        if np.any(np.abs(sample_times-times)>16*np.spacing(np.maximum(np.abs(times),1.))):
            raise ReferenceCoverageError('G1 sample times exceed the recorded source interval')
        times=sample_times
        native_q, mapping_residual = self._map_anatomical(source.root_positions_m, source.world_positions_m, source.bone_names,
            root_rotations=source.global_rot_mats[:, 0], global_rotations=source.global_rot_mats)
        source_audit = self._constraint_audit(native_q, source.timestamps_s, scope="every_raw_native_sample")
        if constraints is None and source_audit["status"] != "PASS":
            raise ValueError(f"Anatomical reference constraint violated: {source_audit['first_violation']}; no clipping")
        qpos = np.empty((len(times), 36))
        qpos[:, :3] = np.stack([np.interp(sample_times, source.timestamps_s, native_q[:, i]) for i in range(3)], -1)
        qpos[:, 3:7] = _quat(Slerp(source.timestamps_s, Rotation.from_quat(native_q[:, [4, 5, 6, 3]]))(sample_times).as_matrix())
        angles = native_q[:, 7:]
        qpos[:, 7:] = np.stack([np.interp(sample_times, source.timestamps_s, angles[:, i]) for i in range(29)], -1)
        native_pose_prior = qpos.copy()
        native_pose_prior.flags.writeable = False
        solve_report = None
        if constraints is not None:
            qpos, solve_report = constraints.solve(qpos, times, boundary=boundary)
        low, high = self.kinematics.limits.T
        bad = (qpos[:, 7:] < low - 1e-6) | (qpos[:, 7:] > high + 1e-6)
        if bad.any():
            frame, joint = np.argwhere(bad)[0]
            raise ValueError(f"Retarget joint limit violated: {G1_JOINT_NAMES[joint]} at {times[frame]:.6f}s, "
                             f"{qpos[frame, 7 + joint]:.6f} not in [{low[joint]:.6f},{high[joint]:.6f}]; no clipping")
        pos, rot = self.kinematics.fk(qpos)
        pos, rot = pos[:, self.kinematics.output_ids], rot[:, self.kinematics.output_ids]
        dt = np.diff(times)
        lin = np.diff(pos, axis=0) / dt[:, None, None]
        ang = Rotation.from_matrix((rot[1:] @ rot[:-1].swapaxes(-1, -2)).reshape(-1, 3, 3)).as_rotvec().reshape(-1, len(OFFICIAL_BODY_NAMES), 3) / dt[:, None, None]
        lin, ang = np.concatenate((lin, lin[-1:])), np.concatenate((ang, ang[-1:]))
        qdot = np.diff(qpos[:, 7:], axis=0) / dt[:, None]
        dense_audit = self._constraint_audit(qpos, times, scope="every_returned_50Hz_sample")
        if dense_audit["status"] != "PASS":
            raise ValueError(f"Dense reference constraint violated: {dense_audit['first_violation']}; no clipping")
        qvel = np.concatenate((lin[:, 0], np.einsum("nij,nj->ni", rot[:, 0].swapaxes(-1, -2), ang[:, 0]),
                               np.concatenate((qdot, qdot[-1:]))), -1)
        return G1Reference(times, qpos, qvel, OFFICIAL_BODY_NAMES, pos, _quat(rot), lin, ang,
                           source.scene_id, source.command_id, float(times[-1]),
                           provenance=dict(kind="calibrated_soma_reference", calibration=self.calibration,
                                           source_provenance=dict(source.provenance), optimizer_updates=0,
                                           source_time_bounds_s=source.timestamps_s[[0, -1]].tolist(),
                                           selected_start_time_s=start, selected_trusted_until_s=end,
                                           temporal_resampling="timestamped_30_to_50Hz; no_clock_rebase",
                                           velocity_source="forward_difference_at_50Hz; terminal_reuses_last_derivative",
                                           duration_ns=perf_counter_ns() - started),
                           adaptation=dict(joint_projection=("calibrated_anatomical_FK_gauge; analytic_hinge_and_Euler; no_clipping"
                                               if constraints is None else "joint_constrained_reference_IK"),
                                           source_gauge=SOURCE_GAUGE,
                                           limb_rotation_source=self.limb_rotation_source,
                                           root_mapping=('authorized_reference_trajectory_XYZ_SO3_constraints'
                                               if solve_report is not None and solve_report['schema'] == 'motionbricks.mb_sonic.trajectory_reference.v1'
                                               else 'unchanged_metric_XYZ_and_native_Hips_rotation'),
                                           model_mapping_residual=mapping_residual,
                                           native_pose_prior=dict(timestamps_s=times, qpos=native_pose_prior,
                                               kind='unconstrained_native_anatomical_mapping;kinematic_observer_prior_not_actuated'),
                                           source_constraint_audit=source_audit, dense_constraint_audit=dense_audit,
                                           joint_velocity_limits_rad_s=dict(zip(G1_JOINT_NAMES, self.kinematics.velocity_limits.tolist())),
                                           velocity_limit_source=self.kinematics.velocity_limit_source,
                                           physical_state_writes=0,
                                           **({} if constraints is None else dict(reference_constraints=solve_report))))

    def _constraint_audit(self, qpos, times, *, scope):
        low, high = self.kinematics.limits.T
        margin = np.minimum(qpos[:, 7:] - low, high - qpos[:, 7:])
        speeds = np.abs(np.diff(qpos[:, 7:], axis=0) / np.diff(times)[:, None])
        ratios = speeds / self.kinematics.velocity_limits
        position_bad = np.argwhere(margin < -1e-6)
        velocity_bad = np.argwhere(speeds > self.kinematics.velocity_limits + 1e-6)
        first = None
        if len(position_bad):
            frame, joint = position_bad[0]
            first = dict(kind="position", joint=G1_JOINT_NAMES[joint], time_s=float(times[frame]),
                         value_rad=float(qpos[frame, 7 + joint]), limits_rad=[float(low[joint]), float(high[joint])])
        elif len(velocity_bad):
            frame, joint = velocity_bad[0]
            first = dict(kind="velocity", joint=G1_JOINT_NAMES[joint], interval_s=np.asarray(times)[frame:frame + 2].tolist(),
                         abs_value_rad_s=float(speeds[frame, joint]), limit_rad_s=float(self.kinematics.velocity_limits[joint]))
        return dict(schema="motionbricks.mb_sonic.reference_constraints.v1", source_gauge=SOURCE_GAUGE,
                    status="PASS" if first is None else "FAIL", scope=scope, sample_count=len(qpos),
                    time_bounds_s=np.asarray(times)[[0, -1]].tolist(),
                    position_violation_count=len(position_bad), velocity_violation_count=len(velocity_bad),
                    minimum_position_margin_rad=float(margin.min()), maximum_velocity_ratio=float(ratios.max()),
                    maximum_abs_velocity_rad_s=float(speeds.max()), first_violation=first)

    def audit_joint_limits(self, source: SomaReference):
        """Preserved legacy raw-global gauge diagnostic, including unconsumed tails."""
        if tuple(source.bone_names) != self.names:
            raise ValueError("Source bone order differs from the calibrated native SOMA skeleton")
        q = self._map_rotations(source.root_positions_m, source.global_rot_mats)
        low, high = self.kinematics.limits.T
        bad = (q[:, 7:] < low - 1e-6) | (q[:, 7:] > high + 1e-6)
        violations = [dict(source_frame=int(frame), time_s=float(source.timestamps_s[frame]),
                           joint=G1_JOINT_NAMES[joint], angle_rad=float(q[frame, joint + 7]),
                           limits_rad=[float(low[joint]), float(high[joint])]) for frame, joint in np.argwhere(bad)]
        return dict(schema="motionbricks.mb_sonic.raw_joint_limit_audit.v1", status="FAIL" if violations else "PASS",
                    source_gauge="legacy_raw_global_rotations; not_runtime_reference",
                    scope="every_raw_native_sample; no_consumer_horizon_filter", sample_count=len(q),
                    source_time_bounds_s=source.timestamps_s[[0, -1]].tolist(),
                    source_trusted_until_s=float(source.trusted_until_s), violation_count=len(violations),
                    violations=violations, clipping=False)

    def retarget_packet(self, source: SomaReference, *, start_time_s: float, consumer_horizon_s: float = CONSUMER_HORIZON_S,
                        constraints=None, boundary=None, required_future_s=SONIC_REQUIRED_FUTURE_S):
        """Explicit run-wide horizon; default v1 is 0.9s SONIC + 0.1s guard.

        The full source and its full native-sample limit audit remain attached.
        An infeasible requested window fails; it is never shortened or clipped.
        """
        started = perf_counter_ns()
        start = float(start_time_s)
        horizon = float(consumer_horizon_s)
        minimum = float(required_future_s)+REPLAN_GUARD_S
        if (not np.isfinite([horizon, required_future_s]).all() or required_future_s <= 0
                or horizon < minimum):
            raise ValueError("Consumer horizon must cover the declared tracker future plus the 0.1s guard")
        if abs(horizon * SONIC_FPS - round(horizon * SONIC_FPS)) > 1e-8:
            raise ValueError("Consumer horizon must contain complete 50Hz intervals; no duration rounding")
        contract = (PACKET_CONTRACT if horizon == CONSUMER_HORIZON_S
                    and required_future_s == SONIC_REQUIRED_FUTURE_S else EXPLICIT_PACKET_CONTRACT)
        end = start + horizon
        if (not np.isfinite(start) or start < source.timestamps_s[0] - 1e-9
                or end > min(source.trusted_until_s, source.timestamps_s[-1]) + 1e-9):
            raise ReferenceCoverageError(f"Explicit {horizon:g}s consumer packet requires actual {required_future_s:g}s future plus {REPLAN_GUARD_S:g}s guard "
                                         "and declared scheduler-reserve coverage")
        audit = self.audit_joint_limits(source)
        selected = replace(source, trusted_until_s=end)
        reference = self.retarget(selected, start_time_s=start, constraints=constraints, boundary=boundary)
        adaptation = dict(reference.adaptation, consumer_contract=contract,
                          scope="uniform_all_commands_and_maps", consumer_horizon_s=horizon,
                          tracker_required_future_s=required_future_s, replan_guard_s=REPLAN_GUARD_S,
                          selected_time_bounds_s=[start, end], raw_source_joint_limit_audit=audit,
                          rejected_tail_preserved=True, automatic_shortening=False)
        if required_future_s == SONIC_REQUIRED_FUTURE_S:
            adaptation["sonic_required_future_s"] = SONIC_REQUIRED_FUTURE_S
        if contract == EXPLICIT_PACKET_CONTRACT:
            adaptation["scheduler_reserve_s"] = horizon - minimum
        return replace(reference,
                       provenance=dict(reference.provenance, consumer_contract=contract,
                                       full_source_trusted_until_s=float(source.trusted_until_s),
                                       packet_duration_ns=perf_counter_ns() - started),
                       adaptation=adaptation)

    def _contacts(self, sample, body_names):
        channels = np.zeros(4, np.float32)
        for side_index, side in enumerate(("left", "right")):
            body = body_names.index(side + "_ankle_roll_link")
            for contact in sample["contacts"]:
                if body not in contact["body_ids"] or 0 not in contact["body_ids"] or contact["wrench_contact"][0] <= self.force_threshold_n:
                    continue
                local = _matrix(sample["body_quat_w"][body]).T @ (contact["position_world"] - sample["body_pos_w"][body])
                channels[2 * side_index + int(local[0] >= self.heel_toe_split_x_m[side])] = 1.
        return channels

    def estimate_history(self, executed, *, end_time_s=None):
        import torch

        samples = executed.samples
        times = np.array([sample["timestamp_s"] for sample in samples])
        if len(times) < 2 or (np.diff(times) <= 0).any():
            raise ReferenceCoverageError("Strictly increasing factual physics samples are required")
        end = float(times[-1] if end_time_s is None else end_time_s)
        requested = end - np.arange(3, -1, -1) / 30
        if requested[0] < times[0] - 1e-9 or requested[-1] > times[-1] + 1e-9:
            raise ReferenceCoverageError("Four 30Hz samples require actual executed coverage, without extrapolation")
        names = tuple(executed.body_names)
        ids = [names.index(name) for name in self.kinematics.body_names]
        right = np.searchsorted(times, requested, side="left")
        right = np.minimum(right, len(times) - 1)
        left = np.maximum(right - 1, 0)
        exact = np.isclose(times[right], requested, atol=1e-9, rtol=0)
        left[exact] = right[exact]
        weights = np.divide(requested - times[left], times[right] - times[left], out=np.zeros(4), where=right != left)
        rotations, roots, contacts = [], [], []
        pelvis = names.index("pelvis")
        for i, (a, b) in enumerate(zip(left, right)):
            pa, pb = samples[a]["body_pos_w"][pelvis], samples[b]["body_pos_w"][pelvis]
            roots.append((1 - weights[i]) * pa + weights[i] * pb)
            qa, qb = _matrix(samples[a]["body_quat_w"][ids]), _matrix(samples[b]["body_quat_w"][ids])
            delta = Rotation.from_matrix(qa.swapaxes(-1, -2) @ qb).as_rotvec()
            rotations.append(qa @ Rotation.from_rotvec(delta * weights[i]).as_matrix())
            contact_index = int(np.searchsorted(times, requested[i] + 1e-9, side="right") - 1)
            contacts.append(self._contacts(samples[contact_index], names))
        estimate = self.inverse(np.asarray(roots), np.asarray(rotations))
        local = torch.as_tensor(estimate.local_rot_mats, dtype=torch.float32)[None]
        global_ = torch.as_tensor(estimate.global_rot_mats, dtype=torch.float32)[None]
        features, origin = self.rep(dict(posed_joints=torch.as_tensor(estimate.world_positions_m)[None],
                                         local_joint_rots=local, global_joint_rots=global_,
                                         foot_contacts=torch.as_tensor(np.asarray(contacts))[None]),
                                    to_normalize=False, lengths=torch.tensor([4]), return_init_heading_info=True)
        heading = 2 * torch.atan2(origin["init_heading_quat"][..., 2], origin["init_heading_quat"][..., 0])
        features = self.rep.change_first_heading(features, heading, False, False)
        features[..., self.rep.indices["global_root_pos"]] = torch.as_tensor(estimate.root_positions_m, dtype=features.dtype)[None]
        return NativeSomaHistory(requested, features[0].numpy(), executed.scene_id,
                                 dict(kind="calibrated_executed_g1", calibration=self.calibration,
                                      robot_observations="actual_pelvis_and_body_FK; not_reference", native_rotation_observed_rank=OBSERVED_RANK,
                                      native_rotation_observed_rank_in_bone_order=[OBSERVED_RANK.get(name, 0) for name in self.names],
                                      constrained_global_rotation_mask=[name in SEGMENTS for name in self.names],
                                      robot_configuration_dofs_used=35, native_configuration_dofs=93,
                                      inferred_rotation_dof_count=58, unobserved_joint_constraint="official_native_neutral_local_rotation",
                                      spine_constraint="length_weighted_root_to_chest_SO3_distribution",
                                      contacts="measured_normal_force_and_declared_local_foot_spatial_split; inferred_SOMA_heel_toe_correspondence",
                                      contact_source="latest_actual_snapshot_at_or_before_target_time; no_distance_labels",
                                      resampling="measured_position_brackets_and_SO3_interpolation; no_extrapolation",
                                      source_brackets=np.stack((times[left], times[right]), -1).tolist(), weights=weights.tolist(),
                                      optimizer_updates=0, physical_state_writes=0))
