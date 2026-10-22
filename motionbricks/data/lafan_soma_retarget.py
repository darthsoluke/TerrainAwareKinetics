"""Offline Geno-to-SOMA adapter for the unmodified GMR/Mink retargeter.

MuJoCo is used only for articulated kinematics/Jacobians here. The generated
inertias are numerical placeholders, not a SOMA physical mass model. This module
does not simulate motion, infer forces, smooth motion or lock feet at runtime.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


GMR_COMMIT = "bb1bbe40774794fceb2a7c579a3464a28e68c844"
Y_TO_Z = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
JOINT_MAP = {
    "Hips": "Hips", "Spine1": "Spine1", "Spine2": "Spine2", "Chest": "Spine3",
    "Neck1": "Neck", "Neck2": "Neck1", "Head": "Head",
    **{side + target: side + source for side in ("Left", "Right")
       for target, source in (("Shoulder", "Shoulder"), ("Arm", "Arm"),
                              ("ForeArm", "ForeArm"), ("Hand", "Hand"),
                              ("Leg", "UpLeg"), ("Shin", "Leg"),
                              ("Foot", "Foot"), ("ToeBase", "ToeBase"))},
}

PRIMARY_CHILDREN = {"Hips": ("Spine", "Spine1"), "Spine1": ("Spine2", "Spine2"),
                    "Spine2": ("Spine3", "Chest"), "Chest": ("Neck", "Neck1"),
                    "Neck1": ("Neck1", "Neck2"), "Neck2": ("Head", "Head"),
                    "Head": ("HeadEnd", None),
                    **{side + target: (side + source, side + child)
                       for side in ("Left", "Right") for target, source, child in (
                           ("Shoulder", "Arm", "Arm"), ("Arm", "ForeArm", "ForeArm"),
                           ("ForeArm", "Hand", "Hand"), ("Hand", "HandMiddle1", "HandMiddleEnd"),
                           ("Leg", "Leg", "Shin"), ("Shin", "Foot", "Foot"),
                           ("Foot", "ToeBase", "ToeBase"))},
                    "LeftToeBase": ("LeftToeBaseEnd", None),
                    "RightToeBase": ("RightToeBaseEnd", None)}


def anatomical_frame(primary, forward):
    """Right-handed frame from a bone axis and projected anatomical forward."""
    primary = primary / np.linalg.norm(primary)
    secondary = forward - np.dot(forward, primary) * primary
    if np.linalg.norm(secondary) < .05:
        # Feet run along forward: up is the non-collinear anatomical reference.
        secondary = np.array([0., 1., 0.]) - primary[1] * primary
    secondary /= np.linalg.norm(secondary)
    return np.stack([primary, secondary, np.cross(primary, secondary)], axis=-1)


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _numbers(value):
    return " ".join(format(float(x), ".17g") for x in np.ravel(value))


def soma_mjcf(skeleton, fps=60.):
    """Generate the official SOMA hierarchy with a free root and ball joints."""
    names = list(skeleton.bone_order_names)
    parents = _array(skeleton.joint_parents).astype(int)
    neutral = _array(skeleton.neutral_joints).astype(np.float64)
    neutral -= neutral[0]
    root = ET.Element("mujoco", model="SOMA30_kinematic_only")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep=str(1. / fps), gravity="0 0 0")
    world = ET.SubElement(root, "worldbody")
    bodies = {}
    for index, name in enumerate(names):
        parent = int(parents[index])
        offset = neutral[index] if parent < 0 else neutral[index] - neutral[parent]
        body = ET.SubElement(world if parent < 0 else bodies[parent], "body",
                             name=name, pos=_numbers(Y_TO_Z @ offset))
        bodies[index] = body
        ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia="0.01 0.01 0.01")
        ET.SubElement(body, "freejoint" if parent < 0 else "joint",
                      **({"name": name + "_joint"} if parent < 0 else
                         {"name": name + "_joint", "type": "ball", "limited": "false"}))
    return ET.tostring(root, encoding="unicode")


def reference_calibration(bind, skeleton):
    """Fit scale from leg lengths; reference offsets are exact geometric differences."""
    names = bind["source_skeleton"]["names"]
    source = bind["posed_joints"][0]
    source_rots = bind["global_rot_mats"][0]
    target = _array(skeleton.neutral_joints).astype(np.float64)
    target -= target[0]
    target_names = list(skeleton.bone_order_names)
    if any(name not in names for name in JOINT_MAP.values()):
        raise ValueError("Geno hierarchy is missing anatomical mapping anchors")
    source_lengths, target_lengths = [], []
    for side in ("Left", "Right"):
        sp = source[[names.index(side + suffix) for suffix in ("UpLeg", "Leg", "Foot")]]
        tp = target[[target_names.index(side + suffix) for suffix in ("Leg", "Shin", "Foot")]]
        source_lengths.append(float(np.linalg.norm(np.diff(sp, axis=0), axis=-1).sum()))
        target_lengths.append(float(np.linalg.norm(np.diff(tp, axis=0), axis=-1).sum()))
    if not all(.5 < length < 1.3 for length in source_lengths):
        raise ValueError("Source leg length is not plausible in metres; verify explicit BVH units")
    scale = float(np.mean(target_lengths) / np.mean(source_lengths))
    aligned_global = np.tile(np.eye(3), (len(target_names), 1, 1))
    rotation_offsets = {}
    for target_name, source_name in JOINT_MAP.items():
        si, ti = names.index(source_name), target_names.index(target_name)
        source_child, target_child = PRIMARY_CHILDREN[target_name]
        source_direction = source[names.index(source_child)] - source[si]
        target_direction = (target[target_names.index(target_child)] - target[ti]
                            if target_child else np.array([0., 0., 1.]) if "ToeBase" in target_name
                            else np.array([0., 1., 0.]))
        source_local_axis = source_rots[si].T @ source_direction
        reference_axis = np.array([0., 1., 0.]) if target_name.endswith(("Foot", "ToeBase")) else np.array([0., 0., 1.])
        source_local_forward = source_rots[si].T @ reference_axis
        source_frame = anatomical_frame(source_local_axis, source_local_forward)
        target_frame = anatomical_frame(target_direction, reference_axis)
        offset = source_frame @ target_frame.T
        rotation_offsets[target_name] = offset
        aligned_global[ti] = source_rots[si] @ offset
    parents = _array(skeleton.joint_parents).astype(int)
    reference_positions = np.zeros_like(target)
    for index, parent in enumerate(parents):
        if parent >= 0:
            reference_positions[index] = (reference_positions[parent] +
                                          aligned_global[parent] @ (target[index] - target[parent]))
    table = {}
    for target_name, source_name in JOINT_MAP.items():
        si, ti = names.index(source_name), target_names.index(target_name)
        offset = aligned_global[ti].T @ (reference_positions[ti] - scale * (source[si] - source[0]))
        rotation_offset = Y_TO_Z @ rotation_offsets[target_name] @ Y_TO_Z.T
        # GMR's orientation-first upper-body / position-first feet decomposition.
        # Do not force different-length arms to the uniformly scaled hand location.
        if target_name == "Hips":
            pos_weight, rot_weight = 1000., 100.
        elif target_name.endswith(("Foot", "ToeBase")):
            pos_weight, rot_weight = 50., 10.
        elif target_name.endswith(("Leg", "Shin")):
            pos_weight, rot_weight = 10., 10.
        else:
            pos_weight, rot_weight = 0., 100.
        table[target_name] = [source_name, pos_weight, rot_weight, (Y_TO_Z @ offset).tolist(),
                              Rotation.from_matrix(rotation_offset).as_quat(scalar_first=True).tolist()]
    config = {"human_root_name": "Hips", "robot_root_name": "Hips", "ground_height": 0.,
              "human_height_assumption": 1., "human_scale_table": {name: scale for name in JOINT_MAP.values()},
              "use_ik_match_table1": True, "use_ik_match_table2": False,
              "ik_match_table1": table, "ik_match_table2": {}}
    return config, {"uniform_scale": scale, "source_leg_lengths_m": source_lengths,
                    "target_leg_lengths_m": target_lengths, "joint_mapping": JOINT_MAP,
                    "unmapped_target_joints": [name for name in target_names if name not in JOINT_MAP],
                    "reference_pose": "bone-local anatomical axes; actual Geno bind is not forced to a T-pose",
                    "reference_global_rot_mats": aligned_global.tolist(),
                    "reference_positions_relative_root_m": reference_positions.tolist(),
                    "runtime_ik": False, "per_frame_floor_shift": False,
                    "objective_weights": "GMR orientation-first upper body and position-first feet; root trajectory prioritized; custom SOMA adapter"}


def load_gmr_core(repository):
    """Load the upstream core without importing its viewer/SMPL dependencies."""
    repository = Path(repository).resolve()
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if commit != GMR_COMMIT:
        raise ValueError(f"GMR must be pinned to {GMR_COMMIT}, got {commit}")
    package_path = repository / "general_motion_retargeting"
    hashes = {}
    for filename in ("params.py", "motion_retarget.py"):
        path = package_path / filename
        original = subprocess.check_output(["git", "-C", str(repository), "show",
                                            f"HEAD:general_motion_retargeting/{filename}"])
        if path.read_bytes().replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
            raise ValueError(f"Upstream GMR core was modified: {filename}")
        hashes[filename] = hashlib.sha256(original).hexdigest()
    package_name = "_motionbricks_offline_gmr"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package
    modules = {}
    for name in ("params", "motion_retarget"):
        spec = importlib.util.spec_from_file_location(package_name + "." + name, package_path / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["motion_retarget"].GeneralMotionRetargeting, modules["params"], hashes


class SomaGMRKinematics:
    """Shared qpos conversion and official-FK parity for offline SOMA adapters."""

    def qpos_from_motion(self, local, root):
        qpos = self.engine.model.qpos0.copy()
        qpos[:3] = Y_TO_Z @ root
        for index, address in enumerate(self.addresses):
            rotation = Y_TO_Z @ local[index] @ Y_TO_Z.T
            start = address + 3 if index == 0 else address
            qpos[start:start + 4] = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
        return qpos

    def motion_from_qpos(self, qpos):
        local = np.empty((len(self.addresses), 3, 3))
        for index, address in enumerate(self.addresses):
            start = address + 3 if index == 0 else address
            rotation = Rotation.from_quat(qpos[start:start + 4], scalar_first=True).as_matrix()
            local[index] = Y_TO_Z.T @ rotation @ Y_TO_Z
        return local, Y_TO_Z.T @ qpos[:3]

    def validate_fk(self, local, root):
        """Verify solver FK in arbitrary poses, independently of task weights."""
        import mujoco
        import torch
        data = mujoco.MjData(self.engine.model)
        data.qpos[:] = self.qpos_from_motion(local, root)
        mujoco.mj_kinematics(self.engine.model, data)
        ids = [mujoco.mj_name2id(self.engine.model, mujoco.mjtObj.mjOBJ_BODY, name)
               for name in self.skeleton.bone_order_names]
        positions = data.xpos[ids] @ Y_TO_Z
        global_rots = Y_TO_Z.T @ data.xmat[ids].reshape(-1, 3, 3) @ Y_TO_Z
        rotations, official, _ = self.skeleton.fk(torch.tensor(local[None], dtype=torch.float32),
                                               torch.tensor(root[None], dtype=torch.float32))
        error = max(float(np.abs(positions - official[0].numpy()).max()),
                    float(np.abs(global_rots - rotations[0].numpy()).max()))
        if error > 2e-6:
            raise ValueError(f"Solver skeleton disagrees with official SOMA FK: {error}")
        return error


class GenoSomaRetargeter(SomaGMRKinematics):
    """GMR adapter with official SOMA FK cross-checks and no motion postprocessing."""

    def __init__(self, bind, skeleton, repository, work_directory, endpoint_position_cost=50.):
        self.skeleton = skeleton
        self.source_names = bind["source_skeleton"]["names"]
        config, self.calibration = reference_calibration(bind, skeleton)
        if not np.isfinite(endpoint_position_cost) or endpoint_position_cost <= 0:
            raise ValueError("Endpoint position cost must be finite and positive")
        for name, entry in config["ik_match_table1"].items():
            if name.endswith(("Foot", "ToeBase")):
                entry[1] = float(endpoint_position_cost)
        self.calibration["endpoint_position_cost"] = float(endpoint_position_cost)
        directory = Path(work_directory)
        directory.mkdir(parents=True, exist_ok=True)
        xml_path, config_path = directory / "soma30.xml", directory / "geno_to_soma30.json"
        xml_path.write_text(soma_mjcf(skeleton), encoding="utf-8")
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        implementation, params, hashes = load_gmr_core(repository)
        params.ROBOT_XML_DICT["motionbricks_soma30"] = xml_path
        params.IK_CONFIG_DICT["motionbricks_geno"] = {"motionbricks_soma30": config_path}
        self.engine = implementation("motionbricks_geno", "motionbricks_soma30", verbose=False)
        self.calibration["gmr_commit"] = GMR_COMMIT
        self.calibration["gmr_core_sha256"] = hashes
        import mujoco
        self.addresses = [int(self.engine.model.jnt_qposadr[
            mujoco.mj_name2id(self.engine.model, mujoco.mjtObj.mjOBJ_JOINT, name + "_joint")])
            for name in skeleton.bone_order_names]

    def solve(self, positions, global_rotations):
        human = {}
        for source_name in JOINT_MAP.values():
            index = self.source_names.index(source_name)
            rotation = Y_TO_Z @ global_rotations[index] @ Y_TO_Z.T
            human[source_name] = [Y_TO_Z @ positions[index],
                                 Rotation.from_matrix(rotation).as_quat(scalar_first=True)]
        qpos = self.engine.retarget(human, offset_to_ground=False).copy()
        local, root = self.motion_from_qpos(qpos)
        if not np.isfinite(local).all() or not np.isfinite(root).all():
            raise ValueError("Retarget solver produced a non-finite pose")
        return local, root
