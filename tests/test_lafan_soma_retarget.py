import xml.etree.ElementTree as ET
import json
import sys
import types

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from motionbricks.data.lafan_soma_retarget import (
    GenoSomaRetargeter,
    JOINT_MAP,
    PRIMARY_CHILDREN,
    Y_TO_Z,
    anatomical_frame,
    reference_calibration,
    soma_mjcf,
)
from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.soma_skin_reference import load_official_skinner, surface_heights
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30


def bind_fixture(skeleton):
    target_names = list(skeleton.bone_order_names)
    source_names = list(dict.fromkeys(
        list(JOINT_MAP.values())
        + [child for pair in PRIMARY_CHILDREN.values() for child in pair if child]
    ))
    source_index = {name: index for index, name in enumerate(source_names)}
    positions = np.zeros((len(source_names), 3), dtype=np.float64)
    rotations = np.empty((len(source_names), 3, 3), dtype=np.float64)
    target = skeleton.neutral_joints.numpy().astype(np.float64)
    target -= target[0]

    for target_name, source_name in JOINT_MAP.items():
        positions[source_index[source_name]] = (
            target[target_names.index(target_name)] / 1.1 + [0., .9, 0.]
        )
    positions[source_index["Spine"]] = (
        positions[source_index["Hips"]] +
        .5 * (positions[source_index["Spine1"]] - positions[source_index["Hips"]])
    )

    # A-pose arms: neither arm's shoulder-to-elbow axis is the target T-pose axis.
    for side, sign in (("Left", 1.), ("Right", -1.)):
        shoulder = positions[source_index[side + "Shoulder"]].copy()
        arm = shoulder + [sign * .18, -.085, -.025]
        forearm = arm + [sign * .25, -.055, .01]
        hand = forearm + [sign * .19, -.025, .015]
        positions[source_index[side + "Arm"]] = arm
        positions[source_index[side + "ForeArm"]] = forearm
        positions[source_index[side + "Hand"]] = hand
        positions[source_index[side + "HandMiddle1"]] = hand + [sign * .07, 0., .01]
        positions[source_index[side + "HandMiddleEnd"]] = hand + [sign * .14, 0., .02]
        positions[source_index[side + "ToeBaseEnd"]] = (
            positions[source_index[side + "ToeBase"]] + [0., 0., .06]
        )
    positions[source_index["HeadEnd"]] = positions[source_index["Head"]] + [0., .12, 0.]

    for index in range(len(source_names)):
        rotvec = np.array([.013 * (index + 1), -.009 * (index + 2), .007 * (index + 3)])
        rotations[index] = Rotation.from_rotvec(rotvec).as_matrix()
    return {
        "source_skeleton": {"names": source_names},
        "posed_joints": positions[None],
        "global_rot_mats": rotations[None],
    }


def test_coordinate_and_anatomical_frames_are_proper_rotations():
    np.testing.assert_allclose(Y_TO_Z @ Y_TO_Z.T, np.eye(3))
    assert np.linalg.det(Y_TO_Z) == pytest.approx(1.)
    np.testing.assert_allclose(Y_TO_Z @ [0., 1., 0.], [0., 0., 1.])
    for primary, forward in (([1., 0., 0.], [0., 1., 0.]), ([1., 0., 0.], [1., 0., 0.])):
        frame = anatomical_frame(np.asarray(primary), np.asarray(forward))
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        assert np.linalg.det(frame) == pytest.approx(1.)


def test_mjcf_hierarchy_offsets_fk_and_placeholder_inertias():
    skeleton = SOMASkeleton30()
    xml_text = soma_mjcf(skeleton)
    xml = ET.fromstring(xml_text)
    assert xml.attrib["model"] == "SOMA30_kinematic_only"
    assert xml.find("option").attrib["gravity"] == "0 0 0"
    assert len(list(xml.iter("freejoint"))) == 1
    assert len(list(xml.iter("joint"))) == 29
    assert not list(xml.iter("actuator"))

    bodies = {body.attrib["name"]: body for body in xml.iter("body")}
    assert list(bodies) == skeleton.bone_order_names
    for index, name in enumerate(skeleton.bone_order_names):
        parent = int(skeleton.joint_parents[index])
        offset = skeleton.neutral_joints[index].numpy().astype(np.float64).copy()
        if parent >= 0:
            offset -= skeleton.neutral_joints[parent].numpy()
            parent_name = skeleton.bone_order_names[parent]
            assert bodies[name] in list(bodies[parent_name].findall("body"))
        np.testing.assert_allclose(
            np.fromstring(bodies[name].attrib["pos"], sep=" "), Y_TO_Z @ offset
        )
        inertial = bodies[name].find("inertial")
        assert float(inertial.attrib["mass"]) == 1.
        np.testing.assert_allclose(
            np.fromstring(inertial.attrib["diaginertia"], sep=" "), [.01, .01, .01]
        )

    import mujoco

    model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(1701)
    local = Rotation.from_rotvec(rng.normal(0., .18, (30, 3))).as_matrix()
    root_position = np.array([.2, .9, -.4])
    for joint_index, joint_name in enumerate(skeleton.bone_order_names):
        mujoco_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name + "_joint"
        )
        address = int(model.jnt_qposadr[mujoco_joint])
        start = address + 3 if joint_index == 0 else address
        converted = Y_TO_Z @ local[joint_index] @ Y_TO_Z.T
        data.qpos[start:start + 4] = Rotation.from_matrix(converted).as_quat(scalar_first=True)
    data.qpos[:3] = Y_TO_Z @ root_position
    mujoco.mj_forward(model, data)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in skeleton.bone_order_names]
    mujoco_positions = data.xpos[body_ids] @ Y_TO_Z
    mujoco_rotations = np.stack([
        Y_TO_Z.T @ data.xmat[index].reshape(3, 3) @ Y_TO_Z for index in body_ids
    ])
    expected_rotations, expected_positions, _ = skeleton.fk(
        torch.tensor(local[None], dtype=torch.float32),
        torch.tensor(root_position[None], dtype=torch.float32),
    )
    np.testing.assert_allclose(mujoco_positions, expected_positions[0].numpy(), atol=2e-6)
    np.testing.assert_allclose(mujoco_rotations, expected_rotations[0].numpy(), atol=2e-6)


def test_reference_offsets_preserve_apose_arm_axis_and_actual_bind_rotation():
    skeleton = SOMASkeleton30()
    bind = bind_fixture(skeleton)
    config, calibration = reference_calibration(bind, skeleton)
    assert calibration["uniform_scale"] == pytest.approx(1.1)
    assert "not forced to a T-pose" in calibration["reference_pose"]
    assert config["use_ik_match_table2"] is False

    source_names = bind["source_skeleton"]["names"]
    target_names = skeleton.bone_order_names
    aligned = np.asarray(calibration["reference_global_rot_mats"])
    reference_positions = np.asarray(calibration["reference_positions_relative_root_m"])
    target_neutral = skeleton.neutral_joints.numpy().astype(np.float64)
    target_neutral -= target_neutral[0]

    for target_name, entry in config["ik_match_table1"].items():
        source_name, _, _, offset, quaternion = entry
        source_index = source_names.index(source_name)
        target_index = target_names.index(target_name)
        source_rotation = Y_TO_Z @ bind["global_rot_mats"][0, source_index] @ Y_TO_Z.T
        offset_rotation = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
        desired_rotation = source_rotation @ offset_rotation
        expected_rotation = Y_TO_Z @ aligned[target_index] @ Y_TO_Z.T
        np.testing.assert_allclose(desired_rotation, expected_rotation, atol=1e-12)
        assert np.linalg.det(offset_rotation) == pytest.approx(1., abs=1e-10)

        desired_position = (
            calibration["uniform_scale"] * (Y_TO_Z @ bind["posed_joints"][0, source_index])
            + desired_rotation @ np.asarray(offset)
        )
        root_ref = calibration["uniform_scale"] * bind["posed_joints"][0, 0]
        expected_position = Y_TO_Z @ (reference_positions[target_index] + root_ref)
        np.testing.assert_allclose(desired_position, expected_position, atol=1e-7)

    left_arm_source = source_names.index("LeftArm")
    left_shoulder_source = source_names.index("LeftShoulder")
    source_axis = bind["posed_joints"][0, left_arm_source] - bind["posed_joints"][0, left_shoulder_source]
    target_shoulder = target_names.index("LeftShoulder")
    target_arm = target_names.index("LeftArm")
    target_axis = target_neutral[target_arm] - target_neutral[target_shoulder]
    mapped_axis = aligned[target_shoulder] @ (target_axis / np.linalg.norm(target_axis))
    source_axis /= np.linalg.norm(source_axis)
    np.testing.assert_allclose(mapped_axis, source_axis, atol=1e-12)
    assert source_axis[1] < -.3
    assert not np.allclose(aligned[target_shoulder], np.eye(3), atol=1e-3)


def test_noncentimetre_bind_is_rejected():
    bind = bind_fixture(SOMASkeleton30())
    bind["posed_joints"] *= 100.
    with pytest.raises(ValueError, match="units"):
        reference_calibration(bind, SOMASkeleton30())


def test_endpoint_position_cost_overrides_only_foot_and_toebase(tmp_path, monkeypatch):
    skeleton = SOMASkeleton30()
    bind = bind_fixture(skeleton)

    class Params:
        ROBOT_XML_DICT = {}
        IK_CONFIG_DICT = {}

    class Engine:
        model = types.SimpleNamespace(jnt_qposadr=np.arange(len(skeleton.bone_order_names)) * 4)

    captured = []

    def fake_gmr(*_args, **_kwargs):
        config_path = Params.IK_CONFIG_DICT["motionbricks_geno"]["motionbricks_soma30"]
        captured.append(json.loads(config_path.read_text(encoding="utf-8")))
        return Engine()

    class MjObject:
        mjOBJ_JOINT = 0

    fake_mujoco = types.SimpleNamespace(
        mjtObj=MjObject,
        mj_name2id=lambda _model, _kind, name: skeleton.bone_order_names.index(
            name.removesuffix("_joint")
        ),
    )
    monkeypatch.setattr(
        "motionbricks.data.lafan_soma_retarget.load_gmr_core",
        lambda _repository: (fake_gmr, Params, {}),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    baseline, _ = reference_calibration(bind, skeleton)
    default = GenoSomaRetargeter(bind, skeleton, "unused", tmp_path / "default")
    default_table = captured[-1]["ik_match_table1"]
    for name, entry in default_table.items():
        assert entry[1] == baseline["ik_match_table1"][name][1]
    assert default.calibration["endpoint_position_cost"] == 50.

    custom = GenoSomaRetargeter(
        bind, skeleton, "unused", tmp_path / "custom", endpoint_position_cost=500.
    )
    custom_table = captured[-1]["ik_match_table1"]
    for name, entry in custom_table.items():
        if name.endswith(("Foot", "ToeBase")):
            assert entry[1] == 500.
        else:
            assert entry[1] == baseline["ik_match_table1"][name][1]
    assert custom.calibration["endpoint_position_cost"] == 500.


def test_selected_vertex_official_skin_matches_full_mesh_for_four_poses():
    skeleton = SOMASkeleton30()
    surfaces, _, _, _ = official_sole_surfaces(skeleton)
    skinner = load_official_skinner(skeleton)
    rng = np.random.default_rng(933)
    local = Rotation.from_rotvec(rng.normal(0., .16, (4 * 30, 3))).as_matrix().reshape(
        4, 30, 3, 3
    )
    roots = rng.normal(0., .2, (4, 3)).astype(np.float32)
    local_tensor = torch.from_numpy(local.astype(np.float32))
    root_tensor = torch.from_numpy(roots)
    _, posed, _ = skeleton.fk(local_tensor, root_tensor)

    with torch.inference_mode():
        full_vertices = skinner.skin(local_tensor, posed, rot_is_global=False)
        full_heights = torch.stack(list(surface_heights(full_vertices, surfaces).values()), -1)

    selected = np.unique(np.concatenate([surface["selected_indices"] for surface in surfaces]))
    selected_lookup = {int(vertex): order for order, vertex in enumerate(selected)}
    for attribute in ("bind_vertices", "lbs_indices", "lbs_weights"):
        setattr(skinner, attribute, getattr(skinner, attribute)[selected])
    selected_surfaces = [
        dict(surface, selected_indices=np.asarray([
            selected_lookup[int(vertex)] for vertex in surface["selected_indices"]
        ]))
        for surface in surfaces
    ]
    with torch.inference_mode():
        selected_vertices = skinner.skin(local_tensor, posed, rot_is_global=False)
        selected_heights = torch.stack(
            list(surface_heights(selected_vertices, selected_surfaces).values()), -1
        )
    error = float((full_heights - selected_heights).abs().max())
    assert error <= 1e-6


@pytest.mark.parametrize(
    "missing_anchor",
    ["HeadEnd", "LeftHandMiddle1", "RightHandMiddle1", "LeftToeBaseEnd"],
)
def test_missing_source_child_anchors_fail_calibration(missing_anchor):
    skeleton = SOMASkeleton30()
    bind = bind_fixture(skeleton)
    keep = [index for index, name in enumerate(bind["source_skeleton"]["names"])
            if name != missing_anchor]
    bind["source_skeleton"]["names"] = [bind["source_skeleton"]["names"][index] for index in keep]
    bind["posed_joints"] = bind["posed_joints"][:, keep]
    bind["global_rot_mats"] = bind["global_rot_mats"][:, keep]
    with pytest.raises(ValueError):
        reference_calibration(bind, skeleton)
