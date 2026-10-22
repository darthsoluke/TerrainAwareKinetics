"""Independent CPU contract checks against the pinned official deploy helpers."""

import importlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.protomotions_policy import (
    FrozenProtoMotionsPolicy,
    PROTOMOTIONS_ONNX_SHA256,
    PROTOMOTIONS_REVISION,
)
from motionbricks.mb_sonic.reference import G1Reference, OFFICIAL_BODY_NAMES, file_sha256
from motionbricks.mb_sonic.sonic_policy import ObservationHistory, Proprioception


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "out/vendor/ProtoMotions"
MODEL = VENDOR / "data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx"
METADATA = MODEL.with_suffix(".yaml")
MODEL_SHA = "a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c"
METADATA_SHA = "57d81b60dda9aeb30c57d3428a822f203a83bca399c5abe82ffea7a9aaf8b139"
HELPER_SHAS = {
    "tracker_inputs.py": "5f613be0d5116f49674a6f853dfad47ef21aa27582a52abdebde949837deec9f",
    "state_utils.py": "7c1cbc9880537ced4cc8ab8082616c0fda02de4d1011107166296244a68c421a",
}
REFERENCE = ROOT / "out/mb_sonic_e0_e3_20261018/constrained_reference_fast_v1/failed_dense66_packet.npz"


def wxyz(rotation):
    return rotation.as_quat()[[3, 0, 1, 2]]


@pytest.fixture(scope="module")
def official():
    assert MODEL.stat().st_size == 22606590
    assert file_sha256(MODEL) == MODEL_SHA
    assert file_sha256(METADATA) == METADATA_SHA
    for name, expected in HELPER_SHAS.items():
        assert file_sha256(VENDOR / "deployment" / name) == expected
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(VENDOR))
        state = importlib.import_module("deployment.state_utils")
        tracker = importlib.import_module("deployment.tracker_inputs")
        assert Path(state.__file__).resolve() == (VENDOR / "deployment/state_utils.py").resolve()
        assert Path(tracker.__file__).resolve() == (VENDOR / "deployment/tracker_inputs.py").resolve()
        yield state, tracker


@pytest.fixture(scope="module")
def direct_session(official):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(MODEL), sess_options=options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    return session


@pytest.fixture
def policy(official):
    return FrozenProtoMotionsPolicy(MODEL, METADATA, onnx_sha256=MODEL_SHA,
                                    metadata_sha256=METADATA_SHA, threads=1)


@pytest.fixture(scope="module")
def reference():
    # Genuine saved MB reference after the approved offline constrained conversion.
    with np.load(REFERENCE, allow_pickle=False) as archive:
        times = archive["timestamps_s"]
        return G1Reference(times, archive["solved_qpos"], archive["qvel"], OFFICIAL_BODY_NAMES,
            archive["body_pos_w"], archive["body_quat_w"], archive["body_lin_vel_w"],
            archive["body_ang_vel_w"], "000", "saved_failed_reference_contract", float(times[-1]))


def sensor_fixture(reference, index, torso_yaw):
    """Deliberately distinct sensor poses for contract probing, not a physics trace."""
    pelvis = wxyz(Rotation.from_euler("xyz", [-.19, .23, -1.17]))
    torso = wxyz(Rotation.from_euler("xyz", [.31, -.27, torso_yaw]))
    current = Proprioception(float(reference.timestamps_s[index]),
        reference.joint_pos[index] + np.linspace(-.012, .017, 29),
        reference.joint_vel[index] + np.linspace(.071, -.037, 29),
        pelvis, np.array([.23, -.41, .67]), np.linspace(-.8, .6, 29))
    return current, torso


def expected_inputs(official, metadata, start_reference, window, current, current_torso,
                    reset_torso, previous=None):
    state, tracker = official
    anchor = window.body_names.index("torso_link")
    current_body_rot = state.mujoco_wxyz_to_xyzw(np.stack((current.base_quat_wxyz, current_torso)))
    current_anchor = state.compute_anchor_rot_np(current_body_rot, anchor_body_index=1)
    world_omega = state.quat_rotate_np(current_body_rot[0], current.base_angular_velocity)
    current_body_omega = np.stack((world_omega, [1.7, -2.3, .11]))
    local_omega = state.compute_root_local_ang_vel_np(current_body_rot, current_body_omega, root_body_index=0)
    alignment = tracker.init_tracker_alignment(
        robot_anchor_pos=np.array([1.7, -2.4, .91]),
        robot_anchor_rot=state.mujoco_wxyz_to_xyzw(reset_torso),
        motion_anchor_pos=start_reference.body_pos_w[0, anchor],
        motion_anchor_rot=state.mujoco_wxyz_to_xyzw(start_reference.body_quat_w[0, anchor]))
    future = tracker.align_future_motion_refs(dict(
        body_rot=state.mujoco_wxyz_to_xyzw(window.body_quat_w[1:]),
        dof_pos=window.joint_pos[1:], dof_vel=window.joint_vel[1:]), alignment)
    runtime = metadata["_runtime"]
    return tracker.build_tracker_onnx_inputs(
        onnx_input_names=runtime["onnx_in_names"], onnx_name_to_key=runtime["onnx_name_to_in_key"],
        dof_pos=current.joint_positions, dof_vel=current.joint_velocities,
        anchor_rot=current_anchor, root_local_ang_vel=local_omega,
        future_refs=future, anchor_body_index=anchor, num_dofs=29, prev_actions=previous)


def compare_official_command(command, expected, session, output_names):
    assert command.inputs.keys() == expected.keys()
    for name in expected:
        assert command.inputs[name].dtype == np.float32
        assert command.inputs[name].shape == expected[name].shape
        # Official NumPy quaternion algebra is float32; the adapter uses SciPy float64.
        np.testing.assert_allclose(command.inputs[name], expected[name], atol=2e-7, rtol=0, err_msg=name)
    action, target, kp, kd = session.run(output_names, expected)
    np.testing.assert_allclose(command.raw_action, action[0], atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(command.joint_target, target[0], atol=1e-5, rtol=1e-6)
    np.testing.assert_array_equal(command.processed_action_mujoco, command.joint_target)
    assert np.isfinite(kp).all() and np.isfinite(kd).all()


def test_real_pinned_deployment_identity(policy, direct_session):
    assert PROTOMOTIONS_REVISION == "f69fdcaabff31846b4d2247207417b57dc788828"
    assert PROTOMOTIONS_ONNX_SHA256 == MODEL_SHA
    assert policy.session.get_providers() == direct_session.get_providers() == ["CPUExecutionProvider"]
    assert policy.metadata["robot"]["anchor_body_name"] == "torso_link"
    assert policy.metadata["robot"]["root_body_name"] == "pelvis"
    assert policy.metadata["policy_inputs"][4]["output_key"] == "robot_action"


@pytest.mark.parametrize("torso_yaw", [.63, 2.17, -2.31])
def test_nonzero_heading_torso_anchor_and_pelvis_local_gyro_match_official(
        policy, official, direct_session, reference, torso_yaw):
    window = policy.reference_window(reference, reference.timestamps_s[0])
    current, torso = sensor_fixture(reference, 0, torso_yaw)
    history = ObservationHistory()
    history.append(current)
    anchor = window.body_names.index("torso_link")
    assert anchor != policy.metadata["robot"]["anchor_body_index"]
    assert not np.allclose(torso, current.base_quat_wxyz)
    assert not np.allclose(window.body_quat_w[:, anchor], window.qpos[:, 3:7])
    before = {name: getattr(reference, name).tobytes() for name in
              ("timestamps_s", "qpos", "qvel", "body_pos_w", "body_quat_w")}
    policy.reset_heading(torso, window.body_quat_w[0, anchor])
    command = policy.infer(window, history, anchor_quat_wxyz=torso)
    expected = expected_inputs(official, policy.metadata, reference, window, current, torso, torso)
    compare_official_command(command, expected, direct_session, policy.output_names)
    np.testing.assert_array_equal(command.inputs["current_anchor_rot"][0], torso[[1, 2, 3, 0]].astype(np.float32))
    np.testing.assert_array_equal(command.inputs["current_root_local_ang_vel"][0], current.base_angular_velocity.astype(np.float32))
    np.testing.assert_array_equal(command.inputs["historical_processed_actions"], np.zeros((1, 1, 29), np.float32))
    assert not np.array_equal(command.inputs["current_anchor_rot"][0], current.base_quat_wxyz[[1, 2, 3, 0]].astype(np.float32))
    for name, original in before.items():
        assert getattr(reference, name).tobytes() == original


def test_previous_processed_target_not_raw_action_or_sonic_history(
        policy, official, direct_session, reference):
    window = policy.reference_window(reference, reference.timestamps_s[0])
    current, torso = sensor_fixture(reference, 0, 1.29)
    history = ObservationHistory()
    history.append(current)
    anchor = window.body_names.index("torso_link")
    policy.reset_heading(torso, window.body_quat_w[0, anchor])
    first = policy.infer(window, history, anchor_quat_wxyz=torso)
    expected_first = expected_inputs(official, policy.metadata, reference, window, current, torso, torso)
    compare_official_command(first, expected_first, direct_session, policy.output_names)
    previous = first.joint_target.copy()
    assert not np.allclose(previous, first.raw_action)
    assert not np.allclose(previous, current.last_action_isaac)
    first.joint_target[:] = 41.
    first.processed_action_mujoco[:] = -37.
    next_current, next_torso = sensor_fixture(reference, 1, 1.41)
    history.append(next_current)
    next_window = policy.reference_window(reference, next_current.time_s)
    second = policy.infer(next_window, history, anchor_quat_wxyz=next_torso)
    expected_second = expected_inputs(official, policy.metadata, reference, next_window,
                                      next_current, next_torso, torso, previous=previous)
    compare_official_command(second, expected_second, direct_session, policy.output_names)
    np.testing.assert_array_equal(second.inputs["historical_processed_actions"][0, 0], previous.astype(np.float32))


def test_reset_heading_discards_previous_target_and_realigns_official(
        policy, official, direct_session, reference):
    window = policy.reference_window(reference, reference.timestamps_s[0])
    current, torso = sensor_fixture(reference, 0, .84)
    history = ObservationHistory()
    history.append(current)
    anchor = window.body_names.index("torso_link")
    policy.reset_heading(torso, window.body_quat_w[0, anchor])
    policy.infer(window, history, anchor_quat_wxyz=torso)
    assert policy.previous_target is not None
    new_current, new_torso = sensor_fixture(reference, 1, -1.92)
    next_window = policy.reference_window(reference, new_current.time_s)
    policy.reset_heading(new_torso, next_window.body_quat_w[0, anchor])
    assert policy.previous_target is None
    fresh_history = ObservationHistory()
    fresh_history.append(new_current)
    result = policy.infer(next_window, fresh_history, anchor_quat_wxyz=new_torso)
    expected = expected_inputs(official, policy.metadata, next_window, next_window,
                               new_current, new_torso, new_torso)
    compare_official_command(result, expected, direct_session, policy.output_names)
    np.testing.assert_array_equal(result.inputs["historical_processed_actions"], np.zeros((1, 1, 29), np.float32))
