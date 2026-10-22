"""Material-axis singularities and physical-foot reference constraints."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.mb_sonic.reference_ground import ReferenceFootGround
from motionbricks.mb_sonic.constrained_reference import ReferenceConstraintSolveError
from motionbricks.mb_sonic.retarget import (G1Kinematics, anatomical_frames,
    native_limb_material_frames)
from motionbricks.mb_sonic.runtime import load_physics_scene
from scripts.isolate_mb_continuous_reference import load_fixed_reference


@pytest.fixture(scope='module')
def assets():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/mb_protomotions_wasd_v32_reference_only.json').read_text())
    case = root/'out/mb_sonic_e0_e3_20261018/continuous_reference_isolation_v1'
    _, model, _ = load_physics_scene(case/'scene/scene_manifest.json', config['robot_model'],
        contact_pair_binding=config['robot_contact_pairs'], flat_ground_primitive=config['flat_ground_primitive'])
    return G1Kinematics(config['robot_model']), model, load_fixed_reference(case)[0]


def test_exactly_straight_limb_uses_observed_material_rotation_not_marker_plane():
    rep = build_native_representation()
    names = tuple(rep.skeleton.bone_order_names)
    neutral = rep.skeleton.neutral_joints.numpy()
    global_r = np.tile(np.eye(3), (2, len(names), 1, 1))
    global_r[1] = Rotation.from_euler('z', 1e-5).as_matrix()@global_r[1]
    frames = native_limb_material_frames(global_r, neutral, names)
    for frame in frames.values():
        angle = Rotation.from_matrix(frame[1]@frame[0].T).magnitude()
        assert angle == pytest.approx(1e-5, abs=1e-10)
        np.testing.assert_allclose(np.linalg.det(frame), 1., atol=1e-10)
    positions = np.tile(neutral, (2, 1, 1))
    positions[:, names.index('LeftShin')] = (positions[:, names.index('LeftLeg')]+positions[:, names.index('LeftFoot')])/2
    actual = anatomical_frames(positions, names, limb_frames=frames)
    assert np.isfinite(actual).all()


def test_foot_ground_distance_jacobian_and_physical_model_isolation(assets, monkeypatch):
    kin, model, reference = assets
    original_margin, original_pairs = model.geom_margin.copy(), model.pair_margin.copy()
    ground = ReferenceFootGround(model)
    monkeypatch.setattr(mujoco, 'mj_step', lambda *a: pytest.fail('Reference IK advanced physics'))
    qadr = kin.model.jnt_qposadr[list(kin.joint_ids.values())]
    selected = np.r_[np.arange(6), kin.model.jnt_dofadr[list(kin.joint_ids.values())]]
    pose = reference.qpos[710].copy()
    rows, distances = ground.linearize(pose, qadr, selected)
    assert len(distances) > 0 and distances.min() < -.02
    high, low = pose.copy(), pose.copy()
    high[2] += 1e-6
    low[2] -= 1e-6
    _, plus = ground.linearize(high, qadr, selected)
    _, minus = ground.linearize(low, qadr, selected)
    np.testing.assert_allclose(rows[:, 2], (plus-minus)/2e-6, atol=1e-8)
    np.testing.assert_array_equal(model.geom_margin, original_margin)
    np.testing.assert_array_equal(model.pair_margin, original_pairs)
    assert ground.data.time == 0
    assert ground.audit(pose[None], qadr, selected)['status'] == 'FAIL'


def test_initial_reference_clearance_is_measured_without_moving_initial_state(assets):
    kin, model, reference = assets
    ground = ReferenceFootGround(model)
    qadr = kin.model.jnt_qposadr[list(kin.joint_ids.values())]
    selected = np.r_[np.arange(6), kin.model.jnt_dofadr[list(kin.joint_ids.values())]]
    initial = reference.qpos[:2].copy()
    report = ground.audit(initial, qadr, selected)
    assert report['status'] == 'PASS', report
    np.testing.assert_array_equal(initial, reference.qpos[:2])


def test_support_positions_use_original_capsules_and_exact_point_jacobians(assets):
    kin, model, reference = assets
    root = Path(__file__).resolve().parents[1]
    from motionbricks.mb_sonic.scene import load_scene
    scene = load_scene(root/'out/mb_sonic_e0_e3_20261018/continuous_reference_isolation_v1/scene/scene_manifest.json')
    ground = ReferenceFootGround(model, scene=scene, support_times=np.array([0., 1.]),
        support_confidence=np.array([[1., 1., 0., 0.], [0., 0., 1., 1.]]))
    qadr = kin.model.jnt_qposadr[list(kin.joint_ids.values())]
    selected = np.r_[np.arange(6), kin.model.jnt_dofadr[list(kin.joint_ids.values())]]
    pose = reference.qpos[100].copy()
    ground.linearize(pose, qadr, selected)
    points, jac, normals, gap = ground.support_linearize(selected)
    high = pose.copy()
    high[2] += 1e-6
    ground.linearize(high, qadr, selected)
    higher, _, _, higher_gap = ground.support_linearize(selected)
    np.testing.assert_allclose((higher-points)/1e-6, jac[:, :, 2], atol=1e-8)
    np.testing.assert_allclose((higher_gap-gap)/1e-6, normals[:, 2], atol=1e-8)
    np.testing.assert_array_equal(ground.support_weights([.5]), [[.5, .5, .5, .5]])


@pytest.mark.xfail(strict=True, raises=ReferenceConstraintSolveError,
    reason='Known release blocker: 6cm abrupt reference dip can exhaust the contact QP')
def test_whole_reference_solve_enforces_foot_collision_without_physical_projection(assets, monkeypatch):
    from motionbricks.mb_sonic.trajectory_reference import TrajectoryReferenceSolver
    kin, model, reference = assets
    monkeypatch.setattr(mujoco, 'mj_step', lambda *a: pytest.fail('Reference solve advanced physics'))
    original = model.qpos0.copy()
    poses = np.tile(reference.qpos[0], (39, 1))
    poses[4:, 2] -= .06
    times = np.arange(39)*.02
    ground = ReferenceFootGround(model)
    solver = TrajectoryReferenceSolver(kin, foot_ground=ground, iterations=4)
    result, report = solver.solve(poses, times, boundary=dict(initial_qpos=poses[0].copy(),
        initial_qvel=np.zeros(35), initial_time_s=0., plan_id='same_original_seed'))
    assert report['foot_ground']['status'] == 'PASS'
    assert report['foot_ground']['minimum_distance_m'] >= .001-1e-5
    np.testing.assert_array_equal(result[:2], poses[:2])
    np.testing.assert_array_equal(model.qpos0, original)
    assert ground.data.time == 0 and solver.data.time == 0
