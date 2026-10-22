"""Whole-trajectory IK continuity, manifold derivatives and physical isolation."""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.retarget import G1Kinematics
from motionbricks.mb_sonic.sonic_policy import DEFAULT_ANGLES
from motionbricks.mb_sonic.trajectory_reference import TrajectoryReferenceSolver, right_jacobian


@pytest.fixture(scope='module')
def kinematics():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/mb_protomotions_wasd_v19_feedback.json').read_text())
    return G1Kinematics(config['robot_model'])


def poses(count=31):
    result = np.tile(np.r_[.3, -.2, .79, 1., 0., 0., 0., DEFAULT_ANGLES], (count, 1))
    return result


def boundary(reference, times, until):
    return dict(timestamps_s=times, root_pose=reference[:, :7], joint_positions=reference[:, 7:],
                adoption_until_s=until, plan_id='previous_accepted')


@pytest.mark.parametrize('task_frame',['world','root'])
def test_so3_chart_right_jacobian_and_cartesian_tasks_match_finite_difference(kinematics, task_frame):
    solver = TrajectoryReferenceSolver(kinematics, pose_task_frame=task_frame)
    basis = Rotation.from_euler('xyz', [.3, -.2, .7])
    current = solver._coordinates(poses(1), basis)[0]
    current[3:6] = [.4, -.3, .2]
    current[6:] += .01
    pose = solver._poses(current[None], basis)[0]
    solver._update(pose)
    for task in solver.frames:
        task.set_target_from_configuration(solver.configuration)
    transform = np.eye(35)
    transform[3:6, 3:6] = right_jacobian(current[3:6])
    analytic = [task.compute_jacobian(solver.configuration)[:, solver.selected]@transform for task in solver.frames]
    for i in range(35):
        plus, minus = current.copy(), current.copy()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        solver._update(solver._poses(plus[None], basis)[0])
        positive = [task.compute_error(solver.configuration).copy() for task in solver.frames]
        solver._update(solver._poses(minus[None], basis)[0])
        negative = [task.compute_error(solver.configuration).copy() for task in solver.frames]
        for jacobian, a, b in zip(analytic, positive, negative):
            np.testing.assert_allclose(jacobian[:, i], (a-b)/2e-6, atol=1e-5, rtol=0)


def test_constant_reference_is_unchanged_and_never_steps_physics(kinematics, monkeypatch):
    monkeypatch.setattr(mujoco, 'mj_step', lambda *args: pytest.fail('Reference solve advanced physics'))
    nominal = poses()
    original = nominal.copy()
    times = np.arange(len(nominal))*.02
    start = dict(initial_qpos=nominal[0], initial_qvel=np.zeros(35), initial_time_s=0., plan_id='measured_start')
    solver = TrajectoryReferenceSolver(kinematics)
    result, report = solver.solve(nominal, times, boundary=start)
    np.testing.assert_allclose(result, nominal, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(nominal, original)
    assert solver.data.time == 0 and report['physical_state_writes'] == 0
    assert report['committed_frames'] == 2


def test_reference_root_registration_does_not_change_body_pose(kinematics):
    nominal = poses()
    time = np.arange(len(nominal))*.02
    nominal[:,7] += .1*np.sin(time*5)
    shifted = nominal.copy()
    shifted[:,0] += time*3
    start = dict(initial_qpos=nominal[0], initial_qvel=np.zeros(35), initial_time_s=0., plan_id='same_start')
    first,_ = TrajectoryReferenceSolver(kinematics,pose_task_frame='root').solve(nominal,time,boundary=start)
    second,_ = TrajectoryReferenceSolver(kinematics,pose_task_frame='root').solve(shifted,time,boundary=start)
    np.testing.assert_allclose(first[:,7:],second[:,7:],atol=1e-8,rtol=0)
    assert np.max(np.abs(first[:,:3]-second[:,:3])) > 1.


def test_separate_root_qp_matches_the_full_constrained_objective():
    import daqp
    from motionbricks.mb_sonic.trajectory_reference import solve_root_relative_qp

    rng = np.random.default_rng(12)
    controls = 3
    root = (np.arange(controls)[:,None]*35+np.arange(6)).ravel()
    joints = (np.arange(controls)[:,None]*35+np.arange(6,35)).ravel()
    hessian = np.zeros((controls*35,controls*35))
    for selected in (root,joints):
        a = rng.normal(size=(len(selected),len(selected)))
        hessian[np.ix_(selected,selected)] = a.T@a+np.eye(len(selected))
    gradient = rng.normal(size=controls*35)
    matrix = np.eye(controls*35)[joints]
    lo,hi = np.full(len(joints),-.03),np.full(len(joints),.04)
    expected,_,status,_ = daqp.solve(hessian,gradient,matrix,hi,lo,
        primal_tol=1e-10,dual_tol=1e-9,iter_limit=500,eps_prox=0.)
    actual,actual_status,_ = solve_root_relative_qp(hessian,gradient,matrix,hi,lo,controls)
    assert status == actual_status == 1
    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0)


def test_direct_horizon_projection_preserves_the_full_sparse_objective():
    from scipy import sparse
    from motionbricks.mb_sonic.trajectory_reference import project_reference_objective

    rng = np.random.default_rng(19)
    count,width,controls = 9,35,6
    basis = rng.normal(size=(count,controls))
    a = rng.normal(size=(count,width,width))
    blocks = a.swapaxes(-1,-2)@a
    gradients,current,overlap = (rng.normal(size=(count,width)) for _ in range(3))
    metric,weights = rng.uniform(.5,2,width),rng.uniform(0,2,count)
    d = rng.normal(size=(count,count))
    temporal = sparse.csc_matrix(d.T@d)
    projection = sparse.kron(basis,sparse.eye(width),format='csc')
    regularizer = sparse.kron(temporal,sparse.diags(metric))+sparse.diags((weights[:,None]*metric).ravel())
    full_hessian = sparse.block_diag(blocks)+regularizer+sparse.diags(np.tile(.01*metric,count))
    full_gradient = gradients.ravel()+regularizer@current.ravel()-(weights[:,None]*metric*overlap).ravel()
    actual_hessian,actual_gradient = project_reference_objective(basis,blocks,gradients,
        temporal,weights,metric,current,overlap)
    np.testing.assert_allclose(actual_hessian,(projection.T@full_hessian@projection).toarray(),atol=2e-12,rtol=0)
    np.testing.assert_allclose(actual_gradient,projection.T@full_gradient,atol=2e-12,rtol=0)


def test_reference_safety_margin_changes_feasible_plan_not_mechanical_limits(kinematics):
    before = kinematics.model.jnt_range.copy()
    solver = TrajectoryReferenceSolver(kinematics, position_margin_rad=.04)
    nominal = poses()
    nominal[2:, 7+3] = kinematics.limits[3, 0]+.005
    times = np.arange(len(nominal))*.02
    start = dict(initial_qpos=nominal[0], initial_qvel=np.zeros(35), initial_time_s=0., plan_id='measured_start')
    result, report = solver.solve(nominal, times, boundary=start)
    assert (result[:, 7:] >= kinematics.limits[:, 0]+.04-1e-7).all()
    assert (result[:, 7:] <= kinematics.limits[:, 1]-.04+1e-7).all()
    np.testing.assert_array_equal(kinematics.model.jnt_range, before)
    np.testing.assert_array_equal(result[:2], nominal[:2])
    assert report['physical_state_writes'] == 0


def test_root_and_joint_c0_c1_prefix_survives_a_large_new_reference(kinematics):
    old_times = np.arange(51)*.02
    old = poses(len(old_times))
    old[:, 0] += old_times*.3
    old[:, 3:7] = Rotation.from_euler('z', old_times*.2).as_quat()[:, [3, 0, 1, 2]]
    old[:, 7] += old_times*.1
    times = .2+np.arange(31)*.02
    nominal = poses()
    nominal[:, 0] += 1.
    nominal[:, 3:7] = Rotation.from_euler('z', np.ones(31)*.7).as_quat()[:, [3, 0, 1, 2]]
    nominal[:, 7] += .6
    result, report = TrajectoryReferenceSolver(kinematics).solve(nominal, times,
        boundary=boundary(old, old_times, .28))
    fixed = times <= .30+1e-8
    indices = np.rint(times[fixed]/.02).astype(int)
    np.testing.assert_allclose(result[fixed, :3], old[indices, :3], atol=1e-7, rtol=0)
    np.testing.assert_allclose(result[fixed, 7:], old[indices, 7:], atol=1e-7, rtol=0)
    new_rot = Rotation.from_quat(result[fixed][:, [4, 5, 6, 3]])
    old_rot = Rotation.from_quat(old[indices][:, [4, 5, 6, 3]])
    assert np.max((new_rot.inv()*old_rot).magnitude()) < 1e-7
    np.testing.assert_allclose(np.diff(result[fixed, 7:], axis=0), np.diff(old[indices, 7:], axis=0), atol=1e-7)
    assert result[-1, 0] > result[fixed][-1, 0]+.2
    assert report['maximum_velocity_ratio'] <= 1+1e-7
    assert report['committed_prefix_error'] < 1e-7


def test_whole_horizon_suppresses_alternating_joint_targets(kinematics):
    nominal = poses()
    nominal[:, 7] += np.where(np.arange(31)%2, .5, -.5)
    times = np.arange(31)*.02
    start = dict(initial_qpos=poses(1)[0], initial_qvel=np.zeros(35), initial_time_s=0., plan_id='measured_start')
    result, _ = TrajectoryReferenceSolver(kinematics).solve(nominal, times, boundary=start)
    before = np.mean(np.diff(nominal[:, 7:], n=2, axis=0)**2)
    after = np.mean(np.diff(result[:, 7:], n=2, axis=0)**2)
    assert after < before*.2
    assert np.min(result[:, 7:]-kinematics.limits[:, 0]) >= -1e-8
    assert np.min(kinematics.limits[:, 1]-result[:, 7:]) >= -1e-8


def test_default_solve_matches_saved_v49_scalar_quaternion_implementation(kinematics):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    path = root/'out/mb_sonic_e0_e3_20261018/registered_scene_foot_v49/source/trajectory_reference.py'
    spec = importlib.util.spec_from_file_location('motionbricks.mb_sonic._v49_trajectory_reference', path)
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    times = np.arange(31)*.02
    nominal = poses()
    nominal[:, 0] += .2*times
    nominal[:, 3:7] = Rotation.from_euler('z', times*.25).as_quat()[:, [3, 0, 1, 2]]
    nominal[:, 7] += .1*np.sin(times*10)
    start = dict(initial_qpos=poses(1)[0], initial_qvel=np.zeros(35), initial_time_s=0., plan_id='same_start')
    expected, _ = archived.TrajectoryReferenceSolver(kinematics, pose_task_frame='root').solve(
        nominal, times, boundary=start)
    actual, _ = TrajectoryReferenceSolver(kinematics, pose_task_frame='root').solve(nominal, times, boundary=start)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
