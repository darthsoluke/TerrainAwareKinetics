"""CPU-only reference IK contracts on the original G1 XML; no physical rollout."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.constrained_reference import (
    ConstrainedReferenceSolver, ReferenceConstraintSolveError,
)
from motionbricks.mb_sonic.reference import G1_JOINT_NAMES
from motionbricks.mb_sonic.retarget import G1Kinematics
from motionbricks.mb_sonic.sonic_policy import DEFAULT_ANGLES


ROOT = Path(__file__).resolve().parents[2]
ROBOT_XML = ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"


@pytest.fixture(scope="module")
def kinematics():
    return G1Kinematics(ROBOT_XML)


def nominal(kinematics, count=3):
    poses = np.zeros((count, 36), dtype=np.float64)
    poses[:, :3], poses[:, 3], poses[:, 7:] = [.3, -.2, .79], 1., DEFAULT_ANGLES
    return poses


def assert_joint_constraints(kinematics, solved, times):
    assert np.isfinite(solved).all()
    assert np.all(solved[:, 7:] >= kinematics.limits[:, 0]-1e-8)
    assert np.all(solved[:, 7:] <= kinematics.limits[:, 1]+1e-8)
    speed = np.abs(np.diff(solved[:, 7:], axis=0)/np.diff(times)[:, None])
    assert np.all(speed <= kinematics.velocity_limits+1e-7)


def infeasible_waist(kinematics):
    poses = nominal(kinematics)
    column = 7+G1_JOINT_NAMES.index("waist_roll_joint")
    assert kinematics.limits[column-7, 1] < .65
    poses[:, column] = .65
    return poses


def test_solver_uses_original_named_xml_and_urdf_limits(kinematics):
    solver = ConstrainedReferenceSolver(kinematics, iterations=3)
    model = kinematics.model
    joint_ids = [model.joint(name).id for name in G1_JOINT_NAMES]
    np.testing.assert_array_equal(solver.qadr, model.jnt_qposadr[joint_ids])
    np.testing.assert_array_equal(solver.vadr, model.jnt_dofadr[joint_ids])
    np.testing.assert_array_equal(solver.velocity, kinematics.velocity_limits)
    assert np.all(solver.lower >= model.jnt_range[joint_ids, 0])
    assert np.all(solver.upper <= model.jnt_range[joint_ids, 1])
    assert len(solver.qadr) == len(set(solver.qadr)) == 29


def test_feasible_reference_is_bitwise_unchanged_with_moving_root(kinematics):
    poses = nominal(kinematics)
    poses[:, :3] += np.arange(3)[:, None]*[.01, -.02, .003]
    xyzw = Rotation.from_euler("z", [.1, .15, .2]).as_quat()
    poses[:, 3:7] = xyzw[:, [3, 0, 1, 2]]
    times = .18+np.arange(3)*.02
    original, original_times = poses.copy(), times.copy()
    poses.flags.writeable = times.flags.writeable = False
    solved, report = ConstrainedReferenceSolver(kinematics).solve(poses, times)
    np.testing.assert_array_equal(solved, original)
    np.testing.assert_array_equal(poses, original)
    np.testing.assert_array_equal(times, original_times)
    assert not np.shares_memory(solved, poses)
    assert report["qp_solves"] == 0 and report["solved_frames"] == []
    assert report["root_exact"] and report["optimizer_updates"] == 0


def test_velocity_bounds_use_previous_solved_row_not_previous_nominal(kinematics):
    poses = nominal(kinematics)
    column = 7+G1_JOINT_NAMES.index("waist_yaw_joint")
    poses[:, column] = [0., 1., -1.]
    times = np.array([.18, .181, .182])
    original = poses.copy()
    solved, report = ConstrainedReferenceSolver(kinematics).solve(poses, times)
    assert_joint_constraints(kinematics, solved, times)
    np.testing.assert_array_equal(solved[:, :7], poses[:, :7])
    np.testing.assert_array_equal(solved[0], poses[0])
    np.testing.assert_array_equal(poses, original)
    assert report["qp_solves"] > 0 and report["maximum_velocity_ratio"] <= 1.+1e-7
    assert abs(solved[1, column]-poses[1, column]) > .5


def test_private_mjdata_fingers_and_input_are_untouched_by_projection(kinematics, monkeypatch):
    sentinel = mujoco.MjData(kinematics.model)
    sentinel.time, sentinel.qvel[:] = 12.5, np.arange(kinematics.model.nv)
    before = {name: getattr(sentinel, name).copy() for name in
              ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")}
    solver = ConstrainedReferenceSolver(kinematics)
    assert solver.configuration.data is not sentinel

    def forbidden_step(*args, **kwargs):
        pytest.fail("Reference projection must never advance physical simulation")

    monkeypatch.setattr(mujoco, "mj_step", forbidden_step)
    poses = infeasible_waist(kinematics)
    poses[:, :3] += np.arange(3)[:, None]*[.01, .02, -.001]
    original = poses.copy()
    poses.flags.writeable = False
    solved, report = solver.solve(poses, .18+np.arange(3)*.02)
    np.testing.assert_array_equal(poses, original)
    np.testing.assert_array_equal(solved[:, :7], original[:, :7])
    assert not np.shares_memory(solved, poses)
    assert sentinel.time == 12.5 and solver.configuration.data.time == 0.
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(sentinel, name), value)
    fingers = np.setdiff1d(np.arange(7, kinematics.model.nq), solver.qadr)
    assert len(fingers) == 14
    np.testing.assert_array_equal(solver.configuration.q[fingers], kinematics.model.qpos0[fingers])
    assert report["physical_state_writes"] == 0 and report["root_exact"]


def test_infeasible_waist_fits_frame_objective_instead_of_component_clipping(kinematics):
    solver = ConstrainedReferenceSolver(kinematics)
    poses, times = infeasible_waist(kinematics), .18+np.arange(3)*.02
    solved, report = solver.solve(poses, times)
    assert_joint_constraints(kinematics, solved, times)
    clipped = poses[0].copy()
    clipped[7:] = np.clip(clipped[7:], solver.lower, solver.upper)

    def objective(pose):
        full = kinematics.model.qpos0.copy()
        full[:7], full[solver.qadr] = pose[:7], pose[7:]
        solver.configuration.update(full)
        return sum(float(np.dot(task.cost*task.compute_error(solver.configuration),
                                task.cost*task.compute_error(solver.configuration)))
                   for task in solver.tasks)

    clipped_cost, fitted_cost = objective(clipped), objective(solved[0])
    assert fitted_cost < clipped_cost-1e-8
    assert not np.allclose(solved[0, 7:], clipped[7:], atol=1e-7, rtol=0)
    nominal_positions, nominal_rotations = kinematics.fk(poses)
    solved_positions, solved_rotations = kinematics.fk(solved)
    torso = kinematics.body_names.index("torso_link")
    residual = Rotation.from_matrix(nominal_rotations[:, torso].swapaxes(-1, -2)
                                   @ solved_rotations[:, torso]).magnitude()
    assert (residual > 1e-5).all()
    assert np.max(np.abs(solved_positions-nominal_positions)) > 1e-6
    assert report["solved_frames"] == [0, 1, 2] and report["qp_solves"] == 9


def test_identical_packets_are_deterministic_without_hidden_previous_packet_state(kinematics):
    poses, times = infeasible_waist(kinematics), .18+np.arange(3)*.02
    solver = ConstrainedReferenceSolver(kinematics)
    first, _ = solver.solve(poses, times)
    solver.solve(nominal(kinematics), times+5.)
    second, _ = solver.solve(poses, times)
    independent, _ = ConstrainedReferenceSolver(kinematics).solve(poses, times)
    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(independent, first)


@pytest.mark.parametrize("times", [
    [.18, .18, .20], [.18, .17, .20], [.18, np.nan, .22],
    [.18, .20, np.inf], [.18, .20], [[.18, .20, .22]],
])
def test_invalid_timestamps_are_rejected_before_qp(kinematics, monkeypatch, times):
    import motionbricks.mb_sonic.constrained_reference as module
    monkeypatch.setattr(module.qpsolvers, "solve_problem",
                        lambda *args, **kwargs: pytest.fail("Invalid timestamps reached QP"))
    with pytest.raises(ValueError, match="increasing sample times"):
        ConstrainedReferenceSolver(kinematics).solve(nominal(kinematics), np.array(times))


def test_failed_qp_raises_without_clipping_or_returning_a_partial_packet(kinematics, monkeypatch):
    import motionbricks.mb_sonic.constrained_reference as module
    monkeypatch.setattr(module.qpsolvers, "solve_problem",
                        lambda *args, **kwargs: SimpleNamespace(found=False, x=None))
    poses = infeasible_waist(kinematics)
    original = poses.copy()
    with pytest.raises(ReferenceConstraintSolveError, match="QP failed"):
        ConstrainedReferenceSolver(kinematics).solve(poses, .18+np.arange(3)*.02)
    np.testing.assert_array_equal(poses, original)


def test_every_first_five_adoption_grids_respects_previous_packet_velocity_without_prefix_copy(kinematics):
    times = .18+np.arange(8)*.02
    poses = nominal(kinematics, len(times))
    column = 7+G1_JOINT_NAMES.index("waist_yaw_joint")
    poses[:, column] = 1.7
    poses[:, :3] += np.arange(len(times))[:, None]*[.01, -.002, .001]
    old_times = times-.02
    old_joints = nominal(kinematics, len(times))[:, 7:].copy()
    boundary = dict(timestamps_s=old_times, joint_positions=old_joints,
                    adoption_until_s=float(times[4]), plan_id="previous_accepted_reference")
    original, old_original, old_time_original = poses.copy(), old_joints.copy(), old_times.copy()
    poses.flags.writeable = old_joints.flags.writeable = old_times.flags.writeable = False
    solved, report = ConstrainedReferenceSolver(kinematics).solve(poses, times, boundary=boundary)
    assert_joint_constraints(kinematics, solved, times)
    np.testing.assert_array_equal(solved[:, :7], original[:, :7])
    np.testing.assert_array_equal(poses, original)
    np.testing.assert_array_equal(old_joints, old_original)
    np.testing.assert_array_equal(old_times, old_time_original)
    for frame in range(5):
        old_index = np.flatnonzero(np.isclose(old_times, times[frame]-.02, atol=1e-12, rtol=0)).item()
        speed = np.abs(solved[frame, 7:]-old_joints[old_index])/.02
        assert np.all(speed <= kinematics.velocity_limits+1e-7)
        assert np.max(np.abs(solved[frame, 7:]-old_joints[old_index])) > 1e-3
    assert report["boundary_frame_count"] == 5
    assert report["boundary_plan_id"] == boundary["plan_id"]


@pytest.mark.parametrize("failure", ["disjoint", "missing_grid"])
def test_boundary_disjoint_or_missing_exact_previous_grid_raises(kinematics, monkeypatch, failure):
    import motionbricks.mb_sonic.constrained_reference as module
    times = .18+np.arange(3)*.02
    poses = nominal(kinematics)
    old_times, old_joints = times-.02, poses[:, 7:].copy()
    if failure == "disjoint":
        old_joints[:, G1_JOINT_NAMES.index("waist_yaw_joint")] = 20.
    else:
        old_times += .001
    boundary = dict(timestamps_s=old_times, joint_positions=old_joints,
                    adoption_until_s=float(times[2]), plan_id="unusable_previous_reference")
    original = poses.copy()
    monkeypatch.setattr(module.qpsolvers, "solve_problem",
                        lambda *args, **kwargs: pytest.fail("Impossible boundary must be rejected before QP"))
    message = "incompatible boundary" if failure == "disjoint" else "does not cover"
    with pytest.raises(ReferenceConstraintSolveError, match=message):
        ConstrainedReferenceSolver(kinematics).solve(poses, times, boundary=boundary)
    np.testing.assert_array_equal(poses, original)


@pytest.mark.parametrize("seed", [19, 137, 104729])
def test_cached_cartesian_frame_jacobians_match_joint_angle_finite_differences(kinematics, seed):
    solver = ConstrainedReferenceSolver(kinematics)
    rng = np.random.default_rng(seed)
    current = kinematics.model.qpos0.copy()
    current[:3] = rng.uniform(-1., 1., 3)
    quaternion = rng.normal(size=4)
    current[3:7] = quaternion/np.linalg.norm(quaternion)
    width = kinematics.limits[:, 1]-kinematics.limits[:, 0]
    current[solver.qadr] = kinematics.limits[:, 0]+rng.uniform(.15, .85, 29)*width
    target = current.copy()
    target[solver.qadr] = kinematics.limits[:, 0]+rng.uniform(.15, .85, 29)*width
    configuration = solver.configuration
    configuration.update(target)
    for task in solver.frames:
        task.set_target_from_configuration(configuration)
    configuration.update(current)
    epsilon = 1e-6
    for task in solver.frames:
        analytic = task.compute_jacobian(configuration)[:, solver.vadr].copy()
        finite_difference = np.empty((6, 29))
        for column, qadr in enumerate(solver.qadr):
            plus, minus = current.copy(), current.copy()
            plus[qadr] += epsilon
            minus[qadr] -= epsilon
            configuration.update(plus)
            positive = task.compute_error(configuration).copy()
            configuration.update(minus)
            negative = task.compute_error(configuration).copy()
            finite_difference[:, column] = (positive-negative)/(2*epsilon)
        configuration.update(current)
        np.testing.assert_allclose(analytic, finite_difference, atol=1e-5, rtol=0,
            err_msg=f"body={kinematics.body_names[task.body]}, seed={seed}")
    np.testing.assert_array_equal(configuration.q[:7], current[:7])


def test_cached_cartesian_task_zero_target_cost_weights_and_product_space_identity(kinematics):
    solver = ConstrainedReferenceSolver(kinematics)
    full = kinematics.model.qpos0.copy()
    full[:7], full[solver.qadr] = nominal(kinematics)[0, :7], DEFAULT_ANGLES
    full[3:7] = np.array([1., -2., 3., 4.])/np.sqrt(30.)
    solver.configuration.update(full)
    costs = [(0., 5.), (50., 5.), (50., 5.), (10., 2.), (10., 2.)]
    assert len(solver.frames) == len(costs)
    for task, (position_cost, orientation_cost) in zip(solver.frames, costs):
        task.set_target_from_configuration(solver.configuration)
        np.testing.assert_array_equal(task.cost, [position_cost]*3+[orientation_cost]*3)
        np.testing.assert_allclose(task.compute_error(solver.configuration), 0., atol=1e-14, rtol=0)
        hessian, gradient = task.compute_qp_objective(solver.configuration)
        jacobian = task.cost[:, None]*task.compute_jacobian(solver.configuration)
        np.testing.assert_allclose(hessian, jacobian.T@jacobian, atol=1e-12, rtol=0)
        np.testing.assert_allclose(gradient, 0., atol=1e-12, rtol=0)
    perturbed = full.copy()
    perturbed[solver.qadr] += .01
    solver.configuration.update(perturbed)
    for task in solver.frames:
        error = task.cost*task.compute_error(solver.configuration)
        jacobian = task.cost[:, None]*task.compute_jacobian(solver.configuration)
        hessian, gradient = task.compute_qp_objective(solver.configuration)
        np.testing.assert_allclose(hessian, jacobian.T@jacobian, atol=1e-12, rtol=0)
        np.testing.assert_allclose(gradient, jacobian.T@error, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(task.compute_error(solver.configuration)[:3],
                                      solver.configuration.data.xpos[task.body]-task.position)
    assert solver.identity["task_error"] == "world_position_and_SO3_quaternion_log"


@pytest.mark.parametrize("times", [[.18, .20, .23], [.18, .19, .20]])
def test_adoption_boundary_rejects_nonuniform_or_non50hz_grid_before_qp(kinematics, monkeypatch, times):
    import motionbricks.mb_sonic.constrained_reference as module
    monkeypatch.setattr(module.qpsolvers, "solve_problem",
                        lambda *args, **kwargs: pytest.fail("Invalid adoption grid reached QP"))
    poses = nominal(kinematics)
    boundary = dict(timestamps_s=np.array([.16, .18, .20]), joint_positions=poses[:, 7:].copy(),
                    adoption_until_s=.22, plan_id="uniform_old_reference")
    with pytest.raises(ValueError, match="uniform 50Hz"):
        ConstrainedReferenceSolver(kinematics).solve(poses, np.array(times), boundary=boundary)
