"""Whole-horizon reference IK with an explicit committed execution prefix."""

from time import perf_counter_ns

import mujoco
import numpy as np
import daqp
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

from .constrained_reference import CartesianFrameTask, RootRelativeFrameTask, ReferenceConstraintSolveError


def right_jacobian(vector):
    x, y, z = vector
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = np.linalg.norm(vector)
    a = (1-np.cos(angle))/angle**2 if angle > 1e-5 else .5-angle**2/24
    b = (angle-np.sin(angle))/angle**3 if angle > 1e-5 else 1/6-angle**2/120
    return np.eye(3)-a*cross+b*(cross@cross)


def solve_root_relative_qp(hessian, gradient, matrix, hi, lo, controls):
    """Separate unconstrained Root coordinates from the joint-only QP."""
    root = (np.arange(controls)[:, None]*35+np.arange(6)).ravel()
    joints = (np.arange(controls)[:, None]*35+np.arange(6,35)).ravel()
    if np.any(hessian[np.ix_(root,joints)]) or np.any(matrix[:,root]):
        raise ValueError('Root-relative reference objective must have no Root/joint coupling')
    solution = np.empty(len(gradient))
    solution[root] = np.linalg.solve(hessian[np.ix_(root,root)],-gradient[root])
    value, _, status, info = daqp.solve(hessian[np.ix_(joints,joints)],gradient[joints],
        matrix[:,joints],hi,lo,primal_tol=1e-10,dual_tol=1e-9,iter_limit=500,eps_prox=0.)
    solution[joints] = value
    return solution,status,info


def project_reference_objective(basis, hessians, gradients, temporal, weights, metric, current, overlap):
    """Project frame blocks directly, without constructing a horizon-sized Hessian."""
    temporal_gradient = temporal@current+weights[:,None]*(current-overlap)
    gradient = (basis.T@(gradients+temporal_gradient*metric)).ravel()
    controls = basis.T@(temporal@basis+weights[:,None]*basis+.01*basis)
    hessian = np.einsum('fa,fij,fb->aibj',basis,hessians,basis,optimize=True).reshape(
        basis.shape[1]*len(metric),-1)
    hessian += np.kron(controls,np.diag(metric))
    return hessian,gradient


class TrajectoryReferenceSolver:
    """Optimize reference Root and joints, without advancing a physical MjData."""

    def __init__(self, kinematics, *, iterations=2, acceleration_time_s=.04,
                 jerk_time_s=.025, overlap_time_s=.12, overlap_weight=4., position_margin_rad=1e-7,
                 foot_ground=None, pose_task_frame='world'):
        if iterations < 1 or min(acceleration_time_s, jerk_time_s, overlap_time_s, overlap_weight) <= 0:
            raise ValueError('Positive trajectory iteration budget and time/overlap weights required')
        self.kinematics, self.model, self.iterations = kinematics, kinematics.model, iterations
        self.foot_ground = foot_ground
        if pose_task_frame not in ('world','root'):
            raise ValueError('Pose tasks require an explicit world or Root coordinate frame')
        self.pose_task_frame = pose_task_frame
        self.data = mujoco.MjData(self.model)
        self.configuration = type('ReferenceConfiguration', (), {})()
        self.configuration.model, self.configuration.data = self.model, self.data
        ids = np.array(list(kinematics.joint_ids.values()))
        self.qadr, self.vadr = self.model.jnt_qposadr[ids], self.model.jnt_dofadr[ids]
        self.selected = np.r_[np.arange(6), self.vadr]
        if (not np.isfinite(position_margin_rad) or position_margin_rad < 1e-7
                or np.any(np.diff(kinematics.limits, axis=1)[:, 0] <= 2*position_margin_rad)):
            raise ValueError('Reference safety margin must remain inside every original joint range')
        self.lower, self.upper = kinematics.limits[:, 0]+position_margin_rad, kinematics.limits[:, 1]-position_margin_rad
        self.velocity = kinematics.velocity_limits.copy()
        self.prior = np.r_[[20.]*3, [5.]*3, [.5]*29]**2
        self.acceleration_time_s, self.jerk_time_s = acceleration_time_s, jerk_time_s
        self.overlap_time_s, self.overlap_weight = overlap_time_s, overlap_weight
        specifications = [('torso_link', 0., 5.), ('left_ankle_roll_link', 50., 5.),
            ('right_ankle_roll_link', 50., 5.), ('left_wrist_yaw_link', 10., 2.), ('right_wrist_yaw_link', 10., 2.)]
        task_type = RootRelativeFrameTask if pose_task_frame == 'root' else CartesianFrameTask
        self.frames = [task_type(self.model, name, position, rotation)
                       for name, position, rotation in specifications]
        self.identity = dict(schema='motionbricks.mb_sonic.trajectory_reference.v1',
            solver='whole_horizon_cartesian_Gauss_Newton_DAQP', iterations=iterations,
            acceleration_time_s=acceleration_time_s, jerk_time_s=jerk_time_s,
            overlap_time_s=overlap_time_s, overlap_weight=overlap_weight,
            levenberg_metric_damping=.01,
            trajectory_basis='cubic_B_spline_six_free_controls;fixed_prefix_and_zero_initial_correction_slope',
            root_dofs='reference_XYZ_SO3_optimized;actual_state_never_written',
            prefix='existing_absolute_time_reference_through_adoption_bound_plus_one_control_step',
            position_limits='original_XML_with_explicit_interior_margin', position_margin_rad=position_margin_rad,
            velocity_limits='original_URDF_between_reference_samples',
            pose_task_frame=pose_task_frame,
            physical_state_writes=0, optimizer_updates=0)

    def _update(self, pose):
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:7], self.data.qpos[self.qadr] = pose[:7], pose[7:]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    @staticmethod
    def _coordinates(poses, basis):
        return np.c_[poses[:, :3], (basis.inv()*Rotation.from_quat(poses[:, [4, 5, 6, 3]])).as_rotvec(), poses[:, 7:]]

    @staticmethod
    def _poses(coordinates, basis):
        quaternion = (basis*Rotation.from_rotvec(coordinates[:, 3:6])).as_quat()[:, [3, 0, 1, 2]]
        return np.c_[coordinates[:, :3], quaternion, coordinates[:, 6:]]

    def solve(self, qpos, timestamps_s, *, boundary=None):
        started = perf_counter_ns()
        nominal, times = np.asarray(qpos, float), np.asarray(timestamps_s, float)
        if (nominal.ndim != 2 or nominal.shape[1] != 36 or len(nominal) < 4
                or times.shape != (len(nominal),) or not np.isfinite(nominal).all()
                or not np.allclose(np.diff(times), .02, atol=1e-9, rtol=0)):
            raise ValueError('Trajectory constraints require finite [N,36] reference poses on the 50Hz grid')
        count, width = len(times), 35
        dt = .02
        old, fixed = None, np.zeros(count, bool)
        if boundary is not None:
            if 'initial_qpos' in boundary:
                pose = np.array(boundary['initial_qpos'], dtype=float, copy=True)
                velocity = np.asarray(boundary['initial_qvel'], float)
                full = self.model.qpos0.copy()
                full[:7], full[self.qadr] = pose[:7], pose[7:]
                v = np.zeros(self.model.nv)
                v[self.selected] = velocity
                old = []
                for time in times[:2]:
                    current = full.copy()
                    mujoco.mj_integratePos(self.model, current, v, float(time-boundary['initial_time_s']))
                    old.append(np.r_[current[:7], current[self.qadr]])
                old = np.array(old)
                fixed[:2] = True
            else:
                old_times = np.asarray(boundary['timestamps_s'])
                covered = (times >= old_times[0]-1e-9) & (times <= old_times[-1]+1e-9)
                indices = np.searchsorted(old_times, times[covered]-1e-9)
                if not np.allclose(old_times[indices], times[covered], atol=1e-8, rtol=0):
                    raise ReferenceConstraintSolveError('Committed trajectory requires exact previous sample times')
                old = nominal.copy()
                old[covered, :7] = np.asarray(boundary['root_pose'])[indices]
                old[covered, 7:] = np.asarray(boundary['joint_positions'])[indices]
                fixed = times <= boundary['adoption_until_s']+dt+1e-8
                if not covered[fixed].all():
                    raise ReferenceConstraintSolveError('Previous trajectory does not cover its committed prefix')
        else:
            raise ValueError('Trajectory constraints require an explicit startup or previous-reference boundary')
        basis = Rotation.from_quat(old[0, [4, 5, 6, 3]])
        target = self._coordinates(nominal, basis)
        overlap = target.copy()
        if len(old) == count:
            overlap = self._coordinates(old, basis)
            age = np.maximum(0., times-boundary['adoption_until_s'])
            overlap_weights = self.overlap_weight*np.exp(-(age/self.overlap_time_s)**2)*covered
        else:
            overlap[:2] = self._coordinates(old, basis)
            overlap_weights = np.zeros(count)
        last_fixed = int(np.flatnonzero(fixed)[-1])
        free_frames = np.flatnonzero(~fixed)
        controls = min(6, len(free_frames))
        if controls < 2:
            raise ReferenceConstraintSolveError('Trajectory requires at least two editable future samples')
        base = overlap.copy()
        initial_velocity = (overlap[last_fixed]-overlap[last_fixed-1])/dt
        base[free_frames] = overlap[last_fixed]+(times[free_frames]-times[last_fixed])[:, None]*initial_velocity
        knots = np.r_[np.zeros(4), np.linspace(0., 1., controls)[1:-1], np.ones(4)]
        parameter = (times[free_frames]-times[last_fixed])/(times[-1]-times[last_fixed])
        basis_values = np.zeros((count, controls))
        basis_values[free_frames] = BSpline.design_matrix(parameter, knots, 3).toarray()[:, 2:]
        coefficients = np.linalg.lstsq(basis_values[free_frames], target[free_frames]-base[free_frames], rcond=None)[0]
        current = base+basis_values@coefficients
        projection = sparse.kron(sparse.csc_matrix(basis_values), sparse.eye(width), format='csc')
        lower = np.tile(np.r_[[-np.inf]*6, self.lower], (count, 1))
        upper = np.tile(np.r_[[np.inf]*6, self.upper], (count, 1))
        lower[fixed] = upper[fixed] = overlap[fixed]
        delta1 = sparse.diags([-np.ones(count-1), np.ones(count-1)], [0, 1], shape=(count-1, count), format='csc')
        joint_select = sparse.csc_matrix((np.ones(29), (np.arange(29), np.arange(6, 35))), shape=(29, 35))
        speed_constraint = sparse.kron(delta1, joint_select, format='csc')
        speed_bound = np.tile(self.velocity*dt, count-1)
        constraints = sparse.vstack([sparse.eye(count*width, format='csc'), speed_constraint], format='csc')
        delta2 = sparse.diags([np.ones(count-2), -2*np.ones(count-2), np.ones(count-2)],
                             [0, 1, 2], shape=(count-2, count), format='csc')
        delta3 = sparse.diags([-np.ones(count-3), 3*np.ones(count-3), -3*np.ones(count-3), np.ones(count-3)],
                             [0, 1, 2, 3], shape=(count-3, count), format='csc')
        temporal = ((self.acceleration_time_s/dt)**4*(delta2.T@delta2)
                    +(self.jerk_time_s/dt)**6*(delta3.T@delta3))
        positions, rotations = self.kinematics.fk(nominal)
        if self.pose_task_frame == 'root':
            root = self.kinematics.pelvis
            root_inverse = rotations[:, root].swapaxes(-1,-2)
            targets = [(np.einsum('nij,nj->ni',root_inverse,positions[:,task.body]-positions[:,root]),
                root_inverse@rotations[:,task.body]) for task in self.frames]
        else:
            targets = [(positions[:, task.body].copy(), rotations[:, task.body].copy()) for task in self.frames]
        targets = [(point, Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]])
                   for point, rotation in targets]
        metric = None
        support_weights = None if self.foot_ground is None else self.foot_ground.support_weights(times)
        qp_setups, qp_solves = [], []
        for _ in range(self.iterations):
            poses = self._poses(current, basis)
            hessians, gradients, ground_rows, ground_lower, support_records = [], [], [], [], []
            for frame, pose in enumerate(poses):
                self._update(pose)
                transform = np.eye(width)
                transform[3:6, 3:6] = right_jacobian(current[frame, 3:6])
                if self.foot_ground is not None:
                    rows, distances = self.foot_ground.linearize(pose, self.qadr, self.selected)
                    if fixed[frame] and np.any(distances < self.foot_ground.clearance_m-1e-5):
                        raise ReferenceConstraintSolveError('Committed reference prefix violates foot-ground clearance')
                    if not fixed[frame] and len(rows):
                        ground_rows.append(sparse.csc_matrix(np.kron(basis_values[frame][None], rows@transform)))
                        ground_lower.extend(self.foot_ground.clearance_m-distances)
                hessian = np.diag(self.prior)
                gradient = self.prior*(current[frame]-target[frame])
                if support_weights is not None:
                    points, jac, normals, gap = self.foot_ground.support_linearize(self.selected)
                    jac = jac@transform
                    support_records.append((points, jac))
                    height_jacobian = np.einsum('ki,kij->kj', normals, jac)
                    weight = 200.*support_weights[frame]
                    weighted_jacobian = weight[:, None]*height_jacobian
                    hessian += weighted_jacobian.T@weighted_jacobian
                    gradient += weighted_jacobian.T@(weight*gap)
                for task, (point, quaternion) in zip(self.frames, targets):
                    task.position[:] = point[frame]
                    task.quaternion[:] = quaternion[frame]
                    error = task.cost*task.compute_error(self.configuration)
                    jacobian = task.cost[:, None]*(task.compute_jacobian(self.configuration)[:, self.selected]@transform)
                    hessian += jacobian.T@jacobian
                    gradient += jacobian.T@error
                hessians.append(hessian)
                gradients.append(gradient)
            if metric is None:
                metric = np.maximum(np.mean([np.diag(h) for h in hessians], axis=0), self.prior)
            vector = current.ravel()
            lower_step = np.r_[lower.ravel()-vector, -speed_bound-speed_constraint@vector]
            upper_step = np.r_[upper.ravel()-vector, speed_bound-speed_constraint@vector]
            scales = np.tile(1/np.sqrt(metric),controls)
            selected_constraints = constraints@projection
            active_rows = ((selected_constraints.getnnz(axis=1)>0)
                           & (np.isfinite(lower_step) | np.isfinite(upper_step)))
            reduced_hessian,reduced_gradient = project_reference_objective(basis_values,np.asarray(hessians),
                np.asarray(gradients),temporal,overlap_weights,metric,current,overlap)
            if support_weights is not None:
                velocity_jacobians, velocity_errors = [], []
                for frame in range(count-1):
                    point0, jac0 = support_records[frame]
                    point1, jac1 = support_records[frame+1]
                    weight = 3.*np.minimum(support_weights[frame], support_weights[frame+1])/dt
                    for contact in range(4):
                        velocity_jacobians.append(weight[contact]*(
                            np.kron(basis_values[frame+1][None], jac1[contact])
                            - np.kron(basis_values[frame][None], jac0[contact])))
                        velocity_errors.extend(weight[contact]*(point1[contact]-point0[contact]))
                velocity_jacobian = np.vstack(velocity_jacobians)
                reduced_hessian += velocity_jacobian.T@velocity_jacobian
                reduced_gradient += velocity_jacobian.T@np.asarray(velocity_errors)
            normalized_hessian = scales[:,None]*reduced_hessian*scales[None,:]
            objective_scale = float(normalized_hessian.diagonal().max())
            matrix = selected_constraints[active_rows].toarray()*scales
            lo, hi = lower_step[active_rows], upper_step[active_rows]
            if ground_rows:
                matrix = np.vstack((matrix,sparse.vstack(ground_rows).toarray()*scales))
                lo = np.r_[lo, ground_lower]
                hi = np.r_[hi, np.full(len(ground_lower), np.inf)]
            objective_hessian = normalized_hessian/objective_scale
            objective_gradient = scales*reduced_gradient/objective_scale
            if self.pose_task_frame == 'root' and self.foot_ground is None:
                step_solution,status,info = solve_root_relative_qp(objective_hessian,
                    objective_gradient,matrix,hi,lo,controls)
            else:
                step_solution, _, status, info = daqp.solve(objective_hessian,
                    objective_gradient, matrix, hi, lo,
                    primal_tol=1e-10, dual_tol=1e-9, iter_limit=500, eps_prox=0.)
            if status != 1 or not np.isfinite(step_solution).all():
                raise ReferenceConstraintSolveError(f'Whole-trajectory reference QP failed: status={status}')
            qp_setups.append(info['setup_time'])
            qp_solves.append(info['solve_time'])
            step = projection@(scales*step_solution)
            current += step.reshape(count, width)
        result = self._poses(current, basis)
        prefix_error = float(np.max(np.abs(current[fixed]-overlap[fixed])))
        if prefix_error > 1e-7:
            raise ReferenceConstraintSolveError('Whole-trajectory reference changed its committed prefix')
        speeds = np.abs(np.diff(result[:, 7:], axis=0)/dt)
        position_margin = np.minimum(result[:, 7:]-self.kinematics.limits[:, 0],
                                     self.kinematics.limits[:, 1]-result[:, 7:]).min()
        if position_margin < -1e-8 or np.max(speeds/self.velocity) > 1+1e-7:
            raise ReferenceConstraintSolveError('Whole-trajectory reference violated original joint bounds')
        ground_report = (None if self.foot_ground is None else
            self.foot_ground.require_feasible(result, self.qadr, self.selected))
        return result, dict(self.identity, sample_count=count,
            foot_ground=ground_report,
            solved_frames=np.flatnonzero(np.max(np.abs(result[:, 7:]-nominal[:, 7:]), axis=1)>1e-8).tolist(),
            qp_solves=self.iterations, root_exact=False,
            qp_setup_ms=1000*sum(qp_setups), qp_solve_ms=1000*sum(qp_solves),
            committed_frames=int(fixed.sum()), committed_prefix_error=prefix_error,
            boundary_plan_id=boundary['plan_id'], boundary_frame_count=int(fixed.sum()),
            minimum_position_margin_rad=float(position_margin), maximum_velocity_ratio=float(np.max(speeds/self.velocity)),
            maximum_joint_change_rad=float(np.abs(result[:, 7:]-nominal[:, 7:]).max()),
            maximum_reference_root_change_m=float(np.linalg.norm(result[:, :3]-nominal[:, :3], axis=-1).max()),
            wall_ns=perf_counter_ns()-started)
