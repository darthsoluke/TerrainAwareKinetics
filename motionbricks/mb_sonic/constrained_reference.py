"""Joint-constrained reference IK on private kinematics, never physical state."""

from time import perf_counter_ns

import mink
import mujoco
import numpy as np
import qpsolvers


class ReferenceConstraintSolveError(ValueError):
    pass


class CartesianFrameTask:
    """Cached world-position and SO(3) task using MuJoCo's exact derivatives."""

    def __init__(self, model, name, position_cost, orientation_cost):
        self.body = model.body(name).id
        self.cost = np.array([position_cost]*3+[orientation_cost]*3, dtype=float)
        self.position = np.empty(3)
        self.quaternion = np.empty(4)
        self._jp = np.empty((3, model.nv))
        self._jr = np.empty((3, model.nv))
        self._dq = np.empty((3, 3))

    def set_target_from_configuration(self, configuration):
        self.position[:] = configuration.data.xpos[self.body]
        self.quaternion[:] = configuration.data.xquat[self.body]

    def compute_error(self, configuration):
        data = configuration.data
        error = np.empty(6)
        error[:3] = data.xpos[self.body]-self.position
        mujoco.mju_subQuat(error[3:], data.xquat[self.body], self.quaternion)
        return error

    def compute_jacobian(self, configuration):
        data = configuration.data
        mujoco.mj_jacBody(configuration.model, data, self._jp, self._jr, self.body)
        mujoco.mjd_subQuat(data.xquat[self.body], self.quaternion, self._dq, None)
        rotation = data.xmat[self.body].reshape(3, 3)
        return np.vstack((self._jp, self._dq @ rotation.T @ self._jr))

    def compute_qp_objective(self, configuration):
        error = self.cost*self.compute_error(configuration)
        jacobian = self.cost[:, None]*self.compute_jacobian(configuration)
        return jacobian.T @ jacobian, jacobian.T @ error


class RootRelativeFrameTask(CartesianFrameTask):
    """Body pose in the pelvis frame, independent of global reference registration."""

    def __init__(self, model, name, position_cost, orientation_cost):
        super().__init__(model, name, position_cost, orientation_cost)
        self.root = model.body('pelvis').id
        self._relative_quaternion = np.empty(4)

    def _relative(self, data):
        root_rotation = data.xmat[self.root].reshape(3,3)
        body_rotation = data.xmat[self.body].reshape(3,3)
        relative_rotation = root_rotation.T@body_rotation
        mujoco.mju_mat2Quat(self._relative_quaternion, relative_rotation.ravel())
        return root_rotation.T@(data.xpos[self.body]-data.xpos[self.root]), root_rotation, body_rotation

    def set_target_from_configuration(self, configuration):
        self.position[:], _, _ = self._relative(configuration.data)
        self.quaternion[:] = self._relative_quaternion

    def compute_error(self, configuration):
        position, _, _ = self._relative(configuration.data)
        error = np.empty(6)
        error[:3] = position-self.position
        mujoco.mju_subQuat(error[3:], self._relative_quaternion, self.quaternion)
        return error

    def compute_jacobian(self, configuration):
        data = configuration.data
        _, root_rotation, body_rotation = self._relative(data)
        mujoco.mj_jacBody(configuration.model, data, self._jp, self._jr, self.body)
        self._jp[:, :6], self._jr[:, :6] = 0., 0.
        mujoco.mjd_subQuat(self._relative_quaternion, self.quaternion, self._dq, None)
        return np.vstack((root_rotation.T@self._jp, self._dq@body_rotation.T@self._jr))


class ConstrainedReferenceSolver:
    """Preserve the root and fit G1 frame targets subject to original joint limits."""

    def __init__(self, kinematics, *, iterations=3):
        if type(iterations) is not int or iterations < 1:
            raise ValueError("Reference IK needs a positive fixed iteration budget")
        self.kinematics, self.iterations = kinematics, iterations
        self.model = kinematics.model
        joints = np.array(list(kinematics.joint_ids.values()))
        self.qadr = self.model.jnt_qposadr[joints].copy()
        self.vadr = self.model.jnt_dofadr[joints].copy()
        self.configuration = mink.Configuration(self.model)
        self.posture = mink.PostureTask(self.model, cost=.15)
        specifications = (
            ("torso_link", 0., 5.),
            ("left_ankle_roll_link", 50., 5.),
            ("right_ankle_roll_link", 50., 5.),
            ("left_wrist_yaw_link", 10., 2.),
            ("right_wrist_yaw_link", 10., 2.),
        )
        self.frames = [CartesianFrameTask(self.model, name, position, rotation)
                       for name, position, rotation in specifications]
        self.tasks = [self.posture, *self.frames]
        self.lower = kinematics.limits[:, 0].copy()+1e-7
        self.upper = kinematics.limits[:, 1].copy()-1e-7
        self.velocity = kinematics.velocity_limits.copy()
        self.identity = dict(schema="motionbricks.mb_sonic.constrained_reference.v1",
            solver="mujoco_cartesian_tasks_mink_build_ik_daqp", iterations=iterations,
            task_error="world_position_and_SO3_quaternion_log",
            posture_cost=.15, frame_tasks=[dict(body=n, position_cost=p, orientation_cost=r)
                                         for n, p, r in specifications],
            root_dofs="excluded_from_QP", finger_dofs="excluded_from_QP",
            position_limits="original_XML_inside_margin_1e-7_rad",
            velocity_limits="original_URDF_between_reference_samples",
            temporal_feasibility="backward_reachable_boxes;1e-8_rad_internal_future_margin",
            physical_state_writes=0, optimizer_updates=0)

    def _full(self, pose):
        full = self.model.qpos0.copy()
        full[:7], full[self.qadr] = pose[:7], pose[7:]
        return full

    def solve(self, qpos, timestamps_s, *, boundary=None):
        started_ns = perf_counter_ns()
        nominal = np.asarray(qpos, dtype=np.float64)
        times = np.asarray(timestamps_s, dtype=np.float64)
        if (nominal.ndim != 2 or nominal.shape[1] != 36 or len(nominal) < 2
                or times.shape != (len(nominal),) or not np.isfinite(nominal).all()
                or not np.isfinite(times).all() or not (np.diff(times) > 0).all()):
            raise ValueError("Reference IK requires finite [N,36] poses and increasing sample times")
        result = nominal.copy()
        boundary_indices = {}
        if boundary is not None:
            if not np.allclose(np.diff(times), 1/50, atol=1e-9, rtol=0):
                raise ValueError("Reference adoption boundaries require a uniform 50Hz sample grid")
            old_times = np.asarray(boundary["timestamps_s"], dtype=float)
            old_joints = np.asarray(boundary["joint_positions"], dtype=float)
            if (old_times.ndim != 1 or len(old_times) < 2 or old_joints.shape != (len(old_times), 29)
                    or not np.isfinite(old_joints).all() or not np.isfinite(old_times).all()
                    or not (np.diff(old_times) > 0).all()
                    or not np.isfinite(boundary["adoption_until_s"])):
                raise ValueError("Reference IK boundary must contain an existing finite joint reference")
            step = 1/50
            for frame in np.flatnonzero(times <= boundary["adoption_until_s"]+1e-8):
                requested = times[frame]-step
                index = np.searchsorted(old_times, requested-1e-9)
                if index == len(old_times) or not np.isclose(old_times[index], requested, atol=1e-8, rtol=0):
                    raise ReferenceConstraintSolveError("Previous reference does not cover the adoption boundary")
                boundary_indices[int(frame)] = old_joints[index]
        lower_bands = np.broadcast_to(self.lower, (len(times), 29)).copy()
        upper_bands = np.broadcast_to(self.upper, (len(times), 29)).copy()
        for frame, previous in boundary_indices.items():
            reach = self.velocity*(times[1]-times[0])
            lower_bands[frame] = np.maximum(lower_bands[frame], previous-reach)
            upper_bands[frame] = np.minimum(upper_bands[frame], previous+reach)
        # Restrict each current solution to states that can reach all later boxes.
        # The inward margin prevents independently rounded tight bounds diverging.
        for frame in range(len(times)-2, -1, -1):
            reach = self.velocity*(times[frame+1]-times[frame])-1e-8
            lower_bands[frame] = np.maximum(lower_bands[frame], lower_bands[frame+1]-reach)
            upper_bands[frame] = np.minimum(upper_bands[frame], upper_bands[frame+1]+reach)
        changed, qp_solves = [], 0
        for frame, target in enumerate(nominal):
            lower, upper = lower_bands[frame], upper_bands[frame]
            if frame:
                reach = self.velocity*(times[frame]-times[frame-1])
                lower = np.maximum(lower, result[frame-1, 7:]-reach)
                upper = np.minimum(upper, result[frame-1, 7:]+reach)
            if np.any(lower > upper):
                joint = int(np.argmax(lower-upper))
                raise ReferenceConstraintSolveError("Reference IK has incompatible boundary joint constraints: "
                    f"frame={frame},time={times[frame]:.17g},joint={joint},"
                    f"lower={lower[joint]:.17g},upper={upper[joint]:.17g},gap={lower[joint]-upper[joint]:.17g}")
            if np.all(target[7:] >= lower) and np.all(target[7:] <= upper):
                # Zero error and feasible constraints make the original pose the optimum.
                continue
            changed.append(frame)
            full = self._full(target)
            self.configuration.update(full)
            self.posture.set_target(full)
            for task in self.frames:
                task.set_target_from_configuration(self.configuration)
            current = target[7:].copy()
            for _ in range(self.iterations):
                problem = mink.build_ik(self.configuration, self.tasks, dt=1.,
                                        damping=1e-8, limits=[])
                selected = np.ix_(self.vadr, self.vadr)
                reduced = qpsolvers.Problem(problem.P[selected], problem.q[self.vadr],
                    lb=lower-current, ub=upper-current)
                solution = qpsolvers.solve_problem(reduced, solver="daqp",
                    primal_tol=1e-12, dual_tol=1e-12, iter_limit=100)
                qp_solves += 1
                if not solution.found or solution.x is None or not np.isfinite(solution.x).all():
                    raise ReferenceConstraintSolveError(f"Reference IK QP failed at {times[frame]:.6f}s")
                current += solution.x
                full[self.qadr] = current
                self.configuration.update(full)
            if np.any(current < lower-1e-8) or np.any(current > upper+1e-8):
                raise ReferenceConstraintSolveError("Reference IK violated an original joint constraint")
            result[frame, 7:] = current
        speeds = np.abs(np.diff(result[:, 7:], axis=0)/np.diff(times)[:, None])
        report = dict(self.identity, sample_count=len(result), solved_frames=changed,
            qp_solves=qp_solves, root_exact=bool(np.array_equal(result[:, :7], nominal[:, :7])),
            boundary_frame_count=len(boundary_indices),
            boundary_plan_id=None if boundary is None else boundary["plan_id"],
            minimum_position_margin_rad=float(np.minimum(result[:, 7:]-self.kinematics.limits[:, 0],
                self.kinematics.limits[:, 1]-result[:, 7:]).min()),
            maximum_velocity_ratio=float((speeds/self.velocity).max()),
            maximum_joint_change_rad=float(np.abs(result[:, 7:]-nominal[:, 7:]).max()),
            wall_ns=perf_counter_ns()-started_ns)
        return result, report
