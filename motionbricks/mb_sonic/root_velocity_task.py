"""Contact-dynamics correction of frozen-tracker joint targets, without integration."""

from time import perf_counter_ns

import daqp
import mujoco
import numpy as np


class RootVelocityTask:
    """Fit a bounded actuator residual to measured COM velocity and pelvis yaw."""

    def __init__(self, physics, *, torque_fraction=.08, velocity_time_s=.4,
                 yaw_time_s=.25, acceleration_limit_mps2=1.5, yaw_acceleration_limit=4.):
        if not physics.is_protomotions:
            raise ValueError('Root velocity correction requires the explicit implicit-PD tracker')
        if not 0 < torque_fraction <= .15 or min(velocity_time_s, yaw_time_s,
                acceleration_limit_mps2, yaw_acceleration_limit) <= 0:
            raise ValueError('Invalid bounded Root velocity task settings')
        self.model, self.ids = physics.model, physics.actuator_ids.copy()
        self.lower, self.upper = physics.torque_lower.copy(), physics.torque_upper.copy()
        self.scale = np.minimum(-self.lower, self.upper)/physics.kp
        self.bound = np.minimum(torque_fraction, .15/self.scale)
        self.velocity_time, self.yaw_time = velocity_time_s, yaw_time_s
        self.acceleration_limit, self.yaw_acceleration_limit = acceleration_limit_mps2, yaw_acceleration_limit
        self.pelvis = self.model.body('pelvis').id
        self.data, self.shifted = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self.jac = np.zeros((3, self.model.nv))
        self.shifted_jac = np.zeros_like(self.jac)
        self.identity = dict(schema='motionbricks.mb_sonic.root_velocity_task.v1',
            method='contact_forward_dynamics_linearization_DAQP', torque_fraction=torque_fraction,
            velocity_time_s=velocity_time_s, yaw_time_s=yaw_time_s,
            acceleration_limit_mps2=acceleration_limit_mps2,
            yaw_acceleration_limit_radps2=yaw_acceleration_limit,
            actual_state_writes=0, external_forces=0, optimizer_updates=0,
            input='latest independent world-XY velocity and yaw; measured COM and pelvis state')

    def linearize(self, actual, target):
        """Reuse contact geometry; only private forward dynamics is differentiated."""
        d, m = self.data, self.model
        mujoco.mj_copyData(d, m, actual)
        d.ctrl[self.ids] = target
        mujoco.mj_forward(m, d)
        base_acc, base_force = d.qacc.copy(), d.actuator_force[self.ids].copy()
        warmstart = d.qacc_warmstart.copy()
        mujoco.mj_jacSubtreeCom(m, d, self.jac, self.pelvis)
        velocity = self.jac@d.qvel
        rotation = d.xmat[self.pelvis].reshape(3, 3).copy()
        projection = np.zeros((6, m.nv))
        projection[:3] = self.jac
        projection[3:, 3:6] = rotation
        response, forces = np.empty((6, 29)), np.empty((29, 29))
        eps = 1e-5
        for column, actuator in enumerate(self.ids):
            d.ctrl[self.ids] = target
            d.ctrl[actuator] += eps*self.scale[column]
            d.qacc_warmstart[:] = warmstart
            mujoco.mj_forwardSkip(m, d, mujoco.mjtStage.mjSTAGE_VEL, 1)
            response[:, column] = projection@(d.qacc-base_acc)/eps
            forces[:, column] = (d.actuator_force[self.ids]-base_force)/eps
        # The directional kinematic derivative supplies the COM Jdot*v term.
        s = self.shifted
        s.qpos[:] = actual.qpos
        mujoco.mj_integratePos(m, s.qpos, actual.qvel, eps)
        mujoco.mj_kinematics(m, s)
        mujoco.mj_comPos(m, s)
        mujoco.mj_jacSubtreeCom(m, s, self.shifted_jac, self.pelvis)
        acceleration = projection@base_acc
        acceleration[:3] += ((self.shifted_jac-self.jac)/eps)@actual.qvel
        return dict(response=response, forces=forces, base_force=base_force,
            acceleration=acceleration, velocity=velocity, rotation=rotation,
            angular_velocity=rotation@actual.qvel[3:6], base_qacc=base_acc,
            projection=projection, warmstart=warmstart)

    def solve(self, actual, target, *, velocity_xy_mps, yaw_rate_radps, heading_rad):
        started = perf_counter_ns()
        target, velocity_xy = np.asarray(target, dtype=float), np.asarray(velocity_xy_mps, dtype=float)
        if (target.shape != (29,) or velocity_xy.shape != (2,) or
                not np.isfinite(np.r_[target, velocity_xy, yaw_rate_radps, heading_rad]).all()):
            raise ValueError('Root task requires finite named joint targets and independent planar intent')
        system = self.linearize(actual, target)
        linearized = perf_counter_ns()
        base = system['acceleration']
        desired = base.copy()
        desired[:2] = (velocity_xy-system['velocity'][:2])/self.velocity_time
        length = np.linalg.norm(desired[:2])
        desired[:2] *= min(1., self.acceleration_limit/max(length, 1e-12))
        actual_heading = np.arctan2(system['rotation'][1, 0], system['rotation'][0, 0])
        error = np.arctan2(np.sin(heading_rad-actual_heading), np.cos(heading_rad-actual_heading))
        desired[5] = np.clip((yaw_rate_radps+2*error-system['angular_velocity'][2])/self.yaw_time,
                             -self.yaw_acceleration_limit, self.yaw_acceleration_limit)
        # Preserve the actor's vertical and roll/pitch dynamics while fitting the planar task.
        weights = np.array([1., 1., 4., 2., 2., .25])
        matrix = weights[:, None]*system['response']
        residual = weights*(desired-base)
        hessian = matrix.T@matrix+.1*np.eye(29)
        gradient = -matrix.T@residual
        constraint = np.vstack((np.eye(29), system['forces']))
        lower = np.r_[-self.bound, self.lower-system['base_force']]
        upper = np.r_[self.bound, self.upper-system['base_force']]
        scale = max(1., np.diag(hessian).max())
        correction, _, status, _ = daqp.solve(np.ascontiguousarray(hessian/scale),
            np.ascontiguousarray(gradient/scale), constraint, upper, lower,
            primal_tol=1e-9, dual_tol=1e-9, eps_prox=0., iter_limit=300)
        if status != 1 or not np.isfinite(correction).all():
            raise RuntimeError(f'Root velocity QP failed: DAQP status {status}')
        result = target+self.scale*correction
        finished = perf_counter_ns()
        predicted = base+system['response']@correction
        return result, dict(schema=self.identity['schema'],
            native_joint_target=target.copy(), applied_joint_target=result.copy(),
            measured_com_velocity_mps=system['velocity'].copy(), requested_velocity_xy_mps=velocity_xy.copy(),
            requested_yaw_rate_radps=float(yaw_rate_radps), desired_heading_rad=float(heading_rad),
            baseline_acceleration=base.copy(), desired_acceleration=desired,
            predicted_acceleration=predicted, normalized_torque_correction=correction,
            predicted_actuator_forces=system['base_force']+system['forces']@correction,
            timing_ns=dict(dynamics=linearized-started, qp=finished-linearized, total=finished-started))
