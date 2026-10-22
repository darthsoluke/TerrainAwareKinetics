import mujoco
import numpy as np
import pytest

from motionbricks.mb_sonic.root_velocity_task import RootVelocityTask
from test_mb_sonic_tracker_wiring import initialized, scene_model


def test_actor_action_history_receives_the_actually_applied_target(scene_model):
    from motionbricks.mb_sonic.protomotions_policy import ProtoMotionsCommand

    physics, policy, _ = initialized(scene_model)
    physics.warmup_history()
    physics.root_velocity_task = RootVelocityTask(physics)
    target = physics.data.qpos[physics.qpos_indices].copy()
    command = ProtoMotionsCommand(target.copy(), np.zeros(29), target.copy(), {}, {})
    applied = physics.apply_velocity_task(command,
        dict(velocity_xy_mps=[.5, 0.], yaw_rate_radps=0.), 0.)
    np.testing.assert_array_equal(command.joint_target, target)
    np.testing.assert_array_equal(applied.joint_target, policy.previous_target)
    np.testing.assert_array_equal(applied.processed_action_mujoco, policy.previous_target)
    np.testing.assert_array_equal(applied.execution_task['native_joint_target'], target)
    assert applied.timing_ns['root_velocity_task'] > 0


def test_private_dynamics_never_mutates_actual_state_or_original_model(scene_model):
    physics, _, _ = initialized(scene_model)
    physics.warmup_history()
    task = RootVelocityTask(physics)
    state = {key: getattr(physics.data, key).copy() for key in
             ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl', 'qfrc_applied', 'xfrc_applied')}
    time = physics.data.time
    original = physics.model.actuator_forcerange.copy()
    target = physics.data.qpos[physics.qpos_indices].copy()
    applied, report = task.solve(physics.data, target, velocity_xy_mps=[.5, 0.], yaw_rate_radps=0., heading_rad=0.)
    for key, expected in state.items():
        np.testing.assert_array_equal(getattr(physics.data, key), expected)
    assert physics.data.time == time
    np.testing.assert_array_equal(physics.model.actuator_forcerange, original)
    assert np.max(np.abs((applied-target)/task.scale)) <= .08000001
    force = report['predicted_actuator_forces']
    assert (force >= task.lower-1e-7).all() and (force <= task.upper+1e-7).all()


def test_skipped_forward_derivatives_match_full_forward(scene_model):
    physics, _, _ = initialized(scene_model)
    physics.warmup_history()
    task = RootVelocityTask(physics)
    target = physics.data.qpos[physics.qpos_indices].copy()
    system = task.linearize(physics.data, target)
    private = mujoco.MjData(physics.model)
    for index in (0, 4, 6, 12, 24):
        mujoco.mj_copyData(private, physics.model, physics.data)
        private.ctrl[physics.actuator_ids] = target
        private.ctrl[physics.actuator_ids[index]] += 1e-5*task.scale[index]
        private.qacc_warmstart[:] = system['warmstart']
        mujoco.mj_forward(physics.model, private)
        expected = system['projection']@(private.qacc-system['base_qacc'])/1e-5
        np.testing.assert_allclose(system['response'][:, index], expected, atol=1e-7, rtol=1e-7)


def test_airborne_internal_torques_cannot_accelerate_whole_robot_com(scene_model):
    physics, _, _ = initialized(scene_model)
    # This is an isolated reset fixture, never a state edit during an episode.
    qpos = physics.data.qpos.copy()
    qpos[2] = 3.
    physics.reset(qpos, np.zeros(physics.model.nv))
    task = RootVelocityTask(physics)
    system = task.linearize(physics.data, physics.data.qpos[physics.qpos_indices].copy())
    np.testing.assert_allclose(system['response'][:3], 0., atol=3e-8)
    np.testing.assert_allclose(system['acceleration'][:3], physics.model.opt.gravity, atol=1e-8)


def test_reject_nonfinite_command(scene_model):
    physics, _, _ = initialized(scene_model)
    with pytest.raises(ValueError, match='finite'):
        RootVelocityTask(physics).solve(physics.data, np.zeros(29),
            velocity_xy_mps=[np.nan, 0], yaw_rate_radps=0, heading_rad=0)
