import mujoco
import numpy as np
import pytest

from motionbricks.mb_sonic.physics_numerics import configure_physics_numerics
from motionbricks.mb_sonic.reference import G1_JOINT_NAMES


def model():
    joints = ''.join(f'<body><joint name="{name}" range="-1 1"/><geom size=".02"/></body>'
                     for name in G1_JOINT_NAMES)
    return mujoco.MjModel.from_xml_string(f'<mujoco><compiler angle="radian"/><worldbody>{joints}</worldbody></mujoco>')


def settings():
    return dict(physics_integrator="implicitfast", joint_stop_solver=dict(
        schema="motionbricks.g1.joint_stop_solver.v1", margin_rad=.02,
        time_constant_s=.002, damping_ratio=1.))


def test_stop_solver_preserves_ranges_and_uses_internal_constraints():
    m = model()
    ranges, masses, inertia = m.jnt_range.copy(), m.body_mass.copy(), m.body_inertia.copy()
    receipt = configure_physics_numerics(m, .001, settings())
    np.testing.assert_array_equal(m.jnt_range, ranges)
    np.testing.assert_array_equal(m.body_mass, masses)
    np.testing.assert_array_equal(m.body_inertia, inertia)
    np.testing.assert_array_equal(m.jnt_margin, .02)
    np.testing.assert_allclose(m.jnt_solref, np.tile([.002, 1.], (29, 1)))
    assert not receipt["joint_stop_solver"]["state_projection"]


def test_stop_timescale_must_be_resolved_by_real_physics_steps():
    with pytest.raises(ValueError, match="resolved timescale"):
        configure_physics_numerics(model(), .005, settings())


def test_unconfigured_original_numerics_are_unchanged():
    m = model()
    before = m.jnt_solref.copy()
    configure_physics_numerics(m, .005, {})
    np.testing.assert_array_equal(m.jnt_solref, before)
