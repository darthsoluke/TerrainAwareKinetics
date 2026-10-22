"""Declared integration and internal joint-stop precision; no state projection."""

import mujoco
import numpy as np

from .reference import G1_JOINT_NAMES


def configure_physics_numerics(model, physics_dt, settings):
    if not np.isfinite(physics_dt) or not 0 < physics_dt <= .02:
        raise ValueError("Physics timestep must be finite and positive")
    before = model.jnt_range.copy()
    result = dict(physics_dt_s=physics_dt, integrator=mujoco.mjtIntegrator(model.opt.integrator).name)
    if "physics_integrator" in settings:
        name = settings["physics_integrator"]
        model.opt.integrator = {"Euler": mujoco.mjtIntegrator.mjINT_EULER,
                                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}[name]
        result["integrator"] = name
    if "joint_stop_solver" in settings:
        config = settings["joint_stop_solver"]
        margin, time_constant, damping = (float(config[name]) for name in
                                          ("margin_rad", "time_constant_s", "damping_ratio"))
        if (config["schema"] != "motionbricks.g1.joint_stop_solver.v1"
                or not np.isfinite([margin, time_constant, damping]).all()
                or not 0 < margin <= .03 or time_constant < 2*physics_dt or damping != 1.):
            raise ValueError("Joint stops require a declared conservative margin, resolved timescale and critical damping")
        ids = np.array([model.joint(name).id for name in G1_JOINT_NAMES])
        if not model.jnt_limited[ids].all() or (before[ids, 1]-before[ids, 0] <= 2*margin).any():
            raise ValueError("Joint-stop margins must remain inside the original bounded hinge ranges")
        result["joint_stop_solver"] = dict(config,
            previous_margin_rad=model.jnt_margin[ids].tolist(),
            previous_solref=model.jnt_solref[ids].tolist(),
            original_joint_ranges_rad=before[ids].tolist(),
            joint_names=list(G1_JOINT_NAMES), state_projection=False)
        model.jnt_margin[ids] = margin
        model.jnt_solref[ids] = [time_constant, damping]
    if not np.array_equal(model.jnt_range, before):
        raise AssertionError("Numerical configuration changed a mechanical joint range")
    return result
