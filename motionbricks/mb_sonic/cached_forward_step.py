"""Cached-position/velocity execution for MuJoCo3.8 implicitfast.

Precondition: mj_forward has completed at the current state. Since then only
ctrl or external force inputs may change, never qpos/qvel/act/model parameters.
The owning runtime completes the initial forward and never mutates state outside integration.

Pinned implementation: https://github.com/google-deepmind/mujoco/blob/3.8.0/src/engine/engine_forward.c
API: https://mujoco.readthedocs.io/en/3.8.0/APIreference/APIfunctions.html#mj-forwardskip
"""

import mujoco


def cached_forward_step(model, data):
    """One implicitfast step and full post-forward; no direct state writes."""
    if mujoco.__version__ != "3.8.0":
        raise RuntimeError("Cached-forward parity is scoped to MuJoCo3.8.0")
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        raise ValueError("Cached-forward step supports only implicitfast")
    if not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_AUTORESET:
        raise ValueError("Automatic state reset must be disabled")
    if model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP or model.nplugin:
        raise ValueError("Sleeping trees and plugins are outside the cached-forward contract")
    for name in ("control", "passive", "contactfilter", "sensor", "act_dyn", "act_gain", "act_bias", "time"):
        if getattr(mujoco, "get_mjcb_"+name)() is not None:
            raise ValueError("External MuJoCo callbacks are not permitted: "+name)
    mujoco.mj_checkPos(model, data)
    mujoco.mj_checkVel(model, data)
    mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_VEL, 0)
    mujoco.mj_checkAcc(model, data)
    if model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_FWDINV:
        mujoco.mj_compareFwdInv(model, data)
    mujoco.mj_implicit(model, data)
    mujoco.mj_forward(model, data)
