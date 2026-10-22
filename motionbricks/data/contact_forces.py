"""Resultant ground-force demand from a centre-of-mass trajectory.

With the ground as the only external contact, Newton's second law for the whole body
gives the resultant force the ground would have to exert to explain the observed
centre-of-mass motion:

    F_ground / m = c_ddot - g

Everything here is per unit mass (m/s^2, equivalently N/kg), so no body mass has to
be assumed, no actor weight is guessed, and no per-foot load distribution is
invented. The resultant is the sum over all contact points, never one foot's share.

This is an inverse statement, not a measurement. It says what the ground would have
to do. A zero residual therefore only means "not contradicted by the stated
assumptions"; it never validates a motion and must not be reported as verified
contact, measured force or a solved inverse dynamics.

Time alignment: the acceleration at frame t comes from a Savitzky-Golay fit over a
window of frames centred on t, evaluated at t, so it stays index-aligned with the
input frames. Frames whose window is not fully interior are flagged as edges and
reported separately; they are never silently trusted. The filter is a linear
low-pass, so it preserves polynomials up to its fit order exactly and biases
everything else.
"""

import math

import torch
import torch.nn.functional as F
from scipy.signal import savgol_filter

#: Absolute floor for calling a required resultant force nonzero. A float32
#: trajectory differentiated twice at 60 Hz carries a numerical noise floor far above
#: machine epsilon, so the check needs a stated tolerance rather than a bare zero.
AIRBORNE_FORCE_TOLERANCE_MPS2 = 1e-3

#: Whole-body contact states, always mutually exclusive and exhaustive.
CONTACT_STATES = ("stable_support", "near_ground_undecided", "airborne", "unknown")


def center_of_mass_acceleration(com_m, fps, *, window_frames=7, polyorder=3):
    """Second time derivative of a centre-of-mass trajectory, in m/s^2.

    com_m is [..., frames, xyz]. The Savitzky-Golay fit keeps the derivative
    index-aligned with the frames: the value at frame t uses the window centred on t,
    with a polynomial edge fit at the ends. Returns the acceleration, the edge mask,
    and the filter settings actually used.
    """
    if com_m.ndim < 2 or com_m.shape[-1] != 3:
        raise ValueError("A centre of mass must be [..., frames, xyz]")
    if not com_m.is_floating_point() or not torch.isfinite(com_m).all():
        raise ValueError("The centre of mass must be finite floating-point metres")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("A finite positive FPS is required")
    frames = com_m.shape[-2]
    if window_frames % 2 != 1 or window_frames < 3:
        raise ValueError("The smoothing window must be an odd number of frames, at least 3")
    if polyorder < 2 or polyorder >= window_frames:
        raise ValueError("The polynomial order must be at least 2 and below the window length")
    if window_frames > frames:
        raise ValueError("The smoothing window cannot be longer than the clip")
    # Differentiate in float64 even when the stored trajectory is float32, so the
    # filter does not add its own accumulation error on top of the input precision.
    acceleration = savgol_filter(com_m.detach().to(torch.float64).numpy(), window_frames, polyorder,
                                 deriv=2, delta=1.0 / fps, axis=-2, mode="interp")
    acceleration = torch.as_tensor(acceleration, dtype=com_m.dtype, device=com_m.device)
    half = window_frames // 2
    index = torch.arange(frames, device=com_m.device)
    edge = (index < half) | (index >= frames - half)
    return {
        "acceleration_mps2": acceleration,
        "acceleration_edge": edge,
        "smoothing_window_frames": window_frames,
        "smoothing_polyorder": polyorder,
    }


def required_ground_force_per_mass(com_m, fps, *, gravity_mps2=9.81, window_frames=7, polyorder=3):
    """Resultant ground force per unit mass, F_ground / m = c_ddot - g, in m/s^2.

    Gravity acts along -Y in SOMA's Y-up world. At rest this returns +9.81 m/s^2 of
    upward force per kilogram, which is exactly the weight support a standing body
    needs; in free fall it returns zero.
    """
    if not math.isfinite(gravity_mps2) or gravity_mps2 <= 0:
        raise ValueError("A finite positive gravity is required")
    smoothed = center_of_mass_acceleration(com_m, fps, window_frames=window_frames, polyorder=polyorder)
    gravity = torch.zeros(3, dtype=com_m.dtype, device=com_m.device)
    gravity[1] = -gravity_mps2
    result = dict(smoothed)
    result["gravity_mps2"] = gravity_mps2
    result["required_force_per_mass_mps2"] = smoothed["acceleration_mps2"] - gravity
    return result


def phase_interior_mask(state_codes, window_frames):
    """Frames whose whole smoothing window lies inside one contact phase.

    A derivative window that spans a support-to-flight change mixes two different
    force regimes, so its output describes neither. state_codes is an integer code per
    frame; a frame is interior only when every frame within window_frames // 2 of it
    shares its code. Edge padding keeps the first and last frames non-interior.
    """
    if state_codes.ndim != 1:
        raise ValueError("A one-dimensional per-frame state code is required")
    if window_frames % 2 != 1 or window_frames < 1:
        raise ValueError("The smoothing window must be an odd number of frames")
    if window_frames > state_codes.numel():
        raise ValueError("The smoothing window cannot be longer than the clip")
    half = window_frames // 2
    codes = state_codes.to(torch.float64)[None, None]
    padded = F.pad(codes, (half, half), value=-1.0)
    maximum = F.max_pool1d(padded, window_frames, stride=1)[0, 0]
    minimum = -F.max_pool1d(-padded, window_frames, stride=1)[0, 0]
    return (maximum == codes[0, 0]) & (minimum == codes[0, 0])


def state_codes(contact_state, order=CONTACT_STATES):
    """Integer code per frame from a whole-body contact state, for phase erosion."""
    codes = torch.zeros_like(contact_state[order[0]], dtype=torch.long)
    total = torch.zeros_like(codes)
    for index, name in enumerate(order):
        mask = contact_state[name]
        codes = torch.where(mask, torch.full_like(codes, index), codes)
        total = total + mask.long()
    if int(total.min()) != 1 or int(total.max()) != 1:
        raise ValueError("Contact states must partition the clip exactly once")
    return codes


def violation_counts(residual, *, airborne_tolerance_mps2=AIRBORNE_FORCE_TOLERANCE_MPS2):
    """Count frames that violate each unit-mass residual, at the stated tolerance."""
    if not math.isfinite(airborne_tolerance_mps2) or airborne_tolerance_mps2 < 0:
        raise ValueError("A finite nonnegative airborne tolerance is required")
    return {
        "ground_pull_frames": int((residual["ground_pull_mps2"] > 0).sum()),
        "friction_capacity_frames": int((residual["friction_capacity_excess_mps2"] > 0).sum()),
        "airborne_force_frames": int((residual["airborne_force_magnitude_mps2"]
                                      > airborne_tolerance_mps2).sum()),
    }


def ground_force_feasibility(required_force_per_mass_mps2, contact_state, *, static_friction,
                            evaluated=None):
    """Unit-mass residuals for the required resultant ground force.

    contact_state is the whole-body per-frame state from
    contact_mechanics.body_contact_state. Residuals, all in m/s^2 (= N/kg):

    ground_pull_mps2
        Support frames where the required vertical force points down. The ground can
        push but cannot pull, so any positive value is not explainable.
    friction_capacity_excess_mps2
        Support frames where the required horizontal force exceeds what friction can
        supply at the given coefficient, measured against the upward part of the
        required normal force only.
    airborne_force_magnitude_mps2
        Airborne frames where the required resultant is not zero. With no ground
        contact the body must be in free fall.

    Unknown and contact-transition frames are never evaluated and never treated as
    airborne. The residuals measure different violations and are never summed.
    """
    if required_force_per_mass_mps2.ndim < 1 or required_force_per_mass_mps2.shape[-1] != 3:
        raise ValueError("A required ground force must be [..., frames, xyz]")
    if not required_force_per_mass_mps2.is_floating_point() or not torch.isfinite(
            required_force_per_mass_mps2).all():
        raise ValueError("The required ground force must be finite floating-point")
    if not math.isfinite(static_friction) or static_friction < 0:
        raise ValueError("A finite nonnegative static friction coefficient is required")
    frame_shape = required_force_per_mass_mps2.shape[:-1]
    for name in CONTACT_STATES:
        mask = contact_state[name]
        if mask.dtype != torch.bool or mask.shape != frame_shape:
            raise ValueError("Every contact-state mask must be boolean and match the force frames")
    overlap = sum(contact_state[name].long() for name in CONTACT_STATES)
    if int(overlap.min()) != 1 or int(overlap.max()) != 1:
        raise ValueError("Contact states must partition the clip exactly once")
    force = required_force_per_mass_mps2
    normal = force[..., 1]
    upward = F.relu(normal)
    horizontal = force[..., [0, 2]].norm(dim=-1)
    if evaluated is None:
        evaluated = torch.ones(frame_shape, dtype=torch.bool, device=required_force_per_mass_mps2.device)
    elif evaluated.dtype != torch.bool or evaluated.shape != frame_shape:
        raise ValueError("An evaluation mask must be boolean and match the force frames")
    # Only a labeled stable support carries ground reaction. A sole that is merely near
    # the floor is undecided, so it is neither loaded nor flown.
    support = contact_state["stable_support"] & evaluated
    airborne = contact_state["airborne"] & evaluated
    zeros = torch.zeros_like(normal)
    return {
        "ground_pull_mps2": torch.where(support, F.relu(-normal), zeros),
        "friction_capacity_excess_mps2": torch.where(support, F.relu(horizontal - static_friction * upward),
                                                     zeros),
        "airborne_force_magnitude_mps2": torch.where(airborne, force.norm(dim=-1), zeros),
        "evaluated_contact_frames": support,
        "evaluated_airborne_frames": airborne,
    }
