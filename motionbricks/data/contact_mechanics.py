"""Flat-floor contact diagnostics, not a dynamics solver or a training objective.

Conventions
-----------
World space, SI units, SOMA's Y-up frame: metres, seconds, newtons, watts.
The floor is static, horizontal and rigid at ``ground_height_m`` along +Y.
A contact force is exerted *by the floor on the body*, so +Y pushes the body up.
For a point contact with gap ``d``, normal load ``Fn`` and tangential force ``Ft``
the classical Coulomb model (Drake's friction model) requires

    d >= 0,    Fn >= 0,    d * Fn == 0,
    |Ft| <= mu_s * Fn            while sticking,
    Ft = -mu_k * Fn * v_hat_t    while sliding,    0 <= mu_k <= mu_s.

Friction coefficients are explicit scene parameters. The values used in tests are
analytic samples, not a measurement of any real shoe sole.

Time alignment
--------------
Frame quantities are sampled at ``t = 0 .. T-1``: sole gaps, penetration, contact
labels, confidence and any per-frame contact force. Interval quantities describe
the motion over ``[t, t+1]`` for ``t = 0 .. T-2``: interval gap, normal velocity,
tangential velocity and tangential speed. ``interval_average`` is the two-endpoint
mean that moves a frame quantity onto an interval; that midpoint rule is a discrete
diagnostic convention, not a claim about when a force is applied. Signed per-frame
residuals keep their own frame index and must not be averaged silently.

Forces and labels
-----------------
Normal loads must come from a separate, identified source. Contact labels and
confidence say where the source data reports support; they never measure force and
this module never invents one. Contact state is decided by those labels, never by
velocity and never by a predicted force, so a tangential-speed threshold alone can
never turn a swing interval into support.
"""

import math

import torch
import torch.nn.functional as F


def interval_average(frame_values):
    """Mean of the two endpoints of every ``[t, t+1]`` interval.

    This is the discrete convention used for interval gaps and for aligning frame
    quantities such as contact labels or forces with interval velocities. It is a
    diagnostic midpoint rule, not a statement about when a force is applied.
    """
    if frame_values.ndim < 2 or frame_values.shape[-2] < 2:
        raise ValueError("At least two frames on the second-to-last axis are required")
    return 0.5 * (frame_values[..., :-1, :] + frame_values[..., 1:, :])


def flat_ground_contact_kinematics(soles_m, fps, ground_height_m=0.0):
    """Split ``[batch, frames, contacts, xyz]`` FK sole motion on a static floor.

    Gaps have ``frames`` samples, indexed by frame time ``t``. Velocities and speeds
    have ``frames - 1`` samples and describe the interval ``[t, t+1]``. Interval gaps
    are the two-endpoint mean.

    Sole points are contact proxies from the official skinned mesh; they are not the
    ankle joint and no assumption is made that all four channels touch down together.
    No contact classifier, root gate, pose correction or foot lock is applied here.
    """
    if soles_m.ndim != 4 or soles_m.shape[-1] != 3 or soles_m.shape[1] < 2:
        raise ValueError("Soles must be [batch, frames>=2, contacts, xyz]")
    if not soles_m.is_floating_point() or not torch.isfinite(soles_m).all():
        raise ValueError("Sole positions must be finite floating-point meters")
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(ground_height_m):
        raise ValueError("Finite positive FPS and finite ground height are required")
    gap = soles_m[..., 1] - ground_height_m
    velocity = (soles_m[:, 1:] - soles_m[:, :-1]) * fps
    tangential_velocity = velocity[..., [0, 2]]
    return {
        "gap_m": gap,
        "penetration_m": F.relu(-gap),
        "interval_gap_m": interval_average(gap),
        "interval_normal_velocity_mps": velocity[..., 1],
        "interval_tangential_velocity_mps": tangential_velocity,
        "interval_tangential_speed_mps": tangential_velocity.norm(dim=-1),
    }


def support_frame_mask(contacts, confidence):
    """Frames the source labels as support *and* marks confident, shape ``[..., T, C]``.

    A frame is support only when the label says so and the confidence is true. An
    unconfident frame is unknown, never support.
    """
    _require_aligned_labels(contacts, confidence)
    return contacts.bool() & confidence.bool()


def support_interval_masks(contacts, confidence):
    """Classify every ``[t, t+1]`` interval from both endpoints' label and confidence.

    Returns four mutually exclusive boolean masks of shape ``[..., T-1, C]``:

    ``sustained_support``
        Both endpoints labeled support and both confident. Only these intervals may
        be read as continuous support, and only this mask is valid for slip or drift
        statistics.
    ``transition``
        Both endpoints confident and the labels differ: touch-down or lift-off. The
        zero relative-motion assumption of sustained support does not apply here.
    ``swing``
        Both endpoints confident and neither is support.
    ``unknown``
        At least one endpoint is not confident. Unknown intervals are never counted
        as support and are reported as their own count.
    """
    _require_aligned_labels(contacts, confidence)
    if contacts.shape[-2] < 2:
        raise ValueError("At least two frames are required to form an interval")
    labels = contacts.bool()
    confident = confidence.bool()
    support = support_frame_mask(labels, confident)
    both_confident = confident[..., 1:, :] & confident[..., :-1, :]
    return {
        "sustained_support": support[..., 1:, :] & support[..., :-1, :],
        "transition": both_confident & (labels[..., 1:, :] != labels[..., :-1, :]),
        "swing": both_confident & ~labels[..., 1:, :] & ~labels[..., :-1, :],
        "unknown": ~both_confident,
    }


def body_contact_state(contacts, confidence, gap_m, *, contact_clearance_m=0.035,
                       airborne_clearance_m=0.06):
    """Whole-body per-frame contact state from labels, confidence and sole height.

    Returns four mutually exclusive boolean masks of shape [..., T]:

    stable_support
        At least one channel is labeled support and confident. A foot that is merely
        slow is not enough; this is the label's own stable-support claim.
    near_ground_undecided
        No stable support, but at least one sole is at or below contact_clearance_m.
        Being near the floor is not evidence of contact: the proxies carry centimetres
        of uncertainty, so this state stays undecided. It is never weight-bearing, never
        sliding, never flight and never training truth.
    airborne
        No stable support, every sole strictly above airborne_clearance_m, and every
        channel confident. Flight needs measured clearance and agreeing labels, never
        just a missing support label.
    unknown
        Everything else: the band between the two clearances, and any frame with an
        unconfident channel that is not already stable support.

    Neither a missing support label nor mere proximity to the floor may create contact
    or flight. Confirming contact needs calibrated surface evidence, which the fixed
    proxy points do not yet provide.
    """
    _require_aligned_labels(contacts, confidence)
    if gap_m.shape != contacts.shape:
        raise ValueError("Sole gaps must match the contact label shape, one height per channel")
    if not gap_m.is_floating_point() or not torch.isfinite(gap_m).all():
        raise ValueError("Sole gaps must be finite floating-point metres")
    if (not math.isfinite(contact_clearance_m) or not math.isfinite(airborne_clearance_m)
            or contact_clearance_m < 0 or contact_clearance_m >= airborne_clearance_m):
        raise ValueError("Clearances must satisfy 0 <= contact_clearance_m < airborne_clearance_m")
    lowest = gap_m.amin(dim=-1)
    grounded = lowest <= contact_clearance_m
    confirmed_clear = (lowest > airborne_clearance_m) & confidence.bool().all(dim=-1)
    support = support_frame_mask(contacts.bool(), confidence).any(dim=-1)
    return {
        "stable_support": support,
        "near_ground_undecided": ~support & grounded,
        "airborne": ~support & confirmed_clear,
        "unknown": ~support & ~grounded & ~confirmed_clear,
    }


def masked_summary(values, mask=None):
    """Pooled mean/minimum/P95/peak of the selected samples, or explicit nulls.

    ``values`` and ``mask`` must share a shape. With ``mask=None`` every sample is
    used. An empty selection returns ``samples == 0`` and ``None`` for every
    statistic so that "no valid sample" can never be reported as a zero error.
    Quantiles are taken over the pooled samples, never as an average of per-window
    quantiles.
    """
    if not torch.is_floating_point(values):
        raise ValueError("Summarized values must be floating-point SI quantities")
    if not torch.isfinite(values).all():
        raise ValueError("Summarized values must be finite")
    if mask is None:
        selected = values.reshape(-1)
    else:
        if mask.dtype != torch.bool or mask.shape != values.shape:
            raise ValueError("A boolean mask must match the summarized values")
        selected = values[mask]
    samples = int(selected.numel())
    if samples == 0:
        return {"samples": 0, "mean": None, "minimum": None, "p95": None, "peak": None}
    return {
        "samples": samples,
        "mean": float(selected.mean()),
        "minimum": float(selected.min()),
        "p95": float(torch.quantile(selected, 0.95)),
        "peak": float(selected.max()),
    }


def sustained_support_segments(sole_positions_m, gap_m, sustained_support):
    """Drift of every maximal run of sustained-support intervals, per channel.

    One clip only: ``sole_positions_m`` is ``[frames, channels, xyz]``, ``gap_m`` is
    ``[frames, channels]`` and ``sustained_support`` is ``[frames-1, channels]``.
    Returns one list of segment records per channel.

    A segment is a maximal run of consecutive sustained-support intervals; its frames
    run from the run's first to the run's last endpoint, so ``frames = intervals + 1``.
    ``path_length_m`` accumulates every interval displacement, so back-and-forth
    oscillation counts; ``net_drift_m`` is the straight horizontal distance from the
    first to the last frame; ``max_anchor_m`` is the largest horizontal distance from
    the first frame. The normal channel is kept separate as ``normal_range_m``
    (peak-to-peak gap) and ``normal_net_m`` (gap change over the segment).
    """
    if sole_positions_m.ndim != 3 or sole_positions_m.shape[-1] != 3:
        raise ValueError("One clip of [frames, channels, xyz] sole positions is required")
    if gap_m.shape != sole_positions_m.shape[:2]:
        raise ValueError("Gaps must match the clip's frame and channel axes")
    frames, channels = gap_m.shape
    if sustained_support.shape != (frames - 1, channels) or sustained_support.dtype != torch.bool:
        raise ValueError("A boolean sustained-support mask must cover every [t, t+1] interval")
    segments = []
    for channel in range(channels):
        records = []
        for first, last in _true_runs(sustained_support[:, channel].tolist()):
            start, end = first, last + 1
            horizontal = sole_positions_m[start:end + 1, channel, [0, 2]]
            gap = gap_m[start:end + 1, channel]
            records.append({
                "start_frame": start,
                "end_frame": end,
                "intervals": last - first + 1,
                "frames": end - start + 1,
                "path_length_m": float((horizontal[1:] - horizontal[:-1]).norm(dim=-1).sum()),
                "net_drift_m": float((horizontal[-1] - horizontal[0]).norm()),
                "max_anchor_m": float((horizontal - horizontal[0]).norm(dim=-1).max()),
                "normal_range_m": float(gap.max() - gap.min()),
                "normal_net_m": float(gap[-1] - gap[0]),
            })
        segments.append(records)
    return segments


def coulomb_contact_residuals(gap_m, relative_velocity_mps, contact_force_n, *,
                             static_friction, dynamic_friction, slip_speed_mps=1e-4):
    """Check a candidate point-contact force on a flat floor, in SI units.

    ``gap_m`` is ``[...]``, ``relative_velocity_mps`` and ``contact_force_n`` are
    ``[..., xyz]`` and describe the *same* frame or interval. Force is exerted by the
    floor on the body; positive Y pushes upward. Static friction is set-valued inside
    its cone rather than always equal to ``mu_s * Fn``. While sliding, friction
    opposes relative XZ velocity with magnitude ``mu_k * Fn``.

    ``kinetic_friction_branch`` is selected purely by tangential speed exceeding
    ``slip_speed_mps``. It says which friction law to evaluate; it is not a contact
    state, not a measured slip event and not evidence that the point is supported.
    At zero speed static friction is set-valued, so no direction is invented.

    Every residual keeps its own unit and they are never summed into one loss:
    ``penetration_m`` [m], ``normal_pull_n`` [N], ``contact_complementarity_nm``
    [N*m], ``friction_cone_excess_n`` [N], ``kinetic_friction_residual_n`` [N] and
    ``friction_energy_generation_w`` [W].
    """
    if (relative_velocity_mps.shape != contact_force_n.shape
            or relative_velocity_mps.ndim < 1 or relative_velocity_mps.shape[-1] != 3
            or gap_m.shape != relative_velocity_mps.shape[:-1]):
        raise ValueError("Matching [..., xyz] velocities/forces and [...] gaps are required")
    for value in (gap_m, relative_velocity_mps, contact_force_n):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("Contact quantities must be finite floating-point SI values")
        if value.device != gap_m.device or value.dtype != gap_m.dtype:
            raise ValueError("Contact quantities must share a device and dtype")
    if (not math.isfinite(static_friction) or not math.isfinite(dynamic_friction)
            or not 0 <= dynamic_friction <= static_friction):
        raise ValueError("Friction coefficients must satisfy 0 <= mu_k <= mu_s")
    if not math.isfinite(slip_speed_mps) or slip_speed_mps <= 0:
        raise ValueError("A finite positive slip-speed threshold is required")

    normal_force = contact_force_n[..., 1]
    normal_load = normal_force.clamp(min=0)
    tangential_force = contact_force_n[..., [0, 2]]
    tangential_velocity = relative_velocity_mps[..., [0, 2]]
    speed = tangential_velocity.norm(dim=-1)
    kinetic_friction_branch = speed > slip_speed_mps
    # At zero speed static friction is set-valued; do not invent its direction.
    kinetic_target = (-dynamic_friction * normal_load)[..., None] * (
        tangential_velocity / speed.clamp(min=slip_speed_mps)[..., None])
    kinetic_residual = (tangential_force - kinetic_target).norm(dim=-1)
    friction_power = (tangential_force * tangential_velocity).sum(dim=-1)
    # |d| * Fn decomposes exactly into the two ways d * Fn = 0 can fail.
    penetration_m = F.relu(-gap_m)
    separated_load_nm = F.relu(gap_m) * normal_load
    return {
        "penetration_m": penetration_m,
        "normal_pull_n": F.relu(-normal_force),
        "contact_complementarity_nm": penetration_m * normal_load + separated_load_nm,
        "friction_cone_excess_n": F.relu(tangential_force.norm(dim=-1) - static_friction * normal_load),
        "kinetic_friction_residual_n": torch.where(kinetic_friction_branch, kinetic_residual,
                                                   torch.zeros_like(speed)),
        "friction_energy_generation_w": F.relu(friction_power),
        "kinetic_friction_branch": kinetic_friction_branch,
    }


def _require_aligned_labels(contacts, confidence):
    if contacts.shape != confidence.shape or contacts.ndim < 2:
        raise ValueError("Contact labels and confidence must share a [..., frames, contacts] shape")
    for value in (contacts, confidence):
        if value.dtype != torch.bool:
            raise ValueError("Contact labels and confidence must be boolean")


def _true_runs(flags):
    """Maximal inclusive runs of true values, as (first, last) index pairs."""
    runs, start = [], None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs
