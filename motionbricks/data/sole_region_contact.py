"""Contact geometry and slip from the official skinned sole region.

The fixed proxy points are replaced here by the sole region: a fixed set of numbered
material vertices of the official skinned mesh, 98 per foot, taken as the union of
that foot's heel and forefoot channel selections. Clearance and penetration come from
that region's lowest vertex; slip comes from the displacement of the same numbered
vertices, never from whichever vertex happens to be lowest this frame.

Ground handling: the ground is the fixed plane the conversion already subtracted, and
it stays put. Nothing here moves the ground, moves the body, or re-fits a plane per
frame. If the region sits below or above that plane, that is reported as a measured
residual.

Rolling is expected, not penalised: a heel-to-toe roll hands contact from one group of
vertices to another, and each group's own run carries almost no displacement. Slip is
only measured inside a run of consecutive frames where the same vertex stays within
the contact clearance, so a vertex that dips near the floor for one frame during swing
cannot manufacture slip.
"""

import numpy as np
import torch

#: Candidate thresholds for review. They are declared from the measured sole-region
#: geometry of the standing and slow-walking windows, not tuned to reproduce the old
#: labels or to maximise agreement with them.
SOLE_CONTACT_CLEARANCE_M = 0.010
SOLE_CLEARANCE_M = 0.030
SOLE_SLIP_SPEED_MPS = 0.020
SOLE_MIN_CONTACT_RUN_FRAMES = 3
SOLE_REST_VERTICAL_SPEED_MPS = 0.05
SOLE_LABEL_THRESHOLDS = {
    "contact_clearance_m": SOLE_CONTACT_CLEARANCE_M,
    "clearance_m": SOLE_CLEARANCE_M,
    "slip_speed_mps": SOLE_SLIP_SPEED_MPS,
    "min_contact_run_frames": SOLE_MIN_CONTACT_RUN_FRAMES,
    "rest_vertical_speed_mps": SOLE_REST_VERTICAL_SPEED_MPS,
    "status": "candidate thresholds requiring review, not approved",
    "derivation": "In the standing window the sole region sits within -3.8 mm (left) and +7.5 mm (right) of the "
                  "fixed plane, so a 10 mm contact band covers a planted foot on both sides while still reporting "
                  "the left foot's penetration. 30 mm separates a clearly lifted foot. 20 mm/s is one third of a "
                  "millimetre per frame at 60 Hz, well below any real skate and well above the standing jitter "
                  "measured at 1.5-2.0 mm/s.",
    "not_fitted_to_old_labels": "The old contact labels were used only to describe where the near-floor population "
                                "is. No threshold was chosen to reproduce them.",
}

FRAME_STATES = ("contact", "undecided", "clearance")
INTERVAL_STATES = ("stable_contact", "sliding_contact", "contact_transition", "clearance", "undecided")


def sole_region_indices(surfaces, side):
    """Fixed material vertex indices for one foot: its heel and forefoot selections."""
    if side not in ("Left", "Right"):
        raise ValueError("Side must be Left or Right")
    channels = [surface for surface in surfaces if surface["channel"].startswith(side)]
    if not channels:
        raise ValueError(f"No sole channels found for {side}")
    return np.array(sorted({int(index) for surface in channels
                            for index in surface["selected_indices"]}), dtype=np.int64)


def sole_region_heights(vertices_m, indices, ground_height_m=0.0):
    """Height of every region vertex above the fixed ground, shaped [frames, vertices]."""
    if vertices_m.ndim != 3 or vertices_m.shape[-1] != 3:
        raise ValueError("Skinned vertices must be [frames, vertices, xyz]")
    if not vertices_m.is_floating_point() or not torch.isfinite(vertices_m).all():
        raise ValueError("Skinned vertices must be finite floating-point metres")
    selected = torch.as_tensor(np.asarray(indices), dtype=torch.long, device=vertices_m.device)
    if selected.numel() == 0 or int(selected.max()) >= vertices_m.shape[1]:
        raise ValueError("Region indices must select vertices that exist in the mesh")
    return vertices_m.index_select(1, selected)[..., 1] - ground_height_m


def region_gap_m(vertex_height_m):
    """Lowest region vertex above the ground, and its penetration, per frame."""
    if vertex_height_m.ndim != 2:
        raise ValueError("Region heights must be [frames, vertices]")
    lowest = vertex_height_m.amin(dim=-1)
    return {"region_gap_m": lowest, "region_penetration_m": torch.relu(-lowest)}


def contact_runs(in_contact):
    """Maximal runs of consecutive true frames per vertex, as -1 or a run id."""
    if in_contact.dtype != torch.bool or in_contact.ndim != 2:
        raise ValueError("A boolean [frames, vertices] contact mask is required")
    frames, vertices = in_contact.shape
    runs = torch.full((frames, vertices), -1, dtype=torch.long, device=in_contact.device)
    next_id = 0
    for vertex in range(vertices):
        column = in_contact[:, vertex].tolist()
        start = None
        for frame, flag in enumerate(column + [False]):
            if flag and start is None:
                start = frame
            elif not flag and start is not None:
                runs[start:frame, vertex] = next_id
                next_id += 1
                start = None
    return runs


def material_point_slip(vertex_height_m, vertex_horizontal_m, fps, *,
                        contact_clearance_m=SOLE_CONTACT_CLEARANCE_M,
                        min_run_frames=SOLE_MIN_CONTACT_RUN_FRAMES,
                        rest_vertical_speed_mps=SOLE_REST_VERTICAL_SPEED_MPS):
    """Slip of the same numbered material vertices while they stay in contact.

    A vertex is in contact on a frame when its own height is at or below the contact
    clearance. Slip is only measured inside a contact run that is long enough and
    vertically at rest: a foot descending into or rising out of contact passes through
    the band while still translating fast, and that is a landing or a take-off, not a
    skate. A single-frame dip during swing cannot be read as a skate either, and a
    heel-to-toe roll simply ends one vertex's run and starts another's.
    """
    if vertex_height_m.shape != vertex_horizontal_m.shape[:2] or vertex_horizontal_m.shape[-1] != 2:
        raise ValueError("Heights [frames, vertices] and horizontal positions [frames, vertices, xz] are required")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("A finite positive FPS is required")
    if min_run_frames < 2:
        raise ValueError("A contact run needs at least two frames to carry a displacement")
    if not np.isfinite(rest_vertical_speed_mps) or rest_vertical_speed_mps < 0:
        raise ValueError("A finite nonnegative rest vertical speed is required")
    frames = vertex_height_m.shape[0]
    in_contact = vertex_height_m <= contact_clearance_m
    runs = contact_runs(in_contact)
    step = (vertex_horizontal_m[1:] - vertex_horizontal_m[:-1]).norm(dim=-1) * fps
    # A landing must not veto the resting stretch that follows it. The resting
    # condition is judged interval by interval, and a contact run is cut into maximal
    # sub-runs of consecutive resting intervals. A whole run is never accepted or
    # rejected as a unit.
    vertical = (vertex_height_m[1:] - vertex_height_m[:-1]).abs() * fps
    resting_interval = in_contact[1:] & in_contact[:-1] & (vertical <= rest_vertical_speed_mps)
    sub_runs = contact_runs(resting_interval)
    n_sub = int(sub_runs.max()) + 1 if sub_runs.numel() and int(sub_runs.max()) >= 0 else 0
    sub_lengths = torch.zeros(n_sub, dtype=torch.long, device=sub_runs.device)
    if n_sub:
        sub_lengths = torch.bincount(sub_runs[sub_runs >= 0], minlength=n_sub)
    long_enough = torch.zeros_like(resting_interval)
    if n_sub:
        long_enough = resting_interval & (sub_lengths[sub_runs] >= min_run_frames)
    measured = torch.where(long_enough, step, torch.zeros_like(step))
    return {
        "contact_run_id": runs,
        "vertex_in_contact": in_contact,
        "vertex_slip_mps": measured,
        "vertex_measured": long_enough,
        "region_slip_mps": measured.amax(dim=-1),
        "region_measured": long_enough.any(dim=-1),
        "measured_vertex_count": long_enough.sum(dim=-1),
        "resting_interval": resting_interval,
        "resting_sub_run_id": sub_runs,
        "resting_sub_run_lengths": sub_lengths,
        "vertex_vertical_speed_mps": vertical,
    }


def sole_region_labels(region_gap, slip, *, contact_clearance_m=SOLE_CONTACT_CLEARANCE_M,
                       clearance_m=SOLE_CLEARANCE_M, slip_speed_mps=SOLE_SLIP_SPEED_MPS):
    """Candidate per-frame and per-interval contact labels for one foot.

    Frame states: contact, undecided (the band between the two clearances) and
    clearance. Interval states keep contact transitions apart from sustained contact.
    A transition needs both endpoints decided, so an interval that touches the
    undecided band stays undecided.

    A contact interval with no valid slip measurement is undecided as well. Zero slip
    and unmeasured slip are different things, and only a measured interval may be
    called stable contact. Nothing here converts an undecided frame, or an interval
    without evidence, into contact or into flight.
    """
    if contact_clearance_m >= clearance_m:
        raise ValueError("The contact clearance must be below the clearance that marks flight")
    gap = region_gap
    contact = gap <= contact_clearance_m
    clear = gap > clearance_m
    undecided = ~contact & ~clear
    frame_codes = torch.where(contact, torch.zeros_like(gap, dtype=torch.long),
                              torch.where(clear, torch.full_like(gap, 2, dtype=torch.long),
                                          torch.ones_like(gap, dtype=torch.long)))
    both_contact = contact[1:] & contact[:-1]
    both_clear = clear[1:] & clear[:-1]
    # A transition needs both endpoints decided. An interval that touches the
    # undecided band stays undecided rather than being read as leaving contact.
    decided = ~undecided
    transition = (contact[1:] ^ contact[:-1]) & decided[1:] & decided[:-1]
    measured = slip.get("region_measured")
    if measured is None:
        raise ValueError("Slip must carry a region_measured mask; zero slip and unmeasured slip differ")
    if measured.shape != both_contact.shape or measured.dtype != torch.bool:
        raise ValueError("The region_measured mask must be boolean and cover every interval")
    sliding = both_contact & measured & (slip["region_slip_mps"] > slip_speed_mps)
    stable = both_contact & measured & ~sliding
    interval_codes = torch.full((max(int(gap.shape[0]) - 1, 0),), 4, dtype=torch.long, device=gap.device)
    interval_codes = torch.where(both_clear, torch.full_like(interval_codes, 3), interval_codes)
    interval_codes = torch.where(transition, torch.full_like(interval_codes, 2), interval_codes)
    interval_codes = torch.where(stable, torch.zeros_like(interval_codes), interval_codes)
    interval_codes = torch.where(sliding, torch.ones_like(interval_codes), interval_codes)
    return {
        "frame_state": frame_codes,
        "frame_state_names": FRAME_STATES,
        "interval_state": interval_codes,
        "interval_state_names": INTERVAL_STATES,
        "measured_interval": measured,
        "frame_counts": {name: int((frame_codes == index).sum())
                         for index, name in enumerate(FRAME_STATES)},
        "interval_counts": {name: int((interval_codes == index).sum())
                            for index, name in enumerate(INTERVAL_STATES)},
    }
