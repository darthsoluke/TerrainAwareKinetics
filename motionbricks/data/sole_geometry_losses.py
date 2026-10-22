"""Differentiable sole-region geometry supervision from the candidate labels.

The supervised surface is the sole region, evaluated with the official skinned
implementation rather than an approximation: local rotations are padded from SOMA30 to
the 77-joint rig, forward kinematics runs on that rig, and the official linear blend
skinning produces the vertex positions. Each region vertex therefore keeps its own
skinning weights, so a toe joint that rotates moves the vertices it actually drives.

Rules enforced here rather than left to the caller:

* Only a measured stable-contact interval supervises low slip, and inside it only the
  vertices that carry their own measurement. A lifted part of the sole is never
  penalised for moving.
* The contact-height term uses the same measured stable-contact evidence. Frames whose
  intervals are all undecided are not pinned to the floor.
* The masks come from the source labels and are detached, so a prediction can never
  choose which frames or which vertices it is judged on.

Every term keeps its own unit and they are returned separately; nothing here sums
metres and metres per second into one number.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from motionbricks.data.sole_region_contact import FRAME_STATES, INTERVAL_STATES, sole_region_indices

try:  # pragma: no cover - import shape depends on the vendored layout
    from kimodo.skeleton import batch_rigid_transform
except ImportError:  # pragma: no cover
    from kimodo.skeleton.kinematics import batch_rigid_transform


def region_geometry_from_motion(local_rot_mats, root_positions, skinner, region_indices,
                                ground_height_m=0.0):
    """Official differentiable skinning of the sole region, [frames, vertices].

    local_rot_mats is the predicted SOMA30 local rotation [frames, 30, 3, 3] and
    root_positions is [frames, 3]. Returns the region vertex heights above the fixed
    ground and their horizontal positions, both differentiable with respect to the
    motion. Vertices keep their own skinning weights, so foot and toe joints act on the
    vertices they actually drive.
    """
    if local_rot_mats.ndim != 4 or local_rot_mats.shape[-3:] != (30, 3, 3):
        raise ValueError("Local rotations must be [frames, 30, 3, 3]")
    if root_positions.shape != (local_rot_mats.shape[0], 3):
        raise ValueError("Root positions must be [frames, 3]")
    rig = skinner.skeleton_skin
    device, dtype = local_rot_mats.device, local_rot_mats.dtype
    local77 = skinner.skeleton_input.to_SOMASkeleton77(local_rot_mats)
    neutral = rig.neutral_joints[None].repeat(local77.shape[0], 1, 1).to(device=device, dtype=dtype)
    posed_noroot, global77 = batch_rigid_transform(local77, neutral,
                                                   rig.joint_parents.cpu(), rig.root_idx)
    joint_positions = posed_noroot + root_positions[:, None, :]
    transform = torch.eye(4, dtype=dtype, device=device)[None, None].repeat(
        local77.shape[0], rig.nbjoints, 1, 1)
    transform[..., :3, :3] = global77
    transform[..., :3, 3] = joint_positions
    if isinstance(region_indices, torch.Tensor):
        # Model buffers arrive on the compute device; never round-trip them through numpy.
        indices = region_indices.to(device=device, dtype=torch.long)
    else:
        indices = torch.as_tensor(np.asarray(region_indices), dtype=torch.long, device=device)
    # LBS is independent per vertex. Select its inputs, then run the unchanged
    # official implementation instead of materializing the full 18k-vertex mesh.
    selected_skinner = copy.copy(skinner)
    for name in ("bind_vertices", "lbs_indices", "lbs_weights"):
        setattr(selected_skinner, name, getattr(skinner, name).index_select(0, indices))
    region = selected_skinner.lbs(transform)
    return region[..., 1] - ground_height_m, region[..., [0, 2]]


def stance_displacement_loss(horizontal, interval_state, region_measured, vertex_measured,
                             horizons=(4, 8, 16)):
    """Metres of drift across uninterrupted measured stance, for each material point.

    A rolling sole may change its supporting vertex. Every intervening interval must
    contain evidence for this same vertex; endpoints alone never bridge a swing.
    This is a training penalty, not an inference foot lock or an oracle anchor.
    """
    stable = interval_state.detach() == INTERVAL_STATES.index("stable_contact")
    eligible = vertex_measured.detach().bool() & region_measured.detach().bool()[..., None] & stable[..., None]
    invalid_prefix = F.pad((~eligible).long().cumsum(dim=-2), (0, 0, 1, 0))
    total, count = horizontal.sum() * 0.0, 0
    for horizon in horizons:
        if horizon < 1:
            raise ValueError("Stance horizons must be positive")
        if horizon >= horizontal.shape[-3]:
            continue
        valid = (invalid_prefix[..., horizon:, :] - invalid_prefix[..., :-horizon, :]) == 0
        drift = (horizontal[..., horizon:, :, :] - horizontal[..., :-horizon, :, :]).norm(dim=-1)
        total = total + (drift * valid).sum()
        count += int(valid.sum())
    return total / max(count, 1), count


def frame_has_evidence(measured):
    """Frames covered by at least one measured interval, shaped like the frame axis."""
    if measured.dtype != torch.bool:
        raise ValueError("The measured mask must be boolean")
    padded = F.pad(measured, (1, 1), value=False)
    return padded[..., :-1] | padded[..., 1:]


def sole_geometry_supervision(predicted_height_m, predicted_horizontal_m, labels, fps, *,
                              penetration_tolerance_m=0.005, frame_mask=None):
    """Named, unit-carrying losses from the candidate labels.

    labels carries frame_state and interval_state, the per-interval region_measured
    mask and the per-interval, per-vertex vertex_measured mask, all from the source
    motion. Only measured stable-contact intervals supervise slip, and inside them only
    the measured vertices. Only contact frames that sit inside a measured interval
    supervise contact height.
    """
    if fps <= 0:
        raise ValueError("A positive FPS is required")
    frame_state = labels["frame_state"].detach()
    interval_state = labels["interval_state"].detach()
    measured = labels["region_measured"].detach()
    vertex_measured = labels["vertex_measured"].detach()
    if frame_mask is not None:
        if frame_mask.dtype != torch.bool or frame_mask.shape != frame_state.shape:
            raise ValueError("Geometry frame mask must match source frame labels")
        frame_mask = frame_mask.detach()
        interval_mask = frame_mask[..., :-1] & frame_mask[..., 1:]
        measured = measured & interval_mask
        vertex_measured = vertex_measured & interval_mask[..., None]
    if measured.dtype != torch.bool or vertex_measured.dtype != torch.bool:
        raise ValueError("The measured masks must be boolean")
    frames = frame_state.shape[-1]
    if predicted_height_m.shape[-2] != frames or predicted_horizontal_m.shape[-3] != frames:
        raise ValueError("Predicted geometry must share the label frame axis")
    if vertex_measured.shape[-2] != frames - 1 or vertex_measured.shape[-1] != predicted_height_m.shape[-1]:
        raise ValueError("The vertex measured mask must be [..., frames-1, vertices]")

    lowest = predicted_height_m.amin(dim=-1)
    stable = interval_state == INTERVAL_STATES.index("stable_contact")
    supervised_slip = stable & measured
    step = (predicted_horizontal_m[..., 1:, :, :] - predicted_horizontal_m[..., :-1, :, :]).norm(dim=-1) * fps
    # A lifted part of the sole is allowed to move: only vertices with their own
    # measurement inside a measured stable interval carry slip.
    per_vertex = torch.where(vertex_measured, step, torch.zeros_like(step))
    region_slip = per_vertex.amax(dim=-1)

    contact = frame_state == FRAME_STATES.index("contact")
    trusted_contact = contact & frame_has_evidence(measured)
    stable_vertex_count = vertex_measured & supervised_slip[..., None]

    def masked_mean(values, mask):
        count = mask.sum()
        if int(count) == 0:
            return values.sum() * 0.0, 0
        weights = mask.to(values.dtype)
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        return (values * weights).sum() / count.clamp(min=1), int(count)

    contact_height, contact_frames = masked_mean(lowest.abs(), trusted_contact)
    penetration, _ = masked_mean(F.relu(-lowest - penetration_tolerance_m),
                                 torch.ones_like(lowest, dtype=torch.bool) if frame_mask is None else frame_mask)
    slip, slip_intervals = masked_mean(region_slip, supervised_slip)
    return {
        "l_contact_height_m": contact_height,
        "l_penetration_m": penetration,
        "l_measured_slip_mps": slip,
        "contact_frames_supervised": contact_frames,
        "measured_stable_intervals_supervised": slip_intervals,
        "measured_vertices_supervised": int(stable_vertex_count.sum()),
        "units": {"l_contact_height_m": "m", "l_penetration_m": "m", "l_measured_slip_mps": "m/s"},
        "note": "Only measured stable-contact intervals supervise slip, and inside them only measured vertices. "
                "Contact height uses the same measured evidence, so undecided frames are never pinned to the floor. "
                "The masks are detached source labels; a prediction cannot choose its own supervision.",
    }
