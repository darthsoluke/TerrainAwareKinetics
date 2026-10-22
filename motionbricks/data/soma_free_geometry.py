"""Observed near-ground geometry, never inferred support/force ground truth."""

import torch

from motionbricks.data.sole_region_contact import SOLE_LABEL_THRESHOLDS, material_point_slip


def distribution(values):
    values = values.detach().double().flatten()
    if not values.numel():
        return None
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite free-motion geometry")
    return {"n": values.numel(), "mean": float(values.mean()), "p95": float(values.quantile(.95)),
            "max": float(values.max()), "min": float(values.min())}


def observed_sole_geometry(vertices, fps, context_frames=4):
    """One fixed material-vertex region in world metres above unchanged y=0."""
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or vertices.shape[0] <= context_frames + 1:
        raise ValueError("Need genuine [frames, material vertices, xyz] after the context")
    if not torch.isfinite(vertices).all() or context_frames < 0:
        raise ValueError("Invalid sole vertices/context")
    heights, horizontal = vertices[..., 1], vertices[..., [0, 2]]
    slip = material_point_slip(heights, horizontal, fps,
        contact_clearance_m=SOLE_LABEL_THRESHOLDS["contact_clearance_m"],
        min_run_frames=SOLE_LABEL_THRESHOLDS["min_contact_run_frames"],
        rest_vertical_speed_mps=SOLE_LABEL_THRESHOLDS["rest_vertical_speed_mps"])
    gap = heights.amin(-1)[context_frames:]
    measured = slip["region_measured"][context_frames:]
    measured_speed = slip["region_slip_mps"][context_frames:][measured]
    return {"region_gap_m": distribution(gap), "penetration_m": distribution(torch.relu(-gap)),
        "near_ground_material_speed_mps": distribution(measured_speed),
        "measured_intervals": int(measured.sum()), "scored_intervals": len(measured),
        "measured_interval_fraction": float(measured.double().mean()),
        "near_ground_frame_fraction": float((gap <= SOLE_LABEL_THRESHOLDS["contact_clearance_m"]).double().mean()),
        "clearance_frame_fraction": float((gap > SOLE_LABEL_THRESHOLDS["clearance_m"]).double().mean()),
        "interpretation": "All measurable near-ground runs, including fast motion; NOT selected by low tangential speed. Geometry is not proof of support or true sliding.",
        "no_measurement_is_zero_slip": False, "ground_y_m": 0.0, "thresholds": dict(SOLE_LABEL_THRESHOLDS)}
