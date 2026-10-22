"""Offline flat-floor contact review of saved SOMA tokenizer review windows.

The inputs are tokenizer reconstruction windows: the M1 pose tokenizer encoded the
recorded source motion and decoded it back. They are not WASD input, not root-only
generated motion, and no generative policy produced them. Source contact labels and
confidence are read from the same file and drive both sides of the comparison.

Three statuses are recorded separately and never merged:

* execution_status  - did this window process at all
* motion_quality_status - do the kinematic screening criteria pass
* physics_check_status  - which physical checks could actually run

A processed window is not a good motion, and a passing physics check is not a
validated motion. The resultant-force check only falsifies: it reports the ground
force that would be required to explain the observed centre of mass, so a zero
residual means "not contradicted by the stated assumptions" and never "verified".
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_seed import (SOLE_PROXY_GEOMETRY, SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET,
                                          SOMA_REVISION, build_representation, official_sole_offsets,
                                          sole_positions)
from motionbricks.data.contact_forces import (AIRBORNE_FORCE_TOLERANCE_MPS2, CONTACT_STATES,
                                              ground_force_feasibility, phase_interior_mask,
                                              required_ground_force_per_mass, state_codes,
                                              violation_counts)
from motionbricks.data.contact_mechanics import (body_contact_state, flat_ground_contact_kinematics,
                                                 masked_summary, support_frame_mask,
                                                 support_interval_masks, sustained_support_segments)
from motionbricks.data.soma_mass_model import PROVENANCE as MASS_PROVENANCE
from motionbricks.data.soma_mass_model import SEGMENTS, whole_body_center_of_mass

CHANNEL_LAYOUT = (("left", "heel", 0, "LeftFoot"), ("left", "forefoot", 1, "LeftToeBase"),
                  ("right", "heel", 2, "RightFoot"), ("right", "forefoot", 3, "RightToeBase"))
SEGMENT_KEYS = ("path_length_m", "net_drift_m", "max_anchor_m", "normal_range_m", "normal_net_m")
COUNT_KEYS = ("frames_scored", "frames_support", "frames_swing_labeled", "frames_unknown", "intervals",
              "intervals_sustained_support", "intervals_transition", "intervals_swing", "intervals_unknown",
              "sustained_support_segments")
STATISTICS = ("mean", "minimum", "p95", "peak")
SIDES = ("left", "right")
GROUP_NAMES = ("left_heel", "left_forefoot", "right_heel", "right_forefoot", "left", "right", "all_channels")
DEFAULT_INPUT = Path("out/bones_pilot_reports/m1-v7-support-repair/review-validation")
DEFAULT_OUTPUT = Path("out/contact_mechanics_review")
DEFAULT_EXPECTED_BEHAVIORS = "walk,run,turn,idle,backward"
DEFAULT_FRICTION_COEFFICIENTS = "0.3,0.6,0.9"
SMOOTHING_WINDOW_FRAMES = 7
SMOOTHING_POLYORDER = 3
MOTION_QUALITY_CRITERIA = (
    {"name": "support_tangential_speed_p95_mps", "quantity": "support_tangential_speed_mps",
     "statistic": "p95", "unit": "m/s", "source_factor": 2.0, "absolute_floor": 0.02},
    {"name": "sustained_support_segment_net_drift_peak_m", "quantity": "segment_net_drift_m",
     "statistic": "peak", "unit": "m", "source_factor": 2.0, "absolute_floor": 0.01},
    {"name": "sole_penetration_peak_m", "quantity": "penetration_m",
     "statistic": "peak", "unit": "m", "source_factor": 2.0, "absolute_floor": 0.005},
)
RESIDUAL_STATES = (("ground_pull_mps2", "evaluated_contact_frames"),
                   ("friction_capacity_excess_mps2", "evaluated_contact_frames"),
                   ("airborne_force_magnitude_mps2", "evaluated_airborne_frames"))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_window(path):
    """Read one saved review window and check the shapes the protocol relies on."""
    required = ("pred_positions_m", "source_positions_m", "pred_local_rot_mats", "source_local_rot_mats",
                "pred_contacts", "source_contacts", "source_confidence", "fps")
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [name for name in required if name not in data]
            if missing:
                raise ValueError(f"{path.name} is missing {missing}")
            window = {name: data[name] for name in required}
    except Exception as error:  # noqa: BLE001 - a corrupt window must be reported, not crash the run
        raise ValueError(f"{path.name} could not be read as a review window: {error}") from error
    window["source_local_rot_mats"] = window["source_local_rot_mats"].astype(np.float32)
    window["pred_local_rot_mats"] = window["pred_local_rot_mats"].astype(np.float32)
    frames = window["source_positions_m"].shape[0]
    for name in ("pred_positions_m", "source_positions_m"):
        if window[name].shape != (frames, 30, 3):
            raise ValueError(f"{path.name}: {name} must be [frames, 30, 3]")
    for name in ("pred_local_rot_mats", "source_local_rot_mats"):
        if window[name].shape != (frames, 30, 3, 3):
            raise ValueError(f"{path.name}: {name} must be [frames, 30, 3, 3]")
    for name in ("pred_contacts", "source_contacts", "source_confidence"):
        if window[name].shape != (frames, 4) or window[name].dtype != np.bool_:
            raise ValueError(f"{path.name}: {name} must be boolean [frames, 4]")
    fps = float(window["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path.name}: a finite positive FPS must be stored in the window")
    window["fps"] = fps
    return window


def rebuild_soles(window, skeleton, offsets):
    """Official FK from the saved local rotations and root position, then sole proxies."""
    rebuilt, fk_error = {}, {}
    for tag in ("source", "pred"):
        root = torch.from_numpy(window[f"{tag}_positions_m"][:, skeleton.root_idx].copy())
        local = torch.from_numpy(window[f"{tag}_local_rot_mats"])
        global_rots, posed, _ = skeleton.fk(local, root)
        fk_error[tag] = float((posed - torch.from_numpy(window[f"{tag}_positions_m"])).norm(dim=-1).max())
        rebuilt[tag] = {"posed": posed, "rotations": global_rots,
                        "soles": sole_positions(posed, global_rots, skeleton, offsets)}
    return rebuilt, fk_error


def channel_quantities(sole_m, gap_m, penetration_m, tangential_speed, normal_velocity, masks, support, channel):
    """Raw per-channel samples for one clip; quantiles are pooled downstream."""
    sustained = masks["sustained_support"][:, channel]
    segments = sustained_support_segments(sole_m, gap_m, masks["sustained_support"])[channel]
    quantities = {
        "gap_m": gap_m[:, channel],
        "penetration_m": penetration_m[:, channel],
        "support_gap_m": gap_m[support[:, channel], channel],
        "support_tangential_speed_mps": tangential_speed[sustained, channel],
        "support_normal_speed_mps": normal_velocity[sustained, channel].abs(),
        "support_normal_velocity_mps": normal_velocity[sustained, channel],
    }
    for key in SEGMENT_KEYS:
        quantities[f"segment_{key}"] = torch.as_tensor([record[key] for record in segments],
                                                       dtype=torch.float64)
    counts = {
        "frames_scored": int(gap_m.shape[0]),
        "frames_support": int(support[:, channel].sum()),
        "frames_unknown": int((~masks["frame_confident"])[:, channel].sum()),
        "intervals": int(tangential_speed.shape[0]),
        "intervals_sustained_support": int(sustained.sum()),
        "intervals_transition": int(masks["transition"][:, channel].sum()),
        "intervals_swing": int(masks["swing"][:, channel].sum()),
        "intervals_unknown": int(masks["unknown"][:, channel].sum()),
        "sustained_support_segments": len(segments),
    }
    counts["frames_swing_labeled"] = (counts["frames_scored"] - counts["frames_support"]
                                      - counts["frames_unknown"])
    return quantities, counts, segments


def pool_quantities(by_channel, channels):
    pooled = {}
    for channel in channels:
        for name, values in by_channel[channel].items():
            pooled.setdefault(name, []).append(values)
    return {name: torch.cat(parts) for name, parts in pooled.items()}


def summarize(quantities):
    return {name: masked_summary(values) for name, values in quantities.items()}


def delta_summary(source, reconstruction):
    """Reconstruction minus source for every scalar statistic, to locate differences."""
    delta = {}
    for name in sorted(set(source) & set(reconstruction)):
        if not isinstance(source[name], dict) or not isinstance(reconstruction[name], dict):
            continue
        for statistic in STATISTICS:
            left, right = source[name].get(statistic), reconstruction[name].get(statistic)
            delta[f"{name}_{statistic}"] = None if left is None or right is None else right - left
    return delta


def sum_counts(counts_list):
    return {key: sum(counts[key] for counts in counts_list) for key in COUNT_KEYS}


def group_report(by_channel, channels, counts_list):
    report = {"channels_pooled": len(channels), "counts_from_source_labels": sum_counts(counts_list)}
    for tag in ("source", "pred"):
        report[tag] = summarize(pool_quantities({c: by_channel[tag][c] for c in channels}, channels))
    report["delta_reconstruction_minus_source"] = delta_summary(report["source"], report["pred"])
    return report


def unavailable_point_contact():
    return {
        "status": "unavailable",
        "reason": "the saved review windows contain no contact force data",
        "policy": "A normal load must come from an identified force source. Fn = m*g, an even split of one load "
                  "across the four contact proxies, and zero-filling a missing force are all forbidden here.",
        "scope": "Per-point Coulomb checks only. The separate resultant-force check below does not need a per-point "
                 "load, but it also does not measure one.",
    }


def coverage_report(contacts, confidence, masks, support, gap_m):
    """Effective coverage and unknown share, with channel and whole-body denominators kept apart.

    Channel rows count one entry per channel per frame or interval and are the correct
    denominator for per-foot statistics. Whole-body rows count one entry per frame and
    describe the body state, which is a different quantity. Dividing a four-channel
    total by a single-channel count is what made an all-unknown window report negative
    coverage before.
    """
    frames, channels = int(contacts.shape[0]), int(contacts.shape[1])
    intervals = frames - 1
    state = body_contact_state(contacts, confidence, gap_m)
    codes = state_codes(state)
    unknown_code = CONTACT_STATES.index("unknown")
    channel_frames = {
        "scored": frames * channels,
        "labeled_support": int(contacts.bool().sum()),
        "labeled_support_confident": int(support.sum()),
        "unknown": int((~confidence.bool()).sum()),
    }
    channel_frames["labeled_swing_confident"] = (channel_frames["scored"]
                                                 - channel_frames["labeled_support_confident"]
                                                 - channel_frames["unknown"])
    channel_intervals = {"scored": intervals * channels}
    channel_intervals.update({name: int(masks[name].sum()) for name in
                              ("sustained_support", "transition", "swing", "unknown")})
    whole_frames = {name: int(state[name].sum()) for name in CONTACT_STATES}
    whole_intervals = {"scored": intervals}
    whole_intervals.update({name: int(((codes[1:] == index) & (codes[:-1] == index)).sum())
                            for index, name in enumerate(CONTACT_STATES)})
    whole_intervals["state_change"] = intervals - sum(whole_intervals[name] for name in CONTACT_STATES)
    whole_known = int(((codes[1:] != unknown_code) & (codes[:-1] != unknown_code)).sum())
    return {
        "frames_scored": frames,
        "intervals_scored": intervals,
        "channels": channels,
        "channel_frames": channel_frames,
        "channel_frame_known_fraction": (channel_frames["scored"] - channel_frames["unknown"])
                                        / channel_frames["scored"],
        "channel_intervals": channel_intervals,
        "channel_interval_known_fraction": (channel_intervals["scored"] - channel_intervals["unknown"])
                                           / channel_intervals["scored"],
        "whole_body_frames": whole_frames,
        "whole_body_frame_known_fraction": 1 - whole_frames["unknown"] / frames,
        "whole_body_intervals": whole_intervals,
        "whole_body_interval_known_fraction": whole_known / intervals,
        "definitions": {
            "channel_rows": "one entry per channel per frame or interval; the denominator for per-foot statistics",
            "whole_body_rows": "one entry per frame; the body state combines all channels and is a different quantity",
            "whole_body_interval_states": "an interval takes a state name only when both endpoints share it, "
                                          "otherwise it is a state_change",
            "whole_body_interval_known_fraction": "fraction of intervals whose both endpoints are a known state",
        },
        "note": "Unknown is reported on its own, never as support and never as airborne.",
    }


def motion_quality(group_source, group_pred, criteria=MOTION_QUALITY_CRITERIA):
    """Kinematic screening against declared thresholds, relative to the source window.

    This is a screen, not an approval. Each criterion is compared with the same
    window's source value scaled by source_factor, with an absolute floor so that a
    near-zero source does not make the bar meaningless.
    """
    details, failed, unassessed = {}, [], []
    for criterion in criteria:
        source = group_source.get(criterion["quantity"], {}).get(criterion["statistic"])
        reconstruction = group_pred.get(criterion["quantity"], {}).get(criterion["statistic"])
        if source is None or reconstruction is None:
            details[criterion["name"]] = {"status": "not_assessed", "unit": criterion["unit"],
                                          "source": source, "reconstruction": reconstruction}
            unassessed.append(criterion["name"])
            continue
        limit = max(source * criterion["source_factor"], criterion["absolute_floor"])
        passed = reconstruction <= limit
        details[criterion["name"]] = {
            "status": "pass" if passed else "fail", "unit": criterion["unit"],
            "source": source, "reconstruction": reconstruction, "limit": limit,
            "criterion": "reconstruction <= max(source * factor, absolute floor)",
            "source_factor": criterion["source_factor"], "absolute_floor": criterion["absolute_floor"],
        }
        if not passed:
            failed.append(criterion["name"])
    if failed:
        status = "fail"
    elif unassessed:
        status = "not_assessed"
    else:
        status = "pass"
    return {"status": status, "failed_criteria": failed, "details": details,
            "not_an_approval": "Screening thresholds only. Passing them is not motion approval."}


def resultant_force_check(rebuilt, skeleton, fps, contacts, confidence, gaps, friction_coefficients):
    """Required resultant ground force per unit mass, and its feasibility residuals.

    Two corrections keep this from manufacturing violations. The derivative edge
    frames are excluded from every statistic, and a frame is evaluated only when its
    whole smoothing window lies inside one contact state, so a window spanning a
    take-off or a landing never produces a verdict about either phase.
    """
    tags = ("source", "pred")
    frames = int(contacts.shape[0])
    if frames < max(SMOOTHING_WINDOW_FRAMES, 9):
        return {"status": "not_assessed", "reason": "the scored clip is too short for a phased derivative"}
    state = {tag: body_contact_state(contacts, confidence, gaps[tag]) for tag in tags}
    codes = {tag: state_codes(state[tag]) for tag in tags}
    interior = {tag: phase_interior_mask(codes[tag], SMOOTHING_WINDOW_FRAMES) for tag in tags}
    contact = {tag: state[tag]["stable_support"] & interior[tag] for tag in tags}
    flight = {tag: state[tag]["airborne"] & interior[tag] for tag in tags}
    demand, centre_of_mass = {}, {}
    for tag in tags:
        centre_of_mass[tag] = whole_body_center_of_mass(rebuilt[tag]["posed"], skeleton)
        demand[tag] = required_ground_force_per_mass(centre_of_mass[tag], fps,
                                                     window_frames=SMOOTHING_WINDOW_FRAMES,
                                                     polyorder=SMOOTHING_POLYORDER)
    report = {
        "status": "assessed_resultant_only",
        "measured_quantity": "F_ground / m = c_ddot - g, per unit mass, in m/s^2 (equivalently N/kg)",
        "not_a_measurement": "This is the resultant the ground would have to exert to explain the observed centre of "
                             "mass. It is not a measured force, the mass cancels so no actor weight is assumed, and a "
                             "zero residual means not-contradicted, never verified.",
        "centre_of_mass_model": {"segments": len(SEGMENTS), "provenance": MASS_PROVENANCE},
        "smoothing": {"filter": "Savitzky-Golay", "window_frames": SMOOTHING_WINDOW_FRAMES,
                      "polyorder": SMOOTHING_POLYORDER,
                      "time_alignment": "the derivative at frame t comes from the window centred on t and stays "
                                        "index-aligned with the frames"},
        "evaluation_filter": {
            "derivative_edge_frames_excluded": True,
            "phase_interior_requirement": "every frame within "
                                          + str(SMOOTHING_WINDOW_FRAMES // 2)
                                          + " frames of an evaluated frame must share its whole-body contact state",
            "airborne_tolerance_mps2": AIRBORNE_FORCE_TOLERANCE_MPS2,
            "why": "A derivative window spanning support and flight mixes two force regimes, a bare zero threshold "
                   "flags the float32 noise floor as a violation, and edge frames are one-sided extrapolations.",
        },
        "frame_state": {tag: {name: int(state[tag][name].sum()) for name in CONTACT_STATES} for tag in tags},
        "phase_interior_frames": {tag: int(interior[tag].sum()) for tag in tags},
        "evaluated_contact_frames": {tag: int(contact[tag].sum()) for tag in tags},
        "evaluated_airborne_frames": {tag: int(flight[tag].sum()) for tag in tags},
        "friction_coefficients": list(friction_coefficients),
        "per_friction_coefficient": {},
    }
    for tag in tags:
        report.setdefault("edge_frames", {})[tag] = int(demand[tag]["acceleration_edge"].sum())
        report.setdefault("contact_normal_force_per_mass_mps2", {})[tag] = masked_summary(
            demand[tag]["required_force_per_mass_mps2"][..., 1], contact[tag])
        report.setdefault("horizontal_force_per_mass_mps2", {})[tag] = masked_summary(
            demand[tag]["required_force_per_mass_mps2"][..., [0, 2]].norm(dim=-1), contact[tag])
    for coefficient in friction_coefficients:
        entry = {}
        for tag in tags:
            residual = ground_force_feasibility(demand[tag]["required_force_per_mass_mps2"], state[tag],
                                                static_friction=coefficient, evaluated=interior[tag])
            entry[tag] = {name: masked_summary(residual[name], residual[state_name])
                          for name, state_name in RESIDUAL_STATES}
            entry[tag]["violations"] = violation_counts(residual)
        report["per_friction_coefficient"][str(coefficient)] = entry
    middle = friction_coefficients[len(friction_coefficients) // 2]
    report["filter_sensitivity"] = {
        "headline_window_frames": SMOOTHING_WINDOW_FRAMES,
        "windows_checked": [5, 9],
        "friction_coefficient_used": middle,
        "purpose": "Check that a violation is a property of the motion and not of the smoothing width. Each width "
                   "re-erodes the phase-interior mask, so an alternative width is judged on its own terms.",
        "per_window": {},
    }
    for window in (5, 9):
        entry = {}
        alternative_interior = {tag: phase_interior_mask(codes[tag], window) for tag in tags}
        for tag in tags:
            alternative = required_ground_force_per_mass(centre_of_mass[tag], fps, window_frames=window,
                                                         polyorder=SMOOTHING_POLYORDER)
            residual = ground_force_feasibility(alternative["required_force_per_mass_mps2"], state[tag],
                                                static_friction=middle, evaluated=alternative_interior[tag])
            counts = violation_counts(residual)
            counts["evaluated_airborne_frames"] = int(residual["evaluated_airborne_frames"].sum())
            counts["airborne_force_mean_mps2"] = masked_summary(
                residual["airborne_force_magnitude_mps2"], residual["evaluated_airborne_frames"])["mean"]
            entry[tag] = counts
        report["filter_sensitivity"]["per_window"][str(window)] = entry
    report["falsifies_only"] = "A violation falsifies the motion under these assumptions. Passing does not confirm it."
    return report


def review_window(path, behavior, skeleton, offsets, context_frames, ground_height_m, friction_coefficients):
    """Contact review of one saved tokenizer reconstruction window."""
    record = {"behavior": behavior, "input_file": str(path),
              "input_sha256": file_sha256(path) if Path(path).exists() else None}
    try:
        window = load_window(path)
        frames = window["source_positions_m"].shape[0]
        if frames - context_frames < 3:
            raise ValueError(f"{path.name}: fewer than three scored frames after the context window")
        rebuilt, fk_error = rebuild_soles(window, skeleton, offsets)
    except (ValueError, OSError, KeyError) as error:
        record.update({"execution_status": "failed", "execution_detail": str(error),
                       "motion_quality_status": "not_assessed", "physics_check_status": "not_assessed",
                       "motion_quality": None, "resultant_force_check": None,
                       "point_contact_check": unavailable_point_contact()})
        return record, None
    record.update({
        "fps": window["fps"], "frames_total": frames, "context_frames_skipped": context_frames,
        "frames_scored": frames - context_frames, "fk_rebuild_max_error_m": fk_error,
        "fk_rebuild_tolerance_m": 1e-5,
        "source_motion": "recorded SOMA30 source motion",
        "reconstruction_motion": "M1 tokenizer reconstruction of the source window; not WASD and not root-only generated",
    })
    if max(fk_error.values()) > record["fk_rebuild_tolerance_m"]:
        record.update({"execution_status": "failed",
                       "execution_detail": "saved local rotations and root position did not reproduce the saved joint "
                                           "positions, so sole reconstruction is not trusted",
                       "motion_quality_status": "not_assessed", "physics_check_status": "not_assessed",
                       "motion_quality": None, "resultant_force_check": None,
                       "point_contact_check": unavailable_point_contact()})
        return record, None
    record["execution_status"] = "processed"

    scored = slice(context_frames, frames)
    contacts = torch.from_numpy(window["source_contacts"][scored].copy())
    confidence = torch.from_numpy(window["source_confidence"][scored].copy())
    masks = support_interval_masks(contacts, confidence)
    masks["frame_confident"] = confidence
    support = support_frame_mask(contacts, confidence)
    kinematics = {tag: flat_ground_contact_kinematics(rebuilt[tag]["soles"][scored][None], window["fps"],
                                                      ground_height_m)
                  for tag in ("source", "pred")}
    by_channel = {tag: {} for tag in ("source", "pred")}
    segment_records = {tag: {} for tag in ("source", "pred")}
    counts = {}
    for _, _, channel, _ in CHANNEL_LAYOUT:
        for tag in ("source", "pred"):
            quantities, channel_counts, segments = channel_quantities(
                rebuilt[tag]["soles"][scored], kinematics[tag]["gap_m"][0], kinematics[tag]["penetration_m"][0],
                kinematics[tag]["interval_tangential_speed_mps"][0],
                kinematics[tag]["interval_normal_velocity_mps"][0], masks, support, channel)
            by_channel[tag][channel] = quantities
            segment_records[tag][channel] = segments
            counts[channel] = channel_counts

    record["labels"] = "source contact labels and confidence from this window drive both sides"
    record["channels"] = {}
    for side, part, channel, joint in CHANNEL_LAYOUT:
        name = f"{side}_{part}"
        record["channels"][name] = {
            "channel": joint, "side": side, "part": part,
            "contact_proxy": "official skinned-mesh sole proxy; see sole_proxy_calibration.json",
            "counts_from_source_labels": counts[channel],
            "source": summarize(by_channel["source"][channel]),
            "pred": summarize(by_channel["pred"][channel]),
        }
        record["channels"][name]["source"]["sustained_support_segments"] = segment_records["source"][channel]
        record["channels"][name]["pred"]["sustained_support_segments"] = segment_records["pred"][channel]
        record["channels"][name]["delta_reconstruction_minus_source"] = delta_summary(
            record["channels"][name]["source"], record["channels"][name]["pred"])
    groups = {f"{side}_{part}": [channel] for side, part, channel, _ in CHANNEL_LAYOUT}
    groups.update({side: [channel for entry_side, _, channel, _ in CHANNEL_LAYOUT if entry_side == side]
                   for side in SIDES})
    groups["all_channels"] = [channel for _, _, channel, _ in CHANNEL_LAYOUT]
    record["groups"] = {name: group_report(by_channel, channels, [counts[c] for c in channels])
                        for name, channels in groups.items()}
    record["coverage"] = coverage_report(contacts, confidence, masks, support,
                                         kinematics["source"]["gap_m"][0])
    record["point_contact_check"] = unavailable_point_contact()
    scored_motion = {tag: {"posed": rebuilt[tag]["posed"][scored]} for tag in ("source", "pred")}
    scored_gaps = {tag: kinematics[tag]["gap_m"][0] for tag in ("source", "pred")}
    record["resultant_force_check"] = resultant_force_check(scored_motion, skeleton, window["fps"], contacts,
                                                            confidence, scored_gaps, friction_coefficients)
    record["physics_check_status"] = "assessed_resultant_only"
    record["motion_quality"] = motion_quality(record["groups"]["all_channels"]["source"],
                                              record["groups"]["all_channels"]["pred"])
    record["motion_quality_status"] = record["motion_quality"]["status"]
    return record, {"quantities": by_channel, "channels_by_group": groups,
                    "counts": {name: sum_counts([counts[c] for c in channels])
                               for name, channels in groups.items()}}


def aggregate(group_records):
    """Pool every included behavior so cross-window quantiles stay pooled."""
    pooled = {}
    for name in group_records[0]["counts"]:
        pooled[name] = {}
        for tag in ("source", "pred"):
            merged = {}
            for record in group_records:
                for channel in record["channels_by_group"][name]:
                    for quantity, values in record["quantities"][tag][channel].items():
                        merged.setdefault(quantity, []).append(values)
            pooled[name][tag] = {quantity: torch.cat(values) for quantity, values in merged.items()}
    summary = {}
    for name, tags in pooled.items():
        summary[name] = {"counts_total": sum_counts([record["counts"][name] for record in group_records]),
                         "source": summarize(tags["source"]), "pred": summarize(tags["pred"])}
        summary[name]["delta_reconstruction_minus_source"] = delta_summary(summary[name]["source"],
                                                                          summary[name]["pred"])
    return summary


def verify_report(result):
    """Fail loudly instead of publishing an internally inconsistent report."""
    for behavior, record in result["behaviors"].items():
        if record["execution_status"] != "processed":
            continue
        groups, names = record["groups"], list(record["channels"])
        for key in COUNT_KEYS:
            total = sum(record["channels"][name]["counts_from_source_labels"][key] for name in names)
            if groups["all_channels"]["counts_from_source_labels"][key] != total:
                raise ValueError(f"{behavior}: all_channels counts disagree for {key}")
            if sum(groups[side]["counts_from_source_labels"][key] for side in SIDES) != total:
                raise ValueError(f"{behavior}: left/right counts disagree for {key}")
        for tag in ("source", "pred"):
            for quantity in ("gap_m", "penetration_m", "support_gap_m",
                             "support_tangential_speed_mps", "segment_net_drift_m"):
                total = sum(record["channels"][name][tag][quantity]["samples"] for name in names)
                if groups["all_channels"][tag][quantity]["samples"] != total:
                    raise ValueError(f"{behavior}: pooled {tag}.{quantity} samples disagree")
    return result


def render_readme(result):
    lines = ["# Flat-floor contact diagnostics review", "",
             "Machine-readable results: contact_mechanics_review.json",
             "Sole proxy calibration: sole_proxy_calibration.json", "",
             "## What this is", "",
             "Offline diagnostics on saved **M1 tokenizer reconstruction windows** of recorded SOMA30 motion. Each "
             "window is the M1 tokenizer encoding and decoding the recorded source clip. They are not WASD input, not "
             "root-only generated motion and not the output of a generative policy.", "",
             "## Three separate statuses", "",
             "| Field | Meaning |", "| --- | --- |",
             "| execution_status | did the window process at all |",
             "| motion_quality_status | do the declared kinematic screening criteria pass |",
             "| physics_check_status | which physical checks could actually run |", "",
             "A processed window is not a good motion, and a passing screening threshold is not approval.", "",
             "## Protocol", "",
             "| Item | Value |", "| --- | --- |",
             f"| Ground height | {result['protocol']['ground']['height_m']} m, fixed at conversion, never re-estimated |",
             f"| Context frames skipped | {result['protocol']['scoring']['context_frames_skipped']} |",
             f"| Expected behaviors | {', '.join(result['protocol']['input_manifest']['expected_behaviors'])} |",
             f"| Input status | {result['input_status']} |",
             "| Sole proxies | official skinned-mesh sole points, four channels |",
             "| Interval masks | sustained support / transition / swing / unknown from source labels and confidence |",
             "| Contact states | stable support / near-ground undecided / confirmed airborne / unknown, from labels, "
             "confidence and measured sole height |",
             f"| Sole proxy geometry | {result['protocol']['sole_proxies']['geometry_revision']} "
             f"(shipped dataset: {result['protocol']['sole_proxies']['geometry_revision_of_shipped_dataset']}) |",
             "| Evaluation filter | derivative edge frames excluded, phase-interior windows required, "
             f"airborne tolerance {result['protocol']['resultant_force_smoothing']['airborne_tolerance_mps2']} m/s^2 |",
             "| Quantiles | P95 pooled over samples, never an average of per-window P95 values |",
             "| Force check | resultant only, per unit mass; no per-foot load is invented |", "",
             "## Behaviors", "",
             "| Behavior | Execution | Motion quality | Physics check | Frames scored | FK rebuild error (m) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for behavior, record in sorted(result["behaviors"].items()):
        error = record.get("fk_rebuild_max_error_m") or {}
        detail = f"{error.get('source', float('nan')):.2e} / {error.get('pred', float('nan')):.2e}" if error else "-"
        lines.append(f"| {behavior} | {record['execution_status']} | {record['motion_quality_status']} | "
                     f"{record['physics_check_status']} | {record.get('frames_scored', '-')} | {detail} |")
    for behavior, record in sorted(result["failed_behaviors"].items()):
        lines.append(f"| {behavior} | failed | not_assessed | not_assessed | - | - |")
    for behavior in result["missing_behaviors"]:
        lines.append(f"| {behavior} | missing | not_assessed | not_assessed | - | - |")
    lines += ["", "## Resultant ground force", "",
              "F_ground / m = c_ddot - g, in m/s^2 per kilogram. This is the force the ground would have to exert to "
              "explain the observed centre of mass. It is not a measurement, the mass cancels so no actor weight is "
              "assumed, and a zero residual means not-contradicted rather than verified. Unknown and transition frames "
              "are never treated as airborne.", "",
              "## Reading the numbers", "",
              "Every kinematic quantity is reported for the source motion and for the tokenizer reconstruction side by "
              "side, with delta_reconstruction_minus_source on each statistic, separately per behavior, per left/right "
              "foot and per heel/forefoot channel. Counts inside a pooled group are channel-frame counts.", "",
              "## Open items", ""]
    lines += [f"- {note}" for note in result["notes"]]
    return "\n".join(lines) + "\n"


def build_report(input_directory, output_directory, expected_behaviors, context_frames, ground_height_m,
                 friction_coefficients):
    torch.set_num_threads(4)
    inputs = sorted(Path(input_directory).glob("*_review_window.npz"))
    representation = build_representation()
    skeleton = representation.skeleton
    offsets, mesh_sha256 = official_sole_offsets(skeleton)
    behaviors, failed, group_records = {}, {}, []
    for path in inputs:
        behavior = path.name[: -len("_review_window.npz")]
        record, pooled = review_window(path, behavior, skeleton, offsets, context_frames, ground_height_m,
                                       friction_coefficients)
        if record["execution_status"] == "processed":
            behaviors[behavior] = record
            group_records.append(pooled)
        else:
            failed[behavior] = record
    present = set(behaviors) | set(failed)
    missing = [name for name in expected_behaviors if name not in present]
    unexpected = sorted(present - set(expected_behaviors))
    script_path = Path(__file__).resolve()
    package = script_path.parents[1] / "motionbricks/data"
    result = {
        "schema_version": 2,
        "report_kind": "flat-floor contact review of saved tokenizer reconstruction windows",
        "input_directory": str(input_directory),
        "output_directory": str(output_directory),
        "input_status": "complete" if not missing and not failed else "incomplete",
        "valid_behavior_count": len(behaviors),
        "failed_behavior_count": len(failed),
        "code_identity": {
            "script": {"path": str(script_path), "sha256": file_sha256(script_path)},
            "contact_mechanics_py": {"path": str(package / "contact_mechanics.py"),
                                     "sha256": file_sha256(package / "contact_mechanics.py")},
            "contact_forces_py": {"path": str(package / "contact_forces.py"),
                                  "sha256": file_sha256(package / "contact_forces.py")},
            "soma_mass_model_py": {"path": str(package / "soma_mass_model.py"),
                                   "sha256": file_sha256(package / "soma_mass_model.py")},
            "bones_seed_py": {"path": str(package / "bones_seed.py"),
                              "sha256": file_sha256(package / "bones_seed.py")},
            "soma_revision": SOMA_REVISION,
        },
        "protocol": {
            "fps_source": "the FPS stored in each window file",
            "input_manifest": {"expected_behaviors": list(expected_behaviors),
                               "extra_behaviors_are_reported_not_required": True},
            "ground": {"height_m": ground_height_m,
                       "source": "the fixed support plane subtracted once at conversion by prepare_bones_locomotion.py; "
                                 "never re-estimated here to hide reconstruction error"},
            "sole_proxies": {
                "channels": [channel for _, _, _, channel in CHANNEL_LAYOUT],
                "layout": "left heel, left forefoot, right heel, right forefoot",
                "source": "median sole-mesh vertex per channel from the official skinned mesh, not ankle-joint height",
                "labels_are_proxies": "heel and forefoot are convenience anatomical labels for the four contact proxies",
                "mesh_sha256": mesh_sha256,
                "calibration_report": "sole_proxy_calibration.json",
                "geometry_revision": SOLE_PROXY_GEOMETRY,
                "geometry_revision_of_shipped_dataset": SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET,
                "geometry_revision_note": "local_frame_v1 applies the inverse bind rotation that "
                                          "position_only_v0 skipped. The shipped dataset was converted with v0, so "
                                          "switching revisions changes sole heights, the inferred plane and the "
                                          "contact labels and requires a versioned rebuild, not a silent overwrite.",
            },
            "scoring": {
                "context_frames_skipped": context_frames,
                "interval_rule": "a [t, t+1] interval is scored only when both endpoint frames are inside the scored range",
                "labels": "source contact labels and confidence from the same window drive source and reconstruction alike",
                "interval_classes": "sustained support, transition, swing, unknown; unknown is never support and never airborne",
                "quantiles": "P95 is computed over pooled samples, never as an average of per-window P95 values",
            },
            "motion_quality_criteria": list(MOTION_QUALITY_CRITERIA),
            "unified_contact_supervision": {
                "purpose": "One time grid and one set of masks for horizontal slip, vertical contact error and stance "
                           "drift, so a later training repair does not add three synonyms of the same penalty.",
                "quantities": {
                    "horizontal_slip_mps": {
                        "report_quantity": "support_tangential_speed_mps",
                        "mask": "sustained-support intervals only",
                        "training_status": "not applied"},
                    "vertical_contact_error": {
                        "report_quantity": "support_gap_m and support_normal_speed_mps",
                        "mask": "known support frames and sustained-support intervals",
                        "training_status": "not applied"},
                    "stance_drift_m": {
                        "report_quantity": "segment_path_length_m, segment_net_drift_m, segment_max_anchor_m",
                        "mask": "maximal runs of sustained-support intervals",
                        "training_status": "not applied"},
                },
                "shared_inputs": ["bones_seed.estimate_contacts labels and confidence",
                                  "contact_mechanics.support_frame_mask",
                                  "contact_mechanics.support_interval_masks",
                                  "contact_mechanics.sustained_support_segments"],
                "not_changed_this_round": "bones_models.contact_geometry_losses is unchanged. Its l_trusted_skate_mps "
                                          "still measures the full three-dimensional stance speed, including the "
                                          "vertical component, while this report measures the horizontal part only.",
            },
            "resultant_force_smoothing": {"filter": "Savitzky-Golay", "window_frames": SMOOTHING_WINDOW_FRAMES,
                                          "polyorder": SMOOTHING_POLYORDER,
                                          "edge_frames_excluded": True,
                                          "phase_interior_required": True,
                                          "airborne_tolerance_mps2": AIRBORNE_FORCE_TOLERANCE_MPS2},
            "contact_states": {
                "states": list(CONTACT_STATES),
                "stable_support": "at least one channel labeled support and confident",
                "near_ground_undecided": "no stable support but a sole at or below the contact clearance. Near the "
                                          "floor is not evidence of contact, so this state is never weight-bearing, never "
                                          "sliding, never flight and never training truth",
                "airborne": "no stable support, every sole above the airborne clearance, and every channel "
                            "confident: measured clearance and agreeing labels, not just a missing label",
                "unknown": "the band in between; never contact and never flight",
            },
        },
        "behaviors": behaviors,
        "failed_behaviors": failed,
        "missing_behaviors": missing,
        "unexpected_behaviors": unexpected,
        "group_names": list(GROUP_NAMES),
        "aggregate_across_included_behaviors": aggregate(group_records) if group_records else None,
        "mechanics": unavailable_point_contact(),
        "notes": [],
    }
    verify_report(result)
    if SOLE_PROXY_GEOMETRY != SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET:
        result["notes"].append(
            "This report uses sole proxy geometry " + SOLE_PROXY_GEOMETRY + " while the saved windows and their "
            "contact labels were produced with " + SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET + ". The labels are reused "
            "unchanged, so a small part of the measured gap difference comes from that revision change. Rebuilding "
            "the dataset on the corrected geometry is a separate versioned action; see sole_proxy_calibration.json.")
    evaluated = [record["resultant_force_check"].get("evaluated_airborne_frames", {}).get("source", 0)
                 for record in behaviors.values()]
    if behaviors and not any(evaluated or 0 for evaluated in evaluated):
        result["notes"].append(
            "No window keeps a whole smoothing window inside a confirmed flight phase, so no ballistic verdict is "
            "issued anywhere. Zero evaluable airborne frames means there is no ballistic evidence either way: it does "
            "not show that the source motion is physical, and it does not explain the earlier residuals. The previous "
            "airborne violation counts were not a valid verdict, but they were also not refuted.")
    if missing:
        result["notes"].append("Expected windows missing: " + ", ".join(missing)
                               + ". An incomplete input set is reported, never silently reduced to all-ok.")
    if failed:
        result["notes"].append("Windows that failed execution: " + ", ".join(sorted(failed))
                               + ". Processing success and motion quality are recorded separately.")
    source_failures = {name: record["resultant_force_check"].get("per_friction_coefficient", {})
                       for name, record in behaviors.items()}
    if source_failures:
        result["interpretation"] = interpret_source_vs_reconstruction(behaviors)
    transitions = sum(record["coverage"]["channel_intervals"]["transition"]
                      for record in behaviors.values())
    if behaviors and transitions == 0:
        result["notes"].append(
            "No scored interval has differing labels on two confident endpoints, so intervals_transition is zero "
            "everywhere and lift-off and touch-down appear as unknown intervals. That is a property of the automatic "
            "label scheme, not of this diagnostic.")
    return result


def interpret_source_vs_reconstruction(behaviors):
    """Compare source and reconstruction violations, so the model is blamed before the network."""
    report = {"guidance": "If the source motion violates these checks as often as the reconstruction, the fault is in "
                          "the model or the data, not in the network. Fix the model before training against it.",
              "per_behavior": {}}
    for name, record in sorted(behaviors.items()):
        check = record["resultant_force_check"]
        entry = {}
        for tag in ("source", "pred"):
            violations = [coefficient[tag]["violations"]
                          for coefficient in check.get("per_friction_coefficient", {}).values()]
            entry[tag] = {
                "ground_pull_frames": max((value["ground_pull_frames"] for value in violations), default=0),
                "friction_capacity_frames": max((value["friction_capacity_frames"] for value in violations), default=0),
                "airborne_force_frames": max((value["airborne_force_frames"] for value in violations), default=0),
            }
        entry["source_violates_too"] = any(value > 0 for value in entry["source"].values())
        report["per_behavior"][name] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description="Flat-floor contact review of saved tokenizer windows")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-behaviors", default=DEFAULT_EXPECTED_BEHAVIORS,
                        help="Comma-separated behaviors that must be present for the input set to be complete")
    parser.add_argument("--friction-coefficients", default=DEFAULT_FRICTION_COEFFICIENTS,
                        help="Comma-separated static friction coefficients for the sensitivity check")
    parser.add_argument("--context-frames", type=int, default=4,
                        help="Observed context frames dropped before scoring, per the tokenizer protocol")
    parser.add_argument("--ground-height-m", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero when the input set is incomplete or any window failed execution")
    args = parser.parse_args()
    if args.context_frames < 0:
        parser.error("context frames cannot be negative")
    expected = [name.strip() for name in args.expected_behaviors.split(",") if name.strip()]
    coefficients = [float(value) for value in args.friction_coefficients.split(",") if value.strip()]
    if not expected or not coefficients or any(not math.isfinite(value) or value < 0 for value in coefficients):
        parser.error("expected behaviors and nonnegative friction coefficients are required")
    result = build_report(args.input, args.output, expected, args.context_frames, args.ground_height_m,
                          coefficients)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "contact_mechanics_review.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (args.output / "README.md").write_text(render_readme(result))
    print(json.dumps({"input_status": result["input_status"], "processed": sorted(result["behaviors"]),
                      "failed": sorted(result["failed_behaviors"]),
                      "missing": result["missing_behaviors"],
                      "unexpected": result["unexpected_behaviors"], "output": str(args.output)}, indent=2))
    if args.strict and result["input_status"] != "complete":
        sys.exit(1)


if __name__ == "__main__":
    main()
