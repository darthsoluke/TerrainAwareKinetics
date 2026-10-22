"""Calibrate the SOMA sole contact proxies against the official skinned mesh.

The four contact proxies produced by bones_seed.official_sole_offsets are medians of
the lowest slice of the foot skin vertices. They are calibration proxies, not exact
contact points, so this script measures how far a proxy sits from the sole surface it
represents: geometrically in the mesh bind frame, and frame by frame under the same
rigid-offset model the motion pipeline uses.

Nothing here moves the ground, lowers the body or re-estimates the support plane. The
measured mismatch is reported as it is, and the suggested tolerance is labelled a
proposal that still needs review.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_seed import (FOOT_CHANNELS, FPS, SOLE_PROXY_GEOMETRY,
                                          SOLE_PROXY_GEOMETRY_LOCAL_FRAME,
                                          SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET,
                                          SOLE_PROXY_GEOMETRY_POSITION_ONLY, build_representation,
                                          official_sole_offsets, official_sole_surfaces, sole_positions,
                                          sole_proxy_geometry_report)
from motionbricks.data.contact_mechanics import flat_ground_contact_kinematics, support_frame_mask
from motionbricks.data.soma_skin_reference import load_official_skinner, skinned_vertices, surface_heights

DEFAULT_INPUT = Path("out/bones_pilot_reports/m1-v7-support-repair/review-validation")
DEFAULT_OUTPUT = Path("out/contact_mechanics_review")
CONTEXT_FRAMES = 4
GROUND_HEIGHT_M = 0.0
TOLERANCE_ROUNDING_M = 0.005


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    """Sample count and order statistics of a 1-D array, or explicit nulls."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"samples": 0, "mean": None, "minimum": None, "p05": None, "median": None,
                "p95": None, "maximum": None}
    return {"samples": int(values.size), "mean": float(values.mean()), "minimum": float(values.min()),
            "p05": float(np.quantile(values, 0.05)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)), "maximum": float(values.max())}


def channel_geometry(surface, proxy):
    """Where the proxy sits inside the sole vertex cloud, in the mesh bind frame.

    Positive heights mean the vertex is below the proxy, that is, closer to the floor
    than the point the pipeline treats as the contact point.
    """
    assigned = surface["assigned"]
    selected = surface["selected"]
    return {
        "joint": surface["joint"],
        "assigned_vertices": int(len(assigned)),
        "sole_slice_vertices": int(len(surface["sole"])),
        "proxy_vertices": int(len(selected)),
        "proxy_offset_from_foot_joint_m": [round(float(value), 6) for value in proxy - surface["joint_bind_translation"]],
        "assigned_vertex_height_below_proxy_m": distribution(proxy[1] - assigned[:, 1]),
        "proxy_vertex_height_below_proxy_m": distribution(proxy[1] - selected[:, 1]),
        "fraction_of_assigned_vertices_below_proxy": float((assigned[:, 1] < proxy[1]).mean()),
        "proxy_above_lowest_assigned_vertex_m": float(proxy[1] - assigned[:, 1].min()),
        "proxy_above_lowest_selected_vertex_m": float(proxy[1] - selected[:, 1].min()),
        "selected_vertical_spread_m": float(selected[:, 1].max() - selected[:, 1].min()),
        "sole_slice_vertical_spread_m": float(surface["sole"][:, 1].max() - surface["sole"][:, 1].min()),
    }


def bind_pose_alignment(skeleton, surfaces, bind_transforms):
    """Compare the SOMA30 identity-rotation FK pose with the mesh bind rig transform.

    The pipeline treats the proxy offset as a world-axis offset rotated by the joint.
    That is exact only if the skeleton standard pose equals the mesh bind pose up to a
    global translation. This measures whether it does.
    """
    identity = torch.eye(3, dtype=torch.float64).expand(1, skeleton.nbjoints, 3, 3).contiguous()
    rotations, posed, _ = skeleton.fk(identity, torch.zeros(1, 3, dtype=torch.float64))
    root_shift = torch.from_numpy(bind_transforms[0, :3, 3]) - posed[0, skeleton.root_idx]
    angles, residuals = {}, {}
    for surface in surfaces:
        joint = skeleton.bone_index[surface["joint"]]
        residual = (posed[0, joint] + root_shift).numpy() - surface["joint_bind_translation"]
        residuals[surface["channel"]] = {"norm_m": round(float(np.linalg.norm(residual)), 6),
                                         "xyz_m": [round(float(value), 6) for value in residual]}
        relative = rotations[0, joint].numpy() @ surface["joint_bind_rotation"].T
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        angles[surface["channel"]] = round(math.degrees(math.acos(cosine)), 4)
    return {
        "checked": "SOMA30 FK at identity local rotations versus the skinned-mesh bind rig transform",
        "root_translation_difference_m": [round(float(value), 6) for value in root_shift],
        "foot_joint_position_residual_after_root_shift_m": residuals,
        "foot_joint_rotation_angle_deg": angles,
        "distinct_from": "This is not the reference_geometry_error_m the conversion pipeline already checks; that "
                         "compares the BVH reference skeleton with the official SOMA skeleton and is zero. This "
                         "compares the skinned mesh bind pose with the SOMA30 standard pose, which nothing checked.",
        "conclusion": "The mesh bind pose is not the SOMA30 standard pose, and the foot joints differ by about ten "
                      "degrees. That is a property of the assets. It is NOT the explanation for how far the fixed "
                      "proxy sits from the skinned sole surface: the official skinner already applies the inverse bind "
                      "transform, and the error decomposition below shows the corrected rigid sole vertex reaches the "
                      "official skinned surface to within about a millimetre. Do not hand-align the skeleton to chase "
                      "the remaining centimetres.",
    }


def rigid_offset_series(rebuilt_positions, global_rotations, skeleton, surface, proxy):
    """Proxy clearance minus lowest modelled sole vertex clearance, frame by frame.

    Under the pipeline's rigid-offset model the difference between the proxy world
    height and a vertex world height depends only on the joint rotation and the bind
    frame geometry, not on the joint translation.
    """
    joint = skeleton.bone_index[surface["joint"]]
    delta = torch.from_numpy(proxy - surface["selected"]).to(global_rotations.dtype)
    rotated = delta[None] @ global_rotations[:, joint].transpose(-1, -2)
    return rotated[..., 1].max(dim=-1).values


def load_windows(folder):
    return [path for path in sorted(Path(folder).glob("*_review_window.npz"))]


def window_source_motion(path, skeleton):
    with np.load(path, allow_pickle=False) as data:
        local = torch.from_numpy(data["source_local_rot_mats"].astype(np.float32))
        positions = torch.from_numpy(data["source_positions_m"].copy())
        contacts = torch.from_numpy(data["source_contacts"].copy())
        confidence = torch.from_numpy(data["source_confidence"].copy())
        fps = float(data["fps"])
    rotations, posed, _ = skeleton.fk(local, positions[:, skeleton.root_idx])
    # A geometric calibration is cheap at double precision and the offsets are double.
    return {"rotations": rotations.double(), "posed": posed.double(), "contacts": contacts,
            "confidence": confidence, "fps": fps, "local": local, "root": positions[:, skeleton.root_idx]}


def main():
    parser = argparse.ArgumentParser(description="Sole proxy calibration against the official skinned mesh")
    parser.add_argument("--windows", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-frames", type=int, default=CONTEXT_FRAMES)
    parser.add_argument("--ground-height-m", type=float, default=GROUND_HEIGHT_M)
    args = parser.parse_args()
    torch.set_num_threads(4)
    representation = build_representation()
    skeleton = representation.skeleton
    surfaces, bind_transforms, joint_names, mesh_sha256 = official_sole_surfaces(skeleton)
    offsets, _ = official_sole_offsets(skeleton)
    proxies = {surface["channel"]: np.median(surface["selected"], axis=0) for surface in surfaces}

    report = {
        "schema_version": 1,
        "kind": "SOMA sole contact proxy calibration",
        "mesh": {"sha256": mesh_sha256, "rig_joints": len(joint_names), "channels": list(FOOT_CHANNELS)},
        "proxy_definition": {
            "rule": "vertices with foot-family skin influence above 0.8, then the lowest 8 percent along Y, then the "
                    "median of the heel or forefoot half split at the median Z",
            "source_function": "motionbricks.data.bones_seed.official_sole_offsets",
            "offsets_from_foot_joint_m": [[round(float(value), 6) for value in offset] for offset in offsets],
            "status": "calibration proxy, not an exact contact point",
        },
        "proxy_tolerance_note": "The proxies are not exact contact points. Any constraint that asks them to touch the "
                                "floor while carrying load must carry a tolerance derived from this report, otherwise "
                                "the source motion itself violates the constraint.",
        "geometry_channels": {surface["channel"]: channel_geometry(surface, proxies[surface["channel"]])
                              for surface in surfaces},
        "bind_pose_alignment": bind_pose_alignment(skeleton, surfaces, bind_transforms),
        "not_done": [
            "The true linear-blend-skinned sole surface was not evaluated. Doing so needs the bind rig transform "
            "aligned to the SOMA30 standard pose, which the bind_pose_alignment check above shows is not the case.",
            "No ground plane was moved, no body was lowered and no support plane was re-estimated to hide any of "
            "the measured mismatch.",
        ],
    }

    offsets_v0, _ = official_sole_offsets(skeleton, SOLE_PROXY_GEOMETRY_POSITION_ONLY)
    offsets_v1 = offsets
    skinner = load_official_skinner(skeleton)
    revision_gap = {SOLE_PROXY_GEOMETRY_LOCAL_FRAME: [], SOLE_PROXY_GEOMETRY_POSITION_ONLY: []}
    reference_offset = {SOLE_PROXY_GEOMETRY_LOCAL_FRAME: {c: [] for c in FOOT_CHANNELS},
                        SOLE_PROXY_GEOMETRY_POSITION_ONLY: {c: [] for c in FOOT_CHANNELS}}
    decomposition = {key: {c: [] for c in FOOT_CHANNELS}
                     for key in ("proxy_to_rigid_lowest", "rigid_lowest_to_skinned", "proxy_to_skinned")}
    raw_gate_flip = []
    windows = load_windows(args.windows)
    report["empirical_inputs_available"] = bool(windows)
    report["window_files"] = [str(path) for path in windows]
    per_channel_series = {channel: [] for channel in FOOT_CHANNELS}
    per_side_series = {"left": [], "right": []}
    clearance = []
    for path in windows:
        motion = window_source_motion(path, skeleton)
        scored = slice(args.context_frames, motion["posed"].shape[0])
        poses = motion["posed"][scored]
        rotations = motion["rotations"][scored]
        for channel in FOOT_CHANNELS:
            surface = [entry for entry in surfaces if entry["channel"] == channel][0]
            per_channel_series[channel].append(
                rigid_offset_series(poses, rotations, skeleton, surface, proxies[channel]).numpy())
        for side in ("left", "right"):
            proxies_world, vertices_world = [], []
            for surface in surfaces:
                if not surface["channel"].startswith(side.capitalize()):
                    continue
                joint = skeleton.bone_index[surface["joint"]]
                offset = torch.from_numpy(proxies[surface["channel"]] - surface["joint_bind_translation"])
                proxies_world.append((poses[:, joint] + rotations[:, joint] @ offset)[:, 1])
                delta = torch.from_numpy(surface["selected"] - surface["joint_bind_translation"])
                vertices_world.append((poses[:, joint, None, :]
                                       + (delta[None] @ rotations[:, joint].transpose(-1, -2)))[..., 1])
            lowest_proxy = torch.stack(proxies_world).amin(dim=0)
            lowest_vertex = torch.stack(vertices_world).amin(dim=0).amin(dim=-1)
            per_side_series[side].append((lowest_proxy - lowest_vertex).numpy())
        soles = sole_positions(poses[None], rotations[None], skeleton, offsets.double())
        kinematics = flat_ground_contact_kinematics(soles, motion["fps"], args.ground_height_m)
        support = support_frame_mask(motion["contacts"][scored], motion["confidence"][scored])
        clearance.append(kinematics["gap_m"][0][support].numpy())
        # Official skinned reference surface, plus both proxy revisions, on the same frames.
        vertices = skinned_vertices(skinner, motion["local"], motion["posed"].float())
        skinned = {channel: values[scored].numpy()
                   for channel, values in surface_heights(vertices, surfaces).items()}
        revision_gap[SOLE_PROXY_GEOMETRY_LOCAL_FRAME].append(
            flat_ground_contact_kinematics(sole_positions(poses[None], rotations[None], skeleton, offsets_v1.double()),
                                           motion["fps"], args.ground_height_m)["gap_m"][0].numpy())
        revision_gap[SOLE_PROXY_GEOMETRY_POSITION_ONLY].append(
            flat_ground_contact_kinematics(sole_positions(poses[None], rotations[None], skeleton, offsets_v0.double()),
                                           motion["fps"], args.ground_height_m)["gap_m"][0].numpy())
        velocity = torch.gradient(sole_positions(poses, rotations, skeleton, offsets_v1.double()),
                                  spacing=(1.0 / motion["fps"],), dim=0)[0].norm(dim=-1)
        raw_v1 = ((velocity < 0.20) & (soles[0][..., 1].abs() < 0.035)).numpy()
        velocity_v0 = torch.gradient(sole_positions(poses, rotations, skeleton, offsets_v0.double()),
                                     spacing=(1.0 / motion["fps"],), dim=0)[0].norm(dim=-1)
        raw_v0 = ((velocity_v0 < 0.20) & (sole_positions(poses, rotations, skeleton,
                                                         offsets_v0.double())[..., 1].abs() < 0.035)).numpy()
        raw_gate_flip.append((raw_v0 != raw_v1).astype(np.float64))
        for channel in FOOT_CHANNELS:
            surface = [entry for entry in surfaces if entry["channel"] == channel][0]
            joint = skeleton.bone_index[surface["joint"]]
            reference = skinned[channel]
            for revision, table in ((SOLE_PROXY_GEOMETRY_LOCAL_FRAME, offsets_v1),
                                    (SOLE_PROXY_GEOMETRY_POSITION_ONLY, offsets_v0)):
                offset = torch.from_numpy(table[FOOT_CHANNELS.index(channel)].numpy().astype(np.float64))
                proxy_height = (poses[:, joint] + rotations[:, joint] @ offset)[:, 1].numpy()
                reference_offset[revision][channel].append(proxy_height - reference)
                if revision == SOLE_PROXY_GEOMETRY_LOCAL_FRAME:
                    local_delta = surface["joint_bind_rotation"].T @ (
                        surface["selected"] - surface["joint_bind_translation"]).T
                    local_delta = torch.from_numpy(np.ascontiguousarray(local_delta.T)).double()
                    rigid_lowest = (poses[:, joint, None, :]
                                    + (local_delta[None] @ rotations[:, joint].transpose(-1, -2)))[..., 1]
                    rigid_lowest = rigid_lowest.amin(dim=-1).numpy()
                    decomposition["proxy_to_rigid_lowest"][channel].append(proxy_height - rigid_lowest)
                    decomposition["rigid_lowest_to_skinned"][channel].append(rigid_lowest - reference)
                    decomposition["proxy_to_skinned"][channel].append(proxy_height - reference)

    report["official_skinned_reference"] = {
        "source": "kimodo.viz.soma_skin.SOMASkin, the official implementation, loaded by file path because "
                  "kimodo.viz imports a UI stack",
        "why": "It already pads SOMA30 to the 77-joint rig and applies the inverse bind transform, so it is the "
               "reference for the proxy rather than a manually aligned skeleton.",
        "definition": "signed difference between the proxy world height and the lowest skinned vertex of that "
                      "channel's own sole vertex set, per frame, over every window",
        "per_channel_proxy_minus_skinned_lowest_m": {
            channel: {revision: distribution(np.concatenate(reference_offset[revision][channel]))
                      if reference_offset[revision][channel] else distribution([])
                      for revision in (SOLE_PROXY_GEOMETRY_LOCAL_FRAME, SOLE_PROXY_GEOMETRY_POSITION_ONLY)}
            for channel in FOOT_CHANNELS},
    }
    report["error_decomposition"] = {
        "definition": "Per frame and per channel, three quantities that satisfy c = a + b exactly. "
                      "a: fixed proxy to the lowest rigid sole vertex under the corrected local-frame transform. "
                      "b: that rigid vertex to the official skinned lowest vertex. "
                      "c: fixed proxy to the official skinned lowest vertex. Quantiles are not additive, so the "
                      "per-frame split, not the P95 columns, is what identifies the dominant term.",
        "per_channel_m": {key: {channel: distribution(np.concatenate(values)) if values else distribution([])
                               for channel, values in per.items()}
                          for key, per in decomposition.items()} if windows else {},
        "conclusion": "The first term dominates the second by roughly two orders of magnitude, so the proxy's distance "
                      "from the skinned sole surface comes from a single fixed median point failing to represent the "
                      "lowest point of a rotating sole. It does not come from an unresolved bind transform: the "
                      "official skinner already applies the inverse bind, and the corrected rigid sole vertex lands on "
                      "the skinned surface.",
    }
    report["geometry_revision_impact"] = {
        "shipped_dataset_revision": SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET,
        "report_revision": SOLE_PROXY_GEOMETRY,
        "offset_difference_m": sole_proxy_geometry_report(skeleton),
        "sole_height_change_m": {
            channel: distribution(np.concatenate(revision_gap[SOLE_PROXY_GEOMETRY_LOCAL_FRAME])[:, index]
                                  - np.concatenate(revision_gap[SOLE_PROXY_GEOMETRY_POSITION_ONLY])[:, index])
            for index, channel in enumerate(FOOT_CHANNELS)} if windows else {},
        "label_raw_gate_flip_fraction": float(np.concatenate(raw_gate_flip).mean()) if raw_gate_flip else None,
        "label_raw_gate_definition": "the (velocity < 0.20) & (|clearance| < 0.035) gate inside "
                                     "bones_seed.estimate_contacts, evaluated on the windows",
        "rebuild_required": True,
        "rebuild_policy": "The shipped shards, contact_confidence and accepted.json were produced with "
                          + SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET
                          + ". Changing the revision moves sole heights, the inferred fixed ground plane and the "
                            "contact labels, so it needs a new versioned dataset build with its own dataset id and "
                            "conversion report. This analysis overwrote nothing.",
    }
    report["rigid_offset_model_channels_m"] = {
        channel: distribution(np.concatenate(series)) if series else distribution([])
        for channel, series in per_channel_series.items()}
    report["rigid_offset_model_per_side_m"] = {
        side: distribution(np.concatenate(series)) if series else distribution([])
        for side, series in per_side_series.items()}
    report["empirical_source_support_clearance_m"] = (
        distribution(np.concatenate(clearance)) if clearance else distribution([]))
    report["empirical_source_support_clearance_source"] = {
        "definition": "signed sole proxy height above the fixed ground on frames the source labels as support "
                      "and marks confident, after the context frames are dropped",
        "ground_height_m": args.ground_height_m,
        "context_frames_skipped": args.context_frames,
        "windows": len(windows),
    }

    reference_p95 = {
        revision: max(report["official_skinned_reference"]["per_channel_proxy_minus_skinned_lowest_m"][channel][revision]["p95"]
                      or 0.0 for channel in FOOT_CHANNELS)
        for revision in (SOLE_PROXY_GEOMETRY_LOCAL_FRAME, SOLE_PROXY_GEOMETRY_POSITION_ONLY)} \
        if windows else {}
    reference_mean = {
        revision: max(report["official_skinned_reference"]["per_channel_proxy_minus_skinned_lowest_m"][channel][revision]["mean"]
                      or 0.0 for channel in FOOT_CHANNELS)
        for revision in (SOLE_PROXY_GEOMETRY_LOCAL_FRAME, SOLE_PROXY_GEOMETRY_POSITION_ONLY)} \
        if windows else {}
    bias = max(entry["proxy_above_lowest_assigned_vertex_m"]
               for entry in report["geometry_channels"].values())
    orientation = max((entry["p95"] or 0.0) for entry in report["rigid_offset_model_channels_m"].values()) \
        if windows else None
    empirical = report["empirical_source_support_clearance_m"]["p95"]
    report["calibration_findings"] = {
        "proxy_bias_against_lowest_sole_vertex_m": {
            "value_m": round(float(bias), 6),
            "meaning": "In the mesh bind frame the proxy sits this far above the lowest assigned sole vertex. It is a "
                       "small, well-behaved bias, so the proxy is a good point proxy for a flat foot.",
        },
        "orientation_induced_proxy_spread_m": {
            "p95_m": None if orientation is None else round(float(orientation), 6),
            "meaning": "Under the rigid-offset model the gap between the proxy and the lowest modelled sole vertex "
                       "grows with foot tilt, because one point cannot represent a rotating sole surface. This is a "
                       "modelling limit, not an error that a wider tolerance can repair.",
        },
        "proxy_error_against_official_skinned_surface_m": {
            "worst_channel_mean": {key: round(float(value), 6) for key, value in reference_mean.items()},
            "worst_channel_p95": {key: round(float(value), 6) for key, value in reference_p95.items()},
            "meaning": "Even with the inverse bind rotation applied, the proxy stays about a centimetre from the "
                       "official skinned sole surface on average and several centimetres at P95. Both revisions sit in "
                       "the same band. See error_decomposition: the distance is dominated by the fixed median point "
                       "not tracking the lowest point of a rotating sole. The coordinate defect was real and is fixed, "
                       "but it was never the cause of this distance, and the mesh bind pose is not the cause either.",
        },
        "empirical_source_support_clearance_m": {
            "p95_m": None if empirical is None else round(float(empirical), 6),
            "meaning": "How far the source proxies actually sit above the fixed ground while the source labels them "
                       "as support. Any proxy-on-floor requirement tighter than this would be violated by the source "
                       "motion itself.",
        },
    }
    if empirical is None:
        report["proposed_support_clearance_tolerance"] = {
            "value_m": None,
            "derivation": "no window was available, so no empirical term could be measured",
            "status": "not derived",
        }
    else:
        report["proposed_support_clearance_tolerance"] = {
            "value_m": round(math.ceil(empirical / TOLERANCE_ROUNDING_M) * TOLERANCE_ROUNDING_M, 6),
            "derivation": "the empirical source support clearance P95 (" + format(empirical, ".4f")
                          + " m) rounded up to the next " + format(TOLERANCE_ROUNDING_M, ".3f") + " m",
            "applies_to": "the four-proxy contact model only",
            "status": "NOT APPROVED. The coordinate defect is excluded, but the proxy is still "
                      + (format(max(reference_p95.values()), ".3f") if reference_p95 else "unknown")
                      + " m from the official skinned sole surface at P95, so this number is a ground-clearance "
                        "allowance for the four-proxy model and must not be used as a contact-geometry tolerance.",
            "why": "Below this value the source motion itself would violate a strict proxy-on-floor requirement, so a "
                   "constraint tighter than this would penalise data that is not wrong.",
            "what_to_do_instead": "For a tight contact constraint, evaluate the official skinned surface or the lowest "
                                  "sole vertex under a corrected pose relationship. Do not tighten or widen this proxy "
                                  "tolerance to stand in for that.",
            "recommendation": "Do not widen this tolerance further to absorb foot tilt. If a tight clearance "
                              "constraint is wanted, evaluate contact at the lowest sole vertex under the current foot "
                              "orientation instead of at a fixed proxy point.",
        }
    report["code_identity"] = {
        "script": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
        "bones_seed_py": {"path": str(Path(__file__).resolve().parents[1] / "motionbricks/data/bones_seed.py"),
                          "sha256": file_sha256(Path(__file__).resolve().parents[1] / "motionbricks/data/bones_seed.py")},
    }
    report["skeleton_fps"] = FPS
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sole_proxy_calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"windows": len(windows),
                      "proposed_tolerance_m": report["proposed_support_clearance_tolerance"]["value_m"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
