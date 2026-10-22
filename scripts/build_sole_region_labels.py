"""Small-sample candidate sole-region contact labels from the recorded source motion.

Scope: the five saved review windows only. Nothing is rebuilt, nothing is trained and
no shipped dataset file is touched. The candidate labels are written to a separate
versioned directory so the existing labels stay exactly where they are.

Rules this script follows:

* Clearance and penetration come from the official skinned sole region, not from the
  fixed proxy points. The ground is the plane the conversion already subtracted and it
  stays put: no per-frame ground fitting and no lowering of the body.
* Slip is measured on fixed numbered material vertices inside their own contact runs,
  never on whichever vertex happens to be lowest on a given frame, and a heel-to-toe
  roll is allowed because it only hands contact from one vertex run to another.
* Labels are generated from the source motion alone. The old labels are compared
  against, never used as the definition of correct, and no threshold was tuned to
  reproduce them.
* Undecided frames and contact-transition intervals are kept and reported. They are
  never folded into contact or into flight to make coverage look better.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_seed import SOMA_REVISION, build_representation, official_sole_surfaces
from motionbricks.data.sole_region_contact import (FRAME_STATES, INTERVAL_STATES, SOLE_LABEL_THRESHOLDS,
                                                   material_point_slip, region_gap_m, sole_region_indices,
                                                   sole_region_heights, sole_region_labels)
from motionbricks.data.soma_skin_reference import load_official_skinner, skinned_vertices

DEFAULT_INPUT = Path("out/bones_pilot_reports/m1-v7-support-repair/review-validation")
DEFAULT_OUTPUT = Path("out/sole_region_labels_v1")
CONTEXT_FRAMES = 4
GROUND_HEIGHT_M = 0.0
SIDES = ("Left", "Right")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "local_rot_mats": torch.from_numpy(data["source_local_rot_mats"].astype(np.float32)),
            "positions_m": torch.from_numpy(data["source_positions_m"].copy()),
            "old_contacts": data["source_contacts"].copy(),
            "old_confidence": data["source_confidence"].copy(),
            "fps": float(data["fps"]),
        }


def analyse_side(vertices_m, indices, fps, ground_height_m):
    """Region heights, gap, penetration, material-point slip and candidate labels."""
    height = sole_region_heights(vertices_m, indices, ground_height_m)
    horizontal = vertices_m.index_select(
        1, torch.as_tensor(np.asarray(indices), dtype=torch.long))[..., [0, 2]]
    gap = region_gap_m(height)
    slip = material_point_slip(height, horizontal, fps,
                               contact_clearance_m=SOLE_LABEL_THRESHOLDS["contact_clearance_m"],
                               min_run_frames=SOLE_LABEL_THRESHOLDS["min_contact_run_frames"])
    labels = sole_region_labels(gap["region_gap_m"], slip,
                                contact_clearance_m=SOLE_LABEL_THRESHOLDS["contact_clearance_m"],
                                clearance_m=SOLE_LABEL_THRESHOLDS["clearance_m"],
                                slip_speed_mps=SOLE_LABEL_THRESHOLDS["slip_speed_mps"])
    return {"height_m": height, "horizontal_m": horizontal, "gap_m": gap["region_gap_m"],
            "penetration_m": gap["region_penetration_m"], "slip": slip, "labels": labels}


def compare_with_old(old_contacts, old_confidence, candidate, side, context_frames):
    """Agreement with the old labels, as a comparison and never as a correctness proof."""
    columns = [0, 1] if side == "Left" else [2, 3]
    trust = (old_contacts[:, columns] & old_confidence[:, columns]).any(axis=1)
    old_support = trust[context_frames:]
    new_state = candidate["labels"]["frame_state"][context_frames:].numpy()
    new_contact = new_state == FRAME_STATES.index("contact")
    new_clear = new_state == FRAME_STATES.index("clearance")
    return {
        "frames_scored": int(old_support.size),
        "old_support_frames": int(old_support.sum()),
        "old_support_now_contact": int((old_support & new_contact).sum()),
        "old_support_now_clearance": int((old_support & new_clear).sum()),
        "old_support_now_undecided": int((old_support & ~new_contact & ~new_clear).sum()),
        "old_swing_now_contact": int((~old_support & new_contact).sum()),
        "old_swing_now_clearance": int((~old_support & new_clear).sum()),
        "old_swing_now_undecided": int((~old_support & ~new_contact & ~new_clear).sum()),
        "not_a_correctness_measure": "Agreement with the old labels is a comparison. The old labels were built from "
                                     "the fixed proxy points and are not the definition of correct.",
    }


def failure_cases(side_result, indices, fps, context_frames):
    """Named frames a reviewer should look at first."""
    gap = side_result["gap_m"]
    penetration = side_result["penetration_m"]
    slip = side_result["slip"]["region_slip_mps"]
    frames = int(gap.shape[0])
    scoring = slice(context_frames, frames)
    contact_frames = side_result["labels"]["frame_state"] == FRAME_STATES.index("contact")
    residual = float(gap[scoring][contact_frames[scoring]].median()) if contact_frames[scoring].any() else None
    worst_penetration = int(torch.argmax(penetration)) if float(penetration.max()) > 0 else None
    worst_slip = int(torch.argmax(slip)) if slip.numel() and float(slip.max()) > 0 else None
    return {
        "ground_plane_residual_on_contact_frames_m": None if residual is None else round(residual, 6),
        "ground_plane_residual_meaning": "Median height of the sole region above the fixed plane on frames this side "
                                         "is labeled contact. A nonzero value is the residual of the conversion-time "
                                         "ground fit, kept as measured rather than fitted away per frame.",
        "frames_penetrating_the_fixed_plane": int((penetration[scoring] > 0).sum()),
        "worst_penetration_frame": worst_penetration,
        "worst_penetration_m": round(float(penetration.max()), 6),
        "worst_material_slip_frame": worst_slip,
        "worst_material_slip_mps": None if worst_slip is None else round(float(slip.max()), 6),
        "measured_slip_intervals": int((slip > 0).sum()),
        "intervals_with_evidence": int(side_result["slip"]["region_measured"].sum()),
        "measured_interval_fraction": round(float(side_result["slip"]["region_measured"].float().mean()), 4),
        "region_vertices": int(np.asarray(indices).size),
        "fps": fps,
    }


def render_overlay(folder, behavior, vertices_m, indices, side_results, old_support, fps, context_frames):
    """Sole vertices over the fixed ground, next to the region gap and label timeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = int(vertices_m.shape[0])
    time = np.arange(frames) / fps
    contact_clearance = SOLE_LABEL_THRESHOLDS["contact_clearance_m"] * 1000
    clearance = SOLE_LABEL_THRESHOLDS["clearance_m"] * 1000
    colors = {"Left": "#007fa3", "Right": "#d1691a"}
    figure, (side_axis, gap_axis) = plt.subplots(1, 2, figsize=(11, 4.2))
    # Frame the whole motion, so a lifted foot is never clipped out of the picture.
    left = np.concatenate([vertices_m.index_select(
        1, torch.as_tensor(np.asarray(indices[side]), dtype=torch.long)).numpy() for side in SIDES], axis=1)
    x_margin = 0.08
    x_limits = (float(left[..., 0].min()) - x_margin, float(left[..., 0].max()) + x_margin)
    y_limits = (float(left[..., 1].min()) - 0.02, max(float(left[..., 1].max()) + 0.03, 0.12))
    gap_limits = (min(-12.0, float(min(side_results[s]["gap_m"].min() for s in SIDES)) * 1000 - 5),
                  float(max(max(side_results[s]["gap_m"].max() for s in SIDES) * 1000 + 10, 40)))

    def draw(frame):
        side_axis.clear()
        gap_axis.clear()
        for side in SIDES:
            result = side_results[side]
            points = vertices_m[frame].index_select(
                0, torch.as_tensor(np.asarray(indices[side]), dtype=torch.long)).numpy()
            in_contact = result["height_m"][frame].numpy() <= SOLE_LABEL_THRESHOLDS["contact_clearance_m"]
            side_axis.scatter(points[~in_contact, 0], points[~in_contact, 1], s=8,
                              color=colors[side], alpha=0.35)
            side_axis.scatter(points[in_contact, 0], points[in_contact, 1], s=26, color=colors[side],
                              edgecolor="black", linewidth=0.4)
        side_axis.axhline(0.0, color="black", linewidth=1.4)
        side_axis.set(xlim=x_limits, ylim=y_limits, xlabel="world X (m)", ylabel="world Y (m)")
        state = " | ".join(f"{side}: {FRAME_STATES[int(side_results[side]['labels']['frame_state'][frame])]}"
                           for side in SIDES)
        side_axis.set_title(f"{behavior} | frame {frame} ({time[frame]:.2f} s) | {state}", fontsize=9)
        for side in SIDES:
            gap_axis.plot(time, side_results[side]["gap_m"].numpy() * 1000, color=colors[side],
                          linewidth=1.1, label=f"{side} region gap")
        gap_axis.axhspan(gap_limits[0], contact_clearance, color="#cfe8cf", alpha=0.7)
        gap_axis.axhspan(contact_clearance, clearance, color="#f2e2c4", alpha=0.7)
        gap_axis.axhline(0.0, color="black", linewidth=1.0)
        gap_axis.plot(time[old_support], np.full(int(old_support.sum()), gap_axis.get_ylim()[1] * 0.0),
                      marker="|", linestyle="none", color="#777777", markersize=5, label="old label support")
        gap_axis.axvline(time[frame], color="black", linewidth=0.9)
        gap_axis.set(xlim=(0, time[-1]), ylim=gap_limits, xlabel="time (s)", ylabel="region gap (mm)")
        gap_axis.set_title("green: contact band, sand: undecided band", fontsize=9)
        gap_axis.legend(loc="upper left", fontsize=7)
        figure.tight_layout()
        return []

    animation = FuncAnimation(figure, draw, frames=range(0, frames, 1), interval=1000 / 30)
    animation.save(folder / f"{behavior}_sole_ground.gif", writer=PillowWriter(fps=30))
    plt.close(figure)


def build(input_directory, output_directory, context_frames, render):
    torch.set_num_threads(4)
    inputs = sorted(Path(input_directory).glob("*_review_window.npz"))
    if not inputs:
        raise ValueError(f"No review windows under {input_directory}")
    skeleton = build_representation().skeleton
    surfaces, _, _, mesh_sha256 = official_sole_surfaces(skeleton)
    indices = {side: sole_region_indices(surfaces, side) for side in SIDES}
    skinner = load_official_skinner(skeleton)
    output_directory.mkdir(parents=True, exist_ok=True)
    behaviours, totals = {}, {state: 0 for state in FRAME_STATES}
    interval_totals = {state: 0 for state in INTERVAL_STATES}
    old_frame_totals = 0
    for path in inputs:
        behavior = path.name[: -len("_review_window.npz")]
        source = load_source(path)
        vertices = skinned_vertices(skinner, source["local_rot_mats"], source["positions_m"])
        results = {side: analyse_side(vertices, indices[side], source["fps"], GROUND_HEIGHT_M)
                   for side in SIDES}
        np.savez_compressed(
            output_directory / f"{behavior}_sole_region_labels.npz",
            region_indices_left=indices["Left"], region_indices_right=indices["Right"],
            **{f"frame_state_{side.lower()}": results[side]["labels"]["frame_state"].numpy()
               for side in SIDES},
            **{f"interval_state_{side.lower()}": results[side]["labels"]["interval_state"].numpy()
               for side in SIDES},
            **{f"region_gap_m_{side.lower()}": results[side]["gap_m"].numpy() for side in SIDES},
            **{f"region_penetration_m_{side.lower()}": results[side]["penetration_m"].numpy()
               for side in SIDES},
            **{f"region_slip_mps_{side.lower()}": results[side]["slip"]["region_slip_mps"].numpy()
               for side in SIDES},
            **{f"vertex_height_m_{side.lower()}": results[side]["height_m"].numpy() for side in SIDES},
            **{f"contact_run_id_{side.lower()}": results[side]["slip"]["contact_run_id"].numpy()
               for side in SIDES},
            **{f"region_measured_{side.lower()}": results[side]["slip"]["region_measured"].numpy()
               for side in SIDES},
            **{f"vertex_measured_{side.lower()}": results[side]["slip"]["vertex_measured"].numpy()
               for side in SIDES},
            old_contacts=source["old_contacts"], old_confidence=source["old_confidence"],
            fps=np.array(source["fps"]), ground_height_m=np.array(GROUND_HEIGHT_M))
        entry = {"input_file": str(path), "input_sha256": file_sha256(path),
                 "frames": int(vertices.shape[0]), "fps": source["fps"], "sides": {}}
        for side in SIDES:
            columns = [0, 1] if side == "Left" else [2, 3]
            old_support = (source["old_contacts"][:, columns] & source["old_confidence"][:, columns]).any(axis=1)
            entry["sides"][side] = {
                "frame_counts": results[side]["labels"]["frame_counts"],
                "interval_counts": results[side]["labels"]["interval_counts"],
                "region_gap_m": {name: round(float(value), 6) for name, value in (
                    ("median", torch.median(results[side]["gap_m"])),
                    ("p05", torch.quantile(results[side]["gap_m"], 0.05)),
                    ("p95", torch.quantile(results[side]["gap_m"], 0.95)),
                    ("min", results[side]["gap_m"].min()),
                    ("max", results[side]["gap_m"].max()))},
                "old_label_comparison": compare_with_old(source["old_contacts"], source["old_confidence"],
                                                         results[side], side, context_frames),
                "failure_cases": failure_cases(results[side], indices[side], source["fps"], context_frames),
            }
            for name, value in results[side]["labels"]["frame_counts"].items():
                totals[name] += value
            for name, value in results[side]["labels"]["interval_counts"].items():
                interval_totals[name] += value
            old_frame_totals += int(old_support[context_frames:].sum())
        behaviours[behavior] = entry
        if render:
            columns = [0, 1]
            old_support = (source["old_contacts"][:, columns] & source["old_confidence"][:, columns]).any(axis=1)
            render_overlay(output_directory, behavior, vertices, indices, results, old_support,
                           source["fps"], context_frames)
        print(json.dumps({"behavior": behavior, "frames": entry["frames"],
                          "left": entry["sides"]["Left"]["frame_counts"],
                          "right": entry["sides"]["Right"]["frame_counts"]}))
    script_path = Path(__file__).resolve()
    package = script_path.parents[1] / "motionbricks/data"
    manifest = {
        "schema_version": 1,
        "kind": "candidate sole-region contact labels from the recorded source motion",
        "status": "candidate version for small-sample review; not approved and not used by training",
        "supersedes_nothing": "The shipped labels are untouched. This directory is a separate candidate build.",
        "scope": "the five saved review windows only; no rebuild and no training",
        "contact_geometry": {
            "region": "fixed numbered material vertices of the official skinned mesh, the union of each foot's "
                      "heel and forefoot channel selections",
            "region_vertices_per_side": {side: int(indices[side].size) for side in SIDES},
            "mesh_sha256": mesh_sha256,
            "skinner": "kimodo.viz.soma_skin.SOMASkin, the official implementation, loaded by file path",
            "ground": {"height_m": GROUND_HEIGHT_M,
                       "policy": "the fixed plane subtracted at conversion; never re-fitted per frame and the body "
                                 "is never lowered to make contact appear"},
            "per_frame_ground_fitting": False,
        },
        "slip_definition": {
            "material_points": "slip uses the same numbered vertices inside their own contact runs",
            "never_the_lowest_point": "the per-frame lowest vertex is not used for slip, because its identity changes "
                                       "during a roll and would be reported as metres per second of skate",
            "rolling_allowed": "a heel-to-toe roll ends one vertex run and starts another; it is not penalised",
            "min_contact_run_frames": SOLE_LABEL_THRESHOLDS["min_contact_run_frames"],
        },
        "thresholds": SOLE_LABEL_THRESHOLDS,
        "frame_policy": {
            "context_frames_skipped_for_scoring": context_frames,
            "labels_written_for_all_frames": True,
            "no_frame_reselection": "The scoring range is fixed once. A reconstruction evaluation must use this same "
                                    "range; it must never re-pick frames from prediction results.",
        },
        "totals": {"frame_states": totals, "interval_states": interval_totals,
                   "old_label_support_frames_scored": old_frame_totals},
        "behaviors": behaviours,
        "code_identity": {
            "script": {"path": str(script_path), "sha256": file_sha256(script_path)},
            "sole_region_contact_py": {"path": str(package / "sole_region_contact.py"),
                                       "sha256": file_sha256(package / "sole_region_contact.py")},
            "soma_skin_reference_py": {"path": str(package / "soma_skin_reference.py"),
                                       "sha256": file_sha256(package / "soma_skin_reference.py")},
            "soma_revision": SOMA_REVISION,
        },
        "open_items": [
            "The fixed ground plane was fitted at conversion from the old fixed proxies. Staying put rather than "
            "re-fitting is deliberate, so the per-side residual in failure_cases is the number to review before any "
            "rebuild, not something to fit away per frame.",
            "Thresholds are candidates. They were declared from the standing and slow-walking geometry and were not "
            "tuned to reproduce the old labels.",
            "This is a five-window sample and says nothing about the rest of the dataset.",
        ],
    }
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Candidate sole-region contact labels for the five review windows")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-frames", type=int, default=CONTEXT_FRAMES)
    parser.add_argument("--no-render", action="store_true", help="Skip the sole-vertex and ground overlay videos")
    args = parser.parse_args()
    if args.context_frames < 0:
        parser.error("context frames cannot be negative")
    manifest = build(args.input, args.output, args.context_frames, not args.no_render)
    print(json.dumps({"output": str(args.output), "totals": manifest["totals"]}, indent=2))


if __name__ == "__main__":
    main()
