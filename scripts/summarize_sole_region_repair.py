"""Write a measured reconstruction outcome; improvement is not deployment approval."""

import argparse
import json
from pathlib import Path
from scripts.evaluate_sole_region_ab import aggregate_region_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    before = json.loads((args.initial / "sole_region_ab_metrics.json").read_text())
    after = json.loads((args.review / "sole_region_ab_metrics.json").read_text())
    before["aggregate"] = aggregate_region_results(before["behaviors"])
    paired = json.loads((args.run / "paired_initialization.json").read_text())
    if before["dataset_snapshot_sha256"] != after["dataset_snapshot_sha256"] or not paired["same_loaded_state"]:
        raise ValueError("The comparison is not bound to the same dataset and loaded state")
    expected = set(before["behaviors"])
    if expected != set(after["behaviors"]):
        raise ValueError("The comparison changed its evaluation windows")
    for window in expected:
        for arm in ("Point", "Region"):
            if after["behaviors"][window]["arms"][arm]["valid_samples"] != before["behaviors"][window]["arms"]["Initial"]["valid_samples"]:
                raise ValueError("The comparison changed its source scoring masks")
    initial = before["aggregate"]["Initial"]
    candidates = {name: after["aggregate"][name] for name in ("Point", "Region")}
    limits = {"mpjpe_m": 0.035, "measured_slip_count_weighted_mean_mps": 0.02, "penetration_peak_m": 0.01}
    screens = {name: {key: values[key] is not None and values[key] <= limit for key, limit in limits.items()}
               for name, values in candidates.items()}
    point, region = candidates["Point"], candidates["Region"]
    balanced = (region["mpjpe_m"] <= point["mpjpe_m"] * 1.05
                and region["measured_slip_count_weighted_mean_mps"] <= point["measured_slip_count_weighted_mean_mps"]
                and region["penetration_peak_m"] <= point["penetration_peak_m"])
    selected = "Region" if balanced else "Point"
    result = {
        "same_initial_state_verified": True, "windows": len(expected), "source_masks_identical": True,
        "initial": initial, "candidates": candidates, "screen_limits": limits, "screen_checks": screens,
        "selected_for_reconstruction_review": selected,
        "reconstruction_screen_passed": all(screens[selected].values()),
        "quality_approved": False, "interactive_locomotion_evaluated": False,
        "note": "These are offline full-source tokenizer reconstructions, not player-driven generation. "
                "Source contacts are candidate geometry labels. Screens are necessary, not sufficient for natural motion.",
    }
    (args.review / "repair_outcome.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ["# SOMA reconstruction repair v3", "", "2026-10-05. Measured offline reconstruction, not generated locomotion.", "",
             "## Changes", "", "- Warm start from the existing 10,000-step support-repair checkpoint, not fresh 1,000-step smoke weights.",
             "- 152 complete training clips, with full-length sole labels; 19 held-out clips / 57 fixed validation windows; test untouched.",
             "- Point and Region each receive 10,000 additional optimizer updates, identical loaded learned/EMA state and seed.",
             "- Official SOMASkin LBS is evaluated on only the selected vertices, once for both feet; output and gradient parity are tested.",
             "- Region adds 4/8/16-frame material-point stance displacement. Every intervening interval must have stable measured evidence for that same vertex.",
             "- Padding remains undecided/unmeasured. CUDA allocation occurs before project imports and actual device UUID is checked.",
             "", "## All-window Results", "", "Slip is count-weighted; pose and acceleration errors are equal-window means. Peaks are maxima, not means.", "",
             "| Arm | MPJPE (mm) | Sole slip (mm/s) | Mean penetration (mm) | Peak penetration (mm) | Rotation (degrees) | Acceleration error (m/s^2) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in (("Initial", initial), *candidates.items()):
        rotation = row.get("local_rotation_error_deg")
        acceleration = row.get("joint_acceleration_error_mps2")
        lines.append(f"| {name} | {row['mpjpe_m'] * 1000:.2f} | {row['measured_slip_count_weighted_mean_mps'] * 1000:.2f} | "
                     f"{row['penetration_mean_window_m'] * 1000:.2f} | {row['penetration_peak_m'] * 1000:.2f} | "
                     f"{rotation:.2f} | {acceleration:.2f} |" if rotation is not None and acceleration is not None else
                     f"| {name} | {row['mpjpe_m'] * 1000:.2f} | {row['measured_slip_count_weighted_mean_mps'] * 1000:.2f} | "
                     f"{row['penetration_mean_window_m'] * 1000:.2f} | {row['penetration_peak_m'] * 1000:.2f} | n/a | n/a |")
    lines += ["", "## Decision", "", f"Selected for the next reconstruction review: **{selected}**. "
              "The Region arm is retained only if it improves slip and peak penetration without increasing pose error more than 5% versus Point.",
              "", f"Reconstruction screen passed: **{result['reconstruction_screen_passed']}**. "
              "Required mean pose error <=35 mm, measured slip <=20 mm/s, peak penetration <=10 mm. "
              "Passing this screen alone would still not approve UE gameplay or interactive generation.",
              "", "The contacts remain candidate labels on the unchanged V7 fixed planes. Root rates are supplied from recordings, "
              "and the encoder observes the entire source window. No IK, inference foot locking or pose correction is applied.",
              "", "## Run Cost", "", "| Arm | Updates | Elapsed (s) | Peak allocated memory (MiB) |", "| --- | ---: | ---: | ---: |"]
    for name in candidates:
        row = after["run_records"][name]
        lines.append(f"| {name} | {row['steps']} | {row['elapsed_seconds']:.2f} | {row['peak_gpu_memory_mb']:.2f} |")
    lines += ["", "Training time and memory are not runtime latency or RTX 3080 acceptance.", "",
              "## Method Boundary", "", "The encoder/decoder and token inference remain upstream MotionBricks. "
              "The official paper uses separate tokenizer, Root and Pose stages and standard skating/velocity losses "
              "([sections 4 and 7.4](https://arxiv.org/html/2604.24833v1)). "
              "Sole-region supervision and multi-frame material-point drift are project extensions, not claims about the paper's implementation.", "",
              "## Artifacts", "", "- sole_region_ab_metrics.json: all fixed windows, source masks, hashes and aggregates.",
              "- sole_region_ab_summary.json: run costs and original saved contracts.",
              "- repair_outcome.json: comparison checks, explicit reconstruction screens and candidate decision.",
              "- Paired initialization and all failed launch records remain in the remote run directory.",
              "- Separate showcase/skin videos use recorded timing; they are not an inference-FPS benchmark.", ""]
    (args.review / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
