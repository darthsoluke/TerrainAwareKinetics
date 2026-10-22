"""Preserve seam/control regressions in raw replanning ablation results."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(folder):
    result = {"post_processing": None, "quality_approved": False,
              "default_bundle_changed": False, "arms": {}}
    identities = None
    for arm in ("original", "C"):
        report = json.loads((folder / ("replanning_" + arm) / "metrics.json").read_text())
        if not report.get("complete") or report["intervals"] != [8, 16, 56] or report["publish_frames"] != 8:
            raise ValueError("An entire matched raw replanning evaluation is required")
        if report["post_processing"] is not None or report["test_used"] or report["fps"] != 60:
            raise ValueError("Unfiltered held-out 60-fps motion is required")
        baseline = report["rollouts"]["8"]
        if len(baseline) != 3:
            raise ValueError("Three initial histories are required")
        baseline_review = folder / "trial_v3" / ("review_" + arm)
        baseline_exact = []
        for index in range(3):
            with np.load(baseline_review / f"long_{index}.npz", allow_pickle=False) as previous, \
                    np.load(folder / ("replanning_" + arm) / f"interval_8_history_{index}.npz", allow_pickle=False) as current_motion:
                baseline_exact.append(np.array_equal(previous["global_features"], current_motion["global_features"]))
        if not all(baseline_exact):
            raise ValueError("Eight-frame scheduler changed the existing deployed baseline motion")
        current = [(row["initial_filename"], row["initial_frame"], row["input_sha256"]) for row in baseline]
        if identities is not None and current != identities:
            raise ValueError("Model arms did not use the same initial histories")
        identities = current
        metrics = {}
        for interval in (8, 16, 56):
            rows = report["rollouts"][str(interval)]
            if [(row["initial_filename"], row["initial_frame"], row["input_sha256"]) for row in rows] != identities:
                raise ValueError("Intervals did not use the same initial histories")
            checks = []
            for base, row in zip(baseline, rows):
                checks.extend(row[key][stat] <= base[key][stat]
                              for key, stat in (("angular_step_deg", "p95"), ("angular_step_deg", "max"),
                                                ("boundary_angular_step_deg", "p95"),
                                                ("root_command_velocity_error_mps", "mean")))
                checks.extend(row["per_command"][name]["velocity_error_mps"]["mean"]
                              <= base["per_command"][name]["velocity_error_mps"]["mean"]
                              for name in base["per_command"])
            metrics[str(interval)] = {
                "model_calls_per_history": [row["model_calls"] for row in rows],
                "angular_step_p95_deg": float(np.mean([row["angular_step_deg"]["p95"] for row in rows])),
                "worst_joint_step_deg": max(row["angular_step_deg"]["max"] for row in rows),
                "replan_boundary_angular_p95_deg": float(np.mean([row["boundary_angular_step_deg"]["p95"] for row in rows])),
                "joint_acceleration_p95_mps2": float(np.mean([row["joint_acceleration_mps2"]["p95"] for row in rows])),
                "command_error_mps": float(np.mean([row["root_command_velocity_error_mps"]["mean"] for row in rows])),
                "idle_error_mps": float(np.mean([row["per_command"]["idle"]["velocity_error_mps"]["mean"] for row in rows])),
                "stop_error_mps": float(np.mean([row["per_command"]["stop"]["velocity_error_mps"]["mean"] for row in rows])),
                "stop_again_error_mps": float(np.mean([row["per_command"]["stop_again"]["velocity_error_mps"]["mean"] for row in rows])),
                "strict_paired_no_regression": all(checks), "quality_approved": False}
        result["arms"][arm] = {"bundle_sha256": report["bundle_sha256"],
                               "baseline_raw_arrays_exact": baseline_exact, "intervals": metrics}
    result["decision"] = "no_default_scheduler_or_model_promotion"
    result["limitations"] = ["pilot corpus, one training seed", "no runtime latency or collision feedback in offline ablation",
                             "boundary percentiles use actual model replan boundaries, not each published block",
                             "56 frames is not the paper's recommended interval", "no live source contact labels for free rollouts"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.folder)
    (args.folder / "replanning_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
