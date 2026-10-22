"""Read paired raw evidence; a successful training run is not quality approval."""

import argparse
import json
from pathlib import Path

from scripts.summarize_soma_joint_rollout import MATCHED_KEYS, metric_row, paired_windows, window_identity


def summarize(folder, decoder_comparison=False, observation_comparison=False):
    if (folder / "INVALIDATED.json").exists():
        raise ValueError("This experiment was explicitly invalidated")
    if decoder_comparison and observation_comparison:
        raise ValueError("Choose one comparison")
    def read(path):
        return json.loads((folder / path).read_text())
    names = (("predicted_features", "executed_fk") if observation_comparison else
             ("frozen_decoder", "train_decoder") if decoder_comparison else ("legacy_splice", "source_v2"))
    candidate = names[1]
    contracts = {name: read(f"{name}/contract.json") for name in names}
    results = {name: read(f"{name}/result.json") for name in names}
    for name in names:
        policy = "source_v2" if decoder_comparison or observation_comparison else name
        if contracts[name]["supervision_policy"] != policy or results[name]["steps"] != contracts[name]["steps"]:
            raise ValueError("Policy or completed update count disagrees with the contract")
        if decoder_comparison and (contracts[name]["train_decoder"] != (name == candidate)
                                   or results[name]["decoder_changed"] != (name == candidate)):
            raise ValueError("Decoder training/change disagrees with the comparison")
        if observation_comparison and (contracts[name]["history_geometry"] != name or
                                       contracts[name]["train_decoder"] or results[name]["decoder_changed"]):
            raise ValueError("Observation comparison must retain a frozen decoder and the declared input policy")
    differing = {"loss_arguments"}
    for key in MATCHED_KEYS:
        if key not in differing and contracts[names[0]][key] != contracts[names[1]][key]:
            raise ValueError(f"Unmatched training field: {key}")
    a, b = (contracts[name]["loss_arguments"] for name in names)
    if {k: v for k, v in a.items() if k != "rollout_rotation_increment_coeff"} != {
            k: v for k, v in b.items() if k != "rollout_rotation_increment_coeff"}:
        raise ValueError("Unmatched base loss coefficients")
    if a["rollout_rotation_increment_coeff"] != (1 if decoder_comparison or observation_comparison else 0) or b["rollout_rotation_increment_coeff"] != 1:
        raise ValueError("Unexpected recorded rotation supervision coefficients")
    for key in ("sampling_sha256", "history_counts", "steps"):
        if results[names[0]][key] != results[names[1]][key]:
            raise ValueError(f"Unmatched executed training: {key}")
    if not all(r["frozen_unchanged"] and r["root_changed"] and r["pose_changed"] for r in results.values()):
        raise ValueError("Trained/frozen modules disagree with the experiment")
    reports = {name: read(f"review_{name}/metrics.json") for name in ("original", *names)}
    reference = reports["original"]
    if reference["bundle_sha256"] != contracts[names[0]]["initial_bundle_sha256"]:
        raise ValueError("Original evaluation does not match the shared initial weights")
    for name, report in reports.items():
        if observation_comparison and report["history_geometry"] != (name if name in names else "predicted_features"):
            raise ValueError("Evaluated input policy differs from the training contract")
        if window_identity(report) != window_identity(reference):
            raise ValueError("Scoring windows or source masks differ")
        for key in ("dataset_sha256", "evaluation_sha256", "commit_frames"):
            if report[key] != reference[key]:
                raise ValueError(f"Unmatched evaluation: {key}")
        if report["post_processing"] is not None or report["test_used"]:
            raise ValueError("No post-processing or test-set tuning is permitted")
        if decoder_comparison:
            for key in ("encoder", "quantizer"):
                if report["frozen_state"][key] != reference["frozen_state"][key]:
                    raise ValueError("Encoder and codebooks must stay fixed")
            changed = report["frozen_state"]["decoder"] != reference["frozen_state"]["decoder"]
            if changed != (name == candidate):
                raise ValueError("Evaluated decoder does not match the trained/frozen contract")
        elif report["tokenizer_state"] != reference["tokenizer_state"]:
            raise ValueError("The complete tokenizer must remain frozen")
        for i in range(3):
            for key in ("initial_filename", "initial_frame"):
                if report["rollouts"][str(i)][key] != reference["rollouts"][str(i)][key]:
                    raise ValueError("Unmatched free-control initial state")
        if name in names and report["bundle_sha256"] != results[name]["bundle_sha256"]:
            raise ValueError("Evaluation did not use the trained bundle")
    rows = {name: metric_row(report) for name, report in reports.items()}
    keys = ("mean_window_mpjpe_mm", "mean_window_measured_slip_mps", "worst_window_penetration_m",
            "mean_window_penetration_m", "long_boundary_angular_step_p95_deg", "long_joint_acceleration_p95_mps2",
            "long_boundary_joint_step_p95_mm", "long_worst_joint_step_deg", "long_command_velocity_error_mps",
            *[key for key in rows["original"] if key.startswith("command_")])
    gates = {key: (rows[candidate][key] is not None and rows["original"][key] is not None
                   and rows[candidate][key] <= rows["original"][key]) for key in keys}
    result = {"schema": "soma.rollout_supervision_review.v2", "verified_matched": True,
              "quality_approved": False, "post_processing": None, "metrics": rows, "decoder_comparison": decoder_comparison,
              "observation_comparison": observation_comparison,
              "training": results, "nonregression_vs_original": gates,
              "all_nonregression_gates": all(gates.values()),
              "paired_windows_vs_original": paired_windows(reports[candidate], reference),
              "limits": ["1000-update pilot, single training seed, four 8-frame commits per update",
                         "source-only target, executed-boundary velocity and rotation form a combined correction",
                         "cannot attribute gains to each subcomponent; not a full locomotion acceptance",
                         "no UE or installed model promotion"]}
    if decoder_comparison:
        result["limits"][1:3] = ["only decoder optimizer membership changes; both use source_v2 supervision",
                                  "the root-output policy remains provided_velocity; not full MB decoder root refinement"]
    if observation_comparison:
        result["limits"][1:3] = ["only observed input geometry changes in training and inference; both use source_v2",
                                  "published motion is untouched; decoder, codebooks and encoder remain frozen"]
    (folder / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    display = [("MPJPE (mm)", "mean_window_mpjpe_mm", 1),
               ("Source-masked slip (mm/s)", "mean_window_measured_slip_mps", 1000),
               ("Worst penetration (mm)", "worst_window_penetration_m", 1000),
               ("Long boundary joint-step P95 (mm)", "long_boundary_joint_step_p95_mm", 1),
               ("Long worst angular step (degrees)", "long_worst_joint_step_deg", 1),
               ("Command velocity error (m/s)", "long_command_velocity_error_mps", 1)]
    lines = ["# Executed-History Supervision Pilot", "", "No candidate is quality approved.", "",
             f"| Metric | Original | {names[0]} | {candidate} |", "|---|---:|---:|---:|"]
    for label, key, scale in display:
        lines.append("| " + label + " | " + " | ".join(
            "not assessed" if rows[n][key] is None else f"{rows[n][key] * scale:.3f}" for n in reports) + " |")
    lines += ["", "## Scope", "", *["- " + text for text in result["limits"]], "",
              "## Mechanism", "", "Ground-truth context is never replaced with predicted context before differentiation.",
              "Only input origin is shared. Absolute source position errors remain observable.",
              "FK/sole targets use global source coordinates, not a locally integrated origin that erases drift.",
              "Velocity and rotation increments start at actually executed history, not reconstructed context.",
              "No output correction, IK, locking, filtering or camera changes.", "",
              "## References", "", "- https://arxiv.org/html/2604.24833v1#S4.SS2",
              "- https://github.com/zkf1997/DART/blob/main/mld/train_mld.py (future/history separation and temporal consistency)",
              "- The MB straight-through adaptation and rotation-increment coefficient are experimental, not official DART reproduction."]
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--decoder-comparison", action="store_true")
    parser.add_argument("--observation-comparison", action="store_true")
    args = parser.parse_args()
    result = summarize(args.trial, args.decoder_comparison, args.observation_comparison)
    print(json.dumps({key: result[key] for key in ("metrics", "all_nonregression_gates", "quality_approved")}))
