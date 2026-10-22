"""Reject quality regressions automatically; never promote by token CE alone."""

import argparse
import json
from pathlib import Path


def summarize(folder):
    folders = {"Original": "review_original", "A": "review_A_v3", "B": "review_B_library",
               "B_no_goals": "review_B_no_goals"}
    reports = {name: json.loads((folder / sub / "metrics.json").read_text()) for name, sub in folders.items()}
    training = {name: json.loads((folder / "contracts" / name / "result.json").read_text()) for name in ("A", "B")}
    contracts = {name: json.loads((folder / "contracts" / name / "contract.json").read_text()) for name in ("A", "B")}
    for key in ("sampling_sha256", "steps_per_stage"):
        if training["A"][key] != training["B"][key]:
            raise ValueError("Unmatched optimizer/sampling budget")
    for key in ("initial_bundle_sha256", "dataset_sha256", "seed", "train_clips", "lr", "batch", "objectives"):
        if contracts["A"][key] != contracts["B"][key]:
            raise ValueError("Unmatched training contract: " + key)
    if not all(result["tokenizer_unchanged"] for result in training.values()):
        raise ValueError("Tokenizer changed")
    original = reports["Original"]
    identity = lambda r: [(w["filename"], w["start"], w["input_sha256"],
                           w["source_contact"], w["predicted_contact"]["valid_samples"]) for w in r["windows"]]
    for report in reports.values():
        if (identity(report) != identity(original) or report["evaluation_sha256"] != original["evaluation_sha256"]
                or report["test_used"] or report["post_processing"] is not None):
            raise ValueError("Scoring/data identity differs")
    rows = {}
    for name, report in reports.items():
        rows[name] = dict(report["aggregate"],
            raw_boundary_step_p95_mm=report["rollouts"]["0"]["boundary_joint_step_mm"]["p95"],
            raw_joint_step_max_deg=report["rollouts"]["0"]["angular_step_deg"]["max"],
            delayed_joint_step_max_deg=report["rollouts"]["12"]["angular_step_deg"]["max"],
            root_command_velocity_error_mean_mps=report["rollouts"]["0"]["root_command_velocity_error_mps"]["mean"])
    gates = {}
    for baseline in ("Original", "A", "B_no_goals"):
        gates[baseline] = {key: {"reference": value, "candidate": rows["B"][key],
                                 "no_regression": rows["B"][key] <= value}
                           for key, value in rows[baseline].items() if key != "windows" and value is not None}
    relative_gate = all(value["no_regression"] for value in gates["Original"].values())
    result = {"schema": "motionbricks.soma_pose_goals_review.v1", "arms": rows,
        "training": training, "matched_scoring_verified": True, "comparisons": gates,
        "raw_quality_no_regression_vs_original": relative_gate, "quality_approved": False,
        "deployment_decision": "not_approved" if relative_gate else "rejected_regression",
        "missing_deployment_evidence": ["multi-seed/long-rollout", "RTX3080 UE deadline and raw gait"],
        "interpretation": "Sparse goal contract restored; this bounded trial does NOT solve natural locomotion.",
        "remaining_architecture_deviation": "decoder still forcibly uses provided Root velocity; no native full-body root refinement"}
    (folder / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.folder), indent=2))


if __name__ == "__main__":
    main()
