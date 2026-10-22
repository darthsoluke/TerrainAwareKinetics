"""Check paired trials and report raw-motion regressions without auto-deployment."""

import argparse
import json
from pathlib import Path


def summarize(folder, stage="decoder"):
    if stage not in ("decoder", "pose"):
        raise ValueError("Unknown matched trial stage")
    contracts = {arm: json.loads((folder / arm / "contract.json").read_text()) for arm in ("A", "B", "C")}
    results = {arm: json.loads((folder / arm / "result.json").read_text()) for arm in contracts}
    for key in ("initial_bundle_sha256", "dataset_sha256", "labels_manifest_sha256", "tokenizer_config_sha256",
                "seed", "steps", "batch", "primitives_per_update", "frames_per_primitive", "learning_rate",
                "loss_arguments", "frozen_state", "source_sha256"):
        if any(contracts[arm][key] != contracts["A"][key] for arm in ("B", "C")):
            raise ValueError("Unmatched training contract: " + key)
    if any(not r["frozen_unchanged"] or r["steps"] != 10000 for r in results.values()):
        raise ValueError("Incomplete training/frozen state failure")
    if len({r["sampling_sha256"] for r in results.values()}) != 1:
        raise ValueError("Sampling order differs")
    if stage == "pose":
        if any(contracts[arm].get("commit_frames") != 8 for arm in contracts):
            raise ValueError("Pose training must match UE's eight-frame prefix")
        if any(not r.get("pose_changed") for r in results.values()):
            raise ValueError("Pose network was not actually trained")
        if (contracts["A"]["history_policy"] != "teacher"
                or any(contracts[a]["history_policy"] != "scheduled_generated" for a in ("B", "C"))
                or [contracts[a]["decoded_motion_weight"] for a in ("A", "B", "C")] != [0., 0., 1.]):
            raise ValueError("Unmatched Pose/history/objective ablation")
    reports = {arm: json.loads((folder / ("review_" + arm) / "metrics.json").read_text())
               for arm in ("original", "A", "B", "C")}
    original = reports["original"]
    if original["bundle_sha256"] != contracts["A"]["initial_bundle_sha256"]:
        raise ValueError("Original scoring is not the paired initial bundle")
    def identity(report):
        return [(w["filename"], w["start"], w["input_sha256"],
                 w["native_root"]["source_contact"], w["native_root"]["predicted_contact"]["valid_samples"],
                 w["teacher_source_root"]["source_contact"], w["teacher_source_root"]["predicted_contact"]["valid_samples"])
                for w in report["windows"]]
    for arm, report in reports.items():
        frozen_keys = ("root", "encoder", "quantizer", "decoder") if stage == "pose" else ("root", "pose", "encoder", "quantizer")
        matching_frozen = (all(report["frozen_state"][key] == original["frozen_state"][key] for key in frozen_keys)
                           if isinstance(report["frozen_state"], dict) else report["frozen_state"] == original["frozen_state"])
        if (len(report["windows"]) != 57 or identity(report) != identity(original)
                or report["evaluation_sha256"] != original["evaluation_sha256"]
                or not matching_frozen or report["test_used"]
                or report["post_processing"] is not None):
            raise ValueError("Unmatched scoring or frozen model identity")
        if stage == "pose":
            if report.get("commit_frames") != contracts["A"]["commit_frames"]:
                raise ValueError("Evaluation prefix differs from deployed training contract")
            if (report["tokenizer_state"] != contracts["A"]["frozen_state"]["tokenizer"]
                    or report["frozen_state"]["root"] != contracts["A"]["frozen_state"]["root"]):
                raise ValueError("Scored frozen networks differ from training contract")
            if arm != "original" and report["frozen_state"]["pose"] == original["frozen_state"]["pose"]:
                raise ValueError("Scored Pose was not adapted")
        for index in ("0", "1", "2"):
            for key in ("initial_filename", "initial_frame"):
                if report["rollouts"][index][key] != original["rollouts"][index][key]:
                    raise ValueError("Long-rollout initial history differs")
        if arm != "original" and report["bundle_sha256"] != results[arm]["bundle_sha256"]:
            raise ValueError("Scored weights differ from trained weights")
    rows = {}
    for arm, report in reports.items():
        long = [report["rollouts"][str(i)] for i in range(3)]
        contact_heights = [w["native_root"]["predicted_contact"]["contact_height_abs_m"]["median"]
                           for w in report["windows"] if w["native_root"]["predicted_contact"]["contact_height_abs_m"] is not None]
        rows[arm] = {**report["aggregate"]["native_root"],
            "mean_window_contact_height_m": sum(contact_heights) / len(contact_heights) if contact_heights else None,
            "worst_window_penetration_m": max(w["native_root"]["predicted_contact"]["region_penetration_m"]["peak"]
                                               for w in report["windows"]),
            "teacher_root_mpjpe_mm": report["aggregate"]["teacher_source_root"]["mean_window_mpjpe_mm"],
            "long_boundary_joint_step_p95_mm": sum(r["boundary_joint_step_mm"]["p95"] for r in long) / 3,
            "long_worst_joint_step_deg": max(r["angular_step_deg"]["max"] for r in long),
            "long_command_velocity_error_mps": sum(r["root_command_velocity_error_mps"]["mean"] for r in long) / 3,
            "delayed_worst_joint_step_deg": report["rollouts"]["delay_12"]["angular_step_deg"]["max"]}
        for command in long[0]["per_command"]:
            rows[arm]["command_" + command + "_error_mps"] = sum(
                r["per_command"][command]["velocity_error_mps"]["mean"] for r in long) / 3
    comparisons = {}
    for candidate, baseline in (("B", "A"), ("C", "B"), ("B", "original"), ("C", "original")):
        comparisons[f"{candidate}_vs_{baseline}"] = {key: {"candidate": rows[candidate][key],
            "baseline": value, "no_regression": rows[candidate][key] is not None and rows[candidate][key] <= value}
            for key, value in rows[baseline].items() if value is not None}
    candidates = {arm: {"raw_no_regression": all(item["no_regression"] for item in comparisons[arm + "_vs_original"].values()),
                        "quality_approved": False} for arm in ("B", "C")}
    summary = {"schema": "motionbricks.pose_rollout_review.v1" if stage == "pose" else "motionbricks.decoder_refinement_review.v1", "arms": rows, "training": results,
        "comparisons": comparisons, "candidates": candidates, "matched_scoring_verified": True,
        "quality_approved": False, "default_bundle_changed": False,
        "deployment_decision": "not_approved" if any(r["raw_no_regression"] for r in candidates.values()) else "rejected_regression",
        "limitations": ["one training seed, pilot corpus", "four short-prefix primitives; frozen Root/tokenizer; biased straight-through gradient" if stage == "pose" else "two-primitive scheduled decoder history, frozen Pose",
                        "source endpoints are goals in paired diagnostics, not free WASD", "RTX3080 UE deadline not measured"],
        "post_processing": None}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--stage", choices=("decoder", "pose"), default="decoder")
    args = parser.parse_args()
    print(json.dumps(summarize(args.folder, args.stage), indent=2))


if __name__ == "__main__":
    main()
