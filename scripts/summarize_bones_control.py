"""Join fixed validation and actual free-control evidence without conflating the two."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_seed import build_representation
from motionbricks.motionlib.core.motion_reps.tools.heading import calc_heading
from scripts.evaluate_bones_control import COMMANDS, distribution


def combine(distributions):
    valid = [value for value in distributions if value["count"]]
    total = sum(value["count"] for value in valid)
    return {"count": total,
            "mean": sum(value["mean"] * value["count"] for value in valid) / total if total else None,
            "max_seed_p95_not_pooled_p95": max(value["p95"] for value in valid) if total else None,
            "peak": max(value["peak"] for value in valid) if total else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--control-runs", type=Path, required=True)
    parser.add_argument("--history-runs", type=Path, required=True)
    parser.add_argument("--pose-baseline-run", type=Path)
    args = parser.parse_args()
    root = args.control_runs / "root"
    runs = {"baseline": args.pose_baseline_run or args.control_runs / "pose-causal",
            "teacher": args.history_runs / "teacher", "generated": args.history_runs / "generated"}
    contracts = {name: json.loads((run / "run_contract.json").read_text()) for name, run in runs.items()}
    if contracts["teacher"]["initial_weights_sha256"] != contracts["generated"]["initial_weights_sha256"]:
        raise ValueError("Paired histories did not start from the same weights")
    reports = {name: json.loads((args.review / name / "metrics.json").read_text()) for name in runs}
    for key in ("root_checkpoint_sha256", "dataset_sha256", "commands", "post_processing", "horizon_frames", "commit_frames"):
        if len({json.dumps(report[key], sort_keys=True) for report in reports.values()}) != 1:
            raise ValueError("Non-paired control review: " + key)
    window_ids = [[(entry["filename"], entry["start"]) for entry in report["paired_validation_windows"]]
                  for report in reports.values()]
    if not window_ids[0] or any(ids != window_ids[0] for ids in window_ids):
        raise ValueError("Validation windows differ or are empty")
    rep = build_representation()
    summaries = {}
    for name, report in reports.items():
        if [r["seed"] for r in report["rollouts"]] != [2333, 2334, 2335] or any(r["frames"] != 3600 for r in report["rollouts"]):
            raise ValueError("Need the full three-seed, sixty-second control protocol")
        result = json.loads((runs[name] / "pilot_result.json").read_text())
        if result["steps"] != (20000 if name == "baseline" else 10000) or not result["frozen_tokenizer_unchanged"]:
            raise ValueError("Incorrect training budget or changed tokenizer")
        paired = {}
        for mode in ("control", "source_root_diagnostic"):
            windows = [entry[mode] for entry in report["paired_validation_windows"]]
            stable = sum(window["valid_samples"]["measured_stable_intervals"] for window in windows)
            paired[mode] = {"windows": len(windows),
                            "mpjpe_m": float(np.mean([window["mpjpe_m"] for window in windows])),
                            "root_xz_error_m": float(np.mean([window["root_xz_error_m"] for window in windows])),
                            "local_rotation_error_deg": float(np.mean([window["local_rotation_error_deg"] for window in windows])),
                            "joint_acceleration_error_mps2": float(np.mean([window["joint_acceleration_error_mps2"] for window in windows])),
                            "measured_stable_intervals": stable,
                            "measured_slip_mps": sum(window["measured_slip_mps"]["mean"] * window["valid_samples"]["measured_stable_intervals"]
                                                      for window in windows if window["measured_slip_mps"] is not None) / stable if stable else None,
                            "slip_peak_mps": max((window["measured_slip_mps"]["peak"] for window in windows
                                                  if window["measured_slip_mps"]), default=None)}
        free = {key: combine([rollout[key] for rollout in report["rollouts"]]) for key in
                ("velocity_error_mps", "heading_error_deg", "sole_penetration_m", "boundary_joint_step_m",
                 "interior_joint_step_m", "displacement_tracking_error_m", "generation_seconds_per_4_committed_frames")}
        free["near_ground_material_speed_mps"] = combine([
            rollout["feet"][side]["near_ground_material_speed_mps"]
            for rollout in report["rollouts"] for side in ("Left", "Right")])
        free["mean_measured_vertex_interval_fraction"] = float(np.mean([
            rollout["feet"][side]["measured_vertex_interval_fraction"]
            for rollout in report["rollouts"] for side in ("Left", "Right")]))
        free["mean_both_feet_clearance_over_3cm_fraction"] = float(np.mean([
            rollout["both_feet_above_3cm_fraction"] for rollout in report["rollouts"]]))
        phases = {}
        for start, stop, phase, command, heading in COMMANDS:
            speed_error, actual_speeds, bone_heading, heading_disagreement = [], [], [], []
            for rollout in report["rollouts"]:
                with np.load(args.review / name / f"rollout_{rollout['seed']}.npz", allow_pickle=False) as data:
                    root = data["global_features"][:, :3]
                    speed = torch.from_numpy(np.diff(root, axis=0)[:, [0, 2]] * 60)
                    start_frame, end_frame = max(start * 60 - 1, 0), stop * 60 - 1
                    part = speed[start_frame:end_frame]
                    actual_speeds.append(part.norm(dim=-1))
                    speed_error.append((part - torch.tensor(command)).norm(dim=-1))
                    joints = torch.from_numpy(data["joints"][start * 60:stop * 60])[None]
                    actual_heading = calc_heading("hips_pos", posed_joints=joints, skeleton=rep.skeleton)[0]
                    bone_heading.append(torch.atan2((actual_heading - heading).sin(), (actual_heading - heading).cos()).abs() * 180 / np.pi)
                    features = data["global_features"][start * 60:stop * 60]
                    feature_heading = torch.from_numpy(np.arctan2(features[:, 4], features[:, 3]))
                    diff = actual_heading - feature_heading
                    heading_disagreement.append(torch.atan2(diff.sin(), diff.cos()).abs() * 180 / np.pi)
            phases[f"{phase}_{start}s"] = {"velocity_error_mps": distribution(torch.cat(speed_error)),
                                          "actual_speed_mps": distribution(torch.cat(actual_speeds)),
                                          "skeletal_heading_error_deg": distribution(torch.cat(bone_heading)),
                                          "skeletal_vs_root_feature_heading_error_deg": distribution(torch.cat(heading_disagreement))}
        summaries[name] = {"training": result, "paired": paired, "free": free, "phases": phases}
    records = json.loads((args.dataset / "accepted.json").read_text())
    train = [record for record in records if record["split"] == "train"]
    behaviors = {name: sum(record["behavior"] == name for record in train)
                 for name in ("idle", "walk", "run", "turn", "backward", "lateral")}
    coverage = {name: sum(name in record["filename"].lower() for record in train) for name in ("start", "stop", "turn")}
    summary = {"schema": "motionbricks.soma_control_summary.v1", "arms": summaries,
               "train_behavior_clip_counts": behaviors, "filename_transition_candidate_counts_not_labels": coverage,
               "paired_initial_weights_equal": True, "post_processing": None, "quality_approved": False,
               "conclusion": "Continuous generation implemented and evaluated; natural locomotion and no-slip are NOT established",
               "scope": "152 train clips; fixed validation only; test unused; no UE changes or foot corrections"}
    (args.review / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ["# SOMA command-driven locomotion", "", "Raw model generation, not tokenizer reconstruction. Quality not approved.",
             "", "Root: 10,000 updates. Pose baseline: 20,000. Teacher/generated continuations: 10,000 each.",
             "Tokenizer and codebooks stayed frozen. Teacher/generated initialization identical.",
             "All arms: 57 fixed validation windows and 3 independent 60-second command rollouts.",
             "", "| Arm | Source-root MPJPE (mm) | Actual-root MPJPE (mm) | Free velocity error (m/s) | Near-ground point speed (m/s) | Penetration mean (mm) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, value in summaries.items():
        lines.append(f"| {name} | {value['paired']['source_root_diagnostic']['mpjpe_m'] * 1000:.1f} | "
                     f"{value['paired']['control']['mpjpe_m'] * 1000:.1f} | {value['free']['velocity_error_mps']['mean']:.3f} | "
                     f"{value['free']['near_ground_material_speed_mps']['mean']:.3f} | {value['free']['sole_penetration_m']['mean'] * 1000:.3f} |")
    lines += ["", "Source-root is an oracle diagnostic and not ordinary player-control performance.",
              "Free point-speed uses fixed geometry criteria, not source contact masks; coverage is recorded in summary.json.",
              "No IK, foot locking, root correction, smoothing, future body truth, or periodic source resets.",
              "", "Complete videos: each arm/rollouts_full.mp4 (three seeds, full sixty seconds).",
              "Worst official-skinned excerpts: each arm/worst_skinned.mp4; raw arrays and inputs: each arm/rollout_*.npz.",
              "", "Unresolved: stop drift, transition bursts, foot sliding, and body prediction error. Longer training alone is not proven to fix these.",
              "Deployment remains unapproved. GPU timing in these reports is remote RTX 4090, not RTX 3080."]
    (args.review / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
