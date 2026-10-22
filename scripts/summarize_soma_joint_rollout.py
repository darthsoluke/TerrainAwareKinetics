"""Verify paired Root/Pose training and raw evaluation; never promote weights."""

import argparse
import json
from pathlib import Path

import numpy as np


MATCHED_KEYS = (
    "schema", "initial_bundle_sha256", "dataset_sha256", "labels_manifest_sha256",
    "tokenizer_config_sha256", "seed", "steps", "batch", "primitives_per_update",
    "frames_per_primitive", "learning_rate", "optimizer", "loss_arguments",
    "source_sha256", "initial_pose_state", "initial_root_state", "history_policy",
    "teacher_steps", "history_ramp_steps", "motion_ce_weight", "decoded_motion_weight",
    "commit_frames", "history_frames", "motion_fps", "gradient_clip_norm",
    "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_deterministic",
    "paired_attention_kernel", "root_condition_policy", "network_mode",
)


def read(path):
    return json.loads(path.read_text())


def window_identity(report):
    return [(w["filename"], w["start"], w["input_sha256"],
             *[(w[k]["source_contact"], w[k]["predicted_contact"]["valid_samples"])
               for k in ("native_root", "teacher_source_root")]) for w in report["windows"]]


def metric_row(report):
    windows, long = report["windows"], [report["rollouts"][str(i)] for i in range(3)]
    heights = [w["native_root"]["predicted_contact"]["contact_height_abs_m"]["median"]
               for w in windows if w["native_root"]["predicted_contact"]["contact_height_abs_m"] is not None]
    row = {**report["aggregate"]["native_root"],
        "mean_window_contact_height_m": float(np.mean(heights)) if heights else None,
        "worst_window_penetration_m": max(w["native_root"]["predicted_contact"]["region_penetration_m"]["peak"] for w in windows),
        "teacher_root_mpjpe_mm": report["aggregate"]["teacher_source_root"]["mean_window_mpjpe_mm"],
        "long_boundary_joint_step_p95_mm": float(np.mean([r["boundary_joint_step_mm"]["p95"] for r in long])),
        "long_boundary_angular_step_p95_deg": float(np.mean([r["boundary_angular_step_deg"]["p95"] for r in long])),
        "long_worst_joint_step_deg": max(r["angular_step_deg"]["max"] for r in long),
        "long_joint_acceleration_p95_mps2": float(np.mean([r["joint_acceleration_mps2"]["p95"] for r in long])),
        "long_command_velocity_error_mps": float(np.mean([r["root_command_velocity_error_mps"]["mean"] for r in long])),
        "delayed_worst_joint_step_deg": report["rollouts"]["delay_12"]["angular_step_deg"]["max"]}
    for command in long[0]["per_command"]:
        row["command_" + command + "_error_mps"] = float(np.mean([
            r["per_command"][command]["velocity_error_mps"]["mean"] for r in long]))
    if any(v is not None and not np.isfinite(v) for v in row.values()):
        raise ValueError("Nonfinite quality metric")
    return row


def paired_windows(candidate, baseline):
    getters = {
        "mpjpe_mm": lambda w: w["native_root"]["root_relative_mpjpe_mm"]["mean"],
        "root_error_m": lambda w: w["native_root"]["root_position_error_m"]["mean"],
        "worst_joint_step_deg": lambda w: w["native_root"]["angular_step_deg"]["max"],
        "boundary_joint_step_p95_mm": lambda w: w["native_root"]["boundary_joint_step_mm"]["p95"],
        "penetration_m": lambda w: w["native_root"]["predicted_contact"]["region_penetration_m"]["mean"],
        "slip_mps": lambda w: (w["native_root"]["predicted_contact"]["measured_slip_mps"] or {}).get("mean"),
    }
    rows = []
    for c, b in zip(candidate["windows"], baseline["windows"]):
        values = {}
        for name, getter in getters.items():
            cv, bv = getter(c), getter(b)
            values[name] = None if cv is None or bv is None else cv - bv
        rows.append({"filename": c["filename"], "start": c["start"], "behavior": c.get("behavior"), "deltas": values})
    summaries = {}
    for name in getters:
        deltas = [r["deltas"][name] for r in rows if r["deltas"][name] is not None]
        summaries[name] = ({"n": len(deltas), "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)), "p95_delta": float(np.quantile(deltas, .95)),
            "improved": sum(d < 0 for d in deltas), "regressed": sum(d > 0 for d in deltas)}
            if deltas else {"n": 0, "mean_delta": None})
    return {"windows": rows, "summary": summaries,
            "scope": "paired descriptive windows, not independent training seeds or significance evidence"}


def write_report(folder, result, contracts):
    metrics = (
        ("Held-out MPJPE (mm)", "mean_window_mpjpe_mm", 1),
        ("Held-out measured slip (mm/s)", "mean_window_measured_slip_mps", 1000),
        ("Mean sole penetration (mm)", "mean_window_penetration_m", 1000),
        ("Worst sole penetration (mm)", "worst_window_penetration_m", 1000),
        ("Held-out worst joint step (deg/frame)", "worst_joint_step_deg", 1),
        ("Long-rollout boundary joint step P95 (mm)", "long_boundary_joint_step_p95_mm", 1),
        ("Long-rollout boundary angular step P95 (deg/frame)", "long_boundary_angular_step_p95_deg", 1),
        ("Long-rollout worst angular step (deg/frame)", "long_worst_joint_step_deg", 1),
        ("Long-rollout command error (m/s)", "long_command_velocity_error_mps", 1),
        ("Idle command error (m/s)", "command_idle_error_mps", 1),
        ("Stop command error (m/s)", "command_stop_error_mps", 1),
        ("Final stop command error (m/s)", "command_stop_again_error_mps", 1),
    )
    lines = ["# Coupled Root/Pose Continuation Result", "",
        "## Outcome", "", f"Decision: `{result['deployment_decision']}`.",
        "No model is quality approved; the installed default bundle is unchanged.", "",
        "Original is the previous trained backend, NOT source motion capture.",
        "A adapts Pose only; B adapts Root and Pose. The complete tokenizer remains frozen.", "",
        "## Raw Metrics", "", "| Metric | Original | A | B |", "|---|---:|---:|---:|"]
    for label, key, scale in metrics:
        values = [result["arms"][arm].get(key) for arm in ("Original", "A", "B")]
        lines.append("| " + label + " | " + " | ".join("unavailable" if v is None else f"{v * scale:.3f}" for v in values) + " |")
    lines += ["", "P95 rows are means of per-history P95 values, not pooled P95.",
        "Held-out geometry uses identical SOURCE evidence; the free rollouts have no aligned ground truth.",
        "All 57 windows and all three 60-second control sequences are included.", "",
        "## Matched Training", "", "| Arm | Updates | Seconds | Peak allocated MiB | Root changed |", "|---|---:|---:|---:|---|"]
    for arm, r in result["training"].items():
        lines.append(f"| {arm} | {r['steps']} | {r.get('elapsed_seconds', 0):.1f} | {r.get('peak_gpu_memory_mb', 0):.1f} | {r['root_changed']} |")
    c = contracts["A"]
    lines += ["", f"Shared initial bundle SHA256: `{c['initial_bundle_sha256']}`.",
        f"Shared dataset SHA256: `{c['dataset_sha256']}`.",
        f"Shared sampling SHA256: `{result['training']['A']['sampling_sha256']}`.",
        "Sampling and history curriculum, source hashes, objectives, frozen tokenizer and scoring identities were automatically verified.",
        "The compared variable is Root optimizer membership; both arms retain the same global gradient clipping rule.",
        "Training uses TF32 off and scoped unfused/math attention. Evaluation independently uses default deployed kernels.",
        "Early mode/precision failures are retained in smoke/full logs; they are not merged into the final matched runs.", "",
        "## Per-Window Changes: B vs A", "", "| Metric | Improved | Regressed | Valid windows |", "|---|---:|---:|---:|"]
    for key, p in result["paired_windows"]["B_vs_A"]["summary"].items():
        lines.append(f"| {key} | {p.get('improved', 0)} | {p.get('regressed', 0)} | {p['n']} |")
    lines += ["", "## Training Time", "",
        "Each update executes four sequential 64-frame primitives, each with a no-grad native reference,",
        "a differentiable native pass, decoded-motion/sole supervision and backward propagation.",
        "10,000 updates therefore execute 40,000 primitives per arm (batch size four).",
        "The tokenizer is frozen but gradients through its decoder and selected sole skinning still cost computation.",
        "Training-only paired FP32/math attention avoids forward-parity failures; deployment uses independent default kernels.",
        "The independent two-update profile includes startup, checkpoint saving and profiling overhead,",
        "and shows many small launches, copies and host/device synchronizations. It is NOT a steady-state bottleneck percentage.",
        "Logging and gradient checks can be batched to reduce synchronization without removing automatic checks.",
        "No performance change was introduced mid-trial. These times are NOT UE inference latency."]
    lines += ["", "## Artifacts", "", "- `summary.json`: verification, all metrics, decisions and per-window deltas.",
        "- `review_{Original,A,B}/metrics.json`: all held-out and free-control metrics.",
        "- `review_{Original,A,B}/long_{0,1,2}.npz`: untouched 60-Hz outputs.",
        "- `render/rollouts_full.mp4`: complete Original/A/B raw first-history comparison at 30 fps.",
        "- `render/worst_skinned.mp4`: worst penetrating excerpt using official SOMA skin.",
        "- `render/free_geometry.json`: first-history near-ground material-point speed, penetration and coverage; NOT support truth.", "",
        "- `render_worst_angular/rollouts_full.mp4`: matched two-second worst angular excerpt, all adjacent 60-Hz frames.",
        "- `render_worst_angular/worst_skinned.mp4`: the same raw failure with official SOMA skin.",
        "- `render_worst_angular/{before_flip,after_flip}.png`: adjacent source frames, not smoothed output.", "",
        "## Basis And Limits", "",
        "[MotionBricks](https://arxiv.org/html/2604.24833v1) supplies the native Root/Pose/conditioned-decoder architecture.",
        "[DART](https://arxiv.org/html/2410.05260v1) motivates generated-history training; our joint discrete predictor adaptation is not its diffusion method.",
        *["- " + item for item in result["limitations"]],
        "- No IK, foot locking, filtering, root correction or camera compensation.",
        "- A rejected candidate does not solve locomotion, regardless of loss reduction or partial metric gains.", ""]
    (folder / "REPORT.md").write_text("\n".join(lines))


def summarize(folder, prefix="full_v2"):
    contracts = {a: read(folder / f"{prefix}_{a}" / "contract.json") for a in ("A", "B")}
    results = {a: read(folder / f"{prefix}_{a}" / "result.json") for a in contracts}
    a, b = contracts["A"], contracts["B"]
    for key in MATCHED_KEYS:
        if a[key] != b[key]:
            raise ValueError("Unmatched training contract: " + key)
    if (any(not c["coupled_root_trial"] for c in contracts.values()) or a["train_root"] or not b["train_root"]
            or a["history_policy"] != "scheduled_generated" or a["decoded_motion_weight"] != 1.
            or a["commit_frames"] != 8 or a["motion_fps"] != 60):
        raise ValueError("Incorrect coupled continuation protocol")
    if a["frozen_state"]["tokenizer"] != b["frozen_state"]["tokenizer"]:
        raise ValueError("Initial tokenizers differ")
    if any(r["steps"] != 10000 or not r["frozen_unchanged"] or not r["pose_changed"] for r in results.values()):
        raise ValueError("Incomplete training or frozen-state failure")
    if results["A"]["root_changed"] or not results["B"]["root_changed"]:
        raise ValueError("Root state change differs from the experiment")
    if results["A"]["sampling_sha256"] != results["B"]["sampling_sha256"] or results["A"]["history_counts"] != results["B"]["history_counts"]:
        raise ValueError("Sampling/history curriculum differs")
    reports = {a: read(folder / f"review_{a}" / "metrics.json") for a in ("Original", "A", "B")}
    original = reports["Original"]
    if original["bundle_sha256"] != a["initial_bundle_sha256"]:
        raise ValueError("Original weights are not the paired starting point")
    for arm, report in reports.items():
        if (len(report["windows"]) != 57 or window_identity(report) != window_identity(original)
                or report["evaluation_sha256"] != original["evaluation_sha256"]
                or report["dataset_sha256"] != a["dataset_sha256"] or report["commit_frames"] != 8
                or report["test_used"] or report["post_processing"] is not None):
            raise ValueError("Unmatched scoring/data/evidence")
        if report["tokenizer_state"] != a["frozen_state"]["tokenizer"]:
            raise ValueError("Scored tokenizer was changed")
        expected_root_equal = arm != "B"
        if (report["frozen_state"]["root"] == a["initial_root_state"]) != expected_root_equal:
            raise ValueError("Scored Root differs from training protocol")
        for key in ("encoder", "quantizer", "decoder"):
            if report["frozen_state"][key] != original["frozen_state"][key]:
                raise ValueError("Scored frozen module changed: " + key)
        for index in ("0", "1", "2"):
            for key in ("initial_filename", "initial_frame"):
                if report["rollouts"][index][key] != original["rollouts"][index][key]:
                    raise ValueError("Long-rollout initial history differs")
        if arm != "Original" and report["bundle_sha256"] != results[arm]["bundle_sha256"]:
            raise ValueError("Scored checkpoint differs from training result")
    rows = {arm: metric_row(report) for arm, report in reports.items()}
    comparisons, paired = {}, {}
    for candidate, baseline in (("A", "Original"), ("B", "Original"), ("B", "A")):
        name = f"{candidate}_vs_{baseline}"
        comparisons[name] = {k: {"baseline": v, "candidate": rows[candidate][k],
            "delta": None if rows[candidate][k] is None else rows[candidate][k] - v,
            "no_regression": rows[candidate][k] is not None and rows[candidate][k] <= v}
            for k, v in rows[baseline].items() if v is not None}
        paired[name] = paired_windows(reports[candidate], reports[baseline])
    candidates = {}
    for arm in ("A", "B"):
        comparison = comparisons[f"{arm}_vs_Original"]
        no_regression = all(v["no_regression"] for v in comparison.values())
        improved = all(k in comparison and comparison[k]["delta"] is not None and comparison[k]["delta"] < 0
            for k in ("mean_window_measured_slip_mps", "long_boundary_angular_step_p95_deg", "long_worst_joint_step_deg"))
        candidates[arm] = {"raw_no_regression": no_regression, "slip_and_continuity_improved": improved,
                           "eligible_for_ue_review": no_regression and improved, "quality_approved": False}
    result = {"schema": "motionbricks.joint_rollout_review.v1", "training": results,
        "arms": rows, "comparisons": comparisons, "paired_windows": paired, "candidates": candidates,
        "matched_scoring_verified": True, "quality_approved": False, "default_bundle_changed": False,
        "deployment_decision": "ue_review_only" if any(c["eligible_for_ue_review"] for c in candidates.values()) else "rejected_regression_or_no_gain",
        "limitations": ["one training seed; pilot corpus", "Root/Pose joint adaptation is our experiment, not official weights",
            "source-root goals and teacher histories in held-out diagnostics are privileged",
            "three free-control initial histories are not three training seeds", "candidate source-derived contact labels",
            "training uses paired FP32/math kernels; evaluation uses default deployed kernels", "UE 30fps not assessed in this experiment"],
        "post_processing": None}
    (folder / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    write_report(folder, result, contracts)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--training-prefix", default="full_v2")
    args = parser.parse_args()
    result = summarize(args.folder, args.training_prefix)
    print(json.dumps({"arms": result["arms"], "candidates": result["candidates"]}, indent=2))
