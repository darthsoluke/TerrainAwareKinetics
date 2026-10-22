"""Compare stair-trained and baseline bundles on fixed flat walk/run windows."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FREE_START, FREE_END, ENDPOINT = 4, 62, 63


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_hashes(run):
    from scripts.evaluate_native_mb_lafan import _sha256

    run = Path(run).resolve()
    files = ("run_contract.json", "native_config.yaml", "training_result.json",
             "checkpoints/final.ckpt")
    return {name: _sha256(run / name) for name in files}


def _scalar_leaves(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            result.update(_scalar_leaves(value[key], f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, bool) or value is None:
        return {prefix: None}
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"Nonfinite measured metric: {prefix}")
        return {prefix: number}
    return {}


def summarize_metrics(windows):
    """Summarize aligned arm metrics by behavior; deltas are child minus baseline."""
    identities = [(row["shard"], row["start"]) for row in windows]
    if len(set(identities)) != len(identities):
        raise ValueError("Window metrics must have unique shard/start identities")
    grouped = {}
    for row in windows:
        behavior = row["behavior"]
        if behavior not in {"walk", "run"}:
            raise ValueError(f"Unexpected flat behavior: {behavior}")
        grouped.setdefault(behavior, []).append(row)
    summary = {}
    for behavior in sorted(grouped):
        rows = grouped[behavior]
        arm_values = {arm: [_scalar_leaves(row["metrics"][arm]) for row in rows]
                      for arm in ("source", "baseline", "child")}
        metric_names = sorted(set().union(*(set(item) for values in arm_values.values()
                                            for item in values)))
        arms, deltas = {}, {}
        for arm, values in arm_values.items():
            arms[arm] = {name: _mean_nullable([item.get(name) for item in values])
                         for name in metric_names}
        for name in metric_names:
            deltas[name] = _mean_nullable([
                (child[name] - baseline[name])
                if child.get(name) is not None and baseline.get(name) is not None else None
                for baseline, child in zip(arm_values["baseline"], arm_values["child"])
            ])
        summary[behavior] = {"windows": len(rows), "metrics": arms,
                             "child_minus_baseline": deltas}
    return summary


def _mean_nullable(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def review(source_dir, dataset_dir, baseline_root_run, baseline_pose_run,
           root_run, pose_run, output_dir, gpu, context_keyframes=4):
    from scripts.train_native_mb import select_gpu, verify_dataset

    gpu_info = select_gpu(gpu)
    from scripts.evaluate_native_mb_lafan import (
        _read_features, _sha256, select_validation_windows,
    )
    from scripts.review_native_mb_lafan_system import (
        raw_fk_diagnostics, score_motion, validate_output, validate_windows,
    )
    source_dir, dataset_dir = Path(source_dir).resolve(), Path(dataset_dir).resolve()
    runs = [Path(value).resolve() for value in
            (baseline_root_run, baseline_pose_run, root_run, pose_run)]
    output = validate_output(output_dir, [source_dir, dataset_dir, *runs])
    manifest_path = dataset_dir / "native_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_dataset(dataset_dir, manifest)
    windows, excluded = select_validation_windows(source_dir, dataset_dir)
    validate_windows(windows)

    import torch
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation
    from scripts.evaluate_native_mb_pipeline import load_bundle
    from scripts.review_native_mb_stairs import stair_prediction
    from scripts.review_native_mb_stair_sequence import generate_sequence

    if context_keyframes not in (1, 4):
        raise ValueError("Use one or four context keyframes")

    def predict(infer, raw):
        if context_keyframes == 4:
            return stair_prediction(infer, raw)
        canonical = infer.global_motion_rep.change_first_heading(raw, 0., False, False)
        prediction, _ = generate_sequence(infer, canonical[:, :4], [(63, canonical[:, 60:65])],
                                          context_count=1, goal_count=1, iterations=1)
        return canonical[:, :64], prediction

    torch.set_num_threads(4)
    torch.manual_seed(2333)
    np.random.seed(2333)
    probe_stats = {name: _sha256(dataset_dir / "stats" / name)
                   for name in ("mean.npy", "std.npy")}
    baseline_infer, baseline_dataset, baseline_bundle = load_bundle(*runs[:2], "cuda")
    child_infer, child_dataset, child_bundle = load_bundle(*runs[2:], "cuda")
    for label, model_dataset in (("baseline", Path(baseline_dataset)),
                                 ("child", Path(child_dataset))):
        model_stats = {name: _sha256(model_dataset / "stats" / name)
                       for name in ("mean.npy", "std.npy")}
        if model_stats != probe_stats:
            raise ValueError(f"{label} model normalization differs byte-for-byte from probe dataset")
    if baseline_infer.global_motion_rep.fps != 30 or child_infer.global_motion_rep.fps != 30:
        raise ValueError("Flat regression review requires native 30 Hz bundles")

    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        "accepted_json_sha256": _sha256(source_dir / "accepted.json"),
        "probe_manifest_sha256": _sha256(manifest_path),
        "probe_accepted_json_sha256": _sha256(dataset_dir / "accepted.json"),
    }
    bundles = {
        "baseline": {"metadata": baseline_bundle, "run_artifact_sha256": {
            "root": _run_hashes(runs[0]), "pose": _run_hashes(runs[1])}},
        "child": {"metadata": child_bundle, "run_artifact_sha256": {
            "root": _run_hashes(runs[2]), "pose": _run_hashes(runs[3])}},
    }
    report = {
        "schema": "native_mb_stair_flat_regression_v1",
        "status": "processing",
        "quality_approved": False,
        "gpu_at_launch": gpu_info,
        "fps": 30,
        "probe_normalization_sha256": probe_stats,
        "bundles": bundles,
        "source_and_probe_hashes": source_hashes,
        "conditions": {
            "predictor": "original motion_inference.predict; explicit context count",
            "context_keyframes": context_keyframes,
            "pose_context_frames": [0, 1, 2, 3, 63] if context_keyframes == 4 else [3, 63],
            "global_root_context_frames": [0, 1, 2, 3, 63] if context_keyframes == 4 else [3, 63],
            "local_velocity_context_intervals": [0, 1, 2] if context_keyframes == 4 else [2],
            "endpoint_velocity_interval": [63, 64],
            "endpoint_frame": ENDPOINT,
            "generated_free_frames_inclusive": [FREE_START, FREE_END],
            "source_interior_tokens_or_root_provided": False,
            "endpoint_scored_separately": True,
            "smoothing": False, "ik": False, "root_overwrite": False,
            "claims": "comparative flat-locomotion regression only; no contact or naturalness claims",
        },
        "excluded_validation_windows": excluded,
        "windows": [],
    }
    with full_precision_evaluation(), torch.inference_mode():
        for index, window in enumerate(windows):
            raw_np = _read_features(dataset_dir, window)
            raw = torch.from_numpy(raw_np).to("cuda")[None]
            source_features, baseline_features = predict(baseline_infer, raw)
            child_source_features, child_features = predict(child_infer, raw)
            if not torch.allclose(source_features, child_source_features, atol=1e-5, rtol=1e-5):
                raise ValueError("Bundle source feature decoding differs for the same probe window")
            baseline_rep = baseline_infer.global_motion_rep
            child_rep = child_infer.global_motion_rep
            source_motion = baseline_rep.inverse(source_features, False, return_all=True)
            baseline_motion = baseline_rep.inverse(baseline_features, False, return_all=True)
            child_motion = child_rep.inverse(child_features, False, return_all=True)
            motions = {"source": source_motion, "baseline": baseline_motion, "child": child_motion}
            root_index = baseline_rep.skeleton.root_idx
            if root_index != child_rep.skeleton.root_idx:
                raise ValueError("Baseline and child root skeleton identities differ")
            metrics = {
                arm: {
                    "scores": score_motion(motion, source_motion, root_index, 30),
                    "raw_fk_diagnostics": raw_fk_diagnostics(
                        motion["posed_joints"][0].detach().cpu().numpy(), root_index, 30),
                }
                for arm, motion in motions.items()
            }
            joints = {arm: motion["posed_joints"][0].detach().cpu().numpy()
                      for arm, motion in motions.items()}
            name = f"{index:02d}_{window['behavior']}_start{window['start']:06d}"
            np.savez_compressed(output / f"{name}_joints.npz", **joints,
                                fps=np.float32(30))
            source_shard = (source_dir / window["source_shard"]).resolve()
            probe_shard = (dataset_dir / window["shard"]).resolve()
            hashes = {
                "window_identity_sha256": _json_hash(window),
                "bundle_lineage_sha256": _json_hash(bundles),
                "source_accepted_json_sha256": source_hashes["accepted_json_sha256"],
                "probe_manifest_sha256": source_hashes["probe_manifest_sha256"],
                "probe_accepted_json_sha256": source_hashes["probe_accepted_json_sha256"],
                "probe_normalization_sha256": probe_stats,
                "source_shard_sha256": _sha256(source_shard),
                "probe_shard_sha256": _sha256(probe_shard),
                "source_features_sha256": hashlib.sha256(raw_np.tobytes()).hexdigest(),
            }
            row = {**window, "hashes": hashes, "metrics": metrics,
                   "joint_artifact": f"{name}_joints.npz"}
            (output / f"{name}.json").write_text(
                json.dumps(row, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            report["windows"].append(row)
            report["summary_by_behavior"] = summarize_metrics(report["windows"])
            (output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report["status"] = "completed_comparative_review"
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset", "baseline-root-run", "baseline-pose-run",
                 "root-run", "pose-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--context-keyframes", type=int, choices=(1, 4), default=4)
    args = parser.parse_args(argv)
    result = review(args.source, args.dataset, args.baseline_root_run,
                    args.baseline_pose_run, args.root_run, args.pose_run,
                    args.output, args.gpu, args.context_keyframes)
    print(json.dumps({"status": result["status"], "windows": len(result["windows"]),
                      "report": str(args.output.resolve() / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
