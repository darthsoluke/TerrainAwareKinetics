"""Bounded held-out LAFAN validation for native Tokenizer and Root/Pose runs."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GLOBAL_WINDOW_FRAMES = 65
DEFAULT_STAGE_STEPS = 2_000_000


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _validate_expected_stage_steps(expected_stage_steps):
    if (isinstance(expected_stage_steps, bool) or not isinstance(expected_stage_steps, int)
            or expected_stage_steps <= 0):
        raise ValueError("expected_stage_steps must be a positive integer")


def validate_stage_budget(run_path, stage, expected_stage_steps=DEFAULT_STAGE_STEPS):
    """Bind a final checkpoint and its recorded updates to the exact review budget."""
    _validate_expected_stage_steps(expected_stage_steps)
    if stage not in {"tokenizer", "root", "pose"}:
        raise ValueError("Expected Tokenizer, Root, or Pose stage")
    import torch

    run_path = Path(run_path).resolve()
    contract = json.loads((run_path / "run_contract.json").read_text(encoding="utf-8"))
    result = json.loads((run_path / "training_result.json").read_text(encoding="utf-8"))
    allowed_status = {"completed"} if stage == "tokenizer" else {"completed", "trained_not_reviewed"}
    if contract.get("stage") != "native_" + stage or contract.get("status") not in allowed_status:
        raise ValueError(f"LAFAN review requires a completed {stage} training run")
    checkpoint = run_path / "checkpoints/final.ckpt"
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != result.get("final_checkpoint_sha256"):
        raise ValueError(f"{stage} checkpoint hash does not match training_result.json")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    global_step = saved.get("global_step") if isinstance(saved, dict) else None
    updates = result.get("optimizer_updates")
    if (isinstance(global_step, bool) or not isinstance(global_step, int)
            or isinstance(updates, bool) or not isinstance(updates, int)
            or global_step != expected_stage_steps or updates != expected_stage_steps):
        raise ValueError(f"{stage} checkpoint global_step and optimizer_updates must both equal "
                         f"expected_stage_steps={expected_stage_steps}")
    return {"run": str(run_path), "checkpoint_sha256": checkpoint_hash,
            "global_step": global_step, "optimizer_updates": updates,
            "config_sha256": contract["config_sha256"]}


def _validate_frozen_tokenizer(tokenizer_info, tokenizer_state, pose_run):
    pose_run = Path(pose_run)
    contract = json.loads((pose_run / "run_contract.json").read_text(encoding="utf-8"))
    result = json.loads((pose_run / "training_result.json").read_text(encoding="utf-8"))
    if (not tokenizer_info.get("checkpoint_sha256") or not tokenizer_state
            or (contract.get("tokenizer") or {}).get("checkpoint_sha256")
                != tokenizer_info["checkpoint_sha256"]
            or contract.get("tokenizer_state_sha256") != tokenizer_state
            or result.get("tokenizer_state_sha256") != tokenizer_state):
        raise ValueError("Pose must bind the reviewed tokenizer checkpoint and frozen tensor identity")


def select_validation_windows(source_dir, dataset_dir):
    """Choose up to three deterministic windows per actual subject4 walk/run clip."""
    source_dir, dataset_dir = Path(source_dir).resolve(), Path(dataset_dir).resolve()
    source_rows = json.loads((source_dir / "accepted.json").read_text(encoding="utf-8"))
    source_by_shard = {Path(row["shard"]).as_posix(): row for row in source_rows}
    if len(source_by_shard) != len(source_rows):
        raise ValueError("Source accepted manifest contains duplicate shard paths")
    accepted = json.loads((dataset_dir / "accepted.json").read_text(encoding="utf-8"))
    rows, excluded = [], []
    for row in accepted:
        if row.get("split") != "validation":
            continue
        original = source_by_shard.get(Path(row.get("source_shard", "")).as_posix())
        if original is None:
            original = next((item for item in source_rows if item.get("filename") == row.get("filename")), None)
        if original is None or original.get("subject_id") != "subject4":
            continue
        behavior = row.get("behavior", original.get("behavior"))
        if behavior not in {"walk", "run"}:
            continue
        shard_path = (dataset_dir / row["shard"]).resolve()
        if dataset_dir not in shard_path.parents or not shard_path.is_file():
            raise ValueError(f"Held-out native shard is missing or escapes dataset: {row['shard']}")
        with np.load(shard_path, allow_pickle=False) as shard:
            frame_count = len(shard["global_features"])
        if frame_count < GLOBAL_WINDOW_FRAMES:
            excluded.append({"shard": row["shard"], "behavior": behavior,
                             "frames": frame_count, "reason": "shorter than 65-frame evaluation window"})
            continue
        starts = sorted(set(np.linspace(0, frame_count - GLOBAL_WINDOW_FRAMES, 3).astype(int).tolist()))
        source_shard_path = (source_dir / original["shard"]).resolve()
        if source_dir not in source_shard_path.parents or not source_shard_path.is_file():
            raise ValueError(f"Held-out source shard is missing or escapes source: {original['shard']}")
        for start in starts:
            rows.append({"shard": row["shard"], "shard_sha256": _sha256(shard_path),
                         "source_shard": original["shard"],
                         "source_shard_sha256": _sha256(source_shard_path),
                         "filename": row.get("filename", original.get("filename")),
                         "subject_id": "subject4", "split": "validation",
                         "behavior": behavior, "start": int(start)})
    if not rows:
        raise ValueError("Subject4 validation has no eligible walk/run windows")
    return rows, excluded


def _read_features(dataset_dir, window):
    with np.load(Path(dataset_dir) / window["shard"], allow_pickle=False) as data:
        raw = data["global_features"][window["start"]:
                                      window["start"] + GLOBAL_WINDOW_FRAMES].copy()
    if raw.shape[0] != GLOBAL_WINDOW_FRAMES or not np.isfinite(raw).all():
        raise ValueError(f"Invalid feature window: {window}")
    return raw


def _joint_arrays(rep, feature_source, feature_prediction):
    source = rep.inverse(feature_source, False, return_all=True)
    prediction = rep.inverse(feature_prediction, False, return_all=True)
    source_joints = source["posed_joints"][0].detach().cpu().numpy()
    prediction_joints = prediction["posed_joints"][0].detach().cpu().numpy()
    if (source_joints.shape != prediction_joints.shape or source_joints.ndim != 3
            or source_joints.shape[1] != 30
            or not np.isfinite(source_joints).all() or not np.isfinite(prediction_joints).all()):
        raise ValueError("Native inference returned invalid full-body SOMA positions")
    return source, prediction


def _render_set(folder, arrays, metrics):
    folder.mkdir(parents=True, exist_ok=False)
    for behavior, (source, prediction, fps) in arrays.items():
        np.savez_compressed(folder / f"{behavior}_reconstruction.npz",
                            source=source, prediction=prediction, fps=np.float32(fps))
        metrics["examples"][behavior] = next(row for row in metrics["windows"]
                                               if row["behavior"] == behavior)
    _write_json(folder / "metrics.json", metrics)
    vendor = Path(__file__).resolve().parents[1] / "out/vendor/kimodo"
    if vendor.is_dir() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from scripts.render_native_mb_review import _render_motion_set

    stage = metrics.get("stage")
    outputs = []
    for behavior, (source, prediction, fps) in arrays.items():
        path = folder / f"{behavior}_reconstruction.npz"
        if stage == "tokenizer":
            title = (f"{behavior} | source-encoded tokenizer reconstruction\n"
                     "no pose keyframes | source root decoder condition")
            labels = ("Source motion", "Tokenizer reconstruction")
        else:
            title = (f"{behavior} | generated tokens + predicted root\n"
                     "poses 3/63 only | no runtime/post-hoc repair")
            labels = ("Source motion", "Generated pose")
        outputs.append(_render_motion_set(path, (source, prediction), fps, folder,
                                          title, panel_labels=labels))
    return {"task": metrics["task"], "quality_approved": False, "gifs": outputs}


def evaluate_lafan(source_dir, dataset_dir, tokenizer_run, root_run, pose_run, output_dir,
                   expected_stage_steps=DEFAULT_STAGE_STEPS):
    _validate_expected_stage_steps(expected_stage_steps)
    import torch
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation
    from scripts.evaluate_native_mb_checkpoint import load_native_checkpoint, reconstruction_metrics
    from scripts.evaluate_native_mb_inbetween import predict_window
    from scripts.evaluate_native_mb_pipeline import load_bundle
    from scripts.train_native_mb import sha256
    from scripts.train_native_mb_control import state_digest

    source_dir, dataset_dir, output_dir = map(Path, (source_dir, dataset_dir, output_dir))
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir == dataset_dir.resolve() or dataset_dir.resolve() in output_dir.parents:
        raise FileExistsError("Review output must be a fresh directory outside the dataset")
    stage_hashes = {stage: validate_stage_budget(run, stage, expected_stage_steps)
                    for stage, run in (("tokenizer", tokenizer_run), ("root", root_run), ("pose", pose_run))}
    budget = {"expected_stage_steps": expected_stage_steps,
              "stage_steps": {stage: info["global_step"] for stage, info in stage_hashes.items()}}
    output_dir.mkdir(parents=True, exist_ok=False)
    windows, excluded = select_validation_windows(source_dir, dataset_dir)
    if {row["behavior"] for row in windows} != {"walk", "run"}:
        raise ValueError("Subject4 validation requires at least one eligible walk and run window")
    torch.manual_seed(2333)
    np.random.seed(2333)

    tokenizer, tokenizer_info, binding = load_native_checkpoint(tokenizer_run, device="cuda")
    if (tokenizer_info["global_step"] != expected_stage_steps
            or tokenizer_info["run_status_at_review"] != "completed"
            or tokenizer_info["checkpoint_sha256"] != stage_hashes["tokenizer"]["checkpoint_sha256"]):
        raise ValueError("LAFAN review requires the completed tokenizer at the exact stage budget")
    tokenizer_state = state_digest(tokenizer.pose_net)
    _validate_frozen_tokenizer(tokenizer_info, tokenizer_state, pose_run)
    tokenizer_rep, local_rep = tokenizer.motion_rep, tokenizer.local_motion_rep
    tokenizer_rows, tokenizer_arrays = [], {}
    tokenizer.pose_net.eval()
    with full_precision_evaluation(), torch.inference_mode():
        for window in windows:
            raw_np = _read_features(dataset_dir, window)
            raw = torch.from_numpy(raw_np).to("cuda")[None]
            raw = tokenizer_rep.change_first_heading(raw, 0.0, False, False)
            normalized = tokenizer_rep.normalize(raw)
            local = tokenizer_rep.dual_rep.global_to_local(
                normalized, True, True,
                lengths=torch.tensor([GLOBAL_WINDOW_FRAMES], device=raw.device),
            )[:, :GLOBAL_WINDOW_FRAMES - 1]
            external = extract_feature_from_motion_rep(
                local, local_rep, tokenizer.pose_net.decoder_external_cond_feature_mode
            )
            prediction = tokenizer.pose_net(
                local, target_cond=None, has_target_cond=None, external_cond=external
            )["recon_state"]
            target_pose = local_rep.inverse(local, True, return_all=True)
            predicted_pose = local_rep.inverse(prediction, True, return_all=True)
            scores = reconstruction_metrics(predicted_pose, target_pose,
                                             tokenizer_rep.skeleton.root_idx, tokenizer_rep.fps)
            tokenizer_rows.append({**window, "conditioning": "none", **scores})
            if window["behavior"] not in tokenizer_arrays:
                tokenizer_arrays[window["behavior"]] = (
                    target_pose["posed_joints"][0].cpu().numpy().copy(),
                    predicted_pose["posed_joints"][0].cpu().numpy().copy(),
                    float(tokenizer_rep.fps))
    del tokenizer, tokenizer_rep, local_rep
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer_metrics = {
        "task": "subject4 LAFAN source-encoded native tokenizer reconstruction",
        "stage": "tokenizer", "dataset_split": "validation", "subject_id": "subject4",
        "fps": 30, "windows": tokenizer_rows, "examples": {},
        "summary": {name: float(np.mean([row[name] for row in tokenizer_rows]))
                    for name in tokenizer_rows[0] if name.endswith("_m") or name.endswith("_mps")
                    or name.endswith("_rad")},
        "quality_approved": False, "postprocessing": "none",
    }
    tokenizer_render = _render_set(output_dir / "tokenizer", tokenizer_arrays, tokenizer_metrics)

    infer, loaded_dataset, bundle = load_bundle(root_run, pose_run, "cuda")
    if loaded_dataset.resolve() != dataset_dir.resolve():
        raise ValueError("Root/Pose inference bundle points at a different native dataset")
    for stage in ("root", "pose"):
        if (bundle.get(stage + "_step") != expected_stage_steps
                or bundle.get(stage + "_checkpoint_sha256") != stage_hashes[stage]["checkpoint_sha256"]):
            raise ValueError(f"Loaded {stage} inference checkpoint differs from the verified stage budget")
    if ((bundle.get("tokenizer") or {}).get("checkpoint_sha256") != tokenizer_info["checkpoint_sha256"]
            or state_digest(infer._vqvae_pose_model) != tokenizer_state):
        raise ValueError("Loaded Pose inference tokenizer differs from the reviewed frozen tokenizer")
    inbetween_rows, inbetween_arrays = [], {}
    rep = infer.global_motion_rep
    for window in windows:
        raw = torch.from_numpy(_read_features(dataset_dir, window)).to("cuda")[None]
        source_features, generated_features = predict_window(infer, raw)
        source_motion, generated_motion = _joint_arrays(rep, source_features, generated_features)
        source_joints = source_motion["posed_joints"][0].detach().cpu().numpy()
        generated_joints = generated_motion["posed_joints"][0].detach().cpu().numpy()
        metrics = reconstruction_metrics(generated_motion, source_motion,
                                         rep.skeleton.root_idx, rep.fps)
        inbetween_rows.append({**window, "conditioning_mode": "native_keyframe1_inbetween", **metrics})
        if window["behavior"] not in inbetween_arrays:
            inbetween_arrays[window["behavior"]] = (source_joints, generated_joints, float(rep.fps))

    inbetween_metrics = {
        "task": "subject4 LAFAN generated native Root-to-Pose in-between; source interior used for evaluation only",
        "stage": "pose", "render_mode": "native_keyframe1_inbetween",
        "root_condition_source": "root_model", "dataset_split": "validation", "subject_id": "subject4",
        "native_inference_bundle": bundle,
        "fps": float(rep.fps), "windows": inbetween_rows, "examples": {},
        "summary": {name: float(np.mean([row[name] for row in inbetween_rows]))
                    for name in ("mpjpe_m", "root_relative_fk_mpjpe_m", "root_error_m",
                                 "joint_velocity_error_mps", "local_joint_rotation_geodesic_rad",
                                 "root_relative_joint_velocity_error_mps")},
        "quality_approved": False, "postprocessing": "none", "root_overwrite": False,
        "wasd_continuous_review": {"status": "not_assessed",
                                   "reason": "No LAFAN command/reference bank for continuous WASD evaluation."},
    }
    inbetween_render = _render_set(output_dir / "inbetween", inbetween_arrays, inbetween_metrics)

    candidate = {
        "schema": "native_mb_lafan_candidate_v1", "status": "trained_evaluated_not_approved",
        "motion_fps": 30, "skeleton": "SOMA30", "dataset": str(dataset_dir.resolve()),
        **budget, "validation_report": str(output_dir / "report.json"),
        "dataset_manifest_sha256": sha256(dataset_dir / "native_manifest.json"),
        "stages": stage_hashes,
        "frozen_tokenizer": {"checkpoint_sha256": tokenizer_info["checkpoint_sha256"],
                             "tensor_state_sha256": tokenizer_state,
                             "run": tokenizer_info["run"]},
        "quality_approved": False, "ue_deployment": False,
        "wasd_continuous_review": "not_assessed_no_command_reference_bank",
        "limitations": ["Subject4 walk/run validation only; subject5 test split remains untouched.",
                        "WASD continuous behavior has no LAFAN command/reference bank and was not assessed.",
                        "No quality approval, UE deployment, or runtime-performance claim."],
    }
    if expected_stage_steps == DEFAULT_STAGE_STEPS:
        candidate["paper_steps"] = DEFAULT_STAGE_STEPS
    else:
        candidate["limitations"].append(
            f"User-selected budget is {expected_stage_steps} updates per stage; no 2M completion claim."
        )
    report = {
        "status": "completed", "task": "bounded native LAFAN validation",
        **budget, "stages": stage_hashes,
        "source_accepted_sha256": sha256(source_dir / "accepted.json"),
        "dataset_manifest_sha256": candidate["dataset_manifest_sha256"],
        "subject_id": "subject4", "behaviors": ["walk", "run"],
        "window_count": len(windows), "windows": windows, "excluded_short_windows": excluded,
        "tokenizer": {"report": "tokenizer/metrics.json", "summary": tokenizer_metrics["summary"],
                      "gifs": tokenizer_render["gifs"]},
        "root_pose_inbetween": {"report": "inbetween/metrics.json",
                                "summary": inbetween_metrics["summary"], "gifs": inbetween_render["gifs"]},
        "wasd_continuous_review": inbetween_metrics["wasd_continuous_review"],
        "quality_approved": False, "ue_deployment": False,
    }
    _write_json(output_dir / "candidate_bundle.json", candidate)
    _write_json(output_dir / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer-run", type=Path, required=True)
    parser.add_argument("--root-run", type=Path, required=True)
    parser.add_argument("--pose-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-stage-steps", type=int, default=DEFAULT_STAGE_STEPS)
    args = parser.parse_args(argv)
    if args.expected_stage_steps <= 0:
        parser.error("--expected-stage-steps must be a positive integer")
    if args.output.exists():
        parser.error("Review output must be fresh")
    report = evaluate_lafan(args.source, args.dataset, args.tokenizer_run,
                            args.root_run, args.pose_run, args.output,
                            expected_stage_steps=args.expected_stage_steps)
    print(json.dumps({"status": report["status"], "window_count": report["window_count"],
                      "quality_approved": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
