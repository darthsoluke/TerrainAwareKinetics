"""Evaluate native Root -> Pose -> decoder generation between two source keyframes.

Intermediate root trajectories are predicted by Root. Source interiors are used
only for comparison; conditional motion is non-unique, not reconstruction truth.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_control_review import (
    GLOBAL_WINDOW_FRAMES, LOCAL_WINDOW_FRAMES, NUM_TOKENS,
    _array_sha256, _fixed_validation_windows, _root_relative_error, _safe_name, _source_window,
)
from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from scripts.train_native_mb import select_gpu, sha256, write_json


MODE = "native_keyframe1_inbetween"
PREDICT_CONFIG = {
    "num_inference_step": 1,
    "pose_token_sampling_use_argmax": True,
    "allow_pred_out_of_reach_num_tokens": False,
    "smooth_root_traj": False,
    "skip_ending_target_cond": False,
    "final_root_pred_mode": "from_pose_module",
}


def inbetween_constraints(infer, raw_global):
    """Pack only frames 3/63 into the original eight-slot, unnormalized API."""
    import torch
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    rep, global_rep, local_rep = infer.motion_rep, infer.global_motion_rep, infer.local_motion_rep
    expected = (1, GLOBAL_WINDOW_FRAMES, len(global_rep.indices["all"]))
    if (raw_global.shape != expected or not raw_global.is_floating_point()
            or not torch.isfinite(raw_global).all()):
        raise ValueError(f"Expected finite floating global source of shape {expected}")
    minimum, maximum = infer._args["min_tokens"], infer._args["max_tokens"]
    if rep.fps != 30 or infer._args["down_t"] != 2 or not minimum <= NUM_TOKENS <= maximum:
        raise ValueError("Requires native 30 Hz, four frames per token, and a 16-token horizon")
    raw = global_rep.change_first_heading(raw_global, 0.0, False, False)
    local = rep.dual_rep.global_to_local(
        raw, is_normalized=False, to_normalize=False,
        lengths=torch.tensor([GLOBAL_WINDOW_FRAMES], device=raw.device),
    )[:, :LOCAL_WINDOW_FRAMES]
    if (raw.shape != expected or local.shape != (1, LOCAL_WINDOW_FRAMES, len(local_rep.indices["all"]))
            or not torch.isfinite(raw).all() or not torch.isfinite(local).all()):
        raise ValueError("Invalid canonical global/local source representation")
    inputs = {}
    for name, values in (
        ("global_root_values", extract_feature_from_motion_rep(raw, global_rep, "root")),
        ("local_root_values", extract_feature_from_motion_rep(local, local_rep, "root")),
        ("local_poses", extract_feature_from_motion_rep(local, local_rep, infer.EXTERNAL_POSE_FEATURE_MODE)),
    ):
        sparse = values.new_zeros((1, 8, values.shape[-1]))
        sparse[:, [3, 7]] = values[:, [3, 63]]
        mask = torch.zeros((1, 8), dtype=torch.bool, device=raw.device)
        mask[:, [3, 7]] = True
        inputs[name], inputs["has_" + name] = sparse, mask

    # predict() reads slot 0 even with its mask false: supply only the canonical
    # frame-0 coordinate reference, never its height, body or velocity.
    reference_indices = np.concatenate((global_rep.indices["global_root_pos_2d"],
                                        global_rep.indices["global_root_heading"]))
    inputs["global_root_values"][:, 0, reference_indices] = raw[:, 0, reference_indices]
    allowed = torch.zeros((1, maximum - minimum + 1), dtype=torch.long, device=raw.device)
    allowed[0, NUM_TOKENS - minimum] = 1
    inputs.update(num_tokens=None, allowed_pred_num_tokens=allowed)
    return inputs, raw[:, :LOCAL_WINDOW_FRAMES]


def predict_window(infer, raw_global):
    """Return untouched native predictions, with source retained only for scoring."""
    import torch

    with full_precision_evaluation(), torch.inference_mode():
        inputs, source = inbetween_constraints(infer, raw_global)
        prediction, count = infer.predict(**inputs, config=dict(PREDICT_CONFIG), info={})
        if (not isinstance(count, torch.Tensor) or count.numel() != 1
                or not torch.isfinite(count).all() or count.item() != NUM_TOKENS):
            raise ValueError("Native prediction must have exactly 16 tokens")
        if (not isinstance(prediction, torch.Tensor) or prediction.shape != source.shape
                or not prediction.is_floating_point() or not torch.isfinite(prediction).all()):
            raise ValueError("Invalid shape or nonfinite native prediction")
    return source, prediction


def evaluate_inbetween(infer, dataset_dir, output, bundle, seed=2333):
    import torch

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    dataset_dir, windows, excluded, identity = _fixed_validation_windows(dataset_dir)
    if bundle.get("dataset_manifest_sha256") != identity["dataset_manifest_sha256"]:
        raise ValueError("Bundle and validation dataset manifest differ")
    output = Path(output).resolve()
    if output == dataset_dir or dataset_dir in output.parents:
        raise ValueError("Evaluation output must be outside the held-out dataset")
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    behaviors = {row["behavior"] for row in windows}
    if len(behaviors) != 5 or len({_safe_name(name) for name in behaviors}) != 5:
        raise ValueError("Expected five distinct validation behavior filenames")
    torch.manual_seed(seed)
    np.random.seed(seed)
    infer.eval()
    rep, device = infer.global_motion_rep, torch.device(infer.device)
    root_index = rep.skeleton.root_idx
    rows, examples, arrays = [], {}, {}
    with full_precision_evaluation(), torch.inference_mode():
        for window in windows:
            source_raw = _source_window(dataset_dir, window)
            source, prediction = predict_window(
                infer, torch.as_tensor(source_raw, dtype=torch.float32, device=device)[None])
            joints = []
            for features in (source, prediction):
                posed = rep.inverse(features, False, return_all=True)["posed_joints"]
                if (posed.shape != (1, LOCAL_WINDOW_FRAMES, rep.skeleton.nbjoints, 3)
                        or not torch.isfinite(posed).all()):
                    raise ValueError("Invalid shape or nonfinite FK output")
                joints.append(posed[0].cpu().numpy().copy())
            source_joints, predicted_joints = joints
            target, generated = source_joints[4:63], predicted_joints[4:63]
            velocity_error = (np.diff(generated, axis=0) - np.diff(target, axis=0)) * rep.fps
            scores = {
                "world_fk_mpjpe_m": float(np.linalg.norm(generated - target, axis=-1).mean()),
                "root_relative_fk_mpjpe_m": _root_relative_error(generated, target, root_index),
                "joint_velocity_error_mps": float(np.linalg.norm(velocity_error, axis=-1).mean()),
                "root_fk_error_m": float(np.linalg.norm(
                    generated[:, root_index] - target[:, root_index], axis=-1).mean()),
            }
            if not all(np.isfinite(value) for value in scores.values()):
                raise ValueError("Nonfinite interior metrics")
            row = {**window, "source_window_sha256": _array_sha256(source_raw),
                   "conditioning_mode": MODE, "scored_frames_inclusive": [4, 62], **scores}
            rows.append(row)
            if window["behavior"] not in examples:
                examples[window["behavior"]] = row
                arrays[window["behavior"]] = (source_joints, predicted_joints)
    report = {
        "task": "native Root -> Pose -> decoder conditional generation; Root predicted intermediate root",
        "stage": "pose", "render_mode": MODE, "root_condition_source": "root_model",
        "root_condition": "Root predicted intermediate root from source global/local root and body keyframes 3 and 63",
        "dataset_split": "validation", **identity, "bundle": bundle, "seed": seed,
        "device": str(device), "fps": float(rep.fps),
        "global_window_frames": GLOBAL_WINDOW_FRAMES, "local_prediction_frames": LOCAL_WINDOW_FRAMES,
        "conditions": {"global_root_frames": [3, 63], "local_root_frames": [3, 63],
                       "pose_frames": [3, 63], "sparse_slots": [3, 7],
                       "local_root_velocity_source_intervals": [[3, 4], [63, 64]],
                       "source_body_velocity_features_used": False,
                       "source_interior_root_trajectory_supplied": False,
                       "source_encoded_tokens": False, "generated_pose_tokens": NUM_TOKENS,
                       "num_tokens": "MASKED (None)", "allowed_pred_num_tokens": [NUM_TOKENS]},
        "coordinate_reference": "Source frame 0 XZ/heading is used only for official canonicalization/recentering; slot 0 is masked and supplies no source height, body or velocity",
        "scored_frames_inclusive": [4, 62], "scored_velocity_interval_starts_inclusive": [4, 61],
        "sampling_config": dict(PREDICT_CONFIG), "postprocessing": "none", "root_overwrite": False,
        "evaluation_precision": "float32; CUDA matmul and cuDNN TF32 disabled",
        "interpretation": "Non-unique conditional generation, not reconstruction correctness or WASD control. Errors measure deviation from one held-out source interior, excluding the supplied endpoint.",
        "quality_approved": False, "summary": {name: float(np.mean([r[name] for r in rows])) for name in scores},
        "windows_per_mode": len(rows), "windows": rows, "examples": examples,
        "excluded_validation_clips": excluded, "evaluator_source_sha256": sha256(Path(__file__)),
    }
    output.mkdir(parents=True)
    for behavior, (source_joints, predicted_joints) in arrays.items():
        np.savez_compressed(output / f"{_safe_name(behavior)}_reconstruction.npz",
                            source=source_joints, prediction=predicted_joints, fps=np.float32(rep.fps))
    write_json(output / "metrics.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-run", type=Path, required=True)
    parser.add_argument("--pose-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int, default=2333)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new output directory")
    if args.seed < 0 or (args.gpu is not None and args.gpu < 0):
        parser.error("Seed and GPU index must be nonnegative")
    if (args.device == "cuda") != (args.gpu is not None):
        parser.error("CUDA requires --gpu; CPU does not accept --gpu")
    gpu = select_gpu(args.gpu) if args.device == "cuda" else None
    # Loading the bundle imports Torch and model modules only after the fresh GPU check.
    import torch
    from scripts.evaluate_native_mb_pipeline import load_bundle

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(4)
    infer, dataset, bundle = load_bundle(args.root_run, args.pose_run, args.device)
    report = evaluate_inbetween(infer, dataset, args.output, {**bundle, "gpu_at_launch": gpu}, args.seed)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
