"""Held-out native Root-only review while Pose trains; not an online control test."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.evaluate_parc_native_terrain import (
    CONTEXT_FRAMES, GLOBAL_WINDOW_FRAMES, LOCAL_WINDOW_FRAMES, NUM_TOKENS,
    SCORE_START, SCORE_STOP, _dataset_windows, _read_window_terrain, _source_window,
)
from scripts.train_native_mb import select_gpu, write_json
from scripts.train_native_mb_control import load_control_checkpoint


def root_inputs(model, raw):
    """Match native training normalization, then retain only keyframes 3 and 63."""
    rep, global_rep, local_rep = model.motion_rep, model.global_motion_rep, model.local_motion_rep
    if raw.shape != (1, GLOBAL_WINDOW_FRAMES, len(global_rep.indices["all"])):
        raise ValueError("Expected a complete native 65-frame global window")
    if not torch.isfinite(raw).all():
        raise ValueError("Source window must be finite")
    canonical = global_rep.change_first_heading(raw, 0.0, False, False)
    normalized = global_rep.normalize(canonical)
    local = rep.dual_rep.global_to_local(
        normalized, is_normalized=True, to_normalize=True,
        lengths=torch.tensor([GLOBAL_WINDOW_FRAMES], device=raw.device))
    inputs = {}
    for value_name, mask_name, full in (
        ("global_root_values", "has_global_root_values",
         extract_feature_from_motion_rep(normalized, global_rep, model.args["global_root_feature"])),
        ("local_root_values", "has_local_root_values",
         extract_feature_from_motion_rep(local, local_rep, model.args["local_root_feature"])),
        ("poses", "has_poses",
         extract_feature_from_motion_rep(local, local_rep, model.args["local_pose_feature"])),
    ):
        sparse = full.new_zeros((1, 8, full.shape[-1]))
        sparse[:, [3, 7]] = full[:, [3, 63]]
        mask = torch.zeros((1, 8), dtype=torch.bool, device=raw.device)
        mask[:, [3, 7]] = True
        inputs[value_name], inputs[mask_name] = sparse, mask
    minimum, maximum = model.args["min_tokens"], model.args["max_tokens"]
    if not minimum <= NUM_TOKENS <= maximum:
        raise ValueError("Root must support a 16-token horizon")
    allowed = torch.zeros((1, maximum - minimum + 1), dtype=torch.long, device=raw.device)
    allowed[:, NUM_TOKENS - minimum] = 1
    inputs["allowed_pred_num_tokens"] = allowed
    inputs["num_tokens"] = torch.full((1, 1), model.backbone_net.MASKED_NUM_TOKENS,
                                     dtype=torch.long, device=raw.device)
    return inputs, extract_feature_from_motion_rep(canonical[:, :LOCAL_WINDOW_FRAMES], global_rep, "root")


def root_metrics(prediction, source, fps):
    if prediction.shape != source.shape or source.shape != (1, LOCAL_WINDOW_FRAMES, 5):
        raise ValueError("Root scores require matching [1,64,5] arrays")
    if not torch.isfinite(prediction).all() or not torch.isfinite(source).all():
        raise ValueError("Root arrays must be finite")
    pred, target = prediction[0, SCORE_START:SCORE_STOP], source[0, SCORE_START:SCORE_STOP]
    delta = pred[:, :3] - target[:, :3]
    p_angle = torch.atan2(pred[:, 4], pred[:, 3])
    t_angle = torch.atan2(target[:, 4], target[:, 3])
    heading_delta = torch.atan2(torch.sin(p_angle - t_angle), torch.cos(p_angle - t_angle)).abs()
    velocity_delta = torch.diff(delta, dim=0) * fps
    return {
        "root_translation_error_m": float(delta.norm(dim=-1).mean()),
        "root_xz_error_m": float(delta[:, [0, 2]].norm(dim=-1).mean()),
        "root_height_error_m": float(delta[:, 1].abs().mean()),
        "root_heading_error_rad": float(heading_delta.mean()),
        "root_velocity_error_mps": float(velocity_delta.norm(dim=-1).mean()),
    }


def review(model, dataset, output, contract, result, max_windows=8):
    dataset, windows, _, _, identity, _, _, _ = _dataset_windows(dataset, max_windows)
    if identity["dataset_manifest_sha256"] != contract["dataset_manifest_sha256"]:
        raise ValueError("Root checkpoint and held-out dataset differ")
    if not model.backbone_net.args.get("terrain_conditioning", False):
        raise ValueError("Root review requires terrain conditioning")
    output = Path(output).resolve()
    if output.exists() or output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError("Use a new review directory separate from the dataset")
    model.eval()
    device = next(model.parameters()).device
    rows = []
    arrays = []
    with full_precision_evaluation(), torch.inference_mode():
        for window in windows:
            raw = torch.as_tensor(_source_window(dataset, window), dtype=torch.float32, device=device)[None]
            inputs, source = root_inputs(model, raw)
            terrain, terrain_info = _read_window_terrain(dataset, window, device)
            predictions = {}
            metrics = {}
            for branch, observation in (
                ("terrain", terrain),
                ("all_invalid", {"points": terrain["points"], "valid": torch.zeros_like(terrain["valid"])}),
            ):
                result_batch = model.backbone_net(**inputs, terrain_observation=observation,
                                                 config={"allow_pred_out_of_reach_num_tokens": False})
                if result_batch["pred_num_tokens"].item() != NUM_TOKENS:
                    raise ValueError("Root predicted a forbidden horizon")
                pred = model.global_motion_rep.unnormalize(
                    result_batch["pred_global_root_values"][:, :LOCAL_WINDOW_FRAMES])
                metrics[branch] = root_metrics(pred, source, model.motion_rep.fps)
                predictions[branch] = pred
            difference = (predictions["terrain"] - predictions["all_invalid"])[0, SCORE_START:SCORE_STOP]
            rows.append({**window, **metrics, "terrain_observation": terrain_info,
                         "terrain_vs_invalid_root_xyz_delta_m": float(difference[:, :3].norm(dim=-1).mean())})
            arrays.append({"source": source.cpu().numpy()[0],
                           **{name: tensor.cpu().numpy()[0] for name, tensor in predictions.items()}})
    summary = {branch: {key: float(np.mean([row[branch][key] for row in rows]))
                        for key in rows[0][branch]} for branch in ("terrain", "all_invalid")}
    summary["terrain_vs_invalid_root_xyz_delta_m"] = float(np.mean([
        row["terrain_vs_invalid_root_xyz_delta_m"] for row in rows]))
    report = {
        "task": "offline Root-only sparse source endpoint diagnostic; not WASD",
        "dataset_split": "validation", **identity,
        "checkpoint_sha256": result["final_checkpoint_sha256"],
        "optimizer_updates": result["optimizer_updates"],
        "conditions": {"sparse_frames": [3, 63], "global_root_xyz_and_heading": True,
                       "local_root_velocity": True, "local_pose": True,
                       "duration_input": "masked; only 16 tokens allowed", "source_interior_supplied": False,
                       "current_terrain_frame": CONTEXT_FRAMES - 1,
                       "future_height_endpoint_supplied": True},
        "scored_frames_inclusive": [SCORE_START, SCORE_STOP - 1],
        "scored_interval_starts_inclusive": [SCORE_START, SCORE_STOP - 2],
        "coordinate_frame": "per-window canonical SOMA Y-up",
        "postprocessing": "none", "summary": summary, "windows": rows,
        "interpretation": "Sensitivity is not terrain skill. Source endpoints include future height and pose; this does not establish an online direction-only contract.",
        "quality_approved": False, "wasd_control_proven": False,
    }
    output.mkdir(parents=True)
    write_json(output / "metrics.json", report)
    for ordinal, array in enumerate(arrays):
        np.savez_compressed(output / f"root_{ordinal:02d}.npz", **array, fps=model.motion_rep.fps)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.max_windows < 1 or args.output.exists():
        parser.error("Use a positive window count and new output directory")
    if args.gpu is not None:
        select_gpu(args.gpu)
    device = "cpu" if args.gpu is None else "cuda"
    torch.set_num_threads(4)
    model, contract, result = load_control_checkpoint(args.root_run, device)
    if contract["stage"] != "native_root":
        raise ValueError("Expected native Root checkpoint")
    report = review(model, Path(contract["dataset"]), args.output, contract, result, args.max_windows)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
