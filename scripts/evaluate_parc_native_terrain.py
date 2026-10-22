"""Offline held-out PARC terrain-conditioned native in-between evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_control_review import (
    GLOBAL_WINDOW_FRAMES,
    LOCAL_WINDOW_FRAMES,
    NUM_TOKENS,
    _array_sha256,
    _fixed_validation_windows,
    _safe_name,
    _source_window,
)
from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from motionbricks.data.native_mb_terrain_protocol import validate_terrain_dataset
from scripts.evaluate_native_mb_inbetween import PREDICT_CONFIG, inbetween_constraints
from scripts.train_native_mb import sha256, write_json


SCORE_START = 4
SCORE_STOP = 63
CONTEXT_FRAMES = 4
WORLD_TRANSFORM_FIELDS = (
    "canonical_init_heading_quat",
    "canonical_root_pos_init_xz",
)


def _stable_rank(identity, *parts):
    payload = "\0".join([identity, *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_windows(windows, records, max_windows, dataset_identity):
    """Choose a repeatable, round-robin sample across terrain families and groups."""
    if isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows < 1:
        raise ValueError("max_windows must be a positive integer")
    by_shard = {row["shard"]: row for row in records if row.get("split") == "validation"}
    families = {}
    for window in windows:
        record = by_shard.get(window["shard"])
        if record is None:
            raise ValueError(f"Validation window has no accepted record: {window['shard']}")
        family = record.get("terrain_family") or record.get("behavior") or window["behavior"]
        group = record.get("source_group") or record.get("base_id") or window["shard"]
        groups = families.setdefault(str(family), {})
        candidates = groups.setdefault(str(group), [])
        candidates.append({
            **window,
            "terrain_family": str(family),
            "source_group": str(group),
        })

    ordered = {}
    group_orders = {}
    group_positions = {}
    for family, groups in families.items():
        ordered_groups = sorted(groups, key=lambda group: _stable_rank(dataset_identity, family, group))
        group_orders[family] = ordered_groups
        group_positions[family] = 0
        ordered[family] = {
            group: sorted(groups[group], key=lambda row: _stable_rank(
                dataset_identity, row["shard"], row["start"]))
            for group in ordered_groups
        }
    family_order = sorted(ordered, key=lambda family: _stable_rank(dataset_identity, family))
    chosen = []
    while len(chosen) < min(max_windows, len(windows)):
        advanced = False
        for family in family_order:
            names = group_orders[family]
            for offset in range(len(names)):
                index = (group_positions[family] + offset) % len(names)
                group = names[index]
                if ordered[family][group]:
                    chosen.append(ordered[family][group].pop(0))
                    group_positions[family] = (index + 1) % len(names)
                    advanced = True
                    break
            if len(chosen) == max_windows:
                break
        if not advanced:
            break
    return chosen


def _dataset_windows(dataset_dir, max_windows):
    dataset_dir, windows, excluded, identity = _fixed_validation_windows(dataset_dir)
    manifest_path = dataset_dir / "native_manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    with (dataset_dir / "accepted.json").open(encoding="utf-8") as stream:
        records = json.load(stream)
    protocol = validate_terrain_dataset(manifest, records)
    selected = select_windows(windows, records, max_windows, identity["dataset_manifest_sha256"])
    return dataset_dir, selected, windows, excluded, identity, manifest, protocol, records


def _read_window_terrain(dataset_dir, window, device):
    import torch

    shard_path = dataset_dir / window["shard"]
    with np.load(shard_path, allow_pickle=False) as shard:
        if not {"terrain_points", "terrain_valid"}.issubset(shard.files):
            raise ValueError(f"Validation shard lacks paired terrain arrays: {window['shard']}")
        points = np.asarray(shard["terrain_points"])
        valid = np.asarray(shard["terrain_valid"])
        frame_index = window["start"] + CONTEXT_FRAMES - 1
        if (points.ndim != 5 or points.shape[1:] != (2, 9, 9, 3)
                or valid.shape != points.shape[:-1] or valid.dtype != np.bool_
                or frame_index >= len(points)):
            raise ValueError(f"Invalid terrain observation arrays: {window['shard']}")
        current_points = points[frame_index].copy()
        current_valid = valid[frame_index].copy()
    if not np.isfinite(current_points).all():
        raise ValueError(f"Nonfinite current terrain observation: {window['shard']}")
    observation = {
        "points": torch.as_tensor(current_points, dtype=torch.float32, device=device)[None],
        "valid": torch.as_tensor(current_valid, dtype=torch.bool, device=device)[None],
    }
    return observation, {
        "frame_index": int(frame_index),
        "valid_points": int(current_valid.sum()),
        "total_points": int(current_valid.size),
        "valid_fraction": float(current_valid.mean()),
    }


def _world_transform(dataset_dir, window, device):
    import torch

    with np.load(dataset_dir / window["shard"], allow_pickle=False) as shard:
        present = [name in shard.files for name in WORLD_TRANSFORM_FIELDS]
        if not any(present):
            return None, "unavailable_missing_canonical_transform_fields"
        if not all(present):
            raise ValueError(f"Incomplete canonical world transform in {window['shard']}")
        heading = np.asarray(shard[WORLD_TRANSFORM_FIELDS[0]])
        root_xz = np.asarray(shard[WORLD_TRANSFORM_FIELDS[1]])
    if (heading.shape not in {(4,), (1, 4)} or root_xz.shape not in {(2,), (1, 2)}
            or not np.isfinite(heading).all() or not np.isfinite(root_xz).all()):
        raise ValueError(
            f"Malformed canonical world transform in {window['shard']}; expected (4,)/(2,) "
            "or singleton-batched (1,4)/(1,2) arrays"
        )
    return {
        "init_heading_quat": torch.as_tensor(heading.copy(), dtype=torch.float32, device=device),
        "root_pos_init_xz": torch.as_tensor(root_xz.copy(), dtype=torch.float32, device=device),
    }, "available"


def _window_world_transform(rep, raw_global, canonical_source, dataset_transform):
    """Compose dataset and per-window canonicalization using the represented FK frames."""
    import torch

    def inverse(features, transform):
        info = None if transform is None else {key: value.clone() for key, value in transform.items()}
        return rep.inverse(features, False, init_heading_info=info, return_all=True)

    source_world = inverse(raw_global[:, :LOCAL_WINDOW_FRAMES], dataset_transform)
    source_canonical = inverse(canonical_source, None)
    world_rots = source_world.get("all_global_rots", {}).get("global_rot_data")
    canonical_rots = source_canonical.get("all_global_rots", {}).get("global_rot_data")
    if (world_rots is None or canonical_rots is None
            or world_rots.shape[:2] != (1, LOCAL_WINDOW_FRAMES)
            or canonical_rots.shape != world_rots.shape
            or not torch.isfinite(world_rots).all() or not torch.isfinite(canonical_rots).all()):
        raise ValueError("Canonical transforms are present but FK global rotations are unavailable")
    root_index = int(rep.skeleton.root_idx)
    rotation = world_rots[0, 0, root_index] @ canonical_rots[0, 0, root_index].transpose(-1, -2)
    world_root = source_world["posed_joints"][0, 0, root_index]
    canonical_root = source_canonical["posed_joints"][0, 0, root_index]
    translation = world_root - rotation @ canonical_root
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    if (not torch.isfinite(rotation).all() or not torch.isfinite(translation).all()
            or not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-3, rtol=0)
            or torch.det(rotation) <= 0):
        raise ValueError("Derived per-window world transform is not a finite rigid rotation")
    return {"rotation": rotation.detach(), "translation": translation.detach()}


def _predict(infer, inputs, terrain, seed):
    import torch

    device = torch.device(infer.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    call_inputs = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    call_terrain = {key: value.clone() for key, value in terrain.items()}
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with full_precision_evaluation(), torch.inference_mode():
            prediction, count = infer.predict(
                **call_inputs,
                config=dict(PREDICT_CONFIG),
                info={},
                terrain_observation=call_terrain,
            )
    if (not isinstance(count, torch.Tensor) or count.numel() != 1
            or not torch.isfinite(count).all() or int(count.item()) != NUM_TOKENS):
        raise ValueError("Native terrain prediction must contain exactly 16 tokens")
    if (not isinstance(prediction, torch.Tensor)
            or prediction.shape != (1, LOCAL_WINDOW_FRAMES, len(infer.global_motion_rep.indices["all"]))
            or not prediction.is_floating_point() or not torch.isfinite(prediction).all()):
        raise ValueError("Native terrain prediction has an invalid shape or nonfinite values")
    return prediction


def _codec_reconstruction(infer, raw_global):
    """Reconstruct a fully source-encoded local sequence through the frozen codec."""
    import torch
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    rep, global_rep, local_rep = infer.motion_rep, infer.global_motion_rep, infer.local_motion_rep
    codec = infer._vqvae_pose_model
    canonical = global_rep.change_first_heading(raw_global, 0.0, False, False)
    normalized = global_rep.normalize(canonical)
    local = rep.dual_rep.global_to_local(
        normalized, is_normalized=True, to_normalize=True,
        lengths=torch.tensor([GLOBAL_WINDOW_FRAMES], device=raw_global.device),
    )[:, :LOCAL_WINDOW_FRAMES]
    external = extract_feature_from_motion_rep(
        local, local_rep, codec.decoder_external_cond_feature_mode)
    tokens = codec.encode_into_idx(local, fetch_overall_indices=False)
    decoded = codec.forward_decoder(
        tokens, target_cond=None, has_target_cond=None, external_cond=external,
        use_overall_indices=False,
    )["recon_state"]
    if (decoded.shape != local.shape or not torch.isfinite(decoded).all()
            or not torch.is_floating_point(decoded)):
        raise ValueError("Frozen codec returned invalid reconstruction features")
    reconstructed = rep.dual_rep.local_to_global(
        decoded, is_normalized=True, to_normalize=False,
        lengths=torch.tensor([LOCAL_WINDOW_FRAMES], device=raw_global.device),
    )
    if not torch.isfinite(reconstructed).all():
        raise ValueError("Codec reconstruction produced nonfinite global features")
    return reconstructed


def _rotation_geodesic(first, second):
    import torch
    from motionbricks.motionlib.core.utils.rotations import matrix_to_quaternion

    # Quaternion chord distance avoids acos amplification for identical FP32 rotations.
    a = matrix_to_quaternion(first.to(torch.float64))
    b = matrix_to_quaternion(second.to(torch.float64))
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    chord = torch.minimum((a - b).norm(dim=-1), (a + b).norm(dim=-1))
    return 4 * torch.asin((chord / 2).clamp(0, 1))


def _metric_arrays(rep, global_rep, source, prediction, world_transform):
    import torch

    def inverse(features):
        result = rep.inverse(features, False, return_all=True)
        joints = result["posed_joints"]
        rotations = result.get("all_local_rots", {}).get("global_rot_data")
        if (joints.shape[:2] != (1, LOCAL_WINDOW_FRAMES) or joints.shape[-1] != 3
                or not torch.isfinite(joints).all()):
            raise ValueError("Invalid or nonfinite native FK output")
        if (rotations is None or rotations.shape[:2] != (1, LOCAL_WINDOW_FRAMES)
                or rotations.shape[-2:] != (3, 3) or not torch.isfinite(rotations).all()):
            raise ValueError("Invalid or nonfinite native local rotation output")
        joints, rotations = joints[0], rotations[0]
        if world_transform is not None:
            rotation = world_transform["rotation"]
            translation = world_transform["translation"]
            joints = torch.einsum("ab,tjb->tja", rotation, joints) + translation
        return joints, rotations

    source_joints, source_rotations = inverse(source)
    pred_joints, pred_rotations = inverse(prediction)
    score = slice(SCORE_START, SCORE_STOP)
    root_index = int(rep.skeleton.root_idx)
    root_source, root_pred = source_joints[:, root_index], pred_joints[:, root_index]
    source_relative = source_joints - root_source[:, None]
    pred_relative = pred_joints - root_pred[:, None]
    rotation_error = _rotation_geodesic(pred_rotations, source_rotations)
    normalized_source = global_rep.normalize(source)
    normalized_prediction = global_rep.normalize(prediction)
    feature_rmse = (normalized_prediction[:, score] - normalized_source[:, score]).square().mean().sqrt()
    position_error = (root_pred[score] - root_source[score]).norm(dim=-1).mean()
    fk_error = (pred_joints[score] - source_joints[score]).norm(dim=-1).mean()
    relative_error = (pred_relative[score] - source_relative[score]).norm(dim=-1).mean()
    velocity_error = (
        (pred_joints[score][1:] - pred_joints[score][:-1])
        - (source_joints[score][1:] - source_joints[score][:-1])
    ).norm(dim=-1).mean() * float(rep.fps)
    scores = {
        "global_feature_normalized_rmse": float(feature_rmse),
        "root_translation_error_m": float(position_error),
        "fk_mpjpe_m": float(fk_error),
        "root_relative_fk_mpjpe_m": float(relative_error),
        "joint_velocity_error_mps": float(velocity_error),
        "joint_rotation_geodesic_error_rad": float(rotation_error[score].mean()),
    }
    if not np.isfinite(list(scores.values())).all():
        raise ValueError("Nonfinite native terrain metrics")
    return scores, {
        "joints": pred_joints.detach().cpu().numpy(),
        "root": root_pred.detach().cpu().numpy(),
        "rotations": pred_rotations.detach().cpu().numpy(),
    }


def _branch_difference(global_rep, terrain_features, invalid_features,
                       terrain_arrays, invalid_arrays):
    import torch

    score = slice(SCORE_START, SCORE_STOP)
    feature_delta = (
        global_rep.normalize(terrain_features)[:, score]
        - global_rep.normalize(invalid_features)[:, score]
    ).square().mean().sqrt()
    root_delta = torch.as_tensor(terrain_arrays["root"]) - torch.as_tensor(invalid_arrays["root"])
    joint_delta = torch.as_tensor(terrain_arrays["joints"]) - torch.as_tensor(invalid_arrays["joints"])
    terrain_rot = torch.as_tensor(terrain_arrays["rotations"])
    invalid_rot = torch.as_tensor(invalid_arrays["rotations"])
    rotation_delta = _rotation_geodesic(terrain_rot, invalid_rot)
    result = {
        "global_feature_normalized_rmse": float(feature_delta),
        "root_trajectory_delta_m": float(root_delta[score].norm(dim=-1).mean()),
        "fk_joint_displacement_m": float(joint_delta[score].norm(dim=-1).mean()),
        "joint_rotation_geodesic_delta_rad": float(rotation_delta[score].mean()),
    }
    epsilon = torch.finfo(terrain_features.dtype).eps
    position_scale = max(1.0, float(np.abs(terrain_arrays["joints"][score]).max()),
                         float(np.abs(invalid_arrays["joints"][score]).max()))
    feature_scale = max(1.0, float(global_rep.normalize(terrain_features)[:, score].abs().max()))
    floors = {"global_feature_normalized_rmse": 64 * epsilon * feature_scale,
              "root_trajectory_delta_m": 64 * epsilon * position_scale,
              "fk_joint_displacement_m": 64 * epsilon * position_scale,
              "joint_rotation_geodesic_delta_rad": 64 * epsilon}
    result["nonzero"] = any(result[key] > floor for key, floor in floors.items())
    result["numerical_floors"] = floors
    result["nonzero_definition"] = "above a conservative 64 machine-epsilon scale floor; not skill evidence"
    return result


def _render_skeleton_examples(output, render_examples, fps):
    from scripts.render_native_mb_review import _render_motion_set

    render_dir = output / "skeleton_renderings"
    render_dir.mkdir()
    name = sorted(render_examples)[0]
    data = render_examples[name]
    rendered = []
    for key, label in (("terrain", "Terrain-conditioned"),
                       ("all_invalid", "All-invalid terrain"),
                       ("codec", "Source-encoded codec baseline")):
        source = data["source"]["joints"].astype(np.float32)
        prediction = data[key]["joints"].astype(np.float32)
        path = render_dir / f"{_safe_name(name)}_{key}_reconstruction.npz"
        np.savez_compressed(path, source=source, prediction=prediction, fps=np.float32(fps))
        _render_motion_set(
            path, (source, prediction), fps, render_dir,
            f"{name} | offline in-between | {label} | no postprocessing",
            panel_labels=("Source motion", label),
        )
        rendered.extend((path, path.with_suffix(".png"), path.with_suffix(".gif")))
    return [str(path.relative_to(output)) for path in rendered]


def evaluate(infer, dataset_dir, output, bundle, *, seed=2333, max_windows=32,
             workflow_provenance=None, render=False):
    import torch

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(max_windows, int) or isinstance(max_windows, bool) or max_windows < 1:
        raise ValueError("max_windows must be a positive integer")
    if (not infer._root_model.backbone_net._args.get("terrain_conditioning", False)
            or not infer._pose_model.backbone_net._args.get("terrain_conditioning", False)):
        raise ValueError("PARC native terrain evaluation requires terrain-enabled Root and Pose models")

    (dataset_dir, windows, candidates, excluded, identity, manifest, protocol,
     records) = _dataset_windows(dataset_dir, max_windows)
    if bundle.get("dataset_manifest_sha256") != identity["dataset_manifest_sha256"]:
        raise ValueError("Workflow bundle and held-out dataset manifest differ")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    if output == dataset_dir or dataset_dir in output.parents or output in dataset_dir.parents:
        raise ValueError("Evaluation output must be separate from the held-out dataset")

    infer.eval()
    device = torch.device(infer.device)
    rep, global_rep = infer.motion_rep, infer.global_motion_rep
    torch.manual_seed(seed)
    np.random.seed(seed)
    rows, render_examples = [], {}
    with full_precision_evaluation(), torch.inference_mode():
        for ordinal, window in enumerate(windows):
            raw_np = _source_window(dataset_dir, window)
            raw = torch.as_tensor(raw_np, dtype=torch.float32, device=device)[None]
            inputs, source = inbetween_constraints(infer, raw)
            terrain, terrain_info = _read_window_terrain(dataset_dir, window, device)
            invalid_terrain = {
                "points": terrain["points"].clone(),
                "valid": torch.zeros_like(terrain["valid"]),
            }
            branch_seed = seed + ordinal
            terrain_prediction = _predict(infer, inputs, terrain, branch_seed)
            invalid_prediction = _predict(infer, inputs, invalid_terrain, branch_seed)
            codec_prediction = _codec_reconstruction(infer, raw)
            dataset_transform, transform_status = _world_transform(dataset_dir, window, device)
            world_transform = None
            if dataset_transform is not None:
                world_transform = _window_world_transform(rep, raw, source, dataset_transform)
            terrain_metrics, terrain_arrays = _metric_arrays(
                rep, global_rep, source, terrain_prediction, world_transform)
            invalid_metrics, invalid_arrays = _metric_arrays(
                rep, global_rep, source, invalid_prediction, world_transform)
            codec_metrics, codec_arrays = _metric_arrays(
                rep, global_rep, source, codec_prediction, world_transform)
            source_metrics, source_arrays = _metric_arrays(
                rep, global_rep, source, source, world_transform)
            difference = _branch_difference(
                global_rep, terrain_prediction, invalid_prediction,
                terrain_arrays, invalid_arrays)
            row = {
                **window,
                "source_window_sha256": _array_sha256(raw_np),
                "scored_frames_inclusive": [SCORE_START, SCORE_STOP - 1],
                "terrain_observation": terrain_info,
                "canonical_world_transform": transform_status,
                "metric_coordinate_frame": "world" if world_transform is not None else "canonicalized_feature_frame",
                "terrain": terrain_metrics,
                "all_invalid_terrain": invalid_metrics,
                "codec_reconstruction": codec_metrics,
                "terrain_vs_all_invalid_difference": difference,
            }
            rows.append(row)
            if window["behavior"] not in render_examples:
                render_examples[window["behavior"]] = {
                    "source": source_arrays,
                    "terrain": terrain_arrays,
                    "all_invalid": invalid_arrays,
                    "codec": codec_arrays,
                }

    metric_names = tuple(rows[0]["terrain"])
    summary = {
        branch: {name: float(np.mean([row[branch][name] for row in rows])) for name in metric_names}
        for branch in ("terrain", "all_invalid_terrain", "codec_reconstruction")
    }
    difference_names = tuple(name for name, value in rows[0]["terrain_vs_all_invalid_difference"].items()
                             if isinstance(value, float))
    summary["terrain_vs_all_invalid_difference"] = {
        name: float(np.mean([row["terrain_vs_all_invalid_difference"][name] for row in rows]))
        for name in difference_names
    }
    summary["terrain_vs_all_invalid_difference"]["nonzero_windows"] = sum(
        row["terrain_vs_all_invalid_difference"]["nonzero"] for row in rows)
    validity = [row["terrain_observation"]["valid_points"] for row in rows]
    report = {
        "task": "offline native Root -> Pose -> decoder in-between terrain comparison",
        "dataset_split": "validation",
        "seed": seed,
        "max_windows": max_windows,
        "window_counts": {
            "eligible_validation_windows": len(candidates),
            "evaluated_windows": len(rows),
            "windows_with_any_valid_terrain": sum(value > 0 for value in validity),
            "windows_with_all_invalid_source_observation": sum(value == 0 for value in validity),
            "excluded_short_validation_shards": len(excluded),
        },
        "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "accepted_json_sha256": identity["accepted_json_sha256"],
        "workflow": workflow_provenance,
        "bundle": bundle,
        "terrain_protocol": protocol,
        "conditions": {
            "global_root_frames": [3, 63],
            "local_root_frames": [3, 63],
            "pose_frames": [3, 63],
            "sparse_slots": [3, 7],
            "source_interior_root_trajectory_supplied": False,
            "source_interior_pose_features_supplied": False,
            "source_pose_tokens_supplied_to_generation": False,
            "frame0_reference": "masked coordinate reference only; no source height/body/velocity condition",
            "global_window_frames": GLOBAL_WINDOW_FRAMES,
            "prediction_frames": LOCAL_WINDOW_FRAMES,
            "predicted_tokens": NUM_TOKENS,
            "terrain_observation_time": "last context frame, source window start + 3",
            "future_terrain_queried": False,
            "terrain_vs_all_invalid_motion_conditions_identical": True,
            "terrain_vs_all_invalid_seed_identical_per_window": True,
            "codec_baseline": "source-encoded full 64-frame sequence through frozen tokenizer/decoder; reconstruction baseline, not generated in-between motion",
        },
        "canonical_world_transform_fields": list(WORLD_TRANSFORM_FIELDS),
        "metric_coordinate_frame": "world where the per-window transform can be composed from both canonical fields; otherwise canonicalized feature frame only",
        "metric_units": {
            "global_feature_normalized_rmse": "dimensionless normalized feature units",
            "root_translation_error_m": "m",
            "fk_mpjpe_m": "m",
            "root_relative_fk_mpjpe_m": "m",
            "joint_velocity_error_mps": "m/s",
            "joint_rotation_geodesic_error_rad": "rad",
            "terrain_vs_all_invalid_difference.root_trajectory_delta_m": "m",
            "terrain_vs_all_invalid_difference.fk_joint_displacement_m": "m",
            "terrain_vs_all_invalid_difference.joint_rotation_geodesic_delta_rad": "rad",
        },
        "scored_frames_inclusive": [SCORE_START, SCORE_STOP - 1],
        "fps": float(rep.fps),
        "sampling_config": dict(PREDICT_CONFIG),
        "evaluation_precision": "float32; CUDA matmul and cuDNN TF32 disabled",
        "postprocessing": "none",
        "ik_or_foot_lock_or_smoothing": "none",
        "summary": summary,
        "windows": rows,
        "excluded_validation_shards": excluded,
        "interpretation": (
            "Offline held-out in-betweening only. Conditional interiors are non-unique; errors against one source "
            "motion are not a quality gate. A branch difference records sensitivity to this observation, not learned "
            "terrain skill. This is not WASD proof and makes no unseen-terrain generalization claim."
        ),
        "world_metrics_status": "available_per_window" if all(
            row["canonical_world_transform"] == "available" for row in rows
        ) else "canonical_only_for_rows_missing_transform",
        "terrain_branch_difference_nonzero_windows": summary["terrain_vs_all_invalid_difference"]["nonzero_windows"],
        "quality_approved": False,
        "wasd_control_proven": False,
        "unseen_terrain_claim": False,
        "render": {"requested": bool(render), "status": "not_requested" if not render else "pending"},
        "evaluator_source_sha256": sha256(Path(__file__)),
    }

    output.mkdir(parents=True)
    write_json(output / "metrics.json", report)
    if render:
        try:
            report["render"] = {
                "requested": True,
                "status": "completed",
                "artifacts": _render_skeleton_examples(output, render_examples, float(rep.fps)),
                "description": "one selected SOMA skeleton example for each branch; no physical assessment",
            }
        except Exception as error:  # Metrics are already durable; rendering is optional.
            report["render"] = {
                "requested": True,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "metrics_persisted": True,
            }
        write_json(output / "metrics.json", report)
    return report


def load_workflow(workflow_path):
    workflow_path = Path(workflow_path).resolve()
    with workflow_path.open(encoding="utf-8") as stream:
        state = json.load(stream)
    if state.get("status") not in {"training_finished_behavior_review_pending", "completed"}:
        raise ValueError("Workflow state must record finished training")
    stages = state.get("stages")
    if not isinstance(stages, dict) or any(
        not isinstance(stages.get(stage), dict) or stages[stage].get("status") != "completed"
        for stage in ("tokenizer", "root", "pose")
    ):
        raise ValueError("Workflow must contain completed tokenizer, Root, and Pose stages")
    dataset_value = state.get("dataset")
    manifest_hash = state.get("dataset_manifest_sha256")
    if not isinstance(dataset_value, str) or not isinstance(manifest_hash, str):
        raise ValueError("Workflow state lacks dataset provenance")

    def stage_run(stage):
        value = stages[stage].get("run")
        if not isinstance(value, str) or not value:
            raise ValueError(f"Workflow state lacks the {stage} run directory")
        path = Path(value)
        return (path if path.is_absolute() else workflow_path.parent / path).resolve()

    dataset_path = Path(dataset_value)
    if not dataset_path.is_absolute():
        dataset_path = workflow_path.parent / dataset_path
    provenance = {
        "state_path": str(workflow_path),
        "state_sha256": sha256(workflow_path),
        "status": state["status"],
        "dataset": str(dataset_path.resolve()),
        "dataset_manifest_sha256": manifest_hash,
        "stages": {
            stage: {"run": str(stage_run(stage)), "status": stages[stage]["status"]}
            for stage in ("tokenizer", "root", "pose")
        },
    }
    return dataset_path.resolve(), stage_run("root"), stage_run("pose"), provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True,
                        help="completed PARC-only training state.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="new review directory, separate from the immutable dataset")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--max-windows", type=int, default=32)
    parser.add_argument("--render", action="store_true",
                        help="write one optional SOMA skeleton example per branch after metrics are saved")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.max_windows < 1 or (args.gpu is not None and args.gpu < 0):
        parser.error("Seed, max windows, and GPU index must be nonnegative; max windows must be positive")
    if (args.device == "cuda") != (args.gpu is not None):
        parser.error("CUDA requires --gpu; CPU does not accept --gpu")
    if args.output.exists():
        parser.error("Use a new output directory")

    dataset, root_run, pose_run, workflow_provenance = load_workflow(args.workflow)
    gpu = None
    if args.device == "cuda":
        from scripts.train_native_mb import select_gpu
        gpu = select_gpu(args.gpu)
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(4)
    from scripts.evaluate_native_mb_pipeline import load_bundle

    infer, bundle_dataset, bundle = load_bundle(root_run, pose_run, args.device)
    if bundle_dataset.resolve() != dataset:
        raise ValueError("Workflow dataset path differs from the checkpoint-bound dataset")
    if bundle.get("dataset_manifest_sha256") != workflow_provenance["dataset_manifest_sha256"]:
        raise ValueError("Workflow and checkpoint dataset manifest hashes differ")
    bundle = {**bundle, "device": args.device, "gpu_at_launch": gpu}
    report = evaluate(
        infer, dataset, args.output, bundle,
        seed=args.seed, max_windows=args.max_windows,
        workflow_provenance=workflow_provenance, render=args.render,
    )
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
