"""Bounded, frozen-weight comparison of legacy and native smart locomotion.

Five commands, six plans each, and eight played frames per plan. Only four
validation idle frames seed playback; reference keyframes come from train clips.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import string
import sys
import time
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from scripts.train_native_mb import select_gpu, sha256, write_json


REFERENCE_STYLES = ("idle", "walk", "run", "backward")
COMMAND_STYLES = {"idle": "idle", "forward": "walk", "turn_right": "walk",
                  "stop": "idle", "backward": "backward"}
CASES = ("legacy_start_only", "native_buffer4", "native_buffer1", "native_buffer4_target1",
         "native_smart4", "native_smart1")
BUFFER_CASES = ("native_buffer4", "native_buffer1", "native_buffer4_target1")
CASE_CONDITIONS = {
    "legacy_start_only": {"context_keyframes": 4, "target_keyframes": 0,
                          "description": "legacy start-only stress probe; fixed eight tokens"},
    "native_buffer4": {"context_keyframes": 4, "target_keyframes": 4,
                       "description": "exact public G1 preview playback cursor port; four keyframes at both ends; FILTER_QPOS=False"},
    "native_buffer1": {"context_keyframes": 1, "target_keyframes": 1,
                       "description": "public G1 preview playback; one keyframe at both ends, training-curriculum diagnostic"},
    "native_buffer4_target1": {"context_keyframes": 4, "target_keyframes": 1,
                               "description": "public G1 preview playback; four context keyframes and one target keyframe"},
    "native_smart4": {"context_keyframes": 4, "target_keyframes": 4,
                      "description": "last-played-four context candidate; four keyframes at both ends"},
    "native_smart1": {"context_keyframes": 1, "target_keyframes": 1,
                      "description": "last-played-four context candidate; one keyframe at both ends, training-curriculum diagnostic"},
    "native_smart4_fixed8": {"context_keyframes": 4, "target_keyframes": 4,
                             "description": "last-played-four context candidate; four keyframes at both ends; fixed eight tokens"},
}
LEGACY_CONFIG = {
    "num_inference_step": 1, "pose_token_sampling_use_argmax": True,
    "allow_pred_out_of_reach_num_tokens": False, "smooth_root_traj": False,
    "skip_ending_target_cond": True, "final_root_pred_mode": "from_pose_module",
}


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _array_hash(value):
    value = np.ascontiguousarray(_array(value))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _jsonable(value):
    if hasattr(value, "detach") or isinstance(value, np.ndarray):
        return _jsonable(_array(value).tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _finite_counts(value):
    value = _array(value)
    return {"elements": int(value.size), "finite_elements": int(np.isfinite(value).sum()),
            "nonfinite_elements": int((~np.isfinite(value)).sum())}


def validate_frozen_bundle(bundle):
    tokenizer = bundle.get("tokenizer") or {}
    for name, step in (("root_step", bundle.get("root_step")), ("pose_step", bundle.get("pose_step")),
                       ("tokenizer.global_step", tokenizer.get("global_step"))):
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError(f"Frozen bundle {name} must be a positive integer")
    for digest in (bundle.get("root_checkpoint_sha256"), bundle.get("pose_checkpoint_sha256"),
                   tokenizer.get("checkpoint_sha256"), bundle.get("dataset_manifest_sha256"),
                   tokenizer.get("dataset_manifest_sha256")):
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in string.hexdigits for char in digest):
            raise ValueError("Bundle lacks a complete 64-hex checkpoint/dataset hash binding")
    if tokenizer.get("dataset_manifest_sha256") != bundle["dataset_manifest_sha256"]:
        raise ValueError("Tokenizer and control dataset binding differ")


def load_sources(dataset_dir, bundle, feature_dim):
    """Read a bound seed4 and complete train-only reference clips, never tokens."""
    validate_frozen_bundle(bundle)
    dataset_dir = Path(dataset_dir).resolve()
    manifest_path, accepted_path = dataset_dir / "native_manifest.json", dataset_dir / "accepted.json"
    if sha256(manifest_path) != bundle["dataset_manifest_sha256"]:
        raise ValueError("Bundle/dataset manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifact_sha256", {})
    accepted_hash = sha256(accepted_path)
    if artifacts.get("accepted.json") != accepted_hash:
        raise ValueError("Manifest/accepted.json hash mismatch")
    records = json.loads(accepted_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Expected nonempty accepted split records")
    splits = {}
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get("shard"), str)
                or record.get("split") not in ("train", "validation", "test")):
            raise ValueError("Malformed accepted split record")
        splits.setdefault(record["shard"], set()).add(record["split"])
    if any(len(value) != 1 for value in splits.values()):
        raise ValueError("A shard appears in multiple splits")

    def read(record, seed_only=False):
        relative = record["shard"]
        path = (dataset_dir / relative).resolve()
        if dataset_dir not in path.parents:
            raise ValueError("Selected shard escapes the dataset")
        digest = sha256(path)
        if (artifacts.get(relative) != digest
                or record.get("converted_shard_sha256", digest) != digest):
            raise ValueError(f"Selected shard hash mismatch: {relative}")
        with np.load(path, allow_pickle=False) as data:
            features = data["global_features"]
            frames = len(features)
            raw = features[:4].copy() if seed_only else features.copy()
            if "fps" in data and float(data["fps"]) != 30.0:
                raise ValueError("References and seed must be native 30 Hz")
        declared = record.get("frames30", record.get("frames_30hz"))
        if declared is not None and declared != frames:
            raise ValueError("Selected shard frame count mismatch")
        if (raw.ndim != 2 or raw.shape[1] != feature_dim or not np.issubdtype(raw.dtype, np.floating)
                or not np.isfinite(raw).all() or (len(raw) != 4 if seed_only else len(raw) <= 4)):
            raise ValueError(f"Invalid selected source features: {relative}")
        metadata = {"shard": relative, "shard_sha256": digest, "split": record["split"],
                    "behavior": record["behavior"], "filename": record.get("filename"),
                    "source_clip": record.get("source_clip"), "shard_frames": frames,
                    "loaded_source_frames_inclusive": [0, len(raw) - 1],
                    "source_feature_sha256": _array_hash(raw), "source_encoded_tokens": False}
        return raw, metadata

    def first_available(candidates):
        return next((row for row in candidates if (dataset_dir / row["shard"]).is_file()), None)

    seed_record = first_available(row for row in records
                                  if row["split"] == "validation" and row.get("behavior") == "idle")
    if seed_record is None:
        raise ValueError("No available validation idle seed")
    seed, seed_metadata = read(seed_record, seed_only=True)
    clips, references = {}, {}
    heldout_hashes = {artifacts.get(row["shard"]) for row in records if row["split"] != "train"}
    for style in REFERENCE_STYLES:
        candidates = [row for row in records if row["split"] == "train" and row.get("behavior") == style]

        def priority(row):
            name = str(row.get("filename") or row.get("source_clip") or "").lower()
            if style == "idle" and ("neutral" in name or "static" in name):
                return 0
            return 1 if "loop" in name else 2

        # Stable sorting preserves accepted.json order within each preference.
        record = first_available(sorted(candidates, key=priority))
        if record is None:
            raise ValueError(f"No available train reference for {style}")
        raw, metadata = read(record)
        if metadata["shard_sha256"] in heldout_hashes:
            raise ValueError("Train reference duplicates held-out shard bytes")
        for field in ("source_clip", "filename", "source_shard"):
            if record.get(field) and any(row.get(field) == record[field] and row["split"] != "train" for row in records):
                raise ValueError(f"Train reference has held-out source identity: {field}")
        clips[style], references[style] = raw, metadata
    return seed, clips, {"dataset_manifest_sha256": bundle["dataset_manifest_sha256"],
                         "accepted_json_sha256": accepted_hash, "seed": seed_metadata,
                         "train_references": references,
                         "selection": "train only; idle neutral/static then loop; other styles loop; stable accepted.json order"}


def _root_stats(root, fps, previous=None, requested=None):
    root = _array(root).astype(np.float64)
    positions = root[:, :3]
    valid = np.isfinite(positions).all(axis=-1)
    finite = positions[valid]
    chain = positions if previous is None else np.concatenate((_array(previous).reshape(1, 3), positions))
    intervals = np.isfinite(chain[:-1]).all(-1) & np.isfinite(chain[1:]).all(-1)
    velocity = np.diff(chain, axis=0)[intervals][:, [0, 2]] * fps
    speed = np.linalg.norm(velocity, axis=-1)
    return _jsonable({"frames": len(positions), "finite_root_frames": int(valid.sum()),
        "finite_velocity_intervals": int(intervals.sum()),
        "root_xyz_first_m": positions[0] if len(positions) else None,
        "root_xyz_last_m": positions[-1] if len(positions) else None,
        "root_xyz_min_m": finite.min(0) if len(finite) else None,
        "root_xyz_max_m": finite.max(0) if len(finite) else None,
        "root_height_min_m": float(finite[:, 1].min()) if len(finite) else None,
        "root_height_max_m": float(finite[:, 1].max()) if len(finite) else None,
        "root_planar_speed_mean_mps": float(speed.mean()) if len(speed) else None,
        "root_planar_speed_max_mps": float(speed.max()) if len(speed) else None,
        "command_velocity_error_mean_mps": float(np.linalg.norm(velocity - np.asarray(requested), axis=-1).mean())
            if requested is not None and len(velocity) else None})


def _velocity_metrics(played, generated, root_index, fps):
    played, generated = _array(played).astype(np.float64), _array(generated).astype(np.float64)
    seam = (generated[0] - played[-1]) * fps
    intra = np.diff(generated, axis=0) * fps
    previous = (played[-1] - played[-2]) * fps
    seam_speed = np.linalg.norm(seam, axis=-1)
    intra_speed = np.linalg.norm(intra, axis=-1)
    return _jsonable({
        "cross_replan_seam": {
            "root_velocity_xyz_mps": seam[root_index],
            "root_planar_speed_mps": float(np.linalg.norm(seam[root_index, [0, 2]])),
            "root_speed_3d_mps": float(seam_speed[root_index]),
            "joint_speed_mean_mps": float(seam_speed.mean()),
            "joint_speed_max_mps": float(seam_speed.max()),
            "joint_velocity_change_mean_mps": float(np.linalg.norm(seam - previous, axis=-1).mean())},
        "intra_committed_clip": {"intervals": len(intra),
            "root_planar_speed_mean_mps": float(np.linalg.norm(intra[:, root_index][:, [0, 2]], axis=-1).mean()),
            "root_planar_speed_max_mps": float(np.linalg.norm(intra[:, root_index][:, [0, 2]], axis=-1).max()),
            "joint_speed_mean_mps": float(intra_speed.mean()),
            "joint_speed_max_mps": float(intra_speed.max())},
        "previous_played_root_planar_speed_mps": float(np.linalg.norm(previous[root_index, [0, 2]])),
    })


@contextmanager
def capture_inference(infer):
    """Observe original method returns; return the same tensors and restore methods."""
    snapshots, originals = {}, {}
    names = ("_predict_root_trajectories", "_decode_motions_from_predicted_root_and_pose_tokens")
    available = [name for name in names if callable(getattr(infer, name, None))]
    try:
        for name in available:
            original = getattr(infer, name)
            originals[name] = (name in vars(infer), original)

            def wrapper(*args, _name=name, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                snapshots[_name] = tuple(_array(value) for value in result)
                return result

            setattr(infer, name, wrapper)
        yield snapshots, available
    finally:
        for name, (on_instance, original) in originals.items():
            if on_instance:
                setattr(infer, name, original)
            else:
                delattr(infer, name)


def _trace_arrays(infer, snapshots, frames):
    import torch

    arrays = {}
    root_result = snapshots.get("_predict_root_trajectories")
    if root_result is not None:
        for name, value, rep in (("root_module_global_root", root_result[1], infer.global_motion_rep),
                                 ("root_module_local_root", root_result[2], infer.local_motion_rep)):
            tensor = torch.as_tensor(value, device=infer.device)
            arrays[name] = _array(rep.unnormalize(tensor))[0, :frames]
    decoded = snapshots.get("_decode_motions_from_predicted_root_and_pose_tokens")
    if decoded is not None:
        for name, value, rep in (("decoded_local_root", decoded[0], infer.local_motion_rep),
                                 ("decoded_global_root_before_world_restore", decoded[1], infer.global_motion_rep)):
            arrays[name] = value[0, :frames][:, rep.indices["root"]].copy()
    return arrays


def _constraints_report(inputs, arrays, prefix):
    result = {"num_tokens_input": "MASKED (None)" if inputs["num_tokens"] is None else _jsonable(inputs["num_tokens"]),
              "allowed_pred_num_tokens": _jsonable(inputs["allowed_pred_num_tokens"]),
              "config": _jsonable(inputs.get("config", {})), "source_encoded_tokens": False}
    for name in ("global_root_values", "local_root_values", "local_poses"):
        value, mask = _array(inputs[name]), _array(inputs["has_" + name])
        if value.ndim != 3 or value.shape[:2] != (1, 8) or mask.shape != (1, 8) or mask.dtype != np.bool_:
            raise ValueError("Invalid eight-slot native constraints")
        arrays[prefix + name] = value[0]
        result["has_" + name] = mask[0].tolist()
        result[name] = (_jsonable(value[0]) if name != "local_poses" else
                        {"shape": list(value[0].shape), "sha256": _array_hash(value), "array_key": prefix + name})
    result["future_pose_constraint_slots"] = (4 + np.flatnonzero(_array(inputs["has_local_poses"])[0, 4:])).tolist()
    result["future_pose_constraint_count"] = len(result["future_pose_constraint_slots"])
    return result


def _token_count(count, minimum, maximum):
    array = _array(count)
    if (array.size != 1 or array.dtype == np.bool_ or not np.isfinite(array).all()
            or float(array.item()) != int(array.item()) or not minimum <= int(array.item()) <= maximum):
        raise ValueError("Invalid predicted token count")
    return int(array.item())


def shipped_buffer_step(prediction, device):
    """Port public G1 preview cursor/getter semantics without a qpos converter."""
    from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent

    offsets, context_frames = full_navigation_agent.DEFAULT_PRED_OFFSETS, full_navigation_agent.NUM_FRAMES_PER_TOKEN
    start = context_frames - offsets
    state = SimpleNamespace(frames={"model_features": prediction}, _device=device,
        NUM_FRAMES_PER_TOKEN=context_frames, PRED_OFFSETS=offsets,
        _current_frame_idx=min(start + 8, prediction.shape[1] - 1))
    # get_next_frame advances this cursor once per played frame. Its qpos return
    # is irrelevant here; the motion-feature getter is called directly.
    context = full_navigation_agent.get_context_motion_features(state).clone()
    indices = [max(0, min(state._current_frame_idx - context_frames + index + offsets,
                         prediction.shape[1] - 1)) for index in range(context_frames)]
    return prediction[:, start:start + 8].clone(), context, start, indices


def rollout_case(infer, seed_frames, case, controller=None, *, seed=2333, plans_per_command=6):
    import torch
    from scripts.evaluate_native_mb_pipeline import COMMANDS, command_constraints

    if case not in (*CASES, "native_smart4_fixed8") or not 1 <= plans_per_command <= 6:
        raise ValueError("Invalid bounded comparison case or plan count")
    if case != "legacy_start_only" and controller is None:
        raise ValueError("Smart case requires its native controller")
    dimension, fps = len(infer.global_motion_rep.indices["all"]), infer.motion_rep.fps
    if seed_frames.shape != (1, 4, dimension) or not torch.isfinite(seed_frames).all():
        raise ValueError("Expected exactly four finite seed frames")
    minimum, maximum = infer._args["min_tokens"], infer._args["max_tokens"]
    is_buffer = case in BUFFER_CASES
    history, pieces, plans, arrays, failure = seed_frames.clone(), [seed_frames.clone()], [], {}, None
    root_index = infer.motion_rep.skeleton.root_idx
    with full_precision_evaluation(), torch.inference_mode(), capture_inference(infer) as (snapshots, available):
        for command, velocity, heading in COMMANDS:
            for command_plan in range(plans_per_command):
                index, first = len(plans), sum(piece.shape[1] for piece in pieces)
                prefix = f"plan_{index:03d}_"
                arrays[prefix + "input_context"] = _array(history)[0]
                history_joints = infer.global_motion_rep.inverse(history, False, return_all=True)["posed_joints"]
                played_history = torch.cat(pieces, 1)[:, -4:]
                played_joints = infer.global_motion_rep.inverse(played_history, False, return_all=True)["posed_joints"]
                row = {"plan_index": index, "command_plan_index": command_plan, "command": command,
                       "velocity_xz_mps": list(velocity), "heading_rad": heading, "sampling_seed": seed + index,
                       "first_generated_frame": first, "committed_frames": 0,
                       "input_history_frames": list(range(first - 4, first)) if not is_buffer or index == 0 else None,
                       "input_context_origin": "validation_seed4" if index == 0 else
                           ("previous_prediction_unplayed_lookahead4" if is_buffer else "last_played_generated4"),
                       "input_context_previous_prediction_frames": [8, 9, 10, 11] if index else None,
                       "input_context_is_played": None if index == 0 else not is_buffer,
                       "seam_origin": "evaluator_seed_prefix_initial_boundary" if index == 0 else "cross_replan_played_boundary",
                       "artificial_initial_buffer_seam": is_buffer and index == 0,
                       "input_history_sha256": _array_hash(history),
                       "input_history_array_key": prefix + "input_context",
                       "input_history_root": _root_stats(history_joints[0, :, root_index], fps),
                       "last_played_root": _root_stats(played_joints[0, :, root_index], fps),
                       "future_source_tokens": 0}
                snapshots.clear()
                inputs = None
                try:
                    if case == "legacy_start_only":
                        inputs = command_constraints(infer, history.clone(), velocity, heading, num_tokens=8)
                        inputs["config"] = dict(LEGACY_CONFIG)
                    if history.is_cuda:
                        torch.cuda.synchronize(history.device)
                    started = time.perf_counter()
                    if case == "legacy_start_only":
                        prediction, count = infer.predict(**inputs, info={})
                    else:
                        prediction, count = controller.predict(history.clone(), velocity, heading,
                            style=COMMAND_STYLES[command], random_seed=seed + index)
                        inputs = controller.last_constraints
                    if history.is_cuda:
                        torch.cuda.synchronize(history.device)
                    row["latency_seconds"] = time.perf_counter() - started
                    tokens = _token_count(count, minimum, maximum)
                    row.update(predicted_tokens=tokens, predicted_active_frames=tokens * 4)
                    if case in ("legacy_start_only", "native_smart4_fixed8") and tokens != 8:
                        raise ValueError("Fixed-eight case predicted a different duration")
                    if (not isinstance(prediction, torch.Tensor) or prediction.ndim != 3
                            or prediction.shape[0] != 1 or prediction.shape[2] != dimension
                            or prediction.shape[1] < tokens * 4 or tokens * 4 < 12):
                        raise ValueError("Prediction cannot supply its real horizon and eight played frames")
                    active = prediction[:, :tokens * 4]
                    row["active_prediction_finiteness"] = _finite_counts(active)
                    row["discarded_prediction_context_max_abs_feature_delta"] = float(
                        np.abs(_array(prediction[:, :4]).astype(np.float64) - _array(history)).max())
                    if is_buffer:
                        committed, next_history, playback_start, context_indices = shipped_buffer_step(prediction, infer.device)
                    else:
                        playback_start, context_indices = 4, [8, 9, 10, 11]
                        committed = prediction[:, 4:12].clone()
                        next_history = torch.cat((history, committed), 1)[:, -4:].clone()
                    row["played_prediction_frames_inclusive"] = [playback_start, playback_start + 7]
                    row["next_context_prediction_frames"] = context_indices
                    pieces.append(committed)
                    row["committed_frames"] = 8
                    posed = infer.global_motion_rep.inverse(active, False, return_all=True)["posed_joints"]
                    played_prediction_joints = posed[0, playback_start:playback_start + 8]
                    row["preview_generated_root"] = _root_stats(posed[0, playback_start:, root_index], fps)
                    row["played_generated_root"] = _root_stats(played_prediction_joints[:, root_index], fps,
                        previous=played_joints[0, -1, root_index], requested=velocity)
                    row.update(_velocity_metrics(played_joints[0], played_prediction_joints, root_index, fps))
                    if not torch.isfinite(active).all() or not torch.isfinite(posed).all():
                        raise ValueError("Nonfinite native prediction/FK; retained actual played frames, ending this case")
                    history = next_history
                except (ValueError, RuntimeError) as error:
                    failure = {"plan_index": index, "type": type(error).__name__, "message": str(error)}
                    row["failure"] = failure
                    if inputs is None and controller is not None:
                        inputs = controller.last_constraints
                if inputs is not None:
                    row["constraints"] = _constraints_report(inputs, arrays, prefix)
                    global_root = _array(inputs["global_root_values"])[0]
                    row["condition_start_root"] = _root_stats(global_root[:4], fps)
                    row["condition_target_root"] = _root_stats(global_root[4:], fps)
                trace = _trace_arrays(infer, snapshots, row.get("predicted_active_frames", maximum * 4))
                row["trace_finiteness"] = {name: _finite_counts(value) for name, value in trace.items()}
                for name, value in trace.items():
                    arrays[prefix + name] = value
                if "root_module_global_root" in trace:
                    row["root_module_root"] = _root_stats(trace["root_module_global_root"], fps)
                plans.append(_jsonable(row))
                if failure:
                    break
            if failure:
                break
        features = torch.cat(pieces, 1)
        try:
            joints = _array(infer.global_motion_rep.inverse(features, False, return_all=True)["posed_joints"])[0]
        except (ValueError, RuntimeError) as error:
            joints = np.empty((0, infer.motion_rep.skeleton.nbjoints, 3), dtype=np.float32)
            failure = failure or {"type": type(error).__name__, "message": str(error)}
    arrays.update(global_features=_array(features)[0], prediction=joints,
                  initial_source_frames=_array(seed_frames)[0], fps=np.float32(fps))
    summary = {"status": "failed" if failure else "completed", "planned_plans": 5 * plans_per_command,
               "attempted_plans": len(plans), "played_plans": sum(row["committed_frames"] == 8 for row in plans),
               "seed_frames": 4, "real_generated_frames": features.shape[1] - 4,
               "seam_summary_includes_artificial_initial_buffer_boundary": is_buffer,
               "real_total_frames": features.shape[1], "generated_seconds": (features.shape[1] - 4) / fps,
               "feature_finiteness": _finite_counts(features), "fk_finiteness": _finite_counts(joints),
               "predicted_token_counts": [row["predicted_tokens"] for row in plans if "predicted_tokens" in row],
               "played_generated_root": _root_stats(joints[4:, root_index], fps,
                   previous=joints[3, root_index]) if len(joints) > 4 else None}
    for section, field in (("cross_replan_seam", "root_planar_speed_mps"),
                           ("cross_replan_seam", "joint_speed_mean_mps"),
                           ("intra_committed_clip", "root_planar_speed_mean_mps"),
                           ("intra_committed_clip", "joint_speed_mean_mps")):
        values = [row[section][field] for row in plans if row.get(section, {}).get(field) is not None]
        summary[section + "_" + field] = {"finite_plans": len(values),
            "mean": float(np.mean(values)) if values else None, "max": max(values) if values else None}
    report = {"case": case, "conditioning": CASE_CONDITIONS[case],
              "summary": summary, "plans": plans, "failure": failure,
              "captured_methods": available,
              "trace_coordinates": "Root module: unnormalized input-recentered global/local root; decoded local root: angular velocity, XZ velocity, height; decoded global root precedes official world restoration",
              "trace_availability": "captured original method returns" if available else "methods unavailable; constraints and final output recorded",
              "timing": "instrumented calls include trace snapshots; not deployment latency",
              "subsequent_history": "previous buffer unplayed prediction8:12" if is_buffer else
                  "last four actually played generated frames; predicted context discarded",
              "playback": "exact public G1 preview playback cursor port; FILTER_QPOS=False; not proven UE runtime" if is_buffer else
                  "last-played-four context candidate; skip prediction0:4 and play prediction4:12",
              "FILTER_QPOS": False, "ue_runtime_equivalence_claimed": False,
              "startup": {"official_startup_simulated": False,
                          "seed_prefix_is_evaluator_artifact": True,
                          "seed_prefix_artifact_frames_inclusive": [0, 3],
                          "model_playback_artifact_start_frame": 4,
                          "first_buffer_seam_is_artificial": is_buffer,
                          "future_validation_frames_supplied": 0,
                          "note": "The four validation seed frames are prepended for review. Buffer plan0's seam from that prefix to prediction0 is artificial; only later boundaries reproduce the public G1 buffer playback cursor semantics. The prefix does not simulate official startup or supply future validation motion."},
              "source_root_realignment": case != "legacy_start_only",
              "source_root_realignment_role": "official INPUT conditioning only; played history and generated output retained",
              "decoder_endpoint_condition": False, "pose_sampling": "one all-mask pass, argmax",
              "postprocessing": "none", "root_overwrite": False}
    return arrays, _jsonable(report)


def _checkpoint_records(root_run, pose_run, bundle):
    records = {}
    for name, run in (("root", root_run), ("pose", pose_run), ("tokenizer", bundle["tokenizer"]["run"])):
        run = Path(run).resolve()
        checkpoint = run / "checkpoints/final.ckpt"
        expected = bundle["tokenizer"]["checkpoint_sha256"] if name == "tokenizer" else bundle[name + "_checkpoint_sha256"]
        if sha256(checkpoint) != expected:
            raise ValueError(f"Frozen {name} checkpoint changed")
        records[name] = {"run": str(run), "checkpoint": str(checkpoint), "checkpoint_sha256": expected,
                         "optimizer_updates": bundle["tokenizer"]["global_step"] if name == "tokenizer" else bundle[name + "_step"],
                         "run_contract_sha256": sha256(run / "run_contract.json"),
                         "training_result_sha256": sha256(run / "training_result.json")}
    return records


def evaluate_comparison(infer, dataset_dir, output, bundle, *, seed=2333, include_fixed8=False):
    import torch
    from motionbricks.data.native_mb_locomotion import NativeReferenceClips, NativeSmartLocomotion
    from scripts.evaluate_native_mb_pipeline import COMMANDS

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32 - 30:
        raise ValueError("Seed must be a bounded nonnegative integer")
    output, dataset_dir = Path(output).resolve(), Path(dataset_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if output == dataset_dir or dataset_dir in output.parents or output in dataset_dir.parents:
        raise ValueError("Output must be separate from the source dataset")
    if infer.motion_rep.fps != 30 or infer._args["down_t"] != 2:
        raise ValueError("Requires native 30 Hz and four frames per token")
    raw_seed, references, source_identity = load_sources(dataset_dir, bundle, len(infer.global_motion_rep.indices["all"]))
    device = torch.device(infer.device)
    with full_precision_evaluation(), torch.inference_mode():
        seed_frames = infer.global_motion_rep.change_first_heading(
            torch.as_tensor(raw_seed, dtype=torch.float32, device=device)[None], 0.0, False, False)
        bank = NativeReferenceClips(infer.motion_rep, {
            name: torch.as_tensor(value, dtype=torch.float32, device=device) for name, value in references.items()})
    cases = (*CASES, "native_smart4_fixed8") if include_fixed8 else CASES
    training_steps = {"root": bundle["root_step"], "pose": bundle["pose_step"],
                      "tokenizer": bundle["tokenizer"]["global_step"]}
    source_root = Path(__file__).resolve().parents[1]
    source_files = ("scripts/evaluate_native_mb_smart_locomotion.py", "scripts/evaluate_native_mb_pipeline.py",
                    "scripts/train_native_mb_control.py", "motionbricks/data/native_mb_locomotion.py",
                    "motionbricks/motion_backbone/demo/full_agent.py",
                    "motionbricks/motion_backbone/inference/motion_inference.py",
                    "motionbricks/motion_backbone/neural_modules/root_backbone.py",
                    "motionbricks/motion_backbone/neural_modules/pose_backbone.py",
                    "motionbricks/vqvae/neural_modules/vqvae.py")
    report = {"task": "frozen legacy versus native smart locomotion command comparison",
              "bundle": bundle, "training_steps": training_steps, "convergence_assessed": False,
              "sources": source_identity, "seed": seed, "fps": 30,
              "commands": [{"name": name, "velocity_xz_mps": list(velocity), "heading_rad": heading,
                            "reference_style": COMMAND_STYLES[name]} for name, velocity, heading in COMMANDS],
              "plans_per_command": 6, "commit_frames": 8, "cases": {}, "summary": {},
              "same_frozen_bundle_for_all_cases": True, "quality_approved": False,
              "ue_runtime_equivalence_claimed": False,
              "startup_note": "Every motion artifact prepends seed4 before 240 model-played frames. This is evaluator initialization, not official G1/UE startup; the first buffer boundary is artificial and included in the recorded seam aggregates. No future validation frames are supplied.",
              "playback_note": "Buffer cases port public G1 preview playback with FILTER_QPOS=False; not a claim about paper UE runtime. Past-four cases use only played context. Buffer4_target1 has four context and one target keyframe; sparse1 cases mask both ends to one.",
              "evaluation_precision": "float32; TF32 disabled; no autocast",
              "randomness": "Python/NumPy/Torch RNG reset to same seed per case; reference selection uses seed + plan index",
              "source_sha256": {name: sha256(source_root / name) for name in source_files},
              "reference_input_note": "Entire train clips build the input reference bank. Per-plan four-frame reference windows are input conditions, never source-encoded future tokens.",
              "duration_note": "Legacy fixed8; native predicts within its reported allowed mask and model min/max; optional native4 fixed8 uses the official one-hot allowed mask",
              "metric_note": "Large finite outputs are retained. Cross-replan seam velocity uses the actual last played frame, even when inference uses unplayed lookahead context; intra-clip velocity uses the seven intervals inside each eight-frame played clip. Nonfinite outputs terminate only that case; JSON uses null with explicit finite counts, arrays retain the raw values.",
              "foot_slip_and_penetration": "not_assessed", "postprocessing": "none"}
    output.mkdir(parents=True, exist_ok=False)
    infer.eval()
    for case in cases:
        conditions = CASE_CONDITIONS[case]
        controller = None if case == "legacy_start_only" else NativeSmartLocomotion(
            infer, bank, keyframes_per_end=conditions["target_keyframes"], context_keyframes=conditions["context_keyframes"],
            fixed_tokens=8 if case == "native_smart4_fixed8" else None, source_root_realignment=True)
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                with full_precision_evaluation():
                    arrays, case_report = rollout_case(infer, seed_frames, case, controller, seed=seed)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        if controller is not None:
            for plan in case_report["plans"]:
                style = COMMAND_STYLES[plan["command"]]
                reference = source_identity["train_references"][style]
                first = plan["sampling_seed"] % (reference["shard_frames"] - 4)
                plan["train_reference_input"] = {"style": style, "shard": reference["shard"],
                    "shard_sha256": reference["shard_sha256"], "split": "train",
                    "source_frames": list(range(first, first + 4)), "source_encoded_tokens": False}
        case_report["bundle"] = bundle
        case_report["training_steps"] = dict(training_steps)
        case_report["convergence_assessed"] = False
        case_output = output / case
        case_output.mkdir(exist_ok=False)
        np.savez_compressed(case_output / "motion.npz", **arrays)
        write_json(case_output / "report.json", case_report)
        report["cases"][case] = {"report": f"{case}/report.json", "motion": f"{case}/motion.npz"}
        report["summary"][case] = case_report["summary"]
        write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-run", type=Path, required=True)
    parser.add_argument("--pose-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--include-fixed8", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new output directory")
    if not 0 <= args.seed < 2 ** 32 - 30 or (args.gpu is not None and args.gpu < 0):
        parser.error("Seed/GPU index is invalid")
    if (args.device == "cuda") != (args.gpu is not None):
        parser.error("CUDA requires --gpu; CPU does not accept --gpu")
    gpu = select_gpu(args.gpu) if args.device == "cuda" else None
    import torch
    from scripts.evaluate_native_mb_pipeline import load_bundle

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(4)
    with full_precision_evaluation():
        infer, dataset, bundle = load_bundle(args.root_run, args.pose_run, args.device)
        validate_frozen_bundle(bundle)
        bundle = {**bundle, "checkpoints": _checkpoint_records(args.root_run, args.pose_run, bundle),
                  "device": args.device, "gpu_at_launch": gpu,
                  "gpu_name": torch.cuda.get_device_name() if args.device == "cuda" else None}
        report = evaluate_comparison(infer, dataset, args.output, bundle,
                                     seed=args.seed, include_fixed8=args.include_fixed8)
    print(json.dumps(report["summary"], indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
