"""Independent held-out evaluation for native MotionBricks Root and Pose stages."""

import hashlib
import json
from pathlib import Path

import numpy as np


FRAMES_PER_TOKEN = 4
NUM_TOKENS = 16
GLOBAL_WINDOW_FRAMES = NUM_TOKENS * FRAMES_PER_TOKEN + 1
LOCAL_WINDOW_FRAMES = NUM_TOKENS * FRAMES_PER_TOKEN
SCORE_START = 4
SCORE_END = LOCAL_WINDOW_FRAMES - 1
POSE_MODES = ("context4_stress_probe", "inbetween_endpoint_pose", "native_keyframe1_inbetween")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _fixed_validation_windows(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    accepted_path = dataset_dir / "accepted.json"
    manifest_path = dataset_dir / "native_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset native_manifest.json not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    artifacts = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict) or artifacts.get("accepted.json") != _sha256(accepted_path):
        raise ValueError("Dataset manifest does not bind the held-out accepted.json split list")
    identity = {"dataset_manifest_sha256": _sha256(manifest_path),
                "accepted_json_sha256": _sha256(accepted_path)}
    with accepted_path.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("accepted.json must contain a list of split records")
    split_by_shard = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("accepted.json contains a malformed split record")
        shard_name = record.get("shard")
        if isinstance(shard_name, str):
            split_by_shard.setdefault(shard_name, set()).add(record.get("split"))
    if any(len(splits) != 1 for splits in split_by_shard.values()):
        raise ValueError("Dataset leakage: a shard is assigned to multiple splits")
    windows, excluded = [], []
    validation_shards = set()
    for record in records:
        if record.get("split") != "validation":
            continue
        shard = record.get("shard")
        behavior = record.get("behavior", "unknown")
        if not isinstance(shard, str) or not shard.strip():
            raise ValueError("Validation record is missing a shard identity")
        if shard in validation_shards:
            raise ValueError(f"Duplicate validation shard identity: {shard}")
        validation_shards.add(shard)
        if not isinstance(behavior, str) or not behavior.strip():
            raise ValueError("Validation record has an invalid behavior identity")
        shard_path = (dataset_dir / shard).resolve()
        if dataset_dir not in shard_path.parents or not shard_path.is_file():
            raise ValueError(f"Validation shard is missing or escapes the dataset: {shard}")
        expected_shard_hash = artifacts.get(shard)
        if not isinstance(expected_shard_hash, str) or _sha256(shard_path) != expected_shard_hash:
            raise ValueError(f"Dataset manifest does not bind validation shard: {shard}")
        with np.load(shard_path, allow_pickle=False) as data:
            if "global_features" not in data:
                raise ValueError(f"Validation shard lacks global_features: {shard}")
            frame_count = len(data["global_features"])
        if frame_count < GLOBAL_WINDOW_FRAMES:
            excluded.append({"shard": shard, "frames": frame_count,
                             "reason": "shorter than the fixed 65-global-frame window"})
            continue
        starts = sorted(set(np.linspace(
            0, frame_count - GLOBAL_WINDOW_FRAMES, 3
        ).astype(int).tolist()))
        shard_hash = _sha256(shard_path)
        for start in starts:
            windows.append({"shard": shard, "shard_sha256": shard_hash,
                            "behavior": behavior, "start": int(start)})
    if not windows:
        raise ValueError("No eligible validation windows in the manifest-bound dataset")
    return dataset_dir, windows, excluded, identity


def _source_window(dataset_dir, window):
    with np.load(dataset_dir / window["shard"], allow_pickle=False) as data:
        source = np.asarray(data["global_features"][
            window["start"]:window["start"] + GLOBAL_WINDOW_FRAMES
        ]).copy()
    if (source.ndim != 2 or source.shape[0] != GLOBAL_WINDOW_FRAMES or source.shape[1] == 0
            or not np.isfinite(source).all()):
        raise ValueError(f"Invalid or nonfinite validation source: {window['shard']}@{window['start']}")
    return source


def _device(model):
    import torch

    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _as_json_finite(value, name):
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite {name}")
    return array


def _canonical_motion(model, raw_global, device):
    import torch
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    rep = model.motion_rep
    raw = torch.as_tensor(raw_global, dtype=torch.float32, device=device)[None]
    raw = rep.change_first_heading(raw, 0.0, is_normalized=False, to_normalize=False)
    global_features = rep.normalize(raw)[:, :LOCAL_WINDOW_FRAMES]
    local = rep.dual_rep.global_to_local(
        raw, is_normalized=False, to_normalize=True,
        lengths=torch.full((1,), GLOBAL_WINDOW_FRAMES, device=device),
    )[:, :LOCAL_WINDOW_FRAMES]
    if not torch.isfinite(local).all() or not torch.isfinite(global_features).all():
        raise ValueError("Canonical source representation is nonfinite")
    return raw, global_features, local


def _root_evaluation(model, dataset_dir, windows, excluded, identity, output, seed):
    import torch
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation

    rep = model.motion_rep
    global_rep, local_rep = rep.dual_rep.global_motion_rep, rep.dual_rep.local_motion_rep
    backbone = model.backbone_net
    token_frames = backbone.get_num_frames_per_token()
    if token_frames != FRAMES_PER_TOKEN:
        raise ValueError(f"Expected four frames per token; model reports {token_frames}")
    device = _device(model)
    rows = []
    min_tokens, max_tokens = int(backbone._args["min_tokens"]), int(backbone._args["max_tokens"])
    if not min_tokens <= NUM_TOKENS <= max_tokens:
        raise ValueError("Root model cannot represent the fixed 16-token primary horizon")
    allowed_tokens = torch.zeros((1, max_tokens - min_tokens + 1), device=device)
    allowed_tokens[0, NUM_TOKENS - min_tokens] = 1
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with full_precision_evaluation(), torch.inference_mode():
            for window in windows:
                source = _source_window(dataset_dir, window)
                raw, _, source_local = _canonical_motion(model, source, device)
                source_root = extract_feature_from_motion_rep(raw, global_rep, "root")[:, :LOCAL_WINDOW_FRAMES]
                if not torch.isfinite(source_root).all():
                    raise ValueError("Nonfinite source root features")
                root_cond = global_rep.normalize(source_root)
                sparse_global = torch.cat(
                    [root_cond[:, :FRAMES_PER_TOKEN], root_cond[:, -FRAMES_PER_TOKEN:]], dim=1
                )
                has_global = torch.ones((1, 2 * FRAMES_PER_TOKEN), dtype=torch.bool, device=device)
                local_root_dim = extract_feature_from_motion_rep(
                    torch.zeros((1, 2 * FRAMES_PER_TOKEN, len(local_rep.indices["all"])), device=device),
                    local_rep, model.args["local_root_feature"],
                ).shape[-1]
                local_pose_dim = extract_feature_from_motion_rep(
                    torch.zeros((1, 2 * FRAMES_PER_TOKEN, len(local_rep.indices["all"])), device=device),
                    local_rep, model.args["local_pose_feature"],
                ).shape[-1]
                empty_local_root = torch.zeros((1, 2 * FRAMES_PER_TOKEN, local_root_dim), device=device)
                empty_local_pose = torch.zeros((1, 2 * FRAMES_PER_TOKEN, local_pose_dim), device=device)
                no_local = torch.zeros_like(has_global)
                local_root = extract_feature_from_motion_rep(source_local, local_rep, model.args["local_root_feature"])
                local_pose = extract_feature_from_motion_rep(source_local, local_rep, model.args["local_pose_feature"])
                local_root = torch.cat((local_root[:, :4], local_root[:, -4:]), dim=1)
                local_pose = torch.cat((local_pose[:, :4], local_pose[:, -4:]), dim=1)
                single_mask = torch.zeros_like(has_global)
                single_mask[:, [3, 7]] = True
                num_tokens = torch.full((1, 1), backbone.MASKED_NUM_TOKENS,
                                        dtype=torch.long, device=device)
                modes = {}
                for mode, allowed in (("fixed_horizon_16", allowed_tokens),
                                      ("implicit_duration_diagnostic", None),
                                      ("root_only_keyframe1", allowed_tokens),
                                      ("native_keyframe1", allowed_tokens)):
                    current_global_mask = single_mask if "keyframe1" in mode else has_global
                    current_local_mask = single_mask if mode == "native_keyframe1" else no_local
                    root_output = backbone(
                        sparse_global * current_global_mask[..., None], current_global_mask,
                        local_root * current_local_mask[..., None], current_local_mask,
                        local_pose * current_local_mask[..., None], current_local_mask,
                        num_tokens, text_embeddings=None, has_text_embeddings=None,
                        allowed_pred_num_tokens=allowed,
                        config={"allow_pred_out_of_reach_num_tokens": False},
                    )
                    predicted_num_tokens = int(root_output["pred_num_tokens"].reshape(-1)[0])
                    predicted_frames = predicted_num_tokens * FRAMES_PER_TOKEN
                    if (predicted_num_tokens < min_tokens or predicted_num_tokens > max_tokens
                            or predicted_frames > root_output["pred_global_root_values"].shape[1]):
                        raise ValueError(f"Invalid predicted duration for {window['shard']} ({mode})")
                    if allowed is not None and predicted_num_tokens != NUM_TOKENS:
                        raise ValueError("Root backbone did not honor the one-hot 16-token duration constraint")
                    predicted = global_rep.unnormalize(
                        root_output["pred_global_root_values"][:, :predicted_frames]
                    )
                    if not torch.isfinite(predicted).all():
                        raise ValueError(f"Nonfinite predicted root trajectory for {window['shard']} ({mode})")
                    overlap_end = min(predicted_frames, LOCAL_WINDOW_FRAMES) - 1
                    if overlap_end <= SCORE_START:
                        raise ValueError(f"Predicted root duration too short to score: {predicted_frames} frames")
                    if mode == "fixed_horizon_16" and overlap_end != SCORE_END:
                        raise ValueError("Fixed-horizon Root output must score all frames 4-63")
                    modes[mode] = _root_track_metrics(
                        source_root, predicted, global_rep, rep.fps, predicted_num_tokens,
                        SCORE_END if mode == "fixed_horizon_16" else overlap_end,
                    )
                rows.append({
                    **window,
                    "source_window_sha256": _array_sha256(source),
                    "source_window_shape": list(source.shape),
                    "source_window_dtype": source.dtype.str,
                    "conditioned_global_root_source_frames": list(range(4)) + list(range(60, 64)),
                    "conditioned_local_root_frames": [], "conditioned_pose_frames": [],
                    "fixed_horizon_16": modes["fixed_horizon_16"],
                    "implicit_duration_diagnostic": modes["implicit_duration_diagnostic"],
                    "root_only_keyframe1": modes["root_only_keyframe1"],
                    "native_keyframe1": modes["native_keyframe1"],
                })

    metric_names = ("root_position_error_m", "root_yaw_error_rad", "root_velocity_error_mps")
    summaries = {
        mode: {name: float(np.mean([row[mode][name] for row in rows])) for name in metric_names}
        | {"predicted_duration_seconds": float(np.mean(
            [row[mode]["predicted_duration_seconds"] for row in rows]
        ))}
        for mode in ("fixed_horizon_16", "implicit_duration_diagnostic", "root_only_keyframe1", "native_keyframe1")
    }

    result = {
        "task": "native MotionBricks Root sparse-condition held-out evaluation",
        "stage": "root", "dataset_split": "validation", "seed": seed,
        **identity, "evaluator_source_sha256": _sha256(Path(__file__)),
        "fps": float(rep.fps), "global_window_frames": GLOBAL_WINDOW_FRAMES,
        "maximum_local_prediction_frames": int(backbone._args["max_tokens"] * FRAMES_PER_TOKEN),
        "scored_frames_start": SCORE_START,
        "conditions": {"global_root_frames": list(range(4)) + list(range(60, 64)),
                       "local_root_frames": [], "pose_frames": [],
                       "future_body_used": False},
        "additional_mode_conditions": {
            "root_only_keyframe1": {"global_root_frames": [3, 63], "local_root_frames": [], "pose_frames": []},
            "native_keyframe1": {"global_root_frames": [3, 63], "local_root_frames": [3, 63], "pose_frames": [3, 63],
                                 "future_body_used": True, "purpose": "native early-curriculum in-between task, not WASD"}},
        "interpretation": "Root-only probes omit local-root and pose constraints that the shipped training always samples. Four-frame probes also exceed the early one-frame-per-end curriculum. native_keyframe1 supplies all native condition types, including a source target pose; keep all probes separate.",
        "duration": "fixed_horizon_16 is primary; implicit_duration_diagnostic separately predicts duration",
        "training_duration_condition_probability": float(model.args["prob_provide_num_tokens"]),
        "endpoint_condition_placement": "fixed mode: frames 60-63; diagnostic: predicted final four frames",
        "temporal_alignment": "fixed mode compares the complete same-time horizon at frames 4-63; the implicit-duration diagnostic compares only prefix overlap and is not the primary score",
        "sampling": "original Root backbone; num_tokens remains MASKED; primary uses one-hot allowed_pred_num_tokens for 16",
        "root_sampling_modes": {
            "fixed_horizon_16": {"num_tokens": "MASKED", "allowed_pred_num_tokens": [NUM_TOKENS]},
            "implicit_duration_diagnostic": {"num_tokens": "MASKED", "allowed_pred_num_tokens": None},
        },
        "evaluation_precision": "float32; CUDA matmul and cuDNN TF32 disabled during evaluation only",
        "postprocessing": "none", "root_overwrite": False,
        "summary": summaries,
        "windows": rows, "excluded_validation_clips": excluded,
        "quality_approved": False,
        "renderer_arrays": "not applicable: Root-only model does not predict full-body joint positions",
    }
    _write_report(output, result)
    return result


def _root_track_metrics(source_root, predicted, global_rep, fps, predicted_num_tokens, score_end):
    import torch

    source_scored = source_root[:, SCORE_START:score_end + 1]
    predicted_scored = predicted[:, SCORE_START:score_end + 1]
    source_pos, source_yaw = global_rep.compute_root_pos_and_rot(
        source_scored, return_quat=False, return_angle=True
    )
    pred_pos, pred_yaw = global_rep.compute_root_pos_and_rot(
        predicted_scored, return_quat=False, return_angle=True
    )
    source_velocity = torch.diff(source_pos, dim=1) * fps
    pred_velocity = torch.diff(pred_pos, dim=1) * fps
    yaw_delta = torch.atan2(torch.sin(pred_yaw - source_yaw), torch.cos(pred_yaw - source_yaw))
    source_yaw_step = torch.diff(source_yaw, dim=1)
    pred_yaw_step = torch.diff(pred_yaw, dim=1)
    source_yaw_velocity = torch.atan2(torch.sin(source_yaw_step), torch.cos(source_yaw_step)) * fps
    pred_yaw_velocity = torch.atan2(torch.sin(pred_yaw_step), torch.cos(pred_yaw_step)) * fps
    predicted_frames = predicted_num_tokens * FRAMES_PER_TOKEN
    return {
        "predicted_num_tokens": predicted_num_tokens,
        "predicted_frames": predicted_frames,
        "predicted_duration_seconds": predicted_frames / float(fps),
        "source_window_duration_seconds": LOCAL_WINDOW_FRAMES / float(fps),
        "scored_frames_inclusive": [SCORE_START, score_end],
        "root_position_error_m": float((pred_pos - source_pos).norm(dim=-1).mean()),
        "root_yaw_error_rad": float(yaw_delta.abs().mean()),
        "root_velocity_error_mps": float((pred_velocity - source_velocity).norm(dim=-1).mean()),
        "source_root_position_xyz_m": source_pos[0].cpu().tolist(),
        "predicted_root_position_xyz_m": pred_pos[0].cpu().tolist(),
        "source_root_yaw_rad": source_yaw[0].cpu().tolist(),
        "predicted_root_yaw_rad": pred_yaw[0].cpu().tolist(),
        "source_root_velocity_xyz_mps": source_velocity[0].cpu().tolist(),
        "predicted_root_velocity_xyz_mps": pred_velocity[0].cpu().tolist(),
        "source_root_yaw_velocity_radps": source_yaw_velocity[0].cpu().tolist(),
        "predicted_root_yaw_velocity_radps": pred_yaw_velocity[0].cpu().tolist(),
    }


def _pose_evaluation(model, dataset_dir, windows, excluded, identity, output, seed):
    import torch
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation

    rep = model.motion_rep
    global_rep, local_rep = rep.dual_rep.global_motion_rep, rep.dual_rep.local_motion_rep
    backbone = model.backbone_net
    # Lightning teardown moves registered modules, not the upstream support dict.
    device = _device(model)
    model.move_supporting_nets_to_device(device)
    vqvae = model.supporting_nets["pose_net"]
    if not model.vqvae_model_loaded:
        raise ValueError("Pose inference requires the parent loader's verified tokenizer binding")
    if backbone.get_num_frames_per_token() != FRAMES_PER_TOKEN:
        raise ValueError("Pose model must use four frames per token")
    if vqvae.motion_rep.name != "local":
        raise ValueError("Pose evaluation requires the original local-representation VQ-VAE decoder")
    args = model.args
    rows, examples = [], {}
    mode_examples = {mode: {} for mode in POSE_MODES}
    generator_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=generator_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with full_precision_evaluation(), torch.inference_mode():
            for window in windows:
                source_raw = _source_window(dataset_dir, window)
                raw, source_global, local = _canonical_motion(model, source_raw, device)
                num_tokens = torch.full((1, 1), NUM_TOKENS, dtype=torch.long, device=device)
                root_source = source_global
                if args["cond_root_feature_is_from_motion_rep"] == "local":
                    root_cond = extract_feature_from_motion_rep(
                        local, local_rep, args["cond_root_feature"]
                    )
                elif args["cond_root_feature_is_from_motion_rep"] == "global":
                    root_cond = extract_feature_from_motion_rep(
                        root_source, global_rep, args["cond_root_feature"]
                    )
                else:
                    raise ValueError("Unsupported native pose root condition representation")
                pose_feature = extract_feature_from_motion_rep(
                    local, local_rep, args["local_pose_feature"]
                )
                source_local = local_rep.unnormalize(local)
                source_motion = local_rep.inverse(source_local, is_normalized=False, return_all=True)
                source_joints = _as_json_finite(source_motion["posed_joints"], "source posed joints")[0]
                external_mode = vqvae.decoder_external_cond_feature_mode
                external_cond = extract_feature_from_motion_rep(local, local_rep, external_mode)
                for mode in POSE_MODES:
                    pose_cond = torch.zeros_like(pose_feature)
                    pose_cond[:, :FRAMES_PER_TOKEN] = pose_feature[:, :FRAMES_PER_TOKEN]
                    has_pose_cond = torch.zeros((1, LOCAL_WINDOW_FRAMES), dtype=torch.bool, device=device)
                    has_pose_cond[:, :FRAMES_PER_TOKEN] = True
                    if mode == "inbetween_endpoint_pose":
                        pose_cond[:, -FRAMES_PER_TOKEN:] = pose_feature[:, -FRAMES_PER_TOKEN:]
                        has_pose_cond[:, -FRAMES_PER_TOKEN:] = True
                    elif mode == "native_keyframe1_inbetween":
                        has_pose_cond[:] = False
                        has_pose_cond[:, [3, 63]] = True
                        pose_cond[:] = 0
                        pose_cond[:, [3, 63]] = pose_feature[:, [3, 63]]
                    pose_tokens = torch.full(
                        (1, NUM_TOKENS, backbone.get_num_heads()[0]),
                        backbone.POSE_MASK_ID, dtype=torch.long, device=device,
                    )
                    logits = backbone(pose_tokens, root_cond, pose_cond, has_pose_cond, num_tokens,
                                      text_embeddings=None, has_text_embeddings=None)["pose_logits"]
                    if not torch.isfinite(logits).all():
                        raise ValueError(f"Nonfinite pose logits for {window['shard']} ({mode})")
                    # Match full_agent's native in-between config: one all-mask pass and argmax.
                    sampled_tokens = logits.argmax(dim=-1)
                    token_mask = torch.ones((1, NUM_TOKENS), dtype=torch.bool, device=device)
                    decoded = vqvae.forward_decoder(
                        sampled_tokens, target_cond=pose_cond, has_target_cond=has_pose_cond,
                        external_cond=external_cond, use_overall_indices=False, token_mask=token_mask,
                    )["recon_state"]
                    if not torch.isfinite(decoded).all():
                        raise ValueError(f"Nonfinite decoded pose for {window['shard']} ({mode})")
                    predicted_local = local_rep.unnormalize(decoded)
                    predicted_motion = local_rep.inverse(predicted_local, is_normalized=False, return_all=True)
                    predicted_joints = _as_json_finite(
                        predicted_motion["posed_joints"], f"predicted posed joints ({mode})"
                    )[0]
                    if source_joints.shape != predicted_joints.shape:
                        raise ValueError("Source and generated joint-array shapes differ")
                    scored_source = source_joints[SCORE_START:SCORE_END + 1]
                    scored_predicted = predicted_joints[SCORE_START:SCORE_END + 1]
                    joint_error = np.linalg.norm(scored_predicted - scored_source, axis=-1)
                    predicted_velocity = np.diff(scored_predicted, axis=0) * float(rep.fps)
                    source_velocity = np.diff(scored_source, axis=0) * float(rep.fps)
                    row = {
                        **window,
                        "source_window_sha256": _array_sha256(source_raw),
                        "source_motion_sha256": _array_sha256(source_joints),
                        "source_window_shape": list(source_raw.shape),
                        "source_window_dtype": source_raw.dtype.str,
                        "conditioning_mode": mode,
                        "conditioning": f"known source root trajectory + source pose frames {has_pose_cond[0].nonzero().flatten().tolist()}; all 16 tokens generated",
                        "conditioned_pose_frames": has_pose_cond[0].nonzero().flatten().tolist(),
                        "known_root_frames": list(range(LOCAL_WINDOW_FRAMES)),
                        "scored_frames_inclusive": [SCORE_START, SCORE_END],
                        "mpjpe_m": float(joint_error.mean()),
                        "joint_velocity_error_mps": float(np.linalg.norm(
                            predicted_velocity - source_velocity, axis=-1
                        ).mean()),
                        "root_relative_fk_mpjpe_m": _root_relative_error(
                            scored_predicted, scored_source, rep.skeleton.root_idx
                        ),
                    }
                    if mode in ("inbetween_endpoint_pose", "native_keyframe1_inbetween"):
                        stop = 60 if mode == "inbetween_endpoint_pose" else 63
                        interior_source = source_joints[4:stop]
                        interior_predicted = predicted_joints[4:stop]
                        interior_predicted_velocity = np.diff(interior_predicted, axis=0) * float(rep.fps)
                        interior_source_velocity = np.diff(interior_source, axis=0) * float(rep.fps)
                        row["unconditioned_interior"] = {
                            "scored_frames_inclusive": [4, stop - 1],
                            "mpjpe_m": float(np.linalg.norm(
                                interior_predicted - interior_source, axis=-1
                            ).mean()),
                            "joint_velocity_error_mps": float(np.linalg.norm(
                                interior_predicted_velocity - interior_source_velocity, axis=-1
                            ).mean()),
                        }
                    rows.append(row)
                    if window["behavior"] not in mode_examples[mode]:
                        mode_examples[mode][window["behavior"]] = row
                        mode_output = output / mode
                        mode_output.mkdir(exist_ok=True)
                        np.savez_compressed(mode_output / f"{_safe_name(window['behavior'])}_reconstruction.npz",
                            prediction=predicted_joints.astype(np.float32), source=source_joints.astype(np.float32),
                            fps=np.asarray(float(rep.fps), dtype=np.float32))
                    if mode == "context4_stress_probe" and window["behavior"] not in examples:
                        examples[window["behavior"]] = row
                        np.savez_compressed(
                            output / f"{_safe_name(window['behavior'])}_reconstruction.npz",
                            prediction=predicted_joints.astype(np.float32),
                            source=source_joints.astype(np.float32),
                            fps=np.asarray(float(rep.fps), dtype=np.float32),
                        )

    expected_behaviors = {window['behavior'] for window in windows}
    if set(examples) != expected_behaviors or len({_safe_name(name) for name in examples}) != len(examples):
        raise ValueError('Renderer examples must cover every validation behavior with unique names')
    summary = {
        mode: {name: float(np.mean([row[name] for row in rows if row["conditioning_mode"] == mode]))
               for name in ("mpjpe_m", "joint_velocity_error_mps", "root_relative_fk_mpjpe_m")}
        for mode in POSE_MODES
    }
    endpoint_rows = [row for row in rows if row["conditioning_mode"] == "inbetween_endpoint_pose"]
    summary["inbetween_endpoint_pose"]["unconditioned_interior"] = {
        "scored_frames_inclusive": [4, 59],
        "mpjpe_m": float(np.mean([
            row["unconditioned_interior"]["mpjpe_m"] for row in endpoint_rows
        ])),
        "joint_velocity_error_mps": float(np.mean([
            row["unconditioned_interior"]["joint_velocity_error_mps"] for row in endpoint_rows
        ])),
    }
    grouped_rows = {}
    identity_fields = ("shard", "shard_sha256", "behavior", "start", "source_window_sha256",
                       "source_motion_sha256", "source_window_shape", "source_window_dtype")
    for row in rows:
        key = (row["shard"], row["start"])
        grouped = grouped_rows.setdefault(key, {field: row[field] for field in identity_fields})
        mode_report = {field: row[field] for field in (
            "conditioning", "conditioned_pose_frames", "known_root_frames",
            "scored_frames_inclusive", "mpjpe_m", "joint_velocity_error_mps",
            "root_relative_fk_mpjpe_m",
        )}
        if "unconditioned_interior" in row:
            mode_report["unconditioned_interior"] = row["unconditioned_interior"]
        grouped.setdefault("modes", {})[row["conditioning_mode"]] = mode_report
    result = {
        "task": "native MotionBricks Pose conditional held-out generation",
        "stage": "pose", "dataset_split": "validation", "seed": seed,
        **identity, "evaluator_source_sha256": _sha256(Path(__file__)),
        "fps": float(rep.fps), "global_window_frames": GLOBAL_WINDOW_FRAMES,
        "local_prediction_frames": LOCAL_WINDOW_FRAMES,
        "duration_seconds": LOCAL_WINDOW_FRAMES / float(rep.fps),
        "first_to_last_frame_seconds": (LOCAL_WINDOW_FRAMES - 1) / float(rep.fps),
        "scored_frames_inclusive": [SCORE_START, SCORE_END],
        "conditions": {"root": "known source trajectory at frames 0-63",
                       "pose_modes": {
                           "context4_stress_probe": list(range(FRAMES_PER_TOKEN)),
                           "inbetween_endpoint_pose": list(range(FRAMES_PER_TOKEN)) + list(range(60, 64)),
                           "native_keyframe1_inbetween": [3, 63],
                       }, "generated_pose_tokens_per_mode": NUM_TOKENS,
                       "source_encoded_future_tokens": False},
        "pose_condition_training_policy": {
            "no_end_keyframe_prob": float(args["no_end_keyframe_prob"]),
            "keyframe_num_warmup_steps": int(args["keyframe_num_warmup_steps"]),
        },
        "pose_sampling_config": {
            "num_inference_step": 1,
            "pose_token_sampling_use_argmax": True,
            "all_pose_tokens_initially_masked": True,
        },
        "sampling": "full_agent native in-between config: one all-mask Pose backbone pass, argmax token selection, fixed 16 tokens",
        "evaluation_precision": "float32; CUDA matmul and cuDNN TF32 disabled during evaluation only",
        "postprocessing": "none", "root_overwrite": False,
        "summary": summary, "windows_per_mode": len(windows),
        "windows": list(grouped_rows.values()), "examples": examples,
        "excluded_validation_clips": excluded,
        "interpretation": "Pose is non-unique given root and boundary poses. context4_stress_probe is out of distribution because the original Pose config has no_end_keyframe_prob=0.0; inbetween_endpoint_pose supplies four source poses per end, beyond the early curriculum. native_keyframe1_inbetween supplies one source pose per end, matching early keyframe counts. None of these modes is WASD control.",
        "quality_approved": False,
    }
    _write_report(output, result)
    for mode in POSE_MODES:
        mode_result = {**result, "render_mode": mode, "artifact_scope": mode,
                       "summary": result["summary"][mode], "examples": mode_examples[mode]}
        _write_report(output / mode, mode_result)
    return result


def _root_relative_error(predicted, source, root_index):
    pred_relative = predicted - predicted[:, root_index:root_index + 1]
    source_relative = source - source[:, root_index:root_index + 1]
    return float(np.linalg.norm(pred_relative - source_relative, axis=-1).mean())


def _safe_name(value):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_") or "behavior"


def _write_report(output, report):
    path = output / "metrics.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate_control(model, stage, dataset_dir: Path, output: Path, seed=2333):
    """Evaluate an already-loaded native Root or Pose Lightning model on fixed validation windows."""
    if stage not in {"root", "pose"}:
        raise ValueError("stage must be 'root' or 'pose'")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = Path(output).resolve()
    dataset_dir, windows, excluded, identity = _fixed_validation_windows(dataset_dir)
    if output == dataset_dir or dataset_dir in output.parents:
        raise ValueError("Evaluation output must be outside the held-out dataset")
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    for window in windows:
        _source_window(dataset_dir, window)
    if stage == "pose" and not model.vqvae_model_loaded:
        raise ValueError("Pose inference requires the parent loader's verified tokenizer binding")
    output.mkdir(parents=True)
    model.eval()
    if stage == "root":
        return _root_evaluation(model, dataset_dir, windows, excluded, identity, output, seed)
    return _pose_evaluation(model, dataset_dir, windows, excluded, identity, output, seed)
