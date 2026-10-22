"""Offline decoder-only response probe for the experimental dense-root-height codec."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_codec_height import (
    CODEC_HEIGHT_CONDITIONING,
    HEIGHT_EXTERNAL_FEATURE_MODE,
    SOURCE_EXTERNAL_FEATURE_MODE,
)
from motionbricks.data.native_mb_online_live import (
    planar_command_goal,
    predict_planar_terrain,
)
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb_control import state_digest


SCHEMA = "motionbricks.parc_decoder_height_response.v1"
NUM_TOKENS = 8
HISTORY_FRAMES = 4
RAMP_START_FRAME = 4
RAMP_FRAME_COUNT = 12
RAMP_ENDPOINT_M = 1.0
INFERENCE_CONFIG = {
    "num_inference_step": 4,
    "pose_token_sampling_use_argmax": True,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(values):
    digest = hashlib.sha256()
    for name in sorted(values):
        value = torch.as_tensor(values[name]).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def height_feature_location(motion_rep, feature_mode):
    """Locate global_root_y in a feature vector using MotionRep feature indices."""
    feature_indices = np.asarray(extract_feature_from_motion_rep(
        None, motion_rep, feature_mode, fetch_feat_idx=True,
    ), dtype=np.int64).reshape(-1)
    height_indices = np.asarray(motion_rep.indices["global_root_y"], dtype=np.int64).reshape(-1)
    if height_indices.size != 1:
        raise ValueError("Expected exactly one global_root_y representation feature")
    matches = np.flatnonzero(feature_indices == int(height_indices[0]))
    return {
        "available": bool(matches.size == 1),
        "feature_mode": feature_mode,
        "feature_width": int(feature_indices.size),
        "feature_indices": feature_indices.tolist(),
        "global_root_y_rep_index": int(height_indices[0]),
        "global_root_y_channel_index": int(matches[0]) if matches.size == 1 else None,
    }


def _require_height_location(motion_rep, feature_mode):
    location = height_feature_location(motion_rep, feature_mode)
    if not location["available"]:
        raise ValueError(f"Feature mode {feature_mode!r} has no global_root_y channel")
    return location


def apply_height_ramp(external_cond, motion_rep, feature_mode, endpoint_m,
                      start_frame=RAMP_START_FRAME, frame_count=RAMP_FRAME_COUNT):
    """Change only the external Y feature, expressing the ramp in meters via Stats."""
    if (external_cond.ndim != 3 or not external_cond.is_floating_point()
            or external_cond.shape[0] != 1):
        raise ValueError("Decoder external condition must be floating-point [1,T,D]")
    if (not isinstance(start_frame, int) or not isinstance(frame_count, int)
            or start_frame < 0 or frame_count < 1
            or start_frame + frame_count > external_cond.shape[1]):
        raise ValueError("Height ramp window is outside the captured decoder condition")
    if not np.isfinite(endpoint_m):
        raise ValueError("Height ramp endpoint must be finite")

    location = _require_height_location(motion_rep, feature_mode)
    channel = location["global_root_y_channel_index"]
    rep_index = np.asarray([location["global_root_y_rep_index"]], dtype=np.int64)
    normalized_y = external_cond[..., channel:channel + 1]
    raw_y = motion_rep.stats.unnormalize(normalized_y, index=rep_index)
    ramp = (torch.arange(1, frame_count + 1, device=external_cond.device,
                         dtype=external_cond.dtype) * (float(endpoint_m) / frame_count))
    changed_raw_y = raw_y.clone()
    changed_raw_y[:, start_frame:start_frame + frame_count] += ramp[None, :, None]
    changed_normalized_y = motion_rep.stats.normalize(changed_raw_y, index=rep_index)

    counterfactual = external_cond.clone()
    counterfactual[..., channel:channel + 1] = changed_normalized_y
    roundtrip_raw_y = motion_rep.stats.unnormalize(
        counterfactual[..., channel:channel + 1], index=rep_index)
    expected_delta = torch.zeros_like(raw_y)
    expected_delta[:, start_frame:start_frame + frame_count] = ramp[None, :, None]
    max_ramp_error = float(((roundtrip_raw_y - raw_y) - expected_delta).abs().max().item())
    if max_ramp_error > 2e-5:
        raise RuntimeError(f"Stats roundtrip failed to apply the requested meter ramp: {max_ramp_error:g}")

    other_channels = [index for index in range(external_cond.shape[-1]) if index != channel]
    if other_channels and not torch.equal(counterfactual[..., other_channels], external_cond[..., other_channels]):
        raise RuntimeError("Height counterfactual changed non-height external channels")
    outside_window = torch.ones(external_cond.shape[1], dtype=torch.bool, device=external_cond.device)
    outside_window[start_frame:start_frame + frame_count] = False
    if not torch.equal(counterfactual[:, outside_window, channel], external_cond[:, outside_window, channel]):
        raise RuntimeError("Height counterfactual changed frames outside its requested window")

    return counterfactual, {
        "location": location,
        "start_frame": start_frame,
        "frame_indices": list(range(start_frame, start_frame + frame_count)),
        "expected_ramp_m": ramp.detach().cpu().tolist(),
        "reference_external_y_m": raw_y[0, :, 0].detach().cpu().tolist(),
        "roundtrip_max_abs_error_m": max_ramp_error,
    }


def decoded_root_y_m(recon_state, motion_rep, output_feature_mode):
    """Read decoder output root Y using its actual output feature layout and Stats."""
    if recon_state.ndim != 3 or not recon_state.is_floating_point():
        raise ValueError("Decoder reconstruction must be floating-point [B,T,F]")
    location = _require_height_location(motion_rep, output_feature_mode)
    channel = location["global_root_y_channel_index"]
    rep_index = np.asarray([location["global_root_y_rep_index"]], dtype=np.int64)
    normalized_y = recon_state[..., channel:channel + 1]
    raw_y = motion_rep.stats.unnormalize(normalized_y, index=rep_index)
    return raw_y[..., 0], location


def unavailable_planar_response(feature_mode, feature_width):
    return {
        "status": "unavailable_not_zero",
        "reason": "The planar codec external feature has no global_root_y channel; no height gain was measured.",
        "external_feature_mode": feature_mode,
        "external_feature_width": int(feature_width),
        "gain_mean": None,
        "gain_last": None,
    }


def _snapshot(value):
    return value.detach().clone() if torch.is_tensor(value) else value


@contextmanager
def capture_decoder_call(codec):
    """Observe the real forward_decoder call while returning its result unchanged."""
    original = codec.forward_decoder
    had_instance_override = "forward_decoder" in codec.__dict__
    previous_instance_value = codec.__dict__.get("forward_decoder")
    calls = []

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append({
            "args": tuple(_snapshot(value) for value in args),
            "kwargs": {key: _snapshot(value) for key, value in kwargs.items()},
            "recon_state": _snapshot(result["recon_state"]),
        })
        return result

    codec.forward_decoder = wrapped
    try:
        yield calls
    finally:
        if had_instance_override:
            codec.forward_decoder = previous_instance_value
        else:
            delattr(codec, "forward_decoder")


def _summary_decoder_call(call, codec, num_frames):
    args, kwargs = call["args"], call["kwargs"]
    if not args:
        raise ValueError("Captured decoder call omitted its token argument")
    tokens = args[0]
    required = {"target_cond", "has_target_cond", "external_cond", "use_overall_indices", "token_mask"}
    if required - kwargs.keys():
        raise ValueError(f"Captured decoder call is missing {sorted(required - kwargs.keys())}")

    local_rep = codec.motion_rep
    external = kwargs["external_cond"]
    external_mode = codec.decoder_external_cond_feature_mode
    y_location = height_feature_location(local_rep, external_mode)
    external_raw_y = None
    if y_location["available"]:
        rep_index = np.asarray([y_location["global_root_y_rep_index"]], dtype=np.int64)
        external_raw_y = local_rep.stats.unnormalize(
            external[..., y_location["global_root_y_channel_index"]:
                     y_location["global_root_y_channel_index"] + 1], index=rep_index)

    known = kwargs["has_target_cond"].detach().bool()
    target_cond = kwargs["target_cond"]
    target_rows = []
    for batch_index, frame_index in known.nonzero(as_tuple=False).detach().cpu().tolist():
        target_rows.append({
            "batch_index": int(batch_index),
            "frame_index": int(frame_index),
            "normalized_feature_values": target_cond[batch_index, frame_index].detach().cpu().tolist(),
        })
    token_mask = kwargs["token_mask"]
    return {
        "tokens": tokens.detach().cpu().tolist(),
        "tokens_shape": list(tokens.shape),
        "tokens_sha256": tensor_digest({"tokens": tokens}),
        "target_cond_shape": list(target_cond.shape),
        "target_cond_sha256": tensor_digest({"target_cond": target_cond}),
        "has_target_cond_shape": list(known.shape),
        "known_target_frame_indices": sorted(set(int(value) for value in known.nonzero()[:, 1].detach().cpu().tolist())),
        "known_target_count": int(known.sum().item()),
        "known_target_cond_normalized": target_rows,
        "has_target_cond_sha256": tensor_digest({"has_target_cond": known}),
        "external_cond_shape": list(external.shape),
        "external_cond_normalized": external[0].detach().cpu().tolist(),
        "external_cond_sha256": tensor_digest({"external_cond": external}),
        "external_feature_mode": external_mode,
        "external_root_y_location": y_location,
        "external_root_y_m": (external_raw_y[0, :num_frames, 0].detach().cpu().tolist()
                               if external_raw_y is not None else None),
        "token_mask": token_mask.detach().cpu().tolist(),
        "token_mask_sha256": tensor_digest({"token_mask": token_mask}),
        "use_overall_indices": bool(kwargs["use_overall_indices"]),
    }


def _decoder_gain_record(reference_recon, counterfactual_recon, codec, sign, ramp_info, num_frames):
    output_mode = codec.decoder_input_feature_mode
    reference_y, y_location = decoded_root_y_m(reference_recon, codec.motion_rep, output_mode)
    counterfactual_y, _ = decoded_root_y_m(counterfactual_recon, codec.motion_rep, output_mode)
    reference_y = reference_y[0, :num_frames].detach().cpu()
    counterfactual_y = counterfactual_y[0, :num_frames].detach().cpu()
    delta_y = counterfactual_y - reference_y
    expected = torch.zeros(num_frames, dtype=delta_y.dtype)
    ramp = torch.tensor(ramp_info["expected_ramp_m"], dtype=delta_y.dtype)
    start = ramp_info["start_frame"]
    stop = start + len(ramp)
    if stop > num_frames:
        raise ValueError("Height response window exceeds the generated token horizon")
    expected[start:stop] = ramp
    observed_window = delta_y[start:stop]
    ratios = observed_window / ramp
    return {
        "sign": "+" if sign > 0 else "-",
        "requested_endpoint_m": float(sign * RAMP_ENDPOINT_M),
        "decoded_output_feature_mode": output_mode,
        "decoded_root_y_location": y_location,
        "reference_decoded_root_y_m": reference_y.tolist(),
        "counterfactual_decoded_root_y_m": counterfactual_y.tolist(),
        "decoded_root_y_delta_vs_reference_m": delta_y.tolist(),
        "expected_external_y_ramp_m": expected.tolist(),
        "response_frame_indices": ramp_info["frame_indices"],
        "response_delta_y_m": observed_window.tolist(),
        "gain_mean": float(ratios.mean().item()),
        "gain_last": float((observed_window[-1] / ramp[-1]).item()),
        "mean_response_delta_m": float(observed_window.mean().item()),
        "target_height_roundtrip_max_abs_error_m": ramp_info["roundtrip_max_abs_error_m"],
    }


def _bundle_source(bundle_path, bundle, inference):
    metadata = bundle["metadata"]
    scene = bundle["scene"]
    stats = {"mean": bundle["stats_mean"], "std": bundle["stats_std"]}
    history = bundle["initial_history"]
    return {
        "bundle_path": str(Path(bundle_path).resolve()),
        "bundle_sha256": sha256_file(bundle_path),
        "dataset_manifest_sha256": metadata.get("dataset_manifest_sha256"),
        "root_checkpoint_sha256": metadata.get("root_checkpoint_sha256"),
        "pose_checkpoint_sha256": metadata.get("pose_checkpoint_sha256"),
        "tokenizer_checkpoint_sha256": metadata.get("tokenizer_checkpoint_sha256"),
        "root_state_sha256": state_digest(inference._root_model),
        "pose_backbone_state_sha256": state_digest(inference._pose_model.backbone_net),
        "tokenizer_state_sha256": state_digest(inference._vqvae_pose_model),
        "initial_history_tensor_sha256": tensor_digest({"initial_history": history}),
        "scene_tensor_sha256": tensor_digest(scene),
        "stats_tensor_sha256": tensor_digest(stats),
        "seed_provenance": bundle.get("seed_provenance", {}),
        "codec_height_adapter": metadata.get("codec_height_adapter"),
    }


@torch.inference_mode()
def run_bundle(bundle_path, device, *, adapted_target):
    inference, bundle = load_terrain_bundle(bundle_path, device=device)
    codec = inference._vqvae_pose_model
    pose_config_args = bundle["pose_config"]["model"]["args"]
    protocol = pose_config_args.get("codec_height_conditioning")
    if adapted_target:
        adapter = bundle["metadata"].get("codec_height_adapter") or {}
        if (protocol != CODEC_HEIGHT_CONDITIONING
                or adapter.get("protocol") != CODEC_HEIGHT_CONDITIONING
                or int(adapter.get("updates", -1)) != 2000
                or codec.decoder_external_cond_feature_mode != HEIGHT_EXTERNAL_FEATURE_MODE):
            raise ValueError("Target bundle is not the completed 2k dense-height codec")
        height_feature_location(codec.motion_rep, codec.decoder_external_cond_feature_mode)
    elif (protocol is not None
          or codec.decoder_external_cond_feature_mode != SOURCE_EXTERNAL_FEATURE_MODE):
        raise ValueError("Baseline bundle must use the planar codec without dense root height")

    device_obj = next(inference._pose_model.parameters()).device
    history = bundle["initial_history"].to(device=device_obj, dtype=torch.float32)
    if history.shape != (1, HISTORY_FRAMES, len(inference.global_motion_rep.indices["all"])):
        raise ValueError("Bundle must contain exactly four full-motion history frames")
    history_root = extract_feature_from_motion_rep(history, inference.global_motion_rep, "root")
    goal = planar_command_goal(history_root, (0.0, 1.6), 0.0, NUM_TOKENS)
    diagnostics = {}
    with capture_decoder_call(codec) as calls:
        predicted_motion, predicted_tokens = predict_planar_terrain(
            inference,
            history,
            goal,
            bundle["scene"],
            num_tokens=NUM_TOKENS,
            config=dict(INFERENCE_CONFIG),
            diagnostics=diagnostics,
        )
    token_count = int(predicted_tokens.reshape(-1)[0].item())
    if token_count != NUM_TOKENS:
        raise ValueError(f"Online inference returned {token_count} tokens, expected {NUM_TOKENS}")
    if len(calls) != 1:
        raise ValueError(f"Expected one actual decoder call, captured {len(calls)}")

    call = calls[0]
    decoder_kwargs = call["kwargs"]
    decoder_args = call["args"]
    reference_recon = call["recon_state"]
    frame_count = token_count * (2 ** codec._down_t)
    if frame_count != NUM_TOKENS * 4:
        raise ValueError("Expected four motion frames per generated token")
    source = _bundle_source(bundle_path, bundle, inference)
    call_record = _summary_decoder_call(call, codec, frame_count)
    output = {
        "source": source,
        "online_inference": {
            "intent_velocity_world_xz_mps": [0.0, 1.6],
            "heading_world_rad": 0.0,
            "planar_goal_world_xz_heading_cos_sin": goal[0].detach().cpu().tolist(),
            "num_tokens": token_count,
            "inference_config": dict(INFERENCE_CONFIG),
            "final_root_pred_mode": "from_pose_module (inference default; no override)",
            "runtime_output_modified": False,
            "predicted_world_motion_shape": list(predicted_motion.shape),
            "predicted_world_motion_sha256": tensor_digest({"predicted_world_motion": predicted_motion}),
            "online_diagnostics": diagnostics,
        },
        "decoder_call": call_record,
    }

    height_location = height_feature_location(codec.motion_rep, codec.decoder_external_cond_feature_mode)
    if not height_location["available"]:
        output["counterfactual_response"] = unavailable_planar_response(
            codec.decoder_external_cond_feature_mode, height_location["feature_width"])
    elif adapted_target:
        original_forward = codec.forward_decoder
        response_rows = []
        for sign in (1.0, -1.0):
            counterfactual_external, ramp_info = apply_height_ramp(
                decoder_kwargs["external_cond"], codec.motion_rep,
                codec.decoder_external_cond_feature_mode,
                endpoint_m=sign * RAMP_ENDPOINT_M,
            )
            counterfactual_kwargs = {
                key: _snapshot(value) for key, value in decoder_kwargs.items()
            }
            counterfactual_kwargs["external_cond"] = counterfactual_external
            for key, value in decoder_kwargs.items():
                if key != "external_cond" and torch.is_tensor(value):
                    if not torch.equal(counterfactual_kwargs[key], value):
                        raise RuntimeError(f"Offline replay changed captured decoder input {key}")
            counterfactual_result = original_forward(*decoder_args, **counterfactual_kwargs)
            response_rows.append(_decoder_gain_record(
                reference_recon,
                counterfactual_result["recon_state"],
                codec,
                sign,
                ramp_info,
                frame_count,
            ))
        output["counterfactual_response"] = {
            "status": "measured_offline_decoder_only",
            "baseline_reference": "the unmodified decoder output from the actual online inference call",
            "same_tokens_and_observed_keyframe": True,
            "only_external_global_root_y_changed": True,
            "runtime_output_modified": False,
            "replays": response_rows,
        }
    else:
        output["counterfactual_response"] = unavailable_planar_response(
            codec.decoder_external_cond_feature_mode, height_location["feature_width"])

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device_obj)
    del inference, bundle, codec
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return output


def _compare_bundle_inputs(target, baseline):
    target_source, baseline_source = target["source"], baseline["source"]
    comparisons = {
        "same_four_frame_history": target_source["initial_history_tensor_sha256"]
        == baseline_source["initial_history_tensor_sha256"],
        "same_scene": target_source["scene_tensor_sha256"] == baseline_source["scene_tensor_sha256"],
        "same_stats": target_source["stats_tensor_sha256"] == baseline_source["stats_tensor_sha256"],
        "same_root_state": target_source["root_state_sha256"] == baseline_source["root_state_sha256"],
        "same_pose_backbone_state": target_source["pose_backbone_state_sha256"]
        == baseline_source["pose_backbone_state_sha256"],
    }
    if not all(comparisons.values()):
        raise ValueError(f"Baseline bundle is not the matching planar source: {comparisons}")
    return comparisons


def review(target_bundle, output_path, *, device="cuda:0", baseline_bundle=None):
    target = run_bundle(target_bundle, device, adapted_target=True)
    baseline = None
    comparisons = None
    if baseline_bundle is not None:
        baseline = run_bundle(baseline_bundle, device, adapted_target=False)
        adapter = target["source"].get("codec_height_adapter") or {}
        if adapter.get("source_bundle_sha256") != baseline["source"]["bundle_sha256"]:
            raise ValueError("Baseline bundle SHA does not match the adapter's bound source bundle")
        comparisons = _compare_bundle_inputs(target, baseline)
    else:
        adapter = target["source"].get("codec_height_adapter") or {}

    report = {
        "schema": SCHEMA,
        "scope": "Offline decoder control-gain diagnostic; not physically validated motion or deployment acceptance",
        "quality_approved": False,
        "deployment_validated": False,
        "future_reference_used": False,
        "runtime_override_applied": False,
        "target_adapted_codec": target,
        "planar_baseline": ({
            "source": baseline["source"],
            "decoder_call": baseline["decoder_call"],
            "online_inference": baseline["online_inference"],
            "counterfactual_response": baseline["counterfactual_response"],
        } if baseline is not None else {
            "status": "not_loaded",
            "source_bundle_sha256_expected_by_adapter": adapter.get("source_bundle_sha256"),
            "external_feature_mode": SOURCE_EXTERNAL_FEATURE_MODE,
            "counterfactual_response": unavailable_planar_response(SOURCE_EXTERNAL_FEATURE_MODE, 2),
        }),
        "baseline_matches_target_inputs_and_models": comparisons,
    }
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ValueError("Refuse to overwrite an existing diagnostic report")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="Completed 2k codec-height portable bundle")
    parser.add_argument("--baseline-bundle", type=Path,
                        help="Optional bound 10k planar source bundle for structural baseline verification")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new report path")
    if args.threads < 1:
        parser.error("--threads must be positive")

    from scripts.train_native_mb import select_gpu

    gpu = select_gpu(args.gpu)
    torch.set_num_threads(args.threads)
    report = review(args.bundle, args.output, device="cuda:0", baseline_bundle=args.baseline_bundle)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "gpu_selection": gpu,
        "target_gain_status": report["target_adapted_codec"]["counterfactual_response"]["status"],
        "baseline_gain_status": report["planar_baseline"]["counterfactual_response"]["status"],
        "quality_approved": False,
        "runtime_override_applied": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
