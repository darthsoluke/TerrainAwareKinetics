"""Independently review a native MotionBricks tokenizer checkpoint."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from motionbricks.data.native_mb_evaluation import (
    full_precision_evaluation, verify_tokenizer_decode_parity,
)


FRAMES_PER_TOKEN = 4
NUM_TOKENS = 16
GLOBAL_WINDOW_FRAMES = NUM_TOKENS * FRAMES_PER_TOKEN + 1
SCORE_START = 4
SCORE_END = 63
CONDITIONS = ("none", "context4")
AVAILABLE_CONDITIONS = ("none", "context1", "context4")
CONDITION_FRAME_INDICES = {
    "none": (),
    # The native runtime marks frames 0-3 as observed context and masks slot 3
    # as the single Pose keyframe in the one-context stair diagnostic.
    "context1": (3,),
    "context4": (0, 1, 2, 3),
}


def _report_conditions(report):
    conditions = report.get("conditions", CONDITIONS)
    if (not isinstance(conditions, (list, tuple)) or not conditions
            or len(set(conditions)) != len(conditions)
            or any(condition not in AVAILABLE_CONDITIONS for condition in conditions)):
        raise ValueError("Report conditions must be a nonempty list of unique supported conditions")
    return tuple(conditions)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def resolve_checkpoint(run_dir, checkpoint=None):
    run_dir = Path(run_dir).resolve()
    candidate = Path(checkpoint) if checkpoint is not None else Path("final.ckpt")
    if not candidate.is_absolute():
        options = (run_dir / "checkpoints" / candidate, run_dir / candidate)
        candidate = next((path for path in options if path.is_file()), options[0])
    if not candidate.is_file():
        raise FileNotFoundError(f"Native checkpoint not found: {candidate}")
    return candidate.resolve()


def validate_run_contract(run_dir):
    """Validate the native run bindings without requiring a completed run."""
    run_dir = Path(run_dir).resolve()
    contract_path = run_dir / "run_contract.json"
    config_path = run_dir / "native_config.yaml"
    if not contract_path.is_file() or not config_path.is_file():
        raise ValueError("Run must contain run_contract.json and native_config.yaml")
    contract = _read_json(contract_path)
    if contract.get("stage") != "native_tokenizer":
        raise ValueError("Run contract stage must be native_tokenizer")
    dataset_value = contract.get("dataset")
    if not dataset_value:
        raise ValueError("Run contract is missing its dataset path")
    dataset_dir = Path(dataset_value).expanduser()
    manifest_path = dataset_dir / "native_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset native_manifest.json not found: {manifest_path}")
    actual_manifest_hash = sha256_file(manifest_path)
    if contract.get("dataset_manifest_sha256") != actual_manifest_hash:
        raise ValueError("Run contract dataset_manifest_sha256 does not match the dataset")
    if contract.get("config_sha256") != sha256_file(config_path):
        raise ValueError("Run contract config_sha256 does not match native_config.yaml")
    manifest = _read_json(manifest_path)
    artifact_hashes = manifest.get("artifact_sha256", {})
    required_artifacts = {"accepted.json", "stats/mean.npy", "stats/std.npy"}
    if not isinstance(artifact_hashes, dict) or not required_artifacts.issubset(artifact_hashes):
        raise ValueError("Dataset manifest does not bind accepted.json and normalization stats")
    for relative, expected in artifact_hashes.items():
        artifact = (dataset_dir / relative).resolve()
        if dataset_dir.resolve() not in artifact.parents or not artifact.is_file():
            raise ValueError(f"Dataset manifest artifact is missing or escapes the dataset: {relative}")
        if sha256_file(artifact) != expected:
            raise ValueError(f"Dataset manifest artifact hash mismatch: {relative}")
    accepted = _read_json(dataset_dir / "accepted.json")
    accepted_shards = {row["shard"] for row in accepted}
    missing_shards = accepted_shards - artifact_hashes.keys()
    if missing_shards:
        raise ValueError("Dataset manifest does not bind every accepted shard: "
                         + ", ".join(sorted(missing_shards)))
    return {
        "run_dir": run_dir,
        "contract": contract,
        "config_path": config_path,
        "dataset_dir": dataset_dir.resolve(),
        "manifest_path": manifest_path.resolve(),
        "dataset_manifest_sha256": actual_manifest_hash,
    }


def load_native_checkpoint(run_dir, checkpoint=None, device="cpu"):
    """Build the official tokenizer and strictly load a Lightning checkpoint."""
    import torch
    from omegaconf import OmegaConf

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    binding = validate_run_contract(run_dir)
    checkpoint_path = resolve_checkpoint(binding["run_dir"], checkpoint)
    # Lightning checkpoints contain optimizer and callback objects beyond the state dict.
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get("state_dict"), dict):
        raise ValueError("Checkpoint must contain a Lightning state_dict")
    global_step = saved.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("Checkpoint must contain a nonnegative integer global_step")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_path.name == "final.ckpt":
        result_path = binding["run_dir"] / "training_result.json"
        if not result_path.is_file():
            raise ValueError("final.ckpt requires training_result.json with its recorded hash")
        training_result = _read_json(result_path)
        if training_result.get("final_checkpoint_sha256") != checkpoint_hash:
            raise ValueError("final.ckpt hash does not match training_result.json")
        if training_result.get("optimizer_updates") != global_step:
            raise ValueError("final.ckpt global_step does not match training_result.json")
    else:
        checkpoints_dir = (binding["run_dir"] / "checkpoints").resolve()
        if checkpoints_dir not in checkpoint_path.parents:
            raise ValueError("Intermediate checkpoint must be inside run/checkpoints")
        contract = binding["contract"]
        stop_steps = contract.get("stop_steps")
        if isinstance(stop_steps, bool) or not isinstance(stop_steps, int) or stop_steps < 1:
            raise ValueError("Run contract must contain a positive stop_steps bound")
        resume = contract.get("resume") or {}
        saved_step = resume.get("saved_step", 0)
        if isinstance(saved_step, bool) or not isinstance(saved_step, int) or saved_step < 0:
            raise ValueError("Run contract resume.saved_step must be a nonnegative integer")
        if not saved_step <= global_step <= stop_steps:
            raise ValueError("Intermediate checkpoint global_step is outside the run bounds")

    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.data.native_mb_models import build_native_model

    config = OmegaConf.load(binding["config_path"])
    representation = build_native_representation(binding["dataset_dir"] / "stats")
    model = build_native_model("tokenizer", representation, config)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, {
        "run": str(binding["run_dir"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "global_step": global_step,
        "run_status_at_review": binding["contract"].get("status", "unspecified"),
        "dataset_manifest_sha256": binding["dataset_manifest_sha256"],
        "config_sha256": sha256_file(binding["config_path"]),
    }, binding


def root_relative_joints(joints, root_index):
    """Remove only per-frame root translation; preserve every orientation."""
    import torch

    _require_finite("posed joints", joints)
    return joints - joints[..., root_index:root_index + 1, :]


def _require_finite(name, tensor):
    import torch

    if not torch.isfinite(tensor).all():
        raise ValueError(f"Nonfinite {name}")


def reconstruction_metrics(predicted, target, root_index, fps, frame_start=SCORE_START,
                           frame_end=SCORE_END):
    """Compute root-relative FK, local-rotation, and interval velocity errors."""
    import torch

    for name, tensor in (("predicted joints", predicted["posed_joints"]),
                         ("target joints", target["posed_joints"]),
                         ("predicted rotations", predicted["local_joint_rots"]),
                         ("target rotations", target["local_joint_rots"])):
        _require_finite(name, tensor)
    joints_p = root_relative_joints(predicted["posed_joints"], root_index)
    joints_g = root_relative_joints(target["posed_joints"], root_index)
    world_p = predicted["posed_joints"][:, frame_start:frame_end + 1]
    world_g = target["posed_joints"][:, frame_start:frame_end + 1]
    relative_p = joints_p[:, frame_start:frame_end + 1]
    relative_g = joints_g[:, frame_start:frame_end + 1]
    if relative_p.shape[1] < 2:
        raise ValueError("Scored joint window must contain at least two frames")
    rotations_p = predicted["local_joint_rots"][:, frame_start:frame_end + 1]
    rotations_g = target["local_joint_rots"][:, frame_start:frame_end + 1]
    relative_rotation = rotations_p.transpose(-1, -2) @ rotations_g
    cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
    rotation_error = torch.acos(cosine.clamp(-1.0, 1.0))
    velocity_p = torch.diff(relative_p, dim=1) * fps
    velocity_g = torch.diff(relative_g, dim=1) * fps
    world_velocity_p = torch.diff(world_p, dim=1) * fps
    world_velocity_g = torch.diff(world_g, dim=1) * fps
    for name, tensor in (("derived rotation error", rotation_error),
                         ("derived joint positions", relative_p - relative_g),
                         ("derived joint velocity error", velocity_p - velocity_g),
                         ("derived world joint positions", world_p - world_g),
                         ("derived world joint velocity error", world_velocity_p - world_velocity_g)):
        _require_finite(name, tensor)
    return {
        "mpjpe_m": float((world_p - world_g).norm(dim=-1).mean()),
        "root_error_m": float((world_p[:, :, root_index] - world_g[:, :, root_index])
                               .norm(dim=-1).mean()),
        "joint_velocity_error_mps": float(
            (world_velocity_p - world_velocity_g).norm(dim=-1).mean()
        ),
        "root_relative_fk_mpjpe_m": float((relative_p - relative_g).norm(dim=-1).mean()),
        "local_joint_rotation_geodesic_rad": float(rotation_error.mean()),
        "root_relative_joint_velocity_error_mps": float(
            (velocity_p - velocity_g).norm(dim=-1).mean()
        ),
    }


def _fixed_windows(dataset_dir, split="validation"):
    if split not in {"train", "validation"}:
        raise ValueError("Review split must be train or validation; the test set stays untouched")
    records = [row for row in _read_json(dataset_dir / "accepted.json")
               if row.get("split") == split]
    windows = []
    excluded = []
    eligible = []
    import numpy as np

    for row in records:
        shard = row["shard"]
        shard_path = (dataset_dir / shard).resolve()
        if dataset_dir.resolve() not in shard_path.parents:
            raise ValueError(f"Review shard escapes dataset directory: {shard}")
        with np.load(shard_path, allow_pickle=False) as data:
            frame_count = len(data["global_features"])
        if frame_count < GLOBAL_WINDOW_FRAMES:
            excluded.append({"shard": shard, "frames": frame_count,
                             "reason": "shorter than required global evaluation window"})
            continue
        starts = sorted(set(np.linspace(
            0, frame_count - GLOBAL_WINDOW_FRAMES, 3
        ).astype(int).tolist()))
        eligible.append((row, shard_path, starts))

    expected_validation_windows = sum(
        len(starts) for _, _, starts in eligible
    ) if split == "validation" else None
    for row, shard_path, starts in eligible:
        for start in starts:
            windows.append({
                "shard": row["shard"],
                "shard_sha256": sha256_file(shard_path),
                "behavior": row.get("behavior", "unknown"),
                "start": start,
            })
    if split == "validation" and len(windows) != expected_validation_windows:
        raise ValueError("Validation window count does not match eligible accepted shards: "
                         f"expected {expected_validation_windows}, found {len(windows)}")
    if not windows:
        raise ValueError(f"No eligible {split} windows")
    return windows, excluded


def _window_signature(manifest_hash, windows, conditions=CONDITIONS):
    return _canonical_sha256({
        "dataset_manifest_sha256": manifest_hash,
        "global_window_frames": GLOBAL_WINDOW_FRAMES,
        "score_frames_inclusive": [SCORE_START, SCORE_END],
        "conditions": list(conditions),
        "windows": [{key: row[key] for key in
                     ("shard", "shard_sha256", "behavior", "start")} for row in windows],
    })


def _array_identity(array):
    import numpy as np

    if hasattr(array, "detach") and hasattr(array, "cpu"):
        array = array.detach().cpu().numpy()
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _source_skeleton_identity(skeleton):
    names = getattr(skeleton, "bone_order_names", None)
    parents = getattr(skeleton, "joint_parents", None)
    neutral = getattr(skeleton, "neutral_joints", None)
    root_index = getattr(skeleton, "root_idx", None)
    if names is None or parents is None or neutral is None:
        raise ValueError("Physical comparison requires ordered skeleton names, parents, and neutral geometry")
    if isinstance(root_index, bool) or not isinstance(root_index, int):
        raise ValueError("Physical comparison requires a valid skeleton root index")
    return {
        "joint_names": [str(name) for name in names],
        "joint_parents": _array_identity(parents),
        "neutral_geometry": _array_identity(neutral),
        "root_index": root_index,
    }


def _source_window_identity(window):
    fields = ("shard", "start", "behavior", "conditioning", "source_motion_sha256",
              "source_motion_shape", "source_motion_dtype")
    identity = {field: window.get(field) for field in fields}
    shape = identity["source_motion_shape"]
    digest = identity["source_motion_sha256"]
    if (not isinstance(identity["shard"], str) or not identity["shard"].strip()
            or isinstance(identity["start"], bool) or not isinstance(identity["start"], int)
            or not isinstance(identity["behavior"], str) or not identity["behavior"].strip()
            or identity["conditioning"] not in AVAILABLE_CONDITIONS
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not isinstance(shape, list) or not shape
            or any(isinstance(size, bool) or not isinstance(size, int) or size < 1
                   for size in shape)
            or not isinstance(identity["source_motion_dtype"], str)
            or not identity["source_motion_dtype"].strip()):
        raise ValueError("Physical comparison refused: missing or malformed source motion identity")
    return identity


def _validate_skeleton_identity(skeleton):
    if (not isinstance(skeleton.get("joint_names"), list) or not skeleton["joint_names"]
            or any(not isinstance(name, str) or not name for name in skeleton["joint_names"])
            or isinstance(skeleton.get("root_index"), bool)
            or not isinstance(skeleton.get("root_index"), int)
            or not 0 <= skeleton["root_index"] < len(skeleton["joint_names"])):
        return False
    for key in ("joint_parents", "neutral_geometry"):
        array_identity = skeleton.get(key)
        if not isinstance(array_identity, dict):
            return False
        shape, dtype, digest = (array_identity.get(name) for name in ("shape", "dtype", "sha256"))
        if (not isinstance(shape, list) or not shape
                or any(isinstance(size, bool) or not isinstance(size, int) or size < 1
                       for size in shape)
                or not isinstance(dtype, str) or not dtype.strip()
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            return False
        expected_shape = [len(skeleton["joint_names"])]
        if key == "neutral_geometry":
            expected_shape.append(3)
        if shape != expected_shape:
            return False
    return True


def _source_motion_signature(skeleton, fps, windows, conditions=CONDITIONS):
    return _canonical_sha256({
        "skeleton": skeleton,
        "fps": float(fps),
        "global_window_frames": GLOBAL_WINDOW_FRAMES,
        "score_frames_inclusive": [SCORE_START, SCORE_END],
        "conditions": list(conditions),
        "windows": [_source_window_identity(row) for row in windows],
    })


PHYSICAL_METRICS = (
    "mpjpe_m", "root_error_m", "joint_velocity_error_mps",
    "root_relative_fk_mpjpe_m", "local_joint_rotation_geodesic_rad",
    "root_relative_joint_velocity_error_mps",
)


def compare_physical_baseline(current, baseline_path):
    """Compare physical metrics only when the raw source motions are identical."""
    baseline = _read_json(baseline_path)
    if current.get("dataset_split", "validation") != baseline.get("dataset_split", "validation"):
        raise ValueError("Physical baseline comparison refused: dataset split mismatch")
    report_conditions = None
    for report in (current, baseline):
        conditions = _report_conditions(report)
        if report_conditions is None:
            report_conditions = conditions
        elif conditions != report_conditions:
            raise ValueError("Physical baseline comparison refused: source signature metadata "
                             "is missing or malformed (condition mismatch)")
        signature = report.get("source_motion_signature")
        if not isinstance(signature, str) or not signature.strip():
            raise ValueError("Physical baseline comparison refused: source_motion_signature is missing; reevaluate report")
        windows = report.get("windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError("Physical baseline comparison refused: source motion windows are missing")
        for window in windows:
            _source_window_identity(window)
        skeleton = report.get("source_skeleton_identity")
        fps = report.get("fps")
        if (not isinstance(skeleton, dict) or not _validate_skeleton_identity(skeleton)
                or isinstance(fps, bool) or not isinstance(fps, (int, float))
                or not math.isfinite(fps) or fps <= 0
                or report.get("frame_scoring_inclusive") != [SCORE_START, SCORE_END]
                or report.get("interval_scoring_endpoint_frames_inclusive") != [SCORE_START, SCORE_END]
                or report.get("conditions", list(CONDITIONS)) != list(conditions)):
            raise ValueError("Physical baseline comparison refused: source signature metadata is missing or malformed")
        if _source_motion_signature(skeleton, fps, windows, conditions) != signature:
            raise ValueError("Physical baseline comparison refused: source_motion_signature does not match report identities")
    if current["source_motion_signature"] != baseline["source_motion_signature"]:
        raise ValueError("Physical baseline comparison refused: source_motion_signature mismatch")
    current_keys = [_source_window_identity(row) for row in current["windows"]]
    baseline_keys = [_source_window_identity(row) for row in baseline["windows"]]
    if current_keys != baseline_keys:
        raise ValueError("Physical baseline comparison refused: ordered source motion windows mismatch")

    deltas = {}
    for condition in report_conditions:
        selected_current = [row for row in current["windows"] if row["conditioning"] == condition]
        selected_baseline = [row for row in baseline["windows"] if row["conditioning"] == condition]
        if len(selected_current) != len(selected_baseline) or not selected_current:
            raise ValueError("Physical baseline comparison refused: condition/window mismatch")
        deltas[condition] = {}
        for metric in PHYSICAL_METRICS:
            values = [row.get(metric) for row in selected_current]
            prior = [row.get(metric) for row in selected_baseline]
            if any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in values + prior):
                raise ValueError(f"Physical baseline comparison refused: missing or nonfinite {metric}")
            current_mean = sum(values) / len(values)
            baseline_mean = sum(prior) / len(prior)
            deltas[condition][metric] = {
                "current_mean": current_mean,
                "baseline_mean": baseline_mean,
                "delta": current_mean - baseline_mean,
            }
    return {
        "baseline": str(Path(baseline_path).resolve()),
        "comparison_scope": "physical_metrics_only",
        "excluded_metrics": {
            "normalized_smooth_l1": "excluded because normalization statistics can differ across training datasets",
        },
        "deltas": deltas,
    }


def compare_baseline(current, baseline_path):
    baseline = _read_json(baseline_path)
    if current.get("dataset_split", "validation") != baseline.get("dataset_split", "validation"):
        raise ValueError("Baseline comparison refused: dataset split mismatch")
    conditions = _report_conditions(current)
    if _report_conditions(baseline) != conditions:
        raise ValueError("Baseline comparison refused: condition mismatch")
    for key in ("dataset_manifest_sha256", "input_signature"):
        current_value, baseline_value = current.get(key), baseline.get(key)
        if (not isinstance(current_value, str) or not current_value.strip()
                or not isinstance(baseline_value, str) or not baseline_value.strip()):
            raise ValueError(f"Baseline comparison refused: {key} must be a nonempty string")
        if current_value != baseline_value:
            raise ValueError(f"Baseline comparison refused: {key} mismatch or missing")
    current_windows = current.get("windows")
    baseline_windows = baseline.get("windows")
    if not isinstance(current_windows, list) or not isinstance(baseline_windows, list):
        raise ValueError("Baseline comparison refused: validation windows are missing")
    identity_fields = ("shard", "shard_sha256", "behavior", "start", "conditioning")
    for row in current_windows + baseline_windows:
        if (not isinstance(row.get("shard"), str) or not row["shard"].strip()
                or not isinstance(row.get("shard_sha256"), str) or not row["shard_sha256"].strip()
                or not isinstance(row.get("behavior"), str) or not row["behavior"].strip()
                or isinstance(row.get("start"), bool) or not isinstance(row.get("start"), int)
                or row.get("conditioning") not in AVAILABLE_CONDITIONS):
            raise ValueError("Baseline comparison refused: malformed window identity")
    current_keys = [tuple(row[field] for field in identity_fields) for row in current_windows]
    baseline_keys = [tuple(row[field] for field in identity_fields) for row in baseline_windows]
    if current_keys != baseline_keys:
        raise ValueError("Baseline comparison refused: ordered validation windows mismatch")
    metrics = ("mpjpe_m", "root_error_m", "joint_velocity_error_mps",
               "root_relative_fk_mpjpe_m", "local_joint_rotation_geodesic_rad",
               "root_relative_joint_velocity_error_mps", "normalized_smooth_l1")
    deltas = {}
    for condition in conditions:
        deltas[condition] = {}
        for metric in metrics:
            values = [row[metric] for row in current_windows
                      if row["conditioning"] == condition and metric in row]
            prior = [row[metric] for row in baseline_windows
                     if row["conditioning"] == condition and metric in row]
            if len(values) != len(prior) or not values:
                raise ValueError(f"Baseline comparison refused: missing {metric} or condition/window mismatch")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in values + prior):
                raise ValueError(f"Baseline comparison refused: nonfinite {metric}")
            deltas[condition][metric] = {
                "current_mean": sum(values) / len(values),
                "baseline_mean": sum(prior) / len(prior),
                "delta": sum(values) / len(values) - sum(prior) / len(prior),
            }
    return {"baseline": str(Path(baseline_path).resolve()), "deltas": deltas}


@full_precision_evaluation()
def evaluate_tokenizer(model, dataset_dir, manifest_hash, output, split="validation",
                       conditions=CONDITIONS):
    import numpy as np
    import torch
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    conditions = tuple(conditions)
    if (not conditions or len(set(conditions)) != len(conditions)
            or any(condition not in AVAILABLE_CONDITIONS for condition in conditions)):
        raise ValueError("conditions must be a nonempty sequence of unique supported conditions")
    device = next(model.parameters()).device
    rep, local_rep = model.motion_rep, model.local_motion_rep
    network = model.pose_net
    local_frames = model.args["max_tokens"] * model.get_num_frames_per_code()
    if local_frames != GLOBAL_WINDOW_FRAMES - 1:
        raise ValueError(f"Native config requires 64 local frames; model has {local_frames}")
    windows, excluded = _fixed_windows(dataset_dir, split)
    input_signature = _window_signature(manifest_hash, windows, conditions)
    rows = []
    source_skeleton = _source_skeleton_identity(rep.skeleton)
    code_counts = torch.zeros(network.quantizer.num_heads,
                              network.quantizer.nb_code_per_head,
                              dtype=torch.long, device=device)
    max_decode_error = 0.0
    roundoff_replays = []
    with torch.inference_mode():
        for window in windows:
            with np.load(dataset_dir / window["shard"], allow_pickle=False) as data:
                raw_features = data["global_features"][
                    window["start"]:window["start"] + GLOBAL_WINDOW_FRAMES
                ]
                raw_identity = _array_identity(raw_features)
                raw = torch.from_numpy(raw_features.copy()).to(device)[None]
            raw = rep.change_first_heading(raw, 0.0, False, False)
            global_motion = rep.normalize(raw)
            local = rep.dual_rep.global_to_local(
                global_motion, True, True,
                lengths=torch.tensor([GLOBAL_WINDOW_FRAMES], device=device),
            )[:, :local_frames]
            external = extract_feature_from_motion_rep(
                local, local_rep, network.decoder_external_cond_feature_mode
            )
            target = local_rep.inverse(local, True, return_all=True)
            indices = network.encode_into_idx(local, fetch_overall_indices=False)
            for head in range(len(code_counts)):
                code_counts[head] += torch.bincount(
                    indices[..., head].flatten(), minlength=code_counts.shape[1]
                )
            for condition in conditions:
                mask = torch.zeros(local.shape[:2], dtype=torch.bool, device=device)
                if CONDITION_FRAME_INDICES[condition]:
                    mask[:, list(CONDITION_FRAME_INDICES[condition])] = True
                conditioned_target = local * mask[..., None] if mask.any() else None
                conditioned_mask = mask if mask.any() else None
                prediction = network(
                    local, target_cond=conditioned_target,
                    has_target_cond=conditioned_mask, external_cond=external,
                )["recon_state"]
                decoded = network.forward_decoder(
                    indices, target_cond=conditioned_target,
                    has_target_cond=conditioned_mask, external_cond=external,
                    use_overall_indices=False,
                )["recon_state"]
                if not torch.isfinite(prediction).all() or not torch.isfinite(decoded).all():
                    raise ValueError("Nonfinite tokenizer prediction or token decode")
                decode_error = float((prediction - decoded).abs().max())
                max_decode_error = max(max_decode_error, decode_error)
                if decode_error > 1e-5:
                    try:
                        parity = verify_tokenizer_decode_parity(
                            network, local, indices, prediction, decoded,
                            target_cond=conditioned_target, has_target_cond=conditioned_mask,
                            external_cond=external,
                        )
                    except (ValueError, AttributeError) as error:
                        raise ValueError(f"Token-index decoder parity failure: max_error={decode_error:.9g}, "
                                         f"window={window}, condition={condition}: {error}") from error
                    roundoff_replays.append({**window, 'conditioning': condition, **parity})
                reconstructed = local_rep.inverse(prediction, True, return_all=True)
                metrics = reconstruction_metrics(
                    reconstructed, target, rep.skeleton.root_idx, rep.fps
                )
                metrics["normalized_smooth_l1"] = float(torch.nn.functional.smooth_l1_loss(
                    prediction[:, SCORE_START:SCORE_END + 1],
                    local[:, SCORE_START:SCORE_END + 1],
                ))
                rows.append({
                    **window,
                    "conditioning": condition,
                    "source_motion_sha256": raw_identity["sha256"],
                    "source_motion_shape": raw_identity["shape"],
                    "source_motion_dtype": raw_identity["dtype"],
                    **metrics,
                })
    source_motion_signature = _source_motion_signature(
        source_skeleton, rep.fps, rows, conditions
    )
    metric_names = ("mpjpe_m", "root_error_m", "joint_velocity_error_mps",
                    "root_relative_fk_mpjpe_m", "local_joint_rotation_geodesic_rad",
                    "root_relative_joint_velocity_error_mps", "normalized_smooth_l1")
    summary = {}
    for condition in conditions:
        selected = [row for row in rows if row["conditioning"] == condition]
        summary[condition] = {
            name: sum(row[name] for row in selected) / len(selected) for name in metric_names
        }
        summary[condition]["windows"] = len(selected)
    result = {
        "task": "source-encoded original native tokenizer reconstruction; not generation",
        "quality_approved": False,
        "fps": rep.fps,
        "frame_scoring_inclusive": [SCORE_START, SCORE_END],
        "interval_scoring_endpoint_frames_inclusive": [SCORE_START, SCORE_END],
        "root_alignment": "subtract each frame's root joint translation only; no rotation or Procrustes alignment",
        "conditions": list(conditions),
        "input_conditions": {
            condition: {
                "pose_context_frame_indices": list(CONDITION_FRAME_INDICES[condition]),
                "source_encoded_pose_values": True,
            }
            for condition in conditions
        },
        "dataset_manifest_sha256": manifest_hash,
        "dataset_split": split,
        "input_signature": input_signature,
        "source_skeleton_identity": source_skeleton,
        "source_motion_signature": source_motion_signature,
        "evaluation_source_sha256": sha256_file(Path(__file__).resolve()),
        "precision_policy": {
            "name": "full_precision_evaluation",
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "token_decode_max_feature_error": max_decode_error,
        "token_decode_roundoff_replays": roundoff_replays,
        "active_codes_per_head": (code_counts > 0).sum(-1).cpu().tolist(),
        "contact_diagnostics": {"status": "unavailable_not_assessed",
                                "reason": "No physical contact threshold or contact labels are assessed here"},
        "summary": summary,
        "windows": rows,
        "excluded_clips": excluded,
        "excluded_validation_clips": excluded if split == "validation" else [],
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Native tokenizer run directory containing its run contract")
    parser.add_argument("--checkpoint", type=Path,
                        help="Checkpoint path or filename relative to the run (default final.ckpt)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--split", choices=("train", "validation"), default="validation",
                        help="Train is a fitting diagnostic, never a held-out quality result")
    parser.add_argument("--conditions", nargs="+", choices=AVAILABLE_CONDITIONS,
                        default=list(CONDITIONS),
                        help="Pose input conditions to evaluate (default: none context4)")
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument("--baseline", type=Path,
                                help="Prior metrics.json for strict same-dataset comparison")
    baseline_group.add_argument("--physical-baseline", type=Path,
                                help="Prior report for exact raw-motion physical-metric comparison")
    args = parser.parse_args(argv)
    model, checkpoint_info, binding = load_native_checkpoint(
        args.run, args.checkpoint, args.device
    )
    result = evaluate_tokenizer(
        model, binding["dataset_dir"], binding["dataset_manifest_sha256"], args.output,
        split=args.split, conditions=args.conditions,
    )
    result["checkpoint"] = checkpoint_info
    if args.baseline:
        result["baseline_comparison"] = compare_baseline(result, args.baseline)
    if args.physical_baseline:
        result["physical_baseline_comparison"] = compare_physical_baseline(
            result, args.physical_baseline
        )
    with (args.output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"checkpoint": checkpoint_info, "summary": result["summary"],
                      "baseline_comparison": result.get("baseline_comparison"),
                      "physical_baseline_comparison": result.get("physical_baseline_comparison")},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
