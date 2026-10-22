"""Bounded Root-only height forecast comparison for online PARC checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.local_terrain_observation import extract_local_terrain_observation
from motionbricks.data.native_mb_online_conditions import (
    ONLINE_PROTOCOL,
    goal_direction_local,
    sparse_planar_conditions,
)
from motionbricks.data.native_mb_terrain_protocol import terrain_observation_config
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.train_native_mb_control import load_control_checkpoint, state_digest


BUNDLE_SCHEMA = "motionbricks.parc_online_soma_bundle.v1"
CONTEXT_FRAMES = 4
NUM_TOKENS = 8
POSE_EXTERNAL_FEATURE = "joint_positions_and_rotations"
POSE_INTERNAL_FEATURE = "joint_positions_and_rotations_and_hip_height"
NORMALIZATION_EPS = 1e-5
SCENARIOS = (
    ("idle", (0.0, 0.0)),
    ("forward_1_6m", (0.0, 1.6)),
    ("backward_1_0m", (0.0, -1.0)),
    ("left_0_8m", (-0.8, 0.0)),
    ("right_0_8m", (0.8, 0.0)),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(values):
    """Hash named tensors independent of device, matching control state digests."""
    digest = hashlib.sha256()
    for name in sorted(values):
        value = torch.as_tensor(values[name]).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def stats_digest(mean, std):
    holder = torch.nn.Module()
    holder.register_buffer("mean", torch.as_tensor(mean).detach().cpu())
    holder.register_buffer("std", torch.as_tensor(std).detach().cpu())
    return state_digest(holder)


def load_input_bundle(bundle_path):
    """Read only the portable bundle's four-frame seed, scene, stats and provenance."""
    bundle_path = Path(bundle_path).resolve()
    bundle_sha256 = sha256_file(bundle_path)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if (bundle.get("schema") != BUNDLE_SCHEMA or bundle.get("protocol") != ONLINE_PROTOCOL
            or bundle.get("quality_approved") is not False or bundle.get("post_processing") is not None
            or "reference_clips" in bundle):
        raise ValueError("Expected an experimental online bundle without future references")

    metadata = bundle.get("metadata") or {}
    dependency = metadata.get("pose_root_dependency") or {}
    if (metadata.get("parc_only") is not True
            or metadata.get("root_checkpoint_sha256") != dependency.get("checkpoint_sha256")
            or dependency.get("frozen") is not True
            or dependency.get("state_digest") != dependency.get("end_state_digest")):
        raise ValueError("Portable bundle Root provenance is incomplete or inconsistent")

    history = torch.as_tensor(bundle.get("initial_history")).detach().cpu().clone()
    if (history.ndim != 3 or history.shape[:2] != (1, CONTEXT_FRAMES)
            or not history.is_floating_point() or not torch.isfinite(history).all()):
        raise ValueError("Portable seed must contain exactly four finite observed frames")
    scene = bundle.get("scene")
    required_scene_keys = {"heightfield_y_m", "min_point_world_xyz_m", "dx_m"}
    if not isinstance(scene, dict) or set(scene) != required_scene_keys:
        raise ValueError("Portable scene contains missing or undeclared inputs")
    scene = {key: torch.as_tensor(scene[key]).detach().cpu().clone()
             for key in required_scene_keys}
    heights, origin, spacing = (scene[key] for key in
                                ("heightfield_y_m", "min_point_world_xyz_m", "dx_m"))
    if (heights.ndim != 2 or min(heights.shape) < 2 or origin.shape != (3,)
            or spacing.numel() != 1 or float(spacing) <= 0
            or not all(torch.isfinite(value).all() for value in (heights, origin, spacing))):
        raise ValueError("Portable scene geometry is invalid")

    mean = torch.as_tensor(bundle.get("stats_mean")).detach().cpu().clone()
    std = torch.as_tensor(bundle.get("stats_std")).detach().cpu().clone()
    recorded_stats_digest = bundle.get("stats_tensor_sha256")
    if (mean.ndim != 1 or std.shape != mean.shape or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or stats_digest(mean, std) != recorded_stats_digest):
        raise ValueError("Portable normalization statistics are invalid or changed")

    provenance = bundle.get("seed_provenance") or {}
    inputs = {
        "bundle_path": str(bundle_path),
        "bundle_sha256": bundle_sha256,
        "schema": bundle["schema"],
        "protocol": bundle["protocol"],
        "metadata": {
            "root_checkpoint_sha256": metadata.get("root_checkpoint_sha256"),
            "dataset_manifest_sha256": metadata.get("dataset_manifest_sha256"),
            "pose_root_dependency": dependency,
        },
        "seed_provenance": {
            key: provenance[key] for key in
            ("split", "shard", "filename", "start", "frames", "role") if key in provenance
        },
        "history": history,
        "scene": scene,
        "stats_mean": mean,
        "stats_std": std,
        "stats_tensor_sha256": recorded_stats_digest,
        "history_tensor_sha256": tensor_digest({"initial_history": history}),
        "scene_tensor_sha256": tensor_digest(scene),
    }
    del bundle
    return inputs


def _verify_checkpoint_stats(model, bundle):
    full_stats = model.motion_rep.dual_rep.stats
    mean = full_stats.mean.detach().cpu()
    std = full_stats.std.detach().cpu()
    if (not torch.equal(mean, bundle["stats_mean"])
            or not torch.equal(std, bundle["stats_std"])):
        raise ValueError("Portable stats do not exactly match this Root run's dataset statistics")
    return stats_digest(mean, std)


def _load_root_run(run_path, bundle, expected_updates, device):
    model, contract, result = load_control_checkpoint(run_path, device=device)
    if contract.get("stage") != "native_root":
        raise ValueError("Both comparison runs must be native Root checkpoints")
    if result.get("optimizer_updates") != expected_updates:
        raise ValueError(f"Expected a {expected_updates}-update Root checkpoint")
    if (contract.get("dataset_manifest_sha256")
            != bundle["metadata"].get("dataset_manifest_sha256")):
        raise ValueError("Root run and portable seed do not share the same dataset manifest")
    if (model.args.get("online_conditioning") != ONLINE_PROTOCOL["version"]
            or model.backbone_net.args.get("online_conditioning") != ONLINE_PROTOCOL["version"]
            or not model.backbone_net.args.get("terrain_conditioning", False)
            or not model.backbone_net.args.get("root_component_conditioning", False)):
        raise ValueError("Root checkpoint does not implement the online terrain/component contract")
    if model.training:
        raise ValueError("Root checkpoint must be in evaluation mode")
    stats_hash = _verify_checkpoint_stats(model, bundle)

    run = Path(run_path).resolve()
    dataset = Path(contract["dataset"]).resolve()
    checkpoint_path = run / "checkpoints" / "final.ckpt"
    if sha256_file(checkpoint_path) != result["final_checkpoint_sha256"]:
        raise ValueError("Root checkpoint changed after load_control_checkpoint verification")
    source = {
        "run_path": str(run),
        "stage": contract["stage"],
        "status": contract["status"],
        "optimizer_updates": result["optimizer_updates"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": result["final_checkpoint_sha256"],
        "run_contract_sha256": sha256_file(run / "run_contract.json"),
        "training_result_sha256": sha256_file(run / "training_result.json"),
        "native_config_sha256": sha256_file(run / "native_config.yaml"),
        "dataset_manifest_sha256": contract["dataset_manifest_sha256"],
        "dataset_manifest_file_sha256": sha256_file(dataset / "native_manifest.json"),
        "stats_mean_file_sha256": sha256_file(dataset / "stats" / "mean.npy"),
        "stats_std_file_sha256": sha256_file(dataset / "stats" / "std.npy"),
        "stats_tensor_sha256": stats_hash,
        "backbone_state_sha256": state_digest(model.backbone_net),
        "portable_stats_exact_match": True,
    }
    return model, source


def _planar_goal(history_root, local_displacement_xz):
    current = history_root[:, -1]
    local = torch.as_tensor(local_displacement_xz, device=current.device, dtype=current.dtype).reshape(1, 2)
    yaw = torch.atan2(current[:, 4], current[:, 3])
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    world_x = cosine * local[:, 0] + sine * local[:, 1]
    world_z = -sine * local[:, 0] + cosine * local[:, 1]
    endpoint = current[:, [0, 2]] + torch.stack((world_x, world_z), dim=-1)
    return torch.cat((endpoint, current[:, [3, 4]]), dim=-1)


def prepare_root_inputs(model, history, local_displacement_xz):
    """Mirror motion_inference.predict normalization without constructing Pose."""
    device = next(model.parameters()).device
    history = history.to(device=device, dtype=torch.float32)
    expected_dim = len(model.global_motion_rep.indices["all"])
    if history.shape != (1, CONTEXT_FRAMES, expected_dim) or not torch.isfinite(history).all():
        raise ValueError("Root review requires exactly four finite raw world-history frames")

    global_rep, local_rep = model.global_motion_rep, model.local_motion_rep
    history_root = extract_feature_from_motion_rep(history, global_rep, "root")
    local_history = model.motion_rep.dual_rep.global_to_local(
        history, is_normalized=False, to_normalize=False,
        lengths=torch.tensor([CONTEXT_FRAMES], device=device),
    )
    history_velocity = extract_feature_from_motion_rep(local_history, local_rep, "root")
    history_pose = extract_feature_from_motion_rep(local_history, local_rep, POSE_EXTERNAL_FEATURE)
    goal = _planar_goal(history_root, local_displacement_xz)
    conditions = sparse_planar_conditions(history_root, history_velocity, history_pose, goal)

    component_mask = conditions["global_root_component_mask"]
    global_root = torch.where(component_mask, conditions["global_root_values"], 0.0)
    xz_indices = torch.as_tensor(global_rep.indices["global_root_pos_2d"], device=device)
    initial_xz = global_root[:, :1, xz_indices].clone()
    recentered = global_root.clone()
    recentered[..., xz_indices] -= initial_xz
    normalized_global = global_rep.normalize(recentered)
    normalized_global = torch.where(component_mask, normalized_global, 0.0)

    local_mask = conditions["has_local_root_values"]
    local_root = torch.where(local_mask[..., None], conditions["local_root_values"], 0.0)
    normalized_local = local_rep.normalize(local_root)
    normalized_local = torch.where(local_mask[..., None], normalized_local, 0.0)

    pose_mask = conditions["has_local_poses"]
    external_pose = torch.where(pose_mask[..., None], conditions["local_poses"], 0.0)
    y_index = int(global_rep.indices["global_root_pos"][1])
    recentered_height = recentered[..., y_index:y_index + 1]
    internal_pose = torch.cat((recentered_height, external_pose), dim=-1)
    pose_indices = torch.as_tensor(extract_feature_from_motion_rep(
        torch.zeros((1, 1, len(local_rep.indices["all"])), device=device),
        local_rep, POSE_INTERNAL_FEATURE, fetch_feat_idx=True,
    ), dtype=torch.long, device=device)
    mean = local_rep.stats.mean.to(device=device, dtype=internal_pose.dtype)[pose_indices]
    std = local_rep.stats.std.to(device=device, dtype=internal_pose.dtype)[pose_indices]
    normalized_pose = (internal_pose - mean[None, None]) / torch.sqrt(std[None, None] ** 2 + NORMALIZATION_EPS)
    normalized_pose = torch.where(pose_mask[..., None], normalized_pose, 0.0)

    inputs = {
        "global_root_values": normalized_global,
        "has_global_root_values": conditions["has_global_root_values"],
        "global_root_component_mask": component_mask,
        "local_root_values": normalized_local,
        "has_local_root_values": local_mask,
        "poses": normalized_pose,
        "has_poses": pose_mask,
        "num_tokens": torch.full((1, 1), NUM_TOKENS, dtype=torch.long, device=device),
    }
    return {
        "inputs": inputs,
        "history_root": history_root,
        "goal": goal,
        "initial_xz": initial_xz,
        "input_sha256": tensor_digest(inputs),
    }


def build_current_terrain_observation(history_root, goal, scene):
    device = history_root.device
    direction = goal_direction_local(history_root, goal)
    observation = extract_local_terrain_observation(
        scene["heightfield_y_m"].to(device=device, dtype=torch.float32),
        scene["min_point_world_xyz_m"].to(device=device, dtype=torch.float32),
        scene["dx_m"].to(device=device, dtype=torch.float32),
        history_root[:, -1, :3],
        torch.atan2(history_root[:, -1, 4], history_root[:, -1, 3]),
        direction,
        config=terrain_observation_config(),
    )
    return observation, direction


def _terrain_observation_record(observation):
    points = observation.points.detach().cpu()
    valid = observation.valid.detach().cpu()
    world_xz = observation.world_query_xz.detach().cpu()
    return {
        "points_relative_xyz_m_fed_to_root": points[0].tolist(),
        "valid_mask_fed_to_root": valid[0].tolist(),
        "query_world_xz_m_diagnostic": world_xz[0].tolist(),
        "valid_cell_count": int(valid.sum().item()),
        "tensor_sha256": tensor_digest({"points": points, "valid": valid}),
        "observation_time": "last observed history frame only",
    }


def _terrain_at_predicted_roots(scene, predicted_root):
    heightfield = scene["heightfield_y_m"].detach().cpu().to(torch.float32)
    origin = scene["min_point_world_xyz_m"].detach().cpu().to(torch.float32)
    spacing = scene["dx_m"].detach().cpu().to(torch.float32)
    world_xz = predicted_root[CONTEXT_FRAMES:, [0, 2]].to(torch.float32)
    cells = torch.floor((world_xz - origin[[0, 2]]) / spacing + 0.5).to(torch.long)
    rows = []
    for frame_index, (root, cell) in enumerate(zip(predicted_root[CONTEXT_FRAMES:], cells)):
        cell_x, cell_z = int(cell[0]), int(cell[1])
        inside = 0 <= cell_x < heightfield.shape[1] and 0 <= cell_z < heightfield.shape[0]
        known = inside and bool(torch.isfinite(heightfield[cell_z, cell_x]))
        rows.append({
            "prediction_frame_index": frame_index + CONTEXT_FRAMES,
            "predicted_root_world_xyz_m": [float(v) for v in root[:3]],
            "heightfield_cell_zx": [cell_z, cell_x],
            "known_heightfield_cell": known,
            "heightfield_world_y_m": float(heightfield[cell_z, cell_x].item()) if known else None,
        })
    return rows


def _height_summary(predicted_root, current_y):
    future = predicted_root[CONTEXT_FRAMES:]
    if future.shape != (NUM_TOKENS * 4 - CONTEXT_FRAMES, 5):
        raise ValueError("Root forecast must provide the 8-token, 32-frame horizon")

    def block_summary(start, stop):
        block = future[start:stop]
        dy = block[:, 1] - current_y
        return {
            "prediction_frame_indices_inclusive": [start + CONTEXT_FRAMES, stop - 1 + CONTEXT_FRAMES],
            "predicted_y_m": block[:, 1].tolist(),
            "delta_y_from_current_m": dy.tolist(),
            "predicted_y_range_m": [float(block[:, 1].min()), float(block[:, 1].max())],
            "delta_y_from_current_range_m": [float(dy.min()), float(dy.max())],
        }

    return {
        "future_prediction_frame_indices_inclusive": [CONTEXT_FRAMES, NUM_TOKENS * 4 - 1],
        "current_history_root_y_m": float(current_y),
        "future_predicted_y_range_m": [float(future[:, 1].min()), float(future[:, 1].max())],
        "future_delta_y_from_current_range_m": [
            float((future[:, 1] - current_y).min()), float((future[:, 1] - current_y).max())
        ],
        "first_published_8_future_frames": block_summary(0, 8),
        "forecast_tail_8_future_frames": block_summary(future.shape[0] - 8, future.shape[0]),
    }


@torch.inference_mode()
def run_root_forecast(model, prepared, observation, scene):
    inputs = prepared["inputs"]
    output = model.backbone_net(
        inputs["global_root_values"], inputs["has_global_root_values"],
        inputs["local_root_values"], inputs["has_local_root_values"],
        inputs["poses"], inputs["has_poses"], inputs["num_tokens"],
        config={"allow_pred_out_of_reach_num_tokens": False},
        terrain_observation=observation.as_model_inputs(),
        global_root_component_mask=inputs["global_root_component_mask"],
    )
    predicted_tokens = int(output["pred_num_tokens"].reshape(-1)[0].item())
    if predicted_tokens != NUM_TOKENS:
        raise ValueError(f"Root returned {predicted_tokens} tokens; expected {NUM_TOKENS}")
    frame_count = NUM_TOKENS * model.backbone_net.get_num_frames_per_token()
    normalized_root = output["pred_global_root_values"][:, :frame_count]
    if normalized_root.shape != (1, NUM_TOKENS * 4, 5) or not torch.isfinite(normalized_root).all():
        raise ValueError("Root returned an invalid 8-token forecast")

    # Match online inference: unnormalize, then restore initial XZ by translation only.
    predicted_root = model.global_motion_rep.unnormalize(normalized_root).clone()
    xz_indices = torch.as_tensor(model.global_motion_rep.indices["global_root_pos_2d"],
                                 device=predicted_root.device)
    predicted_root[..., xz_indices] += prepared["initial_xz"]
    predicted_root = predicted_root[0].detach().cpu()
    current_y = float(prepared["history_root"][0, -1, 1].item())
    return {
        "pred_num_tokens": predicted_tokens,
        "model_input_sha256": prepared["input_sha256"],
        "planar_goal_world_xz_heading_cos_sin": prepared["goal"][0].detach().cpu().tolist(),
        "initial_world_xz_translation_only": prepared["initial_xz"][0, 0].detach().cpu().tolist(),
        "root_world_xyz_heading_cos_sin": predicted_root.tolist(),
        "height": _height_summary(predicted_root, current_y),
        "terrain_at_predicted_root_points_diagnostic_only": _terrain_at_predicted_roots(
            scene, predicted_root),
        "postprocessing": "unnormalize plus initial-XZ translation only; no yaw rotation or final-root replacement",
        "terrain_lookup_interpretation": "known heightfield values only; not contact or support evidence",
    }


def review(root_1k_run, root_10k_run, bundle_path, *, device="cpu"):
    bundle = load_input_bundle(bundle_path)
    report = {
        "schema": "motionbricks.parc_root_height_review.v1",
        "scope": "Root-only static forecast diagnostic; not a complete controller or acceptance test",
        "future_reference_inputs_used": False,
        "pose_model_instantiated": False,
        "quality_approved": False,
        "source": {
            "bundle_path": bundle["bundle_path"],
            "bundle_sha256": bundle["bundle_sha256"],
            "schema": bundle["schema"],
            "protocol": bundle["protocol"],
            "seed_provenance": bundle["seed_provenance"],
            "history_tensor_sha256": bundle["history_tensor_sha256"],
            "scene_tensor_sha256": bundle["scene_tensor_sha256"],
            "stats_tensor_sha256": bundle["stats_tensor_sha256"],
            "dataset_manifest_sha256": bundle["metadata"].get("dataset_manifest_sha256"),
            "history_frames": CONTEXT_FRAMES,
            "history_root_world_xyz_heading_cos_sin": [],
        },
        "conditions": {},
        "runs": {},
    }

    runs = (("root_1k", root_1k_run, 1000), ("root_10k", root_10k_run, 10000))
    for run_name, run_path, expected_updates in runs:
        model, source = _load_root_run(run_path, bundle, expected_updates, device)
        report["runs"][run_name] = {"source": source, "scenarios": {}}
        for scenario_name, displacement in SCENARIOS:
            prepared = prepare_root_inputs(model, bundle["history"], displacement)
            observation, direction = build_current_terrain_observation(
                prepared["history_root"], prepared["goal"], bundle["scene"])
            forecast = run_root_forecast(model, prepared, observation, bundle["scene"])
            forecast.update({
                "local_endpoint_displacement_xz_m": list(displacement),
                "terrain_direction_local_xz": direction[0].detach().cpu().tolist(),
                "current_terrain_observation": _terrain_observation_record(observation),
            })
            report["runs"][run_name]["scenarios"][scenario_name] = forecast
            if not report["source"]["history_root_world_xyz_heading_cos_sin"]:
                report["source"]["history_root_world_xyz_heading_cos_sin"] = (
                    prepared["history_root"][0].detach().cpu().tolist())
            condition = report["conditions"].setdefault(scenario_name, {
                "local_endpoint_displacement_xz_m": list(displacement),
                "num_tokens": NUM_TOKENS,
                "future_root_y_supplied": False,
                "future_pose_supplied": False,
                "local_root_velocity_supplied_frames": [0, 1, 2],
                "history_pose_supplied_frames": [0, 1, 2, 3],
            })
            condition[run_name + "_model_input_sha256"] = prepared["input_sha256"]
            condition[run_name + "_terrain_observation_sha256"] = forecast[
                "current_terrain_observation"]["tensor_sha256"]
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    for scenario in report["conditions"].values():
        scenario["same_normalized_inputs_across_runs"] = (
            scenario["root_1k_model_input_sha256"] == scenario["root_10k_model_input_sha256"])
        scenario["same_current_terrain_observation_across_runs"] = (
            scenario["root_1k_terrain_observation_sha256"]
            == scenario["root_10k_terrain_observation_sha256"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-1k-run", type=Path, required=True)
    parser.add_argument("--root-10k-run", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="Existing portable online bundle; only seed, scene, stats and provenance are read")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int,
                        help="Select a GPU only when it is idle according to the training runner's gate")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new review path")
    if args.threads < 1:
        parser.error("--threads must be positive")

    gpu_selection = None
    device = "cpu"
    if args.gpu is not None:
        from scripts.train_native_mb import select_gpu

        gpu_selection = select_gpu(args.gpu)
        device = "cuda:0"
    torch.set_num_threads(args.threads)
    report = review(args.root_1k_run, args.root_10k_run, args.bundle, device=device)
    report["device"] = device
    if gpu_selection is not None:
        report["gpu_selection"] = gpu_selection

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "quality_approved": False,
        "scenario_count": len(report["conditions"]),
        "same_inputs_across_runs": all(
            row["same_normalized_inputs_across_runs"]
            and row["same_current_terrain_observation_across_runs"]
            for row in report["conditions"].values()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
