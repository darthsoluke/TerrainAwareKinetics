"""Read-only, six-window LAFAN reconstruction and endpoint-generation review."""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from scripts.evaluate_native_mb_checkpoint import load_native_checkpoint, reconstruction_metrics
from scripts.evaluate_native_mb_inbetween import predict_window
from scripts.evaluate_native_mb_lafan import (
    _read_features as read_features, _sha256, _validate_frozen_tokenizer,
    select_validation_windows,
)


FREE_START, FREE_END, ENDPOINT = 4, 62, 63
HORIZON, WARMUP, ITERATIONS = 64, 3, 20
SEED = 2333


def load_bundle(*args, **kwargs):
    from scripts.evaluate_native_mb_pipeline import load_bundle as original

    return original(*args, **kwargs)


def state_digest(model):
    from scripts.train_native_mb_control import state_digest as original

    return original(model)


def validate_windows(windows):
    identities = {(row["shard"], row["start"]) for row in windows}
    if (len(windows) != 6 or len(identities) != 6
            or any(row.get("subject_id") != "subject4" or row.get("split") != "validation"
                   for row in windows)
            or any(sum(row.get("behavior") == behavior for row in windows) != 3
                   for behavior in ("walk", "run"))):
        raise ValueError("Requires exactly six unique subject4 validation windows: three walk and three run")


def validate_output(output, inputs):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Review output must be a fresh directory")
    for value in inputs:
        source = Path(value).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Review output must be separate from all source, dataset and run directories")
    return output


def _check_motion(motion):
    import torch

    for name, shape in (("posed_joints", (1, HORIZON, 30, 3)),
                        ("local_joint_rots", (1, HORIZON, 30, 3, 3))):
        value = motion[name]
        if value.shape != shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Invalid finite full-body {name}; expected {shape}")


def score_motion(predicted, source, root_index, fps):
    """Exclude both conditioned keyframes from free scores; endpoint has no velocity."""
    import torch

    _check_motion(predicted)
    _check_motion(source)
    free = reconstruction_metrics(predicted, source, root_index, fps,
                                  frame_start=FREE_START, frame_end=FREE_END)
    joint_error = (predicted["posed_joints"][:, ENDPOINT] - source["posed_joints"][:, ENDPOINT])
    relative = (predicted["local_joint_rots"][:, ENDPOINT].transpose(-1, -2)
                @ source["local_joint_rots"][:, ENDPOINT])
    angle = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    return {"free_interior": free, "endpoint_63": {
        "mpjpe_m": float(joint_error.norm(dim=-1).mean()),
        "root_error_m": float(joint_error[:, root_index].norm(dim=-1).mean()),
        "local_joint_rotation_geodesic_rad": float(angle.mean()),
    }}


def raw_fk_diagnostics(joints, root_index, fps):
    """Raw acceleration is a relative diagnostic, not a naturalness score."""
    joints = np.asarray(joints)
    if (joints.shape != (HORIZON, 30, 3) or not np.isfinite(joints).all()
            or not math.isfinite(fps) or fps <= 0 or not 0 <= root_index < 30):
        raise ValueError("Diagnostics require finite 64x30x3 metre FK, valid root and FPS")
    free = joints[FREE_START:FREE_END + 1].astype(np.float64)
    acceleration = np.diff(free, n=2, axis=0) * fps ** 2
    magnitude = np.linalg.norm(acceleration, axis=-1)
    root_magnitude = magnitude[:, root_index]
    roots = joints[:, root_index].astype(np.float64)
    displacement = roots[-1] - roots[0]
    return {
        "raw_fk_acceleration_mean_mps2": float(magnitude.mean()),
        "raw_fk_acceleration_p95_mps2": float(np.quantile(magnitude, 0.95)),
        "raw_root_acceleration_mean_mps2": float(root_magnitude.mean()),
        "raw_root_acceleration_p95_mps2": float(np.quantile(root_magnitude, 0.95)),
        "acceleration_stencil_frames_inclusive": [FREE_START, FREE_END],
        "acceleration_center_frames_inclusive": [FREE_START + 1, FREE_END - 1],
        "root_displacement_frames_inclusive": [0, ENDPOINT],
        "root_displacement_xyz_m": displacement.tolist(),
        "root_displacement_norm_m": float(np.linalg.norm(displacement)),
        "root_path_length_m": float(np.linalg.norm(np.diff(roots, axis=0), axis=-1).sum()),
    }


def tokenizer_reconstruction(model, raw):
    import torch
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    rep, local_rep, network = model.motion_rep, model.local_motion_rep, model.pose_net
    with full_precision_evaluation(), torch.inference_mode():
        canonical = rep.change_first_heading(raw, 0.0, False, False)
        local = rep.dual_rep.global_to_local(
            rep.normalize(canonical), True, True,
            lengths=torch.tensor([65], device=raw.device),
        )[:, :HORIZON]
        external = extract_feature_from_motion_rep(local, local_rep,
                                                   network.decoder_external_cond_feature_mode)
        prediction = network(local, target_cond=None, has_target_cond=None,
                             external_cond=external)["recon_state"]
        source = local_rep.inverse(local, True, return_all=True)
        reconstructed = local_rep.inverse(prediction, True, return_all=True)
    _check_motion(source)
    _check_motion(reconstructed)
    return source, reconstructed


def _cpu_motion(motion):
    return {name: motion[name].detach().cpu().clone()
            for name in ("posed_joints", "local_joint_rots")}


def prepare_sole_review(source_dir, dataset_dir, skeleton):
    """Require the existing conversion's fixed-floor provenance, never fit a plane."""
    manifest = json.loads((dataset_dir / "native_manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((source_dir / "accepted.json").read_text(encoding="utf-8"))
    ground = (manifest.get("contract") or {}).get("ground")
    floors = {row["shard"]: row.get("fixed_plane_source_y_m") for row in rows}
    if (ground != "source fixed ground plane retained; no per-frame fitting or vertical correction"
            or manifest.get("source_manifest_sha256") != _sha256(source_dir / "accepted.json")
            or any(value is None or not math.isfinite(value) for value in floors.values())):
        return None, {"status": "not_assessed", "reason": "Missing bound per-recording fixed-floor provenance"}
    try:
        import torch
        from motionbricks.data.bones_seed import official_sole_surfaces
        from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner
        from motionbricks.data.sole_region_contact import sole_region_indices

        surfaces, _, _, mesh_digest = official_sole_surfaces(skeleton)
        skinner = move_skinner(load_official_skinner(skeleton), torch.device("cpu"), torch.float32)
        regions = {side: sole_region_indices(surfaces, side) for side in ("Left", "Right")}
    except (ImportError, FileNotFoundError) as error:
        return None, {"status": "not_assessed", "reason": "Official sole assets unavailable: " + str(error)}
    return (skinner, regions), {"status": "assessed", "ground_height_m": 0.0,
        "plane": "One estimated plane per recording, retained from stationary source-anchor/official sole calibration; not a capture-calibrated plane",
        "mesh_sha256": mesh_digest, "source_fixed_plane_y_m": floors,
        "mask": "Source-derived same-numbered material vertex mask only; never reselected from predictions"}


def sole_diagnostics(motions, root_index, fps, sole_setup):
    import torch
    from motionbricks.data.soma_skin_reference import skinned_vertices
    from motionbricks.data.sole_region_contact import (
        material_point_slip, SOLE_CONTACT_CLEARANCE_M, SOLE_MIN_CONTACT_RUN_FRAMES,
        SOLE_REST_VERTICAL_SPEED_MPS,
    )

    skinner, regions = sole_setup
    # SOMASkin.skin requires [T,J,3] positions, not a [T,3] root trajectory.
    vertices = {name: skinned_vertices(skinner, motion["local_joint_rots"][0],
                                       motion["posed_joints"][0])
                for name, motion in motions.items()}
    result = {}
    for side, indices in regions.items():
        points = {name: value[:, indices] for name, value in vertices.items()}
        source = points["source"]
        slip = material_point_slip(source[..., 1], source[..., [0, 2]], fps)
        mask = slip["vertex_measured"][FREE_START:FREE_END]
        metrics = {}
        for name, value in points.items():
            speed = torch.diff(value[..., [0, 2]], dim=0).norm(dim=-1) * fps
            measured = speed[FREE_START:FREE_END][mask]
            penetration = torch.relu(-value[FREE_START:FREE_END + 1, :, 1])
            metrics[name] = {
                "slip_status": "measured_on_source_mask" if measured.numel() else "not_assessed",
                "slip_mean_mps": float(measured.mean()) if measured.numel() else None,
                "slip_p95_mps": float(torch.quantile(measured, .95)) if measured.numel() else None,
                "slip_max_mps": float(measured.max()) if measured.numel() else None,
                "penetration_mean_m": float(penetration.mean()),
                "penetration_p95_m": float(torch.quantile(penetration.flatten(), .95)),
                "penetration_max_m": float(penetration.max()),
            }
        result[side] = {"selected_vertex_indices": indices.tolist(),
            "source_measured_vertex_intervals": int(mask.sum()),
            "source_measured_intervals": int(mask.any(dim=-1).sum()),
            "source_mask_sha256": hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest(),
            "thresholds": {"contact_clearance_m": SOLE_CONTACT_CLEARANCE_M,
                           "min_contact_run_frames": SOLE_MIN_CONTACT_RUN_FRAMES,
                           "rest_vertical_speed_mps": SOLE_REST_VERTICAL_SPEED_MPS},
            "metrics": metrics,
            "limitation": "Source-timed relative slip diagnostic; airborne or wrong-timed predicted feet are not proof of stable contact"}
    return result


def hardware_precision(infer, raw):
    import torch

    devices = []
    if raw.device.type == "cuda":
        properties = torch.cuda.get_device_properties(raw.device)
        devices.append({"name": properties.name, "total_memory_bytes": properties.total_memory,
                        "compute_capability": [properties.major, properties.minor],
                        "device_index": raw.device.index or 0})
    models = {"root": infer._root_model, "pose": infer._pose_model,
              "tokenizer": infer._vqvae_pose_model}
    dtypes = {name: sorted({str(parameter.dtype) for parameter in model.parameters()})
              for name, model in models.items()}
    return {"hardware": {"device": str(raw.device), "gpus": devices,
                          "platform": platform.platform(), "cpu": platform.processor(),
                          "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                          "cudnn_version": torch.backends.cudnn.version()},
            "precision": {"input_dtype": str(raw.dtype), "parameter_dtypes": dtypes,
                          "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                          "float32_matmul_precision": torch.get_float32_matmul_precision(),
                          "autocast_enabled": torch.is_autocast_enabled(),
                          "policy": "TF32 disabled; no weight casting or autocast requested"}}


def benchmark_prediction(infer, raws, *, predictor=None, synchronize=None, timer=None):
    """Three warmups plus twenty synchronized batch-one complete predictions total."""
    import torch

    if len(raws) != 6 or any(raw.shape[0] != 1 or raw.shape[1] != 65 for raw in raws):
        raise ValueError("Benchmark requires six batch-one 65-frame inputs")
    predictor = predictor or predict_window
    timer = timer or time.perf_counter
    if synchronize is None:
        synchronize = (lambda: torch.cuda.synchronize(raws[0].device)) if raws[0].is_cuda else lambda: None
    durations, indices = [], []
    with full_precision_evaluation(), torch.inference_mode():
        for iteration in range(WARMUP + ITERATIONS):
            index = iteration % len(raws)
            synchronize()
            started = timer() if iteration >= WARMUP else None
            source, prediction = predictor(infer, raws[index])
            synchronize()
            if started is not None:
                durations.append((timer() - started) * 1000)
                indices.append(index)
            if (prediction.shape != source.shape or prediction.shape[:2] != (1, HORIZON)
                    or not torch.isfinite(prediction).all()):
                raise ValueError("Benchmark prediction must be finite batch-one horizon64")
        metadata = hardware_precision(infer, raws[0])
    if not all(math.isfinite(value) and value > 0 for value in durations):
        raise ValueError("Invalid measured prediction duration")
    return {**metadata, "status": "measured_on_reported_hardware_only", "batch_size": 1,
            "warmup_iterations": WARMUP, "measured_iterations": ITERATIONS,
            "sample_window_indices": indices, "latency_ms": {
                "p50": float(np.quantile(durations, 0.50)),
                "p95": float(np.quantile(durations, 0.95)), "max": max(durations),
                "samples": durations},
            "output_horizon_frames": HORIZON, "output_horizon_seconds": HORIZON / 30,
            "scope": "complete predict_window: conditions + Root + Pose + decoder; synchronized; excludes loading, FK, rendering",
            "not_measured": ["UE runtime", "RTX3080 deployment target"],
            "ue_measured": False, "rtx3080_target_measured": False}


def save_window(output, index, window, motions, fps, *, renderer=None):
    if renderer is None:
        from scripts.render_native_mb_review import _render_motion_set

        renderer = _render_motion_set
    for motion in motions.values():
        _check_motion(motion)
    stem = f"{index:02d}_{window['behavior']}_start{window['start']:06d}"
    path = output / f"{stem}.npz"
    arrays = {name: motion["posed_joints"][0].detach().cpu().numpy().copy()
              for name, motion in motions.items()}
    np.savez_compressed(path, **arrays, fps=np.float32(fps),
                        **{name + "_local_rot_mats": motion["local_joint_rots"][0].cpu().numpy()
                           for name, motion in motions.items()})
    renderer(path, tuple(arrays[name] for name in ("source", "tokenizer", "generated")),
             fps, output, f"{window['behavior']} | free 4-62; endpoint 63 separate\n"
             "Source-encoded reconstruction vs endpoint generation | no output repair",
             three_panel=True, panel_labels=("Source SOMA motion", "Tokenizer reconstruction",
                                             "Root/Pose generation"))
    return {"npz": path.name, "png": path.with_suffix(".png").name,
            "gif": path.with_suffix(".gif").name}


def review_system(source_dir, dataset_dir, tokenizer_run, root_run, pose_run, output_dir,
                  *, device="cuda"):
    import torch

    inputs = list(map(Path, (source_dir, dataset_dir, tokenizer_run, root_run, pose_run)))
    source_dir, dataset_dir, tokenizer_run, root_run, pose_run = inputs
    output = validate_output(output_dir, inputs)
    windows, excluded = select_validation_windows(source_dir, dataset_dir)
    validate_windows(windows)
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested review device is unavailable")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tokenizer, tokenizer_info, binding = load_native_checkpoint(tokenizer_run, device=device)
    manifest_hash = _sha256(dataset_dir / "native_manifest.json")
    if (binding["dataset_dir"].resolve() != dataset_dir.resolve()
            or tokenizer_info["dataset_manifest_sha256"] != manifest_hash
            or tokenizer_info["run_status_at_review"] != "completed"):
        raise ValueError("Tokenizer must bind this completed native dataset/run")
    tokenizer_digest = state_digest(tokenizer.pose_net)
    _validate_frozen_tokenizer(tokenizer_info, tokenizer_digest, pose_run)
    tokenizer.eval()
    cached, raws = [], []
    for window in windows:
        raw = torch.from_numpy(read_features(dataset_dir, window)).to(device)[None]
        source, reconstructed = tokenizer_reconstruction(tokenizer, raw)
        cached.append((_cpu_motion(source), _cpu_motion(reconstructed)))
        raws.append(raw)
    del tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    infer, loaded_dataset, bundle = load_bundle(root_run, pose_run, device)
    if (Path(loaded_dataset).resolve() != dataset_dir.resolve()
            or bundle["dataset_manifest_sha256"] != manifest_hash
            or (bundle.get("tokenizer") or {}).get("checkpoint_sha256") != tokenizer_info["checkpoint_sha256"]
            or state_digest(infer._vqvae_pose_model) != tokenizer_digest):
        raise ValueError("Root/Pose bundle or frozen tokenizer identity differs from reviewed inputs")
    infer.eval()
    rep = infer.global_motion_rep
    root_index, fps = rep.skeleton.root_idx, float(rep.fps)
    if fps != 30:
        raise ValueError("Review requires native 30 Hz")
    sole_setup, sole_status = prepare_sole_review(source_dir, dataset_dir, rep.skeleton)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with full_precision_evaluation(), torch.inference_mode():
        for index, (window, raw, (local_source, reconstructed)) in enumerate(zip(windows, raws, cached)):
            source_features, generated_features = predict_window(infer, raw)
            source = _cpu_motion(rep.inverse(source_features, False, return_all=True))
            generated = _cpu_motion(rep.inverse(generated_features, False, return_all=True))
            _check_motion(source)
            for name in source:
                if not torch.allclose(source[name], local_source[name], atol=1e-5, rtol=1e-5):
                    raise ValueError("Tokenizer and generator canonical source FK disagree")
            motions = {"source": source, "tokenizer": reconstructed, "generated": generated}
            row = {**window, "scores": {
                name: score_motion(motions[name], source, root_index, fps)
                for name in ("tokenizer", "generated")},
                "raw_fk_diagnostics": {name: raw_fk_diagnostics(
                    motion["posed_joints"][0].numpy(), root_index, fps)
                    for name, motion in motions.items()}}
            row["artifacts"] = save_window(output, index, window, motions, fps)
            if sole_setup is not None:
                row["sole_diagnostics"] = sole_diagnostics(motions, root_index, fps, sole_setup)
            rows.append(row)
    performance = benchmark_prediction(infer, raws)
    report = {
        "schema": "native_mb_lafan_system_review_v1", "status": "completed_not_approved",
        "subject_id": "subject4", "split": "validation", "window_count": 6,
        "windows_per_behavior": {"walk": 3, "run": 3}, "fps": fps, "seed": SEED,
        "tokenizer": tokenizer_info, "bundle": bundle,
        "dataset_manifest_sha256": manifest_hash,
        "source_accepted_sha256": _sha256(source_dir / "accepted.json"),
        "review_script_sha256": _sha256(Path(__file__)),
        "free_interior_frames_inclusive": [FREE_START, FREE_END],
        "velocity_interval_endpoint_frames_inclusive": [FREE_START, FREE_END],
        "endpoint_frame": ENDPOINT, "rendered_frames_inclusive": [0, ENDPOINT],
        "conditions": {"tokenizer": "complete source encoding plus source planar root velocity; no pose keyframes",
                       "generated": "source global/local root and pose keyframes 3/63 only; frame0 coordinate reference; 16-token horizon; endpoint derivatives use 3->4 and 63->64",
                       "source_interior_body_conditioning_for_generation": False},
        "postprocessing": "none", "ik": False, "smoothing": False, "root_overwrite": False,
        "quality_approved": False, "ue_approved": False,
        "slip_and_penetration": sole_status,
        "wasd_continuous_review": {"status": "not_assessed", "reason": "No WASD command/reference bank"},
        "diagnostic_interpretation": "Raw FK acceleration mean/P95 are relative diagnostics, not naturalness or physical-feasibility conclusions; generated source error is not unique-motion truth.",
        "limitations": ["Only three windows per accepted subject4 walk/run behavior; not full clips or collection coverage",
                        "Subject5 test untouched; source is prepared SOMA FK, not a Geno retarget-quality audit",
                        "No UE/RTX3080 target performance or runtime approval; no 2M training-completion claim"],
        "excluded_short_windows": excluded, "windows": rows, "performance": performance,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset", "tokenizer-run", "root-run", "pose-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    report = review_system(args.source, args.dataset, args.tokenizer_run, args.root_run,
                           args.pose_run, args.output, device=args.device)
    print(json.dumps({"status": report["status"], "window_count": report["window_count"],
                      "report": str(args.output.resolve() / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
