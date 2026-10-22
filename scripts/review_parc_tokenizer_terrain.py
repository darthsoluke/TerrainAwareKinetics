"""Export a bounded, browser-skinned review of native PARC tokenizer reconstruction."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_control_review import (
    GLOBAL_WINDOW_FRAMES,
    LOCAL_WINDOW_FRAMES,
    _source_window,
)
from scripts import evaluate_native_mb_checkpoint as native_evaluation
from scripts import evaluate_parc_native_terrain as terrain_evaluation


SCHEMA = "parc_tokenizer_terrain_review.v1"
COORDINATE_FRAME = "soma_y_up_world"
FPS = 30.0
MAX_WINDOWS = 8
JOINTS = 30
SCORE_START = native_evaluation.SCORE_START
SCORE_END = native_evaluation.SCORE_END


def _as_numpy(value):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_review_arrays(roots_source, local_source, roots_codec, local_codec,
                           hf, min_point, dx, fps=FPS):
    """Validate the fixed SOMA30/browser archive dimensions and finite values."""
    roots_source = np.asarray(roots_source)
    roots_codec = np.asarray(roots_codec)
    local_source = np.asarray(local_source)
    local_codec = np.asarray(local_codec)
    hf = np.asarray(hf)
    min_point = np.asarray(min_point)
    if roots_source.shape != (LOCAL_WINDOW_FRAMES, 3) or roots_codec.shape != roots_source.shape:
        raise ValueError("roots_source and roots_codec must both have shape (64,3)")
    expected_rotations = (LOCAL_WINDOW_FRAMES, JOINTS, 3, 3)
    if local_source.shape != expected_rotations or local_codec.shape != expected_rotations:
        raise ValueError("local_source and local_codec must both have shape (64,30,3,3)")
    if hf.ndim != 2 or min(hf.shape) < 1:
        raise ValueError("hf must be a nonempty native [Z,X] heightfield")
    if min_point.shape != (3,):
        raise ValueError("min_point must have shape (3,)")
    if (not np.isfinite(roots_source).all() or not np.isfinite(roots_codec).all()
            or not np.isfinite(local_source).all() or not np.isfinite(local_codec).all()
            or not np.isfinite(hf).all() or not np.isfinite(min_point).all()):
        raise ValueError("Review arrays must contain only finite values")
    dx_value = np.asarray(dx)
    if (isinstance(dx, (bool, np.bool_)) or dx_value.shape != ()
            or not np.isfinite(dx_value).all() or float(dx_value) <= 0):
        raise ValueError("dx must be a finite positive scalar")
    if (isinstance(fps, (bool, np.bool_)) or not np.isscalar(fps)
            or not np.isfinite(fps) or float(fps) != FPS):
        raise ValueError("Native PARC tokenizer review requires fps=30")
    for name, rotations in (("local_source", local_source), ("local_codec", local_codec)):
        _validate_rotation_matrices(rotations, name)
    return {
        "roots_source": roots_source.astype(np.float32, copy=False),
        "local_source": local_source.astype(np.float32, copy=False),
        "roots_codec": roots_codec.astype(np.float32, copy=False),
        "local_codec": local_codec.astype(np.float32, copy=False),
        "hf": hf.astype(np.float32, copy=False),
        "min_point": min_point.astype(np.float32, copy=False),
        "dx": np.asarray(dx, dtype=np.float32),
        "fps": np.asarray(fps, dtype=np.float32),
    }


def _validate_rotation_matrices(rotations, name):
    identity = np.eye(3, dtype=np.float64)
    values = np.asarray(rotations, dtype=np.float64)
    gram = np.swapaxes(values, -1, -2) @ values
    determinants = np.linalg.det(values)
    if (not np.allclose(gram, identity, atol=3e-3, rtol=0)
            or np.any(determinants <= 0)
            or not np.allclose(determinants, 1.0, atol=3e-3, rtol=0)):
        raise ValueError(f"{name} must contain proper 3x3 rotation matrices")


def transform_motion_to_world(joints, local_rotations, world_transform, root_index):
    """Apply a rigid world transform to FK positions and the root local orientation."""
    joints = np.asarray(joints)
    local_rotations = np.asarray(local_rotations)
    if joints.shape != (LOCAL_WINDOW_FRAMES, JOINTS, 3):
        raise ValueError("posed joints must have shape (64,30,3)")
    if local_rotations.shape != (LOCAL_WINDOW_FRAMES, JOINTS, 3, 3):
        raise ValueError("local rotations must have shape (64,30,3,3)")
    if (isinstance(root_index, bool) or not isinstance(root_index, (int, np.integer))
            or not 0 <= int(root_index) < JOINTS):
        raise ValueError("root_index must identify one of the 30 SOMA joints")
    rotation = _as_numpy(world_transform["rotation"]).astype(np.float64, copy=False)
    translation = _as_numpy(world_transform["translation"]).astype(np.float64, copy=False)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0)
            or np.linalg.det(rotation) <= 0):
        raise ValueError("world_transform must be a finite proper rigid transform")
    world_joints = np.einsum("ab,tjb->tja", rotation, joints) + translation
    world_rotations = np.asarray(local_rotations, dtype=np.float64).copy()
    world_rotations[:, int(root_index)] = np.einsum(
        "ab,tbc->tac", rotation, world_rotations[:, int(root_index)])
    _validate_rotation_matrices(world_rotations, "world local rotations")
    roots = world_joints[:, int(root_index)]
    return roots.astype(np.float32), world_rotations.astype(np.float32), world_joints.astype(np.float32)


def select_review_windows(windows, records, max_windows, dataset_identity):
    if (isinstance(max_windows, bool) or not isinstance(max_windows, int)
            or not 1 <= max_windows <= MAX_WINDOWS):
        raise ValueError(f"max_windows must be an integer from 1 to {MAX_WINDOWS}")
    return terrain_evaluation.select_windows(windows, records, max_windows, dataset_identity)


def _safe_component(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return value[:48] or "example"


def _load_terrain_geometry(dataset_dir, record, manifest):
    relative = record.get("terrain_geometry_snapshot")
    expected = record.get("terrain_geometry_sha256")
    artifacts = manifest.get("artifact_sha256", {})
    if (not isinstance(relative, str) or not relative
            or not isinstance(expected, str) or len(expected) != 64
            or artifacts.get(relative) != expected):
        raise ValueError("Accepted record does not bind its terrain geometry snapshot")
    path = (dataset_dir / relative).resolve()
    if dataset_dir.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Terrain geometry snapshot is missing or escapes the dataset: {relative}")
    actual = native_evaluation.sha256_file(path)
    if actual != expected:
        raise ValueError(f"Terrain geometry snapshot SHA256 mismatch: {relative}")
    with np.load(path, allow_pickle=False) as data:
        required = {"heightfield_y_m", "min_point_world_xyz_m", "dx_m"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Terrain geometry snapshot lacks {sorted(missing)}: {relative}")
        geometry = {
            "hf": data["heightfield_y_m"].copy(),
            "min_point": data["min_point_world_xyz_m"].copy(),
            "dx": np.asarray(data["dx_m"]).copy(),
        }
    if geometry["dx"].ndim != 0:
        raise ValueError(f"Terrain geometry dx_m must be scalar: {relative}")
    return geometry, {"file": relative, "sha256": actual}


def _decode_source_codec(model, raw_window, device):
    """Use the checkpoint evaluator's source encode and token-index decoder path."""
    import torch
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    rep = model.motion_rep
    local_rep = model.local_motion_rep
    network = model.pose_net
    if rep.fps != FPS or rep.skeleton.nbjoints != JOINTS:
        raise ValueError("Checkpoint representation must be 30 Hz SOMA30")
    expected_dim = len(rep.indices["all"])
    if raw_window.shape != (GLOBAL_WINDOW_FRAMES, expected_dim) or not np.isfinite(raw_window).all():
        raise ValueError(f"Source window must be finite with shape (65,{expected_dim})")

    raw = torch.as_tensor(raw_window, dtype=torch.float32, device=device)[None]
    canonical = rep.change_first_heading(raw, 0.0, False, False)
    global_motion = rep.normalize(canonical)
    local = rep.dual_rep.global_to_local(
        global_motion, is_normalized=True, to_normalize=True,
        lengths=torch.tensor([GLOBAL_WINDOW_FRAMES], device=device),
    )[:, :LOCAL_WINDOW_FRAMES]
    if local.shape != (1, LOCAL_WINDOW_FRAMES, len(local_rep.indices["all"])):
        raise ValueError("Native global-to-local conversion returned an unexpected shape")
    external = extract_feature_from_motion_rep(
        local, local_rep, network.decoder_external_cond_feature_mode)
    indices = network.encode_into_idx(local, fetch_overall_indices=False)
    if (not isinstance(indices, torch.Tensor) or indices.ndim != 3
            or indices.shape[:2] != (1, native_evaluation.NUM_TOKENS)):
        raise ValueError("Native tokenizer must encode the 64-frame source into exactly 16 tokens")
    decoded_result = network.forward_decoder(
        indices, target_cond=None, has_target_cond=None, external_cond=external,
        use_overall_indices=False,
    )
    decoded = decoded_result.get("recon_state") if isinstance(decoded_result, dict) else None
    if (not isinstance(decoded, torch.Tensor) or decoded.shape != local.shape
            or not torch.is_floating_point(decoded) or not torch.isfinite(decoded).all()):
        raise ValueError("Native token-index decoder returned invalid reconstruction features")
    reconstructed = rep.dual_rep.local_to_global(
        decoded, is_normalized=True, to_normalize=False,
        lengths=torch.tensor([LOCAL_WINDOW_FRAMES], device=device),
    )
    if (reconstructed.shape != (1, LOCAL_WINDOW_FRAMES, expected_dim)
            or not torch.isfinite(reconstructed).all()):
        raise ValueError("Native codec reconstruction returned invalid global features")
    return canonical[:, :LOCAL_WINDOW_FRAMES], reconstructed


def _fk_result(rep, features):
    result = rep.inverse(features, is_normalized=False, return_all=True)
    joints = _as_numpy(result["posed_joints"])[0]
    local_rotations = _as_numpy(result["local_joint_rots"])[0]
    if joints.shape != (LOCAL_WINDOW_FRAMES, JOINTS, 3):
        raise ValueError("Native representation FK did not return 64 SOMA30 joint positions")
    if local_rotations.shape != (LOCAL_WINDOW_FRAMES, JOINTS, 3, 3):
        raise ValueError("Native representation did not return 64 SOMA30 local rotations")
    if not np.isfinite(joints).all() or not np.isfinite(local_rotations).all():
        raise ValueError("Native representation FK returned nonfinite arrays")
    _validate_rotation_matrices(local_rotations, "native local rotations")
    return joints, local_rotations


def _verify_fk_roundtrip(skeleton, roots, local_rotations, expected_joints,
                         *, tolerance_m=2e-4):
    import torch

    device = skeleton.joint_parents.device
    _, positions, _ = skeleton.fk(
        torch.as_tensor(np.asarray(local_rotations, dtype=np.float32), device=device),
        torch.as_tensor(np.asarray(roots, dtype=np.float32), device=device),
    )
    actual = _as_numpy(positions)
    if actual.shape != (LOCAL_WINDOW_FRAMES, JOINTS, 3) or not np.isfinite(actual).all():
        raise ValueError("SOMA FK returned invalid world positions")
    error = float(np.linalg.norm(actual - expected_joints, axis=-1).max())
    if not np.isfinite(error) or error > tolerance_m:
        raise ValueError(f"Exported world arrays fail SOMA FK roundtrip: {error:.9g} m")
    return error


def _window_metrics(model, source_features, codec_features, source_world_joints,
                    source_world_local, codec_world_joints, codec_world_local):
    import torch

    source = {
        "posed_joints": torch.as_tensor(source_world_joints, dtype=torch.float32)[None],
        "local_joint_rots": torch.as_tensor(source_world_local, dtype=torch.float32)[None],
    }
    codec = {
        "posed_joints": torch.as_tensor(codec_world_joints, dtype=torch.float32)[None],
        "local_joint_rots": torch.as_tensor(codec_world_local, dtype=torch.float32)[None],
    }
    metrics = native_evaluation.reconstruction_metrics(
        codec, source, model.motion_rep.skeleton.root_idx, model.motion_rep.fps,
        frame_start=SCORE_START, frame_end=SCORE_END,
    )
    relative = terrain_evaluation._rotation_geodesic(
        codec["local_joint_rots"], source["local_joint_rots"])[0]
    score = slice(SCORE_START, SCORE_END + 1)
    metrics["local_joint_rotation_geodesic_rad"] = float(relative[score].mean())
    root_index = int(model.motion_rep.skeleton.root_idx)
    metrics["root_local_rotation_geodesic_rad"] = float(relative[score, root_index].mean())
    keep = [joint for joint in range(JOINTS) if joint != root_index]
    metrics["nonroot_local_rotation_geodesic_rad"] = float(relative[score][:, keep].mean())
    normalized_source = model.motion_rep.normalize(source_features)
    normalized_codec = model.motion_rep.normalize(codec_features)
    metrics["global_feature_normalized_rmse"] = float(
        (normalized_codec[:, score] - normalized_source[:, score]).square().mean().sqrt())
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError("Nonfinite tokenizer terrain review metrics")
    return metrics


def _stable_example_id(window, record):
    identity = f"{window['shard']}\0{window['start']}\0{record.get('source_id', '')}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return "-".join((
        _safe_component(window["terrain_family"]),
        _safe_component(record.get("source_id") or Path(window["shard"]).stem),
        f"{int(window['start']):06d}", suffix,
    ))


def _summary(examples):
    if not examples:
        return {"example_count": 0, "mean": {}, "median": {}}
    keys = tuple(examples[0]["metrics"])
    return {
        "example_count": len(examples),
        "mean": {key: float(np.mean([row["metrics"][key] for row in examples])) for key in keys},
        "median": {key: float(np.median([row["metrics"][key] for row in examples])) for key in keys},
    }


def _assert_new_isolated_output(output, protected_paths):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Review output already exists: {output}")
    for protected in protected_paths:
        protected = Path(protected).resolve()
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError(f"Review output must not overlap immutable input: {protected}")
    return output


def export(run_dir, output, *, device="cpu", max_windows=MAX_WINDOWS):
    """Export primary source and full-codec branches from a completed native run."""
    import torch

    if (isinstance(max_windows, bool) or not isinstance(max_windows, int)
            or not 1 <= max_windows <= MAX_WINDOWS):
        raise ValueError(f"max_windows must be an integer from 1 to {MAX_WINDOWS}")
    model, checkpoint_info, binding = native_evaluation.load_native_checkpoint(
        run_dir, checkpoint=None, device=device)
    if binding["contract"].get("status") != "completed":
        raise ValueError("Tokenizer run contract must record status='completed'")
    output = _assert_new_isolated_output(
        output, (binding["dataset_dir"], binding["run_dir"]))
    model.eval()
    device = next(model.parameters()).device
    rep = model.motion_rep
    if (rep.fps != FPS or rep.skeleton.nbjoints != JOINTS
            or int(rep.skeleton.root_idx) not in range(JOINTS)):
        raise ValueError("Review requires a 30 Hz SOMA30 native tokenizer")

    (dataset_dir, selected, candidates, excluded, identity, manifest,
     protocol, records) = terrain_evaluation._dataset_windows(
        binding["dataset_dir"], max_windows)
    if identity["dataset_manifest_sha256"] != binding["dataset_manifest_sha256"]:
        raise ValueError("Checkpoint-bound and review dataset manifests differ")
    by_shard = {row["shard"]: row for row in records if row.get("split") == "validation"}
    prepared = []
    with native_evaluation.full_precision_evaluation(), torch.inference_mode():
        for window in selected:
            record = by_shard.get(window["shard"])
            if record is None:
                raise ValueError(f"Selected source has no accepted validation record: {window['shard']}")
            raw_np = _source_window(dataset_dir, window)
            raw = torch.as_tensor(raw_np, dtype=torch.float32, device=device)[None]
            source_features, codec_features = _decode_source_codec(model, raw_np, device)
            dataset_transform, transform_status = terrain_evaluation._world_transform(
                dataset_dir, window, device)
            if dataset_transform is None or transform_status != "available":
                raise ValueError(
                    f"World-coordinate export requires bound canonical transform fields: {window['shard']}"
                )
            world_transform = terrain_evaluation._window_world_transform(
                rep, raw, source_features, dataset_transform)
            source_joints, source_local = _fk_result(rep, source_features)
            codec_joints, codec_local = _fk_result(rep, codec_features)
            roots_source, local_source, source_world_joints = transform_motion_to_world(
                source_joints, source_local, world_transform, rep.skeleton.root_idx)
            roots_codec, local_codec, codec_world_joints = transform_motion_to_world(
                codec_joints, codec_local, world_transform, rep.skeleton.root_idx)
            source_roundtrip = _verify_fk_roundtrip(
                rep.skeleton, roots_source, local_source, source_world_joints)
            codec_roundtrip = _verify_fk_roundtrip(
                rep.skeleton, roots_codec, local_codec, codec_world_joints)
            geometry, geometry_identity = _load_terrain_geometry(
                dataset_dir, record, manifest)
            arrays = validate_review_arrays(
                roots_source, local_source, roots_codec, local_codec,
                geometry["hf"], geometry["min_point"], geometry["dx"], FPS)
            metrics = _window_metrics(
                model, source_features, codec_features,
                source_world_joints, local_source,
                codec_world_joints, local_codec)
            example_id = _stable_example_id(window, record)
            prepared.append({
                "id": example_id,
                "label": (f"{window['terrain_family']} | {record.get('source_id', window['shard'])}"
                          f" | frame {window['start']}"),
                "arrays": arrays,
                "metrics": metrics,
                "source_shard": window["shard"],
                "source_shard_sha256": window["shard_sha256"],
                "source_id": record.get("source_id"),
                "source_record_shard": record.get("source_shard"),
                "source_record_shard_sha256": record.get("source_shard_sha256"),
                "source_window_sha256": hashlib.sha256(
                    np.ascontiguousarray(raw_np).tobytes()).hexdigest(),
                "source_window_shape": list(raw_np.shape),
                "source_group": window["source_group"],
                "terrain_family": window["terrain_family"],
                "window_start": int(window["start"]),
                "terrain_geometry": geometry_identity,
                "source_fk_roundtrip_error_m": source_roundtrip,
                "codec_fk_roundtrip_error_m": codec_roundtrip,
            })

    output.mkdir(parents=True, exist_ok=False)
    examples_dir = output / "examples"
    examples_dir.mkdir()
    examples = []
    for item in prepared:
        relative_file = f"examples/{item['id']}.npz"
        archive_path = output / relative_file
        np.savez_compressed(archive_path, **item["arrays"])
        example = {key: value for key, value in item.items()
                   if key not in {"arrays", "source_fk_roundtrip_error_m", "codec_fk_roundtrip_error_m"}}
        example.update({
            "file": relative_file,
            "sha256": native_evaluation.sha256_file(archive_path),
            "metrics": {
                **item["metrics"],
                "source_fk_roundtrip_error_m": item["source_fk_roundtrip_error_m"],
                "codec_fk_roundtrip_error_m": item["codec_fk_roundtrip_error_m"],
            },
            "branch_roles": {
                "roots_source": "SOURCE reference root translation in SOMA Y-up world meters",
                "local_source": "SOURCE local rotations with world-frame root orientation",
                "roots_codec": "CODEC reconstruction root translation in SOMA Y-up world meters",
                "local_codec": "CODEC local rotations with world-frame root orientation",
            },
        })
        examples.append(example)

    review_manifest = {
        "schema": SCHEMA,
        "coordinate_frame": COORDINATE_FRAME,
        "fps": FPS,
        "interpretation": (
            "CODEC is the full source-encoded native tokenizer reconstruction, not generated motion or WASD. "
            "No IK, smoothing, root correction, or postprocessing is applied."
        ),
        "source_identity": {
            "run": checkpoint_info["run"],
            "checkpoint": checkpoint_info["checkpoint"],
            "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
            "config_sha256": checkpoint_info["config_sha256"],
            "global_step": checkpoint_info["global_step"],
            "run_status": checkpoint_info["run_status_at_review"],
            "dataset_dir": str(binding["dataset_dir"]),
            "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
            "accepted_json_sha256": identity["accepted_json_sha256"],
            "native_manifest_sha256": native_evaluation.sha256_file(
                binding["dataset_dir"] / "native_manifest.json"),
        },
        "metrics_protocol": {
            "source_split": "validation",
            "scored_frames_inclusive": [SCORE_START, SCORE_END],
            "root_error": "mean Euclidean root-joint position error in world meters",
            "root_relative_fk": "per-frame root-translation-removed FK MPJPE in meters",
            "local_rotation": "geodesic error in radians; root and nonroot joints also reported separately",
            "postprocessing": "none",
            "eligible_validation_windows": len(candidates),
            "excluded_short_shards": len(excluded),
            "max_windows": max_windows,
        },
        "branch_roles": {
            "SOURCE": "reference source motion in SOMA Y-up world coordinates",
            "CODEC": "actual full source-encoded native tokenizer reconstruction; not generation/WASD",
        },
        "archive_contract": {
            "roots_source": [LOCAL_WINDOW_FRAMES, 3],
            "local_source": [LOCAL_WINDOW_FRAMES, JOINTS, 3, 3],
            "roots_codec": [LOCAL_WINDOW_FRAMES, 3],
            "local_codec": [LOCAL_WINDOW_FRAMES, JOINTS, 3, 3],
            "hf": "native Y-up heightfield in [Z,X] order; heights are world Y meters",
            "min_point": "native cell-center world XYZ origin in meters; preserved from bound geometry snapshot",
            "dx": "native horizontal cell spacing in meters",
            "fps": FPS,
        },
        "terrain_protocol": protocol,
        "summary": _summary(examples),
        "examples": examples,
    }
    manifest_path = output / "review_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(review_manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return review_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="completed native tokenizer run containing final.ckpt")
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for a bounded browser review package")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-windows", type=int, default=MAX_WINDOWS,
                        help=f"deterministic validation sample, capped at {MAX_WINDOWS}")
    args = parser.parse_args(argv)
    if not 1 <= args.max_windows <= MAX_WINDOWS:
        parser.error(f"--max-windows must be from 1 to {MAX_WINDOWS}")
    if args.output.exists():
        parser.error("Use a new output directory")
    review = export(args.run, args.output, device=args.device, max_windows=args.max_windows)
    print(json.dumps({
        "review_manifest": str(Path(args.output).resolve() / "review_manifest.json"),
        "examples": len(review["examples"]),
        "summary": review["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
