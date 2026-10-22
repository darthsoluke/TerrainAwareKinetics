"""Build an explicitly admitted 30 Hz terrain-aware native MotionBricks dataset.

Admissions JSON schema (paths are relative to that JSON unless absolute)::

    {"status":"passed", "retarget_status":"passed",
     "motion_terrain_pairing_status":"passed",
     "physics_retest_status":"passed",
     "engine_origin_measurement_status":"passed", "items":[{
       "source_id":"clip-001", "status":"passed",
       "source_path":".../sample.pkl", "source_sha256":"...",
       "retarget_path":".../retarget.npz", "retarget_sha256":"...",
       "intent_path":".../intent.npz", "intent_sha256":"...",
       "physics_report_path":".../physics/report.json",
       "physics_report_sha256":"...", "export_manifest_path":".../export/manifest.json",
       "export_manifest_sha256":"...", "physics_candidate_key":"clip-001"}]}

Split JSON has one row per source_id with explicit source_group, terrain_family,
terrain_seed, and split (train/validation/test). intent.npz contains the online
command_world_xy already aligned to post-cleanup frames, source_frame_indices
identifying their original source frames, and fps=30. Commands must not be
indexed a second time. The export manifest binds the simulated raw recording to
the official converted pickle consumed by prepare_parc_soma.
"""

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np
import torch

# Use the repository-pinned official Kimodo implementation when not installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "out/vendor/kimodo"))

from kimodo.exports.motion_io import complete_motion_dict
from kimodo.skeleton.definitions import SOMASkeleton30 as KimodoSOMA30

from motionbricks.data.lafan_soma_retarget import Y_TO_Z
from motionbricks.data.local_terrain_observation import (
    extract_local_terrain_observation,
)
from motionbricks.data.native_mb_data import (
    CONTRACT_VERSION,
    FPS,
    REPRESENTATION_NAME,
    build_native_representation,
    build_train_stats,
)
from motionbricks.data.native_mb_terrain_protocol import (
    TERRAIN_PROTOCOL,
    terrain_observation_config,
)


MIN_TRAINABLE_FRAMES = 65
OBSERVATION_CHUNK_FRAMES = 512
SPLITS = {"train", "validation", "test"}
_ADMISSION_FIELDS = (
    "source_id", "source_path", "source_sha256", "retarget_path", "retarget_sha256",
    "intent_path", "intent_sha256", "physics_report_path", "physics_report_sha256",
    "export_manifest_path", "export_manifest_sha256", "physics_candidate_key",
)

_EXPORT_PARITY_FIELDS = (
    "root_pos_exact", "root_rot_vs_official_expmap_exact",
    "joint_rot_vs_official_dof_exact", "body_contacts_exact", "terrain_hf_exact",
    "terrain_hf_maxmin_exact", "terrain_min_point_exact", "terrain_dx_exact",
    "observations_preserved", "path_nodes_absent",
)

TERRAIN_ARCHITECTURE_STATUS = {
    "stage": "pilot",
    "final_architecture": False,
    "active_protocol": TERRAIN_PROTOCOL["version"],
    "conditioning_location": "opt-in Root/Pose local-geometry branches",
    "intent_encoding": "online direction selects a bounded local terrain patch in current root-yaw coordinates",
    "tokenizer": "official native MotionBricks tokenizer and motion representation are unchanged",
    "terrain_conditioned_decoder": "not implemented",
    "open_architecture_reviews": [
        "spatial/visual encoding of intent direction and local geometry",
        "whether and how terrain tokens should enter the tokenizer/decoder",
    ],
    "training_performed_by_dataset_builder": False,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _resolve_path(base, value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Admission field {field} must be a nonempty path")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _validate_physics_retest(report_path, report_hash, export_manifest_path,
                             export_manifest_hash, candidate_key, source_path, source_hash):
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing full physics retest report: {report_path}")
    if not isinstance(report_hash, str) or _sha256(report_path) != report_hash:
        raise ValueError("Physics retest report SHA256 mismatch")
    if not export_manifest_path.is_file():
        raise FileNotFoundError(f"Missing official physics export manifest: {export_manifest_path}")
    if not isinstance(export_manifest_hash, str) or _sha256(export_manifest_path) != export_manifest_hash:
        raise ValueError("Physics export manifest SHA256 mismatch")
    report = _json(report_path)
    export_manifest = _json(export_manifest_path)
    if not isinstance(report, dict) or not isinstance(export_manifest, dict):
        raise ValueError("Physics report and export manifest must be JSON objects")
    if (report.get("status") != "official_recording_finished" or
            report.get("physics_recording_completed") is not True or
            report.get("inputs_unchanged_after_run") is not True):
        raise ValueError("Physics report is not a completed, unchanged full recording run")
    if (export_manifest.get("schema") != "parc.physics_recording_export.v1" or
            export_manifest.get("status") != "exported_all_reported_physics_recordings" or
            export_manifest.get("run_status") != report.get("status") or
            export_manifest.get("physics_recording_completed") is not True or
            export_manifest.get("run_report_sha256") != report_hash or
            export_manifest.get("conversion") != "PARC scripts/convert_legacy_motions.py:motion_to_ms_file"):
        raise ValueError("Official export manifest does not bind this completed physics report and conversion")

    alignment = report.get("engine_origin_alignment", {})
    after = alignment.get("after", {}) if isinstance(alignment, dict) else {}
    contract = report.get("engine_origin_contract", {})
    if (not isinstance(alignment, dict) or not isinstance(contract, dict) or
            alignment.get("source") != "gym.get_env_origin" or
            alignment.get("applied_before_first_reset") is not True or
            alignment.get("shared_buffer_preserved") is not True or
            not isinstance(after, dict)):
        raise ValueError("Physics retest lacks a measured, applied, pre-reset engine-origin alignment")
    try:
        after_error = float(after["max_error_m"])
        contract_error = float(contract["max_error_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Physics retest has invalid engine-origin error measurements") from error
    if (not np.isfinite(after_error) or not np.isfinite(contract_error) or
            after_error > 1e-5 or contract_error > 1e-5):
        raise ValueError("Physics retest lacks a measured, applied, pre-reset engine-origin alignment")
    origins = np.asarray(contract.get("engine_origins"), dtype=np.float64)
    offsets = np.asarray(contract.get("observation_offsets"), dtype=np.float64)
    if (origins.ndim != 2 or origins.shape[1:] != (3,) or offsets.shape != origins.shape or
            not len(origins) or not np.isfinite(origins).all() or not np.isfinite(offsets).all() or
            np.max(np.abs(origins - offsets)) > 1e-5):
        raise ValueError("Physics retest engine origins do not match recorded observation offsets")
    after_origins = np.asarray(after.get("engine_origins"), dtype=np.float64)
    after_offsets = np.asarray(after.get("observation_offsets"), dtype=np.float64)
    if (after_origins.shape != origins.shape or after_offsets.shape != offsets.shape or
            not np.isfinite(after_origins).all() or not np.isfinite(after_offsets).all() or
            np.max(np.abs(after_origins - origins)) > 1e-5 or
            np.max(np.abs(after_offsets - offsets)) > 1e-5):
        raise ValueError("Final physics origin alignment differs from the measured engine-origin contract")

    terrain = report.get("packed_terrain_validation", {})
    if not isinstance(terrain, dict) or terrain.get("grid_spacing_matches") is not True:
        raise ValueError("Physics retest has not validated packed terrain against collision terrain")
    try:
        cells_checked = int(terrain["all_tile_cells_checked"])
        terrain_error = float(terrain["maximum_height_error_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Physics retest has invalid packed terrain measurements") from error
    if cells_checked <= 0 or not np.isfinite(terrain_error) or terrain_error > 1e-5:
        raise ValueError("Physics retest has not validated packed terrain against collision terrain")
    input_rows = report.get("inputs")
    saved_rows = report.get("saved")
    if not isinstance(input_rows, list) or not isinstance(saved_rows, list):
        raise ValueError("Physics retest is missing its input or saved-recording inventory")
    inputs = [row for row in input_rows
              if isinstance(row, dict) and row.get("candidate_key") == candidate_key]
    saved = [row for row in saved_rows
             if isinstance(row, dict) and row.get("candidate_key") == candidate_key]
    if len(inputs) != 1 or len(saved) != 1:
        raise ValueError(f"Physics report must contain exactly one input and recording for {candidate_key}")
    input_row, saved_row = inputs[0], saved[0]
    recordings = [row for row in export_manifest.get("recordings", [])
                  if isinstance(row, dict) and row.get("candidate_key") == candidate_key]
    if len(recordings) != 1:
        raise ValueError(f"Export manifest must contain exactly one recording for {candidate_key}")
    recording = recordings[0]
    try:
        frames = int(input_row["source_frames"])
        start_fraction = float(saved_row["start_fraction"])
        recorded_frames = int(saved_row["frames"])
        recorded_source_frames = int(saved_row["source_frames"])
        recorded_fps = float(saved_row["fps"])
        recorded_terrain_error = float(saved_row["recorded_terrain_error_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Physics recording for {candidate_key} has incomplete metrics") from error
    converted_path = Path(recording.get("converted_path", ""))
    if not converted_path.is_absolute():
        converted_path = export_manifest_path.parent / converted_path
    parity = recording.get("conversion_parity", {})
    if (frames < 2 or not isinstance(input_row.get("sha256"), str) or
            saved_row.get("source_sha256") != input_row.get("sha256") or
            recording.get("source_sha256") != saved_row.get("source_sha256") or
            recording.get("sha256") != saved_row.get("sha256") or
            recording.get("converted_sha256") != source_hash or
            not converted_path.is_file() or converted_path.resolve() != source_path.resolve() or
            _sha256(converted_path) != source_hash or
            not isinstance(parity, dict) or
            any(parity.get(field) is not True for field in _EXPORT_PARITY_FIELDS) or
            saved_row.get("recorded_state") != "simulated" or
            saved_row.get("whole_motion") is not True or
            not np.isfinite([start_fraction, recorded_fps, recorded_terrain_error]).all() or
            start_fraction != 0.0 or recorded_frames != frames or
            recorded_source_frames != frames or recorded_fps != FPS or
            recorded_terrain_error > 1e-5 or
            int(recording.get("frames", -1)) != frames or
            int(recording.get("source_frames", -1)) != frames or
            float(recording.get("fps", 0)) != FPS or
            recording.get("whole_motion") is not True or
            float(recording.get("start_fraction", -1)) != 0.0):
        raise ValueError(f"Physics recording/export conversion for {candidate_key} is partial, stale, or not hash-linked to the admitted source")
    return report, export_manifest


def _matrix_is_rotation(matrices, name, atol=2e-3):
    matrices = np.asarray(matrices)
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError(f"{name} must contain finite 3x3 matrices")
    identity = np.eye(3, dtype=np.float64)
    gram = matrices.astype(np.float64).swapaxes(-1, -2) @ matrices.astype(np.float64)
    if np.max(np.abs(gram - identity)) > atol or np.min(np.linalg.det(matrices)) < 1.0 - atol:
        raise ValueError(f"{name} contains a non-rotation matrix")


def parc_heightfield_to_native(heightfield_z_m, min_point_xy_m, dx_m):
    """Convert PARC [source-X,source-Y] Z-up terrain using the shared Y_TO_Z basis.

    The target grid is [Z,X], with target +Z corresponding to PARC -Y. The
    returned origin is the center of target cell [0,0]; heights remain target Y.
    """
    heightfield = np.asarray(heightfield_z_m, dtype=np.float64)
    origin_xy = np.asarray(min_point_xy_m, dtype=np.float64)
    dx = float(dx_m)
    if (heightfield.ndim != 2 or min(heightfield.shape) < 1 or
            not np.isfinite(heightfield).all() or origin_xy.shape != (2,) or
            not np.isfinite(origin_xy).all() or not np.isfinite(dx) or dx <= 0):
        raise ValueError("Invalid PARC terrain grid, origin, or spacing")

    # PARC source cell [x=0,y=ny-1] maps to the first target [z=0,x=0] cell.
    nx, ny = heightfield.shape
    source_first_cell = np.array([origin_xy[0], origin_xy[1] + (ny - 1) * dx, 0.0])
    origin_native = Y_TO_Z.T @ source_first_cell
    native_heightfield = np.ascontiguousarray(heightfield.T[::-1, :], dtype=np.float32)
    if native_heightfield.shape != (ny, nx):
        raise AssertionError("PARC heightfield axis conversion changed dimensions unexpectedly")
    return native_heightfield, origin_native.astype(np.float32), dx


def parc_world_command_to_native_local_xz(command_world_xy, root_heading_yaw):
    """Map recorded PARC world XY intent through Y_TO_Z, then into current root yaw."""
    command = np.asarray(command_world_xy, dtype=np.float64)
    yaw = np.asarray(root_heading_yaw, dtype=np.float64)
    if command.ndim != 2 or command.shape[1] != 2 or not np.isfinite(command).all():
        raise ValueError("command_world_xy must be finite [T,2] in original PARC z-up XY")
    if yaw.shape != (len(command),) or not np.isfinite(yaw).all():
        raise ValueError("root heading must be finite [T]")

    # Column source vector [dx,dy,0] -> target = Y_TO_Z.T @ source.
    source_xyz = np.column_stack((command, np.zeros(len(command))))
    native_xyz = source_xyz @ Y_TO_Z
    world_x, world_z = native_xyz[:, 0], native_xyz[:, 2]
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local = np.column_stack((cosine * world_x - sine * world_z,
                             sine * world_x + cosine * world_z))
    return local.astype(np.float32)


def _yaw_from_global_root_rotations(global_rotations):
    rotations = np.asarray(global_rotations)
    if rotations.ndim != 4 or rotations.shape[1:] != (30, 3, 3):
        raise ValueError("target_global_rot_mats must have shape [T,30,3,3]")
    _matrix_is_rotation(rotations, "target_global_rot_mats")
    # SOMA forward is local +Z; project it into the Y-up world XZ plane.
    forward = rotations[:, 0, :, 2]
    projected = np.linalg.norm(forward[:, [0, 2]], axis=-1)
    if np.any(projected < 1e-5):
        raise ValueError("Root orientation has no stable projected +Z heading")
    return np.arctan2(forward[:, 0], forward[:, 2])


def _load_json_inputs(admissions_path, splits_path):
    admissions_path, splits_path = Path(admissions_path).resolve(), Path(splits_path).resolve()
    admissions = _json(admissions_path)
    split_manifest = _json(splits_path)
    if not isinstance(admissions, dict) or admissions.get("status") != "passed":
        raise ValueError("Admissions JSON must explicitly declare status='passed'")
    for key in ("retarget_status", "motion_terrain_pairing_status",
                "physics_retest_status", "engine_origin_measurement_status"):
        if admissions.get(key) != "passed":
            raise ValueError(f"Admissions JSON must explicitly declare {key}='passed'")
    admission_rows = admissions.get("items")
    split_rows = split_manifest.get("items") if isinstance(split_manifest, dict) else None
    if not isinstance(admission_rows, list) or not admission_rows:
        raise ValueError("Admissions JSON needs a nonempty items list")
    if not isinstance(split_rows, list) or not split_rows:
        raise ValueError("Split JSON needs a nonempty explicit items list")

    records = {}
    for row in admission_rows:
        if not isinstance(row, dict) or row.get("status") != "passed":
            raise ValueError("Every admission item must explicitly declare status='passed'")
        missing = set(_ADMISSION_FIELDS) - row.keys()
        if missing:
            raise ValueError(f"Admission row is missing fields: {sorted(missing)}")
        source_id = row["source_id"]
        if not isinstance(source_id, str) or not source_id.strip() or source_id in records:
            raise ValueError("Admission source_id values must be unique nonempty strings")
        paths = {key: _resolve_path(admissions_path.parent, row[path_key], path_key)
                 for key, path_key in (("source", "source_path"),
                                       ("retarget", "retarget_path"),
                                       ("intent", "intent_path"),
                                       ("physics_report", "physics_report_path"),
                                       ("export_manifest", "export_manifest_path"))}
        hashes = {name: row[f"{name}_sha256"]
                  for name in ("source", "retarget", "intent", "physics_report", "export_manifest")}
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Admission {name} file does not exist: {path}")
            if not isinstance(hashes[name], str) or _sha256(path) != hashes[name]:
                raise ValueError(f"Admission {name} SHA256 mismatch for {source_id}")
        candidate_key = row["physics_candidate_key"]
        if not isinstance(candidate_key, str) or not candidate_key.strip():
            raise ValueError(f"Admission physics_candidate_key must be nonempty for {source_id}")
        physics_report, export_manifest = _validate_physics_retest(
            paths["physics_report"], hashes["physics_report"],
            paths["export_manifest"], hashes["export_manifest"], candidate_key,
            paths["source"], hashes["source"])
        report_path = _resolve_path(paths["retarget"].parent, row.get("report_path", "report.json"), "report_path")
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing prepare_parc_soma report: {report_path}")
        report = _json(report_path)
        if report.get("execution_status") != "processed" or report.get("screen_status") != "pass":
            raise ValueError(f"Retarget screen is not passed for {source_id}; explicit admission cannot bypass a failed screen")
        if report.get("shard_sha256") != hashes["retarget"]:
            raise ValueError(f"Retarget report does not bind the admitted shard for {source_id}")
        report_source = report.get("source")
        if not isinstance(report_source, dict) or report_source.get("sha256") != hashes["source"]:
            raise ValueError(f"Retarget report does not bind the admitted source for {source_id}")
        records[source_id] = {"admission": row, "paths": paths,
                              "hashes": hashes, "report_path": report_path,
                              "report": report, "physics_report": physics_report,
                              "export_manifest": export_manifest}

    splits = {}
    for row in split_rows:
        if not isinstance(row, dict):
            raise ValueError("Split rows must be objects")
        source_id = row.get("source_id")
        required = ("source_group", "terrain_family", "terrain_seed", "split")
        if source_id not in records or source_id in splits:
            raise ValueError("Split source_ids must uniquely match admitted source_ids")
        if any(not isinstance(row.get(key), (str, int)) or not str(row[key]).strip()
               for key in required[:-1]):
            raise ValueError("Each split row needs explicit source_group, terrain_family, and terrain_seed")
        if row.get("split") not in SPLITS:
            raise ValueError(f"Unknown split for {source_id}: {row.get('split')!r}")
        splits[source_id] = {key: str(row[key]) if key != "split" else row[key]
                             for key in required}
    if set(splits) != set(records):
        raise ValueError("Split JSON must contain exactly one row for every admitted source_id")

    for identity_key in ("source_group", "terrain_family", "terrain_seed"):
        owner = {}
        for source_id, row in splits.items():
            identity = row[identity_key]
            split = row["split"]
            if identity in owner and owner[identity] != split:
                raise ValueError(f"{identity_key} {identity!r} leaks across splits")
            owner[identity] = split
    return admissions_path, splits_path, admissions, split_manifest, records, splits


def _load_clip(source_id, record, split):
    path = record["paths"]["retarget"]
    report = record["report"]
    with np.load(path, allow_pickle=False) as data:
        required = {"local_rot_mats", "root_positions", "target_names", "target_parents",
                    "target_global_rot_mats", "hf", "min_point", "dx",
                    "source_contacts", "source_names", "source_fps"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Retarget shard {path} lacks {sorted(missing)}")
        arrays = {key: data[key].copy() for key in required}
        scale_key = "scale_scene" if "scale_scene" in data.files else "uniform_scale"
        if scale_key not in data.files:
            raise ValueError(f"Retarget shard {path} lacks scale_scene/uniform_scale")
        scene_scale = float(np.asarray(data[scale_key]).item())

    local = np.asarray(arrays["local_rot_mats"], dtype=np.float32)
    root = np.asarray(arrays["root_positions"], dtype=np.float32)
    global_rotations = np.asarray(arrays["target_global_rot_mats"], dtype=np.float32)
    contacts = np.asarray(arrays["source_contacts"], dtype=np.float32)
    source_names = np.asarray(arrays["source_names"]).astype(str)
    target_names = np.asarray(arrays["target_names"]).astype(str)
    parents = np.asarray(arrays["target_parents"], dtype=np.int64)
    fps = float(np.asarray(arrays["source_fps"]).item())
    frames = len(local)
    if not np.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError(f"Invalid scene scale in {path}")
    if fps != FPS:
        raise ValueError(f"{source_id} must be full 30 FPS; got {fps:g}")
    if frames < MIN_TRAINABLE_FRAMES:
        raise ValueError(f"{source_id} is too short for native training: {frames} < {MIN_TRAINABLE_FRAMES}")
    if (local.shape != (frames, 30, 3, 3) or root.shape != (frames, 3) or
            global_rotations.shape != (frames, 30, 3, 3) or contacts.shape != (frames, 15)):
        raise ValueError(f"Invalid SOMA/contact frame schema in {path}")
    if (not np.isfinite(local).all() or not np.isfinite(root).all() or
            not np.isfinite(global_rotations).all() or not np.isfinite(contacts).all()):
        raise ValueError(f"Nonfinite retarget data in {path}")
    _matrix_is_rotation(local, "local_rot_mats")
    _matrix_is_rotation(global_rotations, "target_global_rot_mats")
    if len(source_names) != 15 or contacts.shape[1] != len(source_names):
        raise ValueError("All 15 PARC source contact channels and their names are required")

    from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
    skeleton = SOMASkeleton30()
    expected_names = np.asarray(skeleton.bone_order_names).astype(str)
    expected_parents = skeleton.joint_parents.cpu().numpy().astype(np.int64)
    if not np.array_equal(target_names, expected_names) or not np.array_equal(parents, expected_parents):
        raise ValueError(f"Retarget target skeleton does not match official SOMASkeleton30: {path}")
    with torch.no_grad():
        fk_rotations, fk_positions, _ = skeleton.fk(torch.from_numpy(local), torch.from_numpy(root))
    fk_rotations = fk_rotations.cpu().numpy()
    fk_positions = fk_positions.cpu().numpy()
    if np.max(np.abs(fk_rotations - global_rotations)) > 2e-3:
        raise ValueError(f"Retarget global rotations disagree with official SOMA FK: {path}")
    root_index = int(skeleton.root_idx)
    if np.max(np.abs(fk_positions[:, root_index] - root)) > 2e-4:
        raise ValueError(f"Retarget root positions disagree with official SOMA FK: {path}")
    if (report.get("fps") != FPS or report.get("frames") != frames or
            report.get("source_frames") != frames):
        raise ValueError(f"{source_id} is not a full, untrimmed prepare_parc_soma 30 Hz retarget")

    with np.load(record["paths"]["intent"], allow_pickle=False) as data:
        if not {"command_world_xy", "source_frame_indices", "fps"}.issubset(data.files):
            raise ValueError("intent.npz needs aligned command_world_xy, source_frame_indices, and fps")
        commands = np.asarray(data["command_world_xy"], dtype=np.float32).copy()
        source_frame_indices = np.asarray(data["source_frame_indices"]).copy()
        intent_fps = float(np.asarray(data["fps"]).item())
    if (commands.shape != (frames, 2) or not np.isfinite(commands).all() or
            source_frame_indices.shape != (frames,) or
            not np.issubdtype(source_frame_indices.dtype, np.integer) or intent_fps != FPS):
        raise ValueError(f"Invalid online intent contract for {source_id}")
    source_frame_indices = source_frame_indices.astype(np.int64)
    if (source_frame_indices < 0).any() or (np.diff(source_frame_indices) <= 0).any():
        raise ValueError(f"source_frame_indices must be nonnegative and strictly increasing for {source_id}")

    hf, min_point, dx = parc_heightfield_to_native(
        arrays["hf"], arrays["min_point"], float(np.asarray(arrays["dx"]).item()))
    yaw = _yaw_from_global_root_rotations(global_rotations)
    local_commands = parc_world_command_to_native_local_xz(commands, yaw)
    return {
        "source_id": source_id, "split": split, "local_rot_mats": local,
        "root_positions": root, "target_global_rot_mats": global_rotations,
        "source_contacts": contacts, "source_contact_names": source_names,
        "source_frame_indices": source_frame_indices, "command_world_xy": commands,
        "command_local_xz": local_commands, "heightfield_y_m": hf,
        "min_point_world_xyz_m": min_point, "dx_m": dx,
        "scene_scale": scene_scale, "frames": frames,
    }


def _build_terrain_observations(clip):
    config = terrain_observation_config()
    points, valid = [], []
    for start in range(0, clip["frames"], OBSERVATION_CHUNK_FRAMES):
        end = min(start + OBSERVATION_CHUNK_FRAMES, clip["frames"])
        observation = extract_local_terrain_observation(
            clip["heightfield_y_m"], clip["min_point_world_xyz_m"], clip["dx_m"],
            clip["root_positions"][start:end],
            _yaw_from_global_root_rotations(clip["target_global_rot_mats"][start:end]),
            clip["command_local_xz"][start:end], config=config,
        )
        points.append(observation.points.cpu().numpy())
        valid.append(observation.valid.cpu().numpy())
    terrain_points = np.concatenate(points).astype(np.float32)
    terrain_valid = np.concatenate(valid).astype(np.bool_)
    if (terrain_points.shape != (clip["frames"], *TERRAIN_PROTOCOL["patch_shape"]) or
            terrain_valid.shape != terrain_points.shape[:-1]):
        raise ValueError("Generated terrain observations violate the native MB frame contract")
    if not np.isfinite(terrain_points).all():
        raise ValueError("Generated terrain observations contain nonfinite values")
    return terrain_points, terrain_valid


def _terrain_relative_foot_contacts(posed_joints, skeleton, *, clip_heightfield_y_m,
                                    clip_origin_xyz_m, dx_m):
    """Use official MB velocity/height thresholds with terrain-relative foot height."""
    from motionbricks.motionlib.core.motion_reps.tools.feet import foot_detect_from_pos_and_vel
    from motionbricks.motionlib.core.motion_reps.tools.motion_features import compute_vel_xyz

    positions = torch.as_tensor(posed_joints)
    heightfield = torch.as_tensor(clip_heightfield_y_m, device=positions.device,
                                  dtype=positions.dtype)
    origin = torch.as_tensor(clip_origin_xyz_m, device=positions.device,
                             dtype=positions.dtype)
    dx = float(dx_m)
    foot_indices = list(skeleton.left_foot_joint_idx) + list(skeleton.right_foot_joint_idx)
    if (positions.ndim != 3 or positions.shape[1:] != (skeleton.nbjoints, 3) or
            len(foot_indices) != 4 or heightfield.ndim != 2 or origin.shape != (3,) or
            not torch.isfinite(positions).all() or not torch.isfinite(origin).all() or
            not np.isfinite(dx) or dx <= 0):
        raise ValueError("Invalid positions, official SOMA foot joints, or terrain geometry for MB contacts")

    foot_positions = positions[:, foot_indices]
    cell_x = torch.floor((foot_positions[..., 0] - origin[0]) / dx + 0.5).long()
    cell_z = torch.floor((foot_positions[..., 2] - origin[2]) / dx + 0.5).long()
    inside = ((cell_x >= 0) & (cell_x < heightfield.shape[1]) &
              (cell_z >= 0) & (cell_z < heightfield.shape[0]))
    safe_x = cell_x.clamp(0, heightfield.shape[1] - 1)
    safe_z = cell_z.clamp(0, heightfield.shape[0] - 1)
    ground_y = heightfield[safe_z, safe_x]
    known = inside & torch.isfinite(ground_y)
    if not known.all():
        missing = int((~known).sum().item())
        raise ValueError(f"Unknown terrain under {missing} SOMA foot samples; refusing false-clearance labels")

    terrain_relative_positions = positions.unsqueeze(0).clone()
    terrain_relative_positions[0, :, foot_indices, 1] = foot_positions[..., 1] - ground_y
    lengths = torch.tensor([len(positions)], device=positions.device, dtype=torch.long)
    world_velocity = compute_vel_xyz(positions.unsqueeze(0), FPS, lengths=lengths)
    left, right = foot_detect_from_pos_and_vel(
        terrain_relative_positions, world_velocity, skeleton, 0.15, 0.10)
    contacts = torch.cat((left, right), dim=-1).squeeze(0)
    if contacts.shape != (len(positions), 4) or not torch.isfinite(contacts).all():
        raise ValueError("Official terrain-relative MB detector returned invalid four-channel contacts")
    return contacts


def _native_encode(local, root, representation, clip, *, return_frame=False):
    local_t = torch.from_numpy(local.copy())
    root_t = torch.from_numpy(root.copy())
    motion = complete_motion_dict(local_t, root_t, representation.skeleton, FPS)
    # Replace Kimodo's global-floor heuristic only for this terrain adapter.
    foot_contacts = _terrain_relative_foot_contacts(
        motion["posed_joints"], representation.skeleton,
        clip_heightfield_y_m=clip["heightfield_y_m"],
        clip_origin_xyz_m=clip["min_point_world_xyz_m"], dx_m=clip["dx_m"],
    )
    dual, initial_transform = representation.dual_rep({
        "posed_joints": motion["posed_joints"],
        "global_joint_rots": motion["global_rot_mats"],
        "local_joint_rots": motion["local_rot_mats"],
        "translation": motion["root_positions"],
        "foot_contacts": foot_contacts,
    }, to_normalize=False, return_init_heading_info=True)
    global_features = representation.dual_rep.get_feature_subset(dual, mode="global")
    if not torch.isfinite(dual).all() or len(dual) != len(local):
        raise ValueError("Official MotionBricks encoding produced invalid features")
    roundtrip = representation.inverse(
        global_features, is_normalized=False, init_heading_info=initial_transform,
        return_all=True, return_quat=False,
    )["posed_joints"]
    error = (roundtrip - motion["posed_joints"]).norm(dim=-1).max().item()
    if error > 0.002:
        raise ValueError(f"Official MotionBricks feature/FK roundtrip mismatch: {error:.6f} m")
    canonical = representation.inverse(
        global_features, is_normalized=False, return_all=True, return_quat=False)
    local_out = canonical["local_joint_rots"].cpu().numpy().astype(np.float32)
    root_out = canonical["posed_joints"][:, representation.skeleton.root_idx].cpu().numpy().astype(np.float32)
    result = (global_features.cpu().numpy().astype(np.float32),
            dual.cpu().numpy().astype(np.float32), local_out, root_out, float(error),
            foot_contacts.cpu().numpy().astype(np.float32))
    if return_frame:
        frame = {key: value.detach().cpu().numpy().astype(np.float32)
                 for key, value in initial_transform.items()}
        return (*result, frame)
    return result


def _slug(source_id):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_id).strip("._-")[:72] or "clip"
    return f"{stem}-{hashlib.sha256(source_id.encode()).hexdigest()[:10]}"


def _code_hashes():
    root = Path(__file__).resolve().parents[1]
    package = root / "motionbricks"
    official = package / "motionlib/core/motion_reps"
    names = {
        "dual_root_global_joints": official / "dual_root_global_joints.py",
        "dual_root_local_body": official / "motion_reps_base/dual_root_local_body.py",
        "global_root_local_body": official / "motion_reps_base/global_root_local_body.py",
        "separated_root_local_body": official / "motion_reps_base/seperate_root_local_body.py",
        "motion_features": official / "tools/motion_features.py",
        "official_contact_features": official / "tools/feet.py",
        "soma_skeleton_adapter": package / "motionlib/core/skeletons/soma.py",
        "native_mb_data": package / "data/native_mb_data.py",
        "terrain_protocol": package / "data/native_mb_terrain_protocol.py",
        "terrain_observation": package / "data/local_terrain_observation.py",
        "parc_soma_retarget": package / "data/parc_soma_retarget.py",
        "coordinate_adapter": package / "data/lafan_soma_retarget.py",
        "local_terrain_encoder_pilot": package / "motion_backbone/neural_modules/local_terrain_encoder.py",
        "root_backbone_terrain_pilot": package / "motion_backbone/neural_modules/root_backbone.py",
        "pose_backbone_terrain_pilot": package / "motion_backbone/neural_modules/pose_backbone.py",
    }
    result = {name: _sha256(path) for name, path in names.items()}
    result["converter"] = _sha256(__file__)
    result["kimodo_motion_io"] = _sha256(Path(inspect.getsourcefile(complete_motion_dict)))
    result["kimodo_soma30"] = _sha256(Path(inspect.getsourcefile(KimodoSOMA30)))
    return result


def _verify_reuse_stats(source):
    source = Path(source).resolve()
    manifest_path = source / "native_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("--reuse-stats-from must be a verified native dataset directory")
    manifest = _json(manifest_path)
    contract = manifest.get("contract", {})
    if contract.get("version") != CONTRACT_VERSION or contract.get("representation") != REPRESENTATION_NAME:
        raise ValueError("Normalization source uses an incompatible native representation")
    if manifest.get("terrain_conditioning"):
        raise ValueError("Normalization reuse source must be a flat native dataset")
    if contract.get("target_fps") != FPS:
        raise ValueError("Normalization source is not 30 Hz")
    stats_contract = _json(source / "stats/contract.json")
    if stats_contract.get("split") != "train":
        raise ValueError("Normalization source statistics are not explicitly train-only")
    if manifest.get("stats_contract", {}).get("split") != "train":
        raise ValueError("Normalization source manifest does not bind train-only statistics")
    artifacts = manifest.get("artifact_sha256", {})
    accepted_path = source / "accepted.json"
    if not accepted_path.is_file() or artifacts.get("accepted.json") != _sha256(accepted_path):
        raise ValueError("Normalization source does not hash-bind accepted.json")
    accepted = _json(accepted_path)
    if not isinstance(accepted, list) or not any(row.get("split") == "train" for row in accepted):
        raise ValueError("Normalization source has no explicit train split")
    paths = {name: source / "stats" / name for name in ("mean.npy", "std.npy")}
    for name, path in paths.items():
        relative = f"stats/{name}"
        if not path.is_file() or artifacts.get(relative) != _sha256(path):
            raise ValueError(f"Normalization source does not hash-bind {relative}")
    mean, std = (np.load(paths[name], allow_pickle=False) for name in ("mean.npy", "std.npy"))
    if (mean.ndim != 1 or std.shape != mean.shape or not np.isfinite(mean).all() or
            not np.isfinite(std).all() or (std < 0).any()):
        raise ValueError("Normalization source has invalid mean/std arrays")
    build_native_representation(source / "stats")
    return source, manifest, stats_contract, paths


def build_dataset(admissions_path, splits_path, output, *, reuse_stats_from=None):
    (admissions_path, splits_path, admissions, split_manifest,
     admission_records, splits) = _load_json_inputs(admissions_path, splits_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    if any(output == record["paths"][key] or output in record["paths"][key].parents
           or record["paths"][key] in output.parents
           or record["paths"][key].parent in output.parents
           for record in admission_records.values()
           for key in ("source", "retarget", "intent", "physics_report", "export_manifest")):
        raise ValueError("Output must be separate from and outside every source input")

    # Validate and fully load every item before creating any output directory.
    clips = {}
    for source_id, record in admission_records.items():
        clips[source_id] = _load_clip(source_id, record, splits[source_id])

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        shard_dir, contact_dir, intent_dir, physics_dir = (
            staging / "shards", staging / "source_contacts", staging / "source_intent",
            staging / "physics_reports")
        shard_dir.mkdir()
        contact_dir.mkdir()
        intent_dir.mkdir()
        physics_dir.mkdir()
        representation = build_native_representation()
        accepted, artifact_hashes, acceptance_items = [], {}, []
        for source_id, clip in clips.items():
            source_record = admission_records[source_id]
            (global_features, dual_features, saved_local, saved_root, roundtrip_error,
             mb_foot_contacts) = _native_encode(
                clip["local_rot_mats"], clip["root_positions"], representation, clip)
            terrain_points, terrain_valid = _build_terrain_observations(clip)
            stem = _slug(source_id)
            shard_rel = Path("shards") / f"{stem}_native30.npz"
            contact_rel = Path("source_contacts") / f"{stem}_parc15.npz"
            intent_rel = Path("source_intent") / f"{stem}_online_intent.npz"
            physics_rel = Path("physics_reports") / f"{stem}_full_retest.json"
            export_rel = Path("physics_reports") / f"{stem}_export_manifest.json"
            shard_path = staging / shard_rel
            np.savez_compressed(
                shard_path,
                global_features=global_features,
                dual_features=dual_features,
                local_rot_mats=saved_local,
                root_positions=saved_root,
                terrain_points=terrain_points,
                terrain_valid=terrain_valid,
                mb_foot_contacts=mb_foot_contacts,
                fps=np.asarray(FPS, dtype=np.float32),
                feature_fk_roundtrip_error_m=np.asarray(roundtrip_error, dtype=np.float32),
            )
            contact_path = staging / contact_rel
            np.savez_compressed(
                contact_path,
                source_contacts=clip["source_contacts"].astype(np.float32),
                source_contact_names=clip["source_contact_names"].astype(str),
                source_frame_indices=clip["source_frame_indices"].astype(np.int64),
                fps=np.asarray(FPS, dtype=np.float32),
            )
            intent_path = staging / intent_rel
            np.savez_compressed(
                intent_path,
                command_world_xy=clip["command_world_xy"].astype(np.float32),
                source_frame_indices=clip["source_frame_indices"].astype(np.int64),
                fps=np.asarray(FPS, dtype=np.float32),
                command_coordinates=np.asarray("original PARC z-up world XY"),
            )
            physics_snapshot = staging / physics_rel
            shutil.copy2(source_record["paths"]["physics_report"], physics_snapshot)
            physics_snapshot_hash = _sha256(physics_snapshot)
            if physics_snapshot_hash != source_record["hashes"]["physics_report"]:
                raise ValueError(f"Physics report changed while building {source_id}")
            export_snapshot = staging / export_rel
            shutil.copy2(source_record["paths"]["export_manifest"], export_snapshot)
            export_snapshot_hash = _sha256(export_snapshot)
            if export_snapshot_hash != source_record["hashes"]["export_manifest"]:
                raise ValueError(f"Physics export manifest changed while building {source_id}")
            shard_hash, contact_hash, intent_hash = (
                _sha256(shard_path), _sha256(contact_path), _sha256(intent_path))
            artifact_hashes[shard_rel.as_posix()] = shard_hash
            artifact_hashes[contact_rel.as_posix()] = contact_hash
            artifact_hashes[intent_rel.as_posix()] = intent_hash
            artifact_hashes[physics_rel.as_posix()] = physics_snapshot_hash
            artifact_hashes[export_rel.as_posix()] = export_snapshot_hash
            row_split = splits[source_id]
            record = {
                "filename": source_record["paths"]["source"].name,
                "recording_group": row_split["source_group"],
                "behavior": source_record["admission"].get("behavior", "mixed_terrain"),
                "split": row_split["split"],
                "source_id": source_id,
                "source_group": row_split["source_group"],
                "terrain_family": row_split["terrain_family"],
                "terrain_seed": row_split["terrain_seed"],
                "source_shard": str(source_record["paths"]["retarget"]),
                "source_shard_sha256": source_record["hashes"]["retarget"],
                "source_original_path": str(source_record["paths"]["source"]),
                "source_original_sha256": source_record["hashes"]["source"],
                "intent_sidecar": str(source_record["paths"]["intent"]),
                "intent_sidecar_sha256": source_record["hashes"]["intent"],
                "physics_report_snapshot": physics_rel.as_posix(),
                "physics_report_sha256": physics_snapshot_hash,
                "physics_export_manifest_snapshot": export_rel.as_posix(),
                "physics_export_manifest_sha256": export_snapshot_hash,
                "physics_candidate_key": source_record["admission"]["physics_candidate_key"],
                "source_contact_sidecar": contact_rel.as_posix(),
                "source_contact_sidecar_sha256": contact_hash,
                "source_intent_sidecar": intent_rel.as_posix(),
                "source_intent_sidecar_sha256": intent_hash,
                "source_contact_channels": 15,
                "source_contact_names": clip["source_contact_names"].tolist(),
                "source_contact_semantics": "PARC source labels retained verbatim; not MB loss channels or certified physical truth",
                "source_shard_fps": FPS,
                "source_original_fps": FPS,
                "shard": shard_rel.as_posix(),
                "converted_shard_sha256": shard_hash,
                "frames_30hz": clip["frames"],
                "duration_seconds": (clip["frames"] - 1) / FPS,
                "source_frame_indices_sha256": hashlib.sha256(
                    np.ascontiguousarray(clip["source_frame_indices"]).tobytes()).hexdigest(),
                "scene_scale": clip["scene_scale"],
                "terrain_observation_valid_fraction": float(terrain_valid.mean()),
                "feature_fk_roundtrip_error_m": roundtrip_error,
            }
            accepted.append(record)
            acceptance_items.append({
                "source_id": source_id, "status": "passed",
                "source_sha256": source_record["hashes"]["source"],
                "retarget_sha256": source_record["hashes"]["retarget"],
                "intent_sha256": source_record["hashes"]["intent"],
                "prepare_report_sha256": _sha256(source_record["report_path"]),
                "physics_report_sha256": source_record["hashes"]["physics_report"],
                "physics_export_manifest_sha256": source_record["hashes"]["export_manifest"],
                "physics_candidate_key": source_record["admission"]["physics_candidate_key"],
                "engine_origin_alignment_max_error_m": source_record["physics_report"][
                    "engine_origin_contract"]["max_error_m"],
                "split": row_split["split"], "source_group": row_split["source_group"],
                "terrain_family": row_split["terrain_family"],
                "terrain_seed": row_split["terrain_seed"],
                "frames_30hz": clip["frames"],
                "source_contact_channels_preserved": 15,
                "terrain_valid_fraction": record["terrain_observation_valid_fraction"],
            })

        accepted_path = staging / "accepted.json"
        accepted_path.write_text(json.dumps(accepted, indent=2), encoding="utf-8")
        accepted_hash = _sha256(accepted_path)
        artifact_hashes["accepted.json"] = accepted_hash
        shutil.copy2(admissions_path, staging / "admissions.json")
        shutil.copy2(splits_path, staging / "source_splits.json")
        artifact_hashes["admissions.json"] = _sha256(staging / "admissions.json")
        artifact_hashes["source_splits.json"] = _sha256(staging / "source_splits.json")

        reused_stats = None
        if reuse_stats_from is None:
            stats_contract = build_train_stats(staging, accepted, _sha256(admissions_path))
        else:
            source, source_manifest, source_stats_contract, source_paths = _verify_reuse_stats(reuse_stats_from)
            (staging / "stats").mkdir()
            for name, path in source_paths.items():
                shutil.copy2(path, staging / "stats" / name)
                artifact_hashes[f"stats/{name}"] = _sha256(staging / "stats" / name)
            reused_stats = {"source_dataset": str(source),
                            "source_manifest_sha256": _sha256(source / "native_manifest.json"),
                            "source_stats_contract": source_stats_contract,
                            "normalization_byte_identical": True}
            stats_contract = {
                **source_stats_contract,
                "normalization_source": str(source),
                "normalization_source_manifest_sha256": reused_stats["source_manifest_sha256"],
                "normalization_reuse": "verified byte-identical train-only flat native MB mean/std",
            }
            (staging / "stats/contract.json").write_text(
                json.dumps(stats_contract, indent=2), encoding="utf-8")
        for name in ("mean.npy", "std.npy"):
            relative = f"stats/{name}"
            artifact_hashes[relative] = _sha256(staging / relative)
        artifact_hashes["stats/contract.json"] = _sha256(staging / "stats/contract.json")
        stats_hashes = {name: artifact_hashes[f"stats/{name}"]
                        for name in ("mean.npy", "std.npy")}

        acceptance_report = {
            "schema": "parc-native-training-admission-v1",
            "status": "passed",
            "architecture_status": TERRAIN_ARCHITECTURE_STATUS,
            "retarget_status": admissions["retarget_status"],
            "motion_terrain_pairing_status": admissions["motion_terrain_pairing_status"],
            "physics_retest_status": admissions["physics_retest_status"],
            "engine_origin_measurement_status": admissions["engine_origin_measurement_status"],
            "physics_gate": "full official 30 Hz recording; measured Isaac Gym origins applied before reset; collision/observation terrain parity verified",
            "admissions_json_sha256": _sha256(admissions_path),
            "split_json_sha256": _sha256(splits_path),
            "admission_is_explicit": True,
            "retarget_screen_is_not_automatically_promoted": True,
            "coordinate_contract": {
                "source_basis": "PARC z-up world XY commands and [source-X,source-Y] heightfield",
                "shared_basis_matrix": "motionbricks.data.lafan_soma_retarget.Y_TO_Z",
                "terrain_conversion": "source hf [X,Y] -> native Y-up hf [Z,X] via transpose and source-Y reversal; terrain cells scaled/aligned as stored in retarget.npz",
                "command_conversion": "PARC [dx,dy,0] -> native via Y_TO_Z.T, then world-XZ -> current root-yaw-local XZ",
                "localization": "retarget root and fixed source terrain origin are already registered and scaled together; apply only the Y_TO_Z basis conversion, with no extra rigid localization or rotation",
                "command_provenance": "intent.npz recorded online before generation; no motion-derived or future direction",
            },
            "frame_contract": {
                "source_fps": FPS, "target_fps": FPS,
                "resampling": "none; full 30 FPS retarget frames are encoded in place",
                "source_frame_indices": "commands are already aligned by the correction writer; IDs are monotonic source-frame provenance and are never used to reindex commands",
                "minimum_frames": MIN_TRAINABLE_FRAMES,
            },
            "source_contact_contract": {
                "channels": 15, "preserved_verbatim": True,
                "storage": "source_contacts sidecars bound in native manifest",
                "used_as_mb_contact_loss": False,
                "hand_contact_loss_invented": False,
            },
            "terrain_aware_mb_contacts": {
                "detector": "official foot_detect_from_pos_and_vel",
                "channels": 4, "representation_and_model_dimensions_changed": False,
                "height_test": "official 0.10 m threshold applied to each foot joint height relative to its nearest known terrain cell",
                "velocity_test": "official 0.15 m/s threshold on actual 30 Hz world-space joint velocities; no terrain-relative velocity adjustment",
                "unknown_terrain": "reject the clip if any official foot joint has no in-bounds finite terrain cell",
                "scope": "PARC terrain adapter only; native flat-dataset builder is unchanged",
            },
            "split_isolation": {
                "keys": ["source_group", "terrain_family", "terrain_seed"],
                "policy": "each identity is assigned to exactly one explicit split",
                "split_counts": {split: sum(row["split"] == split for row in accepted)
                                 for split in sorted(SPLITS)},
            },
            "items": acceptance_items,
            "normalization_reuse": reused_stats,
        }
        report_path = staging / "acceptance_report.json"
        report_path.write_text(json.dumps(acceptance_report, indent=2), encoding="utf-8")
        artifact_hashes["acceptance_report.json"] = _sha256(report_path)

        code_hashes = _code_hashes()
        terrain_contract = {
            **TERRAIN_PROTOCOL,
            "architecture_status": TERRAIN_ARCHITECTURE_STATUS,
            "acceptance": {
                "retarget_status": admissions["retarget_status"],
                "motion_terrain_pairing_status": admissions["motion_terrain_pairing_status"],
                "physics_retest_status": admissions["physics_retest_status"],
                "engine_origin_measurement_status": admissions["engine_origin_measurement_status"],
                "report": "acceptance_report.json",
            },
        }
        manifest = {
            "architecture_status": TERRAIN_ARCHITECTURE_STATUS,
            "contract": {
                "version": CONTRACT_VERSION,
                "representation": REPRESENTATION_NAME,
                "source_manifest_sha256": _sha256(admissions_path),
                "split_manifest_sha256": _sha256(splits_path),
                "source_shard_fps": FPS,
                "target_fps": FPS,
                "resampling": "none; full-length official PARC->SOMA retarget is already 30 Hz",
                "features": "official MotionBricks SOMA30 dual_root_global_joints encoding recomputed from retarget rotations/root",
                "terrain_observations": "current retarget pose/yaw plus explicit online intent only; bounded local patches; no future map observations",
                "root_localization": "retarget root and fixed source terrain origin are already registered and scaled together; no additional rigid alignment",
                "statistics": "native MB train-only 65-frame windows with four-way yaw augmentation, or verified byte-identical train-only flat statistics",
                "custom_contact_representation_or_loss": "none; all 15 PARC channels preserved only in provenance sidecars",
                "code_sha256": code_hashes,
            },
            "terrain_conditioning": terrain_contract,
            "source_admissions": str(admissions_path),
            "source_splits": str(splits_path),
            "source_admissions_sha256": _sha256(admissions_path),
            "source_splits_sha256": _sha256(splits_path),
            "admissions_snapshot": "admissions.json",
            "split_snapshot": "source_splits.json",
            "accepted_json_sha256": accepted_hash,
            "stats_artifacts_sha256": stats_hashes,
            "artifact_sha256": artifact_hashes,
            "source_rows": len(admission_records),
            "converted_rows": len(accepted),
            "split_counts": acceptance_report["split_isolation"]["split_counts"],
            "stats_contract": stats_contract,
            "normalization_reuse": reused_stats,
            "limitations": [
                "PARC source contact channels are provenance only; native MB keeps its official four heuristic foot/toe representation.",
                "Heightfield observations cannot represent overhangs, vertical faces, ceilings, or handholds.",
                "Terrain observations are local height patches queried from the offline paired map using current pose and recorded command; they are not full-map or future-motion inputs.",
                "Explicit admissions and pairing status are required; a passed retarget screen alone never creates training approval.",
                "Physics inputs are admitted only when a hash-bound complete official recording proves measured engine-origin alignment before reset and terrain/collision parity; earlier physics outputs are not implicitly grandfathered.",
                "native-mb-local-terrain-v2 and its Root/Pose local-geometry encoder are a pilot, not a finalized intent-visual-encoding, tokenizer, or decoder architecture.",
            ],
        }
        (staging / "native_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _json(output / "native_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admissions", type=Path, required=True,
                        help="Passed admissions JSON binding source, retarget, intent and full-physics hashes")
    parser.add_argument("--splits", type=Path, required=True,
                        help="Explicit per-source split, source_group, terrain_family and terrain_seed JSON")
    parser.add_argument("--output", type=Path, required=True,
                        help="New versioned native dataset directory; existing paths are never overwritten")
    parser.add_argument("--reuse-stats-from", type=Path,
                        help="Optional verified 30 Hz native dataset with train-only byte-identical normalization")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(build_dataset(args.admissions, args.splits, args.output,
                                   reuse_stats_from=args.reuse_stats_from), indent=2))


if __name__ == "__main__":
    main()
