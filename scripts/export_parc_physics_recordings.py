"""Export hash-verified PARC physics recordings to the current MS file format."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import sys
import tempfile
from typing import Any, Dict, Mapping, Tuple


SCHEMA = "parc.physics_recording_export.v1"
PROVENANCE_KEY = "parc_physics_export"
HASH_LENGTH = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != HASH_LENGTH
            or any(char not in "0123456789abcdef" for char in value.lower())):
        raise ValueError("%s must be a 64-character SHA-256 hex digest" % label)
    return value.lower()


def _resolve_inside(root: Path, relative: Any, label: str) -> Tuple[Path, str]:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ValueError("%s must be a nonempty relative path" % label)
    normalized = relative.replace("\\", "/")
    posix = PurePosixPath(normalized)
    if posix.is_absolute() or PureWindowsPath(relative).is_absolute():
        raise ValueError("%s must be relative to its root" % label)
    if any(part in ("", ".", "..") for part in posix.parts):
        raise ValueError("%s contains an unsafe path component" % label)

    root = Path(root).resolve()
    path = root.joinpath(*posix.parts)
    if path.is_symlink():
        raise ValueError("%s must not be a symbolic link" % label)
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("%s resolves outside its root" % label)
    if not resolved.is_file():
        raise FileNotFoundError("%s does not exist: %s" % (label, resolved))
    return resolved, posix.as_posix()


def _read_completed_report(run: Path) -> Tuple[Dict[str, Any], Path, str]:
    run = Path(run).resolve()
    report_path = run / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError("run/report.json was not found: %s" % report_path)
    report_hash = sha256_file(report_path)
    with report_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise ValueError("run report must be a JSON object")
    if report.get("status") != "official_recording_finished":
        raise ValueError("run report is not a completed official recording")
    if report.get("physics_recording_completed") is not True:
        raise ValueError("run report does not confirm physics recording completion")
    if report.get("inputs_unchanged_after_run") is not True:
        raise ValueError("run report does not confirm unchanged bundle inputs")
    terrain = report.get("packed_terrain_validation", {})
    if (not terrain.get("grid_spacing_matches") or not terrain.get("all_tile_cells_checked")
            or terrain.get("maximum_height_error_m", float("inf")) > 1e-5):
        raise ValueError("run terrain/collision pairing has not been validated")
    if not isinstance(report.get("saved"), list):
        raise ValueError("run report saved field must be a list")
    if not isinstance(report.get("inputs"), list):
        raise ValueError("run report inputs field must be a list")
    return report, report_path, report_hash


def _source_inputs(report: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for row in report["inputs"]:
        if not isinstance(row, dict):
            raise ValueError("run report input rows must be objects")
        key = row.get("candidate_key")
        if not isinstance(key, str) or not key:
            raise ValueError("run report input is missing candidate_key")
        if key in result:
            raise ValueError("run report has duplicate input candidate_key: %s" % key)
        result[key] = row
    return result


def _prepare_numpy_pickle_compatibility() -> None:
    import numpy as np

    if int(np.__version__.split(".", 1)[0]) < 2:
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def _device_type(device: Any) -> str:
    return str(getattr(device, "type", device))


def _load_official_converter(bundle: Path):
    """Load the vendored converter and character model without initializing CUDA."""
    bundle = Path(bundle).resolve()
    parc_root = bundle / "PARC"
    package_init = parc_root / "PARC" / "__init__.py"
    converter_path = parc_root / "scripts" / "convert_legacy_motions.py"
    character_path = parc_root / "data" / "assets" / "humanoid.xml"
    for path in (package_init, converter_path, character_path):
        if not path.is_file():
            raise FileNotFoundError("Official PARC file is missing: %s" % path)

    _prepare_numpy_pickle_compatibility()
    parc_root_text = str(parc_root)
    if parc_root_text not in sys.path:
        sys.path.insert(0, parc_root_text)
    package = importlib.import_module("PARC")
    if Path(package.__file__).resolve() != package_init.resolve():
        raise RuntimeError("Imported PARC package does not belong to --bundle")

    lowercase_package = sys.modules.get("parc")
    if lowercase_package is not None:
        package_file = getattr(lowercase_package, "__file__", None)
        if package_file is None or Path(package_file).resolve() != package_init.resolve():
            raise RuntimeError("A different parc package is already loaded")
    else:
        sys.modules["parc"] = package

    module_name = "_parc_bundle_convert_legacy_motions"
    existing = sys.modules.get(module_name)
    if existing is not None and Path(existing.__file__).resolve() != converter_path.resolve():
        raise RuntimeError("A different PARC legacy converter is already loaded")
    if existing is None:
        spec = importlib.util.spec_from_file_location(module_name, str(converter_path))
        if spec is None or spec.loader is None:
            raise ImportError("Could not load official PARC converter: %s" % converter_path)
        existing = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = existing
        spec.loader.exec_module(existing)

    character = existing.kin_char_model.KinCharModel(device="cpu")
    character.load_char_file(str(character_path))
    if _device_type(character._device) != "cpu":
        raise RuntimeError("PARC character model must be loaded on CPU")
    return existing, character, character_path


def _as_numpy(value: Any, label: str):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array):
        raise ValueError("%s must be a real numeric array" % label)
    return array


def _max_abs_difference(left: Any, right: Any) -> float:
    import numpy as np

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return float("inf")
    if left_array.size == 0:
        return 0.0
    return float(np.max(np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))))


def _same_value(left: Any, right: Any) -> bool:
    import numpy as np

    if hasattr(left, "detach"):
        left = left.detach().cpu().numpy()
    if hasattr(right, "detach"):
        right = right.detach().cpu().numpy()
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (set(left.keys()) == set(right.keys()) and
                all(_same_value(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right))
        except (TypeError, ValueError):
            return False
    try:
        result = left == right
        return bool(result) if not hasattr(result, "all") else bool(result.all())
    except (TypeError, ValueError):
        return False


def _validate_saved_row(row: Any) -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("run report saved rows must be objects")
    required = ("candidate_key", "source_sha256", "sha256", "path", "frames", "fps",
                "start_fraction", "whole_motion")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError("saved recording is missing fields: %s" % ", ".join(missing))
    if not isinstance(row["candidate_key"], str) or not row["candidate_key"]:
        raise ValueError("saved recording candidate_key must be nonempty")
    _require_hash(row["source_sha256"], "saved source_sha256")
    _require_hash(row["sha256"], "saved sha256")
    if row.get("recorded_state") != "simulated":
        raise ValueError("saved recording is not marked as a simulated state")
    if row.get("recorded_terrain_error_m", float("inf")) > 1e-5:
        raise ValueError("recorded terrain has not been validated in world coordinates")
    if isinstance(row["frames"], bool) or not isinstance(row["frames"], int) or row["frames"] <= 0:
        raise ValueError("saved recording frames must be a positive integer")
    if isinstance(row["fps"], bool) or not isinstance(row["fps"], (int, float)):
        raise ValueError("saved recording fps must be numeric")
    if not math.isfinite(float(row["fps"])) or row["fps"] <= 0:
        raise ValueError("saved recording fps must be finite and positive")
    fraction = row["start_fraction"]
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction)) or not 0 <= fraction < 1):
        raise ValueError("saved recording start_fraction must be in [0, 1)")
    if not isinstance(row["whole_motion"], bool):
        raise ValueError("saved recording whole_motion must be boolean")
    return dict(row)


def _validate_legacy(raw_data: Mapping[str, Any], saved: Mapping[str, Any], character: Any):
    import numpy as np

    if not isinstance(raw_data, Mapping):
        raise ValueError("legacy recording root must be a dictionary")
    frames = _as_numpy(raw_data.get("frames"), "legacy frames").astype(np.float32, copy=False)
    contacts = _as_numpy(raw_data.get("contacts"), "legacy contacts").astype(np.float32, copy=False)
    if frames.ndim != 2 or frames.shape[0] != saved["frames"]:
        raise ValueError("legacy frames do not match the saved frame count")
    if frames.shape[1] != 6 + int(character.get_dof_size()):
        raise ValueError("legacy frame width does not match the official KinCharModel DOF size")
    if int(character.get_dof_size()) != 28 or int(character.get_num_non_root_joints()) != 14:
        raise ValueError("official humanoid must expose 28 DOFs and 14 non-root joints")
    if int(character.get_num_bodies()) != 15 or int(character.get_num_contact_bodies()) != 15:
        raise ValueError("official humanoid must expose 15 bodies/contact channels")
    if contacts.shape != (len(frames), int(character.get_num_contact_bodies())):
        raise ValueError("legacy contacts do not match frames and official contact bodies")
    if not np.isfinite(frames).all() or not np.isfinite(contacts).all():
        raise ValueError("legacy motion and contact arrays must be finite")
    fps = raw_data.get("fps")
    if (isinstance(fps, bool) or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps)) or float(fps) != float(saved["fps"])):
        raise ValueError("legacy fps does not match run report")
    if "terrain" not in raw_data or raw_data["terrain"] is None:
        raise ValueError("legacy physics recording is missing its localized terrain")
    terrain = raw_data["terrain"]
    for field in ("hf", "hf_maxmin", "min_point", "dxdy"):
        if not hasattr(terrain, field):
            raise ValueError("legacy terrain is missing %s" % field)
    hf = _as_numpy(terrain.hf, "terrain hf").astype(np.float32, copy=False)
    hf_maxmin = _as_numpy(terrain.hf_maxmin, "terrain hf_maxmin").astype(np.float32, copy=False)
    min_point = _as_numpy(terrain.min_point, "terrain min_point").astype(np.float32, copy=False)
    dxdy = _as_numpy(terrain.dxdy, "terrain dxdy").astype(np.float32, copy=False)
    if (hf.ndim != 2 or min(hf.shape) < 1 or hf_maxmin.shape != hf.shape + (2,)
            or min_point.shape != (2,) or dxdy.shape != (2,)):
        raise ValueError("legacy localized terrain has unexpected dimensions")
    if not all(np.isfinite(value).all() for value in (hf, hf_maxmin, min_point, dxdy)):
        raise ValueError("legacy localized terrain must be finite")
    if dxdy[0] <= 0 or not np.isclose(dxdy[0], dxdy[1], rtol=0, atol=1e-8):
        raise ValueError("current PARC MS terrain requires equal positive x/y spacing")
    return frames, contacts, hf, hf_maxmin, min_point, dxdy


def _build_converted_file(raw_data: Dict[str, Any], saved: Dict[str, Any], source: Dict[str, Any],
                          recording_rel: str, report_hash: str, converter_hash: str, converter: Any,
                          character: Any):
    import numpy as np

    frames, contacts, hf, hf_maxmin, min_point, dxdy = _validate_legacy(
        raw_data, saved, character)
    if "path_nodes" in raw_data:
        raise ValueError("physics recorder input unexpectedly contains path_nodes")

    motion_data = converter.motion_edit_lib.MotionData(copy.deepcopy(raw_data), device="cpu")
    converted = converter.motion_to_ms_file(raw_data, motion_data, character)
    motion = converted.motion_data
    terrain = converted.terrain_data
    if motion is None or terrain is None:
        raise ValueError("official converter did not produce motion_data and terrain_data")

    torch = converter.torch
    root_expmap = torch.as_tensor(frames[:, 3:6], dtype=torch.float32, device="cpu")
    dof = torch.as_tensor(frames[:, 6:], dtype=torch.float32, device="cpu")
    expected_root_rot = converter.torch_util.quat_pos(
        converter.torch_util.exp_map_to_quat(root_expmap))
    expected_joint_rot = converter.torch_util.quat_pos(character.dof_to_rot(dof))
    root_pos_error = _max_abs_difference(motion.root_pos, frames[:, 0:3])
    root_rot_error = _max_abs_difference(motion.root_rot, expected_root_rot)
    joint_rot_error = _max_abs_difference(motion.joint_rot, expected_joint_rot)
    converted_contacts = _as_numpy(motion.body_contacts, "converted body_contacts")
    terrain_hf_error = _max_abs_difference(terrain.hf, hf)
    terrain_bounds_error = _max_abs_difference(terrain.hf_maxmin, hf_maxmin)
    terrain_origin_error = _max_abs_difference(terrain.min_point, min_point)
    terrain_dx_error = abs(float(terrain.dx) - float(dxdy[0]))
    parity = {
        "root_pos_max_abs": root_pos_error,
        "root_pos_exact": root_pos_error == 0.0,
        "root_rot_vs_official_expmap_max_abs": root_rot_error,
        "root_rot_vs_official_expmap_exact": root_rot_error == 0.0,
        "joint_rot_vs_official_dof_max_abs": joint_rot_error,
        "joint_rot_vs_official_dof_exact": joint_rot_error == 0.0,
        "body_contacts_exact": np.array_equal(converted_contacts, contacts),
        "terrain_hf_max_abs": terrain_hf_error,
        "terrain_hf_exact": terrain_hf_error == 0.0,
        "terrain_hf_maxmin_max_abs": terrain_bounds_error,
        "terrain_hf_maxmin_exact": terrain_bounds_error == 0.0,
        "terrain_min_point_max_abs": terrain_origin_error,
        "terrain_min_point_exact": terrain_origin_error == 0.0,
        "terrain_dx_abs": terrain_dx_error,
        "terrain_dx_exact": terrain_dx_error == 0.0,
    }
    if not all((parity["root_pos_exact"], parity["root_rot_vs_official_expmap_exact"],
                parity["joint_rot_vs_official_dof_exact"], parity["body_contacts_exact"],
                parity["terrain_hf_exact"], parity["terrain_hf_maxmin_exact"],
                parity["terrain_min_point_exact"], parity["terrain_dx_exact"])):
        raise ValueError("official conversion changed source pose, contacts, or localized terrain")

    misc_data = dict(converted.misc_data or {})
    if "path_nodes" in misc_data:
        raise ValueError("official conversion unexpectedly supplied path_nodes")
    for key in ("obs", "obs_shapes"):
        if key in raw_data and (key not in misc_data or
                                not _same_value(misc_data[key], converter._convert_misc_value(raw_data[key]))):
            raise ValueError("official conversion did not preserve legacy %s" % key)
    parity["observations_preserved"] = all(
        key not in raw_data or key in misc_data for key in ("obs", "obs_shapes"))
    parity["path_nodes_absent"] = True

    provenance = {
        "schema": SCHEMA,
        "recorded_state": "simulated",
        "candidate_key": saved["candidate_key"],
        "source_sha256": saved["source_sha256"],
        "source_candidate_path": source["path"],
        "recording_sha256": saved["sha256"],
        "recording_path": recording_rel,
        "start_fraction": float(saved["start_fraction"]),
        "phase": {"start_fraction": float(saved["start_fraction"])},
        "whole_motion": saved["whole_motion"],
        "frames": int(saved["frames"]),
        "fps": float(saved["fps"]),
        "run_report_sha256": report_hash,
        "conversion": "PARC scripts/convert_legacy_motions.py:motion_to_ms_file",
        "converter_sha256": converter_hash,
        "device": "cpu",
        "conversion_parity": parity,
    }
    misc_data[PROVENANCE_KEY] = provenance
    converted.misc_data = misc_data
    return converted, parity


def _assert_roundtrip(expected: Any, actual: Any) -> None:
    import numpy as np

    for field in ("root_pos", "root_rot", "joint_rot", "body_contacts"):
        if not np.array_equal(_as_numpy(getattr(expected.motion_data, field), field),
                              _as_numpy(getattr(actual.motion_data, field), field)):
            raise ValueError("nested MS roundtrip changed motion_data.%s" % field)
    for field in ("hf", "hf_maxmin", "min_point"):
        if not np.array_equal(_as_numpy(getattr(expected.terrain_data, field), field),
                              _as_numpy(getattr(actual.terrain_data, field), field)):
            raise ValueError("nested MS roundtrip changed terrain_data.%s" % field)
    if float(expected.terrain_data.dx) != float(actual.terrain_data.dx):
        raise ValueError("nested MS roundtrip changed terrain spacing")
    if not _same_value(expected.misc_data, actual.misc_data):
        raise ValueError("nested MS roundtrip changed misc_data")
    if actual.misc_data is None or PROVENANCE_KEY not in actual.misc_data:
        raise ValueError("nested MS file is missing per-recording provenance")
    if "path_nodes" in actual.misc_data:
        raise ValueError("nested MS file must not contain path_nodes")


def export_run(bundle: Path, run: Path, output: Path) -> Dict[str, Any]:
    import numpy as np

    bundle, run = Path(bundle).resolve(), Path(run).resolve()
    output = Path(output).resolve()
    report, report_path, report_hash = _read_completed_report(run)
    sources = _source_inputs(report)
    converter, character, character_path = _load_official_converter(bundle)
    converter_path = bundle / "PARC" / "scripts" / "convert_legacy_motions.py"
    converter_hash = sha256_file(converter_path)

    if output.exists():
        raise FileExistsError("output already exists; refusing to overwrite: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=str(output.parent)))
    recordings = []
    source_snapshots = []
    raw_snapshots = []
    try:
        (staging / "recordings").mkdir()
        for index, raw_row in enumerate(report["saved"]):
            saved = _validate_saved_row(raw_row)
            candidate = sources.get(saved["candidate_key"])
            if candidate is None:
                raise ValueError("saved candidate is absent from run report inputs")
            source_hash = _require_hash(candidate.get("sha256"), "input candidate sha256")
            if source_hash != saved["source_sha256"].lower():
                raise ValueError("saved source_sha256 does not match run report input hash")
            source_path, source_rel = _resolve_inside(bundle, candidate.get("path"),
                                                      "source candidate path")
            actual_source_hash = sha256_file(source_path)
            if actual_source_hash != source_hash:
                raise ValueError("source candidate hash mismatch: %s" % source_rel)

            recording_path, recording_rel = _resolve_inside(run, saved["path"],
                                                             "saved recording path")
            actual_recording_hash = sha256_file(recording_path)
            if actual_recording_hash != saved["sha256"].lower():
                raise ValueError("saved recording hash mismatch: %s" % recording_rel)
            source_snapshots.append((source_path, source_hash))
            raw_snapshots.append((recording_path, saved["sha256"].lower()))

            raw_data = converter.motion_edit_lib.load_motion_file(
                str(recording_path), convert_to_class=False)
            if not isinstance(raw_data, dict):
                raise ValueError("legacy physics recording root must be a dictionary")
            converted, parity = _build_converted_file(
                raw_data, saved, candidate, recording_rel, report_hash, converter_hash,
                converter, character)
            converted_rel = Path("recordings") / ("%04d_%s" % (index, recording_path.name))
            converted_path = staging / converted_rel
            converter.file_io.save_ms_file(converted, str(converted_path))
            reloaded = converter.file_io.load_ms_file(str(converted_path))
            _assert_roundtrip(converted, reloaded)

            if sha256_file(recording_path) != saved["sha256"].lower():
                raise ValueError("raw physical recording changed during conversion")
            recordings.append(dict(saved, source_candidate_path=source_rel,
                                   converted_path=converted_rel.as_posix(),
                                   converted_sha256=sha256_file(converted_path),
                                   conversion_parity=parity))

        for source_path, expected_hash in source_snapshots:
            if sha256_file(source_path) != expected_hash:
                raise ValueError("source candidate changed during conversion: %s" % source_path)
        for raw_path, expected_hash in raw_snapshots:
            if sha256_file(raw_path) != expected_hash:
                raise ValueError("raw physics recording changed during conversion: %s" % raw_path)
        if sha256_file(converter_path) != converter_hash:
            raise ValueError("supplemental official converter changed during export")
        if sha256_file(report_path) != report_hash:
            raise ValueError("run report changed during export")

        manifest = {
            "schema": SCHEMA,
            "status": "exported_all_reported_physics_recordings",
            "run_report": "report.json",
            "run_report_sha256": report_hash,
            "run_status": report["status"],
            "physics_recording_completed": True,
            "official_parc_root": str((bundle / "PARC").resolve()),
            "official_character_xml": str(character_path.resolve()),
            "official_converter_sha256": converter_hash,
            "conversion": "PARC scripts/convert_legacy_motions.py:motion_to_ms_file",
            "conversion_device": "cpu",
            "recordings": recordings,
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(str(staging), str(output))
        manifest["manifest_path"] = str(output / "manifest.json")
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(str(staging))
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="PARC bundle root containing PARC/ and source candidate files")
    parser.add_argument("--run", type=Path, required=True,
                        help="completed physics-recording run directory containing report.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory; existing directories are never overwritten")
    args = parser.parse_args(argv)
    manifest = export_run(args.bundle, args.run, args.output)
    print(json.dumps({"manifest": manifest["manifest_path"],
                      "recordings": len(manifest["recordings"]),
                      "whole_motion": sum(bool(row["whole_motion"])
                                           for row in manifest["recordings"]),
                      "partial_motion": sum(not bool(row["whole_motion"])
                                             for row in manifest["recordings"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
