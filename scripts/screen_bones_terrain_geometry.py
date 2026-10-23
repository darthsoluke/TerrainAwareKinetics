"""Geometric screen of selected BONES-SEED clips (step 2 of terrain fitting).

Reads SOMA Uniform BVHs, converts them with the official Kimodo path already used
for the stair data (header check, 30 Hz SLERP resampling, SOMA77 FK, official
skin sole regions) and screens support levels, limb support and posture with
motionbricks.data.bones_terrain_screen. Source files and motion are never
modified and nothing is written next to the source.

CPU only. Examples:

    # Selection only; needs no Kimodo.
    python scripts/screen_bones_terrain_geometry.py --bvh-root /path/to/soma_uniform/bvh \
        --config configs/bones_terrain_screen_v1.json --output runs/screen_dry --dry-run

    # Pilot: two clips per stem.
    python scripts/screen_bones_terrain_geometry.py --bvh-root /path/to/soma_uniform/bvh \
        --config configs/bones_terrain_screen_v1.json --output runs/screen_pilot \
        --limit-per-stem 2 --workers 8
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motionbricks.data.bones_terrain_screen import (  # noqa: E402
    LANDMARKS, SCHEMA_VERSION, ScreenConfig, screen_clip, select_clips, summarize,
)
from motionbricks.data.stair_support import SupportConfig  # noqa: E402

FPS = 30.0
MIN_FRAMES = 16
_STATE = {}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from kimodo.skeleton.registry import build_skeleton
    from motionbricks.data.bones_seed import build_representation, official_sole_offsets, official_sole_surfaces
    from motionbricks.data.soma_skin_reference import load_official_skinner

    full = build_skeleton(77)
    names = list(full.bone_order_names)
    missing = [name for name in LANDMARKS if name not in names]
    if missing:
        raise RuntimeError(f"SOMA77 lacks landmarks {missing}")
    skeleton30 = build_representation().skeleton
    offsets, _ = official_sole_offsets(skeleton30)
    # Same reference as bones_seed.validate_planar_posture: neutral hips above the soles.
    reference = skeleton30.neutral_joints[skeleton30.foot_joint_idx] + offsets
    standing_hip = float(-reference[:, 1].median())
    surfaces, _, _, _ = official_sole_surfaces(skeleton30)
    _STATE.update(full=full, names=names, landmark_index={name: names.index(name) for name in LANDMARKS},
                  skinner=load_official_skinner(full), surfaces=surfaces,
                  standing_hip=standing_hip, np=np, torch=torch)


def convert(path, extra_names=()):
    """Official BVH -> 30 Hz SOMA77 joints and sole regions, in source world coordinates.

    Returns LANDMARKS plus any extra SOMA77 joint names that exist (for review drawings).
    """
    from kimodo.exports.bvh import bvh_to_kimodo_motion
    from kimodo.exports.motion_io import complete_motion_dict
    from kimodo.skeleton.bvh import Bvh, SkeletonBvh
    from motionbricks.data.bones_seed import resample_rotations_root
    from scripts.prepare_native_mb_stairs import _skin_sole_regions, validate_header

    np, torch, full = _STATE["np"], _STATE["torch"], _STATE["full"]
    bvh = Bvh(Path(path).read_text(encoding="utf-8", errors="strict"), backend="np")
    validate_header(bvh, full)
    wrapper = np.asarray(bvh.frames_joint_channels("Root", bvh.joint_channels("Root")))
    if wrapper.size and np.max(np.abs(wrapper)) > 1e-5:
        raise ValueError("Animated Root wrapper cannot be silently discarded")
    parsed = SkeletonBvh()
    parsed.load_from_bvh(str(path), exclude_bones={"Root"}, mocap=bvh)
    neutral = torch.as_tensor(parsed.get_neutral_joints() * 0.01)
    neutral -= neutral[0].clone()
    geometry_error = float((neutral - full.bvh_neutral_joints.cpu()).norm(dim=-1).max())
    if geometry_error > 0.002:
        raise ValueError(f"BVH reference geometry differs from official SOMA77 ({geometry_error:.4f} m)")
    del bvh, parsed
    native, source_fps = bvh_to_kimodo_motion(path, full)
    rotations, root = resample_rotations_root(native["local_rot_mats"].cpu().numpy().astype(np.float32),
                                              native["root_positions"].cpu().numpy().astype(np.float32),
                                              source_fps, FPS)
    if len(root) < MIN_FRAMES:
        raise ValueError(f"Fewer than {MIN_FRAMES} frames at {FPS} Hz")
    motion = complete_motion_dict(torch.from_numpy(rotations).double(), torch.from_numpy(root).double(), full, FPS)
    posed = motion["posed_joints"].cpu().numpy().astype(np.float32)
    soles, _, parity = _skin_sole_regions(rotations, posed, _STATE["skinner"], _STATE["surfaces"])
    joints = {name: posed[:, index].astype(np.float64) for name, index in _STATE["landmark_index"].items()}
    for name in extra_names:
        if name in _STATE["names"] and name not in joints:
            joints[name] = posed[:, _STATE["names"].index(name)].astype(np.float64)
    return soles, joints, root.astype(np.float64), float(source_fps), parity


def work(item):
    started = time.perf_counter()
    base = {key: item[key] for key in ("filename", "stem", "group")}
    try:
        base["source_sha256"] = sha256_file(item["path"])
        soles, joints, root, source_fps, parity = convert(item["path"])
        row, _ = screen_clip(soles, joints, root, FPS, _STATE["standing_hip"],
                             dimension_tokens=item["dimension_tokens"])
        row.update(base, source_fps=source_fps, sole_lbs_parity_m=parity,
                   standing_hip_height_m=_STATE["standing_hip"])
    except Exception as error:  # report every failure; never skip silently
        row = dict(base, error=f"{type(error).__name__}: {error}",
                   traceback=traceback.format_exc(limit=4))
    row["seconds"] = round(time.perf_counter() - started, 3)
    return row


COMPACT_COLUMNS = ("filename", "stem", "group", "status", "duration_seconds", "level_count", "level_span_m",
                   "levels_m", "flags", "standing", "crouch", "quadruped", "prone", "other_low",
                   "palm_seconds_at_level", "knee_seconds_at_level", "head_clearance_p5_by_posture",
                   "dimension_level_span_error_m", "dimension_clearance_minus_size_m", "error")


def write_compact(rows, path):
    """One small line per clip for review; the full rows stay in clips.jsonl."""
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\t".join(COMPACT_COLUMNS) + "\n")
        for row in sorted(rows, key=lambda r: (r["group"], r["stem"], r["filename"])):
            if "error" in row:
                values = dict(filename=row["filename"], stem=row["stem"], group=row["group"], status="failed",
                              error=row["error"].replace("\t", " ").replace("\n", " ")[:200])
            else:
                fraction = row["posture"]["fraction"]
                values = dict(
                    filename=row["filename"], stem=row["stem"], group=row["group"], status="ok",
                    duration_seconds=row["duration_seconds"], level_count=row["support"]["level_count"],
                    level_span_m=row["support"]["level_span_m"], levels_m=json.dumps(row["support"]["levels_m"]),
                    flags=",".join(row["flags"]), **{name: fraction[name] for name in fraction},
                    palm_seconds_at_level=row["limb_support"]["palm_seconds_at_level"],
                    knee_seconds_at_level=row["limb_support"]["knee_seconds_at_level"],
                    head_clearance_p5_by_posture=json.dumps({k: v["p5"] for k, v in row["head_clearance_m"].items()}),
                    dimension_level_span_error_m=json.dumps([d["level_span_error_m"] for d in row["dimensions"]]),
                    dimension_clearance_minus_size_m=json.dumps(
                        [d["lowest_body_clearance_minus_size_m"] for d in row["dimensions"]]))
            stream.write("\t".join(str(values.get(column, "")) for column in COMPACT_COLUMNS) + "\n")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bvh-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bones_terrain_screen_v1.json")
    parser.add_argument("--output", type=Path, required=True, help="New directory (must not exist)")
    parser.add_argument("--groups", nargs="*", help="Subset of config groups")
    parser.add_argument("--limit-per-stem", type=int, help="Cap clips per stem (pilot runs)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--dry-run", action="store_true", help="Write the selection only")
    args = parser.parse_args(argv)
    if args.limit_per_stem is not None and args.limit_per_stem < 1:
        parser.error("--limit-per-stem must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = sorted(args.bvh_root.rglob("*.bvh"))
    selected, selection = select_clips(paths, config, groups=set(args.groups) if args.groups else None,
                                       limit_per_stem=args.limit_per_stem)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "selection.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row) + "\n")
    report = {
        "schema_version": SCHEMA_VERSION, "git_commit": git_commit(),
        "config": {"path": str(args.config), "sha256": sha256_file(args.config), "version": config.get("version")},
        "bvh_root": str(args.bvh_root), "bvh_files_found": len(paths),
        "arguments": {"groups": args.groups, "limit_per_stem": args.limit_per_stem, "workers": args.workers},
        "selection": selection,
        "thresholds": {"screen": ScreenConfig().__dict__, "support": SupportConfig(require_stairs=True).__dict__},
        "protocol": {"fps": FPS, "coordinates": "source world, Y up, metres", "source_changed": False,
                     "conversion": "Kimodo BVH -> 30 Hz SLERP -> SOMA77 FK; official skin sole regions (98 vertices per foot)",
                     "status": "screen evidence for selecting fitting inputs; not fitted terrain"},
        "module_sha256": {name: sha256_file(ROOT / name) for name in (
            "motionbricks/data/bones_terrain_screen.py", "motionbricks/data/stair_support.py",
            "scripts/screen_bones_terrain_geometry.py")},
    }
    print(json.dumps({"selected": selection["selected"], "per_group": selection["per_group"],
                      "configured_stems_without_files": selection["configured_stems_without_files"]}))
    if args.dry_run:
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    started = time.time()
    rows = []
    context = mp.get_context("spawn")
    with context.Pool(args.workers, initializer=_init_worker) as pool, \
            (args.output / "clips.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(pool.imap_unordered(work, selected, chunksize=4), 1):
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            if index % 100 == 0 or index == len(selected):
                failed = sum("error" in r for r in rows)
                print(f"{index}/{len(selected)} screened, {failed} failed, {time.time() - started:.0f}s", flush=True)
    by_stem, by_group = summarize(rows)
    write_compact(rows, args.output / "clips_compact.tsv")
    errors = {}
    for row in rows:
        if "error" in row:
            key = row["error"].split(":")[0] + ": " + row["error"].split(":", 1)[-1].strip()[:80]
            errors[key] = errors.get(key, 0) + 1
    report.update(elapsed_seconds=round(time.time() - started, 1), screened=len(rows),
                  failed=sum("error" in r for r in rows), failure_reasons=errors,
                  standing_hip_height_m=sorted({r["standing_hip_height_m"] for r in rows
                                                if "standing_hip_height_m" in r}))
    (args.output / "by_stem.json").write_text(json.dumps(by_stem, indent=2), encoding="utf-8")
    (args.output / "by_group.json").write_text(json.dumps(by_group, indent=2), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(by_group, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
