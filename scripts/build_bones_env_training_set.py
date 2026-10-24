"""Build the BONES motion-scene training set from a finished terrain-fit run (step 6).

Inputs: the fit run (fits/*.json from fit_bones_terrain_support.py), the motion cache it
used (--motion-cache, exact float32 conversion outputs) and the screen selection (paths
and metadata). For each clip whose scene is training-eligible it writes one shard:

    features [F,366]   native world SOMA30 frames, floor translated to y = 0
    contacts [F,4]     heel/toe contact against the fitted scene
    commands [F,4]     source-derived left/right stick commands; command_valid, stop_event
    box_* [V,K,...]    accepted scene variants, packed (K = 64), same translation

plus manifest.jsonl (one row per shard) and report.json (counts, seconds, failures by
group and split, FK roundtrip errors). Source motion is never edited; a clip whose
native-frame FK differs from the source FK by more than 2 mm is rejected.

    python scripts/build_bones_env_training_set.py --fit-run runs/bones_fit_v9_full \
        --motion-cache runs/bones_motion_cache_s2 --screen-run runs/bones_screen_v1_full \
        --output runs/bones_env_v1 --workers 32
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motionbricks.data.bones_env_dataset import (  # noqa: E402
    BOX_CAPACITY, actor_of, assign_split, contact_channels, pack_variants, roundtrip_error, scene_variants,
    soma30_motion, world_features,
)

SCHEMA = "bones-env-training-v1"
MAX_FK_ERROR_M = 0.002
_STATE = {}


def _init():
    import torch
    torch.set_num_threads(1)
    from motionbricks.data.bones_seed import build_representation
    _STATE["rep"] = build_representation()


def build_clip(job):
    item, fit_path, cache_path, output, options = job
    started = time.perf_counter()
    name = item["filename"]
    try:
        import torch
        from motionbricks.data.pfnn_full_training_data import input_intent_labels
        rep = _STATE["rep"]
        fit = json.loads(Path(fit_path).read_text())
        scene = fit.get("scene") or {}
        eligibility = scene.get("eligibility")
        if eligibility not in options["eligibility"]:
            return {"filename": name, "group": item["group"], "skipped": f"eligibility:{eligibility}"}
        variants = scene_variants(scene, include_geometry_only=eligibility != "environment_explained")
        if not variants:
            return {"filename": name, "group": item["group"], "skipped": "no_usable_variant"}
        with np.load(cache_path, allow_pickle=False) as data:
            rotations, root = data["rotations"], data["root"].astype(np.float64)
            sole_left, sole_right = data["sole_left"].astype(np.float64), data["sole_right"].astype(np.float64)
        ground = float(fit["ground_m"])
        # Contacts in the original world, against the fitted floor and support boxes.
        local30, posed30 = soma30_motion(rep, rotations, root)
        foot_index = np.asarray(rep.skeleton.foot_joint_idx)
        feet = posed30.numpy()[:, foot_index].astype(np.float64)
        contacts = contact_channels(sole_left, sole_right, feet, fit["boxes"]["minimal"])
        # One constant vertical translation of motion and scene: the fitted floor becomes y = 0.
        shifted = root.copy()
        shifted[:, 1] -= ground
        local30, posed30 = soma30_motion(rep, rotations, shifted)
        with torch.no_grad():
            features = world_features(rep, local30, shifted, contacts)
            error = roundtrip_error(rep, features, posed30)
        if error > MAX_FK_ERROR_M:
            return {"filename": name, "group": item["group"], "error": f"fk_roundtrip {error:.6f} m"}
        intent = input_intent_labels(features.numpy())
        frames = len(features)
        commands = np.full((frames, 4), np.nan, dtype=np.float32)
        valid = np.zeros(frames, dtype=bool)
        stop = np.zeros(frames, dtype=bool)
        at = intent["source_frames"].numpy()
        commands[at], valid[at], stop[at] = intent["command"].numpy(), True, intent["stop_event"].numpy()
        centers, half, rotation, mask = pack_variants([boxes for _, boxes in variants], -ground, BOX_CAPACITY)
        actor = actor_of(name)
        shard_rel = f"shards/{name}.npz"
        np.savez(Path(output) / shard_rel, features=features.numpy().astype(np.float32), contacts=contacts,
                 commands=commands, command_valid=valid, stop_event=stop,
                 box_centers=centers.astype(np.float32), box_half=half.astype(np.float32),
                 box_rotation=rotation.astype(np.float32), box_mask=mask,
                 variant_index=np.array([i for i, _ in variants]), ground_translation_m=np.array(-ground))
        return {"filename": name, "group": item["group"], "stem": item["stem"], "actor": actor,
                "split": assign_split(actor), "eligibility": eligibility, "shard": shard_rel,
                "frames": int(frames), "seconds": round((frames - 1) / 30, 3), "variants": len(variants),
                "boxes_max": int(mask.sum(axis=1).max()), "fk_roundtrip_m": round(error, 7),
                "contact_rate": [round(float(v), 4) for v in contacts.mean(axis=0)],
                "command_valid_frames": int(valid.sum()), "ground_translation_m": -ground,
                "build_seconds": round(time.perf_counter() - started, 3)}
    except Exception as error:   # noqa: BLE001 - recorded per clip
        return {"filename": name, "group": item.get("group"), "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=4)}


def summarize(results):
    groups = {}
    for row in results:
        entry = groups.setdefault(row.get("group") or "unknown", {"clips": 0, "built": 0, "skipped": {},
                                                                    "errors": 0, "seconds": {}, "variants": 0})
        entry["clips"] += 1
        if "error" in row:
            entry["errors"] += 1
        elif "skipped" in row:
            entry["skipped"][row["skipped"]] = entry["skipped"].get(row["skipped"], 0) + 1
        else:
            entry["built"] += 1
            entry["variants"] += row["variants"]
            entry["seconds"][row["split"]] = round(entry["seconds"].get(row["split"], 0.0) + row["seconds"], 2)
    built = [r for r in results if "shard" in r]
    splits = {}
    for row in built:
        slot = splits.setdefault(row["split"], {"clips": 0, "seconds": 0.0, "actors": set()})
        slot["clips"] += 1
        slot["seconds"] += row["seconds"]
        slot["actors"].add(row["actor"])
    return {"groups": dict(sorted(groups.items())),
            "splits": {k: {"clips": v["clips"], "seconds": round(v["seconds"], 1), "actors": len(v["actors"])}
                       for k, v in sorted(splits.items())},
            "fk_roundtrip_max_m": max((r["fk_roundtrip_m"] for r in built), default=None),
            "boxes_max": max((r["boxes_max"] for r in built), default=None)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit-run", type=Path, required=True)
    parser.add_argument("--motion-cache", type=Path, required=True)
    parser.add_argument("--screen-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", type=Path, help="Optional filename list (e.g. the dev set)")
    parser.add_argument("--include-geometry-only", action="store_true",
                        help="Also build clips whose low posture has no environmental cause (kept separate)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    selection = {json.loads(line)["filename"]: json.loads(line)
                 for line in (args.screen_run / "selection.jsonl").read_text().splitlines() if line}
    fits = sorted((args.fit_run / "fits").glob("*.json"))
    wanted = set(args.clips.read_text().split()) if args.clips else None
    eligibility = ("environment_explained",) + (("geometry_only_low_posture_unexplained",)
                                                if args.include_geometry_only else ())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "shards").mkdir()
    jobs = []
    for fit_path in fits:
        name = fit_path.stem
        if (wanted is not None and name not in wanted) or name not in selection:
            continue
        cache = args.motion_cache / f"{name}__s2.npz"
        jobs.append((selection[name], str(fit_path), str(cache), str(args.output), {"eligibility": eligibility}))
    print(json.dumps({"clips": len(jobs), "workers": args.workers}), flush=True)
    started = time.time()
    results = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        for index, row in enumerate(pool.imap_unordered(build_clip, jobs, chunksize=4), 1):
            results.append(row)
            if index % 500 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} built {sum('shard' in r for r in results)} "
                      f"errors {sum('error' in r for r in results)} {time.time() - started:.0f}s", flush=True)
    built = sorted((r for r in results if "shard" in r), key=lambda r: r["filename"])
    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in built:
            stream.write(json.dumps(row) + "\n")
    report = {"schema": SCHEMA, "fit_run": str(args.fit_run), "motion_cache": str(args.motion_cache),
              "eligibility": list(eligibility), "elapsed_seconds": round(time.time() - started, 1),
              "summary": summarize(results),
              "errors": [{k: r[k] for k in ("filename", "error")} for r in results if "error" in r],
              "module_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
                  "motionbricks/data/bones_env_dataset.py", "scripts/build_bones_env_training_set.py")}}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
