"""Fit support geometry to screened BONES-SEED clips and render review visuals (step 3a).

Uses a finished screen run (selection.jsonl + clips.jsonl) to choose clips, converts
each BVH with the same official path as the screen, fits ground and support boxes
(motionbricks.data.bones_terrain_fit), and renders review material for a sample:
contact-sheet PNGs, optional GIFs and JSON for web/terrain_fit_review.

Motion is never modified. CPU only.

    python scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v1_full \
        --output runs/bones_fit_v1 --workers 16 --viz-per-stem 2 --gif
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motionbricks.data.bones_terrain_fit import (  # noqa: E402
    SCHEMA_VERSION, FitConfig, actor_scale_report, fit_support, foot_vertices_by_side, leg_vertices_by_side,
)
from motionbricks.data.bones_terrain_scene import (  # noqa: E402
    PART_FOOT, PART_HAND, PART_TRUNK, SceneConfig, scene_for_clip, vertex_part_codes,
)
from motionbricks.data.bones_terrain_viz import (  # noqa: E402
    VIEW_JOINTS, render_contact_sheet, render_gif, viewer_payload,
)
from scripts import screen_bones_terrain_geometry as screen  # noqa: E402

COMPACT = ("filename", "stem", "group", "status", "levels_m", "support_boxes", "flags", "penetration_authority",
           "minimal_penetration_max_m", "minimal_frames_over", "maximal_penetration_max_m", "maximal_frames_over",
           "sphere_minimal_penetration_max_m",
           "unsupported_evidence", "minimal_sizes_m", "maximal_sizes_m",
           "ceiling_status", "ceiling_range_m", "explanation_tight", "explanation_loose", "walls",
           "variants_accepted", "upright_ceilings", "clutter", "variant_ground_penetration_max_m",
           "eligibility", "rejection_reasons", "wall_status", "ground_source", "ground_skin_max_m", "seconds", "error")


def _init():
    screen._init_worker()
    import matplotlib
    matplotlib.use("Agg")
    skinner = screen._STATE.get("skinner")
    if skinner is not None and "standing_top" not in screen._STATE:
        # Uniform skin height in the bind pose: top of the head above the sole.
        bind = skinner.bind_vertices
        bind = bind.detach().cpu().numpy() if hasattr(bind, "detach") else np.asarray(bind)
        screen._STATE["standing_top"] = float(bind[:, 1].max() - bind[:, 1].min())
        screen._STATE["vertex_parts"] = skin_vertex_parts(skinner, screen._STATE["names"], bind)
        dominant = dominant_joints(skinner)
        screen._STATE["foot_vertices"] = foot_vertices_by_side(dominant, screen._STATE["names"])
        screen._STATE["leg_vertices"] = leg_vertices_by_side(dominant, screen._STATE["names"])
        counts = {f"{kind}_{side}": len(table[side]) for kind, table in
                  (("foot", screen._STATE["foot_vertices"]), ("leg", screen._STATE["leg_vertices"]))
                  for side in ("left", "right")}
        if min(counts.values()) < 100:
            raise RuntimeError(f"foot/leg skin labels look wrong: {counts}")
        screen._STATE["limb_vertex_counts"] = counts


def dominant_joints(skinner):
    """Index of the largest-weight LBS joint of each skin vertex."""
    indices, weights = skinner.lbs_indices, skinner.lbs_weights
    indices = indices.detach().cpu().numpy() if hasattr(indices, "detach") else np.asarray(indices)
    weights = weights.detach().cpu().numpy() if hasattr(weights, "detach") else np.asarray(weights)
    return np.take_along_axis(indices, weights.argmax(axis=1)[:, None], axis=1)[:, 0]


def skin_vertex_parts(skinner, names, bind):
    """Trunk/hand/foot code per skin vertex from its dominant LBS joint, checked on the bind pose."""
    dominant = dominant_joints(skinner)
    parts = vertex_part_codes(dominant, names)
    height = bind[:, 1] - bind[:, 1].min()
    counts = {code: int((parts == code).sum()) for code in (PART_TRUNK, PART_HAND, PART_FOOT)}
    foot, hand, trunk = (float(np.median(height[parts == code])) for code in (PART_FOOT, PART_HAND, PART_TRUNK))
    if min(counts.values()) < 100 or not (foot < 0.25 and foot < hand < trunk + 0.6):
        raise RuntimeError(f"skin part labels fail the bind-pose check: {counts}")
    return parts


def skin_vertices(rotations, posed, stride):
    """Full official SOMA77 skin (every stride-th frame) for the penetration metric."""
    torch = screen._STATE["torch"]
    skinner = screen._STATE["skinner"]
    local = torch.as_tensor(rotations[::stride], dtype=torch.float32)
    joints = torch.as_tensor(posed[::stride], dtype=torch.float32)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(local), 48):
            chunks.append(skinner.skin(local[start:start + 48], joints[start:start + 48],
                                       rot_is_global=False).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def clip_seed(filename):
    return int(hashlib.sha256(filename.encode()).hexdigest()[:8], 16)


#: Groups whose motion needs geometry the boxes cannot represent yet (thin ladder rungs,
#: obstacles under jumps). Their scenes are fitted and reported but never training-eligible.
UNMODELLED_GROUPS = ("ladder", "obstacle_clearance")


def _cache_path(cache, item, skin_stride):
    return Path(cache) / f"{item['filename']}__s{skin_stride}.npz"


def _source_stamp(path):
    stat = Path(path).stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def load_motion(item, skin_stride, cache=None, cache_skin=False):
    """Converted motion (and optionally skin) for one clip, from the cache when it is current.

    The cache only stores outputs of the unchanged official conversion path (BVH parse,
    30 Hz resampling, FK, sole regions, skinning); it is keyed by the BVH size and mtime and
    the skin stride, so fitting code changes never need it rebuilt.
    """
    path = _cache_path(cache, item, skin_stride) if cache else None
    if path is not None and path.exists():
        data = np.load(path, allow_pickle=False)
        if str(data["stamp"]) == _source_stamp(item["path"]) and (not skin_stride or not cache_skin or "skin" in data):
            names = [str(n) for n in data["joint_names"]]
            joints = {name: data["joints"][:, k].astype(np.float64) for k, name in enumerate(names)}
            soles = {"left": data["sole_left"].astype(np.float64), "right": data["sole_right"].astype(np.float64)}
            skin = None
            if skin_stride:
                if "skin" in data:     # float16 offsets from each skinned frame's root: about 1 mm steps
                    skin = data["skin"].astype(np.float32) + data["root"][::skin_stride][:, None].astype(np.float32)
                else:                  # exact: the same float32 rotations through the same skinner
                    skin = skin_vertices(data["rotations"], data["posed"], skin_stride)
            return soles, joints, data["root"].astype(np.float64), float(data["source_fps"]), skin, True
    if skin_stride:
        soles, joints, root, source_fps, _, (rotations, posed) = screen.convert(
            item["path"], extra_names=VIEW_JOINTS, return_motion=True)
        skin = skin_vertices(rotations, posed, skin_stride)
    else:
        soles, joints, root, source_fps, _ = screen.convert(item["path"], extra_names=VIEW_JOINTS)
        skin, rotations, posed = None, None, None
    if path is not None:
        names = sorted(joints)
        arrays = {"stamp": np.array(_source_stamp(item["path"])), "joint_names": np.array(names),
                  "joints": np.stack([joints[n] for n in names], axis=1).astype(np.float32),
                  "sole_left": soles["left"].astype(np.float32), "sole_right": soles["right"].astype(np.float32),
                  "root": root.astype(np.float32), "source_fps": np.array(source_fps)}
        if rotations is not None:
            arrays.update(rotations=np.asarray(rotations, dtype=np.float32), posed=np.asarray(posed, dtype=np.float32))
        if skin is not None and cache_skin:
            arrays["skin"] = (skin - root[::skin_stride][:, None]).astype(np.float16)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, **arrays)
        temporary.replace(path)
    return soles, joints, root, source_fps, skin, False


def _process(item, render, skin_stride=2, variants=4, cache=None, cache_skin=False):
    timing = {}
    clock = time.perf_counter()

    def lap(name):
        nonlocal clock
        now = time.perf_counter()
        timing[name] = round(now - clock, 4)
        clock = now

    soles, joints, root, source_fps, skin, cached = load_motion(item, skin_stride, cache, cache_skin)
    lap("motion_cached" if cached else "motion_converted")
    _, support, posture = screen.screen_clip(soles, joints, root, screen.FPS, screen._STATE["standing_hip"],
                                             dimension_tokens=item.get("dimension_tokens", ()), return_posture=True)
    lap("screen")
    fit = fit_support(soles, joints, root, screen.FPS, support, skin_vertices=skin,
                      skin_frame_stride=max(1, skin_stride), foot_vertices=screen._STATE.get("foot_vertices"),
                      leg_vertices=screen._STATE.get("leg_vertices"))
    lap("support_fit")
    if skin is not None:
        fit["skin"] = {"vertices": int(skin.shape[1]), "frame_stride": skin_stride}
        fit["scene"] = scene_for_clip(skin, posture[::skin_stride], joints["Hips"][::skin_stride][:, [0, 2]], fit,
                                      screen._STATE["standing_top"], screen.FPS / skin_stride,
                                      variants=variants, seed=clip_seed(item["filename"]),
                                      vertex_parts=screen._STATE.get("vertex_parts"))
        if item.get("group") in UNMODELLED_GROUPS:
            fit["scene"].update(eligibility="excluded_unmodelled_geometry", training_eligible=False,
                                eligibility_before_exclusion=fit["scene"]["eligibility"])
    lap("scene")
    fit["meta"] = {key: item[key] for key in ("filename", "stem", "group")}
    fit["meta"].update(source_fps=source_fps, dimension_tokens=item.get("dimension_tokens", []),
                       seconds=round(len(joints["Hips"]) / screen.FPS, 3))
    if render:
        folder = Path(render["folder"])
        title = f"{item['group']} / {item['stem']} / {item['filename']}"
        render_contact_sheet(joints, fit, folder / f"{item['filename']}.png", title=title)
        if render.get("gif"):
            render_gif(joints, fit, folder / f"{item['filename']}.gif", title=item["filename"])
        payload = viewer_payload(joints, fit, meta=fit["meta"])
        (Path(render["viewer"]) / f"{item['filename']}.json").write_text(json.dumps(payload), encoding="utf-8")
        lap("render")
    fit["timing_s"] = timing
    return fit


def work_fit(job):
    item, options = job
    started = time.perf_counter()
    try:
        fit = _process(item, None, options["skin_stride"], options["variants"],
                       options.get("motion_cache"), options.get("cache_skin", False))
        result = {"filename": item["filename"], "stem": item["stem"], "group": item["group"], "fit": fit}
    except Exception as error:
        result = {"filename": item["filename"], "stem": item["stem"], "group": item["group"],
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(limit=4)}
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def work_render(job):
    item, render, options = job
    try:
        _process(item, render, options["skin_stride"], options["variants"],
                 options.get("motion_cache"), options.get("cache_skin", False))
        return {"filename": item["filename"], "rendered": True}
    except Exception as error:
        return {"filename": item["filename"], "rendered": False, "error": f"{type(error).__name__}: {error}"}


def stage_summary(results):
    """Total and per-clip median seconds per processing stage, to see where time goes."""
    stages = {}
    for result in results:
        for name, value in result.get("fit", {}).get("timing_s", {}).items():
            stages.setdefault(name, []).append(value)
    return {name: {"clips": len(v), "total": round(float(np.sum(v)), 1), "p50": round(float(np.median(v)), 4),
                   "p95": round(float(np.percentile(v, 95)), 4)} for name, v in sorted(stages.items())}


def penetration(fit, variant):
    """(max depth, frames over 2 cm) from the authoritative metric.

    With skin, support boxes are judged on their own (skin_support); the ground slab is
    reported separately because mocap skin sinks a little into any floor."""
    metric = fit["metrics"][variant]
    if "skin_support" in metric:
        return metric["skin_support"]["max_m"], metric["skin_support"]["frames_over_report"]
    if "skin" in metric:
        return metric["skin"]["max_m"], metric["skin"]["frames_over_report"]
    return metric["penetration_max_m"], metric["penetration_frames_over_report"]


def compact_row(result):
    if "error" in result:
        return {"filename": result["filename"], "stem": result["stem"], "group": result["group"],
                "status": "failed", "error": result["error"].replace("\t", " ").replace("\n", " ")[:200]}
    fit = result["fit"]
    pen = {variant: penetration(fit, variant) for variant in ("minimal", "maximal")}
    sizes = {variant: [[round(2 * b["half_extent_xyz_m"][0], 3), round(2 * b["half_extent_xyz_m"][2], 3)]
                       for b in fit["boxes"][variant] if b["role"] != "ground"] for variant in ("minimal", "maximal")}
    return {
        "filename": result["filename"], "stem": result["stem"], "group": result["group"], "status": "ok",
        "levels_m": json.dumps([round(v, 4) for v in fit["levels_m"]]),
        "support_boxes": len(sizes["minimal"]), "flags": ",".join(fit["flags"]),
        "penetration_authority": fit.get("penetration_authority", "landmark_spheres"),
        "minimal_penetration_max_m": round(pen["minimal"][0], 4), "minimal_frames_over": pen["minimal"][1],
        "maximal_penetration_max_m": round(pen["maximal"][0], 4), "maximal_frames_over": pen["maximal"][1],
        "sphere_minimal_penetration_max_m": round(fit["metrics"]["minimal"]["penetration_max_m"], 4),
        "unsupported_evidence": fit["metrics"]["minimal"]["support_evidence_unsupported"],
        "minimal_sizes_m": json.dumps(sizes["minimal"]), "maximal_sizes_m": json.dumps(sizes["maximal"]),
        "ground_source": fit.get("ground_source", ""), "seconds": fit.get("meta", {}).get("seconds", ""),
        "ground_skin_max_m": (round(fit["metrics"]["minimal"]["skin_ground"]["max_m"], 4)
                              if "skin_ground" in fit["metrics"]["minimal"] else ""),
        **scene_columns(fit.get("scene")),
    }


def scene_columns(scene):
    if not scene:
        return {}
    report = scene["ceiling_report"]
    explanation = report.get("explanation", {})
    variants = scene["variants"]
    return {
        "ceiling_status": report["status"],
        "ceiling_range_m": json.dumps([round(v, 3) for v in scene["ceiling"]["range_above_floor_m"]]) if scene["ceiling"] else "",
        "explanation_tight": "" if explanation.get("tight") is None else round(explanation["tight"], 3),
        "explanation_loose": "" if explanation.get("loose") is None else round(explanation["loose"], 3),
        "walls": len(scene["walls"]),
        "variants_accepted": f"{sum(v['accepted'] for v in variants)}/{len(variants)}",
        "upright_ceilings": sum(v["params"]["upright_ceiling_m"] is not None for v in variants),
        "clutter": sum(v["params"]["clutter"] for v in variants),
        "variant_ground_penetration_max_m": round(max((v["penetration_m"]["ground"] for v in variants), default=0.0), 4),
        "eligibility": scene.get("eligibility", ""),
        "rejection_reasons": ",".join(sorted({r for v in variants for r in v.get("rejection_reasons", [])})),
        "wall_status": scene["wall_report"].get("status", ""),
    }


def _count(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def group_summary(results):
    groups = {}
    for result in results:
        groups.setdefault(result["group"], []).append(result)
    summary = {}
    for group, items in sorted(groups.items()):
        ok = [r["fit"] for r in items if "fit" in r]
        entry = {"clips": len(items), "fitted": len(ok), "failed": len(items) - len(ok)}
        if ok:
            boxes = [len([b for b in f["boxes"]["minimal"] if b["role"] != "ground"]) for f in ok]
            entry["support_boxes_distribution"] = {str(k): boxes.count(k) for k in sorted(set(boxes))}
            for variant in ("minimal", "maximal"):
                over = [penetration(f, variant)[1] for f in ok]
                peak = [penetration(f, variant)[0] for f in ok]
                entry[variant] = {"clips_with_frames_over_2cm": int(sum(v > 0 for v in over)),
                                  "penetration_max_p95_m": round(float(np.percentile(peak, 95)), 4)}
            entry["unsupported_evidence_points"] = int(sum(f["metrics"]["minimal"]["support_evidence_unsupported"] for f in ok))
            entry["flags"] = _count(flag for f in ok for flag in f["flags"])
            entry["ground_source"] = _count(f.get("ground_source", "unknown") for f in ok)
            checks = [c for f in ok for c in f.get("patch_checks", [])]
            entry["raised_sole_patches"] = {"checked": len(checks), "kept": sum(not c["dropped"] for c in checks),
                                            "dropped_by": _count(r for c in checks for r in c["dropped"])}
            ground_skin = [f["metrics"]["minimal"]["skin_ground"]["max_m"] for f in ok
                           if "skin_ground" in f["metrics"]["minimal"]]
            if ground_skin:
                entry["ground_skin_depth_m"] = {q: round(float(np.percentile(ground_skin, int(q[1:]))), 4)
                                                for q in ("p50", "p95", "p99")}
            scenes = [f["scene"] for f in ok if f.get("scene")]
            if scenes:
                status = {}
                for scene in scenes:
                    status[scene["ceiling_report"]["status"]] = status.get(scene["ceiling_report"]["status"], 0) + 1
                tight = [s["ceiling_report"]["explanation"]["tight"] for s in scenes
                         if s["ceiling"] and s["ceiling_report"].get("explanation", {}).get("tight") is not None]
                variants = [v for s in scenes for v in s["variants"]]
                fitted_pen = [c["frames_over_report"] for s in scenes for name, c in s["fitted_checks"].items()
                              if name.startswith("ceiling")]
                eligibility = {}
                for f in ok:
                    if f.get("scene"):
                        key = f["scene"].get("eligibility", "unknown")
                        slot = eligibility.setdefault(key, {"clips": 0, "seconds": 0.0})
                        slot["clips"] += 1
                        slot["seconds"] = round(slot["seconds"] + f.get("meta", {}).get("seconds", 0.0), 2)
                entry["scene"] = {
                    "eligibility": eligibility,
                    "rejection_reasons": _count(r for v in variants for r in v.get("rejection_reasons", [])),
                    "wall_status": _count(s["wall_report"].get("status", "unknown") for s in scenes),
                    "wall_kind": _count(s["wall_report"]["kind"] for s in scenes if s["walls"]),
                    "ceiling_status": status,
                    "explanation_tight_p50": round(float(np.median(tight)), 4) if tight else None,
                    "explanation_tight_p10": round(float(np.percentile(tight, 10)), 4) if tight else None,
                    "walls_fitted": sum(len(s["walls"]) > 0 for s in scenes),
                    "fitted_ceilings_with_frames_over_2cm": int(sum(v > 0 for v in fitted_pen)),
                    "variants": len(variants), "variants_accepted": sum(v["accepted"] for v in variants),
                    "upright_ceiling_variants": sum(v["params"]["upright_ceiling_m"] is not None for v in variants),
                    "clutter_mean": round(float(np.mean([v["params"]["clutter"] for v in variants])), 3),
                    "ground_skin_penetration_p95_m": round(float(np.percentile(
                        [v["penetration_m"]["ground"] for v in variants], 95)), 4),
                }
        summary[group] = entry
    return summary


def choose_renders(results, per_stem, worst_per_group, sample_per_group=0):
    """Deterministic sample per stem and per group, plus the worst penetration clips per group."""
    chosen = {}
    by_stem = {}
    for result in results:
        if "fit" in result:
            by_stem.setdefault(result["stem"], []).append(result)
    for stem, items in by_stem.items():
        items = sorted(items, key=lambda r: hashlib.sha256(r["filename"].encode()).hexdigest())
        for result in items[:per_stem]:
            chosen[result["filename"]] = "sample"
    by_group = {}
    for result in results:
        if "fit" in result:
            by_group.setdefault(result["group"], []).append(result)
    for items in by_group.values():
        for result in sorted(items, key=lambda r: hashlib.sha256(r["filename"].encode()).hexdigest())[:sample_per_group]:
            chosen.setdefault(result["filename"], "group_sample")
        worst = sorted(items, key=lambda r: -penetration(r["fit"], "minimal")[0])
        for result in worst[:worst_per_group]:
            if penetration(result["fit"], "minimal")[1] > 0:
                chosen.setdefault(result["filename"], "worst_penetration")
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--screen-run", type=Path, required=True, help="Directory with selection.jsonl and clips.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", nargs="*")
    parser.add_argument("--limit-per-stem", type=int)
    parser.add_argument("--viz-per-stem", type=int, default=2)
    parser.add_argument("--worst-per-group", type=int, default=3)
    parser.add_argument("--sample-per-group", type=int, default=0,
                        help="Also render this many hash-chosen clips per group (worst renders only cover failures)")
    parser.add_argument("--render-list", type=Path, help="Text file of filenames to render in addition")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--skin-stride", type=int, default=2,
                        help="Skin every n-th frame for the penetration metric; 0 disables skinning and scenes")
    parser.add_argument("--variants", type=int, default=4, help="Scene variants per clip")
    parser.add_argument("--clips", type=Path, help="Text file of filenames (one per line) to restrict the run to")
    parser.add_argument("--motion-cache", type=Path,
                        help="Directory caching converted motion per clip; reused while the BVH is unchanged")
    parser.add_argument("--cache-skin", action="store_true",
                        help="Also cache the skin as float16 root offsets (~1 mm steps, ~10 MB per clip) instead "
                             "of re-skinning; approximate, so not for release runs")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Worker processes; 0 runs in-process")
    args = parser.parse_args(argv)

    selection = {json.loads(line)["filename"]: json.loads(line)
                 for line in (args.screen_run / "selection.jsonl").read_text().splitlines() if line}
    screened_ok = [json.loads(line) for line in (args.screen_run / "clips.jsonl").read_text().splitlines() if line]
    screened_ok = [row for row in screened_ok if "error" not in row]
    items = [selection[row["filename"]] for row in sorted(screened_ok, key=lambda r: r["filename"])
             if row["filename"] in selection and (not args.groups or row["group"] in args.groups)]
    if args.clips:
        wanted = {line.strip() for line in args.clips.read_text().splitlines() if line.strip()}
        items = [item for item in items if item["filename"] in wanted]
    if args.limit_per_stem:
        kept, counts = [], {}
        for item in sorted(items, key=lambda i: hashlib.sha256(i["filename"].encode()).hexdigest()):
            if counts.get(item["stem"], 0) < args.limit_per_stem:
                kept.append(item)
                counts[item["stem"]] = counts.get(item["stem"], 0) + 1
        items = kept
    args.output.mkdir(parents=True, exist_ok=False)
    for folder in ("fits", "viz", "viewer"):
        (args.output / folder).mkdir()
    options = {"skin_stride": args.skin_stride, "variants": args.variants,
               "motion_cache": str(args.motion_cache) if args.motion_cache else None, "cache_skin": args.cache_skin}
    print(json.dumps({"clips": len(items), "workers": args.workers, **options}), flush=True)

    started = time.time()
    results = []
    if args.workers == 0:  # in-process, for debugging and tests
        _init()
        pool = None
        mapper = map
    else:
        pool = mp.get_context("spawn").Pool(args.workers, initializer=_init)
        mapper = lambda function, jobs, **kw: pool.imap_unordered(function, jobs, **kw)  # noqa: E731
    try:
        for index, result in enumerate(mapper(work_fit, [(item, options) for item in items],
                                              **({} if pool is None else {"chunksize": 4})), 1):
            results.append(result)
            if "fit" in result:
                (args.output / "fits" / f"{result['filename']}.json").write_text(
                    json.dumps(result["fit"], allow_nan=False), encoding="utf-8")
            if index % 200 == 0 or index == len(items):
                print(f"fit {index}/{len(items)}, failed {sum('error' in r for r in results)}, "
                      f"{time.time() - started:.0f}s", flush=True)
        chosen = choose_renders(results, args.viz_per_stem, args.worst_per_group, args.sample_per_group)
        if args.render_list:
            for name in args.render_list.read_text().split():
                if any(r.get("filename") == name and "fit" in r for r in results):
                    chosen.setdefault(name, "requested")
        lookup = {item["filename"]: item for item in items}
        jobs = []
        for filename, reason in sorted(chosen.items()):
            item = lookup[filename]
            folder = args.output / "viz" / item["group"]
            folder.mkdir(exist_ok=True)
            jobs.append((item, {"folder": str(folder), "viewer": str(args.output / "viewer"), "gif": args.gif},
                         options))
        rendered = list(mapper(work_render, jobs))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(f"rendered {sum(r['rendered'] for r in rendered)}/{len(rendered)}", flush=True)

    with (args.output / "fits_compact.tsv").open("w", encoding="utf-8") as stream:
        stream.write("\t".join(COMPACT) + "\n")
        for result in sorted(results, key=lambda r: (r["group"], r["stem"], r["filename"])):
            row = compact_row(result)
            stream.write("\t".join(str(row.get(column, "")) for column in COMPACT) + "\n")
    viewer_files = sorted(p.name for p in (args.output / "viewer").glob("*.json"))
    (args.output / "viewer" / "index.json").write_text(json.dumps({"files": viewer_files}), encoding="utf-8")
    scale_rows = [{"filename": r["filename"], "stem": r["stem"], "levels_m": r["fit"]["levels_m"],
                   "dimension_tokens": r["fit"]["meta"]["dimension_tokens"]} for r in results if "fit" in r]
    (args.output / "actor_scale.json").write_text(json.dumps(actor_scale_report(scale_rows), indent=2), encoding="utf-8")
    summary = group_summary(results)
    (args.output / "by_group.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = {
        "schema_version": SCHEMA_VERSION, "git_commit": screen.git_commit(),
        "screen_run": str(args.screen_run), "clips": len(items),
        "fitted": sum("fit" in r for r in results), "failed": sum("error" in r for r in results),
        "failures": [{k: r[k] for k in ("filename", "error")} for r in results if "error" in r],
        "rendered": {"clips": len(rendered), "ok": sum(r["rendered"] for r in rendered),
                     "failed": [r for r in rendered if not r["rendered"]], "reasons": chosen},
        "elapsed_seconds": round(time.time() - started, 1), "config": FitConfig().__dict__,
        "stage_seconds": stage_summary(results),
        "scene_config": SceneConfig().__dict__,
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "module_sha256": {name: screen.sha256_file(ROOT / name) for name in (
            "motionbricks/data/bones_terrain_fit.py", "motionbricks/data/bones_terrain_viz.py",
            "motionbricks/data/bones_terrain_scene.py",
            "motionbricks/data/bones_terrain_screen.py", "scripts/fit_bones_terrain_support.py")},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
