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
    SCHEMA_VERSION, FitConfig, actor_scale_report, fit_support,
)
from motionbricks.data.bones_terrain_viz import (  # noqa: E402
    VIEW_JOINTS, render_contact_sheet, render_gif, viewer_payload,
)
from scripts import screen_bones_terrain_geometry as screen  # noqa: E402

COMPACT = ("filename", "stem", "group", "status", "levels_m", "support_boxes", "flags",
           "minimal_penetration_max_m", "minimal_frames_over", "maximal_penetration_max_m", "maximal_frames_over",
           "unsupported_evidence", "minimal_sizes_m", "maximal_sizes_m", "error")


def _init():
    screen._init_worker()
    import matplotlib
    matplotlib.use("Agg")


def _process(item, render):
    soles, joints, root, source_fps, _ = screen.convert(item["path"], extra_names=VIEW_JOINTS)
    _, support = screen.screen_clip(soles, joints, root, screen.FPS, screen._STATE["standing_hip"],
                                    dimension_tokens=item.get("dimension_tokens", ()))
    fit = fit_support(soles, joints, root, screen.FPS, support)
    fit["meta"] = {key: item[key] for key in ("filename", "stem", "group")}
    fit["meta"].update(source_fps=source_fps, dimension_tokens=item.get("dimension_tokens", []))
    if render:
        folder = Path(render["folder"])
        title = f"{item['group']} / {item['stem']} / {item['filename']}"
        render_contact_sheet(joints, fit, folder / f"{item['filename']}.png", title=title)
        if render.get("gif"):
            render_gif(joints, fit, folder / f"{item['filename']}.gif", title=item["filename"])
        payload = viewer_payload(joints, fit, meta=fit["meta"])
        (Path(render["viewer"]) / f"{item['filename']}.json").write_text(json.dumps(payload), encoding="utf-8")
    return fit


def work_fit(item):
    started = time.perf_counter()
    try:
        fit = _process(item, None)
        result = {"filename": item["filename"], "stem": item["stem"], "group": item["group"], "fit": fit}
    except Exception as error:
        result = {"filename": item["filename"], "stem": item["stem"], "group": item["group"],
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(limit=4)}
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def work_render(job):
    item, render = job
    try:
        _process(item, render)
        return {"filename": item["filename"], "rendered": True}
    except Exception as error:
        return {"filename": item["filename"], "rendered": False, "error": f"{type(error).__name__}: {error}"}


def compact_row(result):
    if "error" in result:
        return {"filename": result["filename"], "stem": result["stem"], "group": result["group"],
                "status": "failed", "error": result["error"].replace("\t", " ").replace("\n", " ")[:200]}
    fit = result["fit"]
    sizes = {variant: [[round(2 * b["half_extent_xyz_m"][0], 3), round(2 * b["half_extent_xyz_m"][2], 3)]
                       for b in fit["boxes"][variant] if b["role"] != "ground"] for variant in ("minimal", "maximal")}
    return {
        "filename": result["filename"], "stem": result["stem"], "group": result["group"], "status": "ok",
        "levels_m": json.dumps([round(v, 4) for v in fit["levels_m"]]),
        "support_boxes": len(sizes["minimal"]), "flags": ",".join(fit["flags"]),
        "minimal_penetration_max_m": round(fit["metrics"]["minimal"]["penetration_max_m"], 4),
        "minimal_frames_over": fit["metrics"]["minimal"]["penetration_frames_over_report"],
        "maximal_penetration_max_m": round(fit["metrics"]["maximal"]["penetration_max_m"], 4),
        "maximal_frames_over": fit["metrics"]["maximal"]["penetration_frames_over_report"],
        "unsupported_evidence": fit["metrics"]["minimal"]["support_evidence_unsupported"],
        "minimal_sizes_m": json.dumps(sizes["minimal"]), "maximal_sizes_m": json.dumps(sizes["maximal"]),
    }


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
                over = [f["metrics"][variant]["penetration_frames_over_report"] for f in ok]
                peak = [f["metrics"][variant]["penetration_max_m"] for f in ok]
                entry[variant] = {"clips_with_frames_over_2cm": int(sum(v > 0 for v in over)),
                                  "penetration_max_p95_m": round(float(np.percentile(peak, 95)), 4)}
            entry["unsupported_evidence_points"] = int(sum(f["metrics"]["minimal"]["support_evidence_unsupported"] for f in ok))
            flags = {}
            for f in ok:
                for flag in f["flags"]:
                    flags[flag] = flags.get(flag, 0) + 1
            entry["flags"] = flags
        summary[group] = entry
    return summary


def choose_renders(results, per_stem, worst_per_group):
    """Deterministic sample per stem, plus the worst penetration clips per group."""
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
        worst = sorted(items, key=lambda r: -r["fit"]["metrics"]["minimal"]["penetration_max_m"])
        for result in worst[:worst_per_group]:
            if result["fit"]["metrics"]["minimal"]["penetration_frames_over_report"] > 0:
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
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Worker processes; 0 runs in-process")
    args = parser.parse_args(argv)

    selection = {json.loads(line)["filename"]: json.loads(line)
                 for line in (args.screen_run / "selection.jsonl").read_text().splitlines() if line}
    screened_ok = [json.loads(line) for line in (args.screen_run / "clips.jsonl").read_text().splitlines() if line]
    screened_ok = [row for row in screened_ok if "error" not in row]
    items = [selection[row["filename"]] for row in sorted(screened_ok, key=lambda r: r["filename"])
             if row["filename"] in selection and (not args.groups or row["group"] in args.groups)]
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
    print(json.dumps({"clips": len(items), "workers": args.workers}), flush=True)

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
        for index, result in enumerate(mapper(work_fit, items, **({} if pool is None else {"chunksize": 4})), 1):
            results.append(result)
            if "fit" in result:
                (args.output / "fits" / f"{result['filename']}.json").write_text(
                    json.dumps(result["fit"], allow_nan=False), encoding="utf-8")
            if index % 200 == 0 or index == len(items):
                print(f"fit {index}/{len(items)}, failed {sum('error' in r for r in results)}, "
                      f"{time.time() - started:.0f}s", flush=True)
        chosen = choose_renders(results, args.viz_per_stem, args.worst_per_group)
        lookup = {item["filename"]: item for item in items}
        jobs = []
        for filename, reason in sorted(chosen.items()):
            item = lookup[filename]
            folder = args.output / "viz" / item["group"]
            folder.mkdir(exist_ok=True)
            jobs.append((item, {"folder": str(folder), "viewer": str(args.output / "viewer"), "gif": args.gif}))
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
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "module_sha256": {name: screen.sha256_file(ROOT / name) for name in (
            "motionbricks/data/bones_terrain_fit.py", "motionbricks/data/bones_terrain_viz.py",
            "motionbricks/data/bones_terrain_screen.py", "scripts/fit_bones_terrain_support.py")},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
