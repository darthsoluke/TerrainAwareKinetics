"""Versioned small-subset conversion on the official training split.

Only records whose accepted.json split is train are converted, so the five validation
review windows can never leak into training. The output is a separate versioned
directory; nothing in the shipped pilot-v7 dataset is touched or overwritten.

Each item carries the recorded features needed for training plus the candidate
sole-region contact labels, their sliding state, and the per-interval and per-vertex
measured masks. A zero slip value without a true measured flag means "no evidence",
and training must read the mask rather than the value.
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_seed import build_representation, official_sole_surfaces
from motionbricks.data.sole_region_contact import (FRAME_STATES, SOLE_LABEL_THRESHOLDS, material_point_slip,
                                                   region_gap_m, sole_region_indices, sole_region_heights,
                                                   sole_region_labels)
from motionbricks.data.soma_skin_reference import load_official_skinner
from motionbricks.data.sole_geometry_losses import region_geometry_from_motion

DEFAULT_DATASET = Path("out/bones_pilot_reports/pilot-v7")
DEFAULT_OUTPUT = Path("out/sole_region_labels_v1/train_subset")
SIDES = ("Left", "Right")
GROUND_HEIGHT_M = 0.0


def label_item(local_rot_mats, root_positions, skeleton, skinner, indices, fps):
    """Candidate labels for one clip, from the recorded source motion only."""
    selected = np.concatenate([indices[side] for side in SIDES])
    heights, horizontal_chunks = [], []
    with torch.inference_mode():
        for start in range(0, len(local_rot_mats), 512):
            height, horizontal = region_geometry_from_motion(
                local_rot_mats[start:start + 512], root_positions[start:start + 512], skinner, selected)
            heights.append(height)
            horizontal_chunks.append(horizontal)
    all_height, all_horizontal = torch.cat(heights), torch.cat(horizontal_chunks)
    out = {}
    offset = 0
    for side in SIDES:
        width = len(indices[side])
        height = all_height[:, offset:offset + width]
        horizontal = all_horizontal[:, offset:offset + width]
        offset += width
        gap = region_gap_m(height)
        slip = material_point_slip(height, horizontal, fps,
                                   contact_clearance_m=SOLE_LABEL_THRESHOLDS["contact_clearance_m"],
                                   min_run_frames=SOLE_LABEL_THRESHOLDS["min_contact_run_frames"],
                                   rest_vertical_speed_mps=SOLE_LABEL_THRESHOLDS["rest_vertical_speed_mps"])
        labels = sole_region_labels(gap["region_gap_m"], slip,
                                    contact_clearance_m=SOLE_LABEL_THRESHOLDS["contact_clearance_m"],
                                    clearance_m=SOLE_LABEL_THRESHOLDS["clearance_m"],
                                    slip_speed_mps=SOLE_LABEL_THRESHOLDS["slip_speed_mps"])
        out[side] = {
            "region_gap_m": gap["region_gap_m"].numpy().astype(np.float32),
            "region_penetration_m": gap["region_penetration_m"].numpy().astype(np.float32),
            "frame_state": labels["frame_state"].numpy().astype(np.int8),
            "interval_state": labels["interval_state"].numpy().astype(np.int8),
            "region_measured": labels["measured_interval"].numpy(),
            "vertex_measured": slip["vertex_measured"].numpy(),
            "region_slip_mps": slip["region_slip_mps"].numpy().astype(np.float32),
            "frame_counts": labels["frame_counts"],
            "interval_counts": labels["interval_counts"],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Versioned sole-region label subset on the training split")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-behavior", type=int, default=2)
    parser.add_argument("--all-frames", action="store_true", help="Label whole clips so random training crops are covered")
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"),
                        help="Which accepted.json split to convert; validation rows label the held-out review windows")
    parser.add_argument("--filenames", default="", help="Optional comma-separated filename allow-list")
    parser.add_argument("--frames", type=int, default=129,
                        help="Global frames per item. The existing pipeline needs 129 global frames to produce "
                             "128 local frames, so the last frame is never duplicated to pad a short item.")
    args = parser.parse_args()
    if args.frames < 129 or args.per_behavior < 0:
        parser.error("At least 129 frames are required; per-behavior=0 selects the whole split")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh label version directory; existing labels must not be overwritten")
    torch.set_num_threads(4)
    records = json.loads((args.dataset / "accepted.json").read_text())
    excluded = sorted({record["filename"] for record in records if record["split"] != "train"})
    # The manifest field is "shards/xxx.npz" while the files on disk are "xxx.npz",
    # so the two are compared by basename and resolved by joining the dataset root.
    available = {item.name for item in (args.dataset / "shards").glob("*.npz")}
    allow = {name.strip() for name in args.filenames.split(",") if name.strip()}
    train = [record for record in records if record["split"] == args.split
             and record["frames_60hz"] >= args.frames
             and Path(record["shard"]).name in available
             and (not allow or record["filename"] in allow)]
    missing_shards = sorted({Path(record["shard"]).name for record in records
                             if record["split"] == args.split
                             and Path(record["shard"]).name not in available})
    chosen, per_behavior = [], collections.Counter()
    for record in sorted(train, key=lambda item: item["filename"]):
        if args.per_behavior and per_behavior[record["behavior"]] >= args.per_behavior:
            continue
        per_behavior[record["behavior"]] += 1
        chosen.append(record)
    if not chosen:
        raise ValueError("No records selected for split " + args.split)
    skeleton = build_representation().skeleton
    surfaces, _, _, mesh_sha256 = official_sole_surfaces(skeleton)
    indices = {side: sole_region_indices(surfaces, side) for side in SIDES}
    skinner = load_official_skinner(skeleton)
    args.output.mkdir(parents=True, exist_ok=True)
    items, totals = [], {state: 0 for state in FRAME_STATES}
    measured_total = stable_total = 0
    for record in chosen:
        with np.load(args.dataset / Path(record["shard"]).name
                     if not Path(record["shard"]).parent.name else args.dataset / record["shard"],
                     allow_pickle=False) as shard:
            if shard["global_features"].shape[0] < args.frames:
                raise ValueError(record["filename"] + ": shard is shorter than the frame contract")
            count = len(shard["global_features"]) if args.all_frames else args.frames
            features = shard["global_features"][:count].copy()
            local = torch.from_numpy(shard["local_rot_mats"][:count].astype(np.float32))
            root = torch.from_numpy(shard["root_positions"][:count].copy())
            confidence = shard["contact_confidence"][:count].copy()
        labels = label_item(local, root, skeleton, skinner, indices, 60.0)
        name = f"{record['filename']}.npz"
        np.savez_compressed(
            args.output / name, features=features.astype(np.float32), confidence=confidence,
            local_rot_mats=local.numpy(), root_positions=root.numpy(),
            **{f"region_gap_m_{side.lower()}": labels[side]["region_gap_m"] for side in SIDES},
            **{f"region_penetration_m_{side.lower()}": labels[side]["region_penetration_m"] for side in SIDES},
            **{f"frame_state_{side.lower()}": labels[side]["frame_state"] for side in SIDES},
            **{f"interval_state_{side.lower()}": labels[side]["interval_state"] for side in SIDES},
            **{f"region_measured_{side.lower()}": labels[side]["region_measured"] for side in SIDES},
            **{f"vertex_measured_{side.lower()}": labels[side]["vertex_measured"] for side in SIDES},
            **{f"region_slip_mps_{side.lower()}": labels[side]["region_slip_mps"] for side in SIDES},
            region_indices_left=indices["Left"], region_indices_right=indices["Right"],
            fps=np.array(60.0), ground_height_m=np.array(GROUND_HEIGHT_M))
        for side in SIDES:
            measured_total += int(labels[side]["region_measured"].sum())
            stable_total += labels[side]["interval_counts"]["stable_contact"]
            for state, value in labels[side]["frame_counts"].items():
                totals[state] += value
        items.append({"filename": record["filename"], "behavior": record["behavior"],
                      "split": record["split"], "shard": record["shard"],
                      "frames": int(features.shape[0]), "file": name,
                      "source_sha256": record["source_sha256"]})
        print(json.dumps({"converted": len(items), "total": len(chosen), "filename": record["filename"]}), flush=True)
    manifest = {
        "schema_version": 1,
        "kind": "versioned sole-region label subset on the official training split",
        "status": "candidate; not the shipped dataset",
        "split_policy": {
            "converted_split": args.split,
            "validation_records_excluded": excluded,
            "no_validation_leakage": f"Only records whose accepted.json split is {args.split} were converted.",
            "per_behavior": args.per_behavior,
            "global_frames_per_item": "whole clip" if args.all_frames else args.frames,
            "local_frames_after_window_conversion": "global frames minus one" if args.all_frames else args.frames - 1,
            "frame_contract": "The pipeline converts 129 global frames into 128 local frames. The last frame is "
                              "never duplicated and no item is padded.",
            "shards_present_locally": sorted(available),
            "resolved_shard_name": "compared by basename, loaded by the manifest path",
            "train_shards_not_present_locally": missing_shards,
            "note": "Only shards present on this machine were converted. The missing ones are part of the full "
                    "training split and were not available offline.",
        },
        "labels": {"thresholds": SOLE_LABEL_THRESHOLDS, "mesh_sha256": mesh_sha256,
                   "ground_height_m": GROUND_HEIGHT_M,
                   "measured_mask_required": "Training must read region_measured. A zero slip without a true "
                                             "measured flag means no evidence, not a stationary foot."},
        "totals": {"frame_states": totals, "stable_contact_intervals": stable_total,
                   "intervals_with_evidence": measured_total},
        "items": items,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps({"items": len(items), "per_behavior": dict(per_behavior),
                      "missing_shards": missing_shards, "totals": manifest["totals"]}, indent=2))


if __name__ == "__main__":
    main()
