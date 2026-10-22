"""Check source geometry and loss preconditions without judging learned motion."""

import argparse
import json
from collections import Counter

import numpy as np
import torch

from motionbricks.data.bones_contract import dataset_snapshot
from motionbricks.data.bones_models import contact_geometry_losses
from motionbricks.data.bones_seed import build_representation, official_sole_offsets, sole_positions


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    snapshot = dataset_snapshot(args.dataset)
    records = json.loads((args.dataset / "accepted.json").read_text())
    representation = build_representation()
    offsets, _ = official_sole_offsets(representation.skeleton)
    clips, clearances, speeds, penetrations = [], [], [], []
    maximum_height_roundtrip = 0.0
    violations = Counter()
    for record in records:
        with np.load(args.dataset / record["shard"], allow_pickle=False) as shard:
            features = torch.from_numpy(shard["global_features"].copy())[None]
            confidence = torch.from_numpy(shard["contact_confidence"].copy()).bool()[None]
        with torch.inference_mode():
            motion = representation.inverse(features, False, return_all=True)
            soles = sole_positions(motion["posed_joints"], motion["global_joint_rots"],
                                   representation.skeleton, offsets)
            contacts = motion["foot_contacts"] > 0.5
            support = contacts & confidence
            stance = support[:, 1:] & support[:, :-1]
            height = soles[..., 1]
            speed = (soles[:, 1:] - soles[:, :-1]).norm(dim=-1) * representation.fps
            central_speed = torch.gradient(soles[0], spacing=(1 / representation.fps,), dim=(0,))[0].norm(dim=-1)[None]
            violations["contact_without_confidence"] += int((contacts & ~confidence).sum())
            violations["support_outside_35mm_band"] += int((support & (height.abs() > 0.0351)).sum())
            # The saved labels were estimated before one terminal frame was cropped.
            violations["support_above_20cmps_central_speed"] += int(
                (support[:, 1:-1] & (central_speed[:, 1:-1] > 0.2001)).sum())
            local_features = representation.dual_rep.global_to_local(features[0], False, False)[None]
            local = representation.dual_rep.local_motion_rep.inverse(local_features, False, return_all=True)
            local_soles = sole_positions(local["posed_joints"], local["global_joint_rots"],
                                         representation.skeleton, offsets)
            height_error = float((local_soles[..., 1] - height[:, :local_soles.shape[1]]).abs().max())
            maximum_height_roundtrip = max(maximum_height_roundtrip, height_error)
            losses = contact_geometry_losses(soles, contacts.float(), confidence, representation.fps)
        selected_clearance, selected_speed = height[support].abs(), speed[stance]
        clearances.append(selected_clearance)
        speeds.append(selected_speed)
        penetrations.append(torch.relu(-height).flatten())
        clips.append({
            "filename": record["filename"], "split": record["split"], "behavior": record["behavior"],
            "support_samples": int(support.sum()), "stance_samples": int(stance.sum()),
            "source_skate_mps": float(losses["l_trusted_skate_mps"]),
            "source_penetration_loss_m": float(losses["l_sole_penetration_m"]),
            "source_penetration_peak_m": float(torch.relu(-height).max()),
        })
    clearances, speeds, penetrations = [torch.cat(values) for values in (clearances, speeds, penetrations)]
    checks = {
        "saved_support_labels_consistent": not any(violations.values()),
        "local_root_preserves_fixed_ground": maximum_height_roundtrip < 1e-4,
        "finite_source_losses": all(np.isfinite(clip["source_skate_mps"]) and
                                    np.isfinite(clip["source_penetration_loss_m"]) for clip in clips),
        "every_behavior_has_support": all(any(clip["stance_samples"] for clip in clips if clip["behavior"] == behavior)
                                          for behavior in {clip["behavior"] for clip in clips}),
    }
    result = {
        "schema": "motionbricks.geometry_preflight.v1", "dataset_snapshot_sha256": snapshot["sha256"],
        "clips_audited": len(clips), "checks": checks, "automatic_preconditions_passed": all(checks.values()),
        "manual_contact_ground_truth_validated": False,
        "scope": "source-only engineering checks; not contact annotation accuracy or trained UE acceptance",
        "ground_y_m": 0.0, "penetration_tolerance_m": 0.005, "fps": representation.fps,
        "maximum_local_global_sole_height_difference_m": maximum_height_roundtrip,
        "label_violation_counts": dict(violations),
        "source_support_clearance_abs_p95_m": float(torch.quantile(clearances, 0.95)),
        "source_stance_speed_mean_mps": float(speeds.mean()),
        "source_stance_speed_p95_mps": float(torch.quantile(speeds, 0.95)),
        "source_penetration_mean_m": float(penetrations.mean()),
        "source_penetration_p99_m": float(torch.quantile(penetrations, 0.99)),
        "source_penetration_peak_m": float(penetrations.max()),
        "source_penetration_over_2cm_fraction": float((penetrations > 0.02).float().mean()),
        "clips": clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != "clips"}, indent=2, allow_nan=False))
    if not result["automatic_preconditions_passed"]:
        raise RuntimeError("Source geometry preconditions failed; inspect the saved report before training")


if __name__ == "__main__":
    main()
