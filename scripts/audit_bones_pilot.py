"""Audit the frozen pilot's hierarchy, split isolation, feature shape and provenance."""

import argparse
import json
import hashlib
import importlib.metadata
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from kimodo.skeleton.bvh import Bvh
from kimodo.skeleton.registry import build_skeleton
from prepare_bones_locomotion import behavior, validate_header


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runs", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    records = json.loads((args.dataset / "accepted.json").read_text())
    report = json.loads((args.dataset / "quality_report.json").read_text())
    skeleton = build_skeleton(77)
    groups, hashes = {}, {}
    semantic_failures = []
    for record in records:
        for identifier, mapping in ((record["recording_group"], groups), (record["source_sha256"], hashes)):
            previous = mapping.setdefault(identifier, record["split"])
            if previous != record["split"]:
                raise RuntimeError("Source recording or duplicate leaks across splits")
        header = []
        with Path(record["path"]).open() as stream:
            for line in stream:
                if line.strip() == "MOTION":
                    break
                header.append(line)
        validate_header(Bvh("".join(header), backend="graph"), skeleton)
        label = behavior({"overview_description": record["description"], "events": record["events"]},
                         record["filename"])
        if label != record["behavior"]:
            semantic_failures.append(record["filename"])
        with np.load(args.dataset / record["shard"], allow_pickle=False) as shard:
            if shard["global_features"].shape != (record["frames_60hz"], 366):
                raise RuntimeError("Feature shape/manifest mismatch")
            for name in shard.files:
                if not np.isfinite(shard[name]).all():
                    raise RuntimeError("Nonfinite shard: " + record["filename"])
    if semantic_failures and report.get("semantic_policy"):
        raise RuntimeError("Current semantic policy mismatch: " + ", ".join(semantic_failures))
    coverage = {split: dict(Counter(record["behavior"] for record in records if record["split"] == split))
                for split in ("train", "validation", "test")}
    behaviors = {record["behavior"] for record in records}
    result = {
        "clips_audited": len(records), "recording_groups": len(groups),
        "hierarchy_and_rotation_channels_passed": True, "group_and_hash_split_isolation_passed": True,
        "finite_features_passed": True,
        "calibration_cropped_clips": sum(record["source_crop_start_seconds"] > 0 for record in records),
        "maximum_body_fk_error_m": max(record["body_subset_fk_error_m"] for record in records),
        "maximum_feature_roundtrip_error_m": max(record["feature_roundtrip_error_m"] for record in records),
        "manual_contact_ground_truth_validated": False, "actor_held_out_split_validated": False,
        "automatic_semantic_policy_passed": not semantic_failures,
        "manual_semantic_review_passed": False,
        "behavior_counts_by_split": coverage,
        "held_out_behavior_coverage_passed": all(behaviors <= set(coverage[split])
                                                for split in ("validation", "test")),
    }
    (args.dataset / "frozen_audit.json").write_text(json.dumps(result, indent=2))
    stats_name = report["stats_folder"]
    stats_hash = hashlib.sha256(b"".join(
        (args.dataset / stats_name / name).read_bytes() for name in ("mean.npy", "std.npy"))).hexdigest()
    for run in args.runs:
        contract_path = run / "run_contract.json"
        contract = json.loads(contract_path.read_text())
        if contract["dataset"] != str(args.dataset.resolve()) or contract["stats_folder"] != stats_name:
            raise RuntimeError("Run does not describe this audited dataset/statistics")
        if contract.get("stats_sha256", stats_hash) != stats_hash:
            raise RuntimeError("Run statistics digest differs")
        contract["stats_sha256"] = stats_hash
        contract["provenance_snapshot_after_run"] = {
            package: importlib.metadata.version(package) for package in
            ("vector-quantize-pytorch", "adam-atan2-pytorch", "kimodo", "numpy", "scipy")}
        contract_path.write_text(json.dumps(contract, indent=2))
        (run / "environment-freeze.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
