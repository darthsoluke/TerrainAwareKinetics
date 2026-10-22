"""Bounded CPU fixture diagnostics or local public dataset metadata inspection."""

import argparse
import json

import numpy as np

from motionbricks.scene_control_v2.data import (
    inspect_parc15_sidecar,
    inspect_public_metadata,
    make_synthetic_fixture,
    make_synthetic_scene,
    validate_paired_sample,
)
from motionbricks.scene_control_v2.sampling import offline_teacher_guidance, sample_scene_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("synthetic", help="Explicit offline oracle; no real dataset claims")
    public = sub.add_parser("public", help="Inspect supplied root read-only; no downloads")
    public.add_argument("dataset", choices=("LINGO", "TRUMANS", "CIRCLE"))
    public.add_argument("root")
    parc = sub.add_parser("parc15", help="Inspect existing source labels, not certified contacts")
    parc.add_argument("path")
    args = parser.parse_args()
    if args.mode == "public":
        report = inspect_public_metadata(args.dataset, args.root)
    elif args.mode == "parc15":
        report = inspect_parc15_sidecar(args.path)
    else:
        sample = make_synthetic_fixture()
        validate_paired_sample(sample)
        query = sample_scene_queries(make_synthetic_scene(sample), sample,
                                     guidance=offline_teacher_guidance(sample), mode="offline",
                                     as_of_s=float(sample["timestamps_s"][-1]))
        report = {"source": sample["source"], "purpose": sample["provenance"]["purpose"],
                  "schema": sample["schema"], "query_schema": query["schema"],
                  "policy": query["policy"], "guidance_role": query["guidance_role"],
                  "queries": len(query["valid"]), "valid_queries": int(query["valid"].sum()),
                  "regions": np.bincount(query["region_id"], minlength=3).tolist(),
                  "distance_semantics": query["distance_semantics"], "gpu_launches": 0,
                  "optimizer_updates": 0, "real_entity_samples": 0}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
