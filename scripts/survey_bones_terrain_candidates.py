"""Survey BONES-SEED clips that are candidates for terrain fitting.

Reads only annotations (timeline jsonl, optional metadata table) and, optionally,
the names of BVH files under a source root. No motion file is opened and no
source file is modified.

Examples:

    # Inspect an unfamiliar metadata file first.
    python scripts/survey_bones_terrain_candidates.py --describe-metadata \
        --metadata /path/to/metadata.pkl

    # Timelines only (official NVIDIA SEED timeline jsonl).
    python scripts/survey_bones_terrain_candidates.py \
        --timelines /path/to/seed_timelines.jsonl \
        --bvh-root /path/to/soma_uniform/bvh --output runs/bones_terrain_survey_v1

    # With the official metadata table (51 columns), when available.
    python scripts/survey_bones_terrain_candidates.py \
        --timelines /path/to/seed_timelines.jsonl \
        --metadata /path/to/seed_metadata_v003.parquet --output runs/bones_terrain_survey_v1
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motionbricks.data.bones_terrain_survey import (  # noqa: E402
    BUCKETS, SCHEMA_VERSION, describe_metadata, file_sha256, load_timelines,
    read_metadata_rows, survey,
)


def print_summary(report):
    print(f"clips: {report['clips']}")
    header = f"{'bucket':<28}{'clips':>8}{'orig':>8}{'mirror':>8}{'orig_s':>12}{'event_s':>10}{'families':>10}{'actors':>8}"
    print(header)
    for name in BUCKETS + ("any_candidate", "flat_remainder"):
        summary = report["buckets"].get(name) or report[name]
        print(f"{name:<28}{summary['clips']:>8}{summary['original_clips']:>8}{summary['mirror_clips']:>8}"
              f"{summary['original_duration']['seconds']:>12.1f}"
              f"{summary.get('original_matched_event_seconds', 0.0):>10.1f}"
              f"{summary['families']:>10}{summary['actors']:>8}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timelines", type=Path, help="Official SEED timeline jsonl")
    parser.add_argument("--metadata", type=Path, help="Metadata table (.parquet/.csv/.json/.jsonl/.pkl)")
    parser.add_argument("--bvh-root", type=Path, help="Optional source BVH tree; only file names are read")
    parser.add_argument("--output", type=Path, help="New output directory (must not exist)")
    parser.add_argument("--describe-metadata", action="store_true",
                        help="Print the metadata column inventory and exit")
    args = parser.parse_args(argv)

    metadata_rows = read_metadata_rows(args.metadata) if args.metadata else None
    if args.describe_metadata:
        if metadata_rows is None:
            parser.error("--describe-metadata requires --metadata")
        print(json.dumps(describe_metadata(metadata_rows), indent=2, ensure_ascii=False, default=str))
        return 0
    if args.timelines is None and metadata_rows is None:
        parser.error("Provide --timelines and/or --metadata")
    if args.output is None:
        parser.error("--output is required")
    args.output.mkdir(parents=True, exist_ok=False)

    timelines = load_timelines(args.timelines) if args.timelines else {}
    bvh_stems = None
    if args.bvh_root:
        bvh_stems = {path.stem for path in args.bvh_root.rglob("*.bvh")}
    rows, report = survey(timelines, metadata_rows, bvh_stems)
    report["inputs"] = {
        "timelines": ({"path": str(args.timelines), "sha256": file_sha256(args.timelines),
                       "entries": len(timelines)} if args.timelines else None),
        "metadata": ({"path": str(args.metadata), "sha256": file_sha256(args.metadata),
                      "rows": len(metadata_rows),
                      "columns": describe_metadata(metadata_rows)["official_columns_present"]}
                     if args.metadata else None),
        "bvh_root": str(args.bvh_root) if args.bvh_root else None,
        "survey_module_sha256": file_sha256(ROOT / "motionbricks/data/bones_terrain_survey.py"),
    }
    with (args.output / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            if row["buckets"]:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    (args.output / "survey_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{SCHEMA_VERSION}: wrote {args.output}")
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
