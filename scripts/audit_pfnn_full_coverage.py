"""CPU-only P0 audit of existing full PFNN data and its four-quarter cache."""

import argparse
import json
from pathlib import Path

from motionbricks.data.pfnn_full_training_data import SPAN_FRAMES, audit_full_coverage


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        report = audit_full_coverage(args.cache, args.pfnn, args.inventory, span_frames=args.span_frames)
        per_file = report.pop('per_file')
        (args.output / 'per_file_coverage.json').write_text(json.dumps(per_file, indent=2) + '\n', encoding='utf-8')
        report['per_file_coverage'] = 'per_file_coverage.json'
        (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: report[key] for key in ('passed', 'phase', 'original_files',
                          'augmented_rows', 'admitted_pair_windows', 'zero_admission_files', 'splits')}))
    except Exception as error:
        (args.output / 'failure.json').write_text(json.dumps(dict(passed=False, phase='P0', error=repr(error),
            training_exposures=0, optimizer_updates=0), indent=2) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path, help='Existing cache_pfnn_world_v2 root')
    parser.add_argument('--pfnn', required=True, type=Path, help='Existing dataset_world_v2 root')
    parser.add_argument('--inventory', type=Path, help='Optional original source inventory JSON')
    parser.add_argument('--output', required=True, type=Path, help='New audit output directory')
    parser.add_argument('--span-frames', nargs='+', type=int, default=list(SPAN_FRAMES))
    main(parser.parse_args())
