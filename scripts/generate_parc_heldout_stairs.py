"""Reuse verified native PARC generation on reserved evaluation geometry."""

import argparse
import json
from pathlib import Path

import generate_parc_rugged_candidates as generation
from build_parc_heldout_stair_course import build_heldout_course


def main(args):
    generation.build_rugged_course=build_heldout_course
    generation.main(args)
    path=args.output/'generation.json'
    report=json.loads(path.read_text())
    report.update(course_wrapper='pfnn_evaluation_stairs_20261015_v1',
        heldout_builder_sha256=generation.generation.pilot.sha256(Path(__file__).with_name('build_parc_heldout_stair_course.py')),
        current_full_training_excluded=True,training_started=False)
    path.write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--layout',type=int,choices=range(8),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--terrain-sha256',required=True)
    main(parser.parse_args())
