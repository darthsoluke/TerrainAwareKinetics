"""Use the verified PARC generator on explicitly new stair approach geometry."""
import argparse
import json
from pathlib import Path

import generate_parc_rugged_candidates as generation
from build_parc_pose_entry_course import build_pose_entry_course


def main(args):
    generation.build_rugged_course = build_pose_entry_course
    generation.main(args)
    path = args.output / 'generation.json'
    report = json.loads(path.read_text())
    report['course_wrapper'] = 'pose_entry_20261014_v1'
    report['pose_entry_builder_sha256'] = generation.generation.pilot.sha256(Path(__file__).with_name('build_parc_pose_entry_course.py'))
    report['training_started'] = False
    path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'checkpoint', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--layout', type=int, choices=range(8), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--terrain-sha256', required=True)
    main(p.parse_args())
