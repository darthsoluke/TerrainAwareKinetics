"""Check recorded progress beyond a source-defined stair-top plane, without reruns."""

import argparse
import json
from pathlib import Path

import numpy as np


def main(args):
    report = json.loads((args.review / 'complete.json').read_text())
    rows = []
    for row in report['rows']:
        if row['mode'] == 'source':
            continue
        source_id = row['source_id']
        with np.load(args.review / 'source' / (source_id + '-source.npz'), allow_pickle=False) as archive:
            source = archive['roots'].copy()
        with np.load(args.review / row['mode'] / (source_id + '.npz'), allow_pickle=False) as archive:
            actual = archive['roots'].copy()
        candidate = row['candidate']
        # The end of the measured 16-frame plateau; use the same plane for both
        # recorded modes. Never stop, crop, or repair an actor at that plane.
        index = candidate['plateau_start'] + 15 - candidate['start'] - 4
        if not 0 <= index < len(source):
            raise ValueError('No real source goal available in exported reference')
        goal = source[index]
        direction = np.asarray(row['command_velocity_mps'])
        direction /= np.linalg.norm(direction)
        progress = (actual[:, [0, 2]] - goal[[0, 2]]) @ direction
        crossed = np.flatnonzero(progress >= 0)
        rows.append({'source_id': source_id, 'mode': row['mode'],
                     'goal_source_frame': candidate['plateau_start'] + 15,
                     'goal_world_xyz': goal.tolist(), 'goal_reached': bool(len(crossed)),
                     'first_generated_frame_at_goal': int(crossed[0]) if len(crossed) else None,
                     'final_distance_past_goal_m': float(progress[-1]),
                     'final_root_height_error_m': float(actual[-1, 1] - goal[1]),
                     'postprocessing': None, 'quality_approved': False})
    with args.output.open('x') as stream:
        json.dump({'rows': rows, 'quality_approved': False,
                   'scope': 'Additional source-defined progress measurement; original metrics unchanged'}, stream, indent=2)
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
