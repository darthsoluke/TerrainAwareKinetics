"""Measure planar hindsight-goal ambiguity without changing dataset labels."""

import argparse
import json
from pathlib import Path
import numpy as np

from scripts.train_native_mb import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output')
    manifest = json.loads((args.dataset / 'native_manifest.json').read_text())
    accepted = args.dataset / 'accepted.json'
    if sha256(accepted) != manifest['artifact_sha256']['accepted.json']:
        raise ValueError('Accepted list changed')
    rows = json.loads(accepted.read_text())
    result = {'source_dataset_sha256': sha256(args.dataset / 'native_manifest.json'),
              'scope': 'unweighted train-window diagnostic, not new labels or controller commands',
              'thresholds': {'near_zero_planar_goal_speed_mps': .05,
                             'substantial_vertical_range_m': .15, 'quiet_max_root_speed_mps': .05},
              'horizons': {}}
    for frames in (32, 64):
        counts = {'windows': 0, 'near_zero_planar_goal': 0, 'zero_goal_with_vertical_motion': 0,
                  'quiet_root_windows': 0}
        speeds, examples = [], []
        for row in rows:
            if row['split'] != 'train':
                continue
            path = args.dataset / row['shard']
            if sha256(path) != manifest['artifact_sha256'][row['shard']]:
                raise ValueError('Training shard changed')
            with np.load(path) as data:
                roots = data['global_features'][:, :3]
            for start in range(0, len(roots) - frames + 1, 4):
                future = roots[start+3:start+frames]
                speed = np.linalg.norm(future[-1, [0, 2]] - future[0, [0, 2]]) / ((frames - 4)/30)
                y_range = float(np.ptp(future[:, 1]))
                max_speed = float(np.linalg.norm(np.diff(future, axis=0), axis=-1).max() * 30)
                counts['windows'] += 1
                speeds.append(float(speed))
                if speed <= .05:
                    counts['near_zero_planar_goal'] += 1
                    if y_range > .15:
                        counts['zero_goal_with_vertical_motion'] += 1
                        if len(examples) < 6:
                            examples.append({'shard': row['shard'], 'start': start, 'y_range_m': y_range,
                                             'planar_goal_speed_mps': float(speed)})
                counts['quiet_root_windows'] += int(max_speed <= .05)
        result['horizons'][str(frames)] = {**counts, 'examples': examples,
            'goal_speed_quantiles_mps': np.quantile(speeds, [0, .1, .5, .9, 1]).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
