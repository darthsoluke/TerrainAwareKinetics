"""Author an observed relaxed idle input, excluding stationary calibration poses.

These semantic selection thresholds are for input authoring, not physics labels,
training losses, or edits to generated motion. Existing bundles stay immutable.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.soma_replay import skeleton_manifest


def candidate_starts(raw, representation):
    if raw.ndim != 2 or raw.shape[1] != len(representation.indices['all']) or not np.isfinite(raw).all():
        raise ValueError('Need finite full native global features')
    if len(raw) < 30:
        return []
    positions = raw[:, representation.indices['ric_data']].reshape(-1, 29, 3)
    names = representation.skeleton.bone_order_names
    relaxed = (raw[:, 1] >= .85) & (raw[:, 1] <= 1.15)
    for side in ('Left', 'Right'):
        hand = positions[:, names.index(side + 'Hand') - 1, 1]
        arm = positions[:, names.index(side + 'Arm') - 1, 1]
        relaxed &= (arm - hand >= .25) & (hand <= raw[:, 1] + .25)
    speed = np.linalg.norm(np.diff(raw[:, [0, 2]], axis=0), axis=-1) * representation.fps
    body_speed = np.linalg.norm(np.diff(positions, axis=0), axis=-1).mean(-1) * representation.fps
    result = []
    for start in range(0, len(raw) - 29):
        if (relaxed[start:start + 30].all() and speed[start:start + 29].max() <= .03
                and body_speed[start:start + 29].mean() <= .03):
            result.append((float((speed[start:start + 29] + body_speed[start:start + 29]).mean()), start))
    return result


def author(dataset, output):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError('Reference artifacts are immutable; use a new path')
    manifest = json.loads((dataset / 'native_manifest.json').read_text())
    records = json.loads((dataset / 'accepted.json').read_text())
    rep = build_native_representation()
    best, count = None, 0
    for row in records:
        if row['split'] != 'train' or row['behavior'] not in ('idle', 'walk', 'run'):
            continue
        shard = dataset / row['shard']
        if file_sha256(shard) != manifest['artifact_sha256'][row['shard']]:
            raise ValueError('Training reference shard hash mismatch')
        with np.load(shard, allow_pickle=False) as data:
            raw = data['global_features'].copy()
        candidates = candidate_starts(raw, rep)
        count += len(candidates)
        if candidates:
            score, start = min(candidates)
            if best is None or score < best[0]:
                best = score, row, start, raw[start:start + 30].copy()
    if best is None:
        raise ValueError('No observed relaxed standing segment; do not synthesize a replacement')
    score, row, start, reference = best
    result = dict(schema='motionbricks.authored_idle.v1', source_id=skeleton_manifest()['source_id'],
        fps=30, features=reference[:4].tolist(), reference_30frames=reference.tolist(),
        provenance=dict(split='train', shard=row['shard'], filename=row['filename'], start_frame=start,
            shard_sha256=file_sha256(dataset / row['shard']), dataset_manifest_sha256=file_sha256(dataset / 'native_manifest.json'),
            accepted_sha256=file_sha256(dataset / 'accepted.json'), selection_score=score, candidate_count=count),
        selection=dict(root_speed_max_mps=.03, mean_joint_speed_max_mps=.03, root_height_range_m=[.85, 1.15],
            hand_below_arm_min_m=.25, hand_above_hip_max_m=.25, window_frames=30,
            role='Observed train-only authored idle input, not played motion or a physical contact guarantee'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result['provenance']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(author(args.dataset, args.output), indent=2))
