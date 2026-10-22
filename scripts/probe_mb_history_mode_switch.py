"""Counterfactual observer-mode switch with saved physical observations fixed."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.mb_sonic.reference import file_sha256
from scripts.analyze_mb_online_history_contract import matrices, locals_from_globals, sample_rotations
from scripts.analyze_mb_reference_jitter import distribution
from scripts.isolate_mb_continuous_reference import load_record, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--history-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.history_review.read_text())
    if file_sha256(args.recording) != review['recording_sha256']:
        raise ValueError('History review refers to another recording')
    receipts = {r['record_id']: r for r in map(json.loads, args.recording.read_text().splitlines())
                if r['kind'] == 'numeric_artifact_written'}
    cached = {}

    def load(key):
        if key not in cached:
            item = receipts[key]
            if file_sha256(item['path']) != item['sha256']:
                raise ValueError('Saved sensor/reference evidence changed')
            cached[key] = load_record(item['path'])
        return cached[key]

    eligible = [h for h in review['histories'] if len(h['correction_segments']) == 14
                and all(p != 'explicit_time_zero_seed_pair' for p in h['prior_ids'])]
    rep = build_native_representation()
    names, parents = tuple(rep.skeleton.bone_order_names), rep.skeleton.joint_parents.numpy()
    legs = [names.index(s+n) for s in ('Left', 'Right') for n in ('Leg', 'Shin', 'Foot')]
    chest = names.index('Chest')
    cases = []
    for index in np.unique(np.linspace(0, len(eligible)-1, 12, dtype=int)):
        item = eligible[index]
        plan = load(item['plan_id'])
        if np.any(plan['command']['velocity_xy_mps']):
            raise ValueError('Selected full-observer record is not the stop branch')
        history = plan['conditioning_history']
        actual = matrices(history['features'], rep.indices['global_rot_data'])
        prior = np.concatenate([sample_rotations(load(key)['raw_source'], np.array([time]))
            for key, time in zip(item['prior_ids'], history['timestamps_s'])])
        prior_local = locals_from_globals(prior, parents)
        moving = np.empty_like(actual)
        for joint, parent in enumerate(parents):
            moving[:, joint] = actual[:, joint] if joint in (0, chest) else moving[:, parent]@prior_local[:, joint]
        original_local = locals_from_globals(actual, parents)
        moving_local = locals_from_globals(moving, parents)
        difference = Rotation.from_matrix((original_local[:, legs]@moving_local[:, legs].swapaxes(-1, -2)).reshape(-1, 3, 3)).magnitude()
        np.testing.assert_array_equal(moving[:, [0, chest]], actual[:, [0, chest]])
        cases.append(dict(plan_id=item['plan_id'], time_s=item['time_s'],
            leg_local_change_max_rad=float(difference.max()),
            leg_local_change_rms_rad=float(np.sqrt(np.mean(difference**2))),
            hips_and_chest_unchanged=True, measured_state_and_prior_unchanged=True))
    report = dict(schema='mb_history_mode_counterfactual.v1', recording_sha256=review['recording_sha256'],
        eligible_histories=len(eligible), cases=cases,
        maximum_leg_change_rad=distribution([c['leg_local_change_max_rad'] for c in cases]),
        held_fixed=['actual measured G1 state', 'native prior', 'absolute times', 'Root and torso corrections'],
        changed_only='observer branch selected by zero versus nonzero translation command',
        optimizer_updates=0, physics_steps=0, MB_calls=0, actor_calls=0,
        scope='same-observation history discontinuity; no counterfactual physical rollout')
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
