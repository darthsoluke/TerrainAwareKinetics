"""Compare completed real-time physics against the exact offline source trial."""

import argparse
import json
from pathlib import Path

import numpy as np

from motionbricks.mb_sonic.columnar_physics import DENSE_FIELDS
from motionbricks.mb_sonic.reference import file_sha256
from scripts.audit_mb_tracker_binary import read_index
from scripts.isolate_mb_continuous_reference import load_record, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--live', type=Path, required=True)
    parser.add_argument('--browser', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    loaded = []
    for path in (args.baseline, args.live):
        before = file_sha256(path)
        meta, ticks, artifacts, _, _ = read_index(path)
        values, targets = {key:[] for key in ('time_s', *DENSE_FIELDS)}, []
        for tick in ticks.values():
            receipt = artifacts[tick['numeric_record_id']]
            if file_sha256(receipt['path']) != receipt['sha256']:
                raise ValueError('Physical record hash mismatch')
            record = load_record(receipt['path'])
            for key in values:
                values[key].append(record['physics'][key])
            targets.append(record['tick']['policy_command']['joint_target'])
        values = {key:np.concatenate(value) for key,value in values.items()}
        values['policy_target'] = np.asarray(targets)
        if file_sha256(path) != before:
            raise ValueError('Physical log changed during comparison')
        initial = next(json.loads(line)['actual'] for line in path.read_text().splitlines()
            if json.loads(line)['kind'] == 'explicit_initial_reset')
        loaded.append((values, initial, before))
    original, live = loaded[0][0], loaded[1][0]
    differences = {}
    for key in original:
        if original[key].shape != live[key].shape:
            raise ValueError(f'Real-time physical coverage mismatch: {key}')
        differences[key] = dict(bitwise_equal=np.array_equal(original[key], live[key]),
            maximum_absolute_difference=float(np.max(np.abs(original[key]-live[key]))))
    for key in ('qpos', 'qvel'):
        np.testing.assert_array_equal(loaded[0][1][key], loaded[1][1][key])
    browser = json.loads(args.browser.read_text())
    report = dict(schema='mb_fixed_reference_realtime_equivalence.v1', optimizer_updates=0,
        original_recording_sha256=loaded[0][2], live_recording_sha256=loaded[1][2],
        exact_initial_qpos_qvel=True, comparisons=differences,
        all_numeric_fields_bitwise_equal=all(row['bitwise_equal'] for row in differences.values()),
        controlled_substeps=len(live['time_s']), final_time_s=float(live['time_s'][-1]),
        browser_metrics=browser['metrics'], browser_failures=browser['failures'])
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
