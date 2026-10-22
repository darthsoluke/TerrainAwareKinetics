"""Separate reference, frozen actor, execution residual and physical oscillation."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from motionbricks.mb_sonic.reference import file_sha256
from scripts.audit_mb_tracker_binary import decode, read_index
from scripts.analyze_mb_reference_jitter import temporal_metrics


def signal_metrics(values, dt):
    values = np.asarray(values, dtype=np.float64)
    result = temporal_metrics(values, dt)
    frequency, density = welch(values, fs=1/dt, axis=0,
        nperseg=min(len(values), round(4/dt)), detrend='linear')
    spectrum = density.sum(axis=1)
    spacing = frequency[1]-frequency[0]
    total = float(spectrum.sum()*spacing)
    high = float(spectrum[frequency >= 8].sum()*spacing)
    result.update(power_fraction_above_8hz=high/total if total > 1e-24 else 0.,
        high_frequency_power=high, total_spectral_power=total,
        spectral_resolution_hz=float(spacing), spectral_window_s=min(len(values)*dt, 4.))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=4.)
    args = parser.parse_args()
    metadata, ticks, artifacts, _, _ = read_index(args.recording)
    before = file_sha256(args.recording)
    intents = {}
    for line in args.recording.read_text().splitlines():
        row = json.loads(line)
        if row['kind'] == 'intent_received':
            intents[row['raw_intent']['command_id']] = row['raw_intent']
    results = []
    verified = []
    for command_id, intent in intents.items():
        selected = [t for t in ticks.values() if t['command_id'] == command_id]
        if len(selected) < 100:
            continue
        selected = selected[:round(args.seconds/.02)]
        signals = {key: [] for key in ('reference', 'future_reference', 'actor', 'applied', 'residual',
                                      'actual', 'actual_substeps', 'motor_force_substeps')}
        for tick in selected:
            receipt = artifacts[tick['numeric_record_id']]
            if file_sha256(receipt['path']) != receipt['sha256']:
                raise ValueError('Recorded numeric artifact changed')
            verified.append(receipt['sha256'])
            with np.load(receipt['path'], allow_pickle=False) as archive:
                record = decode(json.loads(str(archive['metadata_json']))['record'], archive)
            command = record['tick']['policy_command']
            task = command.get('execution_task')
            actor = task['native_joint_target'] if task is not None else command['joint_target']
            signals['reference'].append(record['tick']['adapted_reference']['qpos'][0, 7:])
            signals['future_reference'].append(command['inputs']['mimic_future_dof_pos'][0, -1])
            signals['actor'].append(actor)
            signals['applied'].append(command['joint_target'])
            signals['residual'].append(command['joint_target']-actor)
            signals['actual'].append(record['physics']['joint_positions'][-1])
            signals['actual_substeps'].extend(record['physics']['joint_positions'])
            signals['motor_force_substeps'].extend(record['physics']['actuator_forces'])
        report = {}
        for name, values in signals.items():
            values = np.array(values)
            dt = metadata['physics_dt_s'] if name.endswith('substeps') else .02
            report[name] = dict(all_joints=signal_metrics(values, dt),
                leg_joints=signal_metrics(values[:, :12], dt))
        results.append(dict(command_id=command_id, velocity_xy_mps=intent['velocity_xy_mps'],
            yaw_rate_radps=intent['yaw_rate_radps'], seconds=len(selected)*.02, signals=report))
    if before != file_sha256(args.recording):
        raise ValueError('Close the episode before oscillation analysis')
    report = dict(schema='motionbricks.mb_sonic.control_oscillation.v1',
        recording=str(args.recording), recording_sha256=before, optimizer_updates=0, new_physics_steps=0,
        verified_artifacts=len(verified), results=results)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(verified_artifacts=len(verified), commands=len(results),
        output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
