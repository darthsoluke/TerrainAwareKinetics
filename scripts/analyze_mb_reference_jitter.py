"""Measure native/reference/action/physical jitter in a closed recorded episode."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.spatial.transform import Rotation, Slerp

from motionbricks.mb_sonic.reference import file_sha256 as digest
from scripts.analyze_mb_reference_support import decode, sample, transition


def distribution(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return dict(count=int(values.size), mean=float(values.mean()),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)), maximum=float(values.max()))


def temporal_metrics(values, dt=.02):
    values = np.asarray(values)
    velocity = np.diff(values, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    frequency, power = welch(values, fs=1 / dt, axis=0, nperseg=min(256, len(values)), detrend='linear')
    total = power.sum()
    aggregate = power.reshape(len(frequency), -1).sum(axis=1)
    return dict(step_max=distribution(np.max(np.abs(np.diff(values, axis=0)), axis=1)),
                velocity_rms=float(np.sqrt(np.mean(velocity ** 2))),
                acceleration_rms=float(np.sqrt(np.mean(acceleration ** 2))),
                jerk_rms=float(np.sqrt(np.mean(jerk ** 2))),
                power_fraction_above_8hz=float(power[frequency >= 8].sum() / total) if total else 0.,
                dominant_nonzero_frequency_hz=float(frequency[1 + np.argmax(aggregate[1:])])
                if aggregate[1:].sum() else None)


def native_at(source, time):
    times = source['timestamps_s']
    if time < times[0] - 1e-8 or time > times[-1] + 1e-8:
        raise ValueError('Requested native sample is outside saved coverage')
    local = np.stack([Slerp(times, Rotation.from_matrix(source['local_rot_mats'][:, joint]))([time]).as_matrix()[0]
                      for joint in range(len(source['bone_names']))])
    points = np.array([[np.interp(time, times, source['world_positions_m'][:, joint, axis])
                        for axis in range(3)] for joint in range(len(source['bone_names']))])
    return local, points - points[:1]


def analyze(recording):
    log_sha = digest(recording)
    lines = [json.loads(line) for line in recording.read_text(encoding='utf-8').splitlines()]
    if not any(row['kind'] == 'explicit_stop' for row in lines):
        raise ValueError('Close the physical episode before forensic analysis')
    artifacts = {row['record_id']: row for row in lines if row['kind'] == 'numeric_artifact_written'}
    ticks = [row for row in lines if row['kind'] == 'policy_tick_binary']
    uses = [row for row in lines if row['kind'] == 'reference_packet_used']
    if len(ticks) != len(uses) or len(ticks) < 5:
        raise ValueError('Expected paired recorded policy ticks and reference receipts')
    verified, plans, jumps, rows = [], {}, [], []

    def load(record_id):
        row = artifacts[record_id]
        path = Path(row['path']).resolve(strict=True)
        if digest(path) != row['sha256']:
            raise ValueError(f'Numeric artifact changed: {record_id}')
        with np.load(path, allow_pickle=False) as archive:
            record = decode(json.loads(str(archive['metadata_json']))['record'], archive)
        verified.append(dict(record_id=record_id, sha256=row['sha256']))
        return record

    previous_id = None
    for tick, use in zip(ticks, uses, strict=True):
        time, plan_id = float(tick['reference_time_s']), use['plan_id']
        if plan_id not in plans:
            record = load(plan_id)
            plans[plan_id] = {key: record[key] for key in ('raw_source', 'adapted_reference')}
        plan = plans[plan_id]
        reference = plan['adapted_reference']
        source = plan['raw_source']
        target = sample(reference, [time])
        native_rotation, native_points = native_at(source, time)
        if previous_id is not None and previous_id != plan_id:
            old = plans[previous_id]
            feet = [reference['body_names'].index(name) for name in
                    ('left_ankle_roll_link', 'right_ankle_roll_link')]
            jump = transition(old['adapted_reference'], reference, time, feet,
                              reference['body_names'].index('torso_link'))
            old_rotation, old_points = native_at(old['raw_source'], time)
            differences = Rotation.from_matrix(native_rotation[1:] @ old_rotation[1:].swapaxes(-1, -2)).magnitude()
            jump.update(native_local_joint_jump_max_rad=float(differences.max()),
                        native_root_relative_point_jump_max_m=float(np.linalg.norm(native_points-old_points, axis=-1).max()),
                        command_changed=use['generated_from_command_seq'] != rows[-1]['command_seq'])
            jumps.append(jump)
        record = load(tick['numeric_record_id'])
        recorded = record['tick']
        if (recorded['raw_reference']['plan_id'] != plan_id
                or not np.array_equal(recorded['adapted_reference']['qpos'][0], target['qpos'][0])):
            raise ValueError('Consumed target differs from the recorded plan')
        physics, command = record['physics'], recorded['policy_command']
        rows.append(dict(time_s=time, plan_id=plan_id, command_seq=use['generated_from_command_seq'],
                         native_points=native_points, g1_joints=target['qpos'][0, 7:],
                         reference_velocity=target['qvel'][0, 6:], action_target=command['joint_target'],
                         actual_joints=physics['joint_positions'][-1],
                         actual_velocity=physics['joint_velocities'][-1]))
        previous_id = plan_id
    times = np.array([row['time_s'] for row in rows])
    if not np.allclose(np.diff(times), .02, rtol=0, atol=1e-8):
        raise ValueError('Spectral analysis requires the original continuous 50Hz control grid')
    adopted = np.array([a['plan_id'] != b['plan_id'] for a, b in zip(rows, rows[1:])])
    control_changed = np.array([a['command_seq'] != b['command_seq'] for a, b in zip(rows, rows[1:])])
    signals = {key: np.array([row[key] for row in rows]) for key in
               ('native_points', 'g1_joints', 'action_target', 'actual_joints')}
    metrics = {}
    for key, signal in signals.items():
        flattened = signal.reshape(len(signal), -1)
        differences = np.max(np.abs(np.diff(flattened, axis=0)), axis=1)
        metrics[key] = dict(temporal_metrics(flattened),
            step_at_adoption=distribution(differences[adopted]),
            step_within_plan=distribution(differences[~adopted]),
            step_at_adoption_same_command=distribution(differences[adopted & ~control_changed]))
    jump_keys = [key for key, value in jumps[0].items() if isinstance(value, float)]
    summary = dict(schema='motionbricks.mb_sonic.reference_jitter.v1',
        recording=str(recording), recording_sha256=log_sha,
        optimizer_updates=0, physics_resimulations=0, samples=len(rows), duration_s=float(times[-1]-times[0]),
        verified_numeric_artifacts=len(verified), used_plans=len(plans), adoptions=int(adopted.sum()),
        command_changes=int(control_changed.sum()),
        adoption_interval_s=distribution(np.diff(times[1:][adopted])),
        joint_names=reference['joint_names'], signals=metrics,
        same_absolute_time_jumps={key: distribution([jump[key] for jump in jumps]) for key in jump_keys},
        same_command_absolute_time_jumps={key: distribution([jump[key] for jump in jumps if not jump['command_changed']])
                                          for key in jump_keys},
        largest_future_joint_jumps=sorted(jumps, key=lambda row: row['tracker_future_joint_position_jump_max_rad'], reverse=True)[:5],
        retarget_constraints={
            'plans_modified': sum(bool(p['adapted_reference']['adaptation']['reference_constraints']['solved_frames']) for p in plans.values()),
            'maximum_modified_joint_rad': max(p['adapted_reference']['adaptation']['reference_constraints']['maximum_joint_change_rad'] for p in plans.values())},
        scope='Measured 50Hz control samples; high-frequency statistics are descriptive, not causal attribution',
        units=dict(native_points='m root-relative SOMA points', g1_joints='rad', action_target='rad', actual_joints='rad'))
    if digest(recording) != log_sha:
        raise ValueError('Recording changed during analysis')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.recording.resolve(strict=True))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in
          ('samples', 'used_plans', 'adoptions', 'command_changes', 'same_absolute_time_jumps', 'retarget_constraints')}, indent=2))


if __name__ == '__main__':
    main()
