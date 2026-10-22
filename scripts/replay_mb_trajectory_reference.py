"""Replay saved native MB plans through whole-trajectory reference constraints."""

import argparse
import json
from pathlib import Path

import numpy as np

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.mb_sonic.reference import SomaReference, file_sha256
from motionbricks.mb_sonic.retarget import CalibratedSomaG1Adapter
from motionbricks.mb_sonic.trajectory_reference import TrajectoryReferenceSolver
from scripts.analyze_mb_reference_support import decode, transition
from scripts.analyze_mb_reference_jitter import distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--position-margin', type=float, default=1e-7)
    parser.add_argument('--until-sim-time', type=float)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    rows = [json.loads(line) for line in args.recording.read_text().splitlines()]
    if not any(row['kind'] == 'explicit_stop' for row in rows):
        raise ValueError('Recorded physical episode must be closed')
    log_hash = file_sha256(args.recording)
    config = json.loads(args.config.read_text())
    adapter = CalibratedSomaG1Adapter.load(config['calibration_path'], robot_model_path=config['robot_model'],
                                         representation=build_native_representation())
    solver = TrajectoryReferenceSolver(adapter.kinematics, iterations=args.iterations,
        position_margin_rad=args.position_margin)
    artifacts = {row['record_id']: row for row in rows if row['kind'] == 'numeric_artifact_written'}
    adoptions = [row for row in rows if row['kind'] == 'reference_packet_adopted']
    initial = [row['actual'] for row in rows if row['kind'] == 'actual_history_warmup'][-1]
    candidates, original, jumps, old_jumps, reports = {}, {}, [], [], []
    previous = None
    for event in adoptions:
        if args.until_sim_time is not None and event['sim_time_s'] > args.until_sim_time:
            break
        receipt = artifacts[event['plan_id']]
        if file_sha256(receipt['path']) != receipt['sha256']:
            raise ValueError('Saved native MB artifact changed')
        with np.load(receipt['path'], allow_pickle=False) as archive:
            record = decode(json.loads(str(archive['metadata_json']))['record'], archive)
        raw = record['raw_source']
        raw['bone_names'] = tuple(raw['bone_names'])
        source = SomaReference(**raw)
        baseline = record['adapted_reference']
        parent = baseline['adaptation']['reference_constraints']['boundary_plan_id']
        if parent is None or parent == 'measured_startup_state':
            boundary = dict(initial_qpos=np.array(initial['qpos']), initial_qvel=np.array(initial['qvel']),
                initial_time_s=initial['time_s'], plan_id='measured_startup_state')
        else:
            old = candidates[parent]
            boundary = dict(timestamps_s=old.timestamps_s, root_pose=old.qpos[:, :7],
                joint_positions=old.qpos[:, 7:], plan_id=parent,
                adoption_until_s=record['packet']['issued_from_sim_time']+config['max_generation_state_age_s'])
        try:
            candidate = adapter.retarget_packet(source, start_time_s=baseline['timestamps_s'][0],
                consumer_horizon_s=config['consumer_horizon_s'],
                required_future_s=.16, constraints=solver, boundary=boundary)
        except Exception as error:
            failure = dict(status='FAILED', optimizer_updates=0, physics_resimulations=0,
                recording_sha256=log_hash, completed_plans=len(reports), failed_plan_id=event['plan_id'],
                policy=solver.identity, error=f'{type(error).__name__}:{error}')
            with args.output.open('x', encoding='utf-8') as stream:
                json.dump(failure, stream, indent=2)
            raise
        report = dict(candidate.adaptation['reference_constraints'])
        report['plan_id'] = event['plan_id']
        report['recorded_joint_difference_rad'] = float(np.max(np.abs(candidate.qpos[:, 7:]-baseline['qpos'][:, 7:])))
        report['recorded_root_xyz_difference_m'] = float(np.max(np.abs(candidate.qpos[:, :3]-baseline['qpos'][:, :3])))
        reports.append(report)
        if previous is not None:
            body = candidate.body_names
            feet = [body.index(name) for name in ('left_ankle_roll_link', 'right_ankle_roll_link')]
            keys = ('timestamps_s', 'qpos', 'qvel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'joint_names')
            old = candidates[previous]
            a = {key: getattr(old, key) for key in keys}
            b = {key: getattr(candidate, key) for key in keys}
            jumps.append(transition(a, b, event['sim_time_s'], feet, body.index('torso_link')))
            old_jumps.append(transition(original[previous], baseline, event['sim_time_s'], feet, body.index('torso_link')))
        candidates[event['plan_id']], original[event['plan_id']] = candidate, baseline
        previous = event['plan_id']
    metrics = ('current_joint_position_jump_max_rad', 'current_joint_velocity_jump_max_rad_s',
        'current_root_position_jump_m', 'current_root_orientation_jump_deg',
        'tracker_future_joint_position_jump_max_rad', 'tracker_future_joint_velocity_jump_max_rad_s',
        'tracker_future_torso_orientation_jump_max_deg', 'future_foot_position_jump_max_m')
    report = dict(schema='motionbricks.mb_sonic.trajectory_reference_replay.v1',
        recording=str(args.recording.resolve()), recording_sha256=log_hash, optimizer_updates=0,
        physics_resimulations=0, source_plans=len(reports), policy=solver.identity,
        solve_wall_ms=distribution([row['wall_ns']/1e6 for row in reports]),
        maximum_velocity_ratio=max(row['maximum_velocity_ratio'] for row in reports),
        minimum_joint_margin_rad=min(row['minimum_position_margin_rad'] for row in reports),
        committed_prefix_error=max(row['committed_prefix_error'] for row in reports),
        maximum_recorded_joint_difference_rad=max(row['recorded_joint_difference_rad'] for row in reports),
        maximum_recorded_root_xyz_difference_m=max(row['recorded_root_xyz_difference_m'] for row in reports),
        baseline={key: distribution([row[key] for row in old_jumps]) for key in metrics},
        candidate={key: distribution([row[key] for row in jumps]) for key in metrics})
    if file_sha256(args.recording) != log_hash:
        raise ValueError('Recording changed during replay')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
