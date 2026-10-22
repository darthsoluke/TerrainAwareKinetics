"""Zero-training paired tracker replay with identical recorded MB/G1 references."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from motionbricks.mb_sonic.physics_numerics import configure_physics_numerics
from motionbricks.mb_sonic.reference import G1Reference, file_sha256
from motionbricks.mb_sonic.root_velocity_task import RootVelocityTask
from motionbricks.mb_sonic.runtime import CurrentTrackerObservation, SonicPhysicsRuntime, _load_tracker_policy, load_physics_scene
from scripts.analyze_mb_reference_jitter import distribution, temporal_metrics
from scripts.audit_mb_tracker_binary import decode, read_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--torque-fraction', type=float, default=.08)
    parser.add_argument('--seconds', type=float, default=4.)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    metadata, ticks, artifacts, _, _ = read_index(args.recording)
    intents, uses = {}, []
    for line in args.recording.read_text().splitlines():
        row = json.loads(line)
        if row['kind'] == 'intent_received':
            intents[row['raw_intent']['command_id']] = row['raw_intent']
        if row['kind'] == 'reference_packet_used':
            uses.append(row)
    scene_id = metadata['scene_manifest']['scene_id']
    scene, model, _ = load_physics_scene(Path(config['scene_root'])/scene_id/'scene_manifest.json',
        config['robot_model'], contact_pair_binding=config['robot_contact_pairs'],
        flat_ground_primitive=config['flat_ground_primitive'])
    policy, _, _ = _load_tracker_policy(config)
    physics = SonicPhysicsRuntime(scene, model, policy, physics_dt=.001, tracker_backend='protomotions',
        physics_execution='cached_forward_v1', warmup_control_ticks=0, contact_layout='batch',
        robot_model_sha256=config['robot_model_sha256'])
    configure_physics_numerics(physics.model, .001, config)
    task = RootVelocityTask(physics, torque_fraction=args.torque_fraction)
    verified = set()

    def load(tick):
        receipt = artifacts[tick['numeric_record_id']]
        if receipt['path'] not in verified:
            assert file_sha256(receipt['path']) == receipt['sha256']
            verified.add(receipt['path'])
        with np.load(receipt['path'], allow_pickle=False) as archive:
            return decode(json.loads(str(archive['metadata_json']))['record'], archive)

    rows, timings = [], []
    tick_values = list(ticks.values())
    for command_id in (2, 4, 6, 8):
        selected = [i for i, tick in enumerate(tick_values) if tick['command_id'] == command_id]
        start = selected[0]
        selected = selected[:round(args.seconds/.02)]
        initial = load(tick_values[start-1])['physics']
        first_window = load(tick_values[start])['tick']['adapted_reference']
        intent = intents[command_id]
        direction = np.asarray(intent['velocity_xy_mps'])
        direction = direction/np.linalg.norm(direction)
        for corrected in (False, True):
            physics.reset(initial['qpos'][-1], initial['qvel'][-1])
            physics.data.time = first_window['timestamps_s'][0]
            physics.history = CurrentTrackerObservation()
            policy.heading_alignment = None
            physics.heading_initialized = False
            source_inputs = load(tick_values[start])['tick']['policy_command']['inputs']
            start_pos = physics.data.qpos[:3].copy()
            progress, min_margin, root_z, forces, peak_correction = [], np.inf, np.inf, 0., 0.
            trace, terminal = [], None
            signals = {k: [] for k in ('actor', 'applied', 'actual', 'residual', 'motor_force_substeps')}
            for index in selected:
                payload = load(tick_values[index])
                reference = payload['tick']['adapted_reference']
                reference['joint_names'], reference['body_names'] = tuple(reference['joint_names']), tuple(reference['body_names'])
                window = G1Reference(**reference)
                physics._append_proprioception()
                _, command = physics.begin_control(window)
                if index == start:
                    policy.previous_target = source_inputs['historical_processed_actions'][0, 0].copy()
                    command = policy.infer(window, physics.history,
                        anchor_quat_wxyz=physics.data.xquat[physics.anchor_body_id])
                if corrected:
                    target, report = task.solve(physics.data, command.joint_target,
                        velocity_xy_mps=intent['velocity_xy_mps'], yaw_rate_radps=intent['yaw_rate_radps'],
                        heading_rad=uses[index]['derived_heading_rad'])
                    timings.append(report['timing_ns']['total']/1e6)
                    peak_correction = max(peak_correction, float(np.abs(report['normalized_torque_correction']).max()))
                    policy.previous_target = target.copy()
                else:
                    target = command.joint_target
                states = physics._advance_pd(target)
                signals['actor'].append(command.joint_target.copy())
                signals['applied'].append(target.copy())
                signals['residual'].append(target-command.joint_target)
                signals['actual'].append(states[-1].joint_positions.copy())
                signals['motor_force_substeps'].extend(s.actuator_forces.copy() for s in states)
                for state in states:
                    margin = np.minimum(state.joint_positions-physics.joint_limits[:, 0],
                        physics.joint_limits[:, 1]-state.joint_positions)
                    min_margin = min(min_margin, float(margin.min()))
                    root_z = min(root_z, float(state.qpos[2]))
                    forces = max(forces, float(np.max(np.abs(state.actuator_forces))))
                    assert not np.any(state.root_qfrc_actuator) and not np.any(state.qfrc_applied) and not np.any(state.xfrc_applied)
                state = states[-1]
                progress.append(float(state.qvel[:2]@direction))
                trace.append(dict(time_s=state.time_s, root_xy=state.qpos[:2].tolist(),
                    velocity_xy=state.qvel[:2].tolist()))
                if min_margin < -1e-6 or root_z < .45:
                    terminal = 'actual_joint_limit' if min_margin < -1e-6 else 'low_root'
                    break
            rows.append(dict(command_id=command_id, corrected=corrected,
                seconds=len(progress)*.02, final_displacement_m=(physics.data.qpos[:3]-start_pos).tolist(),
                average_intent_velocity_mps=float(np.mean(progress)), minimum_joint_margin_rad=min_margin,
                minimum_root_z_m=root_z, maximum_motor_force_nm=forces, terminal=terminal,
                maximum_normalized_correction=peak_correction, trace=trace))
            rows[-1]['oscillation'] = {k: dict(all_joints=temporal_metrics(np.array(v),
                .001 if k.endswith('substeps') else .02), leg_joints=temporal_metrics(np.array(v)[:, :12],
                .001 if k.endswith('substeps') else .02)) for k, v in signals.items()}
            print(json.dumps({k: v for k, v in rows[-1].items() if k not in ('trace', 'oscillation')}), flush=True)
    report = dict(schema='motionbricks.root_velocity_task.paired_replay.v1', optimizer_updates=0,
        recording=str(args.recording), recording_sha256=file_sha256(args.recording),
        config_sha256=file_sha256(args.config), root_task=task.identity, verified_artifacts=len(verified),
        identical_reference=True, actual_feedback_to_generator=False,
        solve_ms=distribution(timings), results=rows)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
