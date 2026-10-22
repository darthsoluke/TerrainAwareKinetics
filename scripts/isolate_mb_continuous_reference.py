"""Bake one complete G1 reference, then track it without online generation."""

import argparse
from dataclasses import fields
import json
from pathlib import Path
from time import perf_counter_ns
import traceback

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.columnar_physics import pack_physics_states
from motionbricks.mb_sonic.mb_reference import decode_soma_reference
from motionbricks.mb_sonic.physics_numerics import configure_physics_numerics
from motionbricks.mb_sonic.reference import G1Reference, OFFICIAL_BODY_NAMES, file_sha256, load_soma_reference
from motionbricks.mb_sonic.retarget import CalibratedSomaG1Adapter
from motionbricks.mb_sonic.runtime import (NumericArtifactWriter, PhysicsRecorder, RuntimeTick,
    SonicPhysicsRuntime, _load_tracker_policy, _numeric_snapshot, load_physics_scene)
from motionbricks.mb_sonic.scene import load_scene
from motionbricks.mb_sonic.trajectory_reference import TrajectoryReferenceSolver
from scripts.audit_mb_tracker_binary import decode


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def load_record(path):
    with np.load(path, allow_pickle=False) as archive:
        return decode(json.loads(str(archive['metadata_json']))['record'], archive)


def load_fixed_reference(case):
    receipt = json.loads((case/'retarget_complete.json').read_text())
    if file_sha256(receipt['reference_path']) != receipt['reference_sha256']:
        raise ValueError('Frozen reference changed after retargeting')
    values = load_record(receipt['reference_path'])
    values['body_names'], values['joint_names'] = tuple(values['body_names']), tuple(values['joint_names'])
    return G1Reference(**values), receipt


def reference_from_poses(adapter, times, qpos, source, source_sha):
    positions, rotations = adapter.kinematics.fk(qpos)
    ids = adapter.kinematics.output_ids
    positions, rotations = positions[:, ids], rotations[:, ids]
    dt = np.diff(times)
    linear = np.diff(positions, axis=0)/dt[:, None, None]
    angular = Rotation.from_matrix((rotations[1:]@rotations[:-1].swapaxes(-1, -2)).reshape(-1, 3, 3))
    angular = angular.as_rotvec().reshape(-1, len(ids), 3)/dt[:, None, None]
    linear, angular = np.r_[linear, linear[-1:]], np.r_[angular, angular[-1:]]
    joint_velocity = np.diff(qpos[:, 7:], axis=0)/dt[:, None]
    qvel = np.c_[linear[:, 0], np.einsum('nij,nj->ni', rotations[:, 0].swapaxes(-1, -2), angular[:, 0]),
                  np.r_[joint_velocity, joint_velocity[-1:]]]
    return G1Reference(times, qpos, qvel, OFFICIAL_BODY_NAMES, positions,
        Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(len(times), len(ids), 4)[..., [3, 0, 1, 2]],
        linear, angular, source.scene_id, source.command_id, float(times[-1]),
        provenance=dict(kind='calibrated_soma_reference', calibration=adapter.calibration,
            source_provenance=dict(source.provenance), source_sha256=source_sha,
            reference_mode='complete_prebaked_track; no_online_switches',
            velocity_source='global50Hzforward_difference;terminal_reuses_last'),
        adaptation=dict(root_mapping='authorized_reference_XYZ_SO3_constraints', physical_state_writes=0))


def retarget(case, config, *, limb_rotation_source='markers', foot_ground=False, support_constraints=False,
             initial_reference_case=None):
    import torch
    from motionbricks.data.native_mb_data import build_native_representation

    started = perf_counter_ns()
    if (case/'retarget_complete.json').exists() or (case/'retarget_failed.json').exists():
        raise ValueError('Retarget evidence already exists; no automatic rerun')
    torch.set_num_threads(2)
    source_path = case/'generation/proposal.npz'
    source_sha = file_sha256(source_path)
    source = load_soma_reference(source_path)
    rep = build_native_representation()
    adapter = CalibratedSomaG1Adapter.load(config['calibration_path'],
        robot_model_path=config['robot_model'], representation=rep)
    with np.load(case/'scene/source_scene.npz', allow_pickle=False) as archive:
        seed = decode_soma_reference(archive['initial_history'][0], (np.arange(4)-3)/30,
            rep, scene_id=source.scene_id, command_id='factual_idle_seed', trusted_until_s=0.,
            provenance=dict(kind='source_seed'))
    seed_q, _ = adapter._map_anatomical(seed.root_positions_m, seed.world_positions_m, seed.bone_names,
        root_rotations=seed.global_rot_mats[:, 0])
    initial = seed_q[-1].copy()
    if initial_reference_case is not None:
        baseline_reference, baseline_receipt = load_fixed_reference(initial_reference_case)
        np.testing.assert_array_equal(initial, baseline_reference.qpos[0])
    elif limb_rotation_source != 'markers' or foot_ground:
        raise ValueError('Repair comparison requires the explicit immutable baseline initial reference')
    adapter.limb_rotation_source = limb_rotation_source
    raw_q, residual = adapter._map_anatomical(source.root_positions_m, source.world_positions_m,
        source.bone_names, root_rotations=source.global_rot_mats[:, 0], global_rotations=source.global_rot_mats)
    options = config['reference_constraints']
    solver_options = {key: options[key] for key in
        ('iterations', 'acceleration_time_s', 'jerk_time_s', 'overlap_time_s', 'overlap_weight')}
    if foot_ground:
        from motionbricks.mb_sonic.reference_ground import ReferenceFootGround
        scene, scene_model, _ = load_physics_scene(case/'scene/scene_manifest.json', config['robot_model'],
            contact_pair_binding=config['robot_contact_pairs'], flat_ground_primitive=config['flat_ground_primitive'])
        support = (dict(scene=scene, support_times=source.timestamps_s,
            support_confidence=source.raw_features[:, rep.indices['foot_contacts']]) if support_constraints else {})
        solver_options.update(foot_ground=ReferenceFootGround(scene_model, **support), iterations=4)
    elif support_constraints:
        raise ValueError('Support tasks require original foot-ground collision constraints')
    solver = TrajectoryReferenceSolver(adapter.kinematics, **solver_options)
    horizon, commit = config['consumer_horizon_s'], config['mb_adopt_interval_s']
    end = round(source.trusted_until_s*50)/50
    final_start = round((end-horizon)*50)/50
    starts = np.arange(0., final_start+1e-8, commit).tolist()
    if not np.isclose(starts[-1], final_start, atol=1e-8, rtol=0):
        starts.append(final_start)
    poses, timestamps, reports, previous = [], [], [], None
    for index, start in enumerate(starts):
        if previous is None:
            boundary = dict(initial_qpos=initial, initial_qvel=np.zeros(35), initial_time_s=0.,
                plan_id='mapped_factual_idle_seed')
        else:
            boundary = dict(timestamps_s=previous.timestamps_s, root_pose=previous.qpos[:, :7],
                joint_positions=previous.joint_pos, plan_id=f'offline_window_{index-1}',
                adoption_until_s=start+config['max_generation_state_age_s'])
        current = adapter.retarget_packet(source, start_time_s=start, consumer_horizon_s=horizon,
            required_future_s=.16, constraints=solver, boundary=boundary)
        stop = starts[index+1] if index+1 < len(starts) else end+.02
        keep = current.timestamps_s < stop-1e-8
        poses.append(current.qpos[keep])
        timestamps.append(current.timestamps_s[keep])
        reports.append(dict(start_s=start, commit_until_s=stop,
            **current.adaptation['reference_constraints']))
        previous = current
    times, qpos = np.concatenate(timestamps), np.concatenate(poses)
    if not np.allclose(np.diff(times), .02, atol=1e-8, rtol=0) or times[-1] != end:
        raise ValueError('Offline G1 track has a gap or duplicate; no interpolation repair')
    reference = reference_from_poses(adapter, times, qpos, source, source_sha)
    audit = adapter._constraint_audit(qpos, times, scope='complete_prebaked50Hz_track')
    if audit['status'] != 'PASS':
        raise ValueError(f'Complete G1 reference violates original bounds: {audit}')
    with PhysicsRecorder(case/'reference_artifacts.jsonl', dict(schema='continuous_reference_records.v1', optimizer_updates=0)) as recorder:
        writer = NumericArtifactWriter(case/'reference_numeric', recorder, pending_plans=1)
        writer._persist(('raw_anatomical_g1', _numeric_snapshot(dict(timestamps_s=source.timestamps_s,
            qpos=raw_q, mapping_residual=residual, valid_for_actuation=False,
            source_audit=adapter._constraint_audit(raw_q, source.timestamps_s, scope='raw_anatomical_mapping'))), 0))
        writer._persist(('fixed_g1_reference', _numeric_snapshot(reference), 0))
    receipts = [json.loads(row) for row in (case/'reference_artifacts.jsonl').read_text().splitlines()]
    receipt = next(row for row in receipts if row.get('record_id') == 'fixed_g1_reference')
    if file_sha256(source_path) != source_sha:
        raise ValueError('MB source changed during retargeting')
    report = dict(schema='continuous_reference_retarget.v1', optimizer_updates=0,
        source_sha256=source_sha, reference_path=receipt['path'], reference_sha256=receipt['sha256'],
        samples=len(times), trusted_until_s=end, offline_window_count=len(reports),
        online_reference_switches=0, window_policy='original0.76s/0.48scommits;all_baked_before_tracking',
        initial_pose_source='mapped_original_factual_idle_seed;reference_only_initial_boundary',
        limb_rotation_source=limb_rotation_source, foot_ground_constraints=foot_ground,
        support_constraints=support_constraints,
        baseline_initial_reference_sha256=(None if initial_reference_case is None else baseline_receipt['reference_sha256']),
        initial_qpos=initial.tolist(), initial_qvel=np.zeros(35).tolist(),
        source_audit=audit, reports=reports, elapsed_ns=perf_counter_ns()-started)
    save_json(case/'retarget_complete.json', report)
    print(json.dumps({key: report[key] for key in ('samples', 'trusted_until_s', 'offline_window_count', 'online_reference_switches')}, indent=2))


def track(case, config, seconds, output):
    output.mkdir(parents=True, exist_ok=False)
    reference, receipt = load_fixed_reference(case)
    scene, model, robot = load_physics_scene(case/'scene/scene_manifest.json', config['robot_model'],
        contact_pair_binding=config['robot_contact_pairs'], flat_ground_primitive=config['flat_ground_primitive'])
    policy, _, _ = _load_tracker_policy(config)
    fall_bodies = {model.body(name).id for name in ('pelvis', 'torso_link')}

    def failure(state):
        margin = np.minimum(state.joint_positions-physics.joint_limits[:, 0], physics.joint_limits[:, 1]-state.joint_positions)
        if margin.min() < -1e-6:
            return 'actual_joint_limit'
        contacts = state.contacts
        hit = (np.isin(contacts.body_ids, list(fall_bodies)).any(axis=1)
               & (contacts.body_ids == 0).any(axis=1) & (contacts.wrench_contact[:, 0] > 1))
        return 'walking_task:pelvis_or_torso_scene_contact' if hit.any() else None

    physics = SonicPhysicsRuntime(scene, model, policy, tracker_backend='protomotions', physics_dt=.001,
        robot_model_sha256=config['robot_model_sha256'], contact_layout='batch',
        physics_execution='cached_forward_v1', warmup_control_ticks=config['warmup_control_ticks'],
        failure_evaluator=failure)
    numerics = configure_physics_numerics(physics.model, .001, config)
    qpos, qvel = model.qpos0.copy(), np.zeros(model.nv)
    qpos[:7], qpos[physics.qpos_indices] = reference.qpos[0, :7], reference.joint_pos[0]
    physics.reset(qpos, qvel)
    started = perf_counter_ns()
    metadata = dict(schema='motionbricks.mb_sonic.physics_record.v1', optimizer_updates=0,
        mode='fixed_continuous_reference', allocation_id='continuous-reference-cpu-v1',
        tracker_backend='protomotions', physics_dt_s=.001, controller_hz=50,
        robot=robot, manifest_sha256=physics.manifest_sha256,
        actuator_control=physics.control_metadata, physics_numerics=numerics,
        reference_sha256=receipt['reference_sha256'], online_generation=False,
        online_reference_switches=0, source_scene_sha256=file_sha256(case/'scene/source_scene.npz'))
    count, actor_ns = 0, []
    with PhysicsRecorder(output/'tracking.jsonl', metadata) as recorder:
        writer = NumericArtifactWriter(output/'numeric', recorder, pending_plans=1)
        recorder.write(dict(kind='explicit_initial_reset', actual=_numeric_snapshot(physics.snapshot())))
        for state in physics.warmup_history():
            recorder.write(dict(kind='actual_history_warmup', actual=_numeric_snapshot(state)))
        while physics.time_s < seconds-1e-8 and physics.terminal_reason is None:
            if perf_counter_ns()-started > 300e9:
                raise TimeoutError('Offline tracking exceeded its declared300s wall cap')
            before = perf_counter_ns()
            window, command = physics.begin_control(reference)
            actor_ns.append(perf_counter_ns()-before)
            states = []
            for _ in range(physics.substeps):
                state = physics._advance_pd(command.joint_target, substeps=1)[0]
                if not state.root_ground_known:
                    state = physics.terminate('unknown_canonical_ground_domain')
                states.append(state)
                if physics.terminal_reason:
                    break
            if len(states) == physics.substeps and physics.terminal_reason is None:
                physics._append_proprioception()
            tick = RuntimeTick(physics.scene_id, 'fixed_reference', 'fixed_continuous_reference',
                float(window.timestamps_s[0]), 0, dict(source_sha256=receipt['source_sha256']),
                window, command, tuple(states), dict(policy_total=actor_ns[-1]))
            values = {field.name: getattr(tick, field.name) for field in fields(tick)
                      if field.name != 'physics_states'}
            key = f'tick:{states[-1].state_version}'
            writer._persist((key, _numeric_snapshot(dict(kind='full_policy_tick_columnar',
                tick=values, physics=pack_physics_states(tuple(states)))), 0))
            recorder.write(dict(kind='policy_tick_binary', numeric_record_id=key,
                scene_id=physics.scene_id, command_id='fixed_reference',
                reference_time_s=tick.reference_time_s, actual_time_s=states[-1].time_s,
                state_version=states[-1].state_version, captured_at_monotonic_ns=states[-1].captured_at_monotonic_ns))
            count += 1
        recorder.write(dict(kind='terminal_failure' if physics.terminal_reason else 'episode_completed',
            actual_time_s=physics.time_s, terminal_reason=physics.terminal_reason))
        recorder.write(dict(kind='explicit_stop', actual_time_s=physics.time_s, automatic_reset=False))
    if file_sha256(receipt['reference_path']) != receipt['reference_sha256']:
        raise ValueError('Fixed reference changed during tracking')
    report = dict(schema='continuous_reference_tracking.v1', optimizer_updates=0,
        requested_end_s=seconds, actual_end_s=physics.time_s, control_ticks=count,
        completed=physics.terminal_reason is None and physics.time_s >= seconds-1e-8,
        terminal_reason=physics.terminal_reason, reference_sha256=receipt['reference_sha256'],
        online_generation=False, online_reference_switches=0, root_residual=False, post_filter=False,
        actor_p99_ms=float(np.percentile(actor_ns, 99)/1e6), wall_ns=perf_counter_ns()-started,
        wall_semantics='unpaced CPU offline simulation; not browserFPS or interactive latency')
    save_json(output/'summary.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('retarget', 'track'), required=True)
    parser.add_argument('--seconds', type=float, default=16.)
    parser.add_argument('--track-output', type=Path)
    parser.add_argument('--limb-rotation-source', choices=('markers', 'native_material_axes'), default='markers')
    parser.add_argument('--foot-ground', action='store_true')
    parser.add_argument('--support-constraints', action='store_true')
    parser.add_argument('--initial-reference-case', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        if args.stage == 'retarget':
            retarget(args.case, config, limb_rotation_source=args.limb_rotation_source,
                foot_ground=args.foot_ground, support_constraints=args.support_constraints,
                initial_reference_case=args.initial_reference_case)
        else:
            if args.track_output is None:
                parser.error('Tracking requires an explicit fresh --track-output directory')
            track(args.case, config, args.seconds, args.track_output)
    except Exception as error:
        path = (args.track_output/'failed.json' if args.stage == 'track' and args.track_output is not None
                else args.case/f'{args.stage}_failed.json')
        if not path.exists():
            save_json(path, dict(error=f'{type(error).__name__}:{error}', traceback=traceback.format_exc(), optimizer_updates=0))
        raise


if __name__ == '__main__':
    main()
