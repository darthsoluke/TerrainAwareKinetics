"""Read-only analysis of online history identities and reference replacement."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.mb_sonic.reference import file_sha256
from scripts.analyze_mb_reference_jitter import distribution
from scripts.isolate_mb_continuous_reference import load_record, save_json


def matrices(features, indices):
    six = features[:, indices].reshape(len(features), 30, 6)
    first = six[..., :3]/np.linalg.norm(six[..., :3], axis=-1, keepdims=True)
    second = six[..., 3:]-(first*six[..., 3:]).sum(-1, keepdims=True)*first
    second /= np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def locals_from_globals(global_r, parents):
    result = global_r.copy()
    result[:, 1:] = global_r[:, parents[1:]].swapaxes(-1, -2)@global_r[:, 1:]
    return result


def sample_rotations(source, times):
    t = source['timestamps_s']
    if times[0] < t[0]-1e-8 or times[-1] > source['trusted_until_s']+1e-8:
        raise ValueError('Counterfactual history is not covered by this original source')
    right = np.minimum(np.searchsorted(t, times), len(t)-1)
    left = np.maximum(right-1, 0)
    span = t[right]-t[left]
    weight = np.divide(times-t[left], span, out=np.zeros(len(times)), where=span > 0)
    first, second = source['global_rot_mats'][left], source['global_rot_mats'][right]
    delta = Rotation.from_matrix((first.swapaxes(-1, -2)@second).reshape(-1, 3, 3)).as_rotvec()
    delta = delta.reshape(len(times), 30, 3)*weight[:, None, None]
    return first@Rotation.from_rotvec(delta.reshape(-1, 3)).as_matrix().reshape(-1, 30, 3, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before = file_sha256(args.recording)
    rows = [json.loads(line) for line in args.recording.read_text().splitlines()]
    if not any(row['kind'] in ('explicit_stop', 'terminal_runtime_failure') for row in rows):
        raise ValueError('Only a completed or terminal recorded episode may be analyzed')
    receipts = {row['record_id']: row for row in rows if row['kind'] == 'numeric_artifact_written'}
    plans, hashes = {}, {}
    for key, receipt in receipts.items():
        if key.startswith('tick:'):
            continue
        actual = file_sha256(receipt['path'])
        if actual != receipt['sha256']:
            raise ValueError('Recorded plan artifact changed')
        record = load_record(receipt['path'])
        if 'conditioning_history' in record and record.get('plan_id') == key:
            plans[key] = record
            hashes[key] = actual
    rep = build_native_representation()
    parents = rep.skeleton.joint_parents.numpy()
    names = tuple(rep.skeleton.bone_order_names)
    legs = [names.index(side+bone) for side in ('Left', 'Right') for bone in ('Leg', 'Shin', 'Foot')]
    adoptees = {row['plan_id'] for row in rows if row['kind'] == 'reference_packet_adopted'}
    steps, histories, disagreements, counterfactuals = [], [], [], []
    for key, plan in plans.items():
        history = plan['conditioning_history']
        h = history['features']
        local = locals_from_globals(matrices(h, rep.indices['global_rot_data']), parents)
        provenance = history['provenance']
        prior_ids = [sample['prior_plan_id'] for sample in provenance['samples']]
        angular = Rotation.from_matrix((local[1:]@local[:-1].swapaxes(-1, -2)).reshape(-1, 3, 3)).magnitude().reshape(3, 30)
        for interval in range(3):
            steps.append(dict(plan_id=key, adopted=key in adoptees,
                mixed_prior=prior_ids[interval] != prior_ids[interval+1],
                leg_max_step_rad=float(angular[interval, legs].max()),
                nonroot_max_step_rad=float(angular[interval, 1:].max())))
        mixed = len(set(prior_ids)) > 1
        histories.append(dict(plan_id=key, time_s=plan['snapshot_time_s'], mixed_prior=mixed,
            adopted=key in adoptees, prior_ids=prior_ids,
            correction_segments=provenance['measured_correction_segments']))
        for index, prior in enumerate(prior_ids):
            if prior not in plans:
                if prior != 'explicit_time_zero_seed_pair':
                    raise ValueError(f'Missing actual conditioning prior: {prior}')
                continue
            source = plans[prior]['raw_source']
            time = history['timestamps_s'][index]
            predicted = np.array([np.interp(time, source['timestamps_s'], source['raw_features'][:, col])
                for col in rep.indices['foot_contacts']]) > .5
            observed = h[index, rep.indices['foot_contacts']] > .5
            disagreements.extend((predicted != observed).tolist())
        moving = len(provenance['measured_correction_segments']) == 2
        if mixed and moving and prior_ids[0] in plans:
            prior = plans[prior_ids[0]]['raw_source']
            times = history['timestamps_s']
            if times[0] >= prior['timestamps_s'][0]-1e-8 and times[-1] <= prior['trusted_until_s']+1e-8:
                consistent = locals_from_globals(sample_rotations(prior, times), parents)
                error = Rotation.from_matrix((local[:, legs]@consistent[:, legs].swapaxes(-1, -2)).reshape(-1, 3, 3)).magnitude()
                old_steps = Rotation.from_matrix((consistent[1:, legs]@consistent[:-1, legs].swapaxes(-1, -2)).reshape(-1, 3, 3)).magnitude()
                counterfactuals.append(dict(plan_id=key, adopted=key in adoptees, time_s=plan['snapshot_time_s'],
                    actual_history_leg_step_max_rad=float(angular[:, legs].max()),
                    same_prior_leg_step_max_rad=float(old_steps.max()),
                    history_leg_difference_max_rad=float(error.max()), common_prior=prior_ids[0]))
    inputs = [row['raw_intent'] for row in rows if row['kind'] == 'intent_received']
    deltas = [dict(velocity=float(np.linalg.norm(np.asarray(b['velocity_xy_mps'])-a['velocity_xy_mps'])),
                  yaw=abs(b['yaw_rate_radps']-a['yaw_rate_radps'])) for a, b in zip(inputs, inputs[1:])]
    mode_switches = sum(bool(np.any(a['velocity_xy_mps'])) != bool(np.any(b['velocity_xy_mps']))
        for a, b in zip(inputs, inputs[1:]))
    adopted = [row for row in rows if row['kind'] == 'reference_packet_adopted']
    adoption = [dict(dt=b['sim_time_s']-a['sim_time_s'],
        changed=b['generated_from_command_seq'] != a['generated_from_command_seq']) for a, b in zip(adopted, adopted[1:])]
    timing = {key: distribution([p['timing_ns'][key]/1e6 for p in plans.values()]) for key in
        ('mb_generation', 'retarget', 'provider_total')}
    report = dict(schema='mb_online_history_contract_review.v1', recording=str(args.recording.resolve()),
        recording_sha256=before, optimizer_updates=0, physics_steps=0, actor_calls=0, MB_calls=0,
        verified_plan_artifacts=len(plans), adopted_plans=len(adoptees),
        mixed_history_count=sum(h['mixed_prior'] for h in histories),
        adopted_mixed_history_count=sum(h['mixed_prior'] and h['adopted'] for h in histories),
        history_step_rad={label: distribution([s['leg_max_step_rad'] for s in steps if s['mixed_prior'] == mixed])
            for label, mixed in (('prior_boundary', True), ('same_prior', False))},
        moving_history_counterfactual=dict(count=len(counterfactuals),
            recorded_step=distribution([c['actual_history_leg_step_max_rad'] for c in counterfactuals]),
            same_prior_step=distribution([c['same_prior_leg_step_max_rad'] for c in counterfactuals]),
            difference=distribution([c['history_leg_difference_max_rad'] for c in counterfactuals])),
        predicted_vs_measured_contact_disagreement_fraction=float(np.mean(disagreements)),
        compared_contact_channels=len(disagreements), received_inputs=len(inputs),
        input_translation_delta_mps=distribution([x['velocity'] for x in deltas]),
        input_yaw_rate_delta_radps=distribution([x['yaw'] for x in deltas]),
        tiny_input_changes_under_001_each=sum(x['velocity'] < .01 and x['yaw'] < .01 for x in deltas),
        stop_move_observer_mode_changes=mode_switches,
        adoption_intervals_s={label: distribution([x['dt'] for x in adoption if x['changed'] == changed])
            for label, changed in (('new_command', True), ('same_command', False))},
        recorded_timing_ms=timing, histories=histories, counterfactuals=counterfactuals,
        artifact_sha256=hashes,
        scope='overlapping diagnostic windows; same-prior counterfactual changes only uncorrected local leg history; no physical causality claim')
    if file_sha256(args.recording) != before:
        raise ValueError('Recording changed while analyzed')
    save_json(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('histories','counterfactuals','artifact_sha256')}, indent=2))


if __name__ == '__main__':
    main()
