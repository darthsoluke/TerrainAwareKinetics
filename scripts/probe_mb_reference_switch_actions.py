"""Hold saved sensors/history fixed and swap only old/new future references."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.reference import file_sha256
from scripts.analyze_mb_reference_jitter import distribution
from scripts.analyze_mb_reference_support import decode, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    log_hash = file_sha256(args.recording)
    lines = [json.loads(line) for line in args.recording.read_text(encoding='utf-8').splitlines()]
    if not any(row['kind'] == 'explicit_stop' for row in lines):
        raise ValueError('The recorded episode must be closed')
    artifacts = {row['record_id']: row for row in lines if row['kind'] == 'numeric_artifact_written'}
    uses = [row for row in lines if row['kind'] == 'reference_packet_used']
    ticks = [row for row in lines if row['kind'] == 'policy_tick_binary']
    settings = json.loads(args.config.read_text())['protomotions_policy']
    if file_sha256(settings['onnx_path']) != settings['onnx_sha256']:
        raise ValueError('Published ONNX changed')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(settings['onnx_path'], sess_options=options, providers=['CPUExecutionProvider'])
    session.disable_fallback()
    candidates = [i for i in range(1, len(uses)) if uses[i]['plan_id'] != uses[i-1]['plan_id']
                  and uses[i]['generated_from_command_seq'] == uses[i-1]['generated_from_command_seq']]
    selected = np.unique(np.linspace(0, len(candidates)-1, 12, dtype=int))
    cases, verified = [], set()

    def load(record_id):
        receipt = artifacts[record_id]
        if file_sha256(receipt['path']) != receipt['sha256']:
            raise ValueError(f'Changed numeric artifact: {record_id}')
        verified.add(record_id)
        with np.load(receipt['path'], allow_pickle=False) as archive:
            return decode(json.loads(str(archive['metadata_json']))['record'], archive)

    for selection in selected:
        index = candidates[selection]
        current = load(ticks[index]['numeric_record_id'])['tick']['policy_command']
        old = load(uses[index-1]['plan_id'])['adapted_reference']
        new = load(uses[index]['plan_id'])['adapted_reference']
        time = float(ticks[index]['reference_time_s'])
        offsets = np.array([.02, .04, .08, .16])
        a, b = sample(old, time+offsets), sample(new, time+offsets)
        anchor = new['body_names'].index('torso_link')
        inputs = current['inputs']
        new_quat = b['body_quat_w'][:, anchor][:, [1, 2, 3, 0]]
        alignment = Rotation.from_quat(inputs['mimic_future_anchor_rot'][0, 0]) * Rotation.from_quat(new_quat[0]).inv()
        counterfactual = dict(inputs)
        counterfactual.update(
            mimic_future_anchor_rot=(alignment * Rotation.from_quat(a['body_quat_w'][:, anchor][:, [1, 2, 3, 0]])).as_quat()[None].astype(np.float32),
            mimic_future_dof_pos=a['qpos'][None, :, 7:].astype(np.float32),
            mimic_future_dof_vel=a['qvel'][None, :, 6:].astype(np.float32))
        repeated = session.run(None, inputs)[1][0]
        alternate = session.run(None, counterfactual)[1][0]
        recorded_native = (current['execution_task']['native_joint_target']
            if current.get('execution_task') is not None else current['joint_target'])
        error = float(np.max(np.abs(repeated-recorded_native)))
        if error > 1e-6:
            raise ValueError(f'Original actor invocation does not reproduce its recorded target: {error}')
        delta = np.abs(repeated-alternate)
        joint = int(delta.argmax())
        cases.append(dict(time_s=time, command_seq=uses[index]['generated_from_command_seq'],
            original_target_replay_error_rad=error, max_target_change_rad=float(delta[joint]),
            target_change_rms_rad=float(np.sqrt(np.mean(delta**2))), largest_joint=new['joint_names'][joint],
            old_target_rad=float(alternate[joint]), new_target_rad=float(repeated[joint])))
    if file_sha256(args.recording) != log_hash:
        raise ValueError('Recording changed during analysis')
    report = dict(schema='motionbricks.mb_sonic.reference_switch_counterfactual.v1',
        recording=str(args.recording.resolve()), recording_sha256=log_hash,
        onnx_sha256=settings['onnx_sha256'], optimizer_updates=0, physics_resimulations=0,
        selected_cases=len(cases), actor_evaluations=2*len(cases), verified_numeric_artifacts=len(verified),
        held_fixed=['current sensors', 'previous actions', 'absolute time', 'original raw command'],
        changed_only=['future joint positions', 'future joint velocities', 'future torso orientations'],
        maximum_target_change_rad=distribution([case['max_target_change_rad'] for case in cases]), cases=cases)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
