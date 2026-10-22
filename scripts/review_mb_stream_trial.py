"""Review the recorded physical gait and input response of a stream trial."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from motionbricks.mb_sonic.reference import file_sha256, OFFICIAL_BODY_NAMES
from scripts.audit_mb_tracker_binary import read_index
from scripts.isolate_mb_continuous_reference import load_record, save_json
from scripts.review_mb_continuous_reference import spectrum


def read_trial(directory):
    config = json.loads((directory/'config.json').read_text())
    path = next((directory/'recording').glob('episode_0001_*.jsonl'))
    before = file_sha256(path)
    meta,ticks,artifacts,_,_ = read_index(path)
    model = mujoco.MjModel.from_xml_path(config['robot_model'])
    body_ids = [model.body(name).id for name in OFFICIAL_BODY_NAMES]
    values = {name:[] for name in ('time','qpos','qvel','body','force','target','control_time','control_joints')}
    for tick in ticks.values():
        item = artifacts[tick['numeric_record_id']]
        if file_sha256(item['path'])!=item['sha256']:
            raise ValueError('Changed physical sample')
        record = load_record(item['path'])
        physical = record['physics']
        values['time'].append(physical['time_s'])
        values['qpos'].append(physical['qpos'])
        values['qvel'].append(physical['qvel'])
        values['body'].append(physical['body_positions_world'][:,body_ids])
        values['force'].append(physical['actuator_forces'])
        values['target'].append(record['tick']['policy_command']['joint_target'])
        values['control_time'].append(physical['time_s'][-1])
        values['control_joints'].append(physical['joint_positions'][-1])
    for key in values:
        values[key] = np.asarray(values[key]) if key in ('target','control_time','control_joints') else np.concatenate(values[key])
    if file_sha256(path)!=before:
        raise ValueError('Physical log is still changing')
    return config,values,before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    config,data,digest = read_trial(args.trial)
    result = json.loads((args.trial/'result.json').read_text())
    commands = result['sent']
    phases=[]
    for index,item in enumerate(commands):
        begin=item['sim_time_s']+.25
        end=commands[index+1]['sim_time_s'] if index+1<len(commands) else data['time'][-1]
        mask=(data['time']>=begin)&(data['time']<end)
        if mask.sum()<100:
            continue
        root=data['qpos'][mask,:3]
        relative=data['body'][mask]-root[:,None]
        velocity=np.asarray(item['command']['velocity_xy_mps'])
        direction=velocity/np.linalg.norm(velocity) if np.linalg.norm(velocity)>0 else np.zeros(2)
        observed=(root[-1,:2]-root[0,:2])/(end-begin)
        phases.append(dict(command_id=item['command']['command_id'],begin_s=begin,end_s=end,
            requested_velocity_xy_mps=velocity.tolist(),requested_yaw_rate=item['command']['yaw_rate_radps'],
            actual_velocity_xy_mps=observed.tolist(),projected_velocity_mps=float(observed@direction),
            planar_displacement_m=(root[-1,:2]-root[0,:2]).tolist(),
            body_relative_spectrum=spectrum(relative,.001),joint_spectrum=spectrum(data['qpos'][mask,7:],.001)))
    tail=data['time']>=data['time'][-1]-3.
    stop_velocity=np.linalg.norm(data['qvel'][tail,:2],axis=1)
    report=dict(schema='mb_stream_gait_review.v1',recording_sha256=digest,status=result['status'],
        new_physics_steps=0,optimizer_updates=0,phases=phases,
        final3s_speed_mean_mps=float(stop_velocity.mean()),
        final3s_speed_p95_mps=float(np.percentile(stop_velocity,95)),
        actual_joint_spectrum=spectrum(data['qpos'][:,7:],.001),
        relative_body_spectrum=spectrum(data['body']-data['qpos'][:,None,:3],.001),
        actor_target_spectrum=spectrum(data['target'],.02),
        raw_recorded_states_used=True,pose_filter=False,
        hardware_profile=result['stable_gpu_profile'],scope='per-command physical samples; source and command histories preserved')
    save_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
