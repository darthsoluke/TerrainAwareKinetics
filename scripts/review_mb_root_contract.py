"""Frozen-sensor reference-frame counterfactuals; no simulation or training."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.protomotions_policy import FrozenProtoMotionsPolicy
from motionbricks.mb_sonic.reference import G1Reference,file_sha256
from scripts.audit_mb_tracker_binary import read_index
from scripts.isolate_mb_continuous_reference import load_record,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads((args.case/'config.json').read_text())
    path=next((args.case/'recording').glob('*.jsonl'))
    _,ticks,artifacts,_,_=read_index(path)
    record_id=next(iter(ticks)); item=artifacts[record_id]
    if file_sha256(item['path'])!=item['sha256']: raise ValueError('Physics record changed')
    record=load_record(item['path'])['tick']
    ref_values=record['adapted_reference'].copy()
    for name in ('body_names','joint_names'): ref_values[name]=tuple(ref_values[name])
    reference=G1Reference(**ref_values)
    sensors=record['policy_command']['inputs']
    policy=FrozenProtoMotionsPolicy(**config['protomotions_policy'],threads=1)
    torso=reference.body_names.index('torso_link')
    policy.heading_alignment=(Rotation.from_quat(sensors['mimic_future_anchor_rot'][0,0])*
        Rotation.from_quat(reference.body_quat_w[1,torso][[1,2,3,0]]).inv())
    current=SimpleNamespace(time_s=reference.timestamps_s[0],
        joint_positions=sensors['current_dof_pos'][0],joint_velocities=sensors['current_dof_vel'][0],
        base_angular_velocity=sensors['current_root_local_ang_vel'][0])
    history=SimpleNamespace(entries=[current])
    qpos=reference.qpos.copy(); qpos[:,:3]+=np.array([100.,-30.,2.])
    qvel=reference.qvel.copy();qvel[:,:3]+=np.array([50.,20.,3.])
    shifted=replace(reference,qpos=qpos,qvel=qvel,
        body_pos_w=reference.body_pos_w+np.array([100.,-30.,2.]),
        body_lin_vel_w=reference.body_lin_vel_w+np.array([50.,20.,3.]))
    outputs=[]
    for ref in (reference,shifted):
        policy.previous_target=sensors['historical_processed_actions'][0,0].copy()
        outputs.append(policy.infer(ref,history,
            anchor_quat_wxyz=sensors['current_anchor_rot'][0][[3,0,1,2]]))
    checks={key:bool(np.array_equal(outputs[0].inputs[key],outputs[1].inputs[key])) for key in outputs[0].inputs}
    report=dict(schema='motionbricks.frozen_tracker_root_observability.v1',
        source_record_id=record_id,source_sha256=item['sha256'],
        controlled_change=dict(reference_translation_m=[100.,-30.,2.],reference_velocity_delta_mps=[50.,20.,3.]),
        actual_sensors_unchanged=True,network_inputs_bitwise_equal=checks,
        network_action_bitwise_equal=bool(np.array_equal(outputs[0].joint_target,outputs[1].joint_target)),
        network_action_max_abs_rad=float(np.max(np.abs(outputs[0].joint_target-outputs[1].joint_target))),
        ONNX_sha256=config['protomotions_policy']['onnx_sha256'],
        CPU_actor_calls=2,MB_calls=0,new_physics_steps=0,optimizer_updates=0,
        conclusion='This pinned tracker is directly insensitive to reference XYZ and linear velocity; torso orientation and joints remain consumed')
    save_json(args.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__': main()
