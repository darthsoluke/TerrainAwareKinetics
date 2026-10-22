"""Measure direction, stopping and joint oscillation from a closed gamepad trial."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.reference import file_sha256
from scripts.audit_mb_tracker_binary import read_index
from scripts.isolate_mb_continuous_reference import load_record,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    path=next((args.trial/'recording').glob('episode_0001_*.jsonl'))
    digest=file_sha256(path)
    receipt=json.loads((args.trial/'browser_receipt.json').read_text())
    events=[json.loads(line) for line in path.read_text().splitlines()]
    intents=[e['raw_intent'] for e in events if e['kind']=='intent_received']
    received=np.array([i['server_received_ns'] for i in intents])
    simulation=receipt['input_simulation']
    phases=[]
    for sample in simulation['raw_samples']:
        if not phases or sample['phase']!=phases[-1][1]:
            phases.append((sample['time_s'],sample['phase']))
    phases.append((simulation['trace_duration_s'],'Post-trace neutral'))
    _,ticks,artifacts,_,_=read_index(path)
    rows=[]
    for tick in ticks.values():
        item=artifacts[tick['numeric_record_id']]
        if file_sha256(item['path'])!=item['sha256']:
            raise ValueError('Physical tick changed')
        r=load_record(item['path']); p=r['physics']; ref=r['tick']['adapted_reference']
        now=int(p['captured_at_monotonic_ns'][-1])
        command=intents[max(0,int(np.searchsorted(received,now,side='right')-1))]
        elapsed=(now-received[0])/1e9+(intents[0]['client_input_ms']-simulation['started_client_ms'])/1000
        rows.append(dict(time=float(p['time_s'][-1]),elapsed=elapsed,qpos=p['qpos'][-1],
            qvel=p['qvel'][-1],target=r['tick']['policy_command']['joint_target'],
            reference=ref['qpos'][0],reference_velocity=ref['qvel'][0],
            command=command['velocity_xy_mps'],yaw=command['yaw_rate_radps']))
    arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    reports=[]
    for i,(start,name) in enumerate(phases):
        end=phases[i+1][0] if i+1<len(phases) else arrays['elapsed'][-1]+.02
        mask=(arrays['elapsed']>=start+.4)&(arrays['elapsed']<end)
        if mask.sum()<10:
            continue
        q=arrays['qpos'][mask]; v=arrays['qvel'][mask]; u=arrays['command'][mask]
        speed=np.linalg.norm(u,axis=-1)
        direction=np.divide(u,speed[:,None],out=np.zeros_like(u),where=speed[:,None]>.01)
        progress=np.sum(v[:,:2]*direction,axis=-1)
        elapsed=arrays['time'][mask][-1]-arrays['time'][mask][0]
        displacement=q[-1,:2]-q[0,:2]
        reference_velocity=arrays['reference_velocity'][mask,:2]
        frequency,power=welch(q[:,7:],fs=50,nperseg=min(len(q),200),axis=0,detrend='linear')
        tilt=np.degrees(np.arccos(np.clip(Rotation.from_quat(q[:,[4,5,6,3]]).as_matrix()[:,2,2],-1,1)))
        report=dict(phase=name,samples=int(mask.sum()),time_range_s=arrays['time'][mask][[0,-1]].tolist(),
            intended_speed_mean_mps=float(speed.mean()),actual_speed_mean_mps=float(np.linalg.norm(v[:,:2],axis=-1).mean()),
            progress_along_current_intent_mean_mps=float(progress.mean()),
            reference_progress_along_current_intent_mean_mps=float(np.sum(reference_velocity*direction,axis=-1).mean()),
            reference_speed_mean_mps=float(np.linalg.norm(reference_velocity,axis=-1).mean()),
            net_velocity_xy_mps=(displacement/elapsed).tolist(),
            net_drift_speed_mps=float(np.linalg.norm(displacement)/elapsed),
            reverse_fraction=float(np.mean(progress[speed>.05]<-.05)) if np.any(speed>.05) else None,
            displacement_xy_m=displacement.tolist(),root_tilt_max_deg=float(tilt.max()),
            joint_energy_8_14_rad2=float(power[(frequency>=8)&(frequency<=14)].sum()*(frequency[1]-frequency[0])),
            reference_root_error_mean_m=float(np.linalg.norm(q[:,:3]-arrays['reference'][mask,:3],axis=-1).mean()))
        reports.append(report)
    if file_sha256(path)!=digest:
        raise ValueError('Recording must be closed')
    report=dict(schema='motionbricks.gamepad_physical_review.v2',recording_sha256=digest,
        phase_alignment='first_input_client_server_offset; excludes_first400ms_of_each_phase',
        tick_samples=len(rows),physical_end_s=float(arrays['time'][-1]),new_physics_steps=0,
        optimizer_updates=0,display=receipt['metrics'],phases=reports)
    save_json(args.output if args.output is not None else args.trial/'gamepad_review.json',report)
    print(json.dumps(dict(tick_samples=len(rows),phases=[{k:r[k] for k in
        ('phase','intended_speed_mean_mps','progress_along_current_intent_mean_mps','actual_speed_mean_mps','reverse_fraction')}
        for r in reports]),indent=2))


if __name__=='__main__':
    main()
