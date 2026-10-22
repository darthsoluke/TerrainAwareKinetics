"""Audit saved native-block joins and tracker-visible reference dynamics on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.reference import file_sha256
from scripts.audit_mb_tracker_binary import read_index
from scripts.isolate_mb_continuous_reference import load_record,save_json


def peaks(times,positions,rotations):
    dt=np.diff(times)
    linear=np.linalg.norm(np.diff(positions,axis=0),axis=-1)/dt
    relative=rotations[1:]@rotations[:-1].swapaxes(-1,-2)
    angular=Rotation.from_matrix(relative).magnitude()/dt
    i=int(np.argmax(linear)); j=int(np.argmax(angular))
    return dict(speed_max_mps=float(linear[i]),speed_time_s=times[[i,i+1]].tolist(),
        speed_step_m=float(np.linalg.norm(positions[i+1]-positions[i])),
        speed_step_dt_s=float(dt[i]),angular_max_radps=float(angular[j]),
        angular_time_s=times[[j,j+1]].tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    before=file_sha256(args.recording)
    _,_,artifacts,_,_=read_index(args.recording)
    with args.recording.open() as stream: events=[json.loads(line) for line in stream]
    records=[]; previous=None
    for event in events:
        if event['kind']!='reference_packet_adopted': continue
        item=artifacts[event['plan_id']]
        if file_sha256(item['path'])!=item['sha256']: raise ValueError('Recorded plan changed')
        record=load_record(item['path'])
        source=record['raw_source']; g1=record['adapted_reference']; history=record['conditioning_history']
        row=dict(time_s=event['sim_time_s'],snapshot_time_s=record['packet']['issued_from_sim_time'],
            command=record['command'],source_record_id=event['plan_id'],
            history_times_s=history['timestamps_s'].tolist(),
            history=peaks(history['timestamps_s'],history['features'][:,:3],
                Rotation.from_euler('y',np.arctan2(history['features'][:,4],history['features'][:,3])).as_matrix()),
            source=peaks(source['timestamps_s'],source['root_positions_m'],source['global_rot_mats'][:,0]),
            g1=peaks(g1['timestamps_s'],g1['qpos'][:,:3],
                Rotation.from_quat(g1['qpos'][:,[4,5,6,3]]).as_matrix()),
            append=g1['adaptation']['stream_append'],
            raw_block_internal=[],registration_yaw_rad=history['provenance'].get('yaw_registration_rad'))
        for block in record['raw_blocks']:
            world=block['raw_features']; time=block['timestamps_s']
            orientation=Rotation.from_euler('y',np.arctan2(world[:,4],world[:,3])).as_matrix()
            row['raw_block_internal'].append(peaks(time[4:28],world[4:28,:3],orientation[4:28]))
        if record['raw_blocks']:
            block=record['raw_blocks'][0]['raw_features']
            row['decoded_context_error_m']=(block[3,:3]-history['features'][3,:3]).tolist()
            row['decoded_first_future_speed_mps']=float(np.linalg.norm(block[4,:3]-block[3,:3])*30)
        if previous is not None and record['raw_blocks']:
            cut=source['provenance']['retained_prefix_until_s']
            a=previous['raw_source']
            left=np.flatnonzero(a['timestamps_s']<=cut+1e-8)[-1]
            right=np.flatnonzero(source['timestamps_s']>cut+1e-8)[0]
            gap=source['root_positions_m'][right]-a['root_positions_m'][left]
            dt=source['timestamps_s'][right]-a['timestamps_s'][left]
            row['join']=dict(cut_time_s=float(cut),left_time_s=float(a['timestamps_s'][left]),
                right_time_s=float(source['timestamps_s'][right]),dt_s=float(dt),gap_m=float(np.linalg.norm(gap)),
                apparent_speed_mps=float(np.linalg.norm(gap)/dt))
            h0,h1=history['timestamps_s'][[0,-1]]
            keep=(a['timestamps_s']>=h0-1e-8)&(a['timestamps_s']<=h1+1e-8)
            if keep.sum()>1:
                row['prior_context']=peaks(a['timestamps_s'][keep],a['root_positions_m'][keep],a['global_rot_mats'][keep,0])
        records.append(row); previous=record
    if before!=file_sha256(args.recording): raise ValueError('Recording is changing')
    summary={key:next((row for row in records if row[key]['speed_max_mps']>10),None)
             for key in ('history','source','g1')}
    report=dict(schema='motionbricks.stream_frame_review.v1',recording_sha256=before,
        optimizer_updates=0,new_physics_steps=0,model_calls=0,plans=len(records),
        first_over10mps=summary,records=records)
    save_json(args.output,report)
    print(json.dumps(dict(plans=len(records),first_over10mps={key:None if row is None else
        dict(time_s=row['time_s'],snapshot_time_s=row['snapshot_time_s'],measurement=row[key],
             join=row.get('join'),raw=row['raw_block_internal'],history=row['history'])
        for key,row in summary.items()}),indent=2))


if __name__=='__main__': main()
