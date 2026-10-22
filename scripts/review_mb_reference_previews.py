"""Measure changes to G1 future targets that the tracker has already observed."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.reference import file_sha256
from scripts.audit_mb_tracker_binary import read_index
from scripts.isolate_mb_continuous_reference import load_record, save_json


def review(path):
    _,_,artifacts,_,_ = read_index(path)
    previous,last_window = None,None
    changes=[]
    before=file_sha256(path)
    with path.open() as stream:
        for line in stream:
            event=json.loads(line)
            if event['kind']=='reference_packet_used':
                last_window=np.asarray(event['used_time_window'])
            if event['kind']!='reference_packet_adopted':
                continue
            item=artifacts[event['plan_id']]
            if file_sha256(item['path'])!=item['sha256']:
                raise ValueError('Recorded reference changed')
            record=load_record(item['path'])
            current=record['adapted_reference']
            if previous is not None and last_window is not None:
                times=last_window[last_window>=event['sim_time_s']-1e-8]
                pairs=[]
                for reference in (previous,current):
                    indices=np.searchsorted(reference['timestamps_s'],times-1e-8)
                    if (np.any(indices>=len(reference['timestamps_s'])) or
                        not np.allclose(reference['timestamps_s'][indices],times,atol=1e-8,rtol=0)):
                        raise ValueError('Previously observed future is not covered')
                    pairs.append((reference['qpos'][indices],reference['body_quat_w'][indices,7]))
                (old,old_torso),(new,new_torso)=pairs
                torso=(Rotation.from_quat(old_torso[:,[1,2,3,0]]).inv()*
                       Rotation.from_quat(new_torso[:,[1,2,3,0]])).magnitude()
                changes.append(dict(time_s=event['sim_time_s'],command_id=record['command']['command_id'],
                    previously_observed_future_times_s=times.tolist(),
                    append=current['adaptation']['stream_append'],
                    joint_max_rad=float(np.abs(new[:,7:]-old[:,7:]).max()),
                    root_max_m=float(np.linalg.norm(new[:,:3]-old[:,:3],axis=1).max()),
                    torso_max_rad=float(torso.max())))
            previous=current
    if file_sha256(path)!=before:
        raise ValueError('The recording is still changing')
    result=dict(schema='motionbricks.observed_future_reference_review.v1',
        recording_sha256=before,new_physics_steps=0,optimizer_updates=0,
        scope='same absolute G1 target timestamps seen in the immediately preceding controller window',
        comparisons=changes)
    for name,rows in [('all',changes),('new_intent',[r for r in changes if not r['append']]),
                      ('append',[r for r in changes if r['append']])]:
        result[name]=dict(count=len(rows),**{key:dict(zip(('median','p95','max'),
            np.percentile([row[key] for row in rows],[50,95,100]).tolist()))
            for key in ('joint_max_rad','root_max_m','torso_max_rad')} ) if rows else dict(count=0)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=review(args.recording)
    save_json(args.output,report)
    print(json.dumps({key:report[key] for key in ('all','new_intent','append')},indent=2))


if __name__=='__main__':
    main()
