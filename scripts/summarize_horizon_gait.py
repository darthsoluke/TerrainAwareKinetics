"""Derive startup and planning-delay evidence from preserved protocol sequences."""
import argparse
import json
from pathlib import Path

import numpy as np


def reach(speed,threshold,hold):
    hits=np.convolve((speed>=threshold).astype(int),np.ones(hold,dtype=int),mode='valid')
    ids=np.flatnonzero(hits==hold)
    return None if not len(ids) else float((ids[0]+1)/30)


def main(a):
    rows=[]
    for folder in a.inputs:
        declaration=json.loads((folder/'complete.json').read_text())
        for item in declaration['arms']:
            tokens=item['tokens'];arm=json.loads((folder/f'arm-{tokens}.json').read_text())
            with np.load(folder/f'frames-{tokens}.npz') as data:roots=data['roots'].copy()
            cut=arm.get('target_start_frame',1)-1
            origin=roots[cut-1] if cut else np.asarray(declaration['initial_root'])
            root=roots[cut:];speed=np.asarray(arm['metrics']['root_planar_speed_mps'])
            steps=[step for packet in arm['packets'] for step in packet['steps'] if step['source_frame']>=cut]
            near=np.asarray([np.mean(step['root_planar_speed_mps'][:4]) for step in steps])
            far=np.asarray([np.mean(step['root_planar_speed_mps'][-4:]) for step in steps])
            rows.append(dict(source=str(folder),tokens=tokens,start_frame=cut+1,
                progress_1s_m=float(root[29,2]-origin[2]),progress_2s_m=float(root[59,2]-origin[2]),
                reach={str(v):dict(first_frame_seconds=reach(speed,v,1),first_sustained_half_second_start=reach(speed,v,15))
                       for v in (.5,1.,2.)},
                root_actual_speed_mean=float(speed.mean()),root_prediction_near4_mean=float(near.mean()),
                root_prediction_last4_mean=float(far.mean()),
                near4_per_replan=near.tolist(),last4_per_replan=far.tolist(),
                root_slow_episodes=arm['metrics']['root_slow_episodes'],leg_still_episodes=arm['metrics']['leg_still_episodes']))
    result=dict(completed=True,actual_optimizer_updates=0,new_model_generation_calls=0,
        metric='Progress is signed worldZ from the declared origin. Sustained reach requires15consecutive raw30Hzspeed samples; no motion is filtered.',rows=rows)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in row.items() if not k.endswith('_per_replan')} for row in rows]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
