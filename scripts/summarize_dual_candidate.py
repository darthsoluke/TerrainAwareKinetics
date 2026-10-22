"""Offline startup and spatial-hypothesis measurements from saved candidate runs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.summarize_horizon_gait import reach


def main(a):
    rows=[]
    for mode in ('cold','warm'):
        for tokens in (12,16):
            run=json.loads((a.input/f'{mode}-{tokens}.json').read_text())
            with np.load(a.input/f'{mode}-{tokens}.npz') as data:
                roots,origin=data['roots'],data['initial_root']
            speed=np.asarray(run['metrics']['root_planar_speed_mps'])
            steps=[step for packet in run['packets'] for step in packet['observations']]
            far=np.asarray([step['far_from_short_endpoint_forward_m'] for step in steps])
            rows.append(dict(mode=mode,lookahead_tokens=tokens,execute_tokens=8,
                progress_1s_m=float(roots[29,2]-origin[2]),progress_2s_m=float(roots[59,2]-origin[2]),
                reach={str(v):dict(first_frame_seconds=reach(speed,v,1),first_sustained_half_second_start=reach(speed,v,15)) for v in (.5,1.,2.)},
                mean_speed=run['metrics']['root_speed_mean'],progress_m=run['metrics']['longitudinal_progress_m'],
                root_slow_episodes=run['metrics']['root_slow_episodes'],leg_still_episodes=run['metrics']['leg_still_episodes'],
                far_station_count=int(far.size),far_behind_short_endpoint_count=int((far<0).sum()),
                far_from_short_endpoint_min_m=float(far.min()),far_from_short_endpoint_max_m=float(far.max()),
                near_root_disagreement_max_m=max(step['near_root_disagreement_max_m'] for step in steps),
                first_stations={k:steps[0][k] for k in ('station_source_frames','station_world_xz_m','station_forward_m','far_from_short_endpoint_forward_m')}))
    result=dict(completed=True,new_model_generation_calls=0,actual_optimizer_updates=0,rows=rows)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
