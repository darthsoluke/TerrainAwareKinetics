"""CPU policy checks and geometry-only declaration before model comparisons."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from motionbricks.data.anticipatory_speed import DEFAULTS, GovernorState, TerrainSpeedMap


def main(output,catalog,review_folder=None):
    output.mkdir(parents=True,exist_ok=False)
    roots=np.tile([0.,1.,0.,1.,0.],(4,1))
    scene=dict(heightfield_y_m=np.zeros((801,801),np.float32),min_point_world_xyz_m=[-20.,0.,-20.],dx_m=.05)
    flat=TerrainSpeedMap(scene)
    assert DEFAULTS['enabled'] is False
    config=dict(DEFAULTS,enabled=True)
    _,state,info=flat.govern(roots,[0.,3.],GovernorState(),config,4)
    assert info['target_speed_mps']==3.
    _,stopped,_=flat.govern(roots,[0.,0.],state,config,4)
    _,restarted,again=flat.govern(roots,[0.,3.],stopped,config,4)
    assert again['target_speed_mps']==3. and restarted.frame==12
    checks=['flat_start','flat_restart','frame_count']
    for height in (.18,-.18):
        data=np.zeros((801,801),np.float32)
        data[410:]=height
        field=TerrainSpeedMap(dict(scene,heightfield_y_m=data))
        original=GovernorState()
        a,advanced,review=field.govern(roots,[0.,3.],original,config,4)
        assert advanced.frame==4 and original.frame==0
        assert review['target_speed_mps']<2. and review['risk_distance_m']>.2
        assert review['target_speed_mps']==review['cap_speed_mps']
        assert review['scan_extent_m']>=4. and review['sample_count']<=3855
        b,other,other_review=field.govern(roots,[0.,3.],original,config,4)
        assert a==b and advanced==other
        checks.append('up_start' if height>0 else 'down_start')
    moving=roots.copy();moving[:,2]=np.arange(-3,1)*.8/30
    _,recover,info=flat.govern(moving,[0.,3.],GovernorState(target_speed=.8,risk_limited=True),config,4)
    assert abs(recover.target_speed-(.8+4/30))<1e-10
    checks.append('bounded_recovery')
    _,_,far=flat.govern(roots,[0.,12.],GovernorState(),config,4)
    assert far['scan_limited'] and far['target_speed_mps']<12. and far['mode']=='horizon_limited'
    checks.append('bounded_visibility')
    edge=roots.copy();edge[:,2]=19.7
    _,_,boundary=flat.govern(edge,[0.,3.],GovernorState(),config,4)
    assert boundary['mode']=='boundary' and boundary['target_speed_mps']==0.
    checks.append('unknown_boundary')
    path=GovernorState(path=((0.,0.),(0.,1.),(.4,2.)),path_frame=0,frame=4)
    assert flat._previous_curve(path,np.zeros(2),np.array([0.,1.]),np.linspace(0,3,257)) is not None
    assert flat._previous_curve(path,np.zeros(2),np.array([0.,-1.]),np.linspace(0,3,257)) is None
    checks.append('turn_rejects_old_path')
    passed_end=GovernorState(path=((0.,0.),(0.,1.)),path_frame=0,frame=1)
    assert flat._previous_curve(passed_end,np.array([0.,1.2]),np.array([0.,1.]),np.linspace(0,3,257)) is None
    checks.append('passed_endpoint_rejected')
    times=[]
    for index in range(105):
        started=time.perf_counter();flat.govern(moving,[0.,3.],path,config,4)
        if index>=5:times.append((time.perf_counter()-started)*1000)
    manifest=json.loads(catalog.read_text())
    entry=next(item for item in manifest['entries'] if item['id']=='007')
    with np.load(catalog.parent/entry['artifact']/'scene.npz',allow_pickle=False) as stored:
        current=TerrainSpeedMap({key:stored[key].copy() for key in ('heightfield_y_m','min_point_world_xyz_m','dx_m')})
        seed=stored['initial_history'][0,:,:5].copy()
    start=seed[-1,[0,2]]
    cases=[]
    for name,direction in [('forward',[0.,1.]),('diagonal',[-2**-.5,2**-.5])]:
        distances=np.linspace(0.,10.,201)
        points=start[None]+distances[:,None]*np.asarray(direction)
        heights,known=current.heights(points)
        # Declare from static geometry, before any generated comparison output.
        transition=np.flatnonzero(known & (np.abs(heights-heights[0])>=.08))
        first=float(distances[transition[0]]) if len(transition) else None
        target=max(3.,first+1.) if first is not None else 3.
        target=min(target,9.)
        cases.append(dict(name=name,direction=direction,target_progress_m=target,
            first_8cm_height_change_m=first,contains_transition=first is not None and target>first,
            requested_speed_mps=3.,max_frames=240,origin_xz_m=start.tolist(),
            profile_distances_m=distances[::5].tolist(),profile_heights_m=heights[::5].tolist(),
            known=known[::5].tolist()))
    guard_review=None
    if review_folder is not None:
        import torch
        old=json.loads((review_folder/'complete.json').read_text())
        limiting=[];other=0
        for row in old['rows']:
            if not row['settings']['enabled']:continue
            saved=torch.load(review_folder/f'{row["case"]}-{row["arm"]}.pt',map_location='cpu',weights_only=False)
            direction=np.asarray(saved['case']['direction'])
            for packet in row['packets']:
                for item in packet['governor']:
                    if item['route']!='session_predicted_path':other+=1;continue
                    position=saved['world'][0,3+item['source_frame'],[0,2]].numpy()
                    point=np.asarray(item['risk_point_world_xyz_m'])[[0,2]]
                    ahead=float((point-position)@direction)
                    assert ahead>0
                    limiting.append(dict(case=row['case'],arm=row['arm'],frame=item['source_frame'],
                        position=position.tolist(),risk_point=point.tolist(),risk_ahead_m=ahead))
        guard_review=dict(non_path_limited_steps=other,path_limited_steps=limiting,
            conclusion='The single path-limited v2 sample lies ahead of intent; a reversed terminal tail would have nonpositive projection. All other limits came from unaffected requested/actual routes.',model_generation_calls=0)
    (output/'complete.json').write_text(json.dumps(dict(passed=True,checks=checks,deployment_defaults=dict(DEFAULTS),
        path_guard_review=guard_review,
        cpu_overhead_ms=dict(n=100,p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95))),
        cases=cases,settings=dict(num_tokens=8,inference_steps=4,refresh_frames=4),
        fixed_before_model_comparison=True,actual_optimizer_updates=0),indent=2))
    print(json.dumps(dict(passed=True,checks=len(checks),cases=[{k:v for k,v in row.items() if not k.startswith('profile') and k!='known'} for row in cases])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--review',type=Path)
    args=p.parse_args();main(args.output,args.catalog,args.review)
