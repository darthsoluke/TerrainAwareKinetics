"""Compare preserved pre-generation direction with outcome-derived motion labels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.lafan_soma_retarget import Y_TO_Z
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_control import demonstration_controls
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.train_native_mb import sha256


def summarize(rows):
    if not rows:return {'windows':0}
    a=np.asarray([r['direction_difference_deg'] for r in rows])
    return {'windows':len(rows),'angle_median_deg':float(np.median(a)),
        'angle_p95_deg':float(np.quantile(a,.95)),'over45_fraction':float((a>45).mean()),
        'over90_fraction':float((a>90).mean()),
        'measured_speed_median_mps':float(np.median([r['outcome_speed_mps'] for r in rows]))}


def main(args):
    torch.set_num_threads(4)
    data=TerrainLearningDataset(args.dataset,build_native_representation(args.dataset/'stats'),'train')
    manifest=json.loads((args.dataset/'native_manifest.json').read_text());artifacts=manifest['artifact_sha256']
    rows=[];records=[]
    for index,row in enumerate(data.records):
        if row.get('source_intent',{}).get('kind')!='recorded_intent_sidecar':continue
        relative=row['source_intent_sidecar'];path=(args.dataset/relative).resolve()
        if args.dataset.resolve() not in path.parents:raise ValueError('Intent path leaves dataset')
        expected=row['source_intent_sidecar_sha256']
        if artifacts.get(relative)!=expected or sha256(path)!=expected:raise ValueError('Intent identity mismatch')
        with np.load(path,allow_pickle=False) as values:
            command=values['command_world_xy'].copy();frames=values['source_frame_indices'].copy();fps=float(values['fps'])
        item=data.full_reference_item(index);world=item['world']
        if command.shape!=(len(world),2) or frames.shape!=(len(world),) or fps!=30:raise ValueError('Intent time axis mismatch')
        if not np.isfinite(command).all() or (np.diff(frames)<=0).any():raise ValueError('Invalid intent values/time index')
        frame_hash=hashlib.sha256(np.ascontiguousarray(frames,dtype=np.int64).tobytes()).hexdigest()
        if row.get('source_frame_indices_sha256') and row['source_frame_indices_sha256']!=frame_hash:
            raise ValueError('Intent/source frame-index hash differs')
        native=np.column_stack((command,np.zeros(len(command))))@Y_TO_Z
        native=torch.from_numpy(native[:,[0,2]].astype(np.float32))
        norms=native.norm(dim=-1)
        if not torch.allclose(norms,torch.ones_like(norms),atol=2e-5,rtol=0):raise ValueError('Recorded direction is not unit length')
        gap,known=heightfield_gap(world[:,:3],item['scene']);ground=world[:,1]-gap
        selected=[]
        for start in range(0,len(world)-32,8):
            inferred=demonstration_controls(world[None,start:start+33],32)[0,:2]
            speed=float(inferred.norm())
            if speed<.2:continue
            current=native[start+3]
            direction_error=float(torch.rad2deg(torch.acos((current.dot(inferred)/speed).clamp(-1,1))))
            change=torch.rad2deg(torch.acos((native[start+3:start+32]*current).sum(-1).clamp(-1,1)))
            stable=bool(change.max()<=5.)
            measured_ground=ground[start:start+33];valid=known[start:start+33]
            relief=float(measured_ground[valid].max()-measured_ground[valid].min()) if valid.any() else None
            result={'source_id':row['source_id'],'window_start':start,'current_source_frame':int(frames[start+3]),
                'direction_difference_deg':direction_error,'recorded_command_change_max_deg':float(change.max()),
                'stable_recorded_command':stable,'outcome_speed_mps':speed,'root_support_relief_m':relief,
                'all_support_known':bool(valid.all()),'recorded_direction_native_xz':current.tolist(),
                'derived_velocity_native_xz':inferred.tolist()}
            selected.append(result);rows.append(result)
        records.append({'source_id':row['source_id'],'family':row.get('terrain_family'),'frames':len(world),
            'intent_sha256':expected,'source_frame_indices_sha256':frame_hash,
            'provenance':row['source_intent'],'windows':summarize(selected),
            'metadata_keys':sorted(k for k in row if any(word in k for word in ('source','intent','snapshot','manifest'))),
            'provenance_snapshots':{k:v for k,v in row.items() if k.endswith('_snapshot')}})
    stable=[r for r in rows if r['stable_recorded_command']]
    relief=[r for r in stable if r['all_support_known'] and r['root_support_relief_m']>.1]
    report={'recorded_clips':len(records),'all_windows':summarize(rows),
        'stable_command_windows':summarize(stable),'stable_command_terrain_change_windows':summarize(relief),
        'records':records,'largest_stable_disagreements':sorted(stable,key=lambda r:-r['direction_difference_deg'])[:24],
        'largest_terrain_disagreements':sorted(relief,key=lambda r:-r['direction_difference_deg'])[:24],
        'scope':'read-only; unit directions are not speed; no new labels or training launch',
        'capture_timing':'builder alone does not establish it; generator/export provenance requires separate review'}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ('recorded_clips','all_windows','stable_command_windows','stable_command_terrain_change_windows',
        'scope')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
