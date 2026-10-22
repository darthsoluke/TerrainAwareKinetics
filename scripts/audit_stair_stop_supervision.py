"""Read-only audit of real stop/restart supervision and the pinned expert config."""

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import sys

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_response_loss import heightfield_gap


def runs(mask,minimum=24):
    start=None
    for i,active in enumerate(mask.tolist()+[False]):
        if active and start is None:start=i
        if not active and start is not None:
            if i-start>=minimum:yield start,i
            start=None


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    sys.path.insert(0,str(args.repo))
    sys.modules.setdefault('parc',importlib.import_module('PARC'))
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['cfg']
    expert={key:cfg.get(key) for key in ('target_type','target_dir_len_eps','target_dir_heading_eps',
        'use_target_loss','w_target','w_vel_root_pos','sequence_fps','sequence_duration','num_prev_states')}
    del saved
    rep=build_native_representation(args.dataset/'stats')
    counts=Counter();examples=[];errors=[];root_quiet_clips=[]
    for split in ('train','validation','test'):
        data=TerrainLearningDataset(args.dataset,rep,split,cache_bytes=128*1024**2)
        for index,row in enumerate(data.records):
            try:
                item=data.full_reference_item(index)
                world=item['world'];root=world[:,:3];frames=len(root)
                speed=torch.diff(root[:,[0,2]],dim=0).norm(dim=-1)*30
                contacts=world[:,rep.indices['foot_contacts']]
                double_support=(contacts[:,:2].amax(-1)>.5)&(contacts[:,2:].amax(-1)>.5)
                quiet=torch.cat((torch.tensor([False]),speed<=.12))&double_support
                domain='bones' if row['origin']=='bones_seed' else 'parc'
                counts[split+':'+domain+':clips']+=1
                root_only=torch.cat((torch.tensor([False]),speed<=.12))
                root_runs=list(runs(root_only))
                counts[split+':'+domain+':root_only_rest_runs']+=len(root_runs)
                if domain=='parc' and root_runs:
                    root_quiet_clips.append(dict(source_id=row['source_id'],split=split,
                        runs=[list(interval) for interval in root_runs],
                        supported_rest_runs=len(list(runs(quiet))),
                        bilateral_contact_fraction=float(double_support[root_only].float().mean())))
                for begin,end in runs(quiet):
                    counts[split+':'+domain+':rest_runs']+=1
                    before=speed[max(0,begin-24):max(0,begin-3)]
                    brake=bool(len(before)>=8 and before.median()>=.3)
                    after=speed[end:min(frames-1,end+30)]
                    restart=bool(len(after)>=8 and after.median()>=.3)
                    if not brake:continue
                    delta=root[begin-3,[0,2]]-root[max(0,begin-24),[0,2]]
                    axis=delta/delta.norm().clamp_min(1e-6)
                    probe=root.new_zeros(41,3)
                    probe[:,[0,2]]=root[begin,[0,2]]+torch.linspace(-2.,2.,41)[:,None]*axis
                    gap,known=heightfield_gap(probe,item['scene'])
                    levels=-gap[known]
                    center_gap,center_known=heightfield_gap(root[begin:begin+1],item['scene'])
                    center_level=root[begin,1]-center_gap[0]
                    middle=bool(len(levels) and center_known.all() and levels.min()+.05<center_level<levels.max()-.05)
                    counts[split+':'+domain+':braking_runs']+=1
                    if restart:counts[split+':'+domain+':brake_rest_restart']+=1
                    if middle:counts[split+':'+domain+':possible_mid_terrain_brake']+=1
                    examples.append(dict(source_id=row['source_id'],split=split,domain=domain,
                        source_index=index,category=row.get('control_category',row.get('terrain_family')),
                        rest_start=begin,rest_end=end,frames=frames,entry_speed_mps=float(before.median()),
                        restart=restart,possible_mid_terrain=middle,
                        weak_intent_provenance='inferred from real motion, not recorded key presses',
                        contact_scope='existing four-channel MB terrain-relative contact labels; full skin not audited'))
            except Exception as error:
                errors.append(dict(source_id=row['source_id'],split=split,error=str(error)))
        (args.output/'progress.json').write_text(json.dumps(dict(counts=dict(counts),errors=errors),indent=2))
    report=dict(expert_config=expert,counts=dict(counts),examples=examples,errors=errors,
        parc_root_only_quiet_clips=root_quiet_clips,
        updates=0,motion_modified=False,quality_approved=False,
        caveat='Mid-terrain root height is a candidate indicator, not proof of a regular stair or full sole support. Examples require a geometry screen before training.')
    (args.output/'complete.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(expert_config=expert,counts=dict(counts),errors=len(errors),examples=len(examples))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','repo','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
