"""Read-only requested-versus-demonstrated direction/facing consistency."""
import argparse
from pathlib import Path

import torch

from scripts.train_joint_terrain_v3 import write_json


def angle(a,b):
    return float(torch.rad2deg(torch.atan2((a-b).sin(),(a-b).cos()).abs()))


@torch.no_grad()
def main(args):
    cases=torch.load(args.cases,map_location='cpu',weights_only=False);rows=[]
    for case in cases:
        world=case['world'];blocks=[]
        for i,command in enumerate(case['controls']):
            begin=i*8;window=world[begin:begin+33]
            actual=(window[31,[0,2]]-window[3,[0,2]])/(28/30)
            requested=command[:2]
            heading=torch.atan2(command[3],command[2])
            source_heading=torch.atan2(window[31,4],window[31,3])
            blocks.append(dict(block=i,requested_speed=float(requested.norm()),source_mean_speed=float(actual.norm()),
                request_vs_motion_degrees=angle(torch.atan2(requested[0],requested[1]),torch.atan2(actual[0],actual[1]))
                    if min(float(requested.norm()),float(actual.norm()))>.2 else None,
                requested_vs_target_facing_degrees=angle(heading,source_heading)))
        rows.append(dict(label=case['label'],category=case['category'],source_group=case['row'].get('source_group'),blocks=blocks))
    write_json(args.output,dict(rows=rows,optimizer_updates=0,source_pose_used_only_for_audit=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
