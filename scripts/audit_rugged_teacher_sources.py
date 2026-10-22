"""Audit new paired teachers only; do not update the accepted controller."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.prepare_joint_sdf_context_pilot import recorded_direction
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    catalog=json.loads(args.catalog.read_text())
    selected={row['source_id']:row for row in catalog['rows'] if row['disposition']=='admitted'}
    rep=build_native_representation(args.dataset/'stats').to('cuda:0')
    geometry=JointGeometryObjective(rep,'cuda:0');rows=[];seen=set()
    for split in ('train','validation','test'):
        data=TerrainLearningDataset(args.dataset,rep,split)
        for index,row in enumerate(data.records):
            if row['source_id'] not in selected:continue
            item=data.full_reference_item(index);world=item['world'][None].cuda()
            direction=recorded_direction(data,row,world.shape[1])
            metrics=exact_skin_metrics(geometry,world,item['scene'])
            _,points=geometry.geometry(world)
            gaps,known=heightfield_gap(points[0,:,:geometry.sole_count],item['scene'])
            lowest=gaps.amin(-1)
            source_speed=torch.diff(world[0,:,:3][:,[0,2]],dim=0).norm(dim=-1)*30
            angles=torch.rad2deg(torch.acos((direction[1:]*direction[:-1]).sum(-1).clamp(-1,1)))
            passed=metrics['body_prism_max_m']<=.02 and metrics['unknown_body_fraction']==0 and metrics['uncertain_inside_samples']==0
            report=dict(source_id=row['source_id'],split=split,family=selected[row['source_id']]['terrain_family'],
                frames=world.shape[1],metrics=metrics,source_exact_skin_pass=bool(passed),
                source_support_audit=dict(both_soles_above5cm_fraction=float((lowest>.05).float().mean()),
                    lowest_sole_gap_median_m=float(lowest.median()),unknown_fraction=float((~known).float().mean())),
                captured_command_frame_count=len(direction),command_max_change_degrees=float(angles.max()),
                command_semantics='unit direction; not metric speed or future teacher body',
                actual_root_speed_median_mps=float(source_speed.median()),motion_modified=False)
            rows.append(report);seen.add(row['source_id'])
            write_json(args.output/'progress.json',dict(rows=rows));print(json.dumps(report),flush=True)
    if seen!=set(selected):raise ValueError('Some admitted sources were not audited')
    counts=Counter((r['split'],r['family']) for r in rows if r['source_exact_skin_pass'])
    write_json(args.output/'complete.json',dict(rows=rows,optimizer_updates=0,
        eligible_by_split_family=[dict(split=s,family=f,count=n) for (s,f),n in sorted(counts.items())],
        dataset_manifest_sha256=sha256(args.dataset/'native_manifest.json'),catalog_sha256=sha256(args.catalog),
        full_tracker_route_recheck_pending=True,actor_quality_approved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','catalog','output'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
