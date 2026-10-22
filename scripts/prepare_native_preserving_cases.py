"""Keep real stair approaches/exits and align source coverage to native24-frame reuse."""

import argparse
import math
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    original = torch.load(args.cases, map_location='cpu', weights_only=False)
    rep = build_native_representation(args.dataset/'stats')
    data = TerrainLearningDataset(args.dataset, rep, 'train')
    lookup = {row['source_id']:index for index,row in enumerate(data.records)}
    geometry = JointGeometryObjective(rep, 'cpu')
    cases, audit = [], []
    selected_train = set()
    for case in original:
        source = data.full_reference_item(lookup[case['row']['source_id']])
        start = case['start_frame']
        torch.testing.assert_close(source['world'][start:start+len(case['world'])],case['world'],atol=0,rtol=0)
        world = source['world'][start:]
        gap, known = heightfield_gap(world[:,:3],source['scene'])
        level = world[:,1]-gap
        goal = case['evaluation_goal_source_frame']-start
        sign = 1 if case['entry']=='up' else -1
        steps = torch.nonzero(torch.diff(level[:goal+1])*sign>.05).flatten()+1
        if len(steps)<3:
            raise ValueError('Original stair passage no longer has its physical risers')
        last = int(steps[-1])
        segments = math.ceil((max(goal,last+24)-3)/24)
        length = 33+(segments-1)*24
        published_end = 3+segments*24
        row = dict(label=case['label'],start=start,last_riser_frame=last,source_goal_frame=goal,
            native_segments=segments,required_real_frames=length,published_exit_frame=published_end)
        if len(world)<length:
            audit.append(dict(row,eligible=False,reason='Missing real exit/tail'));continue
        world = world[:length].clone()
        axis = world[published_end,[0,2]]-world[3,[0,2]]
        axis /= axis.norm()
        side = torch.stack((-axis[1],axis[0]))
        lateral = float(((world[3:published_end+1,[0,2]]-world[3,[0,2]])@side).abs().max())
        if not known[:length].all() or float(level[last:published_end+1].max()-level[last:published_end+1].min())>.03 or lateral>.35:
            audit.append(dict(row,eligible=False,reason='Exit is not a known straight flat continuation',lateral_m=lateral));continue
        metrics = exact_skin_metrics(geometry,world[None],source['scene'])
        if metrics['body_prism_max_m']>.02 or metrics['unknown_body_fraction'] or metrics['uncertain_inside_samples']:
            audit.append(dict(row,eligible=False,reason='Existing2cm source screen',metrics=metrics));continue
        train = case['pilot_split']=='train' and case['entry'] not in selected_train
        if train:selected_train.add(case['entry'])
        prepared = dict(case,world=world,native_segments=segments,publication_stride=24,
            original_goal_source_frame=case['evaluation_goal_source_frame'],
            evaluation_goal_source_frame=start+published_end,
            native_split='train' if train else 'review_only',source_screen=metrics)
        cases.append(prepared)
        audit.append(dict(row,eligible=True,native_split=prepared['native_split'],exit_frames=published_end-last,
            metrics=metrics,lateral_m=lateral))
        write_json(args.output/'progress.json',dict(rows=audit))
    write_json(args.output/'source_audit.json',dict(rows=audit))
    if selected_train != {'up','down'} or len(cases)<4:
        raise ValueError('Need one valid up/down train pair and at least two independent review cases')
    torch.save(cases,args.output/'cases.pt')
    write_json(args.output/'complete.json',dict(cases_sha256=sha256(args.output/'cases.pt'),
        original_cases_sha256=sha256(args.cases),dataset_manifest_sha256=sha256(args.dataset/'native_manifest.json'),
        rows=audit,source_frames_modified=False,synthetic_padding=False,publication_stride=24,
        control='Retained constant weak approach speed/facing labels;not captured keyboard input',
        quality_approved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cases','dataset','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
