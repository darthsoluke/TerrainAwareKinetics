"""Keep regression routes and select genuinely split-held-out complete sources."""

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.prepare_unified_pose_cases import stair_intervals
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    records=json.loads((a.mixed/'accepted.json').read_text())
    groups=defaultdict(set)
    for row in records:
        groups[row['source_group']].add(row['split'])
    if any(len(split)!=1 for split in groups.values()):
        raise ValueError('Source recording families cross splits')
    old=torch.load(a.regression_cases,map_location='cpu',weights_only=False)
    regression=[]
    for case in old:
        if case.get('entry') not in ('up','down'):
            continue
        split=next(iter(groups[case['row']['source_group']])) if case['row']['source_group'] in groups else 'absent'
        regression.append(dict(label=case['label'],source_id=case['row']['source_id'],
            source_group=case['row']['source_group'],current_corpus_group_split=split,
            role='legacy_regression',lifecycle_unseen=False))
    geometry=JointGeometryObjective(build_native_representation(a.mixed/'stats').to(a.device),a.device)
    selected,dispositions=[],[]
    counts=Counter()
    used=set()
    seen=set()
    train_groups={row['source_group'] for row in records if row['split']=='train'}
    train_origins={row.get('source_original_sha256') for row in records if row['split']=='train'}-{None}
    train_converted={row.get('converted_shard_sha256') for row in records if row['split']=='train'}-{None}
    if a.reuse_preparation:
        previous=json.loads((a.reuse_preparation/'complete.json').read_text())
        if (not previous['preparation_completed'] or previous['accepted_sha256']!=sha256(a.mixed/'accepted.json')
                or previous['regression_cases_sha256']!=sha256(a.regression_cases)
                or previous['heldout_cases_sha256']!=sha256(a.reuse_preparation/'heldout-cases.pt')):
            raise ValueError('Prior source preparation identity differs')
        selected=torch.load(a.reuse_preparation/'heldout-cases.pt',map_location='cpu',weights_only=False)
        dispositions=list(previous['dispositions'])
        seen={row['source_id'] for row in dispositions}
        for case in selected:
            counts[(case['native_split'],case['entry'])]+=1
            used.add((case['native_split'],case['row']['source_group']))
    paths=list(a.additional_heldout) if a.reuse_preparation else [a.mixed,*a.additional_heldout]
    sources=[]
    duplicates=0
    for path in paths:
        sources.append(dict(path=str(path),accepted_sha256=sha256(path/'accepted.json'),
            native_manifest_sha256=sha256(path/'native_manifest.json')))
    for path,split in [(path,split) for path in paths for split in ('validation','test')]:
        # Each dataset normalizes with its own stats before restoring physical world values.
        data=TerrainLearningDataset(path,build_native_representation(path/'stats'),split)
        order=sorted(range(len(data.records)),key=lambda i:(data.records[i].get('terrain_family')!='stairs',data.records[i]['source_id']))
        for index in order:
            row=data.records[index]
            if not row['origin'].startswith('parc'):
                continue
            if row['source_id'] in seen:
                duplicates+=1
                continue
            seen.add(row['source_id'])
            if (row['source_group'] in train_groups or row.get('source_original_sha256') in train_origins
                    or row.get('converted_shard_sha256') in train_converted):
                dispositions.append(dict(source_id=row['source_id'],split=split,source_dataset=str(path),
                    disposition='excluded_current_training_identity_overlap'))
                continue
            item=data.full_reference_item(index)
            intervals=stair_intervals(item['world'],item['scene'])
            if not intervals:
                dispositions.append(dict(source_id=row['source_id'],split=split,
                    terrain_family=row.get('terrain_family'),frames=len(item['world']),
                    disposition='no_complete_stair_interval_under_existing_definition'))
            for interval in intervals:
                entry=interval['entry']
                if counts[(split,entry)]>=3 or (split,row['source_group']) in used:
                    dispositions.append(dict(source_id=row['source_id'],split=split,entry=entry,
                        interval=interval,disposition='complete_candidate_reserved_after_quota_or_family'))
                    continue
                start,length=interval['start'],interval['length']
                source=item['world'][start:start+length].clone()
                metrics=exact_skin_metrics(geometry,source[None].to(a.device),item['scene'])
                good=metrics['body_prism_max_m']<=.02 and metrics['unknown_body_fraction']==0 and metrics['uncertain_inside_samples']==0
                dispositions.append(dict(source_id=row['source_id'],split=split,entry=entry,
                    interval=interval,metrics=metrics,disposition='selected' if good else 'source_geometry_rejected'))
                if not good:
                    continue
                selected.append(dict(name='stairs',label=row['source_id']+'-'+entry,
                    entry=entry,row=row,scene=item['scene'],world=source,start_frame=start,
                    interval=interval,native_split=split,source_screen=metrics,
                    source_dataset=str(path),
                    evaluation_role='current_corpus_held_out',lifecycle_unseen='not_exhaustively_audited'))
                counts[(split,entry)]+=1
                used.add((split,row['source_group']))
            write_json(a.output/'progress.json',dict(inspected_parc_sources=len({row['source_id'] for row in dispositions}),
                selected_counts={':'.join(key):value for key,value in counts.items()}))
    torch.save(selected,a.output/'heldout-cases.pt')
    write_json(a.output/'complete.json',dict(preparation_completed=True,
        heldout_cases_sha256=sha256(a.output/'heldout-cases.pt'),
        mixed_manifest_sha256=sha256(a.mixed/'native_manifest.json'),
        accepted_sha256=sha256(a.mixed/'accepted.json'),regression_cases_sha256=sha256(a.regression_cases),
        additional_source_manifests=sources,reused_preparation_sha256=
            sha256(a.reuse_preparation/'complete.json') if a.reuse_preparation else None,
        duplicate_source_records_skipped=duplicates,
        regression_sources=regression,heldout_selected_counts={':'.join(key):value for key,value in counts.items()},
        requested_counts={split+':'+entry:3 for split in ('validation','test') for entry in ('up','down')},
        target_counts_met=all(counts[(split,entry)]==3 for split in ('validation','test') for entry in ('up','down')),
        dispositions=dispositions,source_frames_modified=False,actor_evaluated=False,
        unchanged_source_gate=dict(peak_m=.02,unknown_fraction=0,uncertain_inside_samples=0),
        route_definition='Existing stair_intervals:40real approach frames,>=3rise events,>=30cm elevation,24real plateau frames.',
        quality_approved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('mixed','regression-cases','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--additional-heldout',nargs='*',type=Path,default=[])
    parser.add_argument('--reuse-preparation',type=Path)
    args=parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output/'failure.json',dict(error=repr(error)))
        raise
