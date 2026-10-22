"""Verify newly generated evaluation-only SOMA routes without a train manifest."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation,_native_root_world
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.build_parc_only_dataset import _proxy_clip,_validate_prepare_report,_HELPERS
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.prepare_unified_pose_cases import stair_intervals
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'intent_proxies').mkdir()
    torch.set_num_threads(4)
    plan=json.loads((a.collection/'plan.json').read_text())
    job=plan['jobs'][a.layout]
    if job['split'] not in ('validation','test') or not plan['current_full_training_excluded']:
        raise ValueError('Explicit evaluation-only layout required')
    directory=a.collection/('layout_%02d'%a.layout)
    generated=json.loads((directory/'generation/generation.json').read_text())
    if len(generated['candidates'])!=4 or generated['prefix_contract']['max_prefix_error']>3e-5:
        raise ValueError('Candidate inventory or factual-prefix contract differs')
    catalog=json.loads((directory/'admission/catalog.json').read_text())
    training=json.loads((a.mixed/'accepted.json').read_text())
    forbidden_groups={row['source_group'] for row in training if row['split']=='train'}
    forbidden_motion={row.get('source_original_sha256') for row in training if row['split']=='train'}-{None}
    forbidden_terrain={row.get('terrain_content_sha256') for row in training if row['split']=='train'}-{None}
    rep=build_native_representation(a.mixed/'stats')
    geometry=JointGeometryObjective(build_native_representation(a.mixed/'stats').to(a.device),a.device)
    cases,rows=[],[]
    for original in catalog['rows']:
        row=dict(original)
        if row['disposition']!='admitted':
            rows.append(dict(source_id=row['source_id'],disposition='legacy_admission_rejected',original=row))
            continue
        if (row['split']!=job['split'] or row['source_group'] in forbidden_groups
                or row['source_sha256'] in forbidden_motion or row['terrain_content_sha256'] in forbidden_terrain):
            raise ValueError('New evaluation source overlaps current training identity')
        paths={}
        for name,path_key,hash_key in (
            ('source','source_path','source_sha256'),('retarget','retarget_path','retarget_sha256'),
            ('report','report_path','report_sha256'),('intent','intent_path','intent_sha256'),
            ('physics_report','physics_report_path','physics_report_sha256'),
            ('export_manifest','export_manifest_path','export_manifest_sha256'),
            ('upstream_manifest','upstream_manifest_path','upstream_manifest_sha256')):
            path=Path(row[path_key])
            if sha256(path)!=row[hash_key]:
                raise ValueError('Source provenance changed: '+path_key)
            paths[name]=path
        report=json.loads(paths['report'].read_text())
        _validate_prepare_report(row,paths,report)
        _HELPERS._validate_physics_retest(paths['physics_report'],row['physics_report_sha256'],
            paths['export_manifest'],row['export_manifest_sha256'],row['physics_candidate_key'],
            paths['source'],row['source_sha256'])
        row.update(_paths=paths,_report=report)
        clip=_proxy_clip(row['source_id'],row,a.output/'intent_proxies')
        features,_,_,_,roundtrip,_,frame=_HELPERS._native_encode(
            clip['local_rot_mats'],clip['root_positions'],rep,clip,return_frame=True)
        canonical=torch.from_numpy(features)
        roots=_native_root_world(canonical,torch.from_numpy(frame['init_heading_quat']),
            torch.from_numpy(frame['root_pos_init_xz']),rep.indices['global_root_pos'],rep.indices['global_root_heading'])
        world=rep.change_first_heading(canonical[None],torch.atan2(roots[:1,4],roots[:1,3]),False,False)
        world[...,[0,2]]+=roots[:1,[0,2]][None]
        torch.testing.assert_close(world[0,:,:3],torch.from_numpy(clip['root_positions']),atol=5e-5,rtol=1e-5)
        expected=rep.skeleton.fk(torch.from_numpy(clip['local_rot_mats']),torch.from_numpy(clip['root_positions']))[1]
        actual=rep.inverse(world,False,return_all=False,joint_positions_from='global_rot_data')['posed_joints'][0]
        fk_error=float((actual-expected).norm(dim=-1).max())
        if fk_error>5e-5:
            raise ValueError('Physical world restoration differs from original retarget FK')
        scene={key:torch.as_tensor(clip[key]).clone() for key in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
        whole=exact_skin_metrics(geometry,world.to(a.device),scene)
        intervals=stair_intervals(world[0],scene)
        record=dict(source_id=row['source_id'],source_group=row['source_group'],frames=len(canonical),
            original_native_fk_error_m=roundtrip,restored_world_fk_error_m=fk_error,
            full_source_skin=whole,intervals=[],all_source_frames_preserved=True)
        for interval in intervals:
            start,length=interval['start'],interval['length']
            source=world[0,start:start+length].clone()
            skin=exact_skin_metrics(geometry,source[None].to(a.device),scene)
            good=skin['body_prism_max_m']<=.02 and skin['unknown_body_fraction']==0 and skin['uncertain_inside_samples']==0
            record['intervals'].append(dict(interval=interval,skin=skin,source_gate_pass=good))
            if good:
                cases.append(dict(name='stairs',label=row['source_id']+'-'+interval['entry'],
                    entry=interval['entry'],row=original,world=source,scene=scene,start_frame=start,
                    native_split=job['split'],interval=interval,source_screen=skin,
                    evaluation_role='new_reserved_geometry',current_full_training_excluded=True,
                    lifecycle_unseen='Generated after initial model; complete upstream model history not claimed.'))
        rows.append(record)
        write_json(a.output/'progress.json',dict(rows=rows,selected=len(cases)))
    torch.save(cases,a.output/'heldout-cases.pt')
    counts=Counter(case['entry'] for case in cases)
    write_json(a.output/'complete.json',dict(screening_complete=True,rows=rows,selected_counts=dict(counts),
        distinct_qualified_geometries=len({case['row']['source_group'] for case in cases}),
        first_layout_has_clean_up_and_down=counts['up']>0 and counts['down']>0,
        generated_candidates=4,source_catalog_rows=len(catalog['rows']),
        generation_report_sha256=sha256(directory/'generation/generation.json'),
        catalog_sha256=sha256(directory/'admission/catalog.json'),plan_sha256=sha256(a.collection/'plan.json'),
        cases_sha256=sha256(a.output/'heldout-cases.pt'),current_full_training_excluded=True,
        geometry_gate=dict(peak_m=.02,unknown_fraction=0,uncertain_inside_samples=0),
        model_evaluated=False,optimizer_updates=0,quality_approved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','mixed','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--layout',type=int,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output/'failure.json',dict(error=repr(error)))
        raise
