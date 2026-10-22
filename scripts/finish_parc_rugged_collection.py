"""Retain every candidate and assemble screened data with the parent's statistics."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

from scripts.build_parc_only_dataset import build_dataset
from scripts.prepare_parc_only_catalog import bind_geometry_components,generated_recording_identity
from scripts.train_native_mb import sha256,verify_dataset


def write(path,value):
    pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False));pending.replace(path)


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='waiting_for_collection',training_started=False,quality_approved=False)
    plan=json.loads((args.collection/'plan.json').read_text())
    expected=sha256(args.collection/'plan.json')
    while True:
        workers=[json.loads((args.collection/('worker_'+str(gpu)+'.json')).read_text()) for gpu in (0,2)]
        if any(w['plan_sha256']!=expected for w in workers):raise ValueError('Collection plan changed')
        state['workers']=[dict(gpu=w['gpu'],status=w['status'],jobs=[
            dict(layout=j['layout'],status=j['status']) for j in w['jobs']]) for w in workers]
        write(args.output/'state.json',state)
        if any(w['status']=='failed' for w in workers):raise ValueError('Collection worker failed; preserve all partial artifacts')
        if all(w['status']=='collection_complete' for w in workers):break
        time.sleep(15)
    inventory=[];rows=[]
    for job in plan['jobs']:
        directory=args.collection/('layout_%02d'%job['layout'])
        generated=json.loads((directory/'generation/generation.json').read_text())
        if generated['seed']!=job['seed'] or len(generated['candidates'])!=job['candidates']:
            raise ValueError('Candidate identity/budget differs')
        inventory.extend(dict(layout=job['layout'],family=job['family'],split=job['split'],
            candidate_id=r['candidate_id'],frames=r['frames'],duration_s=r['duration_s'],
            endpoint_reached=r['endpoint_reached'],geometric_route_covered=r['geometric_route_covered'],
            motion_sha256=r['sha256'],intent_sha256=r['intent_sha256'],motion=r['motion'])
            for r in generated['candidates'])
        catalog=directory/'admission/catalog.json'
        if catalog.exists():
            new=json.loads(catalog.read_text())['rows']
            for row in new:
                if row['disposition']=='admitted' and row['split']!=job['split']:
                    raise ValueError('Post-screen split change')
                row['upstream_identity']=generated_recording_identity(row['physics_report_sha256'],row['physics_candidate_key'])
            rows.extend(new)
    if len(inventory)!=32:raise ValueError('Must preserve all32 candidates')
    write(args.output/'candidate_inventory.json',dict(rows=inventory,failures_preserved=True))
    if not rows:raise ValueError('No physics/SOMA records available')
    counts=Counter((r.get('split'),r['terrain_family']) for r in rows if r['disposition']=='admitted')
    state.update(status='building_dataset',candidate_count=len(inventory),
        native_complete_routes=sum(r['geometric_route_covered'] for r in inventory),
        admitted_by_split_family=[dict(split=s,family=f,count=n) for (s,f),n in sorted(counts.items())])
    write(args.output/'state.json',state)
    binding=bind_geometry_components(rows)
    catalog=args.output/'catalog.json'
    write(catalog,dict(schema='parc.only_teacher_catalog.v1',rows=rows,
        geometry_split_binding=binding,quality_approved=False))
    parent=Path(plan['approved_model'])
    if sha256(parent)!=plan['approved_model_sha256']:raise ValueError('Approved model changed')
    checkpoint=torch.load(parent,map_location='cpu',mmap=True,weights_only=False)
    statistics_dataset=Path(checkpoint['contract']['dataset'])
    native=args.output/'parc_native'
    if args.anchor_dataset is not None:
        for name in ('mean.npy','std.npy'):
            if sha256(args.anchor_dataset/'stats'/name)!=sha256(statistics_dataset/'stats'/name):
                raise ValueError('PARC anchor data uses different model normalization')
        manifest=build_dataset(catalog,native,incremental_base=args.anchor_dataset)
        state['preserved_parc_anchor_records']=manifest['incremental_preserved_records']
        state['anchor_dataset']=str(args.anchor_dataset)
    else:
        manifest=build_dataset(catalog,native,reuse_stats_from=statistics_dataset)
    verify_dataset(native,manifest)
    state.update(status='dataset_built_source_audit_pending',dataset=str(native),
        native_manifest_sha256=sha256(native/'native_manifest.json'),
        statistics_reused_from=str(statistics_dataset),approved_model_sha256=sha256(parent),
        actor_optimizer_updates=0,source_exact_skin_and_route_audit_pending=True,
        command_semantics='recorded unit direction, not m/s; speed mapping not yet approved')
    write(args.output/'state.json',state);write(args.output/'complete.json',state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--anchor-dataset',type=Path)
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.exists():write(args.output/'failure.json',dict(error=type(error).__name__+': '+str(error)))
        raise
