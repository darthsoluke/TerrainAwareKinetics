"""Incrementally retain additional qualified long clips without altering a benchmark."""

import argparse
import json
from pathlib import Path

from scripts.build_parc_only_dataset import build_dataset
from scripts.prepare_parc_only_catalog import bind_geometry_components,generated_recording_identity
from scripts.train_native_mb import sha256,verify_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dataset',type=Path,required=True)
    p.add_argument('--catalog',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for path in args.catalog:rows.extend(json.loads(path.read_text())['rows'])
    for row in rows:
        row['upstream_identity']=generated_recording_identity(row['physics_report_sha256'],row['physics_candidate_key'])
    binding=bind_geometry_components(rows)
    catalog=args.output/'catalog.json'
    catalog.write_text(json.dumps({'schema':'parc.only_teacher_catalog.v1','rows':rows,
        'geometry_split_binding':binding,'quality_approved':False},indent=2))
    native=args.output/'parc_native'
    manifest=build_dataset(catalog,native,incremental_base=args.base_dataset)
    verify_dataset(native,manifest)
    summary={'dataset':str(native),'dataset_manifest_sha256':sha256(native/'native_manifest.json'),
        'base_dataset':str(args.base_dataset),'added_clips':sum(r.get('disposition')=='admitted' for r in rows),
        'added_train_seconds':sum(r.get('admitted_seconds',0) for r in rows if r.get('disposition')=='admitted' and r['split']=='train'),
        'added_validation_seconds':sum(r.get('admitted_seconds',0) for r in rows if r.get('disposition')=='admitted' and r['split']=='validation'),
        'benchmark_unchanged':True,'training_approved':False}
    (args.output/'dataset_receipt.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))


if __name__=='__main__':main()
