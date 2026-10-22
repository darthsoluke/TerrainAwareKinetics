"""Measure newly admitted training seconds and build their native dataset."""

import argparse
import json
from pathlib import Path
import time

from scripts.build_parc_only_dataset import build_dataset
from scripts.prepare_parc_only_catalog import bind_geometry_components,generated_recording_identity
from scripts.train_native_mb import sha256


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--cohort-size',type=int)
    args=p.parse_args()
    launch=read(args.run/'launch.json')
    if args.cohort_size is not None:
        if not 1<=args.cohort_size<=len(launch['jobs']):raise ValueError('Invalid original cohort size')
        launch['jobs']=launch['jobs'][:args.cohort_size]
    previous=read(args.run/'throughput.json')
    state={'status':'waiting_for_long_batches','target_new_train_seconds':3600,
           'deadline_seconds':1800,'started_at':launch['started_at'],'reserved_gpus':[0,7],
           'prior_pilot_data_counted':False,'training_approved':False}
    same_cohort=(previous and previous.get('started_at')==launch['started_at'] and
                 {(j['gpu'],j['seed']) for j in previous.get('jobs',[])}=={(j['gpu'],j['seed']) for j in launch['jobs']})
    achieved_at=previous.get('target_achieved_at') if same_cohort else None
    while True:
        rows=[];jobs=[];finished_count=0
        for job in launch['jobs']:
            directory=Path(job['finished']);summary=read(directory/'admission/summary.json')
            status=read(directory/'status.json') or {}
            generated=read(Path(job['generation_run'])/'status.json') or {}
            stage=status.get('status','not_started')
            catalog=read(directory/'admission/catalog.json')
            if summary and catalog:
                finished_count+=1;rows.extend(catalog['rows'])
            elif stage=='failed' or generated.get('status')=='failed':
                finished_count+=1
            jobs.append({'gpu':job['gpu'],'seed':job['seed'],'status':stage,
                         'error':status.get('error',generated.get('error')),
                         'admitted_seconds':summary.get('admitted_seconds',0) if summary else 0})
        seconds=0.;seen=set();admitted=0
        for row in rows:
            if row.get('disposition')!='admitted':continue
            if not row.get('whole_motion') or row.get('split')!='train':
                raise ValueError('Benchmark only counts complete new physical training clips')
            identity=row['motion_content_sha256']
            if identity in seen:
                row.update(disposition='excluded',exclusion_reason='duplicate canonical motion in long corpus')
                row.pop('split',None);continue
            seen.add(identity);seconds+=row['admitted_seconds'];admitted+=1
        if seconds>=3600 and achieved_at is None:achieved_at=time.time()
        state.update(jobs=jobs,new_admitted_train_seconds=seconds,new_admitted_clips=admitted,
            elapsed_seconds=time.time()-launch['started_at'],target_achieved_at=achieved_at,
            target_seconds_elapsed=achieved_at-launch['started_at'] if achieved_at else None,
            one_hour_within_30_minutes=(achieved_at is not None and achieved_at-launch['started_at']<=1800))
        save(args.run/'throughput.json',state)
        if finished_count==len(launch['jobs']):break
        time.sleep(15)
    if not admitted:
        state['status']='no_admitted_long_motion';save(args.run/'throughput.json',state);return
    for row in rows:
        row['previous_upstream_identity']=row['upstream_identity']
        row['upstream_identity']=generated_recording_identity(row['physics_report_sha256'],row['physics_candidate_key'])
    binding=bind_geometry_components(rows)
    catalog={'schema':'parc.only_teacher_catalog.v1','rows':rows,
             'geometry_split_binding':binding,'quality_approved':False}
    save(args.run/'catalog.json',catalog)
    state['status']='building_native_dataset';save(args.run/'throughput.json',state)
    native=args.run/'parc_native'
    build_dataset(args.run/'catalog.json',native,incremental_base=args.base/'parc_native')
    state.update(status='long_corpus_built',dataset=str(native),
                 dataset_manifest_sha256=sha256(native/'native_manifest.json'),
                 native_dataset_ready_elapsed_seconds=time.time()-launch['started_at'])
    save(args.run/'throughput.json',state)
    save(args.run/'dataset_receipt.json',state)


if __name__=='__main__':main()
