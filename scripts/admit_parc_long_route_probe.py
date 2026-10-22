"""Apply existing SOMA screens to complete physical long-route recordings."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.prepare_parc_only_catalog import generated_rows,bind_geometry_components
from scripts.finish_parc_diverse_dataset import validate_retarget_execution
from scripts.parc_collection_budget import motion_identity
from scripts.train_native_mb import sha256


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def partition_selection(selection,workers):
    if workers<1:
        raise ValueError('workers must be positive')
    if len({item['behavior'] for item in selection}) != len(selection):
        raise ValueError('Duplicate retarget source identity')
    return [selection[index::workers] for index in range(min(workers,len(selection)))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--split',choices=('train','validation','test'),default='validation')
    parser.add_argument('--terrain-family',choices=('mixed','stairs','rough','ramp'),default='mixed')
    parser.add_argument('--sdf-gpu',type=int,choices=range(8))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from run_parc_curated_pipeline import attach_export_intent
    attach_export_intent(args.recording/'inputs',args.recording/'export')
    rows=generated_rows([args.recording/'export'])
    selection=[]
    for row in rows:
        row.update(terrain_family=args.terrain_family,scene_id=row['source_group'],
                   split=args.split,collection_role='long_route_experiment',
                   kinematic_optimization_applied=False)
        if row['whole_motion']:
            selection.append({'behavior':row['source_id'],'source_group':row['source_group'],
                'source':{'local_file':row['source_path'],'sha256':row['source_sha256']}})
        else:
            row.update(disposition='excluded',exclusion_reason='partial physics recording')
            row.pop('split',None)
    save(args.output/'selection.json',{'selected_items':selection})
    processes=[]
    for index,items in enumerate(partition_selection(selection,args.workers)):
        selected=args.output/f'selection_{index:02d}.json'
        save(selected,{'selected_items':items})
        target=args.output/f'retarget_{index:02d}'
        log=(args.output/f'retarget_{index:02d}.log').open('w')
        child=subprocess.Popen([sys.executable,'-u','-m','scripts.prepare_parc_soma',
            '--selection',str(selected),'--output',str(target),
            *(['--sdf-device','cuda:0','--sdf-chunk-size','128'] if args.sdf_gpu is not None else [])],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if args.sdf_gpu is None else str(args.sdf_gpu),
                     OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
            stdout=log,stderr=subprocess.STDOUT)
        processes.append((child,log,target))
    report={'items':[]}
    for child,log,target in processes:
        child.wait();log.close()
        report['items'].extend(json.loads((target/'report.json').read_text())['items'])
    validate_retarget_execution(report,[item['behavior'] for item in selection])
    prepared={item['behavior']:item for item in report['items']}
    for row in rows:
        if row.get('disposition')=='excluded':continue
        result=prepared[row['source_id']]
        shard=Path(result['shard']);path=shard.parent/'report.json';quality=json.loads(path.read_text())
        if quality['screen_status']=='pass' and quality['frames']>=65:
            identity,seconds=motion_identity(shard)
            row.update(disposition='admitted',retarget_path=str(shard),retarget_sha256=sha256(shard),
                report_path=str(path),report_sha256=sha256(path),admitted_seconds=seconds,
                motion_content_sha256=identity)
        else:
            row.update(disposition='excluded',exclusion_reason='SOMA geometry or continuity screen failed',
                       screen_failures=quality.get('failures'))
            row.pop('split',None)
    binding=bind_geometry_components(rows)
    save(args.output/'catalog.json',{'schema':'parc.only_teacher_catalog.v1','rows':rows,
         'geometry_split_binding':binding,'quality_approved':False})
    summary={'status':'diagnostic_admission_complete','admitted_clips':sum(r['disposition']=='admitted' for r in rows),
        'admitted_seconds':sum(r.get('admitted_seconds',0.) for r in rows),
        'split':args.split,'retarget_workers':args.workers,'sdf_gpu':args.sdf_gpu,
        'training_approved':False,'stage_specific_geometric_teacher':True,
        'complete_planned_route_verified':False,'limitations':'Only existing physical full-recording and SOMA screens assessed; not all requested skills certified'}
    save(args.output/'summary.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':main()
