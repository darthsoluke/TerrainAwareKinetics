"""Wait for finite PARC collection, retarget admitted physics and export via Drive."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from scripts.prepare_parc_only_catalog import generated_rows, bind_geometry_components
from scripts.build_parc_only_dataset import build_dataset
from scripts.train_native_mb import sha256


def save(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def validate_retarget_execution(report, expected_ids):
    items = report.get('items', [])
    processed = {row['behavior'] for row in items if row.get('execution_status') == 'processed'}
    failed = sorted(set(expected_ids)-processed)
    if failed:
        errors = {row['behavior']:row.get('error','missing processed result') for row in items
                  if row['behavior'] in failed}
        raise RuntimeError('Retarget execution failed, not a motion quality rejection: '+json.dumps(errors))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','base-dataset','output','drive-config'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--collection-cloud')
    p.add_argument('--local-only',action='store_true')
    p.add_argument('--scene-id')
    p.add_argument('--catalog-only',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    state={'status':'waiting_for_collection','training_approved':False}
    state_path=args.output/'state.json'
    save(state_path,state)
    try:
        while True:
            if args.collection_cloud:
                env=dict(os.environ,HTTP_PROXY='http://127.0.0.1:27890',HTTPS_PROXY='http://127.0.0.1:27890')
                subprocess.run(['rclone','--config',str(args.drive_config),'copy',args.collection_cloud,
                    str(args.collection),'--retries','1','--low-level-retries','1'],env=env,check=False)
            if not (args.collection/'plan/plan.json').exists():
                time.sleep(30)
                continue
            plan=json.loads((args.collection/'plan/plan.json').read_text())
            if args.scene_id:
                plan['jobs'] = [job for job in plan['jobs'] if job['id'] == args.scene_id]
                if len(plan['jobs']) != 1:
                    raise ValueError('Unknown scene for incremental admission')
            workers=[args.collection/f'worker_{i}.json' for i in (0,1)]
            complete=all(path.exists() and json.loads(path.read_text())['status']=='collection_finished_admission_pending'
                         for path in workers) and all((args.collection/f'recovery_{i}.json').exists() for i in (0,1))
            if args.scene_id:
                complete = (args.collection/args.scene_id/'receipt.json').is_file()
            state.update(completed_scenes=sum((args.collection/job['id']/'receipt.json').exists() for job in plan['jobs']),
                         total_scenes=len(plan['jobs']),updated_at=time.time())
            save(state_path,state)
            if complete:
                break
            time.sleep(30)
        exports, receipts = [], []
        for job in plan['jobs']:
            receipt=json.loads((args.collection/job['id']/'receipt.json').read_text())
            receipts.append(receipt)
            if receipt['status']=='physics_exported' and receipt['full_recordings']:
                export=args.collection/job['id']/'export'
                if sha256(export/'manifest.json')!=receipt['manifest_sha256']:
                    raise ValueError('Export changed after collection')
                exports.append(export)
        if not exports:
            if args.catalog_only:
                save(args.output/'catalog.json', {'schema':'parc.only_teacher_catalog.v1', 'rows':[]})
                state.update(status='no_complete_physics_inputs', new_admitted=0)
                save(state_path,state)
                return
            raise ValueError('No complete physics recordings available')
        fresh=generated_rows(exports)
        jobs_by_id={job['id']:job for job in plan['jobs']}
        selection=[]
        for row in fresh:
            scene=Path(row['export_manifest_path']).parent.parent.name
            job=jobs_by_id[scene]
            row.update(terrain_family=job['family'], split=job['split'], scene_id=scene)
            if row['whole_motion']:
                selection.append({'behavior':row['source_id'],'source_group':row['source_group'],
                    'source':{'local_file':row['source_path'],'sha256':row['source_sha256']}})
            else:
                row.update(disposition='excluded',exclusion_reason='partial physics recording')
                row.pop('split',None)
        save(args.output/'selection.json',{'selected_items':selection})
        state.update(status='retargeting',complete_physics_inputs=len(selection),export_groups=len(exports))
        save(state_path,state)
        with (args.output/'retarget.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.prepare_parc_soma','--selection',
                str(args.output/'selection.json'),'--output',str(args.output/'retarget')],
                stdout=log,stderr=subprocess.STDOUT,check=False)
        retarget_report=json.loads((args.output/'retarget/report.json').read_text())
        validate_retarget_execution(retarget_report,[item['behavior'] for item in selection])
        prepared={row['behavior']:row for row in retarget_report['items']}
        for row in fresh:
            if row.get('disposition')=='excluded':
                continue
            result=prepared.get(row['source_id'],{})
            shard=Path(result['shard']) if result.get('shard') else None
            report=shard.parent/'report.json' if shard else None
            quality=json.loads(report.read_text()) if report and report.exists() else {}
            if result.get('execution_status')=='processed' and quality.get('screen_status')=='pass' and quality.get('frames',0)>=65:
                row.update(disposition='admitted',retarget_path=str(shard),retarget_sha256=sha256(shard),
                           report_path=str(report),report_sha256=sha256(report))
            else:
                row.update(disposition='excluded',exclusion_reason='SOMA execution, geometry, continuity or frame contract failed')
                row.pop('split',None)
        rows=fresh
        binding=bind_geometry_components(rows)
        owners={}
        for row in rows:
            if row.get('disposition')!='admitted':
                continue
            group=row['source_group']
            if group in owners and owners[group]!=row['split']:
                raise ValueError('New geometry conflicts with immutable source splits; do not silently reassign')
            owners[group]=row['split']
        counts=Counter((row['terrain_family'],row['split']) for row in fresh if row.get('disposition')=='admitted')
        state.update(status='building_native_dataset',new_admitted=sum(counts.values()),
                     new_coverage={f'{family}/{split}':count for (family,split),count in counts.items()})
        save(state_path,state)
        if not args.catalog_only and not any(split=='train' for _,split in counts):
            raise ValueError('No newly admitted training clips')
        catalog={'schema':'parc.only_teacher_catalog.v1','rows':rows,'geometry_split_binding':binding,
                 'quality_approved':False,'split_policy':'old splits immutable; new complete layout groups preassigned'}
        save(args.output/'catalog.json',catalog)
        if args.catalog_only:
            state.update(status='scene_admission_complete')
            save(state_path,state)
            return
        native=args.output/'parc_native'
        build_dataset(args.output/'catalog.json',native,incremental_base=args.base_dataset)
        if args.local_only:
            state.update(status='dataset_ready_local',dataset=str(native),
                         dataset_manifest_sha256=sha256(native/'native_manifest.json'))
            save(state_path,state)
            save(args.output/'dataset_ready.json',state)
            return
        archive=args.output/'parc-diverse-native.tar.gz'
        with tarfile.open(archive,'w:gz') as stream:
            stream.add(native,arcname='parc_native')
        transfer={'archive':archive.name,'sha256':sha256(archive),'bytes':archive.stat().st_size,
                  'dataset_manifest_sha256':sha256(native/'native_manifest.json'),
                  'new_coverage':state['new_coverage'],'scene_count':len(plan['jobs']),
                  'inference_location':'local PC only','quality_approved':False}
        save(args.output/'transfer.json',transfer)
        env=dict(os.environ,HTTP_PROXY='http://127.0.0.1:27890',HTTPS_PROXY='http://127.0.0.1:27890')
        remote='mb-transfer:datasets/parc-diverse-20261010-v1'
        for file in (archive,args.output/'transfer.json'):
            subprocess.run(['rclone','--config',str(args.drive_config),'copyto',str(file),remote+'/'+file.name,
                            '--immutable'],env=env,check=True)
        state.update(status='dataset_uploaded',transfer=transfer,private_drive=remote)
        save(state_path,state)
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error))
        save(state_path,state)
        raise


if __name__=='__main__':
    main()
