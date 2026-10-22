"""Screen completed reserved layouts and assemble evaluation-only route cases."""

import argparse
from collections import Counter,defaultdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


def save(path,value):
    temporary=path.with_suffix('.tmp')
    write_json(temporary,value)
    temporary.replace(path)


def check_screen(root,index,plan_hash,collection):
    folder=root/('heldout_layout'+str(index)+'_screen_v1')
    if not (folder/'complete.json').exists():
        return None
    report=json.loads((folder/'complete.json').read_text())
    if (not report['screening_complete'] or report['plan_sha256']!=plan_hash
            or report['catalog_sha256']!=sha256(collection/('layout_%02d'%index)/'admission/catalog.json')
            or report['cases_sha256']!=sha256(folder/'heldout-cases.pt')
            or report['geometry_gate']!=dict(peak_m=.02,unknown_fraction=0,uncertain_inside_samples=0)):
        raise ValueError('Existing exact source screen identity differs for layout'+str(index))
    return folder,report


def main(a):
    root=a.base/'pfnn_mixed_20261015'
    code=Path(__file__).resolve().parents[1]
    if not a.worker:
        if a.output.exists() or a.output.with_suffix('.process.json').exists():
            raise ValueError('Evaluation finisher already exists')
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--base',str(a.base),
            '--output',str(a.output),'--gpu',str(a.gpu),'--worker']
        with a.output.with_suffix('.log').open('x') as stream:
            child=subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONPATH='.:out/vendor/kimodo',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        receipt=dict(pid=child.pid,command=command,code=str(code),gpu=a.gpu,
            training_started=False,current_full_training_excluded=True)
        save(a.output.with_suffix('.process.json'),receipt)
        print(json.dumps(receipt),flush=True)
        return
    a.output.mkdir(parents=True,exist_ok=False)
    state=dict(pid=os.getpid(),status='waiting_for_source_collection',finished_layouts=[],training_started=False)
    collection=root/'heldout_collection_v1'
    plan=json.loads((collection/'plan.json').read_text())
    plan_hash=sha256(collection/'plan.json')
    reports={}
    skipped={}
    try:
        while len(reports)+len(skipped)<8:
            workers=[json.loads((collection/('worker_'+str(gpu)+'.json')).read_text()) for gpu in (3,4,5)]
            if (sorted(index for worker in workers for index in worker['layouts'])!=list(range(8))
                    or any(worker['plan_sha256']!=plan_hash for worker in workers)):
                raise ValueError('Worker assignments or plan identity differ')
            jobs={job['layout']:job for worker in workers for job in worker['jobs']}
            changed=False
            state['failed_workers']=[dict(gpu=worker['gpu'],jobs=worker['jobs']) for worker in workers if worker['status']=='failed']
            for index in range(8):
                if index in reports or index in skipped:
                    continue
                existing=check_screen(root,index,plan_hash,collection)
                if existing:
                    reports[index]=existing
                    changed=True
                    continue
                job=jobs.get(index,{})
                if job.get('status')=='no_complete_route':
                    skipped[index]='No generated candidate covered the full native route; all raw candidates retained.'
                    changed=True
                    continue
                if job.get('status')!='screened':
                    continue
                output=root/('heldout_layout'+str(index)+'_screen_v1')
                if output.exists():
                    raise ValueError('Incomplete or failed screen requires diagnosis: '+str(output))
                with (a.base/'parc_runtime_v1'/('gpu'+str(a.gpu)+'.lock')).open('a') as lock:
                    try:
                        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        state['status']='waiting_for_screen_gpu_lock'
                        break
                    occupancy=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'],text=True)
                    line=next(line for line in occupancy.splitlines() if int(line.split(',')[0])==a.gpu)
                    _,memory,busy=map(int,line.split(','))
                    if memory>500 or busy>5:
                        state.update(status='waiting_for_idle_screen_gpu',occupancy=occupancy)
                        break
                    command=[sys.executable,'-u','-m','scripts.screen_parc_heldout_collection',
                        '--collection',str(collection),'--mixed',str(a.base/'long_route_mixed_10k_ab_v1/mixed_dataset'),
                        '--layout',str(index),'--output',str(output)]
                    with output.with_suffix('.log').open('x') as stream:
                        child=subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH='.:out/vendor/kimodo',
                                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                        state.update(status='screening',layout=index,child_pid=child.pid,command=command,occupancy=occupancy)
                        save(a.output/'state.json',state)
                        if child.wait()!=0:
                            raise RuntimeError('Exact source screen failed: '+str(output))
                reports[index]=check_screen(root,index,plan_hash,collection)
                if reports[index] is None:
                    raise ValueError('Screen exited without a verified receipt')
                changed=True
            state.update(finished_layouts=sorted(reports),skipped_layouts=skipped,
                status='waiting_for_source_collection' if state.get('status')=='screening' else state['status'])
            save(a.output/'state.json',state)
            if len(reports)+len(skipped)<8:
                time.sleep(1 if changed else 20)
        import torch
        base=root/'evaluation_expanded_sources_v1'
        baseline=json.loads((base/'complete.json').read_text())
        if sha256(base/'heldout-cases.pt')!=baseline['heldout_cases_sha256']:
            raise ValueError('Existing evaluation cases changed')
        cases=torch.load(base/'heldout-cases.pt',map_location='cpu',weights_only=False)
        for case in cases:
            case['evaluation_geometry_identity']=case['row']['source_group']
        screen_receipts=[]
        for index,(folder,report) in sorted(reports.items()):
            added=torch.load(folder/'heldout-cases.pt',map_location='cpu',weights_only=False)
            for case in added:
                if case['native_split']!=plan['jobs'][index]['split'] or not case['current_full_training_excluded']:
                    raise ValueError('New source is not reserved for the planned evaluation split')
                case['evaluation_geometry_identity']='authored:'+plan['jobs'][index]['terrain_sha256']
                case['evaluation_layout_index']=index
                cases.append(case)
            screen_receipts.append(dict(layout=index,report=str(folder/'complete.json'),
                sha256=sha256(folder/'complete.json'),cases_sha256=report['cases_sha256']))
        if len({case['label'] for case in cases})!=len(cases):
            raise ValueError('Evaluation case labels overlap')
        families=defaultdict(set)
        independent=defaultdict(set)
        for case in cases:
            families[case['evaluation_geometry_identity']].add(case['native_split'])
            independent[(case['native_split'],case['entry'])].add(case['evaluation_geometry_identity'])
        if any(len(value)!=1 for value in families.values()):
            raise ValueError('An evaluation geometry crossed validation/test splits')
        inventory=[]
        for index in range(8):
            generated=json.loads((collection/('layout_%02d'%index)/'generation/generation.json').read_text())
            if len(generated['candidates'])!=4 or generated['seed']!=plan['jobs'][index]['seed']:
                raise ValueError('Generated candidate inventory differs')
            inventory.extend(dict(layout=index,candidate_id=row['candidate_id'],frames=row['frames'],
                complete_route=row.get('geometric_route_covered',False),sha256=row['sha256']) for row in generated['candidates'])
        write_json(a.output/'candidate_inventory.json',dict(rows=inventory,failures_preserved=True))
        torch.save(cases,a.output/'heldout-cases.pt')
        complete=dict(baseline,heldout_cases_sha256=sha256(a.output/'heldout-cases.pt'),
            parent_preparation_sha256=sha256(base/'complete.json'),generated_plan_sha256=plan_hash,
            generated_screen_receipts=screen_receipts,skipped_generated_layouts=skipped,
            heldout_selected_counts=dict(Counter(case['native_split']+':'+case['entry'] for case in cases)),
            independent_geometry_counts={':'.join(key):len(value) for key,value in independent.items()},
            target_counts_met=all(len(independent[(split,entry)])>=3 for split in ('validation','test') for entry in ('up','down')),
            target_counts_unit='Distinct authored geometry, not number of candidate passages.',
            current_full_training_excluded=True,quality_approved=False,actor_evaluated=False)
        write_json(a.output/'complete.json',complete)
        state.update(status='complete',selected_cases=len(cases),target_counts_met=complete['target_counts_met'])
    except Exception as error:
        state.update(status='failed',error=repr(error))
        save(a.output/'failure.json',state)
        raise
    finally:
        save(a.output/'state.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=2)
    parser.add_argument('--worker',action='store_true')
    main(parser.parse_args())
