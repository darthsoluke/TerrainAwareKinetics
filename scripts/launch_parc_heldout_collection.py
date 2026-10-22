"""Prepare reserved geometry and launch only its first four expert candidates."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import build_parc_stair_course as course_base
from build_parc_heldout_stair_course import build_heldout_course
from run_parc_rugged_collection import digest,write


def main(a):
    if a.gpu in (0,1,7) or a.gpu not in range(8):
        raise ValueError('Preserve trainingGPU0, other-userGPU1 and pairingGPU7')
    if a.output.exists() or a.output.with_suffix('.process.json').exists():
        raise ValueError('Collection already exists; inspect before any recovery')
    code=Path(__file__).resolve().parents[1]
    course_base.PARC_REPO=a.base/'parc_runtime_v1/wide_v1/out/vendor/PARC'
    jobs=[]
    for index in range(8):
        _,nodes,course=build_heldout_course(index)
        jobs.append(dict(layout=index,family='stairs',split=course['split'],seed=2026101550+index,
            terrain_sha256=course['terrain_sha256'],route_nodes=len(nodes),course=course,candidates=4))
    identities={row['terrain_sha256'] for row in jobs}
    if len(identities)!=8:
        raise ValueError('Duplicate held-out geometry')
    known=set()
    for name in ('parc_pose_entry_20261014_v1','parc_rugged32_20261013_v2'):
        plan=a.base/'model_terrain_v3_20261011'/name/'plan.json'
        if plan.exists():
            known.update(row['terrain_sha256'] for row in json.loads(plan.read_text())['jobs'])
    if identities&known:
        raise ValueError('New geometry matches an existing generation layout')
    protected=a.base/'model_terrain_v3_20261011/parc_rugged32_20261013_v2/unified_native_pose_fullbank_20261015_v2/weights.pt'
    protected_hash='03580db6f19c407b605fb43a6dd23514f8d5f17c5ad7b869c7ec95059ca68ae1'
    if digest(protected)!=protected_hash:
        raise ValueError('Preserved controller identity differs')
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'plan.json',dict(schema='motionbricks.parc_heldout_stairs.v1',jobs=jobs,
        total_candidates=32,initial_dispatch_candidates=4,approved_model=str(protected),
        approved_model_sha256=protected_hash,training_started=False,current_full_training_excluded=True,
        split_unit='Whole authored geometry; all candidates and retries retain validation/test split.',
        scope='Only layout0dispatched. Inspect native prefix, physical recording and exact SOMA source geometry before expansion.',
        admission_is_not_controller_acceptance=True))
    occupancy=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    row=next(line for line in occupancy.splitlines() if int(line.split(',')[0])==a.gpu)
    _,memory,busy=map(int,row.split(','))
    if memory>256 or busy>5:
        raise ValueError('Selected GPU is occupied; plan prepared, no generation launched: '+row)
    command=[sys.executable,'-u','-m','scripts.run_parc_rugged_collection','--base',str(a.base),
        '--output',str(a.output),'--gpu',str(a.gpu),'--layouts','0',
        '--generator-script','generate_parc_heldout_stairs.py','--recording-subdir','recording_v1']
    env=dict(os.environ,PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with a.output.with_suffix('.log').open('x') as stream:
        child=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=child.pid,command=command,code=str(code),gpu=a.gpu,occupancy_before=occupancy,
        candidates_dispatched=4,training_started=False,current_full_training_excluded=True)
    write(a.output.with_suffix('.process.json'),receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    main(parser.parse_args())
