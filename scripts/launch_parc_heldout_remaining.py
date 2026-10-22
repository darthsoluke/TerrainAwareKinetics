"""Dispatch remaining reserved expert layouts after exact first-layout screening."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_parc_rugged_collection import digest,write


def main(a):
    code=Path(__file__).resolve().parents[1]
    root=a.base/'pfnn_mixed_20261015'
    collection=root/'heldout_collection_v1'
    receipt=root/'heldout_remaining_v1.process.json'
    if receipt.exists():
        raise ValueError('Remaining-layout dispatch already exists')
    plan=json.loads((collection/'plan.json').read_text())
    first=json.loads((root/'heldout_layout0_screen_v1/complete.json').read_text())
    worker=json.loads((collection/'worker_3.json').read_text())
    if (worker['status']!='collection_complete' or worker['layouts']!=[0]
            or first['first_layout_has_clean_up_and_down'] is not True
            or first['plan_sha256']!=digest(collection/'plan.json')
            or first['catalog_sha256']!=digest(collection/'layout_00/admission/catalog.json')
            or first['cases_sha256']!=digest(root/'heldout_layout0_screen_v1/heldout-cases.pt')
            or not plan['current_full_training_excluded']):
        raise ValueError('Initial collection and exact source screen must match')
    if any((collection/('layout_%02d'%index)).exists() for index in range(1,8)):
        raise ValueError('A remaining layout already exists; inspect before recovery')
    jobs=[]
    environment=dict(os.environ,PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for gpu,layouts in ((4,[1,2,3]),(5,[4,5,6,7])):
        occupancy=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True)
        row=next(line for line in occupancy.splitlines() if int(line.split(',')[0])==gpu)
        _,memory,busy=map(int,row.split(','))
        if memory>256 or busy>5 or (collection/('worker_'+str(gpu)+'.json')).exists():
            raise ValueError('Selected GPU or worker is occupied: '+row)
        command=[sys.executable,'-u','-m','scripts.run_parc_rugged_collection','--base',str(a.base),
            '--output',str(collection),'--gpu',str(gpu),'--layouts',','.join(map(str,layouts)),
            '--generator-script','generate_parc_heldout_stairs.py','--recording-subdir','recording_v1']
        with (root/('heldout_remaining_gpu'+str(gpu)+'.log')).open('x') as stream:
            child=subprocess.Popen(command,cwd=code,env=environment,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append(dict(pid=child.pid,gpu=gpu,layouts=layouts,command=command,occupancy_before=occupancy))
        write(receipt,dict(jobs=jobs,candidates_dispatched=sum(4*len(row['layouts']) for row in jobs),
            first_screen_sha256=digest(root/'heldout_layout0_screen_v1/complete.json'),
            plan_sha256=digest(collection/'plan.json'),training_started=False,current_full_training_excluded=True))
    print(json.dumps(dict(jobs=jobs,candidates_dispatched=28,current_full_training_excluded=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    main(parser.parse_args())
