"""Resume a stopped readiness-check failure, preserving its generated candidates."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_parc_rugged_collection import digest,write


def main(a):
    root=a.base/'pfnn_mixed_20261015'
    collection=root/'heldout_collection_v1'
    state=json.loads((collection/('worker_'+str(a.gpu)+'.json')).read_text())
    if state['status']!='failed' or state['gpu']!=a.gpu:
        raise ValueError('Recovery only applies to a failed matching worker')
    row=next(job for job in state['jobs'] if job['status']=='failed')
    log=Path(row['log']).read_text()
    if 'Probe GPU is occupied' not in log:
        raise ValueError('This recovery only addresses the diagnosed GPU-readiness check')
    for pid in (state['pid'],row['pid']):
        proc=Path('/proc')/str(pid)/'cmdline'
        if proc.exists() and str(collection).encode() in proc.read_bytes():
            raise ValueError('A prior worker or child is still active')
    plan=json.loads((collection/'plan.json').read_text())
    if state['plan_sha256']!=digest(collection/'plan.json'):
        raise ValueError('Collection plan identity changed')
    for job in state['jobs']:
        generated=json.loads((collection/('layout_%02d'%job['layout'])/'generation/generation.json').read_text())
        if generated['seed']!=plan['jobs'][job['layout']]['seed'] or len(generated['candidates'])!=4:
            raise ValueError('Saved generation does not match the declared plan')
    occupancy=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    line=next(line for line in occupancy.splitlines() if int(line.split(',')[0])==a.gpu)
    _,memory,busy=map(int,line.split(','))
    if memory>256 or busy>5:
        raise ValueError('Recovery GPU is currently occupied: '+line)
    receipt=root/('heldout_recovery_gpu'+str(a.gpu)+'_v1.process.json')
    if receipt.exists():
        raise ValueError('Recovery already dispatched')
    code=Path(__file__).resolve().parents[1]
    command=[sys.executable,'-u','-m','scripts.run_parc_rugged_collection','--base',str(a.base),
        '--output',str(collection),'--gpu',str(a.gpu),'--layouts',','.join(map(str,state['layouts'])),
        '--generator-script','generate_parc_heldout_stairs.py','--recording-subdir','recording_v2','--resume']
    with receipt.with_suffix('.log').open('x') as stream:
        child=subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=stream,
            stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
    result=dict(pid=child.pid,command=command,code=str(code),gpu=a.gpu,occupancy_before=occupancy,
        reused_generation=True,prior_failed_worker=state['pid'],old_physics_outputs_preserved=True,
        training_started=False,current_full_training_excluded=True)
    write(receipt,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--gpu',type=int,choices=(4,5),required=True)
    main(parser.parse_args())
