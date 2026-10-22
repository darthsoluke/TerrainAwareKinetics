"""Dispatch an explicitly bounded world/contact repair pilot or its full rebuild."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(a):
    code=Path(__file__).resolve().parents[1]
    root=a.base/'pfnn_mixed_20261015'
    output=root/a.name
    receipt=output.with_suffix('.process.json')
    if output.exists() or receipt.exists():
        raise ValueError('Repair job already exists')
    if a.job=='audit':
        command=[sys.executable,'-u','-m','scripts.audit_pfnn_world_repair',
            '--repaired',str(root/'world_repair_pilot_v1'),'--previous',str(root/'dataset_full_v1'),
            '--retarget',str(root/'retarget_full_v2')]
        occupancy=None
    else:
        occupancy=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True)
        row=next(line for line in occupancy.splitlines() if int(line.split(',')[0])==a.gpu)
        _,memory,busy=map(int,row.split(','))
        if memory>500 or busy>5:
            raise ValueError('Repair GPU occupied: '+row)
        if a.job in ('preflight','mechanics'):
            old=a.base/'model_terrain_v3_20261011/parc_rugged32_20261013_v2'
            module='scripts.check_unified_full_training' if a.job=='preflight' else 'scripts.check_full_training_resume'
            command=[sys.executable,'-u','-m',module,'--bundle',str(a.base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'),
                '--initial',str(old/'unified_native_pose_fullbank_20261015_v2/weights.pt'),
                '--cases',str(old/'unified_pose_data_20261014_v3/cases.pt'),'--pfnn',str(root/'world_repair_pilot_v1')]
            if a.job=='mechanics':
                command+=['--preflight',str(root/'preflight_world_v1/complete.json'),
                    '--stats',str(a.base/'long_route_mixed_10k_ab_v1/mixed_dataset/stats')]
        else:
            command=[sys.executable,'-u','-m','scripts.build_pfnn_terrain_dataset','--source',str(root/'source/unpacked'),
                '--retarget',str(root/'retarget_full_v2'),'--reuse-geometry',str(root/'dataset_full_v1')]
        if a.job=='pilot':
            command+=['--clips','LocomotionFlat01_000','NewCaptures03_000','NewCaptures07_000_mirror',
                '--max-episodes','2','--pilot']
        elif a.job=='full':
            audit=json.loads((root/'world_repair_audit_v1/complete.json').read_text())
            if not audit['passed']:
                raise ValueError('Actual repair pilot must pass before the full rebuild')
    command+=['--output',str(output)]
    env=dict(os.environ,PYTHONPATH='.:out/vendor/kimodo',CUDA_VISIBLE_DEVICES='' if a.job=='audit' else str(a.gpu),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with output.with_suffix('.log').open('x') as stream:
        child=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(pid=child.pid,command=command,code=str(code),gpu=None if a.job=='audit' else a.gpu,
        occupancy_before=occupancy,actual_optimizer_updates=0)
    receipt.write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--job',choices=('pilot','audit','full','preflight','mechanics'),required=True)
    parser.add_argument('--gpu',type=int,default=6)
    main(parser.parse_args())
