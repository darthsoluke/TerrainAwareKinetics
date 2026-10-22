"""Launch independent coverage and batch-capacity checks beside source pairing."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(a):
    code=Path(__file__).resolve().parents[1]
    root=a.base/'pfnn_mixed_20261015'
    bundle=a.base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'
    old=a.base/'model_terrain_v3_20261011/parc_rugged32_20261013_v2'
    receipt=root/(a.name+'.process.json')
    if receipt.exists():
        raise ValueError('Review already launched; inspect its receipt')
    if a.job=='coverage':
        command=[sys.executable,'-u','-m','scripts.audit_ready_mixed_cache','--cache',str(root/'cache_mixed_v1'),
            '--bundle',str(bundle)]
        occupancy=None
    else:
        occupancy=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True)
        row=next(line for line in occupancy.splitlines() if int(line.split(',')[0])==a.gpu)
        _,memory,busy=map(int,row.split(','))
        if memory>500 or busy>5:
            raise ValueError('Selected GPU is occupied: '+row)
        module={'heldout-screen':'scripts.screen_parc_heldout_collection','evaluation-expanded':'scripts.prepare_full_corpus_evaluation','evaluation-smoke':'scripts.evaluate_full_corpus_controller','evaluation':'scripts.prepare_full_corpus_evaluation','handoff':'scripts.check_full_corpus_handoff','profile':'scripts.profile_full_training_stage',
            'gradients':'scripts.check_short_convolution_gradients'}.get(a.job,'scripts.check_full_training_capacity')
        command=[sys.executable,'-u','-m',module,'--bundle',str(bundle),
            '--initial',str(old/'unified_native_pose_fullbank_20261015_v2/weights.pt'),
            '--fixture',str(root/'mechanics_v4/fixture')]
        if a.job=='handoff':
            command=[sys.executable,'-u','-m',module,'--bundle',str(bundle),
                '--checkpoint',str(root/'mechanics_v4/pose/weights.pt'),
                '--cases',str(old/'unified_pose_data_20261014_v3/cases.pt')]
        if a.job in ('evaluation','evaluation-expanded'):
            command=[sys.executable,'-u','-m',module,'--mixed',str(a.base/'long_route_mixed_10k_ab_v1/mixed_dataset'),
                '--regression-cases',str(old/'unified_pose_data_20261014_v3/cases.pt')]
        if a.job=='evaluation-expanded':
            command+=['--reuse-preparation',str(root/'evaluation_sources_v2'),'--additional-heldout',
                str(old/'dataset_v3/parc_native'),
                str(a.base/'model_terrain_v3_20261011/parc_pose_entry_20261014_v1/dataset_v3/parc_native')]
        if a.job=='evaluation-smoke':
            command=[sys.executable,'-u','-m',module,'--bundle',str(bundle),
                '--checkpoint',str(root/'mechanics_v4/pose/weights.pt'),
                '--cases',str(old/'unified_pose_data_20261014_v3/cases.pt'),
                '--evaluation',str(root/'evaluation_sources_v2'),'--smoke']
        if a.job=='heldout-screen':
            command=[sys.executable,'-u','-m',module,'--collection',str(root/'heldout_collection_v1'),
                '--mixed',str(a.base/'long_route_mixed_10k_ab_v1/mixed_dataset'),'--layout','0']
        if a.job in ('capacity','short-capacity'):
            command+=['--cases',str(old/'unified_pose_data_20261014_v3/cases.pt'),
                '--mixed',str(root/'cache_mixed_v1')]
        if a.job=='short-capacity':
            command+=['--short-conv']
    output=root/a.name
    if output.exists():
        raise ValueError('Output already exists')
    command+=['--output',str(output)]
    env=dict(os.environ,PYTHONPATH='.:out/vendor/kimodo',CUDA_VISIBLE_DEVICES=str(a.gpu) if a.job!='coverage' else '',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (root/(a.name+'.log')).open('x') as stream:
        child=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(pid=child.pid,command=command,gpu=a.gpu if a.job!='coverage' else None,
        occupancy_before=occupancy,actual_optimizer_updates=0,code=str(code))
    receipt.write_text(json.dumps(value,indent=2))
    print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--job',choices=['capacity','coverage','profile','gradients','short-capacity','handoff','evaluation','evaluation-smoke','evaluation-expanded','heldout-screen'],required=True)
    parser.add_argument('--gpu',type=int,default=2)
    main(parser.parse_args())
