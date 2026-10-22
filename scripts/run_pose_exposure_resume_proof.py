"""Use idle GPUs for the bounded Pose warm-reference/cold-resume diagnostic."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_pfnn_bias_workflow import save,read,digest
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(a):
    run=a.base/'pfnn_mixed_20261015';code=Path(__file__).resolve().parents[1]
    output=run/'pose_exposure_resume_workflow_v1';output.mkdir(exist_ok=False)
    state=dict(pid=os.getpid(),production_budget_updates=0,production_modified=False)
    for phase in ('reference','resume'):
        destination=run/('pose_exposure_'+phase+'_v1')
        if destination.exists():raise ValueError('Refuse to repeat an existing phase')
        while True:
            snapshot,available=gpu_snapshot()
            if available:break
            state.update(status='waiting_for_idle_gpu',phase=phase,gpu_snapshot=snapshot)
            save(output/'state.json',state);time.sleep(15)
        gpu=available[0]
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
        command=['/usr/bin/timeout','--kill-after=10s','900s',sys.executable,'-u','-m','scripts.check_pose_exposure_resume',
            '--gpu',str(gpu),'--checkpoint',str(run/'codec_pipeline_inputs_v1/checkpoint.pt'),
            '--output',str(destination),'--batch','3072']
        if phase=='resume':command+=['--reference',str(run/'pose_exposure_reference_v1')]
        with (output/(phase+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,TMPDIR='${PRIVATE_STORAGE}/mbtmp',PYTHONPATH='.:out/vendor/kimodo',
                         OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            state.update(status='running',phase=phase,child_pid=child.pid,gpu=gpu,command=command,
                gpu_snapshot_before=snapshot,compute_processes_before=apps);save(output/'state.json',state)
            result=child.wait()
        if result or not (destination/'complete.json').exists():raise RuntimeError(phase+' failed; inspect retained evidence')
        if not read(destination/'complete.json')['passed']:raise ValueError(phase+' numerical gate failed')
        save(output/(phase+'-verified.json'),dict(path=str(destination/'complete.json'),sha256=digest(destination/'complete.json')))
    state.update(status='complete',actual_diagnostic_updates=6,production_approved=False,quality_approved=False)
    save(output/'state.json',state);save(output/'complete.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        save(args.base/'pfnn_mixed_20261015/pose_exposure_resume_workflow_v1/failure.json',dict(error=repr(error)))
        raise
