"""Bounded four-GPU exposure-policy proofs, separate from healthy production."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_pfnn_bias_workflow import read,save,digest
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(a):
    run=a.base/'pfnn_mixed_20261015';code=Path(__file__).resolve().parents[1]
    output=run/'codec_four_exposure_workflow_v1';output.mkdir(exist_ok=False)
    state=dict(pid=os.getpid(),status='starting',production_budget_updates=0,production_modified=False)
    save(output/'state.json',state)
    reference=run/'codec_four_exposure_reference_v1'
    phases=[('reference',['--serial-only']),
            ('parallel',['--cold-reference',str(reference),'--cold-start-step','0']),
            ('resume',['--cold-reference',str(reference),'--resume-state',str(run/'codec_four_exposure_parallel_v1/pipeline_midpoint.pt')])]
    for label,extra in phases:
        destination=run/('codec_four_exposure_'+label+'_v1')
        if destination.exists():raise ValueError('Refuse to repeat an existing diagnostic phase')
        while True:
            snapshot,available=gpu_snapshot()
            if len(available)>=4:break
            state.update(status='waiting_for_four_idle_gpus',phase=label,gpu_snapshot=snapshot)
            save(output/'state.json',state);time.sleep(15)
        gpus=available[:4]
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                      '--format=csv,noheader'],text=True)
        command=['/usr/bin/timeout','--kill-after=10s','900s',sys.executable,'-u','-m','scripts.check_ordered_codec_pipeline',
            '--gpus',*map(str,gpus),'--checkpoint',str(run/'codec_four_exposure_inputs_v1/checkpoint.pt'),
            '--output',str(destination),'--exposure-mode','--compute-micro','512','--compute-count','4',
            '--reference-memory-fraction','0.9','--pipeline-memory-fraction','0.9','--validation-between',*extra]
        with (output/(label+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,TMPDIR='${PRIVATE_STORAGE}/mbtmp',PYTHONPATH='.:out/vendor/kimodo',
                         OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            state.update(status='running',phase=label,child_pid=child.pid,gpus=gpus,command=command,
                         gpu_snapshot_before=snapshot,compute_processes_before=apps)
            save(output/'state.json',state)
            result=child.wait()
        if result or not (destination/'complete.json').exists():
            raise RuntimeError(label+' diagnostic failed; retain evidence without retry')
        report=read(destination/'complete.json')
        if not report['passed']:raise ValueError(label+' numerical or continuation gate failed')
        save(output/(label+'-verified.json'),dict(path=str(destination/'complete.json'),sha256=digest(destination/'complete.json')))
    state.update(status='proofs_complete_production_integration_required',actual_diagnostic_updates=10,
                 production_approved=False,quality_approved=False)
    save(output/'state.json',state);save(output/'complete.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        save(args.base/'pfnn_mixed_20261015/codec_four_exposure_workflow_v1/failure.json',dict(error=repr(error)))
        raise
