"""Finish one measured long-route batch with parallel existing SOMA screens."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--generation-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--split',choices=('train','validation','test'),required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--wait-physics-pid',type=int)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    runtime=args.base/'parc_runtime_v1';code=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    state={'status':'waiting_for_generation','split':args.split,'gpu':args.gpu,'training_approved':False,
           'pipeline':'official_eval_generator_then_direct_official_tracker_then_unchanged_SOMA_screens',
           'kinematic_optimization_applied':False}
    def save():
        temp=args.output/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(args.output/'status.json')
    def stage(name,command):
        state.update(status=name,updated_at=time.time());save()
        with (args.output/(name+'.log')).open('w') as log:
            subprocess.run(command,env=env,cwd=code,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        save()
        while True:
            path=args.generation_run/'status.json'
            if path.exists():
                generated=json.loads(path.read_text())
                if generated['status']=='kinematic_candidates_generated_physics_pending':break
                if generated['status']=='failed':raise RuntimeError(generated.get('error'))
            time.sleep(10)
        physics_report=args.output/'recording/physics/report.json'
        if args.resume and (args.output/'recording').exists():
            state['status']='waiting_for_preserved_physics';save()
            while True:
                report=json.loads(physics_report.read_text()) if physics_report.exists() else {}
                if report.get('status')=='official_recording_finished':break
                if report.get('status')=='failed':raise RuntimeError(report.get('error'))
                proc=Path('/proc')/str(args.wait_physics_pid)/'cmdline'
                if not args.wait_physics_pid or not proc.exists() or str(args.output/'recording') not in proc.read_bytes().decode():
                    raise RuntimeError('Incomplete physical recording has no matching preserved worker')
                time.sleep(10)
        else:
            stage('physics',[str(runtime/'env/bin/python'),'-u',str(runtime/'wide_v1/scripts/probe_parc_long_route_physics.py'),
                '--runtime',str(runtime),'--generation',str(args.generation_run/'generation'),
                '--output',str(args.output/'recording'),'--gpu',str(args.gpu),'--completed-only'])
        if not (args.output/'recording/export/manifest.json').exists():
            stage('export',[str(runtime/'env/bin/python'),str(runtime/'export_parc_physics_recordings.py'),
                '--bundle',str(args.output/'recording/inputs'),'--run',str(args.output/'recording/physics'),
                '--output',str(args.output/'recording/export')])
        stage('soma',[sys.executable,'-u','-m','scripts.admit_parc_long_route_probe',
            '--recording',str(args.output/'recording'),'--output',str(args.output/'admission'),
            '--workers','8','--split',args.split,'--sdf-gpu',str(args.gpu)])
        summary=json.loads((args.output/'admission/summary.json').read_text())
        state.update(status='long_batch_admission_complete',summary=summary);save()
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error));save();raise


if __name__=='__main__':main()
