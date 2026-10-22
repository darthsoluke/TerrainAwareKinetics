"""One-time same-host handover; preserve in-flight official GPU subprocesses."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def command(pid):
    return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--old-coordinator',type=int,required=True)
    p.add_argument('--old-workers',type=int,nargs=2,required=True)
    args=p.parse_args()
    old=args.base/'parc_collection_a100_v2'
    output=args.base/'parc_duration_campaign_v1'
    receipt=output/'handover.json'
    if receipt.exists():
        raise ValueError('Handover already exists; inspect running processes before retrying')
    verified=[]
    for pid,expected in [(args.old_coordinator,'scripts.bootstrap_parc_a100'),
                         *[(pid,'run_parc_diverse_collection.py') for pid in args.old_workers]]:
        value=command(pid)
        if expected not in value or str(args.base) not in value:
            raise ValueError('PID does not belong to the expected task: '+str(pid))
        verified.append((pid,value))
    frozen=[]
    try:
        for pid,_ in verified:
            os.kill(pid,signal.SIGSTOP);frozen.append(pid)
        children=[]
        for pid in args.old_workers:
            result=subprocess.run(['ps','--ppid',str(pid),'-o','pid='],capture_output=True,text=True)
            for token in result.stdout.split():
                child=int(token); value=command(child)
                if str(args.base) not in value:
                    raise ValueError('Unexpected subprocess under task worker')
                children.append({'pid':child,'command':value})
        scenes={path.parent.name for path in old.glob('*/receipt.json')}
        for role in (0,1):
            state=json.loads((old/f'worker_{role}.json').read_text())
            if state.get('active_scene'):
                scenes.add(state['active_scene'])
        payload={'preserved_children':children,'legacy_scenes':sorted(scenes),
                 'stopped_parents':[{'pid':pid,'command':cmd} for pid,cmd in verified],
                 'time':time.time(),'policy':'finish in-flight stages; reuse immutable receipts and generation'}
        receipt.write_text(json.dumps(payload,indent=2))
        for pid,_ in verified:
            os.kill(pid,signal.SIGTERM)
            os.kill(pid,signal.SIGCONT)
            frozen.remove(pid)
    finally:
        for pid in frozen:
            os.kill(pid,signal.SIGCONT)
    code=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    commands=[
        ('campaign',[sys.executable,'-u','-m','scripts.run_parc_duration_campaign','--base',str(args.base),
          '--output',str(output),'--gpus','2,3','--adopted-children',str(receipt),
          '--legacy-collection',str(old),'--legacy-scenes',*sorted(scenes)]),
        ('training',[sys.executable,'-u','-m','scripts.run_diverse_mixed_handoff','--base',str(args.base),
          '--source',str(args.base/'mixed_velocity_50k_v1/pipeline.pt'),
          '--output',str(args.base/'diverse_mixed_a100_v4'),
          '--drive-config',str(args.base/'drive-auth/rclone.conf'),
          '--local-dataset-receipt',str(output/'dataset_ready.json'),'--steps','10000','--gpus','2,3'])]
    for name,cmd in commands:
        with (output/(name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        payload[name+'_pid']=child.pid
    receipt.write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
