"""Replace only the coordinator, preserving task-owned live GPU workers."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.parc_collection_scheduler import process_identity, process_matches, select_jobs
from scripts.train_native_mb import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--coordinator',type=int,required=True)
    parser.add_argument('--gpus',default='1,2,3,4,5,6')
    parser.add_argument('--stop-campaign',action='store_true')
    args=parser.parse_args()
    allowed={int(value) for value in args.gpus.split(',')}
    if not allowed or any(value not in range(8) for value in allowed):
        raise ValueError('Invalid allowed GPU pool')
    if args.stop_campaign:allowed=set()
    old=process_identity(args.coordinator)
    if old is None or 'scripts.run_parc_duration_campaign' not in old['command'] or str(args.campaign) not in old['command']:
        raise ValueError('Coordinator identity does not match this campaign')
    if (args.campaign/'dataset_ready.json').exists():
        raise ValueError('Dataset already ready; do not restart generation')
    os.kill(args.coordinator,signal.SIGSTOP)
    stopped=True
    try:
        state=json.loads((args.campaign/'state.json').read_text())
        collection=Path(state['collection'])
        candidates=state.get('active_workers')
        if candidates is None:
            plan=json.loads((collection/'plan/plan.json').read_text())
            candidates=[{'pid':pid,'card':card,'collection':str(collection),
                         'state_file':f'worker_{role}.json',
                         'job_ids':[job['id'] for job in select_jobs(plan,role)]}
                        for role,(card,pid) in enumerate(zip(state['physical_gpus'],state['worker_pids']))]
        workers=[]; released=[]
        for worker in candidates:
            identity=process_identity(worker['pid'])
            if identity is None or identity['state']=='Z':
                continue
            if ('run_parc_diverse_collection.py' not in identity['command'] or
                    str(args.base) not in identity['command'] or worker['collection'] not in identity['command']):
                raise ValueError('Unexpected GPU worker; do not adopt or signal it')
            bound={**worker,'identity':identity,
                   'plan_sha256':sha256(Path(worker['collection'])/'plan/plan.json')}
            if worker['card'] in allowed:
                workers.append(bound)
            else:
                if os.getpgid(worker['pid']) != worker['pid']:
                    raise ValueError('Cannot release a worker without its own process group')
                released.append(bound)
        # A running CPU admission may finish while the parent is stopped. Preserve
        # its report; the new owner will reconcile or retry it from immutable inputs.
        child_pids=subprocess.run(['ps','--ppid',str(args.coordinator),'-o','pid='],
                                  text=True,capture_output=True).stdout.split()
        cpu=[]
        adopted={worker['pid'] for worker in workers+released}
        for token in child_pids:
            pid=int(token)
            identity=process_identity(pid)
            if pid in adopted or identity is None or identity['state']=='Z':
                continue
            if 'scripts.finish_parc_diverse_dataset' not in identity['command']:
                raise ValueError('Unexpected live coordinator child')
            cpu.append(identity)
        while any(process_matches(identity) for identity in cpu):
            time.sleep(2)
        handover=args.campaign/('elastic_handover_'+str(time.time_ns())+'.json')
        payload={'coordinator_before':old,'workers':workers,'released_workers':released,'cpu_admissions_preserved':cpu,
                 'gpus':args.gpus,'created_at':time.time(),'generation_and_optimization_unchanged':True}
        handover.write_text(json.dumps(payload,indent=2))
        for worker in released:
            os.killpg(worker['pid'],signal.SIGTERM)
        while any(process_matches(worker['identity']) for worker in released):
            time.sleep(.1)
        os.kill(args.coordinator,signal.SIGTERM)
        os.kill(args.coordinator,signal.SIGCONT)
        stopped=False
        while process_matches(old):
            time.sleep(.1)
        if args.stop_campaign:
            state.update(status='superseded_by_continuous_long_routes',active_workers=[],worker_pids=[],
                         handover=str(handover),updated_at=time.time())
            temporary=args.campaign/'state.tmp';temporary.write_text(json.dumps(state,indent=2));temporary.replace(args.campaign/'state.json')
            print(json.dumps({'stopped_coordinator':args.coordinator,'released_cards':[w['card'] for w in released]}))
            return
        code=Path(__file__).resolve().parents[1]
        env=dict(os.environ,PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),
                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        command=[sys.executable,'-u','-m','scripts.run_parc_duration_campaign',
                 '--base',str(args.base),'--output',str(args.campaign),'--gpus',args.gpus,
                 '--elastic','--adopt-workers',str(handover)]
        log_path=handover.with_suffix('.log')
        with log_path.open('x') as log:
            child=subprocess.Popen(command,cwd=code,env=env,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        payload.update(coordinator_after=child.pid,log=str(log_path))
        handover.write_text(json.dumps(payload,indent=2))
        print(json.dumps({'coordinator':child.pid,'preserved_workers':[w['pid'] for w in workers],
                          'released_cards':[w['card'] for w in released],
                          'handover':str(handover),'log':str(log_path),'gpus':args.gpus}))
    finally:
        if stopped and process_matches(old):
            os.kill(args.coordinator,signal.SIGCONT)


if __name__=='__main__':main()
