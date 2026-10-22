"""Receive an existing PARC runtime via Drive and generate on A100 GPU0/1 only."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from scripts.train_native_mb import sha256
from scripts.train_mixed_control_stage import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--drive-config',type=Path,required=True)
    parser.add_argument('--resume-waiting',action='store_true')
    parser.add_argument('--resume-installed',action='store_true')
    parser.add_argument('--gpus',default='0,1')
    args=parser.parse_args()
    output=args.base/'parc_collection_a100_v2'
    cards=[int(value) for value in args.gpus.split(',')]
    if len(cards)!=2 or len(set(cards))!=2 or any(card not in range(8) for card in cards):
        parser.error('Exactly two physical A100 indices are required')
    previous=json.loads((output/'deployment.json').read_text()) if output.exists() else {}
    if output.exists() and not ((args.resume_waiting and previous['status']=='waiting_for_private_runtime')
            or (args.resume_installed and not any('worker_' in key for key in previous))):
        raise ValueError('Only the idle runtime-receipt wait may be resumed')
    output.mkdir(exist_ok=args.resume_waiting or args.resume_installed)
    state={'status':'waiting_for_private_runtime','gpus':cards,'training_approved':False}
    state_path=output/'deployment.json'; write_json(state_path,state)
    remote='mb-transfer:environments/parc-20261010'
    try:
        while True:
            local_receipt=args.base/'packages/parc_runtime_receipt.json'
            if local_receipt.exists():
                receipt=json.loads(local_receipt.read_text());break
            try:
                result=subprocess.run(['rclone','--config',str(args.drive_config),'cat',remote+'/runtime.json',
                    '--retries','1','--low-level-retries','1','--timeout','20s','--contimeout','10s'],
                    capture_output=True,text=True,timeout=60)
                if result.returncode==0:
                    receipt=json.loads(result.stdout);break
            except subprocess.TimeoutExpired: pass
            time.sleep(30)
        runtime=args.base/'parc_runtime_v1'; runtime.mkdir(exist_ok=args.resume_installed)
        for item in receipt['files']:
            archive=runtime/item['name']
            if archive.parent!=runtime: raise ValueError('Archive escaped runtime directory')
            if args.resume_installed:
                if sha256(archive)!=item['sha256']:raise ValueError('Installed runtime archive changed')
                continue
            state.update(status='downloading_runtime',file=item['name']);write_json(state_path,state)
            while True:
                result=subprocess.run(['rclone','--config',str(args.drive_config),'copyto',remote+'/'+item['name'],str(archive),
                    '--immutable','--retries','1','--tpslimit','1'])
                if result.returncode==0:break
                state.update(status='retrying_private_drive',file=item['name'],last_exit_code=result.returncode)
                write_json(state_path,state);time.sleep(30)
            if sha256(archive)!=item['sha256'] or archive.stat().st_size!=item['bytes']:
                raise ValueError('Runtime archive identity mismatch')
            destination=runtime/'env' if item['name']=='parc-runtime-env.tar.gz' else runtime
            destination.mkdir(exist_ok=True)
            with tarfile.open(archive) as stream:stream.extractall(destination,filter='data')
        python=runtime/'env/bin/python'
        env=dict(os.environ,PATH=str(runtime/'env/bin')+':'+os.environ.get('PATH',''),
            LD_LIBRARY_PATH=str(runtime/'env/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
            PYTHONPATH=str(runtime/'sdk/isaacgym/python'),CUDA_VISIBLE_DEVICES=str(cards[0]),OMP_NUM_THREADS='4')
        if not args.resume_installed:
            subprocess.run([str(python),str(runtime/'env/bin/conda-unpack')],env=env,check=True)
        subprocess.run([str(python),'-c',
            'from isaacgym import gymapi; import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'],env=env,check=True)
        work=runtime/'wide_v1'
        command=[str(python),'-u',str(work/'scripts/run_parc_diverse_collection.py'),
                 '--work',str(work),'--output',str(output),'--asset-bundle',str(runtime/'inputs_v1')]
        if not (output/'plan/plan.json').exists():
            subprocess.run(command+['--prepare'],env=env,check=True)
        from scripts.run_mixed_control_workflow import available_devices
        while True:
            try:available_devices(cards);break
            except RuntimeError:time.sleep(30)
        children=[]
        for gpu in (0,1):
            log=(output/f'worker_{gpu}.log').open('x')
            child=subprocess.Popen(command+['--gpu',str(gpu),'--device-index',str(cards[gpu])],cwd=work,stdout=log,stderr=subprocess.STDOUT,
                                   env={key:value for key,value in env.items() if key!='CUDA_VISIBLE_DEVICES'})
            children.append((child,log));state[f'worker_{gpu}_pid']=child.pid
        state.update(status='generating_correcting_recording');write_json(state_path,state)
        while any(child.poll() is None for child,_ in children):
            state.update(updated_at=time.time(),finished_scenes=sum(1 for _ in output.glob('*/receipt.json')))
            write_json(state_path,state);time.sleep(30)
        for gpu,(child,log) in enumerate(children):
            log.close()
            if child.returncode:raise RuntimeError(f'PARC worker{gpu} failed: {child.returncode}')
            write_json(output/f'recovery_{gpu}.json',{'completed':True,'runtime':'relocated existing official environment'})
        state['status']='collection_ready_for_local_soma_admission';write_json(state_path,state)
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error));write_json(state_path,state);raise


if __name__=='__main__':main()
