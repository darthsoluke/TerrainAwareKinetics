"""Launch only on freshly checked idle GPUs, with a reviewable process receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpus',required=True);parser.add_argument('--name',required=True)
parser.add_argument('--code',type=Path,required=True)
parser.add_argument('command',nargs=argparse.REMAINDER)
args=parser.parse_args()
task=args.code.parent;receipt=task/(args.name+'-process.json')
if receipt.exists():
    old=json.loads(receipt.read_text());proc=Path('/proc')/str(old['pid'])/'cmdline'
    if proc.exists() and args.name.encode() in proc.read_bytes():raise SystemExit('Matching run already active')
status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
available={int(row.split(',')[0]):tuple(int(x.strip()) for x in row.split(',')[1:]) for row in status.splitlines()}
for gpu in map(int,args.gpus.split(',')):
    if available[gpu][0]>256 or available[gpu][1]>5:raise SystemExit(f'GPU {gpu} is occupied: {available[gpu]}')
environment=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus,PYTHONPATH='.:out/vendor/kimodo',OMP_NUM_THREADS='4')
command=args.command[1:] if args.command[:1]==['--'] else args.command
log=task/(args.name+'.log')
with log.open('xb') as stream:
    process=subprocess.Popen(command,cwd=args.code,env=environment,stdin=subprocess.DEVNULL,
        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
value={'pid':process.pid,'command':command,'gpus':args.gpus,'occupancy_before_launch':status,'log':str(log)}
receipt.write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
