"""Select explicitly allowed idle GPUs and start one bounded job."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


GPU_FAMILIES = {'A100': re.compile(r'\bA100\b'),
                'RTX PRO 6000': re.compile(r'\bRTX PRO 6000\b')}


def main(args, extra):
    if args.gpu_family not in GPU_FAMILIES:
        raise ValueError('Unsupported GPU family: ' + args.gpu_family)
    if args.gpus is not None and (not 1 <= len(args.gpus) <= 2
            or len(set(args.gpus)) != len(args.gpus) or any(i < 0 for i in args.gpus)):
        raise ValueError('Select one or two distinct nonnegative GPU indices')
    output = args.output.resolve()
    receipt = output.with_suffix('.launch.json')
    logfile = output.with_suffix('.log')
    if output.exists() or receipt.exists() or logfile.exists():
        raise FileExistsError('This run already has an output, launch receipt or log')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu,display_active',
                                      '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    occupied = set()
    for row in csv.reader(apps.splitlines(), skipinitialspace=True):
        if not row:
            continue
        pid, uuid, _, _ = (part.strip() for part in row)
        if not pid.isdecimal() or not uuid.startswith('GPU-'):
            raise ValueError('Cannot certify compute-process occupancy')
        occupied.add(uuid)
    idle = []
    for row in csv.reader(devices.splitlines(), skipinitialspace=True):
        if not row:
            continue
        index, uuid, name, memory, usage, display = (part.strip() for part in row)
        if (not GPU_FAMILIES[args.gpu_family].search(name) or uuid in occupied
                or display != 'Disabled'):
            continue
        if not memory.isdecimal() or not usage.isdecimal() or not uuid.startswith('GPU-'):
            continue
        if int(memory) <= 32 and int(usage) <= 1:
            idle.append((int(index), uuid))
    if args.gpus is None:
        if len(idle) < 2:
            raise RuntimeError('Two idle ' + args.gpu_family + ' GPUs with no active display or compute PID are required')
        selected = sorted(idle)[:2]
    else:
        by_index = dict(idle)
        if any(i not in by_index for i in args.gpus):
            raise RuntimeError('An explicitly requested GPU is occupied or does not match the required family')
        selected = [(i, by_index[i]) for i in args.gpus]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(row[1] for row in selected),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', PYTHONPATH='.:out/vendor/kimodo')
    timeout = 28800 if args.mode == 'execute' else 1800
    command = ['timeout', '--signal=TERM', '--kill-after=30', str(timeout), sys.executable,
               '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=' + str(len(selected)),
               '-m', 'scripts.train_pfnn_frame_actor', '--mode', args.mode, '--output', str(output), *extra]
    output.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as stream:
        with logfile.open('x') as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        result = dict(pid=process.pid, mode=args.mode, requested_gpu_family=args.gpu_family,
                      requested_gpu_indices=args.gpus, selected=selected, all_devices=devices,
                      all_compute_apps=apps, command=command, wall_timeout_seconds=timeout,
                      launch_unix_time=time.time(), source_root=str(Path.cwd()))
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'execute'), default='preflight')
    parser.add_argument('--gpu-family', choices=tuple(GPU_FAMILIES), default='A100')
    parser.add_argument('--gpus', nargs='+', type=int, help='Use only these physical GPU indices')
    return parser.parse_known_args(argv)


if __name__ == '__main__':
    main(*parse_args())
