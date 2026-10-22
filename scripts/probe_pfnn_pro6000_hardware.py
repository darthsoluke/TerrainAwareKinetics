"""Bounded GRU/autograd and torch.distributed check on two idle PRO6000s."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def worker(output):
    import torch
    import torch.distributed as dist
    rank = int(os.environ['LOCAL_RANK'])
    report = dict(rank=rank, pid=os.getpid(), stage='initializing', optimizer_updates=0, training_exposures=0)
    path = output / f'rank{rank}.json'
    save(path, report)
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2333)
    cell = torch.nn.GRUCell(388, 256)
    x, h = torch.randn(16, 388), torch.randn(16, 256)
    y = cell(x, h)
    y.square().mean().backward()
    reference = {name: p.grad.clone() for name, p in cell.named_parameters()}
    import copy
    gpu = copy.deepcopy(cell).to('cuda:' + str(rank))
    gpu.zero_grad(set_to_none=True)
    actual = gpu(x.to(rank), h.to(rank))
    actual.square().mean().backward()
    error = float((actual.cpu() - y.detach()).abs().max())
    grad_error = max(float((p.grad.cpu() - reference[name]).abs().max()) for name, p in gpu.named_parameters())
    assert error < 3e-5 and grad_error < 3e-5
    report.update(stage='gru_passed_initializing_nccl', forward_error=error, gradient_error=grad_error,
                  torch=str(torch.__version__), cuda=torch.version.cuda, device=torch.cuda.get_device_name(rank))
    save(path, report)
    dist.init_process_group('nccl', timeout=timedelta(seconds=30), device_id=torch.device('cuda', rank))
    value = torch.full((16,), float(rank + 1), device='cuda:' + str(rank))
    dist.all_reduce(value)
    assert torch.equal(value.cpu(), torch.full((16,), 3.))
    dist.destroy_process_group()
    report.update(stage='complete', passed=True, nccl=list(torch.cuda.nccl.version()))
    save(path, report)


def main(args):
    if args.worker:
        worker(args.output)
        return
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu,display_active',
                                      '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader,nounits'], text=True)
    occupied = {line.split(',')[1].strip() for line in apps.splitlines() if line.strip()}
    idle = []
    for row in devices.splitlines():
        index, uuid, name, memory, use, display = (part.strip() for part in row.split(','))
        if ('RTX PRO 6000' in name and uuid not in occupied and int(memory) <= 32
                and int(use) <= 1 and display == 'Disabled'):
            idle.append((int(index), uuid))
    if len(idle) < 2:
        raise RuntimeError('Two idle PRO6000s required')
    args.output.mkdir(parents=True, exist_ok=False)
    selected = sorted(idle)[:2]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(uuid for _, uuid in selected),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', NCCL_DEBUG='INFO')
    started = time.monotonic()
    save(args.output / 'launch.json', dict(selected=selected, devices=devices, compute_apps=apps))
    command = ['timeout', '--kill-after=10', '120', sys.executable, '-m', 'torch.distributed.run',
               '--standalone', '--nproc_per_node=2', str(Path(__file__).resolve()), '--worker', '--output', str(args.output)]
    with (args.output / 'worker.log').open('x') as log:
        result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    rows = [json.loads((args.output / f'rank{i}.json').read_text()) for i in range(2)]
    report = dict(passed=result.returncode == 0 and all(row.get('passed') for row in rows),
                  returncode=result.returncode, rows=rows, elapsed_seconds=time.monotonic() - started,
                  optimizer_updates=0, training_exposures=0)
    save(args.output / 'complete.json', report)
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise RuntimeError('PRO6000 hardware probe failed; see preserved worker log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--worker', action='store_true')
    main(parser.parse_args())
