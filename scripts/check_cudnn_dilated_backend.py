"""Identify and compare deterministic CUDA paths for measured Conv1d shapes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main(a):
    owned_pause = a.paused_owner_pid is not None
    while True:
        snapshot = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        measured = {int(row.split(',')[0]): tuple(map(int, row.split(',')[1:]))
            for row in snapshot.splitlines()}
        memory, busy = measured[a.gpu]
        if owned_pause:
            owner = Path('/proc')/str(a.paused_owner_pid)
            command = (owner/'cmdline').read_bytes().replace(bytes([0]), b' ').decode()
            state = next(line for line in (owner/'status').read_text().splitlines() if line.startswith('State:'))
            if 'train_unified_full_corpus' not in command or '--stage root' not in command or 'full_training_v3/root' not in command:
                raise RuntimeError('Paused owner is not the authorized production Root')
            if state.split()[1] != 'T':
                raise RuntimeError('Owned Root is not stopped')
            gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
            applications = subprocess.check_output(['nvidia-smi',
                '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
            pids = {int(row.split(',')[0]) for row in applications.splitlines()
                if row.split(',')[1].strip() == gpu_uuid}
            free = int(subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if pids != {a.paused_owner_pid} or free < 6144:
                raise RuntimeError('Exclusive owned GPU and 6GiB free are required')
            break
        if memory <= 500 and busy <= 5:
            break
        save(a.output/'state.json', dict(status='waiting_for_idle_gpu', gpu=a.gpu,
            gpu_snapshot=snapshot, pid=os.getpid(), actual_optimizer_updates=0))
        time.sleep(30)
    save(a.output/'state.json', dict(status='running', gpu=a.gpu,
        gpu_snapshot_before=snapshot, actual_optimizer_updates=0))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(4)
    if owned_pause:
        torch.cuda.set_per_process_memory_fraction(.05)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    class BatchedBiasNativeBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, weight, bias, dilation):
            ctx.save_for_backward(value, weight)
            ctx.bias_size, ctx.dilation = list(bias.shape), dilation
            columns = F.unfold(value.unsqueeze(2), (1, 3),
                dilation=(1, dilation), padding=(0, dilation))
            output = torch.empty((value.shape[0], weight.shape[0], value.shape[-1]),
                device=value.device, dtype=value.dtype)
            output.copy_(bias[None, :, None])
            flattened = weight.reshape(weight.shape[0], -1)
            for index in range(value.shape[0]):
                output[index].addmm_(flattened, columns[index])
            return output

        @staticmethod
        def backward(ctx, gradient):
            value, weight = ctx.saved_tensors
            result = torch.ops.aten.convolution_backward.default(gradient, value, weight,
                ctx.bias_size, [1], [ctx.dilation], [ctx.dilation], False, [0], 1,
                list(ctx.needs_input_grad[:3]))
            return *result, None

    reports = []
    for length, dilation in ((8, 27), (16, 9), (32, 3), (32, 27)):
        layer = torch.nn.Conv1d(512, 512, 3, padding=dilation, dilation=dilation).cuda()
        value = torch.randn(64, 512, length, device='cuda', requires_grad=True)
        gradient = torch.randn_like(value)
        backend = str(torch._C._select_conv_backend(value, layer.weight, layer.bias,
            [1], [dilation], [dilation], False, [0], 1))
        reference = None
        for variant in ('native', 'batched_bias_native_backward'):
            times, previous, repeat_exact = [], None, True
            for repeat in range(3):
                torch.cuda.synchronize()
                started = time.perf_counter()
                if variant == 'native':
                    output = F.conv1d(value, layer.weight, layer.bias,
                        padding=dilation, dilation=dilation)
                else:
                    output = BatchedBiasNativeBackward.apply(value, layer.weight, layer.bias, dilation)
                gradients = torch.autograd.grad(output, (value, layer.weight, layer.bias), gradient)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-started)
                observed = [output.detach().cpu().clone(), *[x.detach().cpu().clone() for x in gradients]]
                if previous is not None:
                    repeat_exact = repeat_exact and all(torch.equal(x, y) for x, y in zip(previous, observed))
                previous = observed
                del output, gradients
            if reference is None:
                reference = observed
            checks = []
            for name, actual, expected in zip(('output', 'input_gradient', 'weight_gradient', 'bias_gradient'),
                    observed, reference):
                atol, rtol = (1e-6, 1e-5) if name == 'output' else (5e-6, 1e-3)
                checks.append(dict(name=name, max_abs=float((actual-expected).abs().max()),
                    bitwise_equal=torch.equal(actual, expected),
                    passed=torch.allclose(actual, expected, atol=atol, rtol=rtol), atol=atol, rtol=rtol))
            reports.append(dict(length=length, dilation=dilation, variant=variant,
                selected_native_backend=backend, seconds=times, repeated_bitwise_equal=repeat_exact,
                checks=checks, passed=repeat_exact and all(row['passed'] for row in checks)))
            save(a.output/'progress.json', dict(rows=reports))
            print(json.dumps(reports[-1]), flush=True)
        del layer, value, gradient, reference, observed, previous
        torch.cuda.empty_cache()
    save(a.output/'complete.json', dict(rows=reports, actual_optimizer_updates=0,
        passed=all(row['passed'] for row in reports), production_approved=False,
        synthetic_inputs=True, quality_approved=False, torch_version=torch.__version__,
        cudnn_version=torch.backends.cudnn.version(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    save(a.output/'state.json', dict(status='complete', actual_optimizer_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--paused-owner-pid', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=0))
        save(args.output/'state.json', dict(status='failed', error=repr(error)))
        raise
