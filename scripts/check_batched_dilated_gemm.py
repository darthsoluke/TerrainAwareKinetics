"""Synthetic full-kernel GEMM dispatch probe; no model or optimizer is changed."""

import argparse
import json
import os
from pathlib import Path
import time


def main(args):
    from scripts.check_full_corpus_acceptance import check_idle, save
    allocation = check_idle(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from torch.nn import functional as F
    from motionbricks.data.dilated_bias_convolution import BatchedBiasNativeBackward
    from scripts.train_native_mb import sha256

    check_idle(args.gpu)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.1, 0)
    save(args.output/'state.json', dict(status='running', allocation=allocation,
        actual_optimizer_updates=0))

    class BatchedGemm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, weight, bias, dilation):
            ctx.save_for_backward(value, weight)
            ctx.bias_size, ctx.dilation = list(bias.shape), dilation
            columns = F.unfold(value.unsqueeze(2), (1, 3),
                dilation=(1, dilation), padding=(0, dilation))
            output = torch.empty((value.shape[0], weight.shape[0], value.shape[-1]),
                device=value.device, dtype=value.dtype)
            output.copy_(bias[None, :, None])
            matrices = weight.reshape(1, weight.shape[0], -1).expand(value.shape[0], -1, -1)
            output.baddbmm_(matrices, columns)
            return output

        backward = staticmethod(BatchedBiasNativeBackward.backward)

    graph_execution = None
    candidate = BatchedGemm.apply
    if args.dispatch != 'batched':
        from motionbricks.data.dilated_graph_convolution import GraphConvolutionExecution
        graph_execution = GraphConvolutionExecution(capture_backward=args.dispatch == 'graph-both')
        candidate = graph_execution.apply

    def execute(call, values, dilation, upstream):
        before = (torch.get_rng_state(), torch.cuda.get_rng_state())
        output = call(*values, dilation)
        gradients = torch.autograd.grad((output*upstream).sum(), values)
        after = (torch.get_rng_state(), torch.cuda.get_rng_state())
        return output.detach(), gradients, all(torch.equal(x, y) for x, y in zip(before, after))

    def latency(call, values, dilation, upstream=None):
        def run():
            output = call(*values, dilation)
            if upstream is not None:
                torch.autograd.grad((output*upstream).sum(), values)
        with (torch.enable_grad() if upstream is not None else torch.no_grad()):
            for _ in range(2):
                run()
            timings = []
            for _ in range(3):
                torch.cuda.synchronize()
                started = time.perf_counter()
                run()
                torch.cuda.synchronize()
                timings.append((time.perf_counter()-started)*1000)
        return timings

    rows = []
    for length in (8, 16):
        for dilation in (3, 9, 27):
            for seed in (2333, 9331):
                torch.manual_seed(seed)
                values = (torch.randn(64, 512, length, device='cuda', requires_grad=True),
                    (torch.randn(512, 512, 3, device='cuda')*.02).requires_grad_(True),
                    (torch.randn(512, device='cuda')*.02).requires_grad_(True))
                upstream = torch.randn(64, 512, length, device='cuda')
                expected, reference_gradients, first_rng = execute(
                    BatchedBiasNativeBackward.apply, values, dilation, upstream)
                actual, gradients, second_rng = execute(candidate, values, dilation, upstream)
                row = dict(length=length, dilation=dilation, seed=seed,
                    forward_bitwise_equal=torch.equal(actual, expected),
                    forward_max_abs=float((actual-expected).abs().max()),
                    forward_existing_gate=torch.allclose(actual, expected, atol=1e-6, rtol=1e-5),
                    native_stride=list(expected.stride()), candidate_stride=list(actual.stride()),
                    gradients_bitwise_equal=[torch.equal(x, y) for x, y in zip(gradients, reference_gradients)],
                    gradients_max_abs=[float((x-y).abs().max()) for x, y in zip(gradients, reference_gradients)],
                    gradients_existing_gate=all(torch.allclose(x, y, atol=5e-6, rtol=1e-3)
                        for x, y in zip(gradients, reference_gradients)), rng_unchanged=first_rng and second_rng,
                    reference_forward_ms=latency(BatchedBiasNativeBackward.apply, values, dilation),
                    candidate_forward_ms=latency(candidate, values, dilation),
                    reference_forward_backward_ms=latency(BatchedBiasNativeBackward.apply, values, dilation, upstream),
                    candidate_forward_backward_ms=latency(candidate, values, dilation, upstream))
                rows.append(row)
                save(args.output/'progress.json', dict(rows=rows, actual_optimizer_updates=0))
    eligible = all(row['forward_bitwise_equal'] and row['gradients_existing_gate']
        and row['native_stride'] == row['candidate_stride'] and row['rng_unchanged'] for row in rows)
    save(args.output/'complete.json', dict(passed=eligible, rows=rows, actual_optimizer_updates=0,
        dispatch=args.dispatch, graph_signatures=len(graph_execution.calls) if graph_execution else 0,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        full_model_validated=False, optimizer_validated=False, production_approved=False,
        forward_tolerance=dict(atol=1e-6, rtol=1e-5), gradient_tolerance=dict(atol=5e-6, rtol=1e-3),
        requires_exact_forward_for_next_ema_check=True, script_sha256=sha256(Path(__file__)),
        scope='Synthetic actual convolution shapes; full-model/EMA/optimizer evidence still required.'))
    save(args.output/'state.json', dict(status='complete', eligible_for_full_model_probe=eligible,
        actual_optimizer_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--dispatch',choices=('batched','graph-forward','graph-both'),default='batched')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except Exception as error:
        from scripts.check_full_corpus_acceptance import save
        save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=0))
        raise
