"""Replay saved terminal-convolution derivatives at controlled memory pressure."""

import argparse
import json
import os
from pathlib import Path


def main(a):
    from scripts.run_pfnn_parallel_workflow import gpu_snapshot
    from scripts.probe_full_training_execution import save
    snapshot, available = gpu_snapshot()
    if a.gpu not in available: raise ValueError('Selected GPU is occupied')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDNN_LOGLEVEL_DBG'] = '3'
    os.environ['CUDNN_LOGDEST_DBG'] = str(a.output/'cudnn.log')
    import torch
    from scripts.train_native_mb import sha256
    _, available = gpu_snapshot()
    if a.gpu not in available: raise ValueError('Selected GPU became occupied')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.set_per_process_memory_fraction(.1, 0)
    run = a.checkpoint.parent.parent
    native = torch.load(run/'codec_graph_isolation_v1/native_reference.pt', map_location='cpu', mmap=True, weights_only=False)
    changed = torch.load(run/'codec_graph_isolation_v1/native_with_both_caches.pt', map_location='cpu', mmap=True, weights_only=False)
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    value = native['target']['input'].cuda()
    upstream = native['target']['upstream'].cuda()
    weight = saved['model_state']['codec.decoder.model.6.weight'].cuda()
    reference = native['gradients']['codec.decoder.model.6.weight']
    changed_reference = changed['gradients']['codec.decoder.model.6.weight']
    rows = []
    with torch.no_grad():
        for pressure_mib in (800, 1600, 2400, 800):
            torch.cuda.empty_cache()
            pressure = torch.empty(pressure_mib*1024*1024, dtype=torch.uint8, device='cuda:0')
            log = a.output/'cudnn.log'
            begin = log.stat().st_size if log.exists() else 0
            derivatives = torch.ops.aten.convolution_backward.default(upstream, value, weight,
                [365], [1], [1], [1], False, [0], 1, [True, True, True])
            torch.cuda.synchronize()
            observed = derivatives[1].cpu()
            row = dict(pressure_mib=pressure_mib, native_equal=torch.equal(observed,reference),
                changed_equal=torch.equal(observed,changed_reference),
                native_max_abs=float((observed-reference).abs().max()),
                changed_max_abs=float((observed-changed_reference).abs().max()),
                allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                cudnn_log_byte_start=begin,cudnn_log_byte_end=log.stat().st_size if log.exists() else 0,
                input_stride=list(value.stride()),upstream_stride=list(upstream.stride()))
            rows.append(row)
            torch.save(observed,a.output/f'weight_gradient_{len(rows)}.pt')
            del derivatives, pressure
            save(a.output/'state.json',dict(status='running',rows=rows,actual_optimizer_updates=0))
    save(a.output/'complete.json',dict(completed=True,rows=rows,actual_optimizer_updates=0,
        source_checkpoint_sha256=sha256(a.checkpoint),allocation=snapshot,
        production_approved=False,script_sha256=sha256(Path(__file__)),
        scope='Identical saved layer input, weight and upstream; changed available workspace only.'))
    save(a.output/'state.json',dict(status='complete',actual_optimizer_updates=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:main(args)
    except Exception as error:
        (args.output/'failure.json').write_text(json.dumps(dict(error=repr(error),actual_optimizer_updates=0)))
        raise
