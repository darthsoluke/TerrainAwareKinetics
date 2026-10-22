"""Diagnostic-only spatial-axis layout for the measured slow dilated Conv1d."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
from types import MethodType


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


@contextmanager
def vertical_convolutions(controller, *, channels_last=False):
    import torch
    from torch.nn import functional as F

    changed = []
    try:
        for name, module in controller.named_modules():
            if not (isinstance(module, torch.nn.Conv1d) and module.kernel_size == (3,)
                    and module.dilation[0] > 1 and module.stride == (1,)
                    and module.padding == module.dilation and module.groups == 1
                    and module.padding_mode == 'zeros'):
                continue
            previous = module.__dict__.get('forward')

            def forward(current, value):
                # Preserve every tap; only exchange the singleton spatial axis.
                image, kernel = value.unsqueeze(-1), current.weight.unsqueeze(-1)
                if channels_last:
                    image = image.contiguous(memory_format=torch.channels_last)
                    kernel = kernel.contiguous(memory_format=torch.channels_last)
                return F.conv2d(image, kernel,
                    current.bias, padding=(current.padding[0], 0),
                    dilation=(current.dilation[0], 1)).squeeze(-1).contiguous()

            module.forward = MethodType(forward, module)
            changed.append((name, module, previous))
        yield [name for name, _, _ in changed]
    finally:
        for _, module, previous in reversed(changed):
            if previous is None:
                del module.forward
            else:
                module.forward = previous


def main(a):
    if a.after_probe is not None:
        while not (a.after_probe/'complete.json').exists():
            if (a.after_probe/'failure.json').exists():
                raise RuntimeError('Preceding diagnostic failed; inspect it before continuing')
            save(a.output/'state.json', dict(status='waiting_for_preceding_probe',
                preceding_probe=str(a.after_probe), pid=os.getpid(), actual_optimizer_updates=0))
            time.sleep(30)
    while True:
        snapshot = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        measured = {int(row.split(',')[0]): tuple(map(int, row.split(',')[1:]))
            for row in snapshot.splitlines()}
        memory, busy = measured[a.gpu]
        if a.paused_owner_pid is not None:
            owner = Path('/proc')/str(a.paused_owner_pid)
            command = (owner/'cmdline').read_bytes().replace(bytes([0]), b' ').decode()
            owner_state = next(line.split()[1] for line in (owner/'status').read_text().splitlines()
                if line.startswith('State:'))
            gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                '--format=csv,noheader,nounits'], text=True)
            pids = {int(row.split(',')[0]) for row in apps.splitlines()
                if row.split(',')[1].strip() == gpu_uuid}
            if owner_state != 'T' or pids != {a.paused_owner_pid} or 'full_training_v3/root' not in command:
                raise RuntimeError('GPU must contain only our stopped Root')
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
    import numpy as np
    import torch
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, collate_primitives
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch
    from scripts.train_native_mb_control import state_digest
    from scripts.train_native_mb import sha256

    torch.set_num_threads(4)
    if a.paused_owner_pid is not None:
        torch.cuda.set_per_process_memory_fraction(.1)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    inference, bundle = load_terrain_bundle(a.bundle, 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    saved = torch.load(a.initial, map_location='cpu', mmap=True, weights_only=False)
    controller.load_state_dict(saved['model_state'], strict=True)
    original_hash = state_digest(controller)
    dataset = PrimitiveDataset([p/'data' for p in a.cache], 'train')
    rng, selected = np.random.default_rng(2333), {}
    for domain, count in (('pfnn', 32), ('parc', 16), ('bones', 16)):
        ids = [i for i, index in enumerate(dataset.ids)
            if dataset.records[index]['domain'] == domain]
        selected[domain] = rng.choice(ids, count*a.batches, replace=False).reshape(a.batches, count)
    reports = []
    try:
        for stage_name in ('root', 'codec'):
            torch.use_deterministic_algorithms(True, warn_only=stage_name == 'codec')
            stage = NativeTrainingStage(controller, stage_name).train(True)
            for batch_index in range(a.batches):
                rows = [dataset[int(i)] for domain in ('pfnn', 'parc', 'bones')
                    for i in selected[domain][batch_index]]
                batch = device_batch(collate_primitives(rows), 'cuda:0')
                reference = None
                for variant in ('baseline', a.layout):
                    from contextlib import nullcontext
                    if variant == 'batched_bias':
                        from motionbricks.data.dilated_bias_convolution import coalesced_dilated_bias
                        guard = coalesced_dilated_bias(controller)
                    else:
                        guard = (nullcontext([]) if variant == 'baseline' else
                            vertical_convolutions(controller, channels_last=variant == 'channels_last'))
                    durations = []
                    with guard as replaced:
                        for repeat in range(2):
                            controller.load_state_dict(saved['model_state'], strict=True)
                            controller.zero_grad(set_to_none=True)
                            torch.manual_seed(2333+batch_index)
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            loss = stage(batch, 2333+batch_index)
                            loss.backward()
                            torch.cuda.synchronize()
                            durations.append(time.perf_counter()-started)
                            loss_value = float(loss.detach())
                            del loss
                        observed = dict(loss=loss_value)
                        # Repeating above leaves the last gradients and updated EMA buffers intact.
                        observed['gradients'] = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                            for name, p in controller.named_parameters() if p.requires_grad}
                        observed['buffers'] = {name: value.detach().cpu().clone()
                            for name, value in controller.named_buffers() if name.startswith('codec.quantizer.')}
                        observed['rng'] = [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]
                        if stage_name == 'root':
                            observed['output'] = controller.last_root['pred_global_root_values'].detach().cpu().clone()
                        else:
                            observed['output'] = controller.last_local.detach().cpu().clone()
                    if reference is None:
                        reference = observed
                    errors, gradient_max, exact_gradients = [], 0., True
                    for name, expected in reference['gradients'].items():
                        actual = observed['gradients'][name]
                        if expected is None or actual is None:
                            if (expected is None) != (actual is None):
                                errors.append('Gradient presence: '+name)
                            continue
                        gradient_max = max(gradient_max, float((actual-expected).abs().max()))
                        exact_gradients = exact_gradients and torch.equal(actual, expected)
                        if not torch.allclose(actual, expected, atol=5e-6, rtol=1e-3):
                            errors.append('Gradient tolerance: '+name)
                    buffers_equal = all(torch.equal(value, reference['buffers'][name])
                        for name, value in observed['buffers'].items())
                    rng_equal = all(torch.equal(x, y) for x, y in zip(observed['rng'], reference['rng']))
                    if not buffers_equal:
                        errors.append('Quantizer buffers differ')
                    if not rng_equal:
                        errors.append('RNG differs')
                    loss_difference = abs(observed['loss']-reference['loss'])
                    if loss_difference > 1e-6+1e-5*abs(reference['loss']):
                        errors.append('Loss tolerance')
                    if not torch.allclose(observed['output'], reference['output'], atol=1e-6, rtol=1e-5):
                        errors.append('Native output tolerance')
                    reports.append(dict(stage=stage_name, batch_index=batch_index, variant=variant,
                        seconds=durations, loss=loss_value, loss_difference=loss_difference,
                        gradient_max_abs=gradient_max,
                        gradients_bitwise_equal=exact_gradients, quantizer_buffers_bitwise_equal=buffers_equal,
                        rng_bitwise_equal=rng_equal,
                        output_max_abs=float((observed['output']-reference['output']).abs().max()),
                        replaced_modules=replaced, passed=not errors, errors=errors))
                    save(a.output/'progress.json', dict(rows=reports))
                    print(json.dumps(reports[-1]), flush=True)
                del reference, observed, batch
                controller.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
        controller.load_state_dict(saved['model_state'], strict=True)
        controller.zero_grad(set_to_none=True)
        if state_digest(controller) != original_hash:
            raise ValueError('Initial model tensors were not restored')
        save(a.output/'complete.json', dict(rows=reports, passed=all(row['passed'] for row in reports),
            actual_optimizer_updates=0, quality_approved=False, production_approved=False,
            original_weights_restored=True, gpu=a.gpu, initial_sha256=sha256(a.initial),
            script_sha256=sha256(Path(__file__))))
        save(a.output/'state.json', dict(status='complete', actual_optimizer_updates=0))
    finally:
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'initial', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cache', type=Path, nargs='+', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--layout', choices=('vertical', 'channels_last', 'batched_bias'), default='vertical')
    parser.add_argument('--after-probe', type=Path)
    parser.add_argument('--paused-owner-pid', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=0))
        save(args.output/'state.json', dict(status='failed', error=repr(error)))
        raise
