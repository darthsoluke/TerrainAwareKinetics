"""Bounded native Root/Pose training throughput on identical real PARC batches."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import patch

import numpy as np
import torch

from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import select_gpu, sha256
from motionbricks.data.native_mb_data import NativeMBMotionDataset, native_training_loader, collate_native_batch
from motionbricks.data.mixed_control_training import sampling_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('tokenizer', 'root', 'pose'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--precision', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--batch-snapshot', type=Path)
    parser.add_argument('--export-batches', type=Path)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--tokens', type=int, choices=range(3, 17))
    parser.add_argument('--batches', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or min(args.batch_size, args.steps, args.batches) < 1:
        parser.error('Use a new benchmark output')
    gpu = select_gpu(args.gpu)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(2333)
    np.random.seed(2333)
    inference, bundle = load_terrain_bundle(args.bundle, 'cuda:0')
    dataset_hash = sha256(args.dataset / 'native_manifest.json')
    if dataset_hash != bundle['metadata']['dataset_manifest_sha256']:
        raise ValueError('Benchmark data must be the checkpoint-bound PARC corpus')
    if args.stage == 'tokenizer':
        from omegaconf import OmegaConf
        from motionbricks.data.native_mb_models import build_native_model
        model = build_native_model('tokenizer', inference.motion_rep,
                                   OmegaConf.create(bundle['tokenizer_config']))
        model.pose_net.load_state_dict(bundle['tokenizer_state'], strict=True)
        model.to('cuda:0')
    else:
        model = inference._root_model if args.stage == 'root' else inference._pose_model
    model.train().requires_grad_(True)
    if args.stage == 'pose':
        for frozen in (inference._vqvae_pose_model, inference._root_model):
            frozen.eval()
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
        object.__setattr__(model, 'online_root_model', inference._root_model)
    model._trainer = SimpleNamespace(global_step=52000)
    model.log = lambda *values, **kwargs: None
    original_choice = np.random.choice
    token_choices = np.arange(model.args['min_tokens'], model.args['max_tokens'] + 1)

    def benchmark_choice(values, *positional, **keywords):
        if args.tokens and np.array_equal(values, token_choices):
            return args.tokens
        return original_choice(values, *positional, **keywords)
    if args.batch_snapshot:
        with gzip.open(args.batch_snapshot, 'rb') as stream:
            snapshot = torch.load(stream, map_location='cpu', weights_only=True)
        if snapshot['dataset_manifest_sha256'] != dataset_hash:
            raise ValueError('Frozen batches belong to a different dataset')
        batches = snapshot['batches']
    else:
        torch.manual_seed(2333)
        np.random.seed(2333)
        data = NativeMBMotionDataset(args.dataset, 'train', max_frames=65, min_frames=65,
            require_terrain=args.stage != 'tokenizer', require_online_terrain=args.stage != 'tokenizer',
            online_only=args.stage != 'tokenizer', cache_max_bytes=2 * 1024**3)
        if bundle['metadata'].get('source_policy'):
            sampler = torch.utils.data.WeightedRandomSampler(sampling_weights(data.records),
                args.batch_size * args.batches, replacement=True,
                generator=torch.Generator().manual_seed(2333))
            loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                sampler=sampler, collate_fn=collate_native_batch)
        else:
            if args.batch_size != 256:
                raise ValueError('Non-mixed benchmark retains the original batch256 contract')
            loader = native_training_loader(data, 256, 0, paper=True)
        iterator = iter(loader)
        batches = [next(iterator) for _ in range(args.batches)]
    if any(batch['batch_size'] != args.batch_size for batch in batches):
        raise ValueError('Frozen batch size differs from the requested profile')
    if args.export_batches:
        if args.export_batches.exists():
            raise ValueError('Refuse to overwrite frozen benchmark batches')
        args.export_batches.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.export_batches, 'wb', compresslevel=1) as stream:
            torch.save({'dataset_manifest_sha256': dataset_hash, 'batches': batches}, stream)
    batch_digest = hashlib.sha256()
    for batch in batches:
        batch_digest.update(batch['motion'].numpy().tobytes())
        if 'terrain_root_world' in batch:
            batch_digest.update(batch['terrain_root_world'].numpy().tobytes())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    torch.cuda.reset_peak_memory_stats()
    durations, losses, first_gradient = [], [], None
    initial_loss, initial_inputs = None, {}
    for step in range(args.warmup + args.steps):
        torch.manual_seed(1000 + step)
        np.random.seed(1000 + step)
        torch.cuda.synchronize()
        started = time.perf_counter()
        batch = {key: value.to('cuda:0') if torch.is_tensor(value) else value
                 for key, value in batches[step % len(batches)].items()}
        model._trainer.global_step = 52000 + step
        optimizer.zero_grad(set_to_none=True)
        with patch('numpy.random.choice', side_effect=benchmark_choice), \
                torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bfloat16'):
            loss = model.training_step(batch, step)
        if loss is None or not torch.isfinite(loss):
            raise ValueError('Native benchmark loss is not finite')
        loss.backward()
        if step == 0:
            initial_loss = float(loss)
            for key in ('groundtruth_pose_tokens', 'global_motions', 'local_root_values', 'pose_cond'):
                if torch.is_tensor(batch.get(key)):
                    value = batch[key].detach().cpu()
                    initial_inputs[key] = {'dtype': str(value.dtype),
                        'sha256': hashlib.sha256(value.float().numpy().tobytes()).hexdigest()}
            first_gradient = torch.cat([p.grad.detach().float().reshape(-1)[::max(1, p.numel()//32)][:32].cpu()
                for p in model.parameters() if p.requires_grad and p.grad is not None]).tolist()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        if not torch.isfinite(norm):
            raise ValueError('Native benchmark gradient is not finite')
        optimizer.step()
        torch.cuda.synchronize()
        if step >= args.warmup:
            durations.append(time.perf_counter()-started)
            losses.append(float(loss))
        if step % 10 == 0:
            print(json.dumps({'stage': args.stage, 'update': step, 'loss': float(loss)}), flush=True)
    result = {'gpu_at_launch': gpu, 'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'stage': args.stage, 'batch_size': args.batch_size, 'precision': args.precision,
        'tokens': args.tokens, 'initial_loss': initial_loss, 'initial_gradient_probe': first_gradient,
        'initial_input_digests': initial_inputs,
        'float32_matmul_precision': 'high', 'dataset_manifest_sha256': dataset_hash,
        'bundle_sha256': sha256(args.bundle), 'real_batches_sha256': batch_digest.hexdigest(),
        'measured_updates': len(durations), 'median_update_ms': float(np.median(durations)*1000),
        'p95_update_ms': float(np.percentile(durations, 95)*1000),
        'samples_per_second': args.batch_size / float(np.mean(durations)),
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
        'losses': losses, 'scope': 'native training_step, backward, clipping and Adam; cached real CPU batches with H2D; excludes cold disk loading',
        'quality_approved': False, 'weights_saved': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('losses', 'initial_gradient_probe')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
