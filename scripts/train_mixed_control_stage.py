"""Native mixed-corpus training with explicit batch/precision and held-out selection.

Keep the existing model and objective; record throughput profiles as changes
to the executable training configuration, never as unchanged-paper claims.
"""

import argparse
from copy import deepcopy
from functools import partial
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
from lightning_fabric.utilities.seed import pl_worker_init_function

from motionbricks.data.native_mb_data import NativeMBMotionDataset, build_native_representation, collate_native_batch
from motionbricks.data.native_mb_models import build_native_model
from motionbricks.data.native_mb_paper_protocol import configure_paper_optimizer
from motionbricks.data.native_mb_terrain_protocol import INTENT_RANGE_POLICY
from motionbricks.data.native_mb_online_conditions import enable_velocity_conditioning, VELOCITY_PROTOCOL
from motionbricks.data.mixed_control_training import (ADAPTATION, validate_source_policy, sampling_weights,
                                                      native_validation_loss, summarize_validation)
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import verify_dataset, sha256
from scripts.train_native_mb_control import state_digest


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load_stage(path, stage, representation):
    contract = json.loads((path / 'run_contract.json').read_text())
    result = json.loads((path / 'training_result.json').read_text())
    checkpoint = path / result.get('checkpoint_relative', 'checkpoints/final.ckpt')
    if contract['stage'] != stage or sha256(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Stage checkpoint is missing or changed')
    config = OmegaConf.load(path / 'native_config.yaml')
    model = build_native_model(stage, representation, config)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict'], strict=True)
    return model, contract, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('tokenizer', 'root', 'pose'), required=True)
    parser.add_argument('--tokenizer-run', type=Path)
    parser.add_argument('--root-run', type=Path)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--global-batch', type=int, default=256)
    parser.add_argument('--precision', choices=('32', 'bf16-mixed'), default='32')
    parser.add_argument('--validation-crops', type=int, default=4)
    parser.add_argument('--validate-every', type=int, default=5000)
    parser.add_argument('--checkpoint-selection', choices=('validation', 'final'), default='validation')
    parser.add_argument('--sampling-policy', choices=('clips-v1', 'parc-windows-v1'), default='clips-v1')
    parser.add_argument('--vertical-frame', choices=('observed-pelvis-y-v1', 'observed-terrain-y-v1'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    rank, world, local_rank = (int(os.environ.get(key, default)) for key, default in
                               (('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')))
    if (world < 1 or world > 2 or args.global_batch < world or args.global_batch % world
            or min(args.steps, args.validation_crops, args.validate_every) < 1):
        parser.error('Use positive budgets and a global batch divisible by the rank count')
    if args.output.exists() and not args.resume:
        parser.error('Use a new run or explicitly resume its last checkpoint')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.cuda.set_device(local_rank)
    pl.seed_everything(args.seed, workers=True)
    manifest = json.loads((args.dataset / 'native_manifest.json').read_text())
    validate_source_policy(manifest.get('source_policy'))
    verify_dataset(args.dataset, manifest)
    dataset_hash, source_hash = sha256(args.dataset / 'native_manifest.json'), sha256(args.source)
    source_inference, source = load_terrain_bundle(args.source, 'cpu')
    rep = build_native_representation(args.dataset / 'stats')
    torch.testing.assert_close(rep.dual_rep.stats.mean, source['stats_mean'], atol=0, rtol=0)
    torch.testing.assert_close(rep.dual_rep.stats.std, source['stats_std'], atol=0, rtol=0)
    config = OmegaConf.create(deepcopy(source[args.stage + '_config']))
    if args.vertical_frame:
        config.model.args.vertical_frame = args.vertical_frame
    config.model.args.matched_stage_adaptation = ADAPTATION
    if args.stage in ('tokenizer', 'pose'):
        config.model.args.codec_context_policy = 'played_context4'
    if args.stage in ('root', 'pose'):
        config = enable_velocity_conditioning(config)
        config.model.args.observation_range_policy = INTENT_RANGE_POLICY
    config.trainer.max_steps = args.steps
    config.model.args.mixed_source_policy = manifest['source_policy']['version']
    dependency = None
    if args.stage == 'pose':
        if args.tokenizer_run is None or args.root_run is None:
            parser.error('Pose needs the completed matching Tokenizer and Root runs')
        tokenizer, tc, tr = load_stage(args.tokenizer_run, 'tokenizer', rep)
        root, rc, rr = load_stage(args.root_run, 'root', rep)
        if any(row['dataset_manifest_sha256'] != dataset_hash for row in (tc, rc)):
            raise ValueError('Pose stage dependencies use different data')
        model = build_native_model('pose', rep, config, tokenizer=tokenizer)
        model.load_state_dict(source['pose_state'], strict=True)
        model.backbone_net.init_embedding_from_codebooks(tokenizer.pose_net.get_codebook(), None)
        root.eval().requires_grad_(False)
        object.__setattr__(model, 'online_root_model', root)
        model._vqvae_model_loaded = True
        dependency = {'tokenizer_checkpoint_sha256': tr['checkpoint_sha256'],
                      'tokenizer_state_sha256': state_digest(tokenizer.pose_net),
                      'root_checkpoint_sha256': rr['checkpoint_sha256'], 'root_state_sha256': state_digest(root)}
    else:
        model = build_native_model(args.stage, rep, config)
        if args.stage == 'tokenizer':
            model.pose_net.load_state_dict(source['tokenizer_state'], strict=True)
        else:
            missing, unexpected = model.load_state_dict(source['root_state'], strict=False)
            if set(missing) - {'backbone_net._proj_local_root_component_mask.weight'} or unexpected:
                raise ValueError('Only the new local-condition mask projection may be missing in the warmstart')
    del source_inference, source
    configure_paper_optimizer(model, config)
    needs_scene = args.stage != 'tokenizer' or config.model.args.get('vertical_frame') == 'observed-terrain-y-v1'
    data = NativeMBMotionDataset(args.dataset, 'train', min_frames=65, max_frames=65,
        require_terrain=needs_scene, require_online_terrain=needs_scene,
        online_only=needs_scene,
        cache_max_bytes=4 * 1024**3)
    weights = sampling_weights(data.records, policy=args.sampling_policy)
    per_rank = args.global_batch // world
    base_sampler = WeightedRandomSampler(weights, args.global_batch * 256, replacement=True,
                                         generator=torch.Generator().manual_seed(args.seed))
    sampler = DistributedSamplerWrapper(base_sampler, num_replicas=world, rank=rank,
        shuffle=False, seed=args.seed, drop_last=True) if world > 1 else base_sampler
    loader = DataLoader(data, batch_size=per_rank, sampler=sampler, drop_last=True,
        num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True,
        collate_fn=collate_native_batch, worker_init_fn=partial(pl_worker_init_function, rank=rank),
        generator=torch.Generator().manual_seed(args.seed + rank))
    validation = NativeMBMotionDataset(args.dataset, 'validation', min_frames=65, max_frames=65,
        require_terrain=needs_scene, require_online_terrain=needs_scene,
        online_only=needs_scene, cache_max_bytes=512 * 1024**2)
    # Pin validation crops separately from train sampling; preserve both RNGs.
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1801)
        np.random.seed(1801)
        grouped = {}
        for index, row in enumerate(validation.records):
            domain = 'bones' if row['origin'] == 'bones_seed' else 'parc'
            category = row['control_category'] if domain == 'bones' else row.get('terrain_family', 'unspecified')
            grouped.setdefault((domain, category), []).extend([index] * args.validation_crops)
        validation_batches = []
        for (domain, category), indices in sorted(grouped.items()):
            for start in range(0, len(indices), 16):
                batch = collate_native_batch([validation[i] for i in indices[start:start+16]])
                validation_batches.append((domain, domain + '/' + category, batch))
    np.random.set_state(numpy_state)
    missing_control_categories = sorted(
        {row['control_category'] for row in data.records if row['origin'] == 'bones_seed'}
        - {row['control_category'] for row in validation.records if row['origin'] == 'bones_seed'})
    contract = {'stage': args.stage, 'adaptation': ADAPTATION, 'dataset': str(args.dataset.resolve()),
        'dataset_manifest_sha256': dataset_hash, 'source_bundle_sha256': source_hash,
        'source_policy': manifest['source_policy'], 'steps': args.steps, 'world_size': world,
        'batch': {'global': args.global_batch, 'per_rank': per_rank,
                  'profile': 'paper_per_gpu' if per_rank == 256 else 'explicit_throughput_profile',
                  'native_window_multiplier': config.model.args.batchsize_mul_factor},
        'precision': args.precision, 'matmul_precision': 'high', 'dependencies': dependency,
        'validation': {'crops_per_record': args.validation_crops, 'interval_updates': args.validate_every,
                       'checkpoint_selection': args.checkpoint_selection, 'retain_all_periodic': True,
                       'missing_control_categories': missing_control_categories,
                       'validation_records': len(validation)},
        'optimizer': 'Adam 5e-5; warmup10000; cosine2000000; clip0.5',
        'online_protocol': VELOCITY_PROTOCOL,
        'vertical_frame': config.model.args.get('vertical_frame'),
        'initialization': 'weights-only warmstart; new data/observation protocol; fresh optimizer',
        'sampling': '50 percent PARC; 50 percent BONES, balanced across observed control categories',
        'sampling_policy': args.sampling_policy,
        'parc_sampling': ('uniform eligible 65-frame windows' if args.sampling_policy == 'parc-windows-v1'
                          else 'uniform clips'),
        'inference_location': 'local PC only', 'quality_approved': False}
    if args.resume:
        previous = json.loads((args.output / 'run_contract.json').read_text())
        if previous != contract:
            raise ValueError('Resume contract changed')
    elif rank == 0:
        args.output.mkdir(parents=True)
        write_json(args.output / 'run_contract.json', contract)
        OmegaConf.save(config, args.output / 'native_config.yaml')
    started = time.perf_counter()

    class Progress(pl.Callback):
        def __init__(self):
            self.validation_rows = (json.loads((args.output / 'validation.json').read_text())
                                    if args.resume and (args.output / 'validation.json').exists() else [])
            self.update_seconds = []

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            self.batch_started = time.perf_counter()

        def measure_validation(self, trainer, module):
            previous_numpy = np.random.get_state()
            original_log = module.log
            module.eval()
            module.log = lambda *args, **kwargs: None
            losses = []
            try:
                with torch.random.fork_rng(devices=[module.device]), torch.no_grad():
                    torch.manual_seed(1801)
                    np.random.seed(1801)
                    for index, (domain, group, source_batch) in enumerate(validation_batches):
                        if index % world != rank:
                            continue
                        torch.manual_seed(1801 + index)
                        np.random.seed(1801 + index)
                        current = {key: value.to(module.device) if torch.is_tensor(value) else value
                                   for key, value in source_batch.items()}
                        with trainer.precision_plugin.forward_context():
                            value = native_validation_loss(module, args.stage, current, index)
                        if value is None or not torch.isfinite(value):
                            raise ValueError('Fixed held-out loss is skipped or nonfinite')
                        losses.append({'domain': domain, 'group': group,
                                       'samples': source_batch['batch_size'], 'loss': float(value)})
                if world > 1:
                    collected = [None] * world
                    torch.distributed.all_gather_object(collected, losses)
                    losses = [row for rank_rows in collected for row in rank_rows]
            finally:
                module.log = original_log
                module.train()
                np.random.set_state(previous_numpy)
            result = {'step': trainer.global_step, **summarize_validation(losses),
                      'batches': len(losses), 'missing_control_categories': missing_control_categories,
                      'scope': 'fixed held-out native objective; not closed-loop approval'}
            self.validation_rows.append(result)
            if trainer.is_global_zero:
                write_json(args.output / 'validation.json', self.validation_rows)
                print(json.dumps({'validation': result}), flush=True)

        def on_train_end(self, trainer, module):
            if not self.validation_rows or self.validation_rows[-1]['step'] != trainer.global_step:
                self.measure_validation(trainer, module)

        def on_train_start(self, trainer, module):
            self.measure_validation(trainer, module)

        def on_before_optimizer_step(self, trainer, module, optimizer):
            grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not torch.stack([torch.isfinite(value).all() for value in grads]).all():
                raise ValueError('Missing or nonfinite native gradients')
            if dependency:
                for frozen in (module.supporting_nets['pose_net'], module.online_root_model):
                    if frozen.training or any(p.requires_grad or p.grad is not None for p in frozen.parameters()):
                        raise ValueError('Pose dependencies stopped being frozen')

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if outputs is None:
                raise ValueError('The native training step skipped a batch')
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite native loss')
            self.update_seconds.append(time.perf_counter() - self.batch_started)
            if trainer.is_global_zero and (trainer.global_step % 100 == 0 or trainer.global_step == 1):
                record = {'stage': args.stage, 'step': trainer.global_step, 'loss': float(loss),
                          'elapsed_seconds': time.perf_counter()-started, 'status': 'training'}
                write_json(args.output / 'progress.json', record)
                print(json.dumps(record), flush=True)
            if trainer.global_step % args.validate_every == 0:
                self.measure_validation(trainer, module)

    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.output / 'checkpoints',
        filename='native-{step}', every_n_train_steps=args.validate_every, save_top_k=-1, save_last=True,
        save_on_train_epoch_end=False)
    strategy = (pl.strategies.DDPStrategy(find_unused_parameters=True, process_group_backend='nccl')
                if world > 1 else 'auto')
    if world > 1 and args.stage == 'pose':
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=False, process_group_backend='nccl',
                                             gradient_as_bucket_view=True)
    progress = Progress()
    trainer = pl.Trainer(max_steps=args.steps, accelerator='gpu', devices=world, strategy=strategy,
        precision=32 if args.precision == '32' else args.precision,
        gradient_clip_val=.5, logger=pl.loggers.CSVLogger(str(args.output), name='logs', version=0),
        callbacks=[progress, checkpoint], log_every_n_steps=25, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, use_distributed_sampler=False)
    trainer.fit(model, train_dataloaders=loader,
                ckpt_path=str(args.output / 'checkpoints/last.ckpt') if args.resume else None)
    final = args.output / 'checkpoints/final.ckpt'
    trainer.save_checkpoint(final)
    if world > 1:
        torch.distributed.barrier()
    if dependency:
        if (state_digest(model.supporting_nets['pose_net']) != dependency['tokenizer_state_sha256']
                or state_digest(model.online_root_model) != dependency['root_state_sha256']):
            raise ValueError('Frozen Pose dependency tensors changed')
    if trainer.is_global_zero:
        selected, selected_step = final, trainer.global_step
        if args.checkpoint_selection == 'validation':
            candidates = [(row, args.output / 'checkpoints' / f"native-step={row['step']}.ckpt")
                          for row in progress.validation_rows if row['step'] > 0]
            candidates = [(row, path if path.exists() else final) for row, path in candidates
                          if path.exists() or row['step'] == trainer.global_step]
            best, selected = min(candidates, key=lambda pair: pair[0]['selection_loss'])
            selected_step = best['step']
        final_state = state_digest(model)
        if selected != final:
            model.load_state_dict(torch.load(selected, map_location='cpu', weights_only=False)['state_dict'], strict=True)
        result = {'optimizer_updates': trainer.global_step, 'checkpoint_sha256': sha256(selected),
                  'checkpoint_relative': str(selected.relative_to(args.output)), 'selected_step': selected_step,
                  'final_checkpoint_sha256': sha256(final),
                  'final_state_digest': final_state,
                  'throughput': {'scope': 'training batch after input transfer, excludes validation/checkpoint I/O',
                      'median_update_seconds': float(np.median(progress.update_seconds[5:] or progress.update_seconds)),
                      'input_samples_per_second': args.global_batch / float(np.mean(progress.update_seconds[5:] or progress.update_seconds)),
                      'native_windows_per_second': args.global_batch * config.model.args.batchsize_mul_factor
                          / float(np.mean(progress.update_seconds[5:] or progress.update_seconds)),
                      'peak_allocated_mib_rank0': torch.cuda.max_memory_allocated() / 1024**2},
                  'elapsed_seconds': time.perf_counter()-started, 'state_digest': state_digest(model),
                  'dependencies': dependency, 'quality_approved': False}
        write_json(args.output / 'training_result.json', result)
        print(json.dumps(result), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
