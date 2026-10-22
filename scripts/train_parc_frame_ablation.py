"""Matched three-stage PARC coordinate-frame experiment using native train steps."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from motionbricks.data.native_mb_data import NativeMBMotionDataset, native_training_loader
from motionbricks.data.native_mb_models import build_native_model
from motionbricks.data.native_mb_vertical_frame import VERTICAL_FRAME
from motionbricks.motionlib.train.scheduler import WarmupCosineScheduler
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import select_gpu, verify_training_source_policy, verify_dataset, sha256
from scripts.train_native_mb_control import state_digest


PROTOCOL = 'matched-parc-frame-v1'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame', choices=['absolute', 'observed'], required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, default=2333)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 10000:
        parser.error('Use a new directory and a bounded positive update budget')
    gpu = select_gpu(args.gpu)
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    source_infer, source = load_terrain_bundle(args.source, 'cpu')
    if source['metadata'].get('codec_height_adapter') or source['metadata'].get('matched_stage_adaptation'):
        raise ValueError('Use the original matched online pair and original codec as shared initialization')
    manifest = json.loads((args.dataset / 'native_manifest.json').read_text())
    verify_dataset(args.dataset, manifest)
    provenance = verify_training_source_policy(args.dataset, manifest, parc_only=True)
    if sha256(args.dataset / 'native_manifest.json') != source['metadata']['dataset_manifest_sha256']:
        raise ValueError('Source pipeline and experiment dataset differ')
    version = VERTICAL_FRAME if args.frame == 'observed' else None
    configs = {stage: OmegaConf.create(deepcopy(source[stage + '_config']))
               for stage in ('tokenizer', 'root', 'pose')}
    for stage, config in configs.items():
        config.model.args.vertical_frame = version
        config.model.args.matched_stage_adaptation = PROTOCOL
        if stage in ('tokenizer', 'pose'):
            config.model.args.codec_context_policy = 'played_context4'
    args.output.mkdir(parents=True)
    status = {'schema': PROTOCOL, 'status': 'running', 'vertical_frame': version,
              'source_bundle_sha256': sha256(args.source), 'seed': args.seed,
              'dataset_manifest_sha256': sha256(args.dataset / 'native_manifest.json'),
              'allowed_sources': provenance, 'updates_per_stage': args.steps, 'gpu_at_launch': gpu,
              'architecture_and_loss_coefficients': 'source native MB unchanged',
              'context_policy': 'played_context4',
              'normalization': 'same immutable PARC train-only affine statistics in both arms',
              'vertical_gauge_m': float(source_infer.global_motion_rep.stats.mean[1]),
              'normalization_note': 'coordinate origin centers on the train mean; statistics are fixed scales, not a world-height observation',
              'optimizer': 'fresh Adam 5e-5, warmup10000, cosine2000000, batch256, clip0.5',
              'codebook': 'encoder and codebook adapted; Pose retargeted to the new frozen vocabulary',
              'quality_approved': False, 'stages': {}}
    save_json(args.output / 'state.json', status)
    rep = source_infer.motion_rep
    models = {}
    for stage in ('tokenizer', 'root', 'pose'):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = build_native_model(stage, rep, configs[stage],
                                   tokenizer=models.get('tokenizer') if stage == 'pose' else None)
        if stage == 'tokenizer':
            model.pose_net.load_state_dict(source['tokenizer_state'], strict=True)
        else:
            model.load_state_dict(source[stage + '_state'], strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.to('cuda').train()
        if stage == 'pose':
            codec = models['tokenizer'].pose_net
            codec.eval()
            for parameter in codec.parameters():
                parameter.requires_grad_(False)
            model.backbone_net.init_embedding_from_codebooks(codec.get_codebook(), None)
            root = models['root'].to('cuda').eval().requires_grad_(False)
            object.__setattr__(model, 'online_root_model', root)
            model._vqvae_model_loaded = True
            frozen = {'tokenizer': state_digest(codec), 'root': state_digest(root)}
        else:
            frozen = None
        data = NativeMBMotionDataset(args.dataset, 'train', min_frames=65, max_frames=65,
            require_terrain=stage != 'tokenizer', require_online_terrain=stage != 'tokenizer',
            cache_max_bytes=512 * 1024**2)
        loader = native_training_loader(data, 256, 2, paper=True)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        scheduler = WarmupCosineScheduler(optimizer, num_warmup_steps=10000,
            num_training_steps=2000000, final_lr=2e-6, last_epoch=-1)
        start = time.perf_counter()
        sampling = hashlib.sha256()
        iterator = iter(loader)
        rows, logged = [], {}
        def log(name, value, **kwargs):
            if torch.is_tensor(value) and value.numel() == 1:
                logged[name] = float(value.detach())
        model.log = log
        for step in range(args.steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            sampling.update(batch['motion'].contiguous().numpy().tobytes())
            batch = {key: value.to('cuda') if torch.is_tensor(value) else value for key, value in batch.items()}
            # Same official keyframe curriculum position for both arms.
            model._trainer = SimpleNamespace(global_step=50000 + step)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_step(batch, step)
            if loss is None or not torch.isfinite(loss):
                raise ValueError(f'{stage} skipped or produced nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient norm')
            optimizer.step()
            scheduler.step()
            if step % 100 == 0 or step + 1 == args.steps:
                row = {'stage': stage, 'update': step + 1, 'loss': float(loss), **logged}
                rows.append(row)
                print(json.dumps(row), flush=True)
        if frozen and frozen != {'tokenizer': state_digest(model.supporting_nets['pose_net']),
                                 'root': state_digest(model.online_root_model)}:
            raise ValueError('Frozen Pose dependencies changed')
        model.eval()
        checkpoint = args.output / f'{stage}.pt'
        torch.save({'state_dict': model.state_dict(), 'config': OmegaConf.to_container(configs[stage]),
                    'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
                    'optimizer_updates': args.steps, 'vertical_frame': version}, checkpoint)
        save_json(args.output / f'{stage}_training.json', rows)
        status['stages'][stage] = {'updates': args.steps, 'checkpoint_sha256': sha256(checkpoint),
            'state_digest': state_digest(model), 'input_batch_sampling_sha256': sampling.hexdigest(),
            'elapsed_seconds': time.perf_counter() - start, 'frozen_dependencies': frozen}
        save_json(args.output / 'state.json', status)
        models[stage] = model
        if hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()
        del optimizer, scheduler, iterator, loader, data
        if stage != 'pose':
            model.to('cpu')
            torch.cuda.empty_cache()
    bundle = deepcopy(source)
    codec = models['tokenizer'].pose_net
    bundle['tokenizer_state'] = codec.state_dict()
    bundle['tokenizer_state_sha256'] = state_digest(codec)
    for stage in ('root', 'pose'):
        bundle[stage + '_state'] = models[stage].state_dict()
        bundle[stage + '_backbone_sha256'] = state_digest(models[stage].backbone_net)
    for stage in configs:
        bundle[stage + '_config'] = OmegaConf.to_container(configs[stage], resolve=False)
    metadata = bundle['metadata']
    metadata['root_checkpoint_sha256'] = status['stages']['root']['checkpoint_sha256']
    metadata['pose_checkpoint_sha256'] = status['stages']['pose']['checkpoint_sha256']
    metadata['tokenizer_checkpoint_sha256'] = status['stages']['tokenizer']['checkpoint_sha256']
    metadata['root_step'] = metadata['pose_step'] = args.steps
    metadata['pose_root_dependency'] = {'checkpoint_sha256': metadata['root_checkpoint_sha256'],
        'state_digest': state_digest(models['root']), 'end_state_digest': state_digest(models['root']),
        'dataset_manifest_sha256': metadata['dataset_manifest_sha256'], 'frozen': True}
    metadata['matched_stage_adaptation'] = {'version': PROTOCOL, 'status': 'completed',
        'vertical_frame': version, 'context_policy': 'played_context4',
        'tokenizer_state_sha256': bundle['tokenizer_state_sha256'],
        'pose_tokenizer_state_sha256': frozen['tokenizer'], 'source_bundle_sha256': status['source_bundle_sha256'],
        'stages': status['stages'], 'quality_approved': False}
    output = args.output / 'pipeline.pt'
    torch.save(bundle, output)
    load_terrain_bundle(output, 'cpu')
    status.update(status='completed', bundle_sha256=sha256(output))
    save_json(args.output / 'state.json', status)
    print(json.dumps({'completed': str(output), 'sha256': status['bundle_sha256'], 'quality_approved': False}), flush=True)


if __name__ == '__main__':
    main()
