"""Decoder-only dense-Y adaptation, fixed MB codebook and original MB losses.

Expert trajectories are training labels/conditions, never runtime substitutions.
Rigid vertical translations provide supervised altitude translations while
retaining base pose codes; they do not establish vertical disentanglement or
online quality.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from motionbricks.data.native_mb_codec_height import build_codec_height_tokenizer, rigid_vertical_pair, CODEC_HEIGHT_CONDITIONING
from motionbricks.data.native_mb_data import NativeMBMotionDataset, collate_native_batch
from motionbricks.helper.data_training_util import sample_motion_segments_from_motion_clips, sample_keyframes, extract_feature_from_motion_rep
from motionbricks.motionlib.train.scheduler import WarmupCosineScheduler
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import select_gpu, sha256, verify_training_source_policy, apply_native_matmul_precision
from scripts.train_native_mb_control import state_digest


CONTEXT_POLICIES = ("native_random", "played_context4")


def decoder_step(model, batch, *, translation_range=1., context_policy="native_random"):
    if context_policy not in CONTEXT_POLICIES:
        raise ValueError(f"Unknown context policy: {context_policy!r}")
    device = next(model.pose_net.parameters()).device
    tokens = int(np.random.choice(np.arange(model.args['min_tokens'], model.args['max_tokens'] + 1)))
    frames = tokens * 4
    global_motion = sample_motion_segments_from_motion_clips(batch['motion'].to(device),
        batch['motion_len'].to(device), frames, model.args['batchsize_mul_factor'],
        info={}, motion_rep=model.global_motion_rep)
    global_motion = model.global_motion_rep.change_first_heading(global_motion,
        torch.rand(len(global_motion), device=device) * (2 * np.pi), True, True)
    local = model.motion_rep.dual_rep.global_to_local(global_motion, True, True,
        lengths=torch.full((len(global_motion),), frames + 1, device=device))[:, :frames]
    global_motion = global_motion[:, :frames]
    delta = (torch.rand(len(local), device=device) * 2 - 1) * translation_range
    shifted_global, shifted_local = rigid_vertical_pair(global_motion, local, model.motion_rep, delta)
    probability, _ = model._construct_keyframe_prob()
    known, target = sample_keyframes(shifted_local, model.args['pose_vqvae_max_num_keyframes'], probability)
    executed_history = None
    if context_policy == "played_context4":
        if shifted_local.shape[1] < 4:
            raise ValueError("played_context4 requires at least four motion frames")
        known = known.clone()
        target = target.clone()
        known[:, :4] = True
        target[:, :4] = shifted_local[:, :4]
        with torch.no_grad():
            executed_history = model.local_motion_rep.inverse(
                shifted_local,
                is_normalized=True,
                joint_positions_from="global_rot_data",
                return_all=True,
            )["posed_joints"][:, :4].detach()
    net = model.pose_net
    net.encoder.eval()
    net.quantizer.eval()
    with torch.no_grad():
        latent = net.encoder(net.extract_feature(local, net.encoder_input_feature_mode).permute(0, 2, 1))
        _, commit, perplexity = net.quantizer(latent)
        indices = net.quantizer.forward_into_idx(latent, fetch_overall_indices=False)
    external = extract_feature_from_motion_rep(shifted_local, model.local_motion_rep,
                                              net.decoder_external_cond_feature_mode)
    prediction = net.forward_decoder(indices, target_cond=target, has_target_cond=known,
        external_cond=external, use_overall_indices=False)
    prediction.update(l_commit=commit, perplexity=perplexity)
    loss_batch = {'local_motions': shifted_local, 'global_motions': shifted_global}
    if executed_history is not None:
        loss_batch['executed_history_positions'] = executed_history
    losses = model.loss(loss_batch, prediction, None)
    return losses


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--context-policy', choices=CONTEXT_POLICIES, default='native_random')
    return parser


def _context_policy_metadata(model):
    context_policy = model.args.get('codec_context_policy')
    if context_policy not in CONTEXT_POLICIES:
        raise ValueError("Tokenizer model must declare a supported codec_context_policy")
    return {
        'context_policy': context_policy,
        'observed_context_frames': 4 if context_policy == 'played_context4' else 0,
        'context_policy_note': (
            'explicit online-context adaptation, not unchanged paper training distribution'
            if context_policy == 'played_context4'
            else 'native random-keyframe training distribution'
        ),
    }


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.output.exists() or args.steps < 1 or args.batch_size != 256:
        parser.error('Use a new output, positive updates and paper batch256')
    gpu = select_gpu(args.gpu)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    inference, bundle = load_terrain_bundle(args.bundle, 'cpu')
    apply_native_matmul_precision(OmegaConf.create(bundle['tokenizer_config']))
    manifest = json.loads((args.dataset / 'native_manifest.json').read_text())
    if sha256(args.dataset / 'native_manifest.json') != bundle['metadata']['dataset_manifest_sha256']:
        raise ValueError('Decoder dataset differs from the bound PARC bundle')
    verify_training_source_policy(args.dataset, manifest, parc_only=True)
    model = build_codec_height_tokenizer(inference.motion_rep, inference._vqvae_pose_model,
                                         OmegaConf.create(bundle['tokenizer_config'])).to('cuda')
    model.args['codec_context_policy'] = args.context_policy
    context_metadata = _context_policy_metadata(model)
    model.train()
    model.pose_net.encoder.eval()
    model.pose_net.quantizer.eval()
    frozen_before = {'encoder': state_digest(model.pose_net.encoder), 'quantizer': state_digest(model.pose_net.quantizer)}
    data = NativeMBMotionDataset(args.dataset, 'train', min_frames=65, max_frames=65,
                                cache_max_bytes=2 * 1024**3)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_native_batch, drop_last=True, persistent_workers=args.workers > 0)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    scheduler = WarmupCosineScheduler(optimizer, num_warmup_steps=10000, num_training_steps=2000000,
                                    final_lr=2e-6, last_epoch=-1)
    args.output.mkdir(parents=True)
    contract = {'protocol': CODEC_HEIGHT_CONDITIONING, 'source_bundle_sha256': sha256(args.bundle),
        'dataset_manifest_sha256': bundle['metadata']['dataset_manifest_sha256'], 'parc_only': True,
        'frozen': frozen_before, 'optimizer': 'fresh paper Adam5e-5 warmup10k cosine2M',
        'reason_fresh_optimizer': 'external skip projection shape adds height columns',
        'updates': args.steps, 'gpu_at_launch': gpu, 'seed': args.seed,
        **context_metadata,
        'translation_augmentation_m': [-1., 1.], 'keyframe_curriculum_base_update': 50000,
        'losses': 'original MotionVQVAEModel.loss and unchanged coefficients',
        'matmul_precision': torch.get_float32_matmul_precision(),
        'runtime_ground_truth_root_or_pose': False, 'status': 'running'}
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2))
    iterator, rows = iter(loader), []
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model._trainer = SimpleNamespace(global_step=50000 + step)
        optimizer.zero_grad(set_to_none=True)
        losses = decoder_step(
            model, batch, context_policy=context_metadata['context_policy']
        )
        loss = losses['loss']
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite decoder adaptation loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], .5)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step + 1 == args.steps:
            row = {'update': step + 1, **{k: float(v) for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1}}
            rows.append(row)
            print(json.dumps(row), flush=True)
    frozen_after = {'encoder': state_digest(model.pose_net.encoder), 'quantizer': state_digest(model.pose_net.quantizer)}
    if frozen_before != frozen_after:
        raise RuntimeError('Fixed encoder/codebook changed during decoder-only training')
    contract.update(status='completed', frozen_after=frozen_after, quality_approved=False)
    torch.save({'protocol': CODEC_HEIGHT_CONDITIONING, 'tokenizer_state': model.pose_net.cpu().state_dict(),
        'tokenizer_config': OmegaConf.to_container(enable_config(model, bundle), resolve=False),
        'frozen': frozen_after, 'contract': contract}, args.output / 'decoder.pt')
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2))
    (args.output / 'training.json').write_text(json.dumps(rows, indent=2))


def enable_config(model, bundle):
    from motionbricks.data.native_mb_codec_height import enable_codec_height_conditioning
    context_policy = _context_policy_metadata(model)['context_policy']
    config = enable_codec_height_conditioning(bundle['tokenizer_config'])
    config['model']['args']['codec_context_policy'] = context_policy
    return OmegaConf.create(config)


if __name__ == '__main__':
    main()
