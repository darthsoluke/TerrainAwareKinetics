"""Inspect actual native codec temporal dependencies; never alter actor deployment."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.vqvae.neural_modules.resnet import ResConv1DBlock
from scripts.pilot_joint_sdf_full_context import command, load_cases
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def temporal_support(module, support):
    """Boolean input-frame dependencies, including residual paths and norm scope."""
    if isinstance(module, nn.Conv1d):
        kernel, stride, padding, dilation = (value[0] for value in
                                             (module.kernel_size, module.stride, module.padding, module.dilation))
        length = (support.shape[1] + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
        output = torch.zeros(support.shape[0], length, dtype=torch.bool)
        for position in range(length):
            ids = [position * stride - padding + tap * dilation for tap in range(kernel)]
            ids = [index for index in ids if 0 <= index < support.shape[1]]
            output[:, position] = support[:, ids].any(1)
        return output
    if isinstance(module, ResConv1DBlock):
        branch = temporal_support(module.norm1, support)
        branch = temporal_support(module.conv1, branch)
        branch = temporal_support(module.norm2, branch)
        branch = temporal_support(module.conv2, branch)
        return support | branch
    if isinstance(module, nn.GroupNorm) or isinstance(module, nn.BatchNorm1d) and module.training:
        return support.any(1, keepdim=True).expand_as(support).clone()
    if isinstance(module, (nn.Identity, nn.ReLU, nn.GELU, nn.SiLU, nn.LayerNorm, nn.BatchNorm1d)):
        return support
    if isinstance(module, nn.Sequential):
        for child in module:
            support = temporal_support(child, support)
        return support
    if hasattr(module, 'model'):
        return temporal_support(module.model, support)
    raise ValueError('Temporal support rule missing for ' + type(module).__name__)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    bundle_path = Path(saved['contract']['source'])
    if sha256(bundle_path) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source identity mismatch')
    inference, bundle = load_terrain_bundle(bundle_path, 'cuda:0')
    model = JointSDFController(inference, bundle)
    model.load_state_dict(saved['model_state'], strict=True)
    model.requires_grad_(False)
    model.eval()
    cases = load_cases(args.cases, Path(saved['contract']['dataset']))
    codec = model.codec
    theoretical = temporal_support(codec.encoder, torch.eye(32, dtype=torch.bool))
    rows = []
    for case in cases[:3]:
        world = case['world'][None].cuda()
        history = world[:, :4]
        task = SoftCommandState().from_controls(history, command(case, 0), 8)
        captured = []

        def snapshot(_module, inputs, kwargs):
            captured.append((tuple(value.detach().clone() if torch.is_tensor(value) else value for value in inputs),
                             {key: value.detach().clone() if torch.is_tensor(value) else value for key, value in kwargs.items()}))

        handle = codec.decoder.register_forward_pre_hook(snapshot, with_kwargs=True)
        with torch.no_grad():
            generated = model(history, task, [case['scene']], source=world[:, :33])
            reference_local = model.last_local.detach().clone()
            local = generated['target']['local_motions'].detach()
            alternate = model(history, task, [case['scene']], source=world[:, 8:41])['target']['local_motions'].detach()
            torch.testing.assert_close(generated['world'], model(history, task, [case['scene']])['world'], atol=0, rtol=0)
        handle.remove()
        inputs, kwargs = captured[-1]
        encoded_input = codec.extract_feature(local, codec.encoder_input_feature_mode).permute(0, 2, 1).detach().requires_grad_(True)
        encoded = codec.encoder(encoded_input)
        empirical = []
        for index in range(encoded.shape[-1]):
            gradient = torch.autograd.grad(encoded[:, :, index].square().mean(), encoded_input, retain_graph=True)[0]
            values = gradient.square().sum((0, 1)).sqrt()
            empirical.append({'token': index, 'input_frame_gradient_norms': values.detach().cpu().tolist(),
                              'history_gradient_fraction': float(values[:4].sum() / values.sum().clamp_min(1e-30)),
                              'future_frames_with_gradient_over1e_8': int((values[4:] > 1e-8).sum())})
        perturbations = []
        with torch.no_grad():
            baseline_tokens = codec.encode_into_idx(local, fetch_overall_indices=False)
            baseline_latent = codec.encoder(codec.extract_feature(local, codec.encoder_input_feature_mode).permute(0, 2, 1))
            for name, start, end in (('late_future_only', 24, 32), ('history_only', 0, 4)):
                changed = local.clone()
                changed[:, start:end] = alternate[:, start:end]
                tokens = codec.encode_into_idx(changed, fetch_overall_indices=False)
                latent = codec.encoder(codec.extract_feature(changed, codec.encoder_input_feature_mode).permute(0, 2, 1))
                perturbations.append({'changed_input': name, 'frame_interval': [start, end],
                                      'changed_heads_per_token': (tokens != baseline_tokens).sum((0, 2)).cpu().tolist(),
                                      'latent_difference_l2_per_token': (latent - baseline_latent).square().sum((0, 1)).sqrt().cpu().tolist()})
        decoder_input = inputs[0].detach().clone().requires_grad_(True)
        # Captured input has already passed terrain/SDF hooks. Calling the
        # native forward directly avoids conditioning it a second time.
        output = type(codec.decoder).forward(codec.decoder, decoder_input, *inputs[1:], **kwargs)
        parity = float((output.transpose(1, 2) - reference_local).abs().max())
        torch.testing.assert_close(output.transpose(1, 2), reference_local, atol=1e-6, rtol=1e-5)
        gradient = torch.autograd.grad(output[:, :, 4:12].square().mean(), decoder_input)[0]
        decoder_norm = gradient.square().sum((0, 1)).sqrt()
        row = {'source_id': case['label'], 'token_stride_frames': 2 ** codec._down_t,
               'encoded_tokens': encoded.shape[-1], 'encoder_empirical': empirical,
               'perturbations': perturbations, 'decoder_replay_max_difference': parity,
               'decoder_published4_12_gradient_by_latent_position': decoder_norm.detach().cpu().tolist(),
               'decoder_first_latent_gradient_fraction': float(decoder_norm[0] / decoder_norm.sum().clamp_min(1e-30)),
               'actor_replay_max_difference': 0.}
        rows.append(row)
        write_json(args.output / 'progress.json', {'rows': rows, 'quality_approved': False})
        print(json.dumps({'source_id': case['label'], 'first_token': empirical[0],
                          'perturbations': perturbations, 'decoder_first_latent_gradient_fraction': row['decoder_first_latent_gradient_fraction'],
                          'decoder_replay_max_difference': parity}), flush=True)
    write_json(args.output / 'complete.json', {'rows': rows, 'quality_approved': False,
               'encoder_structural_input_frames_by_token': [torch.nonzero(theoretical[:, index]).flatten().tolist()
                                                           for index in range(theoretical.shape[1])],
               'encoder_configuration': str(codec.encoder),
               'scope': 'Diagnostic perturbations and gradients only; no training or actor input changes. '
                        'Temporal stride is not receptive field. Decoder gradients are w.r.t. latent after '
                        'terrain/SDF conditioning, with original conditions and known-frame masks retained.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
