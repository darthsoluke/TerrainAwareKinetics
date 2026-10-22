"""Bind an adapted decoder to unchanged Root/Pose and fixed token semantics."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb_control import state_digest
from scripts.train_native_mb import sha256
from motionbricks.data.native_mb_codec_height import CODEC_HEIGHT_CONDITIONING


def compose(source, decoder_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new composite bundle')
    inference, original = load_terrain_bundle(source, 'cpu')
    adapter = torch.load(decoder_path, map_location='cpu', weights_only=False)
    if (adapter.get('protocol') != CODEC_HEIGHT_CONDITIONING or adapter['contract'].get('status') != 'completed'
            or adapter['contract']['source_bundle_sha256'] != sha256(source)
            or adapter['contract']['dataset_manifest_sha256'] != original['metadata']['dataset_manifest_sha256']):
        raise ValueError('Decoder adapter does not match the completed source pipeline')
    if adapter['frozen'] != {'encoder': state_digest(inference._vqvae_pose_model.encoder),
                            'quantizer': state_digest(inference._vqvae_pose_model.quantizer)}:
        raise ValueError('Pose encoder or codebook semantics changed')
    context_policy = adapter['contract'].get('context_policy', 'native_random')
    if (context_policy not in {'native_random', 'played_context4'}
            or adapter['tokenizer_config']['model']['args'].get('codec_context_policy', 'native_random')
               != context_policy):
        raise ValueError('Decoder artifact and training context policies differ')
    bundle = deepcopy(original)
    bundle['tokenizer_state'] = adapter['tokenizer_state']
    bundle['tokenizer_config'] = adapter['tokenizer_config']
    # A normal Pose checkpoint does not register the supporting decoder. Bind it
    # explicitly; reject unexpected registered frozen-network layouts.
    if any(key.startswith('pose_net.') for key in bundle['pose_state']):
        raise ValueError('Unexpected registered Pose codec; explicit state conversion required')
    bundle['pose_config']['model']['args']['codec_height_conditioning'] = CODEC_HEIGHT_CONDITIONING
    bundle['pose_config']['model']['args']['codec_context_policy'] = context_policy
    bundle['metadata']['codec_height_adapter'] = {
        'protocol': CODEC_HEIGHT_CONDITIONING, 'decoder_sha256': sha256(decoder_path),
        'source_bundle_sha256': sha256(source), 'frozen': adapter['frozen'],
        'source_tokenizer_checkpoint_sha256': original['metadata']['tokenizer_checkpoint_sha256'],
        'updates': adapter['contract']['updates'], 'context_policy': context_policy,
        'quality_approved': False}
    # State digest uses the same exact tensor representation as portable loader.
    from motionbricks.data.native_mb_models import build_native_model
    from omegaconf import OmegaConf
    codec = build_native_model('tokenizer', inference.motion_rep, OmegaConf.create(bundle['tokenizer_config'])).pose_net
    codec.load_state_dict(bundle['tokenizer_state'], strict=True)
    bundle['tokenizer_state_sha256'] = state_digest(codec)
    bundle['quality_approved'] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    # No future reference motion enters the exported artifact.
    load_terrain_bundle(output, 'cpu')
    receipt = {'sha256': sha256(output), 'bytes': output.stat().st_size,
               'metadata': bundle['metadata'], 'quality_approved': False}
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--decoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(compose(args.source, args.decoder, args.output), indent=2))


if __name__ == '__main__':
    main()
