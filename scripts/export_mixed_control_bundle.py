"""Bind the completed mixed-data native MB stages into one inference artifact."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_models import build_native_model
from motionbricks.data.mixed_control_training import ADAPTATION, MIXED_BUNDLE_SCHEMA, validate_source_policy
from motionbricks.data.native_mb_terrain_protocol import INTENT_RANGE_POLICY
from motionbricks.data.native_mb_online_conditions import online_protocol
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest
from scripts.train_mixed_control_stage import load_stage, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'dataset', 'runs', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new inference artifact')
    torch.set_num_threads(4)
    _, source = load_terrain_bundle(args.source, 'cpu')
    manifest = json.loads((args.dataset / 'native_manifest.json').read_text())
    validate_source_policy(manifest['source_policy'])
    dataset_hash = sha256(args.dataset / 'native_manifest.json')
    rep = build_native_representation(args.dataset / 'stats')
    codec, tc, tr = load_stage(args.runs / 'tokenizer', 'tokenizer', rep)
    root, rc, rr = load_stage(args.runs / 'root', 'root', rep)
    pc = json.loads((args.runs / 'pose/run_contract.json').read_text())
    pr = json.loads((args.runs / 'pose/training_result.json').read_text())
    pose_path = args.runs / 'pose' / pr.get('checkpoint_relative', 'checkpoints/final.ckpt')
    if sha256(pose_path) != pr['checkpoint_sha256']:
        raise ValueError('Pose checkpoint changed')
    if any(row['dataset_manifest_sha256'] != dataset_hash for row in (tc, rc, pc)):
        raise ValueError('Stages used different datasets')
    configs = {stage: OmegaConf.load(args.runs / stage / 'native_config.yaml')
               for stage in ('tokenizer', 'root', 'pose')}
    pose = build_native_model('pose', rep, configs['pose'], tokenizer=codec)
    pose.load_state_dict(torch.load(pose_path, map_location='cpu', weights_only=False)['state_dict'], strict=True)
    dependency = pc['dependencies']
    if (dependency['tokenizer_state_sha256'] != state_digest(codec.pose_net)
            or dependency['root_state_sha256'] != state_digest(root)
            or dependency['root_checkpoint_sha256'] != rr['checkpoint_sha256']):
        raise ValueError('Pose did not use these frozen stage tensors')
    bundle = deepcopy(source)
    bundle['schema'] = MIXED_BUNDLE_SCHEMA
    bundle['protocol'] = online_protocol(configs['root'].model.args.online_conditioning)
    bundle['tokenizer_state'] = codec.pose_net.state_dict()
    bundle['tokenizer_state_sha256'] = state_digest(codec.pose_net)
    for stage, model in (('root', root), ('pose', pose)):
        bundle[stage + '_state'] = model.state_dict()
        bundle[stage + '_backbone_sha256'] = state_digest(model.backbone_net)
    for stage, config in configs.items():
        bundle[stage + '_config'] = OmegaConf.to_container(config, resolve=False)
    metadata = bundle['metadata']
    metadata.update(parc_only=False, source_policy=manifest['source_policy'],
                    observation_range_policy=INTENT_RANGE_POLICY, dataset_manifest_sha256=dataset_hash,
                    root_step=rr.get('selected_step', rr['optimizer_updates']),
                    pose_step=pr.get('selected_step', pr['optimizer_updates']),
                    tokenizer_checkpoint_sha256=tr['checkpoint_sha256'],
                    root_checkpoint_sha256=rr['checkpoint_sha256'], pose_checkpoint_sha256=pr['checkpoint_sha256'])
    metadata['pose_root_dependency'] = {'checkpoint_sha256': rr['checkpoint_sha256'],
        'state_digest': dependency['root_state_sha256'], 'end_state_digest': dependency['root_state_sha256'],
        'dataset_manifest_sha256': dataset_hash, 'frozen': True}
    metadata['matched_stage_adaptation'] = {'version': ADAPTATION, 'status': 'completed',
        'vertical_frame': configs['pose'].model.args.vertical_frame, 'context_policy': 'played_context4',
        'tokenizer_state_sha256': bundle['tokenizer_state_sha256'],
        'pose_tokenizer_state_sha256': dependency['tokenizer_state_sha256'],
        'source_bundle_sha256': sha256(args.source), 'stages': {'tokenizer': tr, 'root': rr, 'pose': pr},
        'quality_approved': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    load_terrain_bundle(args.output, 'cpu')
    write_json(args.output.with_suffix('.json'), {'sha256': sha256(args.output),
        'dataset_manifest_sha256': dataset_hash, 'source_policy': manifest['source_policy'],
        'quality_approved': False, 'inference_location': 'local PC', 'post_processing': None})
    print(json.dumps({'bundle': str(args.output), 'sha256': sha256(args.output)}), flush=True)


if __name__ == '__main__':
    main()
