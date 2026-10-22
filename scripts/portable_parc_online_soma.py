"""Verified online SOMA exports; a four-frame seed, never a future clip."""

import argparse
import json
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.native_mb_online_conditions import ONLINE_PROTOCOL, online_protocol
from motionbricks.data.native_mb_data import NativeMBMotionDataset
from scripts.portable_native_mb_soma import SCHEMA as MODEL_SCHEMA, instantiate_bundle
from scripts.train_native_mb_control import load_control_checkpoint, state_digest
from scripts.evaluate_parc_native_terrain import _dataset_windows

SCHEMA = 'motionbricks.parc_online_soma_bundle.v1'


def world_history(rep, item, start):
    history = rep.unnormalize(item['motion'][start:start + 4])[None]
    world_root = item['terrain_root_world'][start:start + 4][None]
    heading = torch.atan2(world_root[:, 0, 4], world_root[:, 0, 3])
    result = rep.change_first_heading(history, heading, False, False)
    result[..., [0, 2]] += world_root[:, :1, [0, 2]]
    if (result.shape[1] != 4 or not torch.isfinite(result).all()
            or not torch.allclose(result[..., :5], world_root, atol=1e-5, rtol=0)):
        raise ValueError('Four observed history frames did not restore to the paired terrain world')
    return result


def load_terrain_bundle(path, device='cuda:0'):
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    protocol = online_protocol(bundle['root_config']['model']['args'].get('online_conditioning'))
    from motionbricks.data.mixed_control_training import MIXED_BUNDLE_SCHEMA, validate_source_policy
    if (bundle.get('schema') not in (SCHEMA, MIXED_BUNDLE_SCHEMA) or bundle.get('protocol') != protocol
            or bundle.get('quality_approved') is not False or bundle.get('post_processing') is not None
            or 'reference_clips' in bundle):
        raise ValueError('Expected an experimental online SOMA bundle without future references')
    metadata = bundle['metadata']
    mixed = bundle['schema'] == MIXED_BUNDLE_SCHEMA
    if mixed:
        validate_source_policy(metadata.get('source_policy'))
        if metadata.get('parc_only') is not False:
            raise ValueError('Mixed-data bundle cannot claim PARC-only training')
    range_policy = bundle['root_config']['model']['args'].get('observation_range_policy')
    if range_policy != bundle['pose_config']['model']['args'].get('observation_range_policy'):
        raise ValueError('Root and Pose observation range policies differ')
    from motionbricks.data.native_mb_terrain_protocol import terrain_observation_config
    terrain_observation_config(range_policy)
    if range_policy and metadata.get('observation_range_policy') != range_policy:
        raise ValueError('Adaptive observation range lacks a matching training artifact binding')
    from motionbricks.data.native_mb_vertical_frame import validate_vertical_frame
    stage_frames = [validate_vertical_frame(bundle[stage + '_config']['model']['args'].get('vertical_frame'))
                    for stage in ('tokenizer', 'root', 'pose')]
    if len(set(stage_frames)) != 1:
        raise ValueError('Tokenizer, Root and Pose must share one vertical frame')
    matched = metadata.get('matched_stage_adaptation')
    if stage_frames[0] or matched:
        if (not isinstance(matched, dict) or matched.get('status') != 'completed'
                or matched.get('vertical_frame') != stage_frames[0]
                or matched.get('tokenizer_state_sha256') != bundle['tokenizer_state_sha256']
                or matched.get('pose_tokenizer_state_sha256') != bundle['tokenizer_state_sha256']
                or bundle['pose_config']['model']['args'].get('matched_stage_adaptation') != matched.get('version')):
            raise ValueError('Vertical-frame stages lack a completed matched training binding')
        contexts = [bundle[stage + '_config']['model']['args'].get('codec_context_policy')
                    for stage in ('tokenizer', 'pose')]
        if contexts != [matched.get('context_policy')] * 2:
            raise ValueError('Matched stage decoder context policy differs from its training')
    codec_protocol = bundle['pose_config']['model']['args'].get('codec_height_conditioning')
    if codec_protocol:
        from motionbricks.data.native_mb_codec_height import CODEC_HEIGHT_CONDITIONING
        adapter = metadata.get('codec_height_adapter')
        if (codec_protocol != CODEC_HEIGHT_CONDITIONING or not isinstance(adapter, dict)
                or adapter.get('protocol') != codec_protocol
                or set(adapter.get('frozen', {})) != {'encoder', 'quantizer'}
                or adapter.get('source_tokenizer_checkpoint_sha256') != metadata['tokenizer_checkpoint_sha256']):
            raise ValueError('Dense height decoder lacks a fixed-codebook compatibility binding')
        context_policy = adapter.get('context_policy', 'native_random')
        if (context_policy not in {'native_random', 'played_context4'}
                or bundle['pose_config']['model']['args'].get('codec_context_policy', 'native_random')
                   != context_policy
                or bundle['tokenizer_config']['model']['args'].get('codec_context_policy', 'native_random')
                   != context_policy):
            raise ValueError('Decoder playback context does not match its training artifact')
    elif not matched and bundle['pose_config']['model']['args'].get('codec_context_policy', 'native_random') != 'native_random':
        raise ValueError('Played decoder context requires an explicitly bound adapted artifact')
    if ((not mixed and metadata.get('parc_only') is not True)
            or metadata['root_checkpoint_sha256'] != metadata['pose_root_dependency']['checkpoint_sha256']
            or metadata['pose_root_dependency'].get('frozen') is not True
            or metadata['pose_root_dependency']['state_digest']
               != metadata['pose_root_dependency']['end_state_digest']):
        raise ValueError('Portable online stages have an incompatible Root dependency')
    for stage in ('root', 'pose'):
        if bundle[stage + '_config']['model']['args'].get('online_conditioning') != protocol['version']:
            raise ValueError('Portable model does not have the online condition curriculum')
    initial = bundle['initial_history']
    scene = bundle['scene']
    if initial.ndim != 3 or initial.shape[:2] != (1, 4) or not torch.isfinite(initial).all():
        raise ValueError('Online seed must contain exactly four finite observed frames')
    if set(scene) != {'heightfield_y_m', 'min_point_world_xyz_m', 'dx_m'}:
        raise ValueError('Portable scene contains undeclared inputs')
    heights, origin, spacing = (torch.as_tensor(scene[key]) for key in
                                ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m'))
    if (heights.ndim != 2 or min(heights.shape) < 2 or origin.shape != (3,)
            or spacing.numel() != 1 or float(spacing) <= 0
            or not all(torch.isfinite(value).all() for value in (heights, origin, spacing))):
        raise ValueError('Invalid measured scene geometry')
    stats = torch.nn.Module()
    stats.register_buffer('mean', bundle['stats_mean'])
    stats.register_buffer('std', bundle['stats_std'])
    if state_digest(stats) != bundle['stats_tensor_sha256']:
        raise ValueError('Portable normalization statistics changed')
    inference, _ = instantiate_bundle(dict(bundle, schema=MODEL_SCHEMA), device)
    if state_digest(inference._root_model) != metadata['pose_root_dependency']['state_digest']:
        raise ValueError('Loaded Root tensors differ from the frozen Pose training condition')
    if codec_protocol:
        for component in ('encoder', 'quantizer'):
            if state_digest(getattr(inference._vqvae_pose_model, component)) != adapter['frozen'][component]:
                raise ValueError('Adapted decoder changed the Pose token vocabulary')
    return inference, bundle


def export(root_run, pose_run, output, window_index=0):
    output = Path(output)
    if output.exists():
        raise ValueError('Refuse to overwrite a portable experiment')
    root, rc, rr = load_control_checkpoint(root_run, 'cpu')
    pose, pc, pr = load_control_checkpoint(pose_run, 'cpu')
    for model in (root, pose):
        if model.args.get('online_conditioning') != ONLINE_PROTOCOL['version']:
            raise ValueError('Export requires matching trained online Root and Pose')
    dependency = pc['online_contract']['root_dependency']
    if (rc['dataset_manifest_sha256'] != pc['dataset_manifest_sha256']
            or rr['final_checkpoint_sha256'] != dependency['checkpoint_sha256']
            or state_digest(root) != dependency['state_digest']):
        raise ValueError('Exported stages do not match the trained prediction dependency')
    dataset = Path(pc['dataset'])
    _, windows, _, _, _, _, _, _ = _dataset_windows(dataset, 8)
    if not 0 <= window_index < len(windows):
        raise ValueError('Invalid fixed held-out seed selection')
    window = windows[window_index]
    loader = NativeMBMotionDataset(dataset, 'validation', max_frames=10**9, min_frames=65,
                                  require_terrain=True, require_online_terrain=True)
    index = next(i for i, row in enumerate(loader.records) if row['shard'] == window['shard'])
    item = loader[index]
    rep = pose.global_motion_rep
    seed = world_history(rep, item, window['start'])
    stats = torch.nn.Module()
    stats.register_buffer('mean', pose.motion_rep.dual_rep.stats.mean.cpu())
    stats.register_buffer('std', pose.motion_rep.dual_rep.stats.std.cpu())
    tokenizer_run = Path(pc['tokenizer']['run'])
    codec = pose.supporting_nets['pose_net']
    bundle = {'schema': SCHEMA, 'protocol': ONLINE_PROTOCOL, 'fps': 30,
              'initial_history': seed.cpu(),
              'scene': {key: item['terrain_scene'][key].cpu() for key in
                        ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')},
              'seed_provenance': dict(window, role='four observed frames only, no future clip'),
              'bone_names': list(rep.skeleton.bone_order_names),
              'neutral_joints': rep.skeleton.neutral_joints.cpu(),
              'stats_mean': stats.mean, 'stats_std': stats.std,
              'stats_tensor_sha256': state_digest(stats),
              'tokenizer_state': codec.state_dict(), 'tokenizer_state_sha256': state_digest(codec),
              'tokenizer_config': OmegaConf.to_container(OmegaConf.load(tokenizer_run / 'native_config.yaml'), resolve=False),
              'metadata': {'parc_only': True, 'root_step': rr['optimizer_updates'],
                  'pose_step': pr['optimizer_updates'], 'root_checkpoint_sha256': rr['final_checkpoint_sha256'],
                  'pose_checkpoint_sha256': pr['final_checkpoint_sha256'],
                  'tokenizer_checkpoint_sha256': pc['tokenizer']['checkpoint_sha256'],
                  'dataset_manifest_sha256': pc['dataset_manifest_sha256'],
                  'pose_root_dependency': dependency},
              'quality_approved': False, 'post_processing': None}
    for stage, run, model in (('root', root_run, root), ('pose', pose_run, pose)):
        bundle[stage + '_state'] = model.state_dict()
        bundle[stage + '_backbone_sha256'] = state_digest(model.backbone_net)
        bundle[stage + '_config'] = OmegaConf.to_container(OmegaConf.load(Path(run) / 'native_config.yaml'), resolve=False)
    # Verify strict inference reconstruction before publishing the artifact.
    _, reconstructed = instantiate_bundle(dict(bundle, schema=MODEL_SCHEMA), 'cpu')
    assert reconstructed['initial_history'].shape[1] == 4
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    load_terrain_bundle(output, 'cpu')
    receipt = {'schema': SCHEMA, 'file': output.name, 'sha256': file_sha256(output),
               'bytes': output.stat().st_size, 'metadata': bundle['metadata'],
               'seed_provenance': bundle['seed_provenance'], 'quality_approved': False}
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-run', required=True, type=Path)
    parser.add_argument('--pose-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--window-index', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(export(args.root_run, args.pose_run, args.output, args.window_index), indent=2))


if __name__ == '__main__':
    main()
