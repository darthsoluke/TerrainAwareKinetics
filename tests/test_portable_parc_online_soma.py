import copy

import pytest
import torch
from omegaconf import OmegaConf

from test_native_mb_stage_training import _motion_and_representation
from test_native_mb_terrain_integration import small_config
from motionbricks.data.native_mb_models import build_native_model
from motionbricks.data.native_mb_online_conditions import enable_online_conditioning
from scripts import portable_parc_online_soma as portable


@pytest.fixture
def portable_online(tmp_path, monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    rep, batch = _motion_and_representation(frame_count=65)
    configs = {stage: small_config(stage) for stage in ('root', 'pose', 'tokenizer')}
    codec = build_native_model('tokenizer', rep, configs['tokenizer'])
    root = build_native_model('root', rep, enable_online_conditioning(configs['root'])).eval()
    pose = build_native_model('pose', rep, enable_online_conditioning(configs['pose']), tokenizer=codec).eval()
    pose._vqvae_model_loaded = True
    dependency = {'checkpoint_sha256': 'root-test', 'state_digest': portable.state_digest(root),
                  'end_state_digest': portable.state_digest(root), 'frozen': True}
    for stage in configs:
        (tmp_path / stage).mkdir()
        config = configs[stage] if stage == 'tokenizer' else enable_online_conditioning(configs[stage])
        OmegaConf.save(config, tmp_path / stage / 'native_config.yaml')
    contract = {'dataset': str(tmp_path / 'dataset'), 'dataset_manifest_sha256': 'dataset-test',
                'online_contract': {'root_dependency': dependency},
                'tokenizer': {'run': str(tmp_path / 'tokenizer'), 'checkpoint_sha256': 'codec-test'}}
    monkeypatch.setattr(portable, 'load_control_checkpoint', lambda run, device:
        (root if run.name == 'root' else pose, contract,
         {'final_checkpoint_sha256': 'root-test' if run.name == 'root' else 'pose-test',
          'optimizer_updates': 1000}))
    raw = rep.unnormalize(batch['motion'])[0]
    item = {'motion': batch['motion'][0], 'terrain_root_world': raw[:, :5],
            'terrain_scene': {'heightfield_y_m': torch.zeros(81, 81),
                              'min_point_world_xyz_m': torch.tensor([-4., 0., -4.]),
                              'dx_m': torch.tensor(.1)}}

    class Dataset:
        records = [{'shard': 'validation.npz'}]

        def __init__(self, *args, **kwargs):
            pass

        def __getitem__(self, index):
            return item

    window = {'shard': 'validation.npz', 'start': 0, 'split': 'validation'}
    monkeypatch.setattr(portable, 'NativeMBMotionDataset', Dataset)
    monkeypatch.setattr(portable, '_dataset_windows', lambda *args:
                        (None, [window], None, None, None, None, None, None))
    path = tmp_path / 'online.pt'
    portable.export(tmp_path / 'root', tmp_path / 'pose', path)
    bundle = torch.load(path, weights_only=False)
    yield path, bundle, rep, item
    torch.set_num_threads(previous)


def test_portable_online_roundtrip_preserves_predictions_and_no_future_clip(portable_online):
    path, bundle, rep, item = portable_online
    infer, loaded = portable.load_terrain_bundle(path, 'cpu')
    assert 'reference_clips' not in loaded
    assert loaded['initial_history'].shape[1] == 4
    assert torch.allclose(loaded['initial_history'][..., :5], item['terrain_root_world'][:4][None])
    assert torch.equal(loaded['stats_mean'], rep.dual_rep.stats.mean)
    assert portable.state_digest(infer._root_model) == bundle['metadata']['pose_root_dependency']['state_digest']


@pytest.mark.parametrize('mutation', ('future_history', 'root_dependency', 'statistics', 'reference'))
def test_portable_online_rejects_changed_or_future_conditions(portable_online, mutation):
    path, source, _, _ = portable_online
    bundle = copy.deepcopy(source)
    if mutation == 'future_history':
        bundle['initial_history'] = bundle['initial_history'].repeat(1, 2, 1)
    elif mutation == 'root_dependency':
        bundle['metadata']['pose_root_dependency']['checkpoint_sha256'] = 'wrong-root'
    elif mutation == 'statistics':
        bundle['stats_mean'][0] += 1
    else:
        bundle['reference_clips'] = {'future': torch.zeros(1)}
    bad = path.with_name('bad.pt')
    torch.save(bundle, bad)
    with pytest.raises(ValueError):
        portable.load_terrain_bundle(bad, 'cpu')


def test_played_context_decoder_binding_matches_training_and_rejects_policy_changes(portable_online):
    from motionbricks.data.native_mb_codec_height import (
        build_codec_height_tokenizer, enable_codec_height_conditioning, CODEC_HEIGHT_CONDITIONING,
    )
    from scripts.compose_parc_height_bundle import compose
    from scripts.train_native_mb import sha256

    path, bundle, _, _ = portable_online
    inference, _ = portable.load_terrain_bundle(path, 'cpu')
    config = enable_codec_height_conditioning(bundle['tokenizer_config'])
    config['model']['args']['codec_context_policy'] = 'played_context4'
    tokenizer = build_codec_height_tokenizer(inference.motion_rep, inference._vqvae_pose_model, config)
    frozen = {key: portable.state_digest(getattr(tokenizer.pose_net, key))
              for key in ('encoder', 'quantizer')}
    adapter = {'protocol': CODEC_HEIGHT_CONDITIONING, 'tokenizer_config': config,
               'tokenizer_state': tokenizer.pose_net.state_dict(), 'frozen': frozen,
               'contract': {'status': 'completed', 'source_bundle_sha256': sha256(path),
                            'dataset_manifest_sha256': bundle['metadata']['dataset_manifest_sha256'],
                            'context_policy': 'played_context4', 'updates': 1}}
    decoder_path, output = path.with_name('decoder.pt'), path.with_name('adapted.pt')
    torch.save(adapter, decoder_path)
    compose(path, decoder_path, output)
    _, loaded = portable.load_terrain_bundle(output, 'cpu')
    assert loaded['pose_config']['model']['args']['codec_context_policy'] == 'played_context4'
    assert loaded['metadata']['codec_height_adapter']['context_policy'] == 'played_context4'
    assert not loaded['quality_approved']
    loaded['pose_config']['model']['args']['codec_context_policy'] = 'native_random'
    bad = path.with_name('bad-context.pt')
    torch.save(loaded, bad)
    with pytest.raises(ValueError, match='playback context'):
        portable.load_terrain_bundle(bad, 'cpu')
    adapter['tokenizer_config']['model']['args']['codec_context_policy'] = 'native_random'
    torch.save(adapter, decoder_path)
    with pytest.raises(ValueError, match='context policies differ'):
        compose(path, decoder_path, path.with_name('bad-compose.pt'))


@pytest.mark.parametrize('case', ['only_root', 'no_training_binding', 'stale_pose_dictionary'])
def test_rejects_mixed_or_unbound_vertical_representation(portable_online, case):
    from motionbricks.data.native_mb_vertical_frame import VERTICAL_FRAME
    path, original, _, _ = portable_online
    bundle = copy.deepcopy(original)
    stages = ('root',) if case == 'only_root' else ('root', 'pose', 'tokenizer')
    for stage in stages:
        bundle[stage + '_config']['model']['args']['vertical_frame'] = VERTICAL_FRAME
    if case == 'stale_pose_dictionary':
        bundle['metadata']['matched_stage_adaptation'] = {
            'version': 'matched-parc-frame-v1', 'status': 'completed', 'vertical_frame': VERTICAL_FRAME,
            'tokenizer_state_sha256': bundle['tokenizer_state_sha256'],
            'pose_tokenizer_state_sha256': 'different-token-vocabulary'}
    bad = path.with_name(case + '.pt')
    torch.save(bundle, bad)
    with pytest.raises(ValueError, match='vertical frame|matched training binding'):
        portable.load_terrain_bundle(bad, 'cpu')
