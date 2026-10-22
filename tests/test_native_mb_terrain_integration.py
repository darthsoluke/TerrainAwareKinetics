"""Actual MB training/inference entry points, with small test-only networks."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from test_native_mb_stage_training import (
    _motion_and_representation, _attach_minimal_training_context,
)
from motionbricks.data.native_mb_data import NativeMBMotionDataset, collate_native_batch
from motionbricks.data.native_mb_models import build_native_model, native_config
from motionbricks.data.native_mb_terrain_protocol import (
    TERRAIN_PROTOCOL, enable_terrain_conditioning, load_flat_backbone_into_terrain,
    validate_terrain_dataset,
)
from motionbricks.data.native_mb_terrain_batch import root_height_condition


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_config(stage, terrain=True):
    config = native_config(stage, 100)
    if stage != 'tokenizer':
        args = config.model.backbone_network.args
        args.n_embd, args.n_head = 32, 4
        if stage == 'root':
            args.n_layers_shared = args.n_layers_root_token = 1
            args.width, args.depth = 32, 1
            config.model.args.prob_off_range_target_token = 0.0
        else:
            args.n_layers = args.pose_token_mlp_num_layers = 1
            args.pose_feat_width, args.root_feat_width, args.token_length_feat_width = 16, 8, 8
        if terrain:
            config = enable_terrain_conditioning(config)
    return config


def terrain_batch(batch):
    batch = deepcopy(batch)
    n, t = batch['motion'].shape[:2]
    points = (torch.rand(n, t, 2, 9, 9, 3) - 0.5).requires_grad_()
    batch.update(terrain_points=points, terrain_valid=torch.ones(n, t, 2, 9, 9, dtype=torch.bool))
    return batch


@pytest.mark.parametrize('stage', ['root', 'pose'])
def test_real_training_step_uses_terrain_and_backpropagates(stage, monkeypatch):
    torch.manual_seed(741)
    np.random.seed(741)
    rep, batch = _motion_and_representation(frame_count=41)
    batch = terrain_batch(batch)
    tokenizer = build_native_model('tokenizer', rep, small_config('tokenizer')) if stage == 'pose' else None
    model = build_native_model(stage, rep, small_config(stage), tokenizer=tokenizer)
    model.args['min_tokens'] = model.args['max_tokens'] = 6
    model.train()
    _attach_minimal_training_context(model)
    seen = {}
    original = model.backbone_net.forward

    def forward(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(model.backbone_net, 'forward', forward)
    # Initial gate receives a gradient; after it opens, encoder weights do too.
    loss = model.training_step(deepcopy(batch), 0)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    gate = model.backbone_net._terrain_cross_attention.gate
    assert gate.grad is not None and torch.isfinite(gate.grad) and gate.grad.abs() > 0
    assert 'terrain_observation' in seen
    assert not seen['terrain_observation']['points'].requires_grad
    if stage == 'pose':
        assert seen['root_height_trajectory'].shape == (2, 24, 1)
        assert torch.equal(seen['root_height_trajectory'][:, 3], torch.zeros(2, 1))
    model.zero_grad(set_to_none=True)
    gate.data.fill_(0.1)
    model.training_step(deepcopy(batch), 1).backward()
    grads = [p.grad for p in model.backbone_net._terrain_encoder.parameters()]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert batch['terrain_points'].grad is None
    if tokenizer is not None:
        assert all(p.grad is None for p in tokenizer.parameters())


def test_missing_scene_fails_in_real_training_path():
    rep, batch = _motion_and_representation()
    model = build_native_model('root', rep, small_config('root'))
    model.args['min_tokens'] = model.args['max_tokens'] = 6
    model.train()
    _attach_minimal_training_context(model)
    with pytest.raises(ValueError, match='missing required terrain'):
        model.training_step(batch, 0)


def test_dataset_crop_and_padding_keep_exact_frame_alignment(tmp_path):
    rep, batch = _motion_and_representation(frame_count=41)
    (tmp_path / 'stats').mkdir()
    np.save(tmp_path / 'stats/mean.npy', rep.dual_rep.stats.mean.numpy())
    np.save(tmp_path / 'stats/std.npy', rep.dual_rep.stats.std.numpy())
    raw = rep.unnormalize(batch['motion'][0]).numpy()
    points = np.broadcast_to(np.arange(41, dtype=np.float32)[:, None, None, None, None],
                             (41, 2, 9, 9, 3)).copy()
    # Put the same frame marker in one feature to check the actual random crop.
    raw[:, 0] = np.arange(41)
    np.savez(tmp_path / 'clip.npz', global_features=raw, terrain_points=points,
             terrain_valid=np.ones((41, 2, 9, 9), dtype=bool))
    (tmp_path / 'accepted.json').write_text(json.dumps([
        {'shard': 'clip.npz', 'split': 'train', 'frames_30hz': 41, 'filename': 'clip'}]))
    dataset = NativeMBMotionDataset(tmp_path, 'train', min_frames=25, max_frames=25, require_terrain=True)
    for _ in range(5):
        item = dataset[0]
        markers = dataset.stats.unnormalize(item['motion'])[:, 0]
        torch.testing.assert_close(markers, item['terrain_points'][:, 0, 0, 0, 0])
    short = {key: value[:17] if torch.is_tensor(value) else value for key, value in item.items()}
    collated = collate_native_batch([item, short])
    assert not collated['terrain_valid'][1, 17:].any()
    torch.testing.assert_close(collated['terrain_points'][0], item['terrain_points'])
    with pytest.raises(ValueError, match='Every terrain-conditioned'):
        collate_native_batch([item, {'motion': item['motion']}])
    invalid = dict(item, terrain_points=item['terrain_points'][:, :, :8],
                   terrain_valid=item['terrain_valid'][:, :, :8])
    with pytest.raises(ValueError, match='geometry or frame alignment'):
        collate_native_batch([invalid])
    np.savez(tmp_path / 'clip.npz', global_features=raw, terrain_points=points[:, :, :8],
             terrain_valid=np.ones((41, 2, 8, 9), dtype=bool))
    with pytest.raises(ValueError, match='align with every motion frame'):
        dataset[0]


def test_flat_warmstart_only_allows_new_terrain_keys_and_preserves_original_config():
    rep, _ = _motion_and_representation()
    flat_config = small_config('root', terrain=False)
    before = deepcopy(flat_config)
    extended_config = enable_terrain_conditioning(flat_config)
    assert flat_config == before
    flat = build_native_model('root', rep, flat_config)
    extended = build_native_model('root', rep, extended_config)
    checkpoint = {'state_dict': flat.state_dict()}
    missing = load_flat_backbone_into_terrain(extended, checkpoint)
    assert missing and all(key.startswith('_terrain_') for key in missing)
    for key, value in flat.backbone_net.state_dict().items():
        assert torch.equal(value, extended.backbone_net.state_dict()[key])
    invalid = deepcopy(checkpoint)
    invalid['state_dict'].pop(next(key for key in invalid['state_dict'] if key.startswith('backbone_net.')))
    with pytest.raises(ValueError, match='match every original'):
        load_flat_backbone_into_terrain(extended, invalid)


def test_terrain_training_gate_rejects_raw_source_and_split_leakage():
    contract = dict(TERRAIN_PROTOCOL)
    manifest = {'terrain_conditioning': contract, 'artifact_sha256': {'retarget_report.json': 'test'}}
    records = [{'source_group': 'base1', 'split': 'train'}]
    with pytest.raises(ValueError, match='accepted SOMA'):
        validate_terrain_dataset(manifest, records)
    contract['acceptance'] = {'retarget_status': 'passed', 'motion_terrain_pairing_status': 'passed',
                              'report': 'retarget_report.json'}
    assert validate_terrain_dataset(manifest, records)['version'] == TERRAIN_PROTOCOL['version']
    missing_height = deepcopy(manifest)
    missing_height['terrain_conditioning'].pop('root_height_condition')
    with pytest.raises(ValueError, match='root_height_condition'):
        validate_terrain_dataset(missing_height, records)
    with pytest.raises(ValueError, match='leaks'):
        validate_terrain_dataset(manifest, records + [{'source_group': 'base1', 'split': 'validation'}])


def test_complete_native_inference_passes_same_current_scene_and_predicted_root_height(monkeypatch):
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
    from motionbricks.motion_backbone.inference.motion_inference import motion_inference

    rep, batch = _motion_and_representation()
    tokenizer = build_native_model('tokenizer', rep, small_config('tokenizer'))
    root = build_native_model('root', rep, small_config('root')).eval()
    pose = build_native_model('pose', rep, small_config('pose'), tokenizer=tokenizer).eval()
    pose._vqvae_model_loaded = True
    infer = motion_inference({'root': root, 'pose': pose}, pose.args, device='cpu')
    raw = rep.unnormalize(batch['motion'])
    local = rep.dual_rep.global_to_local(raw, False, False, lengths=torch.tensor([25]))
    ids = [0, 1, 2, 3, 20, 21, 22, 23]
    global_root = extract_feature_from_motion_rep(raw, rep, 'root')[:, ids]
    local_root = extract_feature_from_motion_rep(local, rep.dual_rep.local_motion_rep, 'root')[:, ids]
    poses = extract_feature_from_motion_rep(local, rep.dual_rep.local_motion_rep,
                                           infer.EXTERNAL_POSE_FEATURE_MODE)[:, ids]
    mask = torch.tensor([[True] * 4 + [False] * 4])
    scene = {'points': torch.zeros(1, 2, 9, 9, 3), 'valid': torch.ones(1, 2, 9, 9, dtype=torch.bool)}
    seen = {}
    root_forward, pose_forward = root.backbone_net.forward, pose.backbone_net.forward

    def capture_root(*args, **kwargs):
        seen['root_scene'] = kwargs['terrain_observation']
        result = root_forward(*args, **kwargs)
        seen['predicted_root'] = result['pred_global_root_values'].clone()
        return result

    def capture_pose(*args, **kwargs):
        seen['pose_scene'] = kwargs['terrain_observation']
        seen['height'] = kwargs['root_height_trajectory']
        return pose_forward(*args, **kwargs)

    monkeypatch.setattr(root.backbone_net, 'forward', capture_root)
    monkeypatch.setattr(pose.backbone_net, 'forward', capture_pose)
    with torch.inference_mode():
        motion, lengths = infer.predict(global_root, mask, local_root, mask, poses, mask, 6,
            config={'num_inference_step': 1, 'pose_token_sampling_use_argmax': True},
            terrain_observation=scene)
    assert torch.isfinite(motion).all() and int(lengths.item()) == 6
    assert seen['root_scene'] is scene and seen['pose_scene'] is scene
    torch.testing.assert_close(seen['height'], root_height_condition(seen['predicted_root'], rep))
    with pytest.raises(ValueError, match='require a current local observation'):
        infer.predict(global_root, mask, local_root, mask, poses, mask, 6)
