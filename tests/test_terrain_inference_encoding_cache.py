import torch

from motionbricks.motion_backbone.neural_modules.local_terrain_encoder import LocalTerrainEncoder


def observation():
    return {'points': torch.zeros(1, 2, 9, 9, 3),
            'valid': torch.ones(1, 2, 9, 9, dtype=torch.bool),
            '_inference_token_cache': {}}


def test_inference_cache_is_per_encoder_and_per_observation():
    first, second = LocalTerrainEncoder(8).eval(), LocalTerrainEncoder(8).eval()
    seen = []
    hook = first.encoder.register_forward_hook(lambda *_: seen.append('first'))
    data = observation()
    with torch.inference_mode():
        uncached = first({k: v for k, v in data.items() if not k.startswith('_')}, 1, 'cpu', torch.float32)
        cached = first(data, 1, 'cpu', torch.float32)
        reused = first(data, 1, 'cpu', torch.float32)
        for a, b in zip(uncached, cached):
            assert torch.equal(a, b)
        assert reused[0] is cached[0]
        assert len(seen) == 2
        second_result = second(data, 1, 'cpu', torch.float32)
        assert second_result[0] is not cached[0]
        new = observation()
        new['points'][..., 1] = -.5
        changed = first(new, 1, 'cpu', torch.float32)
        assert len(seen) == 3 and not torch.equal(changed[0], cached[0])
    hook.remove()


def test_training_does_not_cache_or_drop_encoder_gradients():
    encoder = LocalTerrainEncoder(8).train()
    data = observation()
    seen = []
    hook = encoder.encoder.register_forward_hook(lambda *_: seen.append(1))
    for _ in range(2):
        tokens, _ = encoder(data, 1, 'cpu', torch.float32)
        tokens.square().sum().backward()
    assert len(seen) == 2 and not data['_inference_token_cache']
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    hook.remove()


def test_complete_native_four_iteration_prediction_is_identical_with_cache():
    from test_native_mb_terrain_integration import small_config
    from test_native_mb_stage_training import _motion_and_representation
    from motionbricks.data.native_mb_models import build_native_model
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
    from motionbricks.motion_backbone.inference.motion_inference import motion_inference

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(2333)
        rep, batch = _motion_and_representation()
        codec = build_native_model('tokenizer', rep, small_config('tokenizer'))
        root = build_native_model('root', rep, small_config('root')).eval()
        pose = build_native_model('pose', rep, small_config('pose'), tokenizer=codec).eval()
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
        scene = observation()
        scene.pop('_inference_token_cache')
        count = []
        hook = pose.backbone_net._terrain_encoder.encoder.register_forward_hook(lambda *_: count.append(1))
        with torch.inference_mode():
            config = {'num_inference_step': 4, 'pose_token_sampling_use_argmax': True}
            baseline, baseline_length = infer.predict(global_root, mask, local_root, mask, poses, mask, 6,
                config=config, terrain_observation=scene)
            assert len(count) == 4
            cached, cached_length = infer.predict(global_root, mask, local_root, mask, poses, mask, 6,
                config=dict(config, cache_terrain_encoder=True), terrain_observation=scene)
        hook.remove()
        assert len(count) == 5
        assert torch.equal(baseline, cached) and torch.equal(baseline_length, cached_length)
        assert set(scene) == {'points', 'valid'}
    finally:
        torch.set_num_threads(previous_threads)
