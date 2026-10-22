from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf

from test_native_mb_online_conditions import online_infer
from test_native_mb_stage_training import _motion_and_representation
from test_native_mb_terrain_integration import small_config
from motionbricks.data.terrain_decoder_pilot import DecoderExperiment, TerrainLatentCondition, training_targets
from motionbricks.data.native_mb_online_conditions import planar_goal
from motionbricks.data.native_mb_online_live import predict_planar_terrain


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_zero_gate_preserves_latent_and_learns_without_observation_gradient():
    torch.manual_seed(3)
    adapter = TerrainLatentCondition(32)
    latent = torch.randn(2, 32, 8, requires_grad=True)
    observation = {'points': torch.randn(2, 2, 9, 9, 3).clamp(-1, 1).requires_grad_(),
                   'valid': torch.ones(2, 2, 9, 9, dtype=torch.bool)}
    output = adapter(latent, observation)
    torch.testing.assert_close(output, latent, atol=0, rtol=0)
    output.square().mean().backward()
    assert observation['points'].grad is None
    assert adapter.attention.gate.grad.abs() > 0
    with torch.no_grad():
        adapter.attention.gate.fill_(.3)
    changed = adapter(latent, observation)
    assert (changed-latent).abs().max() > 0
    unknown = dict(observation, valid=torch.zeros_like(observation['valid']))
    torch.testing.assert_close(adapter(latent, unknown), latent, atol=0, rtol=0)


@pytest.mark.parametrize('frame', [None, 'observed-pelvis-y-v1', 'observed-terrain-y-v1'])
def test_actual_native_request_replay_parity_gradient_and_frozen_vocabulary(frame):
    inference, history = online_infer()
    for model in (inference._root_model, inference._pose_model):
        model.args['vertical_frame'] = frame
    cfg = small_config('tokenizer')
    scene = dict(heightfield_y_m=torch.zeros(81, 81), min_point_world_xyz_m=[-4., 0., -4.], dx_m=.1)
    goal = planar_goal(history[:, -1, :5])
    with torch.no_grad():
        reference, _ = predict_planar_terrain(inference, history, goal, scene,
            num_tokens=8, config={'num_inference_step': 4, 'pose_token_sampling_use_argmax': True})
    pilot = DecoderExperiment(inference, {'tokenizer_config': OmegaConf.to_container(cfg)}, terrain=True)
    try:
        actual, call = pilot.request(history, goal, scene)
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=0)
        local = pilot.replay(call)
        restored = pilot.restore_world(local, history, call['observation'])
        torch.testing.assert_close(restored[:, :32], actual[:, :32], atol=2e-5, rtol=0)
        _, batch = _motion_and_representation(33)
        source = inference.global_motion_rep.unnormalize(batch['motion'])
        losses, _ = pilot.loss(source, history, call)
        losses['loss'].backward()
        assert torch.isfinite(losses['loss'])
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pilot.codec.decoder.parameters())
        for module in (pilot.codec.encoder, pilot.codec.quantizer, inference._root_model,
                       inference._pose_model.backbone_net):
            assert all(p.grad is None and not p.requires_grad for p in module.parameters())
        with torch.no_grad():
            next(pilot.codec.decoder.parameters()).add_(.0001)
        fresh, fresh_call = pilot.request(history, goal, scene)
        assert torch.equal(call['args'][0], fresh_call['args'][0])
        for key in ('target_cond', 'has_target_cond', 'external_cond', 'token_mask'):
            assert torch.equal(call['kwargs'][key], fresh_call['kwargs'][key])
        cached = pilot.restore_world(pilot.replay(call), history, call['observation'])
        torch.testing.assert_close(cached[:, :32], fresh[:, :32], atol=2e-5, rtol=0)
    finally:
        pilot.close()


def test_generated_history_changes_origin_but_never_replaces_source_future():
    inference, history = online_infer()
    _, batch = _motion_and_representation(33)
    source = inference.global_motion_rep.unnormalize(batch['motion'])
    source.requires_grad_()
    moved = history.clone()
    moved[..., 0] += 2.
    observation = {'points': torch.zeros(1, 2, 9, 9, 3), 'valid': torch.ones(1, 2, 9, 9, dtype=torch.bool)}
    target = training_targets(inference, source, moved, observation)
    recovered = inference.global_motion_rep.unnormalize(target['global_motions'])
    torch.testing.assert_close(recovered[..., 0], source.detach()[:, :32, 0]-moved[:, :1, 0])
    assert not target['global_motions'].requires_grad
    assert target['motion_frame_mask'][0, :4].sum() == 0
    assert target['motion_interval_mask'][0, 3]
