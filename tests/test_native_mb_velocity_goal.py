import pytest
import torch

from motionbricks.data.native_mb_online_conditions import (
    VELOCITY_PROTOCOL, sparse_velocity_conditions, validate_online_inputs, apply_terminal_velocity,
)
from motionbricks.data.native_mb_online_live import online_command_goal


def test_stop_is_zero_terminal_velocity_with_unknown_future_position_and_height():
    root = torch.randn(2, 4, 5)
    velocity = torch.randn(2, 4, 4)
    pose = torch.randn(2, 4, 12)
    velocity[:, 3] = torch.nan  # This forward interval is unobserved.
    goal = torch.tensor([[0., 0., 1., 0.], [1.6, -.3, 0., 1.]])
    result = sparse_velocity_conditions(root, velocity, pose, goal)
    assert not result['global_root_component_mask'][:, 4:, :3].any()
    assert not result['local_root_component_mask'][:, 4:, [0, 3]].any()
    assert not result['has_local_poses'][:, 4:].any()
    torch.testing.assert_close(result['local_root_values'][:, 7, 1:3], goal[:, :2])
    validate_online_inputs(result['global_root_values'], result['has_global_root_values'],
        result['global_root_component_mask'], result['local_root_values'], result['has_local_root_values'],
        result['local_poses'], result['has_local_poses'], {}, version=VELOCITY_PROTOCOL['version'],
        local_component_mask=result['local_root_component_mask'])
    result['local_root_component_mask'][:, 7, 3] = True
    with pytest.raises(ValueError, match='future Root height'):
        validate_online_inputs(result['global_root_values'], result['has_global_root_values'],
            result['global_root_component_mask'], result['local_root_values'], result['has_local_root_values'],
            result['local_poses'], result['has_local_poses'], {}, version=VELOCITY_PROTOCOL['version'],
            local_component_mask=result['local_root_component_mask'])


def test_command_speed_does_not_become_a_fabricated_future_position():
    history = torch.zeros(1, 4, 5)
    history[..., :3] = torch.tensor([40., 3., -20.])
    goal = online_command_goal(history, [1.6, -.2], 1., 8, VELOCITY_PROTOCOL['version'])
    torch.testing.assert_close(goal[:, :2], torch.tensor([[1.6, -.2]]))
    history[..., :3] += 500
    shifted = online_command_goal(history, [1.6, -.2], 1., 8, VELOCITY_PROTOCOL['version'])
    torch.testing.assert_close(shifted, goal, atol=0, rtol=0)


def test_terminal_velocity_cannot_replace_predicted_height_or_angular_velocity():
    trajectory = torch.randn(2, 64, 4)
    sparse = torch.zeros(2, 8, 4)
    sparse[:, -1] = torch.tensor([999., 1.6, -.3, 999.])
    mask = torch.zeros_like(sparse, dtype=torch.bool)
    mask[:, -1, 1:3] = True
    lengths = torch.tensor([32, 64])
    actual = apply_terminal_velocity(trajectory, sparse, mask, lengths)
    for row, count in enumerate(lengths):
        torch.testing.assert_close(actual[row, :count-1], trajectory[row, :count-1], atol=0, rtol=0)
        torch.testing.assert_close(actual[row, count-1, [0, 3]], trajectory[row, count-1, [0, 3]], atol=0, rtol=0)
        torch.testing.assert_close(actual[row, count-1, 1:3], sparse[row, -1, 1:3])


def test_real_velocity_root_training_masks_positions_and_trains_the_new_projection(monkeypatch):
    from test_native_mb_online_conditions import (online_config, scene_batch, capture_forward,
        _motion_and_representation, _attach_minimal_training_context, build_native_model)
    from motionbricks.data.native_mb_online_conditions import enable_velocity_conditioning
    rep, batch = _motion_and_representation(frame_count=41)
    model = build_native_model('root', rep, enable_velocity_conditioning(online_config('root')))
    model.args['min_tokens'] = model.args['max_tokens'] = 6
    model.train()
    _attach_minimal_training_context(model)
    seen = capture_forward(model, monkeypatch)
    loss = model.training_step(scene_batch(batch), 0)
    loss.backward()
    values, kwargs, _ = seen[0]
    assert not kwargs['global_root_component_mask'][:, 4:, :3].any()
    assert not kwargs['local_root_component_mask'][:, 4:, [0, 3]].any()
    assert not values[5][:, 4:].any()
    assert model.backbone_net._proj_local_root_component_mask.weight.grad.abs().sum() > 0


def test_velocity_pose_root_cannot_read_teacher_future_xyz_height_or_pose():
    from test_native_mb_online_conditions import (online_config, scene_batch,
        _motion_and_representation, build_native_model, small_config)
    from motionbricks.data.native_mb_online_conditions import enable_velocity_conditioning
    from motionbricks.data.native_mb_online_training import predict_training_root
    rep, batch = _motion_and_representation()
    codec = build_native_model('tokenizer', rep, small_config('tokenizer'))
    root = build_native_model('root', rep, enable_velocity_conditioning(online_config('root'))).eval().requires_grad_(False)
    pose = build_native_model('pose', rep, enable_velocity_conditioning(online_config('pose')), tokenizer=codec)
    object.__setattr__(pose, 'online_root_model', root)
    values = scene_batch(batch)
    values['global_motions'] = batch['motion'][:, :24].clone()
    values['local_motions'] = rep.dual_rep.global_to_local(batch['motion'], True, True,
                                                         lengths=torch.tensor([25]))[:, :24]
    values['num_tokens'] = torch.tensor([[6]])
    info = {'chosen_ids': torch.tensor([0]), 'frame_indices': torch.arange(25)[None]}
    expected, expected_local, _ = predict_training_root(pose, values, info)
    terminal_velocity = values['local_motions'][:, -1, 1:3].clone()
    values['global_motions'][:, 4:, :3] = 999
    values['global_motions'][:, 4:, 5:] = -999
    values['local_motions'][:, 3:, :3] = 999
    values['local_motions'][:, 4:, 3:] = -999
    values['local_motions'][:, -1, 1:3] = terminal_velocity
    actual, actual_local, _ = predict_training_root(pose, values, info)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_local, expected_local, atol=0, rtol=0)
    torch.testing.assert_close(actual_local[:, -1, 1:3], terminal_velocity)


def test_real_velocity_inference_ignores_nan_unknown_height_and_uses_native_decoder(monkeypatch):
    from test_native_mb_online_conditions import (online_config, _motion_and_representation,
        build_native_model, small_config, capture_forward)
    from motionbricks.data.native_mb_online_conditions import enable_velocity_conditioning
    from motionbricks.motion_backbone.inference.motion_inference import motion_inference
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
    rep, batch = _motion_and_representation()
    codec = build_native_model('tokenizer', rep, small_config('tokenizer'))
    root = build_native_model('root', rep, enable_velocity_conditioning(online_config('root'))).eval()
    pose = build_native_model('pose', rep, enable_velocity_conditioning(online_config('pose')), tokenizer=codec).eval()
    pose._vqvae_model_loaded = True
    infer = motion_inference({'root': root, 'pose': pose}, pose.args, device='cpu')
    history = rep.unnormalize(batch['motion'])[:, :4]
    local = rep.dual_rep.global_to_local(history, False, False, lengths=torch.tensor([4]))
    conditions = sparse_velocity_conditions(history[..., :5], local[..., :4],
        extract_feature_from_motion_rep(local, infer.local_motion_rep, infer.EXTERNAL_POSE_FEATURE_MODE),
        torch.tensor([[0., 0., 1., 0.]]))
    observed = capture_forward(root, monkeypatch)
    scene = {'points': torch.zeros(1, 2, 9, 9, 3), 'valid': torch.zeros(1, 2, 9, 9, dtype=torch.bool)}
    def call():
        return infer.predict(conditions['global_root_values'], conditions['has_global_root_values'],
            conditions['local_root_values'], conditions['has_local_root_values'],
            conditions['local_poses'], conditions['has_local_poses'], 6,
            global_root_component_mask=conditions['global_root_component_mask'],
            local_root_component_mask=conditions['local_root_component_mask'], terrain_observation=scene,
            config={'num_inference_step': 1, 'pose_token_sampling_use_argmax': True})[0]
    with torch.no_grad():
        expected = call()
        conditions['local_root_values'][~conditions['local_root_component_mask']] = torch.nan
        conditions['global_root_values'][~conditions['global_root_component_mask']] = torch.nan
        actual = call()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.isfinite(actual).all()
    assert not observed[0][1]['global_root_component_mask'][:, 4:, :3].any()


@pytest.mark.parametrize('stage', ['root', 'pose'])
def test_amp_preserves_motion_targets_and_frozen_pose_labels(stage):
    import numpy as np
    from test_native_mb_online_conditions import (online_config, scene_batch,
        _motion_and_representation, _attach_minimal_training_context, build_native_model, small_config)
    from motionbricks.data.native_mb_online_conditions import enable_velocity_conditioning
    rep, batch = _motion_and_representation(frame_count=41)
    config = enable_velocity_conditioning(online_config(stage))
    codec = build_native_model('tokenizer', rep, small_config('tokenizer'))
    model = build_native_model(stage, rep, config, tokenizer=codec if stage == 'pose' else None)
    if stage == 'pose':
        root = build_native_model('root', rep, enable_velocity_conditioning(online_config('root')))
        root.eval().requires_grad_(False)
        object.__setattr__(model, 'online_root_model', root)
    model.args['min_tokens'] = model.args['max_tokens'] = 6
    model.train()
    _attach_minimal_training_context(model)
    captured = []
    for amp in (False, True):
        torch.manual_seed(881)
        np.random.seed(881)
        current = scene_batch(batch)
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
            loss = model.training_step(current, 0)
        assert torch.isfinite(loss)
        captured.append(current)
    keys = ['global_motions', 'local_motions']
    if stage == 'pose':
        keys += ['groundtruth_pose_tokens', 'local_root_values', 'pose_cond']
    for key in keys:
        torch.testing.assert_close(captured[0][key], captured[1][key], atol=0, rtol=0)
