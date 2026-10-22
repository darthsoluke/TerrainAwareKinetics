"""Training and online inference share planar goals, with unknown future Y/pose."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_native_mb_stage_training import _motion_and_representation, _attach_minimal_training_context
from test_native_mb_terrain_integration import small_config
from motionbricks.data.native_mb_models import build_native_model
from motionbricks.data.native_mb_online_conditions import (
    ONLINE_PROTOCOL, enable_online_conditioning, sparse_planar_conditions, planar_goal,
    dense_history_pose, goal_direction_local,
)
from motionbricks.data.native_mb_online_live import planar_command_goal, predict_planar_terrain
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from motionbricks.motion_backbone.inference.motion_inference import motion_inference


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def online_config(stage):
    return enable_online_conditioning(small_config(stage))


def test_shared_conditions_never_accept_future_height_pose_or_last_context_velocity():
    root, velocity, pose = torch.randn(2, 4, 5), torch.randn(2, 4, 4), torch.randn(2, 4, 268)
    root.requires_grad_()
    velocity[:, 3] = float('nan')
    goal_source = torch.randn(2, 5)
    goal_source[:, 1] = float('nan')
    result = sparse_planar_conditions(root, velocity, pose, planar_goal(goal_source))
    assert result['has_global_root_values'].sum() == 10
    assert not result['global_root_component_mask'][:, 4:, 1].any()
    assert result['global_root_component_mask'][:, 7, [0, 2, 3, 4]].all()
    assert result['has_local_root_values'][:, :3].all()
    assert not result['has_local_root_values'][:, 3:].any()
    assert not result['has_local_poses'][:, 4:].any()
    assert torch.isfinite(result['global_root_values']).all()
    assert not result['global_root_values'].requires_grad
    assert torch.count_nonzero(result['local_root_values'][:, 3:]) == 0


def test_pose_dense_history_drops_future_even_if_nan():
    poses = torch.randn(2, 24, 268)
    poses[:, 4:] = float('nan')
    values, mask = dense_history_pose(poses)
    assert mask[:, :4].all() and not mask[:, 4:].any()
    assert torch.isfinite(values).all() and torch.count_nonzero(values[:, 4:]) == 0


def test_direction_is_current_body_local_and_rotationally_invariant():
    roots = torch.zeros(1, 4, 5)
    roots[..., 3] = 1
    goal = torch.tensor([[0., 2., 1., 0.]])
    expected = goal_direction_local(roots, goal)
    roots[..., 3], roots[..., 4] = 0, 1
    goal[:, :2] = torch.tensor([[2., 0.]])
    torch.testing.assert_close(goal_direction_local(roots, goal), expected, atol=1e-6, rtol=0)


def test_command_goal_uses_only_planar_command_and_actual_remaining_time():
    root = torch.zeros(1, 4, 5)
    root[..., 1] = 789
    root[:, -1, [0, 2]] = torch.tensor([[3., 4.]])
    target = planar_command_goal(root, [1.5, 0], 0, 8)
    torch.testing.assert_close(target, torch.tensor([[4.4, 4., 1., 0.]]))
    root[..., 1] = -999
    assert torch.equal(target, planar_command_goal(root, [1.5, 0], 0, 8))


def scene_batch(batch):
    result = deepcopy(batch)
    n, frames = batch['motion'].shape[:2]
    result['terrain_root_world'] = torch.zeros(n, frames, 5)
    result['terrain_root_world'][..., 1] = 1
    result['terrain_root_world'][..., 3] = 1
    result['terrain_scene'] = [dict(heightfield_y_m=torch.zeros(81, 81),
                                    min_point_world_xyz_m=torch.tensor([-4., 0., -4.]),
                                    dx_m=torch.tensor(.1)) for _ in range(n)]
    return result


def capture_forward(model, monkeypatch):
    seen = []
    original = model.backbone_net.forward

    def forward(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append((tuple(a.detach().clone() if torch.is_tensor(a) else a for a in args),
                     {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in kwargs.items()},
                     {k: v.detach().clone() for k, v in result.items()}))
        return result

    monkeypatch.setattr(model.backbone_net, 'forward', forward)
    return seen


def test_actual_root_training_uses_components_and_no_future_pose(monkeypatch):
    torch.manual_seed(751)
    np.random.seed(751)
    rep, batch = _motion_and_representation(frame_count=41)
    model = build_native_model('root', rep, online_config('root'))
    model.args['min_tokens'] = model.args['max_tokens'] = 6
    model.train()
    _attach_minimal_training_context(model)
    seen = capture_forward(model, monkeypatch)
    model.training_step(scene_batch(batch), 0).backward()
    args, kwargs, _ = seen[0]
    assert kwargs['global_root_component_mask'][:, :4].all()
    assert not kwargs['global_root_component_mask'][:, 4:, 1].any()
    assert not args[3][:, 3:].any() and not args[5][:, 4:].any()
    assert torch.count_nonzero(args[4][:, 4:]) == 0
    assert model.backbone_net._proj_global_root_component_mask.weight.grad.abs().sum() > 0
    assert not kwargs['terrain_observation']['points'].requires_grad


def test_actual_pose_training_uses_predicted_root_not_teacher_future_height(monkeypatch):
    torch.manual_seed(752)
    np.random.seed(752)
    rep, batch = _motion_and_representation(frame_count=41)
    codec = build_native_model('tokenizer', rep, small_config('tokenizer'))
    root = build_native_model('root', rep, online_config('root')).eval().requires_grad_(False)
    pose = build_native_model('pose', rep, online_config('pose'), tokenizer=codec)
    object.__setattr__(pose, 'online_root_model', root)
    pose.args['min_tokens'] = pose.args['max_tokens'] = 6
    pose.train()
    _attach_minimal_training_context(pose)
    root_seen, pose_seen = capture_forward(root, monkeypatch), capture_forward(pose, monkeypatch)
    # Make the predictor's Y visibly unrelated to GT while still generated through Root.
    root.backbone_net._conv_output.model[-1].bias.data[1] = 5.0
    loss = pose.training_step(scene_batch(batch), 0)
    assert torch.isfinite(loss)
    loss.backward()
    from motionbricks.data.native_mb_terrain_batch import root_height_condition
    predicted = root_seen[0][2]['pred_global_root_values'][:, :24]
    expected = root_height_condition(predicted, rep)
    torch.testing.assert_close(pose_seen[0][1]['root_height_trajectory'], expected)
    assert not pose_seen[0][0][3][:, 4:].any()
    assert all(p.grad is None for p in root.parameters())
    assert all(p.grad is None for p in codec.parameters())
    assert not root.training
    assert not any(key.startswith('online_root_model.') for key in pose.state_dict())


def online_infer():
    rep, batch = _motion_and_representation()
    codec = build_native_model('tokenizer', rep, small_config('tokenizer'))
    root = build_native_model('root', rep, online_config('root')).eval()
    pose = build_native_model('pose', rep, online_config('pose'), tokenizer=codec).eval()
    pose._vqvae_model_loaded = True
    infer = motion_inference({'root': root, 'pose': pose}, pose.args, device='cpu')
    return infer, rep.unnormalize(batch['motion'])[:, :4]


def test_complete_online_inference_masks_unknown_nan_and_preserves_native_decoder(monkeypatch):
    infer, history = online_infer()
    root_seen, pose_seen = capture_forward(infer._root_model, monkeypatch), capture_forward(infer._pose_model, monkeypatch)
    goal = planar_goal(extract_feature_from_motion_rep(history[:, -1:], infer.global_motion_rep, 'root')[:, 0])
    scene = dict(heightfield_y_m=torch.zeros(81, 81), min_point_world_xyz_m=[-4., 0., -4.], dx_m=.1)
    result, tokens = predict_planar_terrain(infer, history, goal, scene, num_tokens=6,
                    config={'num_inference_step': 1, 'pose_token_sampling_use_argmax': True})
    assert int(tokens.item()) == 6 and torch.isfinite(result).all()
    assert not root_seen[0][1]['global_root_component_mask'][:, 4:, 1].any()
    assert not root_seen[0][0][5][:, 4:].any()
    assert not pose_seen[0][0][3][:, 4:].any()
    assert root_seen[0][1]['terrain_observation'] is pose_seen[0][1]['terrain_observation']
    assert result[..., 1].std() > 0  # No constant-Y projection.


@pytest.mark.parametrize('override', [
    {'final_root_pred_mode': 'from_root_module'},
    {'debug_ground_truth_root_trajectories': torch.zeros(1, 24, 5)},
    {'debug_ground_truth_pose_tokens': torch.zeros(1, 6, 8, dtype=torch.long)},
])
def test_online_rejects_output_and_ground_truth_substitutions(override):
    infer, history = online_infer()
    goal = planar_goal(history[:, -1, :5])
    scene = dict(heightfield_y_m=torch.zeros(9, 9), min_point_world_xyz_m=[0., 0., 0.], dx_m=.1)
    with pytest.raises(ValueError, match='forbids'):
        predict_planar_terrain(infer, history, goal, scene, num_tokens=6, config=override)


def test_online_pose_missing_frozen_predictor_fails_not_gt_fallback():
    rep, batch = _motion_and_representation()
    codec = build_native_model('tokenizer', rep, small_config('tokenizer'))
    pose = build_native_model('pose', rep, online_config('pose'), tokenizer=codec)
    pose.args['min_tokens'] = pose.args['max_tokens'] = 6
    _attach_minimal_training_context(pose)
    with pytest.raises(ValueError, match='verified frozen online Root'):
        pose.training_step(scene_batch(batch), 0)


def test_frozen_root_prediction_is_invariant_to_teacher_future_y_and_body():
    from motionbricks.data.native_mb_online_training import predict_training_root
    rep, batch = _motion_and_representation()
    codec = build_native_model('tokenizer', rep, small_config('tokenizer'))
    root = build_native_model('root', rep, online_config('root')).eval().requires_grad_(False)
    pose = build_native_model('pose', rep, online_config('pose'), tokenizer=codec)
    object.__setattr__(pose, 'online_root_model', root)
    values = scene_batch(batch)
    values['global_motions'] = batch['motion'][:, :24].clone()
    values['local_motions'] = rep.dual_rep.global_to_local(
        batch['motion'], True, True, lengths=torch.tensor([25]))[:, :24]
    values['num_tokens'] = torch.tensor([[6]])
    info = {'chosen_ids': torch.tensor([0]), 'frame_indices': torch.arange(25)[None]}
    expected, _, _ = predict_training_root(pose, values, info)
    values['global_motions'][:, 4:, 1] += 999
    values['global_motions'][:, 4:, 5:] = -999
    # Local Root's fourth channel is observed hip height, not velocity.
    values['local_motions'][:, 3:, :3] = 999
    values['local_motions'][:, 4:, 3] = 999
    values['local_motions'][:, 4:, 4:] = 999
    actual, _, _ = predict_training_root(pose, values, info)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_online_nan_unknowns_have_no_effect_and_codec_only_has_one_observed_keyframe(monkeypatch):
    infer, history = online_infer()
    local = infer.motion_rep.dual_rep.global_to_local(history, False, False,
                lengths=torch.tensor([4]))
    roots = history[..., :5]
    conditions = sparse_planar_conditions(roots, local[..., :4],
        extract_feature_from_motion_rep(local, infer.local_motion_rep, infer.EXTERNAL_POSE_FEATURE_MODE),
        planar_goal(roots[:, -1]))
    scene = {'points': torch.zeros(1, 2, 9, 9, 3),
             'valid': torch.zeros(1, 2, 9, 9, dtype=torch.bool)}
    seen = {}
    original = infer._vqvae_pose_model.forward_decoder

    def decoder(*args, **kwargs):
        seen['mask'] = kwargs['has_target_cond'].clone()
        return original(*args, **kwargs)

    monkeypatch.setattr(infer._vqvae_pose_model, 'forward_decoder', decoder)

    def call():
        return infer.predict(conditions['global_root_values'], conditions['has_global_root_values'],
            conditions['local_root_values'], conditions['has_local_root_values'],
            conditions['local_poses'], conditions['has_local_poses'], 6,
            global_root_component_mask=conditions['global_root_component_mask'], terrain_observation=scene,
            config={'num_inference_step': 1, 'pose_token_sampling_use_argmax': True})[0]

    with torch.no_grad():
        expected = call()
        conditions['global_root_values'][~conditions['global_root_component_mask']] = float('nan')
        conditions['local_root_values'][~conditions['has_local_root_values']] = float('nan')
        conditions['local_poses'][~conditions['has_local_poses']] = float('nan')
        actual = call()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert seen['mask'].sum() == 1 and seen['mask'][0, 3]


def test_config_is_explicit_extension_without_loss_or_optimizer_changes():
    config = small_config('root')
    before = deepcopy(config)
    changed = enable_online_conditioning(config)
    assert config == before
    assert changed.model.args.global_root_loss_coeff == config.model.args.global_root_loss_coeff
    assert changed.model.optimizer == config.model.optimizer
    assert changed.model.args.online_conditioning == ONLINE_PROTOCOL['version']
