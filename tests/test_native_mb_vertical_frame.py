from copy import deepcopy
import numpy as np
import pytest
import torch

from test_native_mb_stage_training import _motion_and_representation, _attach_minimal_training_context
from test_native_mb_online_conditions import online_infer, online_config, scene_batch
from test_native_mb_terrain_integration import small_config
from motionbricks.data.native_mb_vertical_frame import (
    VERTICAL_FRAME, center_motion_window, shift_global_motion,
)
from motionbricks.data.native_mb_models import build_native_model
from motionbricks.data.native_mb_online_live import predict_planar_terrain, planar_command_goal


@pytest.fixture(autouse=True)
def threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prior)


def test_common_vertical_frame_is_lossless_and_preserves_future_height_changes():
    rep, batch = _motion_and_representation(65)
    raw = rep.unnormalize(batch['motion'])
    raw = raw.expand(2, -1, -1).clone()
    # A changing vertical trajectory must survive the common reference, rather
    # than becoming a constant pelvis height at every frame.
    vertical = torch.linspace(0., 1.4, 65)[None, :, None]
    y_channels = np.concatenate((rep.indices['global_root_pos'][1:2], rep.indices['ric_data'][1::3]))
    raw[..., y_channels] += vertical
    offsets = torch.tensor([4., -2.])
    shifted = shift_global_motion(raw, rep, offsets)
    centered, origin = center_motion_window(raw, rep, normalized=False)
    shifted_centered, shifted_origin = center_motion_window(shifted, rep, normalized=False)
    torch.testing.assert_close(centered, shifted_centered, atol=1e-6, rtol=0)
    torch.testing.assert_close(shifted_origin - origin, offsets)
    recovered = shift_global_motion(centered, rep, origin)
    torch.testing.assert_close(recovered, raw)
    torch.testing.assert_close(centered[:, 1:, 1] - centered[:, :-1, 1],
                               raw[:, 1:, 1] - raw[:, :-1, 1])
    assert torch.equal(raw[..., rep.indices['global_rot_data']], centered[..., rep.indices['global_rot_data']])
    different_future = raw.clone()
    different_future[:, 4:, 1] += torch.linspace(0., 2., 61)
    changed, other_origin = center_motion_window(different_future, rep, normalized=False)
    assert torch.equal(origin, other_origin)
    torch.testing.assert_close(changed[:, 4:, 1] - centered[:, 4:, 1],
                               different_future[:, 4:, 1] - raw[:, 4:, 1])
    torch.testing.assert_close(rep.inverse(recovered, False)['posed_joints'],
                               rep.inverse(raw, False)['posed_joints'])


@pytest.mark.parametrize('vertical_frame', [VERTICAL_FRAME, 'observed-terrain-y-v1'])
def test_complete_pipeline_is_equivariant_with_observed_vertical_frame(monkeypatch, vertical_frame):
    inference, history = online_infer()
    for model in (inference._root_model, inference._pose_model):
        model.args['vertical_frame'] = vertical_frame
    scene = {'heightfield_y_m': torch.zeros(81, 81), 'min_point_world_xyz_m': [-4., 0., -4.], 'dx_m': .1}
    goal = planar_command_goal(history[..., :5], [0., 1.], 0., 6)
    kwargs = dict(num_tokens=6, config={'num_inference_step': 1, 'pose_token_sampling_use_argmax': True})
    with torch.no_grad():
        baseline, _ = predict_planar_terrain(inference, history, goal, scene, **kwargs)
        for offset in [-1., 3.]:
            moved = shift_global_motion(history, inference.global_motion_rep, offset)
            other_scene = dict(scene, heightfield_y_m=scene['heightfield_y_m'] + offset)
            prediction, _ = predict_planar_terrain(inference, moved, goal, other_scene, **kwargs)
            restored = shift_global_motion(prediction, inference.global_motion_rep, -offset)
            torch.testing.assert_close(restored, baseline, atol=2e-5, rtol=0)
    inference._root_model.args['vertical_frame'] = None
    with pytest.raises(ValueError, match='vertical frames differ'):
        predict_planar_terrain(inference, history, goal, scene, **kwargs)


def test_native_duration_prediction_remains_bounded_and_keeps_unknown_future():
    infer, history = online_infer()
    seen = {}
    def before(_module, args, kwargs):
        seen['requested'] = args[6].clone()
        seen['allowed'] = kwargs['allowed_pred_num_tokens'].clone()
        seen['components'] = kwargs['global_root_component_mask'].clone()
    hook = infer._root_model.backbone_net.register_forward_pre_hook(before, with_kwargs=True)
    try:
        _, tokens = predict_planar_terrain(infer, history,
            planar_command_goal(history[..., :5], [0., 1.], 0., 8),
            {'heightfield_y_m': torch.zeros(81, 81), 'min_point_world_xyz_m': [-4., 0., -4.], 'dx_m': .1},
            num_tokens=8, config={'online_predict_duration': True, 'pose_token_sampling_use_argmax': True})
    finally:
        hook.remove()
    assert int(seen['requested'].item()) == infer._root_model.backbone_net.MASKED_NUM_TOKENS
    assert 6 <= int(tokens.item()) <= 16
    assert not seen['components'][:, 4:, 1].any()


@pytest.mark.parametrize('stage', ['tokenizer', 'root', 'pose'])
def test_official_training_path_centers_only_after_sampling_and_backpropagates(stage):
    torch.manual_seed(812)
    np.random.seed(812)
    rep, batch = _motion_and_representation(65)
    raw = rep.unnormalize(batch['motion'])
    batch['motion'] = rep.normalize(shift_global_motion(raw, rep, 6.))
    cfg = small_config('tokenizer') if stage == 'tokenizer' else online_config(stage)
    cfg.model.args['vertical_frame'] = VERTICAL_FRAME
    cfg.model.args['codec_context_policy'] = 'played_context4'
    codec = build_native_model('tokenizer', rep, small_config('tokenizer')) if stage == 'pose' else None
    model = build_native_model(stage, rep, cfg, tokenizer=codec)
    if stage == 'pose':
        root_cfg = online_config('root')
        root_cfg.model.args['vertical_frame'] = VERTICAL_FRAME
        root = build_native_model('root', rep, root_cfg).eval().requires_grad_(False)
        object.__setattr__(model, 'online_root_model', root)
    model.train()
    _attach_minimal_training_context(model)
    model.args['min_tokens'] = model.args['max_tokens'] = 6
    original_loss = model.loss
    observed = []
    def loss(batch, *args, **kwargs):
        observed.append(rep.unnormalize(batch['global_motions'])[:, 3, 1])
        if stage == 'tokenizer':
            assert not batch['executed_history_positions'].requires_grad
            assert batch['pose_has_target_cond'][:, :4].all()
        return original_loss(batch, *args, **kwargs)
    model.loss = loss
    value = model.training_step(deepcopy(batch) if stage == 'tokenizer' else scene_batch(batch), 0)
    value.backward()
    assert torch.isfinite(value)
    torch.testing.assert_close(observed[0], rep.stats.mean[1].expand_as(observed[0]), atol=2e-6, rtol=0)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_observed_terrain_frame_retains_body_height_error_and_whole_world_equivariance():
    from motionbricks.data.native_mb_vertical_frame import TERRAIN_VERTICAL_FRAME, center_sparse_conditions
    infer, history = online_infer()
    rep = infer.global_motion_rep
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
    root=history[..., :5]
    pose=extract_feature_from_motion_rep(history,infer.local_motion_rep,infer.EXTERNAL_POSE_FEATURE_MODE)
    points=torch.zeros(1,2,9,9,3)
    ground=torch.tensor([.3])
    points[...,1]=(ground-root[:,3,1])[:,None,None,None]
    observation={'points':points,'valid':torch.ones(1,2,9,9,dtype=torch.bool)}
    centered,_,origin=center_sparse_conditions(root,pose,rep,infer.local_motion_rep,
        pose_feature_mode=infer.EXTERNAL_POSE_FEATURE_MODE,vertical_frame=TERRAIN_VERTICAL_FRAME,
        terrain_observation=observation)
    torch.testing.assert_close(origin,ground,atol=1e-6,rtol=0)
    shifted=root.clone(); shifted[...,1]+=.1
    changed={'points':points.clone(),'valid':observation['valid']}
    changed['points'][...,1]-=.1
    high,_,high_origin=center_sparse_conditions(shifted,pose,rep,infer.local_motion_rep,
        pose_feature_mode=infer.EXTERNAL_POSE_FEATURE_MODE,vertical_frame=TERRAIN_VERTICAL_FRAME,
        terrain_observation=changed)
    torch.testing.assert_close(high_origin,origin,atol=1e-6,rtol=0)
    torch.testing.assert_close(high[...,1]-centered[...,1],torch.full((1,4),.1),atol=1e-6,rtol=0)
    changed['valid'][:,0,4,4]=False
    with pytest.raises(ValueError,match='unobserved'):
        center_sparse_conditions(shifted,pose,rep,infer.local_motion_rep,
            pose_feature_mode=infer.EXTERNAL_POSE_FEATURE_MODE,vertical_frame=TERRAIN_VERTICAL_FRAME,
            terrain_observation=changed)


@pytest.mark.parametrize('stage',['tokenizer','root','pose'])
def test_terrain_frame_uses_sampled_scene_and_original_training_objective(stage):
    from motionbricks.data.native_mb_vertical_frame import TERRAIN_VERTICAL_FRAME
    rep,batch=_motion_and_representation(65)
    cfg=small_config('tokenizer') if stage=='tokenizer' else online_config(stage)
    cfg.model.args['vertical_frame']=TERRAIN_VERTICAL_FRAME
    codec=build_native_model('tokenizer',rep,small_config('tokenizer')) if stage=='pose' else None
    model=build_native_model(stage,rep,cfg,tokenizer=codec)
    if stage=='pose':
        root_cfg=online_config('root'); root_cfg.model.args['vertical_frame']=TERRAIN_VERTICAL_FRAME
        root=build_native_model('root',rep,root_cfg).eval().requires_grad_(False)
        object.__setattr__(model,'online_root_model',root)
    model.train(); _attach_minimal_training_context(model)
    model.args['min_tokens']=model.args['max_tokens']=6
    current=scene_batch(batch)
    for scene in current['terrain_scene']: scene['heightfield_y_m']+=.3
    value=model.training_step(current,0)
    value.backward()
    assert torch.isfinite(value)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
