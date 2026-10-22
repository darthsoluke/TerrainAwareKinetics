import torch
import pytest

from test_native_mb_online_conditions import online_infer
from test_native_mb_terrain_integration import small_config
from motionbricks.data.native_mb_codec_height import build_codec_height_tokenizer, CODEC_HEIGHT_CONDITIONING
from motionbricks.data.native_mb_online_live import predict_planar_terrain, planar_command_goal
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


@pytest.mark.parametrize('context_policy', ['native_random', 'played_context4'])
def test_dense_decoder_consumes_predicted_root_height_not_future_truth(monkeypatch, context_policy):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        infer, history = online_infer()
        adapted = build_codec_height_tokenizer(infer.motion_rep, infer._vqvae_pose_model, small_config('tokenizer'))
        infer._vqvae_pose_model = adapted.pose_net.eval()
        infer._pose_model._supporting_networks['pose_net'] = adapted.pose_net
        infer._pose_model.args['codec_height_conditioning'] = CODEC_HEIGHT_CONDITIONING
        infer._pose_model.args['codec_context_policy'] = context_policy
        seen = {}
        root_forward = infer._root_model.backbone_net.forward
        decoder_forward = adapted.pose_net.forward_decoder

        def root(*args, **kwargs):
            result = root_forward(*args, **kwargs)
            seen['root'] = result['pred_global_root_values'].clone()
            seen['length'] = result['pred_num_tokens'] * 4
            assert not kwargs['global_root_component_mask'][:, 4:, 1].any()
            return result

        def decoder(*args, **kwargs):
            seen['external'] = kwargs['external_cond'].clone()
            assert kwargs['has_target_cond'].sum() == (4 if context_policy == 'played_context4' else 1)
            assert kwargs['has_target_cond'][0, 3]
            assert not kwargs['has_target_cond'][:, 4:].any()
            return decoder_forward(*args, **kwargs)

        monkeypatch.setattr(infer._root_model.backbone_net, 'forward', root)
        monkeypatch.setattr(adapted.pose_net, 'forward_decoder', decoder)
        with torch.no_grad():
            result, _ = predict_planar_terrain(infer, history, planar_command_goal(history[..., :5], [0., 1.], 0., 6),
                {'heightfield_y_m': torch.zeros(81, 81), 'min_point_world_xyz_m': [-4., 0., -4.], 'dx_m': .1},
                num_tokens=6, config={'num_inference_step': 1, 'pose_token_sampling_use_argmax': True})
        local_root = infer.motion_rep.dual_rep.global_to_local(seen['root'], True, True, lengths=seen['length'])
        expected = extract_feature_from_motion_rep(local_root, infer.local_motion_rep, 'root_without_heading')
        assert seen['external'].shape[-1] == 3 and torch.equal(seen['external'], expected)
        assert torch.isfinite(result).all()
    finally:
        torch.set_num_threads(previous)
