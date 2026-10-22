from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motionbricks.data.native_mb_root_conditioning import (
    decoder_root_interpolation, interpolate_planar_residual,
)


def test_residual_hits_only_authored_anchors_and_preserves_nonaffine_detail():
    prediction = np.array([[0., 0], [1, 2], [3, 4], [4, 5]])
    original = prediction.copy()
    result = interpolate_planar_residual(prediction, np.array([[2., 3], [9, 11]]), np.array([0, 3]))
    np.testing.assert_allclose(result[[0, 3]], [[2, 3], [9, 11]])
    np.testing.assert_allclose(np.diff(result - prediction, n=2, axis=0), 0, atol=1e-12)
    np.testing.assert_array_equal(original, prediction)


def test_translation_and_yaw_equivariance_and_multiple_anchors():
    path = np.array([[0., 0], [1, 2], [3, 4], [4, 5]])
    targets = np.array([[2., 3], [6, 8], [9, 11]])
    ids = np.array([0, 2, 3])
    rotation = np.array([[0., 1], [-1, 0]])
    expected = interpolate_planar_residual(path, targets, ids)
    actual = interpolate_planar_residual(path @ rotation + 100, targets @ rotation + 100, ids)
    np.testing.assert_allclose(actual, expected @ rotation + 100)
    np.testing.assert_allclose(expected[ids], targets)


@pytest.mark.parametrize('indices', [[0], [2, 1], [-1, 3], [0, 4], [0., 3.]])
def test_invalid_anchor_contract_fails(indices):
    with pytest.raises(ValueError):
        interpolate_planar_residual(np.zeros((4, 2)), np.zeros((len(indices), 2)), np.array(indices))


def test_decoder_boundary_preserves_tokens_inputs_height_and_raw_outputs():
    seen = {}
    global_rep = SimpleNamespace(indices={'global_root_pos_2d': [0, 2]},
                                 normalize=lambda x: x, unnormalize=lambda x: x)
    def localize(value, **kwargs):
        return value[..., :4].clone()
    sentinel = object()
    def original(batch, config, info):
        seen.update(batch)
        return sentinel
    infer = SimpleNamespace(global_motion_rep=global_rep, local_motion_rep=None,
        motion_rep=SimpleNamespace(dual_rep=SimpleNamespace(global_to_local=localize)),
        _decode_motions_from_predicted_root_and_pose_tokens=original)
    root = torch.zeros(1, 16, 5)
    root[..., 1] = 1.3
    inputs = torch.zeros(1, 8, 5)
    inputs[0, 3, 0], inputs[0, 7, 0] = 1, 3
    batch = dict(pred_num_tokens=torch.tensor([4]), pred_global_root_values=root,
        global_root_values=inputs, has_global_root_values=torch.tensor([[False]*3+[True]+[False]*3+[True]]),
        pred_local_root_values=torch.zeros(1, 16, 4), pred_pose_tokens=torch.ones(1, 4, 8))
    with decoder_root_interpolation(infer) as trace:
        assert infer._decode_motions_from_predicted_root_and_pose_tokens(batch, {}, {}) is sentinel
        assert seen['pred_pose_tokens'] is batch['pred_pose_tokens']
        torch.testing.assert_close(seen['pred_global_root_values'][..., 1], root[..., 1])
        torch.testing.assert_close(seen['pred_global_root_values'][0, [3, 15], 0], torch.tensor([1., 3.]))
        assert trace[0]['input_anchor_frames'] == [0, 3, 15]
        assert seen['pred_global_root_values'][0, 0, 0] == 0
    assert infer._decode_motions_from_predicted_root_and_pose_tokens is original
    assert root[..., 0].count_nonzero() == 0


def test_origin_anchor_survives_official_velocity_roundtrip():
    from motionbricks.data.native_mb_data import build_native_representation
    rep = build_native_representation()
    coarse = np.stack([np.linspace(.2, .8, 12), np.linspace(-.1, 1.5, 12)], -1).astype(np.float32)
    target = np.array([[0, 0], [.3, .4], [1., 2.]], dtype=np.float32)
    ids = np.array([0, 3, 11])
    refined = interpolate_planar_residual(coarse, target, ids)
    root = torch.zeros(1, 12, 5)
    root[..., 1], root[..., 3] = 1, 1
    root[0, :, [0, 2]] = torch.from_numpy(refined)
    lengths = torch.tensor([12])
    local = rep.dual_rep.global_to_local(root, False, False, lengths=lengths)
    restored = rep.dual_rep.local_to_global(local, False, False, lengths=lengths)
    torch.testing.assert_close(restored[0, ids][:, [0, 2]], torch.from_numpy(target), atol=1e-6, rtol=0)
    without_origin = interpolate_planar_residual(coarse, target[1:], ids[1:])
    assert np.linalg.norm(without_origin[0]) > .05
    root[0, :, [0, 2]] = torch.from_numpy(without_origin)
    wrong = rep.dual_rep.local_to_global(
        rep.dual_rep.global_to_local(root, False, False, lengths=lengths), False, False, lengths=lengths)
    assert (wrong[0, -1, [0, 2]] - torch.from_numpy(target[-1])).norm() > .05
