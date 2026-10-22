import pytest
import torch

from test_joint_sdf_controller import make_pair
from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_decoder_pilot import training_targets
from motionbricks.data.terrain_root_supervision import factual_history_root_targets
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


@pytest.mark.parametrize('frame', [None, 'observed-pelvis-y-v1', 'observed-terrain-y-v1'])
def test_source_history_matches_original_native_root_loss_and_gradient(frame):
    _, model, source, history, task, scene = make_pair()
    inference = model.inference
    for stage in (inference._root_model, inference._pose_model):
        stage.args['vertical_frame'] = frame
    result = model(history, task, [scene], source=source)
    target = result['target']
    actual = factual_history_root_targets(inference, target, history, result['observation'])
    expected = {'global_motions': target['global_motions'], 'local_motions': target['local_motions'],
                'groundtruth_num_tokens': torch.tensor([[8]])}
    old_loss = inference._root_model.loss(expected, result['root_output'])['loss']
    new_loss = inference._root_model.loss(actual, result['root_output'])['loss']
    torch.testing.assert_close(old_loss, new_loss, atol=1e-7, rtol=1e-6)
    predictions = result['root_output']['pred_global_root_values']
    old_grad = torch.autograd.grad(old_loss, predictions, retain_graph=True)[0]
    new_grad = torch.autograd.grad(new_loss, predictions)[0]
    torch.testing.assert_close(old_grad, new_grad, atol=1e-7, rtol=1e-6)


def test_only_observed_root_labels_change_without_touching_future_or_pose_targets():
    _, model, source, history, task, scene = make_pair()
    inference = model.inference
    rep = inference.global_motion_rep
    moved = shift_global_motion(history, rep, .15)
    moved[..., 0] += torch.arange(4) * .07
    moved = moved.detach().requires_grad_()
    result = model(moved, task, [scene], source=source)
    target = result['target']
    copies = {key: value.clone() for key, value in target.items() if torch.is_tensor(value)}
    actual = factual_history_root_targets(inference, target, moved, result['observation'])
    factual = training_targets(inference, torch.cat((moved.detach(), source[:, 4:]), 1),
                               moved, result['observation'])
    expected_global = extract_feature_from_motion_rep(factual['global_motions'], rep, 'root')
    expected_local = extract_feature_from_motion_rep(factual['local_motions'], inference.local_motion_rep, 'root')
    torch.testing.assert_close(actual['global_motions'][:, :4], expected_global[:, :4], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual['local_motions'][:, :3], expected_local[:, :3], atol=1e-7, rtol=1e-6)
    original_global = extract_feature_from_motion_rep(target['global_motions'], rep, 'root')
    original_local = extract_feature_from_motion_rep(target['local_motions'], inference.local_motion_rep, 'root')
    torch.testing.assert_close(actual['global_motions'][:, 4:], original_global[:, 4:], atol=0, rtol=0)
    torch.testing.assert_close(actual['local_motions'][:, 3:], original_local[:, 3:], atol=0, rtol=0)
    assert (actual['global_motions'][:, :4] - original_global[:, :4]).abs().max() > .01
    for key in copies:
        torch.testing.assert_close(target[key], copies[key], atol=0, rtol=0)
    assert all(not value.requires_grad for value in actual.values())
    with torch.no_grad():
        repeated = model(moved.detach(), task, [scene])['world']
    torch.testing.assert_close(result['world'], repeated, atol=1e-6, rtol=1e-5)
