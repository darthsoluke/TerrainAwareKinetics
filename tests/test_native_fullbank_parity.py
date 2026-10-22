import pytest
import torch

from scripts.fit_native_root_fullbank import checked_forward_difference, stack_calls
from scripts.fit_native_pose_fullbank import source_balanced_pose_loss
from motionbricks.motion_backbone.models.pose_model import MotionModel
from types import SimpleNamespace


def test_duration_masks_must_match_without_nan_difference_report():
    actual = torch.tensor([[1., -torch.inf]])
    expected = torch.tensor([[1.00001, -torch.inf]])
    assert 0 < checked_forward_difference(actual, expected) < 2e-5
    with pytest.raises(AssertionError):
        checked_forward_difference(actual, torch.tensor([[1., 0.]]))
    with pytest.raises(AssertionError):
        checked_forward_difference(torch.tensor([torch.nan]), torch.tensor([torch.nan]))


def test_batch_calls_keep_distinct_observations_and_reject_metadata_changes():
    first = {'observation': {'points': torch.zeros(1, 3, 2)}, 'config': {'cache': False}}
    second = {'observation': {'points': torch.ones(1, 3, 2)}, 'config': {'cache': False}}
    result = stack_calls([first, second])
    assert result['observation']['points'].shape == (2, 3, 2)
    assert result['observation']['points'][1].eq(1).all()
    second['config']['cache'] = True
    with pytest.raises(ValueError):
        stack_calls([first, second])


def test_pose_source_weighting_does_not_depend_on_number_of_corrupted_tokens():
    teacher = SimpleNamespace(loss=lambda batch, output: MotionModel.loss(None, batch, output))
    controller = SimpleNamespace(inference=SimpleNamespace(_pose_model=teacher))
    logits = torch.tensor([[[[2., 0.], [0., 3.]]], [[[1., 0.], [0., 4.]]],
                           [[[0., 1.], [1., 0.]]]], requires_grad=True)
    batch = dict(groundtruth_pose_tokens=torch.zeros(3, 1, 2, dtype=torch.long),
                 focused_token_mask=torch.tensor([[True, False], [True, True], [False, False]]))
    loss = source_balanced_pose_loss(controller, batch, logits, [([0], .5), ([1, 2], .5)])
    expected = .5 * torch.nn.functional.cross_entropy(logits[0, 0, :1], torch.tensor([0])) \
        + .25 * torch.nn.functional.cross_entropy(logits[1, 0], torch.tensor([0, 0]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[2].eq(0).all()
    assert logits.grad[0, 0, 1].eq(0).all()
    assert torch.isfinite(logits.grad).all()
