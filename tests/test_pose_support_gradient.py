import pytest
import torch

from motionbricks.data.pose_support_gradient import add_pose_support_gradient


def test_pose_support_gradient_leaves_coupled_root_decoder_gradients_unchanged():
    root = torch.nn.Parameter(torch.tensor(2.))
    pose = torch.nn.Parameter(torch.tensor(3.))
    decoder = torch.nn.Parameter(torch.tensor(4.))
    output = (root * pose + pose.square()) * decoder
    base = output.square()
    support = (output - 10.).square()
    expected = torch.autograd.grad(base + support, pose, retain_graph=True)[0]
    base.backward(retain_graph=True)
    root_before, decoder_before = root.grad.clone(), decoder.grad.clone()
    values = [p.detach().clone() for p in (root, pose, decoder)]
    norm = add_pose_support_gradient(support, (pose,))
    torch.testing.assert_close(pose.grad, expected)
    torch.testing.assert_close(root.grad, root_before, atol=0, rtol=0)
    torch.testing.assert_close(decoder.grad, decoder_before, atol=0, rtol=0)
    for parameter, before in zip((root, pose, decoder), values):
        torch.testing.assert_close(parameter, before, atol=0, rtol=0)
    assert norm > 0


def test_pose_support_gradient_rejects_duplicate_parameters():
    pose = torch.nn.Parameter(torch.tensor(3.))
    with pytest.raises(ValueError, match='Duplicate'):
        add_pose_support_gradient(pose.square(), (pose, pose))
