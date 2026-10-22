import torch
from torch import nn

from scripts.audit_joint_sdf_codec_time_support import temporal_support


def test_temporal_support_matches_real_strided_dilated_convolution_gradients():
    module = nn.Sequential(nn.Conv1d(1, 1, 3, padding=2, dilation=2, bias=False),
                           nn.Conv1d(1, 1, 4, stride=2, padding=1, bias=False))
    for layer in module:
        nn.init.ones_(layer.weight)
    inputs = torch.ones(1, 1, 16, requires_grad=True)
    output = module(inputs)
    support = temporal_support(module, torch.eye(16, dtype=torch.bool))
    for position in range(output.shape[-1]):
        actual = torch.autograd.grad(output[0, 0, position], inputs, retain_graph=True)[0][0, 0] != 0
        assert torch.equal(support[:, position], actual)
