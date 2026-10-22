import torch
from torch.nn import functional as F

from motionbricks.motion_backbone.neural_modules.local_terrain_encoder import terrain_average_pool


def test_deterministic_adaptive_bins_preserve_forward_and_backward():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.manual_seed(123)
    try:
        for shape in ((17,17),(16,19),(3,5)):
            original = torch.randn(2,4,*shape,dtype=torch.float64,requires_grad=True)
            copied = original.detach().clone().requires_grad_(True)
            torch.use_deterministic_algorithms(False)
            reference = F.adaptive_avg_pool2d(original,(3,3))
            weight = torch.randn_like(reference)
            (reference*weight).sum().backward()
            torch.use_deterministic_algorithms(True)
            result = terrain_average_pool(copied,3)
            (result*weight).sum().backward()
            torch.testing.assert_close(result,reference,atol=1e-14,rtol=1e-14)
            torch.testing.assert_close(copied.grad,original.grad,atol=1e-14,rtol=1e-14)
    finally:
        torch.use_deterministic_algorithms(previous)
