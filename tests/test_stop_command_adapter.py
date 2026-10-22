import torch
from torch import nn

from motionbricks.data.stop_command_adapter import StopCommandAdapter


def test_initial_output_and_moving_output_are_identical_to_base():
    torch.manual_seed(2333)
    base = nn.Sequential(nn.Linear(7, 12), nn.GELU(), nn.Linear(12, 12))
    adapter = StopCommandAdapter(base)
    values = torch.randn(4, 7)
    values[0, 2:4] = 0
    original = base(values).detach()
    torch.testing.assert_close(adapter(values), original, atol=0, rtol=0)
    with torch.no_grad():
        adapter.stop[-1].weight.fill_(.2)
        adapter.stop[-1].bias.fill_(.1)
    torch.testing.assert_close(adapter(values)[1:], original[1:], atol=0, rtol=0)
    assert not torch.equal(adapter(values)[0], original[0])


def test_only_stop_residual_has_training_gradients():
    base = nn.Sequential(nn.Linear(7, 12), nn.GELU(), nn.Linear(12, 12))
    adapter = StopCommandAdapter(base)
    values = torch.ones(4, 7)
    values[:2, 2:4] = 0
    adapter(values).square().mean().backward()
    assert all(parameter.grad is None for parameter in base.parameters())
    assert adapter.stop[-1].weight.grad.norm() > 0
    adapter.zero_grad(set_to_none=True)
    adapter(values[2:]).square().mean().backward()
    assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
               for parameter in adapter.stop.parameters())
