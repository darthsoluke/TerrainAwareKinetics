import torch

from motionbricks.data.stop_stage_adapter import StopStageCondition


def test_zero_initialization_and_exact_moving_preservation():
    module = StopStageCondition(12)
    query = torch.randn(3, 8, 12)
    controls = torch.randn(3, 7)
    controls[0, 2:4] = 0
    torch.testing.assert_close(module(query, controls), query, atol=0, rtol=0)
    with torch.no_grad():
        module.update[-1].weight.fill_(.3)
        module.update[-1].bias.fill_(.2)
    torch.testing.assert_close(module(query, controls)[1:], query[1:], atol=0, rtol=0)
    assert not torch.equal(module(query, controls)[0], query[0])


def test_no_release_goal_dependency_and_no_moving_gradient():
    module = StopStageCondition(12)
    query = torch.randn(2, 8, 12)
    controls = torch.ones(2, 7)
    controls[0, 2:4] = 0
    other = controls.clone()
    other[:, :2] += 100
    with torch.no_grad():
        module.update[-1].weight.fill_(.1)
    torch.testing.assert_close(module(query, controls), module(query, other), atol=0, rtol=0)
    module(query[1:], controls[1:]).square().mean().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in module.parameters())
