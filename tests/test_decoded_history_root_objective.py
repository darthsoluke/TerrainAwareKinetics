import torch

from test_joint_sdf_controller import make_pair
from motionbricks.data.decoded_history_root_objective import decoded_history_root_loss


def test_observed_prefix_loss_has_no_future_or_unobserved_boundary_velocity_gradient():
    _, model, source, history, task, scene = make_pair()
    with torch.no_grad():
        result = model(history, task, [scene], source=source)
    target = result['target']['local_motions']
    predicted = target.clone().requires_grad_()
    assert decoded_history_root_loss(model.inference, predicted, history, result['observation']) < 1e-12
    perturbed = (target + .2).requires_grad_()
    loss = decoded_history_root_loss(model.inference, perturbed, history, result['observation'])
    gradient = torch.autograd.grad(loss, perturbed)[0]
    root = list(model.inference.local_motion_rep.indices['root'])
    height = int(model.inference.local_motion_rep.indices['global_root_y'][0])
    expected = torch.zeros_like(gradient, dtype=torch.bool)
    expected[:, :3, root] = True
    expected[:, 3, height] = True
    assert torch.equal(gradient != 0, expected)
    changed = perturbed.detach().clone()
    changed[:, 4:] = 100
    torch.testing.assert_close(decoded_history_root_loss(model.inference, changed, history, result['observation']), loss,
                               atol=0, rtol=0)


def test_history_loss_reaches_adapters_without_changing_actor_or_frozen_codec():
    _, model, source, history, task, scene = make_pair()
    model.freeze_base()
    result = model(history, task, [scene], source=source, straight_through=True)
    before = result['world'].detach().clone()
    loss = decoded_history_root_loss(model.inference, result['local_prediction'], history, result['observation'])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in model.named_parameters() if name.startswith('sdf_'))
    assert all(p.grad is None for p in model.codec.parameters())
    torch.testing.assert_close(result['world'], before, atol=0, rtol=0)
