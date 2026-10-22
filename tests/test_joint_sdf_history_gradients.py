import torch

from test_joint_sdf_controller import make_pair
from motionbricks.data.native_mb_observed_history import reencode_played_history
from scripts.pilot_joint_sdf_reciprocal_source import configure_trainable


def test_history_reencoding_preserves_values_and_finite_authoritative_gradients():
    _, model, _, history, _, _ = make_pair()
    history = history.clone().requires_grad_()
    plain = reencode_played_history(history, model.inference.global_motion_rep)
    attached = reencode_played_history(history, model.inference.global_motion_rep, differentiable=True)
    torch.testing.assert_close(plain, attached, atol=0, rtol=0)
    assert not plain.requires_grad and attached.requires_grad
    gradient = torch.autograd.grad(attached.square().mean(), history)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0


def test_training_history_gradient_changes_no_forward_values_and_resets_by_default():
    _, model, source, history, intent, scene = make_pair()
    configure_trainable(model, 'coordinated')
    history = history.clone().requires_grad_()
    before = model(history, intent, [scene], straight_through=True)['world']
    attached = model(history, intent, [scene], straight_through=True, history_gradients=True)['world']
    torch.testing.assert_close(before, attached, atol=3e-5, rtol=1e-5)
    gradient = torch.autograd.grad(attached[:, 4:12].square().mean(), history)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert gradient[..., model.inference.global_motion_rep.indices['global_rot_data']].norm() > 0
    model(history, intent, [scene])
    assert model._history_gradients is False


def test_next_primitive_loss_reaches_previous_generated_state_only_with_opt_in():
    _, model, source, history, intent, scene = make_pair()
    configure_trainable(model, 'coordinated')
    first = model(history, intent, [scene], straight_through=True)['world']
    observed = reencode_played_history(first[:, 8:12], model.inference.global_motion_rep, differentiable=True)
    second = model(observed, intent, [scene], straight_through=True, history_gradients=True)['world']
    gradient = torch.autograd.grad(second[:, 4:12].square().mean(), first)[0]
    assert torch.isfinite(gradient).all() and gradient[:, 8:12].norm() > 0
    detached = reencode_played_history(first[:, 8:12], model.inference.global_motion_rep)
    output = model(detached, intent, [scene], straight_through=True)['world']
    assert torch.autograd.grad(output.square().mean(), first, allow_unused=True)[0] is None


def test_two_primitive_backward_keeps_native_parameters_finite():
    _, model, source, history, intent, scene = make_pair()
    configure_trainable(model, 'coordinated')
    first = model(history, intent, [scene], source=source, straight_through=True, history_gradients=True)
    native = model.native_losses(first, source)['native_total']
    observed = reencode_played_history(first['world'][:, 8:12], model.inference.global_motion_rep, differentiable=True)
    second = model(observed, intent, [scene], straight_through=True, history_gradients=True)['world']
    (native+first['world'][:, 4:12].square().mean()+second[:, 4:12].square().mean()).backward()
    for module in (model.root, model.pose, model.codec.decoder):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
