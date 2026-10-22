import torch

from test_joint_sdf_controller import make_pair
from motionbricks.data.decoded_factual_history import decoded_factual_history_loss


def test_prefix_gradient_only_covers_observed_root_and_body():
    _, model, source, history, intent, scene = make_pair()
    with torch.no_grad():
        result = model(history, intent, [scene], source=source)
    target = result['target']['local_motions']
    assert decoded_factual_history_loss(model.inference, target, history, result['observation']) < 1e-12
    prediction = (target + .2).requires_grad_()
    loss = decoded_factual_history_loss(model.inference, prediction, history, result['observation'])
    gradient = torch.autograd.grad(loss, prediction)[0]
    indices = model.inference.local_motion_rep.indices
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[:, :3, indices['root']] = True
    for key in ('global_root_y', 'ric_data', 'global_rot_data'):
        mask[:, :4, indices[key]] = True
    assert torch.equal(gradient != 0, mask)
    altered = prediction.detach().clone()
    altered[:, 4:] = 123
    torch.testing.assert_close(decoded_factual_history_loss(model.inference, altered, history,
        result['observation']), loss, atol=0, rtol=0)
    expected = torch.tensor(.5 * .2 ** 2 * mask.sum().item() / (28 * prediction.shape[-1]))
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)


def test_prefix_objective_updates_decoder_not_frozen_predictors_or_actor_state():
    _, model, source, history, intent, scene = make_pair()
    model.requires_grad_(False)
    model.codec.decoder.requires_grad_(True)
    result = model(history, intent, [scene])
    before = result['world'].detach().clone()
    loss = decoded_factual_history_loss(model.inference, model.last_local[:, :32], history, result['observation'])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.codec.decoder.parameters())
    for component in (model.root, model.pose, model.codec.encoder, model.codec.quantizer, model.sdf_decoder):
        assert all(p.grad is None for p in component.parameters())
    torch.testing.assert_close(before, result['world'], atol=0, rtol=0)


def test_independent_root_normalization_leaves_body_scale_and_future_unchanged():
    _, model, source, history, intent, scene = make_pair()
    with torch.no_grad():
        result = model(history, intent, [scene], source=source)
    target = result['target']['local_motions']
    prediction = (target + .2).requires_grad_()
    args = (model.inference, prediction, history, result['observation'])
    old = decoded_factual_history_loss(*args)
    new = decoded_factual_history_loss(*args, root_coefficient=1.)
    old_grad = torch.autograd.grad(old, prediction, retain_graph=True)[0]
    new_grad = torch.autograd.grad(new, prediction)[0]
    indices = model.inference.local_motion_rep.indices
    root = torch.zeros_like(prediction, dtype=torch.bool)
    root[:, :3, indices['root']] = True
    root[:, 3, indices['global_root_y']] = True
    assert root.sum() == 13
    torch.testing.assert_close(new_grad[~root], old_grad[~root], atol=0, rtol=0)
    torch.testing.assert_close(new_grad[root], old_grad[root] * (28 * prediction.shape[-1] / 13))
    future = prediction.detach().clone()
    future[:, 4:] = 0.
    future[:, 3, indices['local_root_vel']] = -123.
    torch.testing.assert_close(decoded_factual_history_loss(model.inference, future, history,
        result['observation'], root_coefficient=1.), new, atol=0, rtol=0)


def test_independent_factual_root_loss_only_reaches_decoder():
    _, model, source, history, intent, scene = make_pair()
    model.requires_grad_(False)
    model.codec.decoder.requires_grad_(True)
    result = model(history, intent, [scene])
    before = result['world'].detach().clone()
    loss = decoded_factual_history_loss(model.inference, model.last_local[:, :32], history,
                                       result['observation'], root_coefficient=1.)
    loss.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.codec.decoder.parameters())
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith('codec.decoder.'))
    torch.testing.assert_close(result['world'], before, atol=0, rtol=0)
