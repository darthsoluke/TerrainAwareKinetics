import torch

from test_joint_sdf_controller import make_pair
from motionbricks.data.native_codec_fidelity import source_fidelity_losses


def test_source_identity_and_decoded_root_gradient_coordinates():
    _, model, source, history, task, scene = make_pair()
    with torch.no_grad():
        result = model(history, task, [scene], source=source)
    target = result['target']
    exact = target['local_motions'].clone().requires_grad_()
    identity = source_fidelity_losses(model.inference, exact, target, history, result['observation'])
    assert identity['total'] < 1e-6
    perturbed = target['local_motions'].clone()
    root_ids = list(model.inference.local_motion_rep.indices['root'])
    perturbed[:, :, root_ids] += .15
    perturbed.requires_grad_()
    before = perturbed.detach().clone()
    losses = source_fidelity_losses(model.inference, perturbed, target, history, result['observation'])
    gradient = torch.autograd.grad(losses['decoded_root_native'], perturbed, retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient[..., root_ids].abs().sum() > 0
    nonroot = [i for i in range(perturbed.shape[-1]) if i not in root_ids]
    assert gradient[..., nonroot].abs().sum() == 0
    fk_grad = torch.autograd.grad(losses['rendered_fk'], perturbed)[0]
    assert torch.isfinite(fk_grad).all() and fk_grad.abs().sum() > 0
    torch.testing.assert_close(perturbed, before, atol=0, rtol=0)
