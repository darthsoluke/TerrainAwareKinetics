"""Training-only support correction restricted to explicit Pose parameters."""

import torch


def add_pose_support_gradient(correction, pose_parameters):
    parameters = tuple(pose_parameters)
    if not parameters or any(not p.is_leaf or not p.requires_grad for p in parameters):
        raise ValueError('Expected trainable Pose parameter leaves')
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError('Duplicate Pose parameter')
    gradients = torch.autograd.grad(correction, parameters, allow_unused=True)
    norm_squared = correction.new_zeros(())
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise ValueError('Nonfinite Pose support gradient')
        norm_squared += gradient.square().sum()
        if parameter.grad is None:
            parameter.grad = gradient.detach().clone()
        else:
            parameter.grad.add_(gradient.detach())
    return norm_squared.sqrt()
