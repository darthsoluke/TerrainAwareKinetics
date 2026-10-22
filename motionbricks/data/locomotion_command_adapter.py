"""Causal desired-motion conditioning inside native Pose and decoder networks."""

import torch
from torch import nn


COMMAND_VERSION = 'causal-all-stage-locomotion-intent-v1'


def desired_motion_features(controls):
    if controls.ndim != 2 or controls.shape[1] != 7 or not torch.isfinite(controls).all():
        raise ValueError('Expected the seven existing causal command features')
    # A retained stop position is a Root goal, not a request for a walking style.
    moving = (controls[:, 2:4].abs().amax(-1) > 1e-6).to(controls.dtype)
    return torch.cat((controls[:, 2:7], moving[:, None]), -1)


class LocomotionCommandCondition(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.update = nn.Sequential(nn.Linear(width+6, width), nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, query, controls):
        if query.ndim != 3 or len(query) != len(controls):
            raise ValueError('Command batch must match native queries')
        features = desired_motion_features(controls)[:, None].expand(-1, query.shape[1], -1)
        return query + self.update(torch.cat((query, features), -1))


def install_locomotion_command(model):
    if hasattr(model, 'locomotion_pose_condition'):
        raise ValueError('Command adapter already installed')
    cache = {}
    def observe(_module, args):
        cache['controls'] = args[0]
    handles = [model.control.register_forward_pre_hook(observe)]
    for stage, target, width, transpose in (
        ('pose', model.pose._transformer_model, model.pose._args['n_embd'], False),
        ('decoder', model.codec.decoder, model.codec.code_dim, True),
    ):
        module = LocomotionCommandCondition(width).to(model.inference.device)
        setattr(model, 'locomotion_'+stage+'_condition', module)
        def condition(_module, args, module=module, transpose=transpose):
            if 'controls' not in cache:
                raise ValueError('Actor command has not been constructed')
            query = args[0].transpose(1, 2) if transpose else args[0]
            value = module(query, cache['controls'])
            return (value.transpose(1, 2) if transpose else value, *args[1:])
        handles.append(target.register_forward_pre_hook(condition))
    return handles


def install_checkpoint_command(model, checkpoint):
    """Build the declared training modules before strict runtime state loading."""
    declaration = checkpoint.get('contract', {}).get('locomotion_command')
    keys = checkpoint['model_state']
    present = [any(key.startswith('locomotion_'+stage+'_condition.') for key in keys)
               for stage in ('pose', 'decoder')]
    if declaration is None and not any(present):
        return False
    if not isinstance(declaration, dict) or declaration.get('version') != COMMAND_VERSION or not all(present):
        raise ValueError('Checkpoint command module declaration and weights disagree')
    install_locomotion_command(model)
    return True
