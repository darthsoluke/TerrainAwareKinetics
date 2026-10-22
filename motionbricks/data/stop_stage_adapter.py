"""Learned stop intent inside the generative network, not an output correction."""

import torch
from torch import nn


STOP_STAGE_VERSION = 'stop-query-and-desired-control-residual-v1'


class StopStageCondition(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.update = nn.Sequential(nn.Linear(width+5, width), nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, query, controls):
        if query.ndim != 3 or controls.shape != (len(query), 7):
            raise ValueError('Expected native queries and seven original control features')
        # The retained release-position delta is not a desired motion style.
        features = controls[:, 2:7].unsqueeze(1).expand(-1, query.shape[1], -1)
        stopped = (controls[:, 2:4].abs().amax(-1) <= 1e-6)[:, None, None]
        return query + self.update(torch.cat((query, features), -1))*stopped


def install_stop_stage(model, stage):
    if stage not in ('pose', 'decoder'):
        raise ValueError('This controlled probe changes exactly one stage')
    width = model.pose._args['n_embd'] if stage == 'pose' else model.codec.code_dim
    module = StopStageCondition(width).to(model.inference.device)
    setattr(model, 'stop_'+stage+'_condition', module)
    cache = {}

    def observe_control(_module, args):
        cache['controls'] = args[0]

    def condition(_module, args):
        if 'controls' not in cache:
            raise ValueError('Stop conditioning requires the actual causal command')
        query = args[0] if stage == 'pose' else args[0].transpose(1, 2)
        value = module(query, cache['controls'])
        return (value if stage == 'pose' else value.transpose(1, 2), *args[1:])

    target = model.pose._transformer_model if stage == 'pose' else model.codec.decoder
    handles = [model.control.register_forward_pre_hook(observe_control),
               target.register_forward_pre_hook(condition)]
    return module, handles
