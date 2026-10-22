"""Scoped codec wiring for the separately verified convolution replay engine."""

from contextlib import contextmanager
from types import MethodType

import torch
from torch import nn


@contextmanager
def graphed_dilated_convolutions(controller, execution):
    changed = []
    try:
        for module in controller.modules():
            if not (isinstance(module, nn.Conv1d) and module.kernel_size == (3,)
                    and module.dilation[0] > 1 and module.stride == (1,)
                    and module.padding == module.dilation and module.groups == 1
                    and module.padding_mode == 'zeros' and module.bias is not None):
                continue
            previous, native = module.__dict__.get('forward'), module.forward
            def forward(current, value, _native=native):
                if value.is_cuda and value.dtype == torch.float32:
                    return execution.apply(value, current.weight, current.bias, current.dilation[0])
                return _native(value)
            module.forward = MethodType(forward, module)
            changed.append((module, previous))
        yield
    finally:
        for module, previous in reversed(changed):
            if previous is None: del module.forward
            else: module.forward = previous
