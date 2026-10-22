"""Exact center-only evaluation when dilated side taps see only zero padding."""

from contextlib import contextmanager
from types import MethodType

from torch import nn
from torch.nn import functional as F


VERSION = 'zero-padded-dilation-center-v1'


def center_or_native(module, value, native_forward):
    if (value.ndim == 3 and module.kernel_size == (3,) and module.stride == (1,)
            and module.padding == module.dilation and module.groups == 1
            and module.padding_mode == 'zeros' and module.dilation[0] >= value.shape[-1]):
        return F.linear(value.transpose(1,2), module.weight[:,:,1], module.bias).transpose(1,2)
    return native_forward(value)


@contextmanager
def short_sequence_convolutions(module):
    """Keep all parameters/state keys and restore every original method on exit."""
    changed=[]
    try:
        for child in module.modules():
            if not isinstance(child,nn.Conv1d):
                continue
            previous=child.__dict__.get('forward')
            native=child.forward
            def forward(current,value,_native=native):
                return center_or_native(current,value,_native)
            child.forward=MethodType(forward,child)
            changed.append((child,previous))
        yield
    finally:
        for child,previous in reversed(changed):
            if previous is None:
                del child.forward
            else:
                child.forward=previous
