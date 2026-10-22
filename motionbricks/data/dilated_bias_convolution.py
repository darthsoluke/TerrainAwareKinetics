"""Preserve native dilated GEMMs while coalescing CUDA bias initialization."""

from contextlib import contextmanager
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


class BatchedBiasNativeBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, bias, dilation):
        ctx.save_for_backward(value, weight)
        ctx.bias_size, ctx.dilation = list(bias.shape), dilation
        columns = F.unfold(value.unsqueeze(2), (1, 3),
            dilation=(1, dilation), padding=(0, dilation))
        output = torch.empty((value.shape[0], weight.shape[0], value.shape[-1]),
            device=value.device, dtype=value.dtype)
        # Native SlowDilated2d initializes each channel separately before GEMM.
        output.copy_(bias[None, :, None])
        flattened = weight.reshape(weight.shape[0], -1)
        for index in range(value.shape[0]):
            output[index].addmm_(flattened, columns[index])
        return output

    @staticmethod
    def backward(ctx, gradient):
        value, weight = ctx.saved_tensors
        gradients = torch.ops.aten.convolution_backward.default(gradient, value, weight,
            ctx.bias_size, [1], [ctx.dilation], [ctx.dilation], False, [0], 1,
            list(ctx.needs_input_grad[:3]))
        return *gradients, None


@contextmanager
def coalesced_dilated_bias(controller):
    changed = []
    try:
        for name, module in controller.named_modules():
            if not (isinstance(module, nn.Conv1d) and module.kernel_size == (3,)
                    and module.dilation[0] > 1 and module.stride == (1,)
                    and module.padding == module.dilation and module.groups == 1
                    and module.padding_mode == 'zeros' and module.bias is not None):
                continue
            previous, native = module.__dict__.get('forward'), module.forward

            def forward(current, value, _native=native):
                if value.is_cuda and value.dtype == torch.float32:
                    return BatchedBiasNativeBackward.apply(value, current.weight,
                        current.bias, current.dilation[0])
                return _native(value)

            module.forward = MethodType(forward, module)
            changed.append((name, module, previous))
        yield [name for name, _, _ in changed]
    finally:
        for _, module, previous in reversed(changed):
            if previous is None:
                del module.forward
            else:
                module.forward = previous
