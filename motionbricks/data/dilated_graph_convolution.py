"""Experimental replay of unchanged dilated-convolution kernels for training."""

import torch

from .dilated_bias_convolution import BatchedBiasNativeBackward as NativeConvolution


def _clone_outputs(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_outputs(item) for item in value)
    if value is None:
        return None
    raise TypeError('Unexpected convolution graph output')


class _Replay:
    def __init__(self, function, inputs, capture_error_mode):
        self.inputs = tuple(value.detach().clone() for value in inputs)
        stream = torch.cuda.Stream(device=inputs[0].device)
        current = torch.cuda.current_stream(inputs[0].device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                function(*self.inputs)
        current.wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream, capture_error_mode=capture_error_mode):
            self.output = function(*self.inputs)
        current.wait_stream(stream)

    def __call__(self, inputs):
        for destination, source in zip(self.inputs, inputs):
            destination.copy_(source)
        self.graph.replay()
        return _clone_outputs(self.output)


class _Convolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, bias, dilation, execution):
        ctx.save_for_backward(value, weight)
        ctx.dilation, ctx.bias_size, ctx.execution = dilation, list(bias.shape), execution
        return execution.forward(value, weight, bias, dilation)

    @staticmethod
    def backward(ctx, gradient):
        value, weight = ctx.saved_tensors
        mask = tuple(ctx.needs_input_grad[:3])
        def native(gradient, value, weight):
            return torch.ops.aten.convolution_backward.default(gradient, value, weight,
                ctx.bias_size, [1], [ctx.dilation], [ctx.dilation], False, [0], 1, list(mask))
        if torch.is_grad_enabled() or not ctx.execution.capture_backward:
            result = native(gradient, value, weight)
        else:
            result = ctx.execution.replay('backward', ctx.dilation, mask, native, (gradient, value, weight))
        return *result, None, None


class GraphConvolutionExecution:
    """Single-worker cache; cloned outputs never alias a later graph replay."""

    def __init__(self, capture_backward=True, *, capture_error_mode='global'):
        if capture_error_mode not in ('global', 'thread_local'):
            raise ValueError('Only global or thread-local capture checking is supported')
        self.capture_backward = capture_backward
        self.capture_error_mode = capture_error_mode
        self.calls = {}

    def replay(self, direction, dilation, mask, function, inputs):
        if any(value.device != inputs[0].device or not value.is_cuda or value.dtype != torch.float32
                for value in inputs):
            raise ValueError('Convolution replay requires matching CUDA FP32 inputs')
        signature = tuple((tuple(value.shape), tuple(value.stride()), value.dtype, value.device) for value in inputs)
        key = (direction, dilation, mask, signature)
        if key not in self.calls:
            self.calls[key] = _Replay(function, inputs, self.capture_error_mode)
        return self.calls[key](inputs)

    def forward(self, value, weight, bias, dilation):
        return self.replay('forward', dilation, (),
            lambda x, w, b: NativeConvolution.apply(x, w, b, dilation), (value, weight, bias))

    def apply(self, value, weight, bias, dilation):
        return _Convolution.apply(value, weight, bias, dilation, self)
