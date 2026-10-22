"""Shared tracing helpers for verified SOMA inference implementations."""

import io
import warnings

import torch


class _FixedCall(torch.nn.Module):
    def __init__(self,function):
        super().__init__();self.function=function
    def forward(self,*inputs):return self.function(*inputs)


def trace_fixed_call(function,inputs,*,cuda_constants=False,shared_cuda_constants=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',torch.jit.TracerWarning)
        result=torch.jit.freeze(torch.jit.trace(_FixedCall(function).eval(),inputs,strict=False,check_trace=False),
                                optimize_numerics=not shared_cuda_constants)
    if cuda_constants and not shared_cuda_constants:
        archive=io.BytesIO();torch.jit.save(result,archive);archive.seek(0)
        result=torch.jit.load(archive,map_location=inputs[0].device)
    if cuda_constants or shared_cuda_constants:
        # Keep MHA shape scalars on the CPU while CUDA array-index constants
        # stay on the GPU, matching the established native execution helper.
        for node in result.graph.nodes():
            if node.kind()=='prim::Constant' and 'value' in node.attributeNames() and node.kindOf('value')=='t':
                value=node.t('value')
                shape_scalar=any(any(item.node().kind()=='prim::NumToTensor' for item in use.user.inputs())
                                 for use in node.output().uses())
                value=value.cpu() if value.ndim==0 and shape_scalar else value.to(inputs[0].device)
                node.t_('value',value)
    return result


def use_dense_transformers(inference):
    for module in inference.modules():
        if isinstance(module,torch.nn.TransformerEncoder):
            if getattr(module,'_soma_dense_installed',False):
                continue
            native,zero_padding=module.forward,module.use_nested_tensor
            module._soma_native_forward=native
            module._soma_original_nested_tensor=zero_padding
            def forward(src,mask=None,src_key_padding_mask=None,is_causal=None,native=native,zero_padding=zero_padding):
                output=native(src,mask=mask,src_key_padding_mask=src_key_padding_mask,is_causal=is_causal)
                if zero_padding and src_key_padding_mask is not None:
                    output=output.masked_fill(src_key_padding_mask[...,None],0)
                return output
            module.use_nested_tensor=False;module.forward=forward
            module._soma_dense_installed=True


def restore_native_transformers(inference):
    for module in inference.modules():
        if isinstance(module,torch.nn.TransformerEncoder) and getattr(module,'_soma_dense_installed',False):
            module.forward=module._soma_native_forward
            module.use_nested_tensor=module._soma_original_nested_tensor
            module._soma_dense_installed=False


def validate_observed_fk(result):
    valid=(torch.isfinite(result['history']).all() & (result['position_error']<=2e-5) & (result['rotation_error']<=2e-5))
    if not valid:raise ValueError('Observed-history re-encoding changed the displayed FK state')
