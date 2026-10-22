"""Diagnostic tracing candidate: separate shape-only scalar constant uses."""

import io

import torch

from .soma_static_execution import trace_fixed_call as native_trace_fixed_call


def isolate_shape_constant_uses(graph):
    replacements = 0
    for node in list(graph.nodes()):
        if (node.kind() != 'prim::Constant' or 'value' not in node.attributeNames()
                or node.kindOf('value') != 't' or node.t('value').ndim != 0):
            continue
        for use in list(node.output().uses()):
            if any(value.node().kind() == 'prim::NumToTensor' for value in use.user.inputs()):
                # A constant may also index motion on CUDA. Only its shape use
                # should move to the CPU; the shared original stays unchanged.
                local = graph.insertConstant(node.t('value').cpu())
                local.node().moveBefore(use.user)
                use.user.replaceInput(use.offset, local)
                replacements += 1
    graph.lint()
    return replacements


def trace_fixed_call(function, inputs, *, cuda_constants=False):
    result = native_trace_fixed_call(function, inputs, cuda_constants=False)
    if cuda_constants:
        archive = io.BytesIO()
        torch.jit.save(result, archive)
        archive.seek(0)
        result = torch.jit.load(archive, map_location=inputs[0].device)
        isolate_shape_constant_uses(result.graph)
    return result
