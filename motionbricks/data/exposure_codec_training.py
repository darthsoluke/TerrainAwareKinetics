"""Larger codec compute batches with the existing ordered 64-example EMA."""

from contextlib import contextmanager

import torch


def quantize_in_chunks(native, value, chunk_size=64):
    if len(value) % chunk_size:
        raise ValueError('Codec EMA must retain complete 64-example chunks')
    if len(value) == chunk_size:
        return native(value)
    outputs = [native(chunk) for chunk in value.split(chunk_size)]
    return (torch.cat([item[0] for item in outputs], dim=0),
            sum(item[1] for item in outputs) / len(outputs),
            sum(item[2] for item in outputs) / len(outputs))


@contextmanager
def ordered_ema_chunks(controller, chunk_size=64):
    quantizer = controller.codec.quantizer
    previous, native = quantizer.__dict__.get('forward'), quantizer.forward

    def forward(value):
        return quantize_in_chunks(native, value, chunk_size)

    quantizer.forward = forward
    try:
        yield
    finally:
        if previous is None:
            del quantizer.forward
        else:
            quantizer.forward = previous


def original_order_indices(schedule, dataset, first_step, count):
    """Retain the old 256-example schedule and its balanced microbatch order."""
    if count % 64:
        raise ValueError('Exposure batches must contain whole original microbatches')
    result = []
    for step in range(first_step, first_step + (count + 255) // 256):
        ids = schedule.indices(step)
        groups = {domain: [index for index in ids
            if dataset.records[dataset.ids[index]]['domain'] == domain]
            for domain in ('pfnn', 'parc', 'bones')}
        for micro in range(4):
            for domain, quota in (('pfnn', 32), ('parc', 16), ('bones', 16)):
                result.extend(groups[domain][micro * quota:(micro + 1) * quota])
    return result[:count]
