"""Capture frozen codec label graphs before concurrent Pose execution."""

import torch

from .dilated_bias_convolution import coalesced_dilated_bias
from .dilated_graph_context import graphed_dilated_convolutions
from .dilated_graph_convolution import GraphConvolutionExecution
from .parallel_pose_training import ParallelPoseMicrobatches
from .parallel_root_training import to_device
from .unified_full_training import prepare_native_batch


class FrozenGraphPoseMicrobatches(ParallelPoseMicrobatches):
    def __init__(self, stages, *, capture_outputs=False):
        super().__init__(stages, capture_outputs=capture_outputs)
        self.label_graphs = [GraphConvolutionExecution(capture_error_mode='thread_local') for _ in stages]
        self.label_graph_shapes = [set() for _ in stages]
        self.label_graph_warmups = 0
        self._parallel_labels_active = False
        self._original_token_methods = [stage.__dict__.get('pose_tokens') for stage in stages]
        self._native_token_methods = [stage.pose_tokens for stage in stages]
        for worker, stage in enumerate(stages):
            stage.pose_tokens = self._label_method(worker)

    def _label_method(self, worker):
        stage = self.stages[worker]
        codec = stage.controller.codec
        execution = self.label_graphs[worker]
        native = self._native_token_methods[worker]

        def labels(batch, local):
            if codec.training or any(parameter.requires_grad for parameter in codec.parameters()):
                raise ValueError('Pose label graphs require an eval-mode frozen codec')
            count = len(execution.calls)
            with graphed_dilated_convolutions(codec, execution):
                result = native(batch, local)
            if self._parallel_labels_active and len(execution.calls) != count:
                raise ValueError('A new label graph appeared during concurrent execution')
            return result

        return labels

    def _prepare_graph(self, worker, batch):
        stage, device = self.stages[worker], self.devices[worker]
        controller = stage.controller
        tokens, layouts = stage._pose_tokens, stage._pose_token_layouts
        hits, misses = stage.token_cache_hits, stage.token_cache_misses
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        # Priming must execute native encoding even when these sample IDs were
        # cached, and must not inject priming labels into the training cache.
        stage._pose_tokens, stage._pose_token_layouts = {}, {}
        try:
            with torch.no_grad(), torch.cuda.device(device), coalesced_dilated_bias(controller):
                value = to_device(batch, device)
                prepared = prepare_native_batch(controller, value['world'], value['observation'])
                stage.pose_tokens(value, prepared['codec_target']['local_motions'])
                torch.cuda.synchronize(device)
            if not torch.equal(cpu, torch.get_rng_state()) or not torch.equal(cuda, torch.cuda.get_rng_state(device)):
                raise ValueError('Frozen label graph priming changed RNG')
            self.label_graph_shapes[worker].add(len(batch['world']))
            self.label_graph_warmups += 1
        finally:
            stage._pose_tokens, stage._pose_token_layouts = tokens, layouts
            stage.token_cache_hits, stage.token_cache_misses = hits, misses
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(cuda, device)

    def backward(self, microbatches, seeds):
        for device in self.devices:
            torch.cuda.synchronize(device)
        for worker, batch in enumerate(microbatches):
            if len(batch['world']) not in self.label_graph_shapes[worker]:
                self._prepare_graph(worker, batch)
        self._parallel_labels_active = True
        result = super().backward(microbatches, seeds)
        self._parallel_labels_active = False
        return result

    def close(self):
        super().close()
        self._parallel_labels_active = False
        for stage, original in zip(self.stages, self._original_token_methods):
            if original is None:
                del stage.pose_tokens
            else:
                stage.pose_tokens = original
