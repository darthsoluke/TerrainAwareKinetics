"""Compute the next unchanged native plan behind an immutable playback block."""
import copy

import torch

from .unified_terrain import UnifiedTerrainBackend


class CommittedUnifiedBackend(UnifiedTerrainBackend):
    committed_history_prefetch = True

    def _future_backend(self, expected):
        future = copy.copy(self)
        future.history = expected.detach().clone()
        future.runtime_controls = None if self.runtime_controls is None else dict(self.runtime_controls)
        future.inference_config = dict(self.inference_config)
        return future

    @torch.inference_mode()
    def prepare_next_plan(self, velocity, heading, observed_history, expected_published_history, overlap_frames=8,
                          *, speed_governor=None, runtime_controls=None, planning_policy=None):
        expected = expected_published_history
        if self.execution is not None:
            validation = getattr(self.execution, 'validation', None)
            # Native neural execution has no traced token-selection path to compare.
            native_geometry = (getattr(self.execution, 'neural_execution', None) == 'native'
                and validation and all(row.get('geometry_verified') for row in validation))
            if not native_geometry and (not validation or not all(row.get('tokens_equal') for row in validation)):
                raise ValueError('Committed playback requires verified acceleration')
        if (overlap_frames not in (4, 6, 8) or expected.shape != self.history.shape
                or not torch.isfinite(expected).all()
                or not torch.allclose(self.history, observed_history, atol=1e-5, rtol=0)
                or self._last_published_history is None
                or not torch.allclose(expected, self._last_published_history, atol=1e-5, rtol=0)):
            raise ValueError('Prefetch must follow the immutable published model block')
        future_backend = self._future_backend(expected)
        if runtime_controls is not None:
            future_backend.configure_runtime(runtime_controls)
        if planning_policy is not None:
            future_backend.configure_policy(planning_policy)
        if speed_governor is not None:
            future_backend.configure_governor(speed_governor)
        result = future_backend.predict_plan(velocity, heading)
        return dict(result, remaining_motion=future_backend._future_buffer.detach().clone(),
            command=future_backend._last_command, expected_prior_history=expected.detach().clone(),
            generation=dict(future_backend.last_generation, prefetch_policy='committed-model-history-v1'),
            forecast_alignment_lag_frames=0)

    @torch.inference_mode()
    def commit_prepared_plan(self, candidate):
        generation = candidate.get('generation', {})
        future, remaining = candidate.get('planned_motion'), candidate.get('remaining_motion')
        expected, published = candidate.get('expected_prior_history'), candidate.get('published_history')
        if (generation.get('prefetch_policy') != 'committed-model-history-v1'
                or not isinstance(future, torch.Tensor) or future.shape != (1, 8, self.history.shape[2])
                or not torch.isfinite(future).all() or not isinstance(remaining, torch.Tensor)
                or remaining.ndim != 3 or remaining.shape[0] != 1 or remaining.shape[2] != self.history.shape[2]
                or not torch.isfinite(remaining).all() or not isinstance(expected, torch.Tensor)
                or expected.shape != self.history.shape or not torch.allclose(self.history, expected, atol=1e-5, rtol=0)
                or not isinstance(published, torch.Tensor) or published.shape != self.history.shape
                or not torch.isfinite(published).all()):
            raise ValueError('Prepared native plan no longer follows actual playback')
        self._future_buffer = remaining.detach().clone()
        self._last_published_history = published.detach().clone()
        self._last_command = candidate['command']
        self._debug_future = candidate.get('debug')
        self.last_generation = dict(generation)
