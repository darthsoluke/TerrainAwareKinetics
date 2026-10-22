"""Preserve the native sampler while admitting the paper's short motion windows."""

import numpy as np
import pytest
import torch

from motionbricks.helper.data_training_util import sample_keyframes


@pytest.mark.parametrize("frames", [0, 1, 4, 8, 24])
def test_requested_keyframe_count_is_capped_by_available_slots(frames):
    motion = torch.ones(3, frames, 6)
    probabilities = [0.0] * 8 + [1.0]
    mask, masked = sample_keyframes(motion, 8, probabilities)
    assert mask.shape == (3, frames)
    assert masked.shape == motion.shape
    assert torch.all(mask.sum(-1) == min(frames, 8))
    assert torch.equal(masked, motion * mask[..., None])


def test_no_keyframe_probability_still_produces_no_conditions():
    motion = torch.ones(3, 4, 6)
    mask, masked = sample_keyframes(motion, 8, [1.0] + [0.0] * 8)
    assert not mask.any()
    assert not masked.any()


def test_long_window_preserves_original_random_sampling():
    motion = torch.arange(3 * 24 * 6).reshape(3, 24, 6).float()
    probabilities = [0.5] + [0.5 / 8] * 8
    np.random.seed(41)
    torch.manual_seed(41)
    counts = torch.tensor(np.random.choice(np.arange(9), p=probabilities, size=3)).clip(max=24)
    expected_mask = torch.rand_like(motion[:, :, 0]).sort()[1] < counts[:, None]
    np.random.seed(41)
    torch.manual_seed(41)
    mask, masked = sample_keyframes(motion, 8, probabilities)
    assert torch.equal(mask, expected_mask)
    assert torch.equal(masked, motion * expected_mask[..., None])


def test_invalid_probability_length_is_still_rejected():
    with pytest.raises(AssertionError, match="compatibility"):
        sample_keyframes(torch.ones(3, 4, 6), 8, [0.5, 0.5])
