import torch

from scripts.probe_parc_stop_history_guidance import prefix_comparison


def test_quaternion_sign_comparison_preserves_inputs():
    factual = torch.tensor([[[.5, .5, .5, .5], [1., 0., 0., 0.]]])
    actual = -factual.clone()
    original = actual.clone()
    aligned, reference = prefix_comparison(actual, factual, rotation=True)
    torch.testing.assert_close(aligned, factual, atol=0, rtol=0)
    torch.testing.assert_close(actual, original, atol=0, rtol=0)
    assert reference.data_ptr() == factual.data_ptr()


def test_real_rotation_or_position_change_is_not_hidden():
    factual = torch.tensor([[[1., 0., 0., 0.]]])
    actual = torch.tensor([[[0., 1., 0., 0.]]])
    aligned, reference = prefix_comparison(actual, factual, rotation=True)
    assert not torch.allclose(aligned, reference)
    positions = torch.tensor([[[1., 2., 3.]]])
    changed = -positions
    aligned, reference = prefix_comparison(changed, positions)
    assert aligned.data_ptr() == changed.data_ptr()
    assert not torch.allclose(aligned, reference)
