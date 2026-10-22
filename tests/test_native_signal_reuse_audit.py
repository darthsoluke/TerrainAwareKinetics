import torch

from scripts.audit_native_signal_reuse import should_refresh


def test_drift_policy_only_discards_unplayed_buffer_and_uses_physical_distance():
    origin = torch.tensor([10., -12.])
    near = origin+torch.tensor([.1, .2])
    far = origin+torch.tensor([.3, .4])
    assert not should_refresh(16, None, far, .4)
    assert not should_refresh(16, origin, near, .4)
    assert not should_refresh(0, origin, far, .4)
    assert should_refresh(8, origin, far, .4)
