from types import SimpleNamespace

import pytest
import torch

from scripts.audit_native_terrain_gaps import pose_ablation


class Residual(torch.nn.Module):
    def forward(self, value, *_args):
        return value+2


def test_offline_ablation_is_scoped_restored_and_does_not_modify_parameters():
    old, volume = Residual(), Residual()
    controller = SimpleNamespace(pose=SimpleNamespace(_terrain_cross_attention=old), adaptation={'pose': volume})
    value = torch.ones(1, 4, 8)
    with pytest.raises(RuntimeError):
        with pose_ablation(controller, 'legacy_pose_off') as calls:
            assert torch.equal(old(value), value)
            assert torch.equal(volume(value), value+2)
            assert calls[0] == 1
            raise RuntimeError('Interrupted diagnostic')
    assert torch.equal(old(value), value+2)
    with pose_ablation(controller, 'both_pose_off'):
        assert torch.equal(old(value), value) and torch.equal(volume(value), value)
    assert torch.equal(old(value), value+2) and torch.equal(volume(value), value+2)
