import pytest
import torch

from scripts.audit_unified_partial_targets import complete_root_targets_only


class Recorder(torch.nn.Module):
    def forward(self, value, **kwargs):
        return value, kwargs


def test_ablation_keeps_factual_and_complete_targets_and_external_conditions():
    module = Recorder()
    mask = torch.zeros(1, 32, 5, dtype=torch.bool)
    mask[:, :4] = True
    mask[:, 12] = True
    mask[:, 31, 3:] = True
    external = torch.randn(1, 32, 8)
    target = torch.randn(1, 32, 5)
    kwargs = dict(target_component_mask=mask, has_target_cond=mask.any(-1), target_cond=target, external_cond=external)
    with complete_root_targets_only(module) as audit:
        _, changed = module(torch.randn(1, 8, 8), **kwargs)
        assert torch.equal(changed['target_component_mask'][:, :4], mask[:, :4])
        assert changed['target_component_mask'][:, 12].all()
        assert not changed['target_component_mask'][:, 31].any()
        assert changed['target_cond'] is target and changed['external_cond'] is external
        assert audit[0]['partial_future_frames'] == 1
    assert mask[:, 31, 3:].all()
    _, normal = module(torch.zeros(1, 8, 8), **kwargs)
    assert normal['target_component_mask'] is mask


def test_partial_target_ablation_is_removed_on_exception():
    module = Recorder()
    with pytest.raises(RuntimeError):
        with complete_root_targets_only(module):
            raise RuntimeError('stop')
    assert not module._forward_pre_hooks
