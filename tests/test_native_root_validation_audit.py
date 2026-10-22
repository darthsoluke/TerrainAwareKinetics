from types import SimpleNamespace

import pytest
import torch

from scripts.audit_native_root_validation import source_root_intervention


class FakeRoot(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('values', torch.zeros(1, 64, 5))

    def forward(self):
        return dict(pred_global_root_values=self.values, pred_num_tokens=torch.tensor([8]))


def test_offline_root_replacement_preserves_storage_tail_and_capture_order():
    root = FakeRoot()
    controller = SimpleNamespace(root=root)
    captured = []
    capture = root.register_forward_hook(lambda _m, _a, out: captured.append(out['pred_global_root_values'].clone()))
    source = torch.ones(1, 32, 5)
    with source_root_intervention(controller, source):
        output = root()
        assert torch.equal(output['pred_global_root_values'][:, :32], source)
        assert torch.equal(captured[-1][:, :32], source)
        assert not output['pred_global_root_values'][:, 32:].any()
        assert not root.values.any()
        assert output['pred_num_tokens'].item() == 8
    assert not root()['pred_global_root_values'].any()
    capture.remove()


def test_offline_root_hook_is_removed_after_failure():
    root = FakeRoot()
    controller = SimpleNamespace(root=root)
    with pytest.raises(RuntimeError):
        with source_root_intervention(controller, torch.ones(1, 32, 5)):
            raise RuntimeError('diagnostic failed')
    assert not root._forward_hooks
    assert not root()['pred_global_root_values'].any()
    with pytest.raises(ValueError):
        with source_root_intervention(controller, torch.ones(1, 31, 5)):
            pass
