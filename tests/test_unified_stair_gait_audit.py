import pytest
import torch

from scripts.audit_unified_stair_gait import source_token_intervention


def test_source_token_diagnostic_is_scoped_and_preserves_native_outputs():
    class Inference:
        def _predict_pose_tokens(self):
            return self.tokens, self.pose, self.mask

    model = Inference()
    model.tokens = torch.zeros(1, 16, 8, dtype=torch.long)
    model.pose = torch.randn(1, 64, 12)
    model.mask = torch.ones(1, 64, dtype=torch.bool)
    source = torch.ones(1, 8, 8, dtype=torch.long)
    with pytest.raises(RuntimeError, match='cleanup'):
        with source_token_intervention(model, source):
            tokens, pose, mask = model._predict_pose_tokens()
            torch.testing.assert_close(tokens[:, :8], source, atol=0, rtol=0)
            assert tokens[:, 8:].count_nonzero() == 0
            assert model.tokens.count_nonzero() == 0
            assert pose is model.pose and mask is model.mask
            raise RuntimeError('cleanup')
    assert '_predict_pose_tokens' not in model.__dict__
    assert model._predict_pose_tokens()[0] is model.tokens
