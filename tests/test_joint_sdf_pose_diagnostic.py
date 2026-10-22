from types import SimpleNamespace

import pytest
import torch

from scripts.audit_joint_sdf_pose_conditions import _snapshot, capture_pose
from scripts.audit_joint_sdf_rotation_recovery import rotation_error_degrees


def test_condition_snapshots_isolate_nested_tensors_and_preserve_types():
    original = {'root': torch.arange(6).reshape(1, 2, 3),
                'nested': [({'mask': torch.ones(2, dtype=torch.bool)},)], 'flag': False}
    copied = _snapshot(original)
    copied['root'].zero_()
    copied['nested'][0][0]['mask'].zero_()
    assert original['root'].sum().item() == 15
    assert original['nested'][0][0]['mask'].all()
    assert isinstance(copied['nested'][0], tuple)
    assert copied['flag'] is False


def test_capture_is_before_call_and_restores_original_on_error():
    def predict(batch, config, info):
        batch['root'].add_(1)
        return batch['root'], torch.ones(1), torch.ones(1, dtype=torch.bool)

    inference = SimpleNamespace(_predict_pose_tokens=predict)
    with pytest.raises(RuntimeError, match='sentinel'):
        with capture_pose(inference) as calls:
            result = inference._predict_pose_tokens({'root': torch.zeros(1)}, {}, {})
            assert calls[0]['batch']['root'].item() == 0
            result[0].add_(5)
            assert calls[0]['result'][0].item() == 1
            raise RuntimeError('sentinel')
    assert inference._predict_pose_tokens is predict


def test_rotation_error_has_identity_and_known_angle():
    identity = torch.eye(3, dtype=torch.float64)
    quarter_turn = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    assert rotation_error_degrees(identity, identity).item() == 0
    torch.testing.assert_close(rotation_error_degrees(identity, quarter_turn), torch.tensor(90., dtype=torch.float64))
    torch.testing.assert_close(rotation_error_degrees(quarter_turn, identity), torch.tensor(90., dtype=torch.float64))
