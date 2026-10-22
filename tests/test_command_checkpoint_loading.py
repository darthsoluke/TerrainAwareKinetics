from types import SimpleNamespace
from unittest.mock import patch

import pytest

from motionbricks.data.locomotion_command_adapter import COMMAND_VERSION, install_checkpoint_command


def checkpoint(declared=True, both=True):
    keys = {'locomotion_pose_condition.update.0.weight': None}
    if both:
        keys['locomotion_decoder_condition.update.0.weight'] = None
    return dict(model_state=keys, contract={'locomotion_command': {'version': COMMAND_VERSION}} if declared else {})


def test_legacy_checkpoint_does_not_change_runtime_graph():
    assert install_checkpoint_command(SimpleNamespace(), dict(model_state={}, contract={})) is False


def test_declared_adapter_installed_before_strict_load():
    with patch('motionbricks.data.locomotion_command_adapter.install_locomotion_command') as install:
        model = SimpleNamespace()
        assert install_checkpoint_command(model, checkpoint()) is True
        install.assert_called_once_with(model)


@pytest.mark.parametrize('saved', [checkpoint(declared=False), checkpoint(both=False),
    dict(model_state={}, contract={'locomotion_command': {'version': COMMAND_VERSION}})])
def test_inconsistent_checkpoint_fails_closed(saved):
    with pytest.raises(ValueError, match='disagree'):
        install_checkpoint_command(SimpleNamespace(), saved)
