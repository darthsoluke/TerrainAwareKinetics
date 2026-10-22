import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from motionbricks.data.codec_repair_pilot import (
    accounting, assert_frozen_scope, configure_scope, learning_rate, validate_plan)
from motionbricks.data.unified_full_training import NativeTrainingStage


@pytest.fixture
def plan():
    path = Path(__file__).parents[1]/'configs/pfnn_codec_repair_pilot_v1.json'
    return json.loads(path.read_text())


def test_budget_is_additional_and_capped(plan):
    validate_plan(plan)
    end = accounting(plan, 250)
    assert end == dict(actual_updates=250, actual_exposures=256000,
                      equivalent_reference_updates=1000, next_reference_step=51000)
    assert plan['total_exposures'] == 2*end['actual_exposures']
    assert learning_rate(plan, 0) == pytest.approx(5e-5)
    assert learning_rate(plan, 250) == pytest.approx(5e-7)
    for invalid in (-1, 251, 1.0, True):
        with pytest.raises(ValueError):
            accounting(plan, invalid)


@pytest.mark.parametrize('key,value', [
    ('parent_sha256', 'wrong'), ('updates_per_arm', 251), ('total_exposures', 0),
    ('mix', dict(pfnn=1, parc=0, bones=0)), ('runtime_changed', True),
    ('arms', dict(native=0, native_fk=2.0)), ('micro_batch', 1024),
])
def test_unreviewed_plan_changes_are_rejected(plan, key, value):
    changed = copy.deepcopy(plan)
    changed[key] = value
    with pytest.raises(ValueError):
        validate_plan(changed)


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('ema_calls', torch.zeros(()))

    def forward(self, value):
        if self.training:
            self.ema_calls.add_(1)
        return value


def controller():
    result = nn.Module()
    result.root = nn.Linear(2, 2)
    result.pose = nn.Linear(2, 2)
    result.codec = nn.Module()
    result.codec.encoder = nn.Linear(2, 2)
    result.codec.decoder = nn.Linear(2, 2)
    result.codec.quantizer = Quantizer()
    result.adaptation = nn.ModuleDict({name: nn.Linear(2, 2)
        for name in ('encoder', 'decoder', 'height_encoder')})
    return result


def test_real_stage_mode_transitions_keep_encoder_and_ema_frozen():
    model = controller()
    stage = NativeTrainingStage(model, 'codec')
    names = configure_scope(stage)
    assert set(names) == {f'{prefix}.{field}' for prefix in
        ('codec.decoder', 'adaptation.decoder', 'adaptation.height_encoder') for field in ('weight', 'bias')}
    for mode in (False, True, False, True):
        stage.train(mode)
        assert_frozen_scope(stage)
        model.codec.quantizer(torch.ones(1))
        assert model.codec.quantizer.ema_calls == 0
        assert model.codec.decoder.training == mode
        assert not model.root.training and not model.pose.training


def test_scope_guard_rejects_encoder_unfreeze_and_ema_training():
    model = controller()
    stage = NativeTrainingStage(model, 'codec')
    configure_scope(stage)
    model.codec.encoder.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        assert_frozen_scope(stage)
    model.codec.encoder.weight.requires_grad_(False)
    model.codec.quantizer.train()
    with pytest.raises(ValueError):
        assert_frozen_scope(stage)
