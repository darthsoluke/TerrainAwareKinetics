import pytest
import torch
from torch import nn

from scripts.train_native_mb_control import terrain_gradient_evidence


def network():
    result = nn.Module()
    result._terrain_encoder = nn.Linear(2, 2)
    result._terrain_cross_attention = nn.Linear(2, 2)
    result._terrain_root_height_gate = nn.Parameter(torch.zeros(()))
    return result


def test_no_gradients_is_not_learning_evidence():
    assert terrain_gradient_evidence(network()) == dict(encoder=False, attention=False, height=False)


def test_only_active_paths_are_counted():
    model = network()
    model._terrain_encoder(torch.ones(1, 2)).sum().backward()
    assert terrain_gradient_evidence(model) == dict(encoder=True, attention=False, height=False)
    for value in model.parameters():
        value.grad = torch.zeros_like(value)
    assert not any(terrain_gradient_evidence(model).values())


def test_bad_terrain_gradient_fails():
    model = network()
    model._terrain_encoder.weight.grad = torch.full_like(model._terrain_encoder.weight, torch.nan)
    with pytest.raises(RuntimeError, match='Nonfinite terrain gradient'):
        terrain_gradient_evidence(model)
