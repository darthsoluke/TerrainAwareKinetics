import pytest
import torch

from scripts.analyze_tokenizer_mechanisms import contact_velocity_decomposition


def test_common_translation_and_opposed_deformation_are_distinguished():
    target = torch.zeros(3, 2, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)
    translation = target.clone()
    translation[:, :, 0] = torch.arange(3)[:, None] * 0.01
    common = contact_velocity_decomposition(translation, target, mask, 60)
    assert common["residual_squared_error_sum_m2ps2"] == pytest.approx(0)
    assert common["common_squared_error_sum_m2ps2"] == pytest.approx(common["squared_error_sum_m2ps2"])
    translation[:, 1] *= -1
    shape = contact_velocity_decomposition(translation, target, mask, 60)
    assert shape["common_squared_error_sum_m2ps2"] == pytest.approx(0)
    assert shape["residual_squared_error_sum_m2ps2"] > 0


def test_unmeasured_vertices_cannot_drive_the_oracle_root_correction():
    target = torch.zeros(3, 2, 2)
    prediction = target.clone()
    prediction[:, 1, 0] = torch.arange(3) * 100
    mask = torch.tensor([[True, False], [True, False]])
    result = contact_velocity_decomposition(prediction, target, mask, 60)
    assert result["squared_error_sum_m2ps2"] == 0
    assert result["unconstrained_oracle_root_offset_peak_m"] == 0


def test_empty_support_has_zero_energy_and_no_fabricated_evidence():
    target = torch.zeros(3, 2, 2)
    result = contact_velocity_decomposition(target, target, torch.zeros(2, 2, dtype=torch.bool), 60)
    assert result["measured_vertices"] == 0
    assert result["squared_error_sum_m2ps2"] == 0
