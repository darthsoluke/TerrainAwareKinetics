import pytest
import torch

from scripts.audit_joint_sdf_reference_alignment import choose_reference


def test_exact_source_history_and_ties_preserve_reference_time():
    costs = (torch.arange(21) - 12).square().float()
    assert choose_reference(costs, preferred=12) == 12
    assert choose_reference(torch.zeros(21), preferred=12) == 12


def test_monotonic_cap_reports_nearest_available_without_skipping_window():
    costs = (torch.arange(33) - 25).square().float()
    assert choose_reference(costs, preferred=16) == 25
    assert choose_reference(costs, preferred=16, previous=8, max_advance=8) == 16
    earlier = torch.arange(33).float()
    assert choose_reference(earlier, preferred=16, previous=8, max_advance=8) == 8
    assert choose_reference(costs, preferred=32, previous=31, max_advance=8) == 31


def test_invalid_or_missing_reference_costs_rejected():
    for costs in (torch.empty(0), torch.tensor([float('nan')]), torch.zeros(2, 2)):
        with pytest.raises(ValueError):
            choose_reference(costs, preferred=0)
