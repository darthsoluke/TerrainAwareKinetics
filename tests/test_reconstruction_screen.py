import pytest
from scripts.evaluate_bones_tokenizer import evaluate_numerical_screen


def test_undeclared_screen_is_not_assessed_not_implicitly_passed():
    assert evaluate_numerical_screen({}, None) is None


def test_empty_declared_screen_cannot_pass():
    with pytest.raises(ValueError, match="empty screen"):
        evaluate_numerical_screen({}, {"upper_bounds": {}, "lower_bounds": {}})


def test_screen_requires_finite_values_and_does_not_approve_deployment():
    result = evaluate_numerical_screen({"pose": 0.03, "slip": float("nan")},
                                      {"upper_bounds": {"pose": 0.035, "slip": 0.02}})
    assert result["checks"] == {"pose": True, "slip": False}
    assert not result["passed"]
    assert result["not_quality_approval"]
