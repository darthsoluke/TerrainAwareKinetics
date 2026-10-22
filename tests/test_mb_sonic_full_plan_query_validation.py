"""CPU-only preparation contract tests; fixtures do not admit terrain or CUDA."""

import inspect
import pickle

import pytest
import torch

from motionbricks.mb_sonic.geometry_execution import SCENE_INPUT_NAMES
from motionbricks.mb_sonic.mb_reference import FullPlanExecution, FullPlanQueryParityError


FLOATS = ("points", "world_query_xz", "sdf_points", "sdf_distance", "sdf_normal", "sdf_features")
MASKS = ("valid", "sdf_valid")


@pytest.fixture
def query_case():
    execution = object.__new__(FullPlanExecution)
    history = torch.zeros(1, 4, 366)
    history[0, -1, 0] = 1e-6
    controls = torch.tensor([[.2, .3, 1., 0.]])
    origin = torch.tensor([[1., 2., 3.]])
    compiled_goal = torch.tensor([[.4, .5]])
    values = {name: torch.zeros(1, 1, 2, 3) for name in SCENE_INPUT_NAMES}
    for name in ("valid", "survey_valid", "sdf_valid"):
        values[name] = torch.tensor([[[True, False]]])
    calls = []

    def goal(current, command):
        calls.append(("goal", current, command))
        return compiled_goal

    def query(current, current_goal, anchor):
        calls.append(("query", current, current_goal, anchor))
        return tuple(values[name] for name in SCENE_INPUT_NAMES)

    execution.goal_function, execution._query_native = goal, query
    actual = dict(history_1=history, query_origin_1=origin)
    actual.update({"block_1_" + name: values[name].clone() for name in (*FLOATS, *MASKS)})
    expected = {name: value.clone() for name, value in actual.items()}
    expected["history_1"].zero_()
    expected["query_origin_1"] += 1e-6
    return execution, actual, expected, controls, compiled_goal, values, calls


def validate(fixture):
    execution, actual, expected, controls, *_ = fixture
    return execution._validate_block_queries(actual, expected, controls, case=7, block=1)


def test_uses_actual_history_compiled_goal_and_actual_origin(query_case):
    _, actual, expected, controls, goal, _, calls = query_case
    before = {name: value.clone() for name, value in actual.items()}
    report = validate(query_case)
    assert calls[0][0] == "goal" and calls[0][1] is actual["history_1"] and calls[0][2] is controls
    assert calls[1][0] == "query" and calls[1][1] is actual["history_1"]
    assert calls[1][2] is goal and calls[1][3] is actual["query_origin_1"]
    assert not torch.equal(actual["history_1"], expected["history_1"])
    assert all(value == 0 for name, value in report["same_input_query_errors"].items() if name.endswith("max_abs"))
    for name in actual:
        torch.testing.assert_close(actual[name], before[name], atol=0, rtol=0)


def test_cross_trajectory_normal_and_mask_deviations_are_recorded_not_gated(query_case):
    expected = query_case[2]
    expected["block_1_sdf_normal"] += .000223459
    expected["block_1_sdf_valid"].logical_not_()
    report = validate(query_case)
    assert report["same_input_query_errors"]["sdf_normal_max_abs"] == 0
    assert report["same_input_query_errors"]["sdf_valid_equal"] is True
    assert report["cross_trajectory_query_errors"]["sdf_normal_max_abs"] == pytest.approx(.000223459)
    assert report["cross_trajectory_query_errors"]["sdf_valid_different_count"] == 2


@pytest.mark.parametrize("name", FLOATS)
def test_same_input_float_gate_keeps_original_absolute_bound(query_case, name):
    query_case[1]["block_1_" + name].fill_(1.01e-5)
    with pytest.raises(FullPlanQueryParityError, match=f"case=7 block=1 name={name}") as failure:
        validate(query_case)
    assert (failure.value.case, failure.value.block, failure.value.name) == (7, 1, name)


@pytest.mark.parametrize("name", MASKS)
def test_same_input_mask_gate_is_exact(query_case, name):
    query_case[1]["block_1_" + name][0, 0, 0].logical_not_()
    with pytest.raises(FullPlanQueryParityError, match=f"name={name}"):
        validate(query_case)


@pytest.mark.parametrize("value", [0., 5e-6, 1e-5, -1e-5])
def test_within_original_absolute_bound_passes(query_case, value):
    query_case[1]["block_1_sdf_normal"].fill_(value)
    report = validate(query_case)
    assert report["same_input_query_errors"]["sdf_normal_max_abs"] == pytest.approx(abs(value))


def test_relative_tolerance_remains_zero(query_case):
    query_case[5]["sdf_normal"].fill_(100.)
    query_case[1]["block_1_sdf_normal"].fill_(100.0005)
    with pytest.raises(FullPlanQueryParityError, match="name=sdf_normal"):
        validate(query_case)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_query_still_fails(query_case, value):
    query_case[1]["block_1_sdf_normal"].fill_(value)
    with pytest.raises(FullPlanQueryParityError, match="name=sdf_normal"):
        validate(query_case)


def test_typed_context_survives_planner_exception_transport(query_case):
    query_case[1]["block_1_sdf_normal"].fill_(.000223459)
    with pytest.raises(FullPlanQueryParityError) as failure:
        validate(query_case)
    restored = pickle.loads(pickle.dumps(failure.value, protocol=5))
    assert type(restored) is FullPlanQueryParityError
    assert (restored.case, restored.block, restored.name) == (7, 1, "sdf_normal")
    assert str(restored) == str(failure.value)


@pytest.mark.parametrize("block", [0, 1])
def test_failure_context_identifies_either_ar_block(query_case, block):
    execution, actual, expected, controls, *_ = query_case
    actual = {name.replace("_1", f"_{block}"): value for name, value in actual.items()}
    expected = {name.replace("_1", f"_{block}"): value for name, value in expected.items()}
    actual[f"block_{block}_sdf_normal"].fill_(.000223459)
    with pytest.raises(FullPlanQueryParityError, match=f"case=3 block={block} name=sdf_normal"):
        execution._validate_block_queries(actual, expected, controls, case=3, block=block)


def test_validation_preserves_rng_and_cuda_initialization(query_case):
    rng, initialized = torch.random.get_rng_state().clone(), torch.cuda.is_initialized()
    validate(query_case)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.cuda.is_initialized() == initialized


def test_validation_only_wired_into_preparation_and_original_gates_retained():
    prepare = inspect.getsource(FullPlanExecution.prepare)
    assert "self._validate_block_queries(actual, expected, inputs[1], case=index, block=block)" in prepare
    assert 'not metrics["active_tokens_equal"]' in prepare
    assert 'metrics["fk_max_error_m"] > 3e-5' in prepare
    assert 'fk_error > 3e-5' in prepare
    assert 'actual[name]) > 2e-5' in prepare
    assert 'expected[f"query_origin_{block}"],' in prepare and "atol=3e-5, rtol=0" in prepare
    for method in (FullPlanExecution._plan, FullPlanExecution._captured_plan, FullPlanExecution.generate):
        assert "_validate_block_queries" not in inspect.getsource(method)
