"""CPU native/trace contracts; CUDA admission is explicitly parent-allocated."""

from functools import partial
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from motionbricks.data.native_mb_online_conditions import VELOCITY_PROTOCOL
from motionbricks.data.native_mb_online_live import online_command_goal
from motionbricks.data.native_mb_terrain_protocol import INTENT_RANGE_POLICY
from motionbricks.data.soma_reusable_terrain_execution import ReusableTerrainExecution
from motionbricks.data.soma_static_execution import trace_fixed_call
from motionbricks.data.unified_scene_field import VolumetricEgoSampler, field_from_heightfield
from motionbricks.mb_sonic.geometry_execution import (
    FK_ATOL, OBSERVED_ATOL, QUERY_ATOL, SCENE_INPUT_NAMES, SOMA_FK_NAMES,
    ObservedHistoryExecution, SceneQueryExecution, SomaFKExecution, _compare,
    admit_observed_history, admit_scene_queries, admit_soma_fk,
    native_observed_history, native_soma_fk, storage_binding,
)
from motionbricks.mb_sonic.mb_reference import CompiledMBNeuralExecution


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "out/mb_sonic_e0_e3_20261018/scenes/000"
CUDA_ADMISSION = pytest.mark.skipif(
    os.environ.get("MB_SONIC_GEOMETRY_CUDA_TESTS") != "1",
    reason="CUDA admission requires explicit parent allocation",
)


@pytest.fixture(scope="module")
def source():
    manifest = json.loads((SCENE / "scene_manifest.json").read_text(encoding="utf-8"))
    with np.load(SCENE / manifest["source"]["scene"]["file"], allow_pickle=False) as archive:
        return {key: torch.from_numpy(archive[key].copy()) for key in (
            "heightfield_y_m", "min_point_world_xyz_m", "dx_m", "initial_history")}


def scene_callback(source, *, actual_origin, holes=False, device="cpu"):
    scene = {key: value.to(device).clone() for key, value in source.items() if key != "initial_history"}
    if holes:
        heights = torch.zeros(81, 81, device=device)
        heights[48:] = .15
        heights[45, 40] = torch.nan
        valid = torch.ones_like(heights, dtype=torch.bool)
        valid[38:43, 43:46] = False
        scene = dict(heightfield_y_m=heights, heightfield_valid=valid,
                     min_point_world_xyz_m=heights.new_tensor([-4., 0., -4.]), dx_m=heights.new_tensor(.1))
    sampler = VolumetricEgoSampler()
    engine = SimpleNamespace(
        scene=scene, field=field_from_heightfield(scene), controller=SimpleNamespace(sampler=sampler),
        inference=SimpleNamespace(_root_model=SimpleNamespace(args={"observation_range_policy": INTENT_RANGE_POLICY})),
        _scene_input_builder=ReusableTerrainExecution._scene_inputs,
    )
    callback = partial(CompiledMBNeuralExecution.scene_inputs, engine)
    history = source["initial_history"].to(device).clone()
    cases = []
    for shift, velocity, heading in (([0., 0., 0.], [0., .5], 0.),
                                     ([.17, .03, -.13], [.6, -.2], .7)):
        current = history.clone()
        current[..., :3] += current.new_tensor(shift)
        goal = online_command_goal(current[..., :5], velocity, heading, 8, VELOCITY_PROTOCOL["version"])
        inputs = (current, goal)
        if actual_origin:
            origin = current[:, -1, :3] + current.new_tensor([[.31, -.21, .19]])
            inputs = (*inputs, origin)
        cases.append(inputs)
    return callback, cases, dict(scene=scene, field=engine.field, sampler=sampler)


def feature_cases(source, representation, frames, *, device="cpu"):
    first = source["initial_history"].to(device).repeat(1, (frames + 3) // 4, 1)[:, :frames].clone()
    second = representation.change_first_heading(first, .6, False, False)
    second = second.clone()
    second[..., :3] += second.new_tensor([.21, .07, -.19])
    return first, second


def assert_native_trace(native, cases, names, atol):
    with torch.inference_mode(), full_precision_evaluation(), torch.jit.optimized_execution(False):
        native(*cases[0])
        traced = trace_fixed_call(native, cases[0], shared_cuda_constants=True)
        for inputs in cases:
            actual, expected = traced(*inputs), native(*inputs)
            if isinstance(expected, dict):
                actual, expected = tuple(actual[name] for name in names), tuple(expected[name] for name in names)
            _compare(actual, expected, names, atol)
    return traced


def test_construction_rejects_cpu_without_initializing_cuda(source):
    initialized = torch.cuda.is_initialized()
    history = source["initial_history"]
    with pytest.raises(ValueError, match="allocated FP32 CUDA"):
        SceneQueryExecution(lambda x: (x,), (history,), scene_id="000", source_sha256="cpu-test", bindings={})
    with pytest.raises(ValueError, match="allocated FP32 CUDA"):
        SomaFKExecution(build_native_representation(), history.repeat(1, 8, 1))
    with pytest.raises(ValueError, match="allocated FP32 CUDA"):
        ObservedHistoryExecution(build_native_representation(), history)
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("frames", [1, 4, 31, 53])
def test_fk_shapes_are_explicit(frames):
    with pytest.raises(ValueError, match="fixed"):
        SomaFKExecution(build_native_representation(), torch.zeros(1, frames, 366))


def test_observer_requires_four_frames():
    with pytest.raises(ValueError, match="exactly"):
        ObservedHistoryExecution(build_native_representation(), torch.zeros(1, 8, 366))


def test_scene_binding_tracks_source_field_storage_and_tensor_version():
    scene = {"height": torch.zeros(2, 2), "dx": .1}
    bindings = dict(scene=scene, field=object(), tree=object())
    execution = object.__new__(SceneQueryExecution)
    execution.scene_id, execution.source_sha256 = "000", "source-hash"
    execution.binding_signature = storage_binding(bindings)
    execution.assert_binding(scene_id="000", source_sha256="source-hash", bindings=bindings)
    for kwargs in (dict(scene_id="001", source_sha256="source-hash", bindings=bindings),
                   dict(scene_id="000", source_sha256="changed-hash", bindings=bindings),
                   dict(scene_id="000", source_sha256="source-hash", bindings={**bindings, "field": object()})):
        with pytest.raises(ValueError, match="binding changed"):
            execution.assert_binding(**kwargs)
    scene["height"].add_(1)
    with pytest.raises(ValueError, match="binding changed"):
        execution.assert_binding(scene_id="000", source_sha256="source-hash", bindings=bindings)


@pytest.mark.parametrize("actual_origin", [False, True])
@pytest.mark.parametrize("holes", [False, True])
def test_cpu_native_query_trace_preserves_dynamic_inputs_and_unknowns(source, actual_origin, holes):
    initialized = torch.cuda.is_initialized()
    callback, cases, bindings = scene_callback(source, actual_origin=actual_origin, holes=holes)
    original = [tuple(value.clone() for value in inputs) for inputs in cases]
    trace = assert_native_trace(callback, cases, SCENE_INPUT_NAMES, QUERY_ATOL)
    first, second = trace(*cases[0]), trace(*cases[1])
    assert len(first) == 13
    assert not torch.equal(first[2], second[2])
    if actual_origin:
        history, goal, origin = cases[0]
        moved = origin + origin.new_tensor([[.12, .08, -.11]])
        expected = callback(history, goal, moved)
        changed = trace(history, goal, moved)
        _compare(changed, expected, SCENE_INPUT_NAMES, QUERY_ATOL)
        assert not torch.equal(first[8], changed[8])
        torch.testing.assert_close(changed[8][0, 0].mean(0), moved[0], rtol=0, atol=2e-6)
    if holes:
        assert not first[1].all()
        assert not first[11].all()
        assert torch.equal(first[0][..., 1][~first[1]], torch.zeros_like(first[0][..., 1][~first[1]]))
        assert torch.count_nonzero(first[12][~first[11]]) == 0
    else:
        assert bindings["scene"]["heightfield_y_m"].shape == (500, 500)
    for before, inputs in zip(original, cases):
        for left, right in zip(before, inputs):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("frames", [28, 32, 52])
def test_cpu_fk_trace_is_exact_native_inverse_and_preserves_state(source, frames):
    representation = build_native_representation().eval().requires_grad_(False)
    features = feature_cases(source, representation, frames)
    state = {name: value.clone() for name, value in representation.state_dict().items()}
    original, rng = features[0].clone(), torch.random.get_rng_state().clone()
    native = partial(native_soma_fk, representation)
    trace = assert_native_trace(native, [(value,) for value in features], SOMA_FK_NAMES, FK_ATOL)
    for value in features:
        expected = representation.inverse(value, False, joint_positions_from="global_rot_data", return_all=False)
        actual = trace(value)
        assert tuple(actual) == SOMA_FK_NAMES
        for name in ("root_pos", "posed_joints"):
            assert (actual[name] - expected[name]).norm(dim=-1).max() <= FK_ATOL
    torch.testing.assert_close(features[0], original, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for name, value in representation.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_cpu_observer_trace_reencodes_native_played_fk_not_raw_channels(source):
    representation = build_native_representation().eval().requires_grad_(False)
    cases = feature_cases(source, representation, 4)
    cases[1][..., representation.indices["ric_data"]] += .4
    native = partial(native_observed_history, representation)
    names = ("history", "position_error", "rotation_error")
    trace = assert_native_trace(native, [(value,) for value in cases], names, QUERY_ATOL)
    for features in cases:
        before = features.clone()
        actual = trace(features)
        assert actual["position_error"] <= OBSERVED_ATOL
        assert actual["rotation_error"] <= OBSERVED_ATOL
        original_pose = native_soma_fk(representation, features)
        observed_pose = native_soma_fk(representation, actual["history"])
        for name in ("posed_joints", "global_joint_rots", "root_pos"):
            torch.testing.assert_close(observed_pose[name], original_pose[name], rtol=0, atol=OBSERVED_ATOL)
        torch.testing.assert_close(features, before, rtol=0, atol=0)
        ids = representation.indices["foot_contacts"]
        torch.testing.assert_close(actual["history"][..., ids], features[..., ids], rtol=0, atol=0)
    assert not torch.equal(trace(cases[1])["history"], cases[1])


@pytest.mark.parametrize("native,changed", [
    (torch.tensor([True]), torch.tensor([False])),
    (torch.tensor([17]), torch.tensor([18])),
    (torch.tensor([1.]), torch.tensor([1.00002])),
    (torch.tensor([torch.nan]), torch.tensor([torch.nan])),
    (torch.tensor([torch.inf]), torch.tensor([torch.inf])),
])
def test_admission_comparison_rejects_mask_id_float_and_nonfinite_changes(native, changed):
    with pytest.raises((AssertionError, ValueError)):
        _compare((changed,), (native,), ("native_value",), QUERY_ATOL)


@CUDA_ADMISSION
@pytest.mark.parametrize("actual_origin", [False, True])
@pytest.mark.parametrize("holes", [False, True])
def test_cuda_scene_native_trace_capture_admission_and_immutable_outputs(source, actual_origin, holes):
    callback, cases, bindings = scene_callback(source, actual_origin=actual_origin, holes=holes, device="cuda")
    manifest = json.loads((SCENE / "scene_manifest.json").read_text(encoding="utf-8"))
    source_sha256 = manifest["source"]["scene"]["sha256"]
    if holes:
        digest = hashlib.sha256()
        for name, value in bindings["scene"].items():
            digest.update(name.encode("ascii"))
            digest.update(value.cpu().numpy().tobytes())
        source_sha256 = digest.hexdigest()
    with torch.inference_mode():
        execution = SceneQueryExecution(callback, cases[0], scene_id="000" if not holes else "authored-holes",
                                        source_sha256=source_sha256, bindings=bindings)
        assert len(admit_scene_queries(execution, cases)) == 4
        first = execution(*cases[0])
        snapshot = tuple(value.clone() for value in first)
        execution(*cases[1])
        _compare(first, snapshot, SCENE_INPUT_NAMES, 0)
        execution.assert_binding(scene_id=execution.scene_id, source_sha256=execution.source_sha256, bindings=bindings)


@CUDA_ADMISSION
@pytest.mark.parametrize("frames", [28, 32, 52])
def test_cuda_fk_native_trace_capture_admission_and_immutable_outputs(source, frames):
    representation = build_native_representation().to("cuda").eval().requires_grad_(False)
    features = feature_cases(source, representation, frames, device="cuda")
    with torch.inference_mode():
        execution = SomaFKExecution(representation, features[0])
        assert len(admit_soma_fk(execution, features)) == 4
        first = execution(features[0])
        snapshot = {name: value.clone() for name, value in first.items()}
        execution(features[1])
        for name in SOMA_FK_NAMES:
            torch.testing.assert_close(first[name], snapshot[name], rtol=0, atol=0)


@CUDA_ADMISSION
def test_cuda_observer_native_trace_capture_admission_and_immutable_outputs(source):
    representation = build_native_representation().to("cuda").eval().requires_grad_(False)
    features = feature_cases(source, representation, 4, device="cuda")
    with torch.inference_mode():
        execution = ObservedHistoryExecution(representation, features[0])
        assert len(admit_observed_history(execution, features)) == 4
        first = execution(features[0])
        snapshot = first["history"].clone()
        execution(features[1])
        torch.testing.assert_close(first["history"], snapshot, rtol=0, atol=0)
