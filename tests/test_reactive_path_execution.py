"""CPU query/composition contracts; no pretrained models or CUDA admission."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from motionbricks.data.native_mb_online_conditions import VELOCITY_PROTOCOL
from motionbricks.data.native_mb_terrain_protocol import INTENT_RANGE_POLICY
from motionbricks.data.reactive_path_execution import (
    COARSE_NAMES, PATH_NAMES, ReactivePathExecution, _assert_output_contract, _assert_queries,
)
from motionbricks.data.root_path_split_execution import SplitRootPathExecution
from motionbricks.data.root_path_terrain import PredictedRootObservation
from motionbricks.data.soma_reusable_terrain_execution import ReusableTerrainExecution
from motionbricks.data.unified_scene_field import VolumetricEgoSampler, field_from_heightfield


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU contracts must not call CUDA')
    for name in ('_lazy_init', 'get_rng_state', 'set_rng_state', 'synchronize', 'Stream', 'CUDAGraph'):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def observe(raw):
    history = raw.clone()
    history[..., 100] = history[..., 0] + .125
    return dict(history=history, position_error=raw.new_zeros(()), rotation_error=raw.new_zeros(()))


def root(history, goal, query, *coarse):
    time = (torch.arange(32, device=history.device, dtype=history.dtype) - 3) / 30
    values = history.new_zeros(1, 32, 5)
    values[..., 0] = history[:, -1:, 0] - history[:, :1, 0] + time * goal[:, :1]
    values[..., 2] = history[:, -1:, 2] - history[:, :1, 2] + time * goal[:, 1:2]
    values[..., 1] = history[:, -1:, 1]
    values[..., 3] = 1
    return dict(root=values, local=history.new_zeros(1, 32, 4),
        count=torch.full((1,), 8, device=history.device, dtype=torch.long))


def pose(history, goal, query, count, predicted, local, *values):
    motion = history[:, -1:].expand(-1, 32, -1).clone()
    motion[..., 0] = predicted[..., 0] + history[:, :1, 0]
    motion[..., 2] = predicted[..., 2] + history[:, :1, 2]
    motion[..., 5] = values[8][..., 1].mean()
    motion[..., 6] = values[13].mean()
    tokens = torch.zeros(1, 8, 8, dtype=torch.long, device=history.device)
    return dict(motion=motion, count=count, root=predicted, tokens=tokens)


def geometry(future):
    joints = future[0, :, None, :3].expand(-1, 30, -1).clone()
    rig = torch.eye(4, device=future.device).expand(24, 77, 4, 4).clone()
    rig[..., :3, 3] = future[0, :, None, :3]
    return dict(rig=rig, joints=joints)


@pytest.fixture
def execution():
    heights = torch.zeros(101, 101)
    heights[54:] = .2
    known = torch.ones_like(heights, dtype=torch.bool)
    known[51:54, 52:55] = False
    scene = dict(heightfield_y_m=heights, heightfield_valid=known,
        min_point_world_xyz_m=torch.tensor([-5., 0., -5.]), dx_m=.1)
    seed = torch.zeros(1, 4, 366)
    seed[..., 1], seed[..., 3] = 1., 1.
    field = field_from_heightfield(scene)
    inference = SimpleNamespace(_root_model=SimpleNamespace(args={'observation_range_policy': INTENT_RANGE_POLICY}),
        global_motion_rep=SimpleNamespace(unnormalize=lambda value: value))
    controller = SimpleNamespace(frames=32, inference=inference, sampler=VolumetricEgoSampler(),
        _fields={id(heights): (scene, field)}, scene_field=lambda _: field)
    observer = object.__new__(PredictedRootObservation)
    observer.controller = controller
    runner = SimpleNamespace(inference=inference, controller=controller, scene=scene, field=field,
        observer=observer, protocol_version=VELOCITY_PROTOCOL['version'],
        graphs={8: (SimpleNamespace(function=root), SimpleNamespace(function=pose))},
        _observe=SimpleNamespace(function=observe))
    runner._scene_inputs = MethodType(ReusableTerrainExecution._scene_inputs, runner)
    runner._path_inputs = MethodType(SplitRootPathExecution._path_inputs, runner)
    inference._terrain_execution = runner
    runtime = SimpleNamespace(inference=inference, terrain_initial_history=seed, terrain_scene=scene,
        current_map='cpu-scene', scene_revision=1,
        acceptance_receipt=dict(scene_sha256='cpu-source', checkpoint_sha256='cpu-weights'))
    return ReactivePathExecution(runtime, geometry)


def prepare_cpu_functions(execution):
    raw = execution.runtime.terrain_initial_history
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        execution._trace_queries(raw, torch.zeros(1, 3))


@pytest.mark.parametrize('shift,command', [
    ((0., 0., 0.), (0., 0., 0.)), ((0., 0., 0.), (0., 1.6, 0.)),
    ((.17, .03, -.13), (.6, -.2, .7)), ((-.21, -.02, .12), (-1.2, .4, -1.1)),
])
def test_exact_query_traces_preserve_dynamic_history_input_and_post_root_path(execution, shift, command):
    prepare_cpu_functions(execution)
    raw = execution.runtime.terrain_initial_history.clone()
    raw[..., :3] += raw.new_tensor(shift)
    controls = raw.new_tensor([command])
    original = raw.clone()
    with torch.inference_mode():
        fused = execution._fused(raw, controls)
        expected = execution._assemble(raw, controls, observe=observe, goal=execution._goal,
            coarse=execution._coarse_native, path=execution._path_native, root=root, pose=pose)
        _assert_output_contract(fused)
        _assert_queries(fused, expected)
    for key in fused:
        torch.testing.assert_close(fused[key], expected[key], atol=1e-5, rtol=0)
    torch.testing.assert_close(raw, original, atol=0, rtol=0)
    assert all('coarse_' + key in fused for key in COARSE_NAMES)
    assert all(key in fused for key in PATH_NAMES)
    assert fused['path_world_xz_m'].shape == (1, 25, 2)
    torch.testing.assert_close(fused['rig'][..., :3, 3], fused['motion'][0, 4:28, None, :3].expand(-1, 77, -1))
    torch.testing.assert_close(fused['history'][..., 100], raw[..., 0] + .125)


def test_coarse_and_post_root_masks_remain_unknown_and_are_checked_exactly(execution):
    prepare_cpu_functions(execution)
    with torch.inference_mode():
        result = execution._fused(execution.runtime.terrain_initial_history, torch.tensor([[.8, .6, .3]]))
    assert not result['coarse_valid'].all()
    changed = dict(result, sdf_valid=~result['sdf_valid'])
    with pytest.raises(AssertionError):
        _assert_queries(changed, result)
    changed = dict(result, sdf_normal=result['sdf_normal'] + 2e-5)
    with pytest.raises(AssertionError):
        _assert_queries(changed, result)


def test_rotated_stop_sensor_preserves_minimum_range_at_float32_boundary(execution):
    prepare_cpu_functions(execution)
    with torch.inference_mode():
        for yaw in torch.linspace(-3.14, 3.14, 41):
            history = execution.runtime.terrain_initial_history.clone()
            history[..., 3] = torch.cos(yaw)
            history[..., 4] = torch.sin(yaw)
            goal = execution._goal(history, torch.zeros(1, 3))
            coarse = execution._coarse_native(history, goal, execution.query)
            values = root(history, goal, execution.query, *coarse)
            result = execution._path_native(history, values['root'], *coarse)
            assert torch.isfinite(result[0]).all()


def test_post_root_trace_resamples_a_changed_root_with_identical_history_and_command(execution):
    prepare_cpu_functions(execution)
    with torch.inference_mode():
        history = observe(execution.runtime.terrain_initial_history)['history']
        goal = execution._goal(history, torch.tensor([[0., 1., 0.]]))
        coarse = execution._coarse_native(history, goal, execution.query)
        predicted = root(history, goal, execution.query, *coarse)['root']
        first = execution.path_function(history, predicted, *coarse)
        shifted = predicted.clone()
        shifted[:, 4:28, 0] += .25
        second = execution.path_function(history, shifted, *coarse)
        native = execution._path_native(history, shifted, *coarse)
        for actual, expected in zip(second, native):
            torch.testing.assert_close(actual, expected, atol=0 if actual.dtype == torch.bool else 1e-5, rtol=0)
    assert (first[2] - second[2]).abs().max() > .01


def test_fixed_input_admission_records_cross_trajectory_nearest_cell_boundary(execution):
    runner = execution.runner
    runner.scene['heightfield_valid'][:, 52:] = False
    field = field_from_heightfield(runner.scene)
    runner.field = field
    runner.controller._fields[id(runner.scene['heightfield_y_m'])] = (runner.scene, field)
    runner.controller.scene_field = lambda _: field
    prepare_cpu_functions(execution)

    def boundary_root(sign):
        def call(*args):
            result = root(*args)
            result['root'][:, 4:28, 0] = .15 + sign * 1e-6
            result['root'][:, 4:28, 2] = .6
            return result
        return call

    with torch.inference_mode():
        def predict(sign):
            return execution._assemble(execution.runtime.terrain_initial_history, torch.zeros(1, 3),
                observe=observe, goal=execution.goal_function, coarse=execution.coarse_function,
                path=execution.path_function, root=boundary_root(sign), pose=pose)
        actual, expected = predict(1), predict(-1)
        assert (actual['root'] - expected['root']).abs().max() < 3e-5
        with pytest.raises(AssertionError):
            _assert_queries(actual, expected)
        proof = execution._admit_queries(actual, expected)
    assert proof['masks_equal']
    assert max(proof['same_input_query_max_abs'].values()) <= 1e-5
    assert proof['cross_trajectory_query_errors']['valid']['different_count'] > 0


@pytest.mark.parametrize('key', ['coarse_sdf_distance', 'sdf_normal', 'valid'])
def test_fixed_input_admission_still_rejects_query_implementation_drift(execution, key):
    prepare_cpu_functions(execution)
    with torch.inference_mode():
        actual = execution._fused(execution.runtime.terrain_initial_history, torch.zeros(1, 3))
        changed = dict(actual)
        changed[key] = ~actual[key] if actual[key].dtype == torch.bool else actual[key] + 2e-5
        with pytest.raises(AssertionError):
            execution._admit_queries(changed, changed)


@pytest.mark.parametrize('coarse_known,path_known', [(False, True), (True, False), (False, False), (True, True)])
def test_live_known_check_preserves_any_known_per_batch_for_both_observations(execution, coarse_known, path_known):
    prepare_cpu_functions(execution)
    with torch.inference_mode():
        raw, controls = execution.runtime.terrain_initial_history, torch.zeros(1, 3)
        baseline = execution._fused(raw, controls)
        coarse = [baseline['coarse_' + key] for key in COARSE_NAMES]
        fine = [baseline[key] for key in PATH_NAMES]
        coarse[11], fine[7] = torch.zeros_like(coarse[11]), torch.zeros_like(fine[7])
        coarse[11].flatten()[0] = coarse_known
        fine[7].flatten()[0] = path_known
        observed = {key: baseline[key] for key in ('history', 'position_error', 'rotation_error')}
        result = execution._outputs(raw, controls, observed, baseline['history'], baseline['goal'],
            baseline['query_controls'], coarse, fine, baseline)
    assert bool(result['observation_known']) == (coarse_known and path_known)
    if coarse_known and path_known:
        _assert_output_contract(result)
    else:
        with pytest.raises(ValueError, match='observations are unknown'):
            _assert_output_contract(result)


@pytest.mark.parametrize('failure', ['token', 'fk'])
def test_preparation_rejects_neural_or_fk_drift_before_admission(execution, failure):
    prepare_cpu_functions(execution)
    with torch.inference_mode():
        result = execution._fused(execution.runtime.terrain_initial_history, torch.zeros(1, 3))
    changed = dict(result)
    if failure == 'token':
        changed['tokens'] = result['tokens'].clone()
        changed['tokens'][:, 0, 0] += 1
        message = 'pose tokens'
    else:
        execution.runner.inference.global_motion_rep.inverse = lambda motion, *args, **kw: {
            'posed_joints': motion[..., :3].unsqueeze(-2).expand(-1, -1, 30, -1)}
        changed['motion'] = result['motion'].clone()
        changed['motion'][..., 0] += 1e-4
        message = 'FK mismatch'
    with pytest.raises(ValueError, match=message):
        execution._admit(changed, result)
    assert execution.proof is None


def test_single_replay_api_preserves_full_motion_and_does_not_advance_history(execution):
    prepare_cpu_functions(execution)
    seed = execution.runtime.terrain_initial_history
    with torch.inference_mode():
        expected = execution._fused(seed, torch.zeros(1, 3))
    received = []
    execution.proof = dict(passed=True)
    execution.binding = execution._scene_binding()
    execution.call = lambda *args: received.append(args) or {key:value.clone() for key,value in expected.items()}
    result = execution(seed, torch.zeros(1, 3))
    assert len(received) == 1 and received[0][0] is seed
    assert result['motion'].shape == (1, 32, 366)
    assert result['rig'].shape == (24, 77, 4, 4)
    assert result['joints'].shape == (24, 30, 3)
    torch.testing.assert_close(seed, torch.zeros_like(seed).index_fill(2, torch.tensor([1, 3]), 1))
    with pytest.raises(ValueError, match='controls'):
        execution(seed, torch.zeros(2, 3))
    with pytest.raises(ValueError, match='controls'):
        execution(seed.double(), torch.zeros(1, 3))


@pytest.mark.parametrize('change', ['map', 'revision', 'source', 'height', 'inplace', 'field', 'root', 'observer'])
def test_replay_rejects_changed_scene_storage_source_and_functions(execution, change):
    prepare_cpu_functions(execution)
    execution.proof = dict(passed=True)
    execution.binding = execution._scene_binding()
    runtime, runner = execution.runtime, execution.runner
    if change == 'map': runtime.current_map = 'other'
    elif change == 'revision': runtime.scene_revision += 1
    elif change == 'source': runtime.acceptance_receipt['scene_sha256'] = 'other'
    elif change == 'height': runner.scene['heightfield_y_m'] = runner.scene['heightfield_y_m'].clone()
    elif change == 'inplace': runner.scene['heightfield_y_m'].add_(.01)
    elif change == 'field': runner.field = SimpleNamespace()
    elif change == 'root': runner.graphs[8] = (SimpleNamespace(function=lambda *args: None), runner.graphs[8][1])
    else: runner._observe = SimpleNamespace(function=lambda *args: None)
    with pytest.raises(ValueError, match='binding changed'):
        execution(runtime.terrain_initial_history, torch.zeros(1, 3))


def test_unadmitted_or_cpu_prepare_never_initializes_cuda(execution):
    with pytest.raises(ValueError, match='successful prepare'):
        execution(execution.runtime.terrain_initial_history, torch.zeros(1, 3))
    with pytest.raises(ValueError, match='allocated CUDA'):
        execution.prepare()


@pytest.mark.parametrize('failure', ['count', 'finite', 'position', 'rotation', 'rig_frames'])
def test_admission_status_rejects_invalid_horizon_history_or_geometry(execution, failure):
    prepare_cpu_functions(execution)
    with torch.inference_mode():
        result = execution._fused(execution.runtime.terrain_initial_history, torch.zeros(1, 3))
    if failure == 'count': result['count'] = torch.tensor([12])
    elif failure == 'finite': result['finite'] = torch.tensor(False)
    elif failure == 'position': result['position_error'] = torch.tensor(2.01e-5)
    elif failure == 'rotation': result['rotation_error'] = torch.tensor(float('nan'))
    else: result['rig'] = result['rig'][:8]
    with pytest.raises(ValueError):
        _assert_output_contract(result)
