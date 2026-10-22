"""CPU-only orchestration fixtures for one/two genuine AR8 plan layouts."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motionbricks.mb_sonic import mb_reference as module
from motionbricks.mb_sonic.reference import NativeSomaHistory


@pytest.fixture(autouse=True)
def no_cuda_initialization():
    initialized = torch.cuda.is_initialized()
    yield
    assert torch.cuda.is_initialized() == initialized


def _history():
    values = np.arange(4 * 366, dtype=np.float32).reshape(4, 366) / 1024
    return NativeSomaHistory((np.arange(4) - 3) / 30 + .18, values, "cpu_fixture",
                             dict(kind="source_seed"))


def _plan(monkeypatch, count=None):
    representation = SimpleNamespace(skeleton=SimpleNamespace(
        bone_order_names=("Hips", *(f"bone_{index}" for index in range(1, 30)))))
    adapter = SimpleNamespace(controller=SimpleNamespace(inference=SimpleNamespace(
        global_motion_rep=representation)), scene={}, scene_id="cpu_fixture",
        identity={"test": "deterministic_cpu_only"}, canonical_scene=object(),
        scene_graph_bindings=lambda: {})
    engine = SimpleNamespace()
    monkeypatch.setattr(module, "CompiledMBNeuralExecution", lambda *_: engine)
    plan = (module.FullPlanExecution(adapter, "B1") if count is None else
            module.FullPlanExecution(adapter, "B1", block_count=count))
    calls = dict(query=[], neural=[], observe=[], fk=[])

    def goal(history, controls):
        return history[:, -1, :4] + controls

    def query(history, target, origin):
        calls["query"].append((history.clone(), target.clone(), origin.clone()))
        return (origin.clone(),)

    def neural(history, target, origin):
        calls["neural"].append(history.clone())
        future = history[:, -1:] + torch.arange(1, 29, dtype=history.dtype)[None, :, None] / 128
        world = torch.cat((history, future), dim=1)
        points = origin[:, None, None, :]
        return dict(world=world, count=torch.tensor(8), tokens=torch.arange(8)[None],
                    root_features=world[..., :5], decoded_local=world.clone(),
                    points=points, valid=torch.ones(points.shape[:-1], dtype=torch.bool),
                    world_query_xz=points[..., [0, 2]], sdf_points=points,
                    sdf_distance=torch.zeros(points.shape[:-1]), sdf_normal=points.clone(),
                    sdf_valid=torch.ones(points.shape[:-1], dtype=torch.bool), sdf_features=points.clone())

    def observe(played):
        calls["observe"].append(played.clone())
        return dict(history=played + 1 / 4096, position_error=torch.tensor(1.25e-6),
                    rotation_error=torch.tensor(2.5e-6))

    def fk(world):
        calls["fk"].append(world.clone())
        rotations = torch.eye(3).expand(1, world.shape[1], 30, 3, 3).clone()
        return dict(root_pos=world[..., :3], local_joint_rots=rotations,
                    global_joint_rots=rotations.clone(),
                    posed_joints=world[..., None, :3].expand(-1, -1, 30, -1).clone())

    plan.query = SimpleNamespace(function=query, assert_binding=lambda **_: None)
    plan.engine.call = neural
    plan.goal_function = goal
    plan.observe = SimpleNamespace(function=observe)
    plan.fk = SimpleNamespace(function=fk)
    plan.source_sha256 = "cpu_fixture"
    inputs = (torch.from_numpy(_history().features.copy())[None],
              torch.tensor([[.2, -.1, 1., 0.]]), torch.tensor([[2., 1., -3.]]))

    def execute():
        return plan._plan(*inputs, query, neural, goal, observe, fk)

    return plan, calls, inputs, execute


def test_short_plan_is_exact_two_block_prefix_and_still_observes(monkeypatch):
    short, calls, _, execute = _plan(monkeypatch, 1)
    one = execute()
    assert len(calls["query"]) == len(calls["neural"]) == len(calls["observe"]) == len(calls["fk"]) == 1
    assert one["world"].shape == (1, 28, 366)
    torch.testing.assert_close(calls["observe"][0], one["block_0_world"][:, 24:28], atol=0, rtol=0)
    assert one["observe_position_error"].item() == pytest.approx(1.25e-6)
    assert one["observe_rotation_error"].item() == pytest.approx(2.5e-6)
    assert "history_1" not in one and "block_1_world" not in one
    _, double_calls, _, double_execute = _plan(monkeypatch, 2)
    two = double_execute()
    assert len(double_calls["neural"]) == 2 and len(double_calls["observe"]) == 1
    assert two["world"].shape == (1, 52, 366)
    torch.testing.assert_close(one["world"], two["world"][:, :28], atol=0, rtol=0)
    for name in one:
        if name != "world" and not name.startswith("fk_"):
            torch.testing.assert_close(one[name], two[name], atol=0, rtol=0)
    torch.testing.assert_close(two["history_1"], two["block_0_world"][:, 24:28] + 1 / 4096, atol=0, rtol=0)
    torch.testing.assert_close(two["world"], torch.cat((two["block_0_world"][:, :28],
                                                      two["block_1_world"][:, 4:28]), dim=1), atol=0, rtol=0)
    assert short.block_count == 1


def test_default_plan_preserves_all_two_block_outputs(monkeypatch):
    default, _, _, execute = _plan(monkeypatch)
    expected, _, _, explicit_execute = _plan(monkeypatch, 2)
    actual, explicit = execute(), explicit_execute()
    assert default.block_count == expected.block_count == 2
    assert actual.keys() == explicit.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], explicit[name], atol=0, rtol=0)


class _CPUEvent:
    def __init__(self, **_):
        pass

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, _):
        return 0.


@pytest.mark.parametrize("count", [1, 2])
def test_payload_generation_layout_times_and_provenance_follow_count(monkeypatch, count):
    plan, _, inputs, execute = _plan(monkeypatch, count)
    expected = execute()
    plan.layout = module.payload_layout(expected)
    plan.call = plan._captured_plan
    monkeypatch.setattr(torch.cuda, "Event", _CPUEvent)
    audits = []

    def audit(*args):
        audits.append(args)
        return {"fixture": "cpu_orchestration"}

    monkeypatch.setattr(module, "audit_legacy_observation", audit)
    proposal = plan.generate(_history(), *inputs, "fixture-command")
    frames = 4 + 24 * count
    assert proposal.source.raw_features.shape == (frames, 366)
    assert len(proposal.raw_blocks) == len(audits) == count
    np.testing.assert_array_equal(proposal.source.raw_features, expected["world"][0].numpy())
    np.testing.assert_array_equal(proposal.source.timestamps_s, .18 + (np.arange(frames) - 3) / 30)
    assert proposal.source.trusted_until_s == pytest.approx(.18 + .8 * count)
    assert proposal.source.provenance["blocks"] == count
    assert proposal.source.provenance["future_coverage_s"] == .8 * count
    assert proposal.source.provenance["native_tokens_per_call"] == 8
    assert proposal.source.provenance["pose_passes_per_call"] == 4
    restored = module.unpack_payload_torch(plan._captured_plan(*inputs), plan.layout)
    assert restored.keys() == expected.keys()
    for name in restored:
        torch.testing.assert_close(restored[name], expected[name], atol=0, rtol=0)
    for index, block in enumerate(proposal.raw_blocks):
        assert block.raw_features.shape == (32, 366)
        np.testing.assert_array_equal(block.timestamps_s, .18 + index * .8 + (np.arange(32) - 3) / 30)
        np.testing.assert_array_equal(block.history_features, expected[f"history_{index}"][0].numpy())
        assert block.trusted_until_s == pytest.approx(.18 + (index + 1) * .8)
        assert block.provenance["history_kind"] == ("source_seed" if index == 0 else "open_loop_generated_context")


@pytest.mark.parametrize("count", [None, 1, 2])
def test_adapter_prepare_forwards_count_and_generate_requires_same_contract(monkeypatch, count):
    created, calls = [], []

    class Execution:
        def __init__(self, adapter, mode, block_count=2):
            self.block_count = block_count
            created.append(self)

        def prepare(self, seed, controls, anchor):
            assert seed.shape == (1, 4, 366) and seed.device.type == "cpu"
            return dict(blocks=self.block_count, fk_frames=4 + 24 * self.block_count,
                        future_coverage_s=.8 * self.block_count)

        def generate(self, *args):
            calls.append(args)
            return SimpleNamespace(timing_ns={})

    monkeypatch.setattr(module, "FullPlanExecution", Execution)
    monkeypatch.setattr(module.TrainedMBReferenceAdapter, "scene_query_receipt", property(lambda _: {}))
    adapter = object.__new__(module.TrainedMBReferenceAdapter)
    adapter.activate_scene = lambda **_: None
    adapter.scene_id = "cpu_fixture"
    adapter.controller = SimpleNamespace(inference=SimpleNamespace(device="cpu"))
    options = {} if count is None else dict(plan_blocks=count)
    report = adapter.prepare_compiled(_history(), velocity_world_xz_mps=[0., 0.], heading_world_rad=0.,
                                      execution="full_plan", **options)
    expected_count = 2 if count is None else count
    assert created[0].block_count == report["blocks"] == expected_count
    assert report["fk_frames"] == 4 + 24 * expected_count
    kwargs = dict(velocity_world_xz_mps=[0., 0.], heading_world_rad=0., command_id="fixture")
    proposal = adapter.generate(_history(), **kwargs, **({} if count is None else dict(blocks=count)))
    assert len(calls) == 1 and "history_preparation_wall_ns" in proposal.timing_ns
    with pytest.raises(ValueError, match="prepared full-plan"):
        adapter.generate(_history(), **kwargs, blocks=3 - expected_count)
    assert len(calls) == 1
    if expected_count == 1:
        with pytest.raises(ValueError, match="prepared full-plan"):
            adapter.generate(_history(), **kwargs)


@pytest.mark.parametrize("count", [0, 3, -1, True, 1., "1", None])
def test_invalid_plan_counts_rejected_before_loading_or_capture(count):
    with pytest.raises(ValueError, match="one or two AR8"):
        module.FullPlanExecution(None, "native", block_count=count)
    with pytest.raises(ValueError, match="one or two AR8"):
        module.TrainedMBReferenceAdapter.prepare_compiled(None, None, velocity_world_xz_mps=[0., 0.],
                                                         heading_world_rad=0., plan_blocks=count)


def test_admission_keeps_real_block_query_and_native_tolerances():
    source = textwrap.dedent(inspect.getsource(module.FullPlanExecution.prepare))
    tree = ast.parse(source)
    conditional = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                       and ast.unparse(node.test) == "self.block_count > 1")
    assert "history_1" in ast.unparse(conditional)
    assert "plan_blocks=self.block_count" in source
    assert "for block in range(self.block_count)" in source
    assert 'metrics["active_tokens_equal"]' in source and 'metrics["fk_max_error_m"] > 3e-5' in source
    assert "query_atol=1e-5" in source
    assert 'actual[name]) > 2e-5' in source
    assert "fk_frames=4 + 24 * self.block_count" in source
    neural = inspect.getsource(module.CompiledMBNeuralExecution.prepare)
    assert "range(plan_blocks if index == 0 else 1)" in neural
    assert "expected_world[:, 24:28]" in neural
    assert inspect.signature(module.TrainedMBReferenceAdapter.prepare_compiled).parameters["plan_blocks"].default == 2
    assert inspect.signature(module.TrainedMBReferenceAdapter.generate).parameters["blocks"].default == 2
