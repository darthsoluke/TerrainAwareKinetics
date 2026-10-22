"""CPU-only binding checks; no scene query kernels or CUDA graphs constructed."""

from dataclasses import replace
import inspect
from types import SimpleNamespace
import weakref

import pytest
import torch

from motionbricks.data.unified_scene_field import VolumetricEgoSampler
from motionbricks.mb_sonic.geometry_execution import SceneQueryExecution, storage_binding
from motionbricks.mb_sonic.mb_reference import FullPlanExecution, TrainedMBReferenceAdapter


class Field:
    pass


class ConstructionList(list):
    forbid_iteration = False

    def __iter__(self):
        if self.forbid_iteration:
            raise AssertionError("Replay walked construction-only prism rows")
        return super().__iter__()


@pytest.fixture
def bound_adapter():
    adapter = object.__new__(TrainedMBReferenceAdapter)
    adapter.scene_id = "003"
    adapter.scene = dict(heightfield_y_m=torch.zeros(4, 4), dx_m=torch.tensor(.1))
    field = Field()
    field.rectangles = ConstructionList([(0, 1, 0, 1, .2), (1, 2, 0, 1, .3)])
    field.holes = ConstructionList([(2, 3, 0, 1, 0.)])
    field.centers, field.owners = torch.zeros(2, 2), torch.arange(2)
    field.cache = {("cpu", "float32"): (torch.zeros(2, 3), torch.ones(2, dtype=torch.bool))}
    field.field_id = 4
    field.solid_tree, field.air_tree = SimpleNamespace(id=101), SimpleNamespace(id=102)
    adapter.scene_field = field
    sampler = VolumetricEgoSampler()
    sampler._grids[("cpu", "float32")] = torch.zeros(8, 3)
    adapter.controller = SimpleNamespace(sampler=sampler, _fields={})
    adapter._scene_query_receipt = dict(backend="warp_bvh_sphere_exact", source_sha256="frozen",
                                      field_kind="SphereBVHHeightfieldPrismSDF")
    query = object.__new__(SceneQueryExecution)
    query.scene_id, query.source_sha256 = adapter.scene_id, "frozen"
    query.bindings = adapter.scene_graph_bindings()
    query.binding_signature = storage_binding(query.bindings)
    query.replay_binding_signature = storage_binding(query.bindings, compact=True)
    adapter._full_plan = SimpleNamespace(query=query, source_sha256="frozen")
    return adapter


def check(adapter, *, replay=True):
    adapter._full_plan.query.assert_binding(scene_id=adapter.scene_id, source_sha256="frozen",
        bindings=adapter.scene_graph_bindings(), replay=replay)


def test_replay_never_iterates_construction_rows(bound_adapter):
    field = bound_adapter.scene_field
    field.rectangles.forbid_iteration = field.holes.forbid_iteration = True
    check(bound_adapter)


@pytest.mark.parametrize("name", ["rectangles", "holes"])
@pytest.mark.parametrize("change", ["replace", "length", "content"])
def test_construction_lists_have_atomic_replay_and_deep_activation_checks(bound_adapter, name, change):
    rows = getattr(bound_adapter.scene_field, name)
    if change == "replace":
        setattr(bound_adapter.scene_field, name, list(rows))
    elif change == "length":
        rows.append(rows[0])
    else:
        rows[0] = (20, 21, 0, 1, .8)
    if change == "content":
        check(bound_adapter)
    else:
        with pytest.raises(ValueError, match="binding changed"):
            check(bound_adapter)
    with pytest.raises(ValueError, match="binding changed"):
        bound_adapter.activate_scene()
    assert not bound_adapter.controller._fields


@pytest.mark.parametrize("change", [
    "scene_mutation", "scene_replacement", "scene_same_storage", "centers_mutation", "owners_mutation",
    "cache_mutation", "cache_replacement", "field_owner", "field_id", "solid_id", "air_owner",
    "sampler_owner", "sampler_config", "sampler_cache", "sampler_cache_replacement", "backend", "source",
])
def test_replay_still_rejects_runtime_binding_changes(bound_adapter, change):
    adapter, field = bound_adapter, bound_adapter.scene_field
    sampler = adapter.controller.sampler
    if change == "scene_mutation":
        adapter.scene["heightfield_y_m"].add_(1)
    elif change == "scene_replacement":
        adapter.scene["heightfield_y_m"] = adapter.scene["heightfield_y_m"].clone()
    elif change == "scene_same_storage":
        adapter.scene["heightfield_y_m"] = adapter.scene["heightfield_y_m"].view(4, 4)
    elif change == "centers_mutation":
        field.centers.add_(1)
    elif change == "owners_mutation":
        field.owners.add_(1)
    elif change == "cache_mutation":
        field.cache[("cpu", "float32")][0].add_(1)
    elif change == "cache_replacement":
        field.cache[("cpu", "float32")] = (torch.zeros(2, 3), torch.ones(2, dtype=torch.bool))
    elif change == "field_owner":
        adapter.scene_field = Field()
        vars(adapter.scene_field).update(vars(field))
    elif change == "field_id":
        field.field_id += 1
    elif change == "solid_id":
        field.solid_tree.id += 1
    elif change == "air_owner":
        field.air_tree = SimpleNamespace(id=102)
    elif change == "sampler_owner":
        adapter.controller.sampler = VolumetricEgoSampler()
    elif change == "sampler_config":
        sampler.config = replace(sampler.config, body_halfwidth_m=.9)
    elif change == "sampler_cache":
        sampler._grids[("cpu", "float32")].add_(1)
    elif change == "sampler_cache_replacement":
        sampler._grids[("cpu", "float32")] = torch.zeros(8, 3)
    elif change == "backend":
        adapter._scene_query_receipt["backend"] = "other"
    else:
        adapter._scene_query_receipt["source_sha256"] = "other"
    with pytest.raises(ValueError, match="binding changed"):
        check(adapter)


@pytest.mark.parametrize("key,value", [("scene_id", "other"), ("source_sha256", "other")])
def test_compact_check_keeps_explicit_scene_and_source_identity(bound_adapter, key, value):
    arguments = dict(scene_id="003", source_sha256="frozen", bindings=bound_adapter.scene_graph_bindings(), replay=True)
    arguments[key] = value
    with pytest.raises(ValueError, match="binding changed"):
        bound_adapter._full_plan.query.assert_binding(**arguments)


def test_explicit_activation_keeps_owner_and_deep_checks(bound_adapter):
    adapter = bound_adapter
    owner = weakref.ref(adapter.scene_field)
    adapter.activate_scene()
    assert adapter.controller._fields[id(adapter.scene["heightfield_y_m"])][1] is owner()
    assert adapter._full_plan.query.bindings["field_owner"] is owner()
    check(adapter)


def test_hot_generation_uses_compact_only_and_preparation_keeps_deep_activation():
    hot = inspect.getsource(TrainedMBReferenceAdapter.generate)
    assert "self.activate_scene(validate_binding=False)" in hot
    assert "self.activate_scene()" in inspect.getsource(TrainedMBReferenceAdapter.prepare_compiled)
    assert "replay=True" in inspect.getsource(FullPlanExecution.generate)
    assert "validate_binding=True" in inspect.getsource(TrainedMBReferenceAdapter.activate_scene)


def test_cpu_binding_checks_preserve_rng_and_cuda_state(bound_adapter):
    initialized, rng = torch.cuda.is_initialized(), torch.random.get_rng_state().clone()
    check(bound_adapter)
    bound_adapter.activate_scene()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.cuda.is_initialized() == initialized
