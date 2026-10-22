"""CPU proof/ownership contracts; no Warp initialization or CUDA execution."""

from collections import OrderedDict
import gc
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from motionbricks.data.unified_scene_field import AnalyticPlaneField, VolumetricEgoSampler, field_from_heightfield
from motionbricks.mb_sonic.geometry_execution import (
    PFNN_BVH_SHA256, SceneQueryExecution, install_scene_field, storage_binding,
    unique_tensor_storage_bytes, verified_pfnn_bvh_proof,
)
from motionbricks.mb_sonic.mb_reference import (
    CompiledMBNeuralExecution, TRAINED_WEIGHTS_SHA256, TrainedMBReferenceAdapter,
)
from motionbricks.mb_sonic.reference import file_sha256
from motionbricks.mb_sonic.scene import load_scene


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "out/pfnn_mixed_20261015/pfnn_map_catalog_v1"
CANONICAL = ROOT / "out/mb_sonic_e0_e3_20261018/scenes"


class CPUSceneController(torch.nn.Module):
    """Isolated owner fixture using the real native field and sampler APIs."""

    def __init__(self):
        super().__init__()
        self.inference = SimpleNamespace(device="cpu")
        self.sampler = VolumetricEgoSampler()
        self._fields = OrderedDict()

    def scene_field(self, scene):
        key = id(scene["heightfield_y_m"])
        if key not in self._fields:
            self._fields[key] = (scene, field_from_heightfield(scene))
        return self._fields[key][1]


def make_adapter(controller=None, *, backend="native", proof=None):
    return TrainedMBReferenceAdapter(
        CPUSceneController() if controller is None else controller,
        canonical_scene=load_scene(CANONICAL / "000/scene_manifest.json"),
        identity=dict(checkpoint_sha256=TRAINED_WEIGHTS_SHA256),
        scene_query_backend=backend, scene_query_proof=proof,
    )


def metadata_query(adapter):
    """Binding admission only, deliberately no constructed CUDA graph."""
    query = object.__new__(SceneQueryExecution)
    query.scene_id = adapter.scene_id
    query.source_sha256 = adapter.canonical_scene.manifest.source["scene"]["sha256"]
    query.bindings = adapter.scene_graph_bindings()
    query.binding_signature = storage_binding(query.bindings)
    return query


def test_backend_is_explicit_and_native_remains_default():
    for method in (TrainedMBReferenceAdapter.load, TrainedMBReferenceAdapter.__init__):
        params = inspect.signature(method).parameters
        assert params["scene_query_backend"].default == "native"
        assert params["scene_query_proof"].default is None
    with pytest.raises(ValueError, match="Unknown scene_query_backend"):
        make_adapter(backend="automatic")
    with pytest.raises(ValueError, match="per-map scene_query_proof"):
        make_adapter(backend="warp_bvh_exact")
    with pytest.raises(ValueError, match="explicit warp_bvh_exact"):
        make_adapter(proof=CATALOG / "000/geometry_query_review.json")


def test_existing_fifteen_proofs_bind_actual_full_resolution_canonical_sources():
    initialized, rng = torch.cuda.is_initialized(), torch.random.get_rng_state().clone()
    catalog = json.loads((CATALOG / "catalog.json").read_text(encoding="utf-8"))
    complete = json.loads((CATALOG / "geometry_complete_v3.json").read_text(encoding="utf-8"))
    ids = [entry["id"] for entry in catalog["entries"]]
    assert len(ids) == 15 and "008" not in ids and "urban" in ids
    assert complete["passed"] is True and {row["id"] for row in complete["maps"]} == set(ids)
    for entry in catalog["entries"]:
        manifest = json.loads((CANONICAL / entry["id"] / "scene_manifest.json").read_text(encoding="utf-8"))
        source = CANONICAL / entry["id"] / manifest["source"]["scene"]["file"]
        proof = CATALOG / entry["artifact"] / "geometry_query_review.json"
        receipt = verified_pfnn_bvh_proof(proof, file_sha256(source))
        assert receipt["implementation_sha256"] == PFNN_BVH_SHA256
        assert receipt["points_checked"] == 62
        assert file_sha256(source) == entry["scene_sha256"] == manifest["source"]["scene"]["sha256"]
        with np.load(source, allow_pickle=False) as actual, np.load(CATALOG / entry["artifact"] / "scene.npz", allow_pickle=False) as original:
            for name in ("heightfield_y_m", "min_point_world_xyz_m", "dx_m", "initial_history"):
                np.testing.assert_array_equal(actual[name], original[name])
            assert actual["heightfield_y_m"].dtype == np.float32
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("key,value", [
    ("passed", False), ("passed", 1), ("scene_sha256", "other-map"),
    ("implementation_sha256", "other-helper"), ("points_checked", 61),
    ("distance_max_abs", 1e-8), ("normal_max_abs", 1e-8),
    ("valid_equal", False), ("valid_equal", 1), ("actual_optimizer_updates", 1),
])
def test_proof_changes_are_rejected_without_geometry_work(tmp_path, key, value):
    proof = json.loads((CATALOG / "000/geometry_query_review.json").read_text(encoding="utf-8"))
    source_sha256 = proof["scene_sha256"]
    proof[key] = value
    path = tmp_path / "invalid-proof.json"
    path.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        verified_pfnn_bvh_proof(path, source_sha256)


def test_opted_in_flat_scene_retains_exact_native_plane_and_rng():
    initialized, rng = torch.cuda.is_initialized(), torch.random.get_rng_state().clone()
    native = make_adapter()
    fast = make_adapter(backend="warp_bvh_exact", proof=CATALOG / "000/geometry_query_review.json")
    assert isinstance(native.scene_field, AnalyticPlaneField)
    assert isinstance(fast.scene_field, AnalyticPlaneField)
    assert native.scene_field.bounds == fast.scene_field.bounds
    points = torch.tensor([[0., 1., 0.], [0., -.5, 0.], [-9.86, 0., 0.], [10., 1., 0.]])
    for actual, expected in zip(fast.scene_field.query(points), native.scene_field.query(points)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name in native.scene:
        torch.testing.assert_close(fast.scene[name], native.scene[name], rtol=0, atol=0)
    receipt = fast.scene_query_receipt
    assert receipt["backend"] == "warp_bvh_exact" and receipt["field_kind"] == "AnalyticPlaneField"
    assert receipt["source_sha256"] == fast.canonical_scene.manifest.source["scene"]["sha256"]
    assert receipt["graph_arenas_and_jit_constants_included"] is False
    assert receipt["accounted_bytes_scope"] == "unique_cuda_torch_scene_and_field_storage"
    json.dumps(receipt, allow_nan=False)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("change", ["cpu", "fp64", "unknown"])
def test_rough_optin_never_falls_back_to_native(change):
    initialized = torch.cuda.is_initialized()
    controller = CPUSceneController()
    with np.load(CANONICAL / "001/source_scene.npz", allow_pickle=False) as source:
        scene = {name: torch.from_numpy(source[name].copy()) for name in (
            "heightfield_y_m", "min_point_world_xyz_m", "dx_m")}
    if change == "fp64":
        scene["heightfield_y_m"] = scene["heightfield_y_m"].double()
    if change == "unknown":
        scene["heightfield_valid"] = torch.ones_like(scene["heightfield_y_m"], dtype=torch.bool)
        scene["heightfield_valid"][0, 0] = False
    proof = CATALOG / "001/geometry_query_review.json"
    source_sha256 = json.loads(proof.read_text(encoding="utf-8"))["scene_sha256"]
    with pytest.raises(ValueError, match="allocated CUDA|fully known FP32"):
        install_scene_field(controller, scene, backend="warp_bvh_exact", source_sha256=source_sha256, proof_path=proof)
    assert not controller._fields
    assert torch.cuda.is_initialized() == initialized


def test_native_and_compiled_construction_use_same_pinned_field_after_cache_removal():
    adapter = make_adapter()
    owner = adapter.scene_field
    adapter.controller._fields.clear()
    adapter.activate_scene()
    engine = CompiledMBNeuralExecution(adapter.controller, adapter.scene)
    assert engine.field is owner
    assert adapter.controller.scene_field(adapter.scene) is owner
    assert adapter.scene_graph_bindings()["field_owner"] is owner


@pytest.mark.parametrize("field_kind", ["BVHHeightfieldPrismSDF", "SphereBVHHeightfieldPrismSDF"])
def test_live_bindings_detect_scene_field_or_handle_identity_changes(field_kind):
    adapter = make_adapter()
    query = metadata_query(adapter)
    query.assert_binding(scene_id=adapter.scene_id, source_sha256=query.source_sha256,
                         bindings=adapter.scene_graph_bindings())
    adapter.scene_field = AnalyticPlaneField(1.)
    with pytest.raises(ValueError, match="binding changed"):
        query.assert_binding(scene_id=adapter.scene_id, source_sha256=query.source_sha256,
                             bindings=adapter.scene_graph_bindings())
    # Synthetic handle metadata only; no numerical BVH or GPU success is asserted.
    adapter.scene_field = SimpleNamespace(field_id=17, solid_tree=SimpleNamespace(id=101), air_tree=SimpleNamespace(id=102))
    adapter._scene_query_receipt["field_kind"] = field_kind
    query = metadata_query(adapter)
    assert query.bindings["field_id"] == 17
    assert query.bindings["solid_tree"] is adapter.scene_field.solid_tree
    assert adapter.scene_query_receipt["warp_bvh_bytes"] is None
    adapter.scene_field.solid_tree.id = 103
    with pytest.raises(ValueError, match="binding changed"):
        query.assert_binding(scene_id=adapter.scene_id, source_sha256=query.source_sha256,
                             bindings=adapter.scene_graph_bindings())


def test_sphere_optin_reuses_actual_000_plane_proof_without_id_scope():
    proof = ROOT / "out/mb_sonic_e0_e3_20261018/wasd_sphere_query_v1/000/proof.json"
    adapter = make_adapter(backend="warp_bvh_sphere_exact", proof=proof)
    assert isinstance(adapter.scene_field, AnalyticPlaneField)
    receipt = adapter.scene_query_receipt
    assert receipt["backend"] == "warp_bvh_sphere_exact"
    assert receipt["plane_unchanged"] is True
    assert receipt["active_prism_comparison_scope"] == "not_applicable_native_plane"
    assert receipt["invalid_internal_id_differences"] is None
    assert receipt["points_checked"] == 6144


def test_release_unprepared_scene_keeps_shared_model_other_scenes_and_external_owners():
    controller = CPUSceneController()
    first, second = make_adapter(controller), make_adapter(controller)
    retained = first.scene_graph_bindings()
    owner = weakref.ref(first.scene_field)
    first_key, second_key = id(first.scene["heightfield_y_m"]), id(second.scene["heightfield_y_m"])
    first.release_scene()
    assert first_key not in controller._fields and second_key in controller._fields
    assert first.execution == "released"
    assert first.controller is second.controller is controller
    assert first._compiled is first._full_plan is first._native_prepared is None
    for method in (first.activate_scene, first.scene_graph_bindings):
        with pytest.raises(RuntimeError, match="Released MB scene"):
            method()
    del first
    gc.collect()
    assert owner() is retained["field_owner"]
    del retained
    gc.collect()
    assert owner() is None


def test_storage_accounting_deduplicates_views_and_separates_device_scope():
    data = torch.arange(100, dtype=torch.float32)
    mask = torch.ones(7, dtype=torch.bool)
    values = dict(scene=data, field=[data[::2], {"view": data[20:], "mask": mask}], handle=object())
    assert unique_tensor_storage_bytes(values, device_type="cpu") == 407
    assert unique_tensor_storage_bytes(values) == 0


def test_native_optional_valid_mask_is_preserved(tmp_path):
    source = tmp_path / "masked-scene.npz"
    height = np.zeros((7, 7), dtype=np.float32)
    valid = np.ones_like(height, dtype=bool)
    valid[3, 3] = False
    np.savez(source, heightfield_y_m=height, min_point_world_xyz_m=np.array([-3., 0., -3.], np.float32),
             dx_m=np.array(1., np.float32), heightfield_valid=valid)
    scene = SimpleNamespace(root=tmp_path, manifest=SimpleNamespace(
        scene_id="masked-unit-fixture", objects=[dict(kind="hfield")],
        source=dict(scene=dict(file=source.name, sha256=file_sha256(source)))))
    adapter = TrainedMBReferenceAdapter(CPUSceneController(), canonical_scene=scene,
                                        identity=dict(checkpoint_sha256=TRAINED_WEIGHTS_SHA256))
    np.testing.assert_array_equal(adapter.scene["heightfield_valid"].numpy(), valid)
    _, _, known = adapter.scene_field.query(torch.tensor([[0., .1, 0.]]))
    assert not known.item()
