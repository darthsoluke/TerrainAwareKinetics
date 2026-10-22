"""Fixed-scene native query/FK execution, independent of neural inference."""

from functools import partial
import json
import math
from pathlib import Path

import torch

from motionbricks.data.native_mb_evaluation import full_precision_evaluation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.soma_cuda_graph import StaticCudaCall
from motionbricks.data.soma_static_execution import trace_fixed_call


SCENE_INPUT_NAMES = (
    "points", "valid", "world_query_xz", "query_envelope_m",
    "survey_points", "survey_valid", "survey_world_query_xz", "survey_query_envelope_m",
    "sdf_points", "sdf_distance", "sdf_normal", "sdf_valid", "sdf_features",
)
SOMA_FK_NAMES = ("root_pos", "local_joint_rots", "global_joint_rots", "posed_joints")
QUERY_ATOL = 1e-5
FK_ATOL = 3e-5
OBSERVED_ATOL = 2e-5
PFNN_BVH_SHA256 = "ccdb3c55616e76999e4a8821d4fe83c0e14888bd24c413ea848746483000ca00"
PFNN_SPHERE_BVH_SHA256 = "0b6230d290782be0c9250daca98b0e10c5a60b1a097ae4ac56e43589c5fee886"


def verified_pfnn_bvh_proof(path, source_sha256):
    """Read retained exact-query evidence; never rerun its geometry comparisons."""
    from .reference import file_sha256

    path = Path(path)
    proof = json.loads(path.read_text(encoding="utf-8"))
    implementation = Path(__file__).parents[1] / "data" / "bvh_heightfield.py"
    expected = dict(passed=True, scene_sha256=source_sha256, implementation_sha256=PFNN_BVH_SHA256,
                    points_checked=62, distance_max_abs=0, normal_max_abs=0, valid_equal=True,
                    actual_optimizer_updates=0)
    if (any(proof.get(key) != value for key, value in expected.items())
            or proof.get("passed") is not True or proof.get("valid_equal") is not True
            or file_sha256(implementation) != PFNN_BVH_SHA256):
        raise ValueError("Exact PFNN BVH proof does not bind this source and implementation")
    return dict(proof_path=str(path.resolve()), proof_sha256=file_sha256(path),
                implementation_sha256=PFNN_BVH_SHA256, points_checked=proof["points_checked"])


def verified_sphere_bvh_proof(path, source_sha256, *, scene_id, plane):
    """Admit the parent's retained same-owner dynamic/edge sphere comparison."""
    from .reference import file_sha256

    path = Path(path)
    proof = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(schema="motionbricks.mb_sonic.sphere_bvh_admission.v1", passed=True,
                    scene_id=scene_id, scene_sha256=source_sha256, plane_unchanged=plane,
                    baseline_implementation_sha256=PFNN_BVH_SHA256,
                    implementation_sha256=PFNN_SPHERE_BVH_SHA256,
                    distance_max_abs=0, normal_max_abs=0, valid_equal=True, active_prism_ids_equal=True,
                    traced_and_captured_equal=True, boundary_ties_equal=True, optimizer_updates=0, physical_steps=0)
    implementation = Path(__file__).with_name("sphere_bvh_heightfield.py")
    baseline = Path(__file__).parents[1] / "data" / "bvh_heightfield.py"
    checked = proof.get("points_checked")
    if (any(proof.get(key) != value for key, value in expected.items())
            or any(proof.get(key) is not value for key, value in expected.items() if isinstance(value, bool))
            or type(checked) is not int or checked < (6144 if plane else 6189)
            or file_sha256(implementation) != PFNN_SPHERE_BVH_SHA256
            or file_sha256(baseline) != PFNN_BVH_SHA256):
        raise ValueError("Sphere BVH proof does not bind this scene, comparison and implementation")
    scope, invalid_ids = "not_applicable_native_plane", None
    if not plane:
        scope = proof.get("active_prism_comparison_scope")
        invalid_ids = proof.get("invalid_internal_id_differences")
        if scope != "native_valid_queries_only" or type(invalid_ids) is not int or invalid_ids < 0:
            raise ValueError("Sphere BVH proof requires valid-query ID scope and an invalid-ID difference count")
    for name in ("native_cube", "sphere"):
        timing = proof.get(name)
        if not isinstance(timing, dict) or type(timing.get("samples")) is not int or timing["samples"] < 20:
            raise ValueError("Sphere BVH proof requires twenty hot graph timings per backend")
        values = [timing.get(key) for key in ("p50_ms", "p99_ms", "max_ms")]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Sphere BVH proof requires finite positive measured graph timings")
    return dict(proof_path=str(path.resolve()), proof_sha256=file_sha256(path), proof_schema=proof["schema"],
                implementation_sha256=PFNN_SPHERE_BVH_SHA256, baseline_implementation_sha256=PFNN_BVH_SHA256,
                points_checked=checked, plane_unchanged=plane, active_prism_comparison_scope=scope,
                invalid_internal_id_differences=invalid_ids, native_cube=proof["native_cube"], sphere=proof["sphere"])


def install_scene_field(controller, scene, *, backend, source_sha256, proof_path, scene_id=None):
    """Explicit deployment selection of the existing native plane/prism algorithms."""
    from motionbricks.data.unified_scene_field import field_from_heightfield

    receipt = dict(backend=backend, source_sha256=source_sha256)
    if backend == "native":
        if proof_path is not None:
            raise ValueError("A BVH proof requires explicit warp_bvh_exact deployment")
        field = controller.scene_field(scene)
    elif backend in ("warp_bvh_exact", "warp_bvh_sphere_exact"):
        if proof_path is None:
            raise ValueError(backend + " requires a per-map scene_query_proof")
        if backend == "warp_bvh_exact":
            receipt.update(verified_pfnn_bvh_proof(proof_path, source_sha256))
        height = scene["heightfield_y_m"]
        if (height.dtype != torch.float32 or not torch.isfinite(height).all()
                or ("heightfield_valid" in scene and not scene["heightfield_valid"].all())):
            raise ValueError("Proven PFNN BVH requires fully known FP32 terrain")
        plane = torch.equal(height, height[:1, :1].expand_as(height))
        if backend == "warp_bvh_sphere_exact":
            if not isinstance(scene_id, str) or not scene_id:
                raise ValueError("Sphere BVH proof requires an explicit scene_id")
            receipt.update(verified_sphere_bvh_proof(proof_path, source_sha256, scene_id=scene_id, plane=plane))
        if plane:
            field = field_from_heightfield(scene)
        else:
            if height.device.type != "cuda":
                raise ValueError("Rough warp_bvh_exact terrain requires allocated CUDA tensors")
            from importlib.metadata import version
            if version("warp-lang") != "1.18.0":
                raise ValueError("Exact PFNN BVH deployment requires the proven Warp 1.18.0")
            if backend == "warp_bvh_exact":
                from motionbricks.data.bvh_heightfield import BVHHeightfieldPrismSDF
                field = BVHHeightfieldPrismSDF(scene, controller.inference.device)
            else:
                from .sphere_bvh_heightfield import SphereBVHHeightfieldPrismSDF
                field = SphereBVHHeightfieldPrismSDF(scene, controller.inference.device)
        controller._fields[id(height)] = (scene, field)
    else:
        raise ValueError("Unknown scene_query_backend: " + backend)
    receipt["field_kind"] = type(field).__name__
    return field, receipt


def unique_tensor_storage_bytes(value, *, device_type="cuda"):
    """Exact owned Torch storage accounting; excludes Warp nodes and graph arenas."""
    storages = {}

    def visit(item):
        if isinstance(item, torch.Tensor) and item.device.type == device_type:
            storage = item.untyped_storage()
            storages[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return sum(storages.values())


def storage_binding(value, *, compact=False):
    """Deep admission, or replay without walking construction-only prism lists."""
    if isinstance(value, torch.Tensor):
        version = None if value.is_inference() else value._version
        return (id(value), value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                str(value.dtype), str(value.device), version)
    if isinstance(value, dict):
        return tuple((key, ((id(item), len(item)) if compact else (id(item), len(item), storage_binding(item))))
                     if key in ("rectangles", "holes") and isinstance(item, list)
                     else (key, storage_binding(item, compact=compact)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(storage_binding(item, compact=compact) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return (type(value).__qualname__, id(value))


def native_soma_fk(representation, features):
    decoded = representation.inverse(features, False, joint_positions_from="global_rot_data", return_all=False)
    return {name: decoded[name] for name in SOMA_FK_NAMES}


def native_observed_history(representation, features):
    checks = {}
    history = reencode_played_history(features, representation, tensor_checks=checks)
    return dict(history=history, **checks)


def _trace_cuda(function, inputs):
    if not inputs or any(not isinstance(x, torch.Tensor) or not x.is_cuda
                         or x.dtype != torch.float32 for x in inputs):
        raise ValueError("Geometry preparation requires allocated FP32 CUDA inputs")
    # Materialize native field/skeleton caches before they become static constants.
    function(*inputs)
    return trace_fixed_call(function, inputs, shared_cuda_constants=True)


class SceneQueryExecution:
    """Capture a native 13-tensor callback; all supplied tensor arguments stay dynamic.

    `function` is the raw tuple-returning TorchScript call for outer composition.
    `bindings` retains the fixed scene, field, sampler and any native tree handles.
    A different scene/source/storage requires a new instance, never recapture.
    """

    @torch.inference_mode()
    def __init__(self, scene_inputs, example_inputs, *, scene_id, source_sha256, bindings):
        self.native = scene_inputs
        self.scene_id, self.source_sha256, self.bindings = scene_id, source_sha256, bindings
        with full_precision_evaluation(), torch.jit.optimized_execution(False):
            self.function = _trace_cuda(self.native, example_inputs)
            self._call = StaticCudaCall(lambda *inputs: dict(zip(SCENE_INPUT_NAMES, self.function(*inputs))))
            self._call(*example_inputs)
        self.binding_signature = storage_binding(bindings)
        self.replay_binding_signature = storage_binding(bindings, compact=True)

    def __call__(self, *inputs):
        values = self._call(*inputs)
        return tuple(values[name] for name in SCENE_INPUT_NAMES)

    def assert_binding(self, *, scene_id, source_sha256, bindings, replay=False):
        """Deep by default; replay retains all storage checks but atomic prism lists."""
        signature = self.replay_binding_signature if replay else self.binding_signature
        if (scene_id != self.scene_id or source_sha256 != self.source_sha256
                or storage_binding(bindings, compact=replay) != signature):
            raise ValueError("Captured scene/source/storage binding changed")


class SomaFKExecution:
    """Exact unnormalized GLOBAL366 inverse; one graph per fixed sequence shape."""

    @torch.inference_mode()
    def __init__(self, representation, example_features):
        if example_features.shape not in ((1, 28, 366), (1, 32, 366), (1, 52, 366)):
            raise ValueError("SOMA FK requires a fixed28/32/52-frame GLOBAL366 sequence")
        self.native = partial(native_soma_fk, representation)
        with full_precision_evaluation(), torch.jit.optimized_execution(False):
            self.function = _trace_cuda(self.native, (example_features,))
            self._call = StaticCudaCall(self.function)
            self._call(example_features)

    def __call__(self, features):
        return self._call(features)


class ObservedHistoryExecution:
    """Existing four-frame played-FK re-encoding, with its original tensor checks."""

    @torch.inference_mode()
    def __init__(self, representation, example_features):
        if example_features.shape != (1, 4, 366):
            raise ValueError("Observed history requires exactly [1,4,366] played frames")
        self.native = partial(native_observed_history, representation)
        with full_precision_evaluation(), torch.jit.optimized_execution(False):
            self.function = _trace_cuda(self.native, (example_features,))
            self._call = StaticCudaCall(self.function)
            self._call(example_features)

    def __call__(self, features):
        return self._call(features)


def _compare(actual, expected, names, atol):
    errors = {}
    for name, left, right in zip(names, actual, expected):
        tolerance = atol if right.is_floating_point() else 0
        torch.testing.assert_close(left, right, rtol=0, atol=tolerance, equal_nan=False)
        if right.is_floating_point():
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError("Nonfinite native/captured geometry: " + name)
            errors[name] = float((left - right).abs().max())
    return errors


@torch.inference_mode()
def admit_scene_queries(execution, cases):
    """Compare native, traced and captured outputs on owner-supplied tensor cases."""
    reports = []
    with full_precision_evaluation(), torch.jit.optimized_execution(False):
        for inputs in cases:
            expected = execution.native(*inputs)
            if len(expected) != len(SCENE_INPUT_NAMES):
                raise ValueError("Exactly 13 native scene tensors required")
            for stage, call in (("trace", execution.function), ("capture", execution)):
                actual = call(*inputs)
                if len(actual) != len(expected):
                    raise ValueError("Scene output arity changed")
                errors = _compare(actual, expected, SCENE_INPUT_NAMES, QUERY_ATOL)
                reports.append(dict(case=len(reports) // 2, stage=stage, max_abs=errors))
    if not reports:
        raise ValueError("Native scene admission cases required")
    return reports


@torch.inference_mode()
def admit_soma_fk(execution, cases):
    """Root/joint Euclidean error <=3e-5 m; rotation elements <=3e-5, rtol=0."""
    reports = []
    with full_precision_evaluation(), torch.jit.optimized_execution(False):
        for features in cases:
            expected = execution.native(features)
            for stage, call in (("trace", execution.function), ("capture", execution)):
                actual = call(features)
                errors = _compare(tuple(actual[name] for name in SOMA_FK_NAMES),
                                  tuple(expected[name] for name in SOMA_FK_NAMES), SOMA_FK_NAMES, FK_ATOL)
                distances = {name: float((actual[name] - expected[name]).norm(dim=-1).max())
                             for name in ("root_pos", "posed_joints")}
                if max(distances.values()) > FK_ATOL:
                    raise ValueError("Captured native FK exceeds 3e-5 m")
                reports.append(dict(case=len(reports) // 2, stage=stage,
                                    max_abs=errors, max_distance_m=distances))
    if not reports:
        raise ValueError("Native FK admission cases required")
    return reports


@torch.inference_mode()
def admit_observed_history(execution, cases):
    """Admission only: retain the native position/rotation reconstruction gates."""
    reports = []
    names = ("history", "position_error", "rotation_error")
    with full_precision_evaluation(), torch.jit.optimized_execution(False):
        for features in cases:
            expected = execution.native(features)
            for stage, call in (("trace", execution.function), ("capture", execution)):
                actual = call(features)
                errors = _compare(tuple(actual[name] for name in names),
                                  tuple(expected[name] for name in names), names, QUERY_ATOL)
                checks = {name: float(actual[name]) for name in names[1:]}
                if max(checks.values()) > OBSERVED_ATOL:
                    raise ValueError("Observed-history FK position/rotation exceeds 2e-5")
                reports.append(dict(case=len(reports) // 2, stage=stage, max_abs=errors, **checks))
    if not reports:
        raise ValueError("Native observed-history admission cases required")
    return reports
