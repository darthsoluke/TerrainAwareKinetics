"""CPU source contracts for the separate sphere candidate; no kernel execution."""

import ast
import copy
from importlib.metadata import version
from pathlib import Path

import pytest
import torch
from warp._src import context

CUDA_BEFORE_IMPORT = torch.cuda.is_initialized()
WARP_BEFORE_IMPORT = context.runtime

from motionbricks.data.bvh_heightfield import BVHHeightfieldPrismSDF, prism_squared
from motionbricks.mb_sonic import sphere_bvh_heightfield as candidate
from motionbricks.mb_sonic.reference import file_sha256


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PATH = ROOT / "motionbricks/data/bvh_heightfield.py"
CANDIDATE_PATH = ROOT / "motionbricks/mb_sonic/sphere_bvh_heightfield.py"


def definition(path, name, class_name=None):
    nodes = ast.parse(path.read_text(encoding="utf-8")).body
    if class_name is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    return next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)


class ReplaceBroadphase(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "bvh_query_aabb":
            node.func.attr = "bvh_query_sphere"
            node.args = [ast.Name(id=name, ctx=ast.Load()) for name in ("tree", "p", "query_radius")]
        return node


class Rename(ast.NodeTransformer):
    def __init__(self, before, after):
        self.before, self.after = before, after

    def visit_Name(self, node):
        if node.id == self.before:
            node.id = self.after
        return node


def body_ast(node):
    return ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)


def test_frozen_candidate_and_v1_sources_are_unchanged():
    frozen = {
        NATIVE_PATH: "ccdb3c55616e76999e4a8821d4fe83c0e14888bd24c413ea848746483000ca00",
        CANDIDATE_PATH: "0b6230d290782be0c9250daca98b0e10c5a60b1a097ae4ac56e43589c5fee886",
    }
    for path, expected in frozen.items():
        assert file_sha256(path) == expected


def test_sphere_builtin_is_registered_in_pinned_warp_without_runtime_start():
    assert version("warp-lang") == "1.18.0"
    assert "bvh_query_sphere" in context.builtin_functions
    assert "bvh_query_aabb" in context.builtin_functions
    assert context.runtime is WARP_BEFORE_IMPORT
    assert torch.cuda.is_initialized() == CUDA_BEFORE_IMPORT


def test_candidate_kernel_changes_only_broadphase_call():
    native = definition(NATIVE_PATH, "select_prism")
    expected = ReplaceBroadphase().visit(copy.deepcopy(native))
    actual = definition(CANDIDATE_PATH, "select_prism_sphere")
    assert ast.dump(native.args, include_attributes=False) == ast.dump(actual.args, include_attributes=False)
    assert body_ast(actual) == body_ast(expected)
    assert candidate.prism_squared is prism_squared


def test_torch_distance_normal_validity_and_original_input_checks_are_identical():
    native = definition(NATIVE_PATH, "query", "BVHHeightfieldPrismSDF")
    expected = Rename("prism_bvh_indices", "selector").visit(copy.deepcopy(native))
    actual = definition(CANDIDATE_PATH, "_query", "SphereBVHHeightfieldPrismSDF")
    assert body_ast(actual) == body_ast(expected)


def test_custom_op_preserves_registry_stream_inputs_and_output_except_kernel_name():
    native = definition(NATIVE_PATH, "prism_bvh_indices")
    expected = Rename("select_prism", "select_prism_sphere").visit(copy.deepcopy(native))
    actual = definition(CANDIDATE_PATH, "prism_bvh_sphere_indices")
    assert body_ast(actual) == body_ast(expected)


def test_field_construction_is_inherited_and_reference_selector_is_proof_only():
    assert candidate.SphereBVHHeightfieldPrismSDF.__init__ is BVHHeightfieldPrismSDF.__init__
    for method, selector in (("query", "prism_bvh_sphere_indices"), ("reference_query", "prism_bvh_indices")):
        node = definition(CANDIDATE_PATH, method, "SphereBVHHeightfieldPrismSDF")
        calls = [item for item in ast.walk(node) if isinstance(item, ast.Call)]
        assert len(calls) == 1
        assert calls[0].func.attr == "_query"
        assert calls[0].args[1].id == selector
    assert candidate.SPHERE_BACKEND == "warp_bvh_sphere_exact"
    assert candidate.SPHERE_PROOF_SCHEMA == "motionbricks.mb_sonic.sphere_bvh_admission.v1"
    tree = ast.parse(CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Name) and node.id == "hasattr" for node in ast.walk(tree))
    assert len(CANDIDATE_PATH.read_text(encoding="utf-8").splitlines()) <= 120


def test_candidate_module_preserves_precision_options():
    tree = ast.parse(CANDIDATE_PATH.read_text(encoding="utf-8"))
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "set_module_options")
    assert ast.literal_eval(call.args[0]) == dict(enable_backward=False, fast_math=False, fuse_fp=False)


@pytest.mark.parametrize("method", ["query", "reference_query"])
def test_cpu_inputs_fail_before_field_or_warp_runtime_use(method):
    field = object.__new__(candidate.SphereBVHHeightfieldPrismSDF)
    with pytest.raises(ValueError, match="CUDA float32 XYZ"):
        getattr(field, method)(torch.zeros(1, 3))
    assert context.runtime is WARP_BEFORE_IMPORT
    assert torch.cuda.is_initialized() == CUDA_BEFORE_IMPORT
