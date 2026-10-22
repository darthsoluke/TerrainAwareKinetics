"""Isolated CPU proof-schema fixtures, not generated GPU admission evidence."""

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from motionbricks.mb_sonic.geometry_execution import (
    PFNN_BVH_SHA256, PFNN_SPHERE_BVH_SHA256, verified_sphere_bvh_proof,
)
from motionbricks.mb_sonic.reference import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "cpu-unit-schema-fixture-only"


def fixture_proof(plane):
    proof = dict(schema="motionbricks.mb_sonic.sphere_bvh_admission.v1", passed=True,
                scene_id="000" if plane else "001", scene_sha256=SOURCE_SHA, plane_unchanged=plane,
                baseline_implementation_sha256=PFNN_BVH_SHA256,
                implementation_sha256=PFNN_SPHERE_BVH_SHA256, points_checked=6144 if plane else 6189,
                distance_max_abs=0, normal_max_abs=0, valid_equal=True, active_prism_ids_equal=True,
                traced_and_captured_equal=True, boundary_ties_equal=True,
                native_cube=dict(samples=20, p50_ms=1., p99_ms=2., max_ms=3.),
                sphere=dict(samples=20, p50_ms=.5, p99_ms=1., max_ms=2.),
                optimizer_updates=0, physical_steps=0)
    if not plane:
        proof.update(active_prism_comparison_scope="native_valid_queries_only", invalid_internal_id_differences=9)
    return proof


def admit_fixture(tmp_path, proof, plane):
    path = tmp_path / "unit-proof.json"
    path.write_text(json.dumps(proof), encoding="utf-8")
    return verified_sphere_bvh_proof(path, SOURCE_SHA, scene_id="000" if plane else "001", plane=plane)


@pytest.mark.parametrize("plane", [False, True])
def test_parent_schema_and_minimum_counts_are_accepted_without_cuda(tmp_path, plane):
    initialized, rng = torch.cuda.is_initialized(), torch.random.get_rng_state().clone()
    proof = fixture_proof(plane)
    receipt = admit_fixture(tmp_path, proof, plane)
    assert receipt["implementation_sha256"] == PFNN_SPHERE_BVH_SHA256
    assert receipt["baseline_implementation_sha256"] == PFNN_BVH_SHA256
    assert receipt["plane_unchanged"] is plane
    assert receipt["points_checked"] == proof["points_checked"]
    assert receipt["native_cube"] == proof["native_cube"]
    assert receipt["active_prism_comparison_scope"] == ("not_applicable_native_plane" if plane else "native_valid_queries_only")
    assert receipt["invalid_internal_id_differences"] == (None if plane else 9)
    proof["points_checked"] += 100
    proof["sphere"]["samples"] = 40
    admit_fixture(tmp_path, proof, plane)
    json.dumps(receipt, allow_nan=False)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("key,value", [
    ("schema", "legacy-62-points"), ("passed", False), ("passed", 1),
    ("scene_id", "other-map"), ("scene_sha256", "other-source"),
    ("baseline_implementation_sha256", "other-baseline"), ("implementation_sha256", "other-candidate"),
    ("plane_unchanged", True), ("plane_unchanged", 0), ("points_checked", 6188),
    ("distance_max_abs", 1e-9), ("normal_max_abs", 1e-9), ("valid_equal", False),
    ("active_prism_ids_equal", False), ("traced_and_captured_equal", False),
    ("boundary_ties_equal", False), ("optimizer_updates", 1), ("physical_steps", 1),
])
def test_proof_contract_changes_are_rejected(tmp_path, key, value):
    proof = fixture_proof(False)
    proof[key] = value
    with pytest.raises(ValueError, match="does not bind"):
        admit_fixture(tmp_path, proof, False)


@pytest.mark.parametrize("key,value", [
    ("active_prism_comparison_scope", None), ("active_prism_comparison_scope", "all_queries"),
    ("invalid_internal_id_differences", None), ("invalid_internal_id_differences", -1),
    ("invalid_internal_id_differences", True),
])
def test_rough_proof_requires_native_valid_id_scope_metadata(tmp_path, key, value):
    proof = fixture_proof(False)
    proof[key] = value
    with pytest.raises(ValueError, match="valid-query ID scope"):
        admit_fixture(tmp_path, proof, False)


@pytest.mark.parametrize("name", ["native_cube", "sphere"])
@pytest.mark.parametrize("key,value", [
    ("samples", 19), ("samples", True), ("p50_ms", 0), ("p99_ms", float("nan")),
    ("max_ms", float("inf")), ("p50_ms", None),
])
def test_missing_or_unmeasured_timings_are_rejected(tmp_path, name, key, value):
    proof = fixture_proof(False)
    proof[name][key] = value
    with pytest.raises(ValueError, match="graph timings"):
        admit_fixture(tmp_path, proof, False)


def test_parent_probe_emits_the_exact_required_field_names():
    probe = ROOT / "out/mb_sonic_e0_e3_20261018/probe_wasd_sphere_queries_v1.py"
    tree = ast.parse(probe.read_text(encoding="utf-8"))
    row = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "row" for target in node.targets))
    assert set(fixture_proof(False)) <= {keyword.arg for keyword in row.value.keywords}
    latency = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "latency")
    returned = next(node for node in ast.walk(latency) if isinstance(node, ast.Return))
    assert set(fixture_proof(False)["sphere"]) <= {keyword.arg for keyword in returned.value.keywords}


def test_previous_unemitted_alias_fields_are_not_required(tmp_path):
    proof = fixture_proof(False)
    before = copy.deepcopy(proof)
    admit_fixture(tmp_path, proof, False)
    assert proof == before
    assert not any(name in proof for name in ("backend", "field_kind", "reference_implementation_sha256",
                                               "dynamic_cases", "same_owner", "trace_equal", "capture_equal"))


def test_actual_all_fifteen_parent_proofs_admit_offline_without_kernel_work():
    initialized, rng = torch.cuda.is_initialized(), torch.random.get_rng_state().clone()
    root = ROOT / "out/mb_sonic_e0_e3_20261018"
    complete = json.loads((root / "wasd_sphere_query_v2/complete.json").read_text(encoding="utf-8"))
    assert complete["passed"] is True and complete["maps"] == 14
    assert complete["cpu_rng_equal"] is True and complete["cuda_rng_equal"] is True
    assert complete["optimizer_updates"] == complete["physical_steps"] == 0
    assert set(complete["source_locks"].values()) == {PFNN_BVH_SHA256, PFNN_SPHERE_BVH_SHA256}
    entries = json.loads((ROOT / "out/pfnn_mixed_20261015/pfnn_map_catalog_v1/catalog.json").read_text(encoding="utf-8"))["entries"]
    assert {row["scene_id"] for row in complete["results"]} == {row["id"] for row in entries if row["id"] != "000"}
    for entry in entries:
        scene_id = entry["id"]
        folder = "wasd_sphere_query_v1" if scene_id == "000" else "wasd_sphere_query_v2"
        source = root / "scenes" / scene_id / "source_scene.npz"
        receipt = verified_sphere_bvh_proof(root / folder / scene_id / "proof.json", file_sha256(source),
                                           scene_id=scene_id, plane=scene_id == "000")
        assert receipt["points_checked"] == (6144 if scene_id == "000" else 6189)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.cuda.is_initialized() == initialized


def test_actual_001_internal_id_differences_are_outside_native_certainty():
    root = ROOT / "out/mb_sonic_e0_e3_20261018"
    with np.load(root / "wasd_sphere_id_diagnosis_v1/failed-001/active_ids.npz", allow_pickle=False) as data:
        differing = data["old_ids"] != data["new_ids"]
        assert differing.sum() == 9
        points = torch.from_numpy(data["points"].reshape(-1, 3)[differing].copy())
        ids = data["old_ids"][differing]
        centers, halves, levels = [torch.from_numpy(data[name][ids].copy()) for name in ("centers", "halves", "levels")]
        origin = torch.from_numpy(data["source_origin"].copy())
        assert np.count_nonzero(data["native_distance"].reshape(-1)[differing]) == 0
        assert np.count_nonzero(data["native_normal"].reshape(-1, 3)[differing]) == 0
    with np.load(root / "scenes/001/source_scene.npz", allow_pickle=False) as scene:
        height = torch.from_numpy(scene["heightfield_y_m"].copy())
        dx = float(scene["dx_m"])
    cell = torch.floor((points[:, [0, 2]] - origin[[0, 2]]) / dx + .5).long()
    x, z = cell.unbind(-1)
    inside = points[:, 1] < height[z.clamp(0, height.shape[0] - 1), x.clamp(0, height.shape[1] - 1)]
    horizontal = (points[:, [0, 2]] - centers).abs().sub(halves).clamp_min(0)
    sign = torch.where(inside, -torch.ones_like(points[:, 1]), torch.ones_like(points[:, 1]))
    vertical = ((points[:, 1] - levels) * sign).clamp_min(0)
    distance = torch.sqrt(horizontal[:, 0].square() + horizontal[:, 1].square() + vertical.square())
    edge = torch.stack((points[:, 0] - (origin[0] - dx / 2), origin[0] + (height.shape[1] - .5) * dx - points[:, 0],
                        points[:, 2] - (origin[2] - dx / 2), origin[2] + (height.shape[0] - .5) * dx - points[:, 2]), -1).amin(-1)
    assert bool((distance > edge + 1e-6).all())
