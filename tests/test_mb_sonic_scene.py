"""Bounded CPU scene checks; analytical fixtures are not runtime motion data."""

import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from motionbricks.mb_sonic.scene import (
    RIGID_BOX_FIXTURE, SCENE_SCHEMA, SOMA_TO_MUJOCO, SceneManifest, export_scene,
    load_scene, mujoco_to_soma, sha256, soma_to_mujoco,
)
from scripts.export_mb_sonic_scenes import _model_xml, contact_probe, verify_scene


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "out/pfnn_mixed_20261015/pfnn_map_catalog_v1/catalog.json"
FRICTION = (1., .005, .0001)


def make_scene(tmp_path, height=None, origin=(-4., 0., -2.), boxes=()):
    height = np.zeros((17, 33), np.float32) if height is None else np.asarray(height, np.float32)
    source = tmp_path / "source.npz"
    np.savez(source, heightfield_y_m=height, min_point_world_xyz_m=np.asarray(origin),
             dx_m=np.float32(.25), metadata_json=json.dumps(dict(coordinate_frame="soma_world_y_up")))
    path = export_scene(source, tmp_path / "export", scene_id="analytical_test_fixture", friction=FRICTION,
                        friction_source="Explicit test coefficients", boxes=boxes)
    return source, path, load_scene(path)


def test_transform_is_shared_proper_rotation_and_inverse():
    points = np.array([[1., 2., 3.], [-4., 5., -6.]])
    np.testing.assert_array_equal(soma_to_mujoco(points), [[1., -3., 2.], [-4., 6., 5.]])
    np.testing.assert_array_equal(mujoco_to_soma(soma_to_mujoco(points)), points)
    np.testing.assert_array_equal(SOMA_TO_MUJOCO.T @ SOMA_TO_MUJOCO, np.eye(3))
    assert np.linalg.det(SOMA_TO_MUJOCO) == 1.


def test_original_bytes_row_order_offsets_and_full_resolution(tmp_path):
    height = np.array([[-2., .1234567, 1.7, .15], [.3, 2.2, 3.7, -.6], [1., .5, -1.8, 2.]], np.float32)
    source, path, scene = make_scene(tmp_path, height=height, origin=(-1., 17., .5))
    manifest = SceneManifest.load(path)
    assert manifest.schema == SCENE_SCHEMA
    assert (path.parent / "source_scene.npz").read_bytes() == source.read_bytes()
    assert manifest.source["scene"]["sha256"] == sha256(source)
    binary = (path.parent / "heightfield.bin").read_bytes()
    assert struct.unpack("<ii", binary[:8]) == height.shape
    np.testing.assert_array_equal(np.frombuffer(binary[8:], "<f4").reshape(height.shape), height[::-1])
    assert manifest.heightfield["source_shape_zx"] == [3, 4]
    assert manifest.heightfield["size_m"][:2] == [.375, .25]
    assert manifest.heightfield["center_xyz_m"] == [-.625, -.75, -2.]
    for z in range(height.shape[0]):
        for x in range(height.shape[1]):
            xy = soma_to_mujoco([-1 + x * .25, 0, .5 + z * .25])[:2]
            assert scene.query_height(xy).height_m == height[z, x]
    assert manifest.friction["assignment"] == "fixed_scene_wide" and not manifest.friction["measured"]


def test_native_nodes_both_triangle_interiors_boundary_unknown(tmp_path):
    _, path, _ = make_scene(tmp_path, height=[[-2., .25, 1.7, .15], [.3, 2.2, 3.7, -.6], [1., .5, -1.8, 2.]])
    report = verify_scene(path, 16)
    assert report["passed"]
    json.dumps(report, allow_nan=False)
    assert report["groups"]["interiors"]["unique_normal_comparisons"] == 32
    assert report["groups"]["unknown"]["native_hits"] == 0


@pytest.mark.parametrize("height", [0., 7.25, -3.5])
def test_flat_normalization_preserves_absolute_height(tmp_path, height):
    _, path, scene = make_scene(tmp_path, height=np.full((4, 5), height))
    report = verify_scene(path, 8)
    assert report["passed"] and report["max_height_error_m"] < 1e-12
    assert scene.manifest.heightfield["size_m"][2] == 1.
    np.testing.assert_array_equal(scene.query_height([scene.bounds_xy_m.mean(axis=0)]).normal_xyz, [[0., 0., 1.]])


def test_unknown_has_no_clamped_height_or_normal(tmp_path):
    _, _, scene = make_scene(tmp_path)
    lower, upper = scene.bounds_xy_m
    points = [lower, upper, np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf), [np.nan, 0.]]
    query = scene.query_height(points)
    assert query.known.tolist() == [True, True, False, False, False]
    assert query.object_id.tolist() == [0, 0, -1, -1, -1]
    assert np.isnan(query.height_m[2:]).all() and np.isnan(query.normal_xyz[2:]).all()
    assert not query.normal_unique.any()


def test_scalar_measured_root_query_is_bitwise_batch_triangle_height(tmp_path):
    rng = np.random.default_rng(733)
    _, _, scene = make_scene(tmp_path, height=rng.normal(size=(7, 9)).astype(np.float32))
    lo, hi = scene.bounds_xy_m
    points = np.r_[rng.uniform(lo, hi, size=(300, 2)), [lo, hi, [lo[0], hi[1]], [hi[0], lo[1]]],
                   [np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf), [np.nan, 0.], [0., np.inf]]]
    expected = scene.query_height(points)
    for index, (x, y) in enumerate(points):
        height, known = scene.query_height_point(x, y)
        assert known == expected.known[index]
        if known:
            assert np.float64(height).tobytes() == expected.height_m[index].tobytes()
        else:
            assert height is None


def test_missing_source_nodes_are_rejected_not_filled(tmp_path):
    height = np.zeros((4, 4), np.float32)
    height[1, 2] = np.nan
    with pytest.raises(ValueError, match="unknown holes"):
        make_scene(tmp_path, height=height)
    assert not (tmp_path / "export").exists()


def test_asset_mutation_does_not_silently_change_physics(tmp_path):
    _, path, _ = make_scene(tmp_path)
    binary = path.parent / "heightfield.bin"
    binary.write_bytes(binary.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_scene(path)


def test_closed_render_mesh_matches_native_domain_and_base(tmp_path):
    _, _, scene = make_scene(tmp_path, height=np.arange(12).reshape(3, 4) * .1)
    mesh, = scene.render_meshes()
    tri = mesh.vertices_m[mesh.faces]
    area_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.linalg.norm(area_normals, axis=-1) > 0).all()
    assert (area_normals[:12, 2] > 0).all()
    assert np.isclose(mesh.vertices_m[:, 2].min(), -.1)
    np.testing.assert_array_equal(mesh.vertices_m[:, :2].min(axis=0), scene.bounds_xy_m[0])
    np.testing.assert_array_equal(mesh.vertices_m[:, :2].max(axis=0), scene.bounds_xy_m[1])
    edges = np.sort(np.concatenate((mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()


def test_independent_rigid_objects_do_not_replace_ground_height(tmp_path):
    _, _, scene = make_scene(tmp_path, boxes=RIGID_BOX_FIXTURE)
    assert [obj["object_id"] for obj in scene.manifest.objects] == [0, 1, 2, 3]
    assert len(scene.render_meshes()) == 4
    assert scene.query_height([0., 0.]).height_m == 0.
    np.testing.assert_allclose(scene.box_signed_distance([[0, 0, 1.2], [0, 0, 1.35]], 1), [.05, -.1])
    np.testing.assert_allclose(scene.box_signed_distance([[-2, 0, .45], [-2, 0, .2]], 3), [.05, -.2])


@pytest.mark.parametrize("kind,position,expected_id", [
    ("sphere", [0., 0., .035], 0),
    ("foot", [0., 0., .02], 0),
    ("palm", [0., 0., .06], 0),
    ("sphere", [0., 0., 1.22], 1),
    ("palm", [1.37, 0., .7], 2),
    ("foot", [-2., 0., .423], 3),
])
def test_genuine_native_contacts_for_ground_ceiling_wall_platform(tmp_path, kind, position, expected_id):
    _, path, _ = make_scene(tmp_path, boxes=RIGID_BOX_FIXTURE)
    report = contact_probe(path, kind=kind, position=position)
    contacts = [c for c in report["last_observed_contacts"] if c["object_id"] == expected_id]
    assert report["passed"] and contacts
    assert min(c["distance_m"] for c in contacts) < 0
    assert any(c["wrench_contact_frame"][0] > 0 for c in contacts)
    for contact in contacts:
        assert np.isclose(np.linalg.norm(contact["normal_geom1_to_geom2"]), 1.)
    assert report["actual_optimizer_updates"] == 0 and not report["qpos_qvel_overwrites"]


def test_free_drop_is_bounded_real_physics_and_empty_space_has_no_contacts(tmp_path):
    _, path, _ = make_scene(tmp_path)
    empty = contact_probe(path, kind="sphere", position=[0., 0., 1.])
    assert not empty["passed"] and empty["max_simultaneous_contacts"] == 0
    drop = contact_probe(path, kind="sphere", position=[0., 0., .3], steps=100)
    assert drop["passed"] and drop["simulation_time_s"] == pytest.approx(.5)
    assert drop["final_position_xyz_m"][2] == pytest.approx(.04, abs=.002)
    assert not drop["external_forces"] and not drop["automatic_reset"]
    with pytest.raises(ValueError, match="at most 10"):
        contact_probe(path, kind="sphere", position=[0., 0., .3], steps=2001)


@pytest.mark.parametrize("edge", ["xmin", "xmax", "ymin", "ymax"])
def test_physical_edge_contact_is_on_surface_not_hfield_base(tmp_path, edge):
    height = np.arange(20, dtype=np.float32).reshape(4, 5) * .01
    _, path, scene = make_scene(tmp_path, height=height)
    point = scene.bounds_xy_m.mean(axis=0)
    axis = 0 if edge[0] == "x" else 1
    point[axis] = scene.bounds_xy_m[0 if edge.endswith("min") else 1, axis]
    top = float(scene.query_height(point).height_m)
    report = contact_probe(path, kind="sphere", position=[*point, top + .035])
    assert report["passed"]
    contacts = report["last_observed_contacts"]
    assert all(c["object_id"] == 0 for c in contacts)
    assert all(abs(c["position_m"][2] - top) < .01 for c in contacts)
    assert any(c["normal_geom1_to_geom2"][2] > .9 for c in contacts)


def test_actual_catalog_has_all_fifteen_original_metric_sources():
    catalog = json.loads(CATALOG.read_text())
    assert len(catalog["entries"]) == 15
    for entry in catalog["entries"]:
        source = CATALOG.parent / entry["artifact"] / "scene.npz"
        assert sha256(source) == entry["scene_sha256"]
        with np.load(source, allow_pickle=False) as data:
            height = data["heightfield_y_m"]
            assert height.dtype == np.float32 and np.isfinite(height).all()
            assert min(height.shape) >= 2 and float(data["dx_m"]) > 0


def test_include_compiles_with_official_g1_without_extra_floor(tmp_path):
    _, _, scene = make_scene(tmp_path)
    robot = ROOT.parent / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
    root = ET.parse(robot).getroot()
    root.find("compiler").set("meshdir", (robot.parent / "meshes").as_posix())
    ET.SubElement(root, "include", file=scene.scene_include_path.as_posix())
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert (model.nq, model.nv, model.nu) == (50, 49, 43)
    assert np.count_nonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD) == 1
    assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mb_ground")
    np.testing.assert_array_equal(model.geom_friction[gid], FRICTION)
