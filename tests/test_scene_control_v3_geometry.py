import os
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

import motionbricks.scene_control_v3.geometry as geometry
from motionbricks.scene_control_v3.geometry import (
    WarpScene, build_pfnn_scene, build_true3d_scene, encode_scene_features,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def box_scene(count, *, batch_size=1, max_distance_m=10):
    return WarpScene((trimesh.creation.box(extents=[2, 2, 2]),), signed=(True,),
                     known_bounds_m=[[-5, -5, -5], [5, 5, 5]], query_count=count,
                     batch_size=batch_size, max_distance_m=max_distance_m)


def query(scene, points, *, pose=None, known=None, enabled=None, mask=None):
    points = points if isinstance(points, torch.Tensor) else torch.tensor(points, dtype=torch.float32, device=scene.device)
    if points.ndim == 2:
        points = points.unsqueeze(0)
    return scene.query(points,
                       scene.initial_object_pose if pose is None else pose,
                       scene.initial_object_known if known is None else known,
                       scene.initial_object_enabled if enabled is None else enabled,
                       torch.ones(points.shape[:-1], dtype=torch.bool, device=points.device) if mask is None else mask)


def test_box_signed_distances_closest_and_face_normals():
    scene = box_scene(4)
    out = query(scene, [[0, 1.3, 0], [0, 0.8, 0], [0, -1.4, 0], [1, 0, 0]])
    torch.testing.assert_close(out["distance_m"], torch.tensor([[0.3, -0.2, 0.4, 0]]), atol=2e-6, rtol=0)
    torch.testing.assert_close(out["normal_world"][0, :3], torch.tensor([[0., 1, 0], [0, 1, 0], [0, -1, 0]]))
    torch.testing.assert_close(out["closest_point_world_m"][0, :3],
                               torch.tensor([[0., 1, 0], [0, 1, 0], [0, -1, 0]]))
    assert out["valid"].all() and out["distance_signed"].all()
    assert out["object_id"].eq(0).all()
    assert not out["distance_gradient_valid"][0, 3]
    assert out["distance_gradient_world"][0, 3].eq(0).all()
    assert all(isinstance(value, torch.Tensor) for value in out.values())


def test_edge_corner_point_vjp_is_distance_gradient_not_face_normal():
    scene = box_scene(3)
    points = torch.tensor([[[1.2, 1.3, 1.4], [1.2, 1.3, 0.1], [0.8, 0.2, 0.1]]], requires_grad=True)
    out = query(scene, points)
    out["distance_m"].sum().backward()
    expected = torch.tensor([[[0.2, 0.3, 0.4], [0.2, 0.3, 0], [1., 0, 0]]])
    expected /= expected.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(points.grad, expected, atol=1e-6, rtol=0)
    assert not torch.allclose(out["normal_world"][0, 0], expected[0, 0])
    torch.testing.assert_close(out["closest_point_world_m"][0, 0], torch.ones(3))
    for axis in range(3):
        step = torch.zeros_like(points)
        step[..., axis] = 0.001
        plus = query(scene, points.detach() + step)["distance_m"]
        minus = query(scene, points.detach() - step)["distance_m"]
        torch.testing.assert_close((plus - minus) / 0.002, expected[..., axis], atol=9e-5, rtol=0)


def test_batched_object_transform_translation_and_rotation():
    scene = box_scene(1, batch_size=2)
    pose = scene.initial_object_pose.clone()
    pose[1, 0] = torch.tensor([2, 2, 0, 0, 0, np.sqrt(0.5), np.sqrt(0.5)])
    points = torch.tensor([[[1.25, 0, 0]], [[2, 3.25, 0]]], requires_grad=True)
    out = query(scene, points, pose=pose)
    torch.testing.assert_close(out["distance_m"], torch.full((2, 1), 0.25), atol=1e-6, rtol=0)
    torch.testing.assert_close(out["normal_world"], torch.tensor([[[1., 0, 0]], [[0, 1, 0]]]), atol=1e-6, rtol=0)
    out["distance_m"].sum().backward()
    torch.testing.assert_close(points.grad, out["normal_world"], atol=1e-6, rtol=0)
    assert not out["closest_point_world_m"].requires_grad


def test_unknown_absent_nonfinite_outside_and_radius_masks():
    scene = box_scene(4)
    points = torch.tensor([[[1.2, 0, 0], [6, 0, 0], [float("nan"), 0, 0], [1.2, 0, 0]]], requires_grad=True)
    out = query(scene, points, mask=torch.tensor([[True, True, True, False]]))
    assert out["valid"].tolist() == [[True, False, False, False]]
    out["distance_m"].sum().backward()
    assert points.grad[0, 1:].eq(0).all()
    for kwargs in ({"known": torch.zeros((1, 1), dtype=torch.bool)},
                   {"enabled": torch.zeros((1, 1), dtype=torch.bool)}):
        result = query(scene, points.detach(), **kwargs)
        assert not result["scene_known"].any()
        assert not result["valid"].any()
        assert result["object_id"].eq(-1).all()
        for key in ("distance_m", "normal_world", "closest_point_world_m", "distance_gradient_world"):
            assert result[key].eq(0).all()
    short = box_scene(1, max_distance_m=0.1)
    assert not query(short, [[1.2, 0, 0]])["valid"].any()
    pose = scene.initial_object_pose.clone()
    pose[..., 6] = 2  # Not a rigid rotation; invalid, not normalized silently.
    assert not query(scene, points.detach(), pose=pose)["valid"].any()


def test_known_absence_is_distinct_from_unknown_component():
    scene = build_true3d_scene(query_count=1)
    enabled = scene.initial_object_enabled.clone()
    enabled[0, 1] = False
    out = query(scene, [[0, 0.1, 0]], enabled=enabled)
    assert out["valid"].all() and out["scene_known"].all()
    known = scene.initial_object_known.clone()
    known[0, 1] = False
    assert not query(scene, [[0, 0.1, 0]], known=known, enabled=enabled)["valid"].any()


def test_open_plane_is_unsigned_and_sign_is_never_guessed():
    plane = trimesh.Trimesh([[-2, 0, -2], [-2, 0, 2], [2, 0, -2], [2, 0, 2]],
                            [[0, 1, 2], [2, 1, 3]], process=False)
    with pytest.raises(ValueError, match="watertight"):
        WarpScene((plane,), signed=(True,), known_bounds_m=[[-3]*3, [3]*3])
    scene = WarpScene((plane,), signed=(False,), known_bounds_m=[[-3]*3, [3]*3], query_count=2)
    out = query(scene, [[0, 0.25, 0], [0, -0.25, 0]])
    torch.testing.assert_close(out["distance_m"], torch.full((1, 2), 0.25))
    torch.testing.assert_close(out["normal_world"], torch.tensor([[[0., 1, 0], [0, 1, 0]]]))
    torch.testing.assert_close(out["distance_gradient_world"], torch.tensor([[[0., 1, 0], [0, -1, 0]]]))
    assert not out["distance_signed"].any()
    points = torch.tensor([[[0., 0.25, 0], [0, -0.25, 0]]])
    features, valid = encode_scene_features(points, out, torch.zeros((1, 3)), torch.zeros(1),
                                           torch.zeros((1, 2)), torch.zeros((1, 2)),
                                           scene.initial_object_pose, torch.zeros((1, 1, 6)))
    assert valid.all() and features[..., 15].eq(0).all()


def test_true3d_overhead_and_side_are_real_surfaces():
    scene = build_true3d_scene(query_count=4)
    out = query(scene, [[0, 1.6, 0], [0, 2., 0], [1.3, 1.2, 0], [1.5, 1.2, 0]])
    assert out["object_id"].tolist() == [[1, 1, 2, 2]]
    torch.testing.assert_close(out["distance_m"], torch.tensor([[0.1, 0.1, 0.1, -0.1]]), atol=2e-6, rtol=0)
    torch.testing.assert_close(out["normal_world"][0, :3], torch.tensor([[0., -1, 0], [0, 1, 0], [-1, 0, 0]]))
    pose = scene.initial_object_pose.clone()
    pose[0, 1, 1] += 0.2
    moved = query(scene, [[0, 1.6, 0], [0, 2., 0], [1.3, 1.2, 0], [1.5, 1.2, 0]], pose=pose)
    torch.testing.assert_close(moved["distance_m"][0, 0], torch.tensor(0.3), atol=2e-6, rtol=0)


def test_minimum_component_signed_selection_not_exact_csg_union():
    meshes = (trimesh.creation.box(extents=[4, 4, 4]), trimesh.creation.box(extents=[0.2, 0.2, 0.2]))
    poses = np.array([[0, 0, 0, 0, 0, 0, 1], [0.3, 0, 0, 0, 0, 0, 1]], np.float32)
    scene = WarpScene(meshes, signed=(True, True), known_bounds_m=[[-4]*3, [4]*3],
                      query_count=1, object_pose=poses)
    out = query(scene, [[0, 0, 0]])
    torch.testing.assert_close(out["distance_m"], torch.tensor([[-2.]]))
    assert out["object_id"].tolist() == [[0]]


@pytest.mark.parametrize("solid_first", [False, True])
def test_open_floor_cannot_hide_inside_solid_with_independent_radii(solid_first):
    floor = trimesh.Trimesh([[-3, 0, -3], [-3, 0, 3], [3, 0, -3], [3, 0, 3]],
                            [[0, 1, 2], [2, 1, 3]], process=False)
    box = trimesh.creation.box(extents=[1, 1, 1])
    poses = np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0.1, 0, 0, 0, 0, 1]], np.float32)
    meshes, signs = (floor, box), (False, True)
    if solid_first:
        meshes, signs, poses = meshes[::-1], signs[::-1], poses[::-1].copy()
    scene = WarpScene(meshes, signed=signs, known_bounds_m=[[-4]*3, [4]*3],
                      query_count=1, object_pose=poses, max_distance_m=0.2)
    # Floor +0.1 must not hide solid -0.5, even beyond ordinary positive radius.
    out = query(scene, [[0, 0.1, 0]])
    torch.testing.assert_close(out["distance_m"], torch.tensor([[-0.5]]))
    assert out["object_id"].tolist() == [[0 if solid_first else 1]]
    assert out["distance_signed"].all()


def test_16d_layout_native_heading_surface_velocity_and_refresh():
    scene = box_scene(2)
    points = torch.tensor([[[1.25, 0, 0], [0, 0, 1.25]]])
    root, heading = torch.zeros((1, 3)), torch.tensor([np.pi/2])
    body, times = torch.tensor([[4, 5]]), torch.tensor([[0., 0.1]])
    velocity = torch.tensor([[[1., 2, 3, 0, 2, 0]]])
    out = query(scene, points)
    features, valid = encode_scene_features(points, out, root, heading, body, times,
                                           scene.initial_object_pose, velocity)
    assert features.shape == (1, 2, 16) and valid.all()
    torch.testing.assert_close(features[0, 0], torch.tensor([
        0., 0, 1.25, 0.25, 0, 0, 1, 4, 0, 0, 0, 1, -1, 2, 1, 1]), atol=2e-6, rtol=0)
    torch.testing.assert_close(features[0, 1, :3], torch.tensor([-1.25, 0., 0]), atol=1e-6, rtol=0)
    owner_id = scene._meshes[0].id
    moved = points + torch.tensor([[[0.1, 0, 0]]])
    next_features, _ = encode_scene_features(moved, query(scene, moved), root, heading, body, times,
                                            scene.initial_object_pose, velocity)
    assert not torch.equal(features, next_features)
    assert scene._meshes[0].id == owner_id


def test_unknown_features_are_zero_not_nan_and_explicitly_invalid():
    scene = box_scene(1)
    points = torch.tensor([[[float("nan"), 0, 0]]])
    out = query(scene, points, known=torch.zeros((1, 1), dtype=torch.bool))
    features, valid = encode_scene_features(points, out, torch.zeros((1, 3)), torch.zeros(1),
                                           torch.zeros((1, 1)), torch.zeros((1, 1)),
                                           scene.initial_object_pose, torch.zeros((1, 1, 6)))
    assert not out["scene_known"].any() and not valid.any()
    assert features.eq(0).all() and torch.isfinite(features).all()


def test_scene_owns_copied_mesh_storage():
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    scene = WarpScene((mesh,), signed=(True,), known_bounds_m=[[-4]*3, [4]*3], query_count=1)
    mesh.vertices[:] += 100
    torch.testing.assert_close(query(scene, [[1.2, 0, 0]])["distance_m"], torch.tensor([[0.2]]), atol=1e-6, rtol=0)


def test_pfnn_triangle_ground_unknown_holes_and_outside(tmp_path):
    height = np.zeros((4, 4), np.float32)
    height[0, 0] = np.nan
    path = tmp_path / "scene.npz"
    np.savez(path, heightfield_y_m=height, min_point_world_xyz_m=np.zeros(3), dx_m=np.float32(1))
    scene = build_pfnn_scene(path, query_count=4)
    out = query(scene, [[0.5, 0.2, 0.5], [2.5, -0.2, 2.5], [3.1, 0.2, 2.5], [3, 0.2, 3]])
    assert out["valid"].tolist() == [[False, True, False, True]]
    assert not out["distance_signed"].any()
    torch.testing.assert_close(out["distance_m"], torch.tensor([[0., 0.2, 0., 0.2]]), atol=1e-6, rtol=0)
    assert scene.storage_summary["objects"][0]["triangles"] == 16
    np.savez(path, heightfield_y_m=np.full((2, 2), np.nan), min_point_world_xyz_m=np.zeros(3), dx_m=np.float32(1))
    with pytest.raises(ValueError, match="entirely unknown"):
        build_pfnn_scene(path)


def test_linear_triangle_ground_not_old_constant_prism(tmp_path):
    path = tmp_path / "slope.npz"
    np.savez(path, heightfield_y_m=np.array([[0., 1], [0., 1]], np.float32),
             min_point_world_xyz_m=np.zeros(3), dx_m=np.float32(1))
    scene = build_pfnn_scene(path, query_count=1)
    out = query(scene, [[0.5, 0.8, 0.5]])
    torch.testing.assert_close(out["distance_m"], torch.tensor([[0.3 / np.sqrt(2)]], dtype=torch.float32))
    torch.testing.assert_close(out["closest_point_world_m"], torch.tensor([[[0.65, 0.65, 0.5]]]))
    assert scene.storage_summary["old_prism_exact_parity"] is False


@pytest.mark.parametrize("relative,shape,sha", [
    ("pfnn_map_catalog_v1/011", (954, 1109), "e9a0c58793634c2c2500cafebc6d6df45ab367f9c91ee88e8a87adfcb9a36b13"),
    ("pfnn_map_catalog_v1/007", (1332, 593), "ad9b3e8747554d6b6ff7849c8704d2cf707f140198848f83e2d7189d3425ef1d"),
    ("final_acceptance_flat_v1", (321, 321), "5c4b2049a2801f8be4268da42dec730f5094e9b992910acd661d0c21027f1464"),
])
def test_actual_pfnn_assets_cpu_only(relative, shape, sha):
    path = Path(__file__).resolve().parents[1] / "out" / "pfnn_mixed_20261015" / relative / "scene.npz"
    scene = build_pfnn_scene(path, query_count=1)
    with np.load(path, allow_pickle=False) as source:
        height, origin, dx = source["heightfield_y_m"], source["min_point_world_xyz_m"], float(source["dx_m"])
        iz, ix = np.array(shape) // 2
        point = [origin[0] + ix * dx, height[iz, ix], origin[2] + iz * dx]
    out = query(scene, [point])
    assert out["valid"].all() and not out["distance_signed"].any()
    torch.testing.assert_close(out["distance_m"], torch.zeros((1, 1)), atol=2e-5, rtol=0)
    assert scene.storage_summary["source_sha256"] == sha
    assert scene.storage_summary["objects"][0]["triangles"] == 2 * (shape[0] - 1) * (shape[1] - 1)
    assert scene.storage_summary["bvh_and_winding_bytes"] is None


def test_fixed_budget_and_missing_geometry_rejected():
    box = trimesh.creation.box()
    with pytest.raises(ValueError, match="1..512"):
        box_scene(513)
    with pytest.raises(ValueError, match="1..8"):
        WarpScene((), signed=(), known_bounds_m=[[-1]*3, [1]*3])
    with pytest.raises(ValueError, match="1..8"):
        WarpScene((box,) * 9, signed=(True,) * 9, known_bounds_m=[[-1]*3, [1]*3])


def test_full_512_query_budget():
    scene = box_scene(512)
    points = torch.zeros((1, 512, 3))
    points[..., 0] = torch.linspace(1.01, 1.99, 512)
    out = query(scene, points)
    torch.testing.assert_close(out["distance_m"], points[..., 0] - 1)
    assert out["valid"].all() and out["distance_signed"].all()


def test_pfnn_strict_math_triangle_trace_and_vjp_cpu():
    options = wp.get_module_options(geometry)
    assert options["fuse_fp"] is False and options["fast_math"] is False
    path = Path(__file__).resolve().parents[1] / "out/pfnn_mixed_20261015/pfnn_map_catalog_v1/011/scene.npz"
    scene = build_pfnn_scene(path, query_count=4)
    with np.load(path, allow_pickle=False) as source:
        height, origin, dx = source["heightfield_y_m"], source["min_point_world_xyz_m"], float(source["dx_m"])
        iz, ix = np.array(height.shape) // 2
        x, z, y = origin[0] + ix * dx, origin[2] + iz * dx, height[iz, ix]
        points = torch.tensor([[[x, y + 0.01, z], [x, y - 0.01, z],
                                [origin[0] - dx, y, z], [x, y + 0.02, z + dx]]],
                              dtype=torch.float32, requires_grad=True)
    out = query(scene, points)
    assert out["valid"].tolist() == [[True, True, False, True]]
    assert out["triangle_id"][0, 2] == -1 and out["barycentric_uv"][0, 2].eq(0).all()
    vertices = scene._meshes[0].points.numpy()
    faces = scene._meshes[0].indices.numpy().reshape(-1, 3)
    for i in (0, 1, 3):
        tri = vertices[faces[int(out["triangle_id"][0, i])]].astype(np.float64)
        u, v = out["barycentric_uv"][0, i].tolist()
        closest = tri[0]*u + tri[1]*v + tri[2]*(1-u-v)
        np.testing.assert_allclose(out["closest_point_world_m"][0, i].numpy(), closest, atol=2e-5, rtol=0)
    out["distance_m"].sum().backward()
    torch.testing.assert_close(points.grad, out["distance_gradient_world"])


@pytest.mark.skipif(os.environ.get("SCENE_CONTROL_V3_CUDA_ALLOCATED") != "1",
                    reason="Parent must explicitly allocate the sole CUDA correctness run")
@pytest.mark.parametrize("kind", ["box", "true3d", "pfnn011"])
def test_allocated_cuda_cpu_parity(kind):
    if kind == "box":
        points = torch.tensor([[[1.2, 1.3, 1.4], [1.2, 1.3, 0.1], [0.8, 0.2, 0.1],
                                [0, 1.3, 0], [6, 0, 0], [float("nan"), 0, 0]]])

        def build(device):
            return WarpScene((trimesh.creation.box(extents=[2, 2, 2]),), signed=(True,),
                              known_bounds_m=[[-5]*3, [5]*3], query_count=6, device=device)
    elif kind == "true3d":
        points = torch.tensor([[[0., 1.6, 0], [0, 2., 0], [1.3, 1.2, 0],
                                [1.55, 1.25, 0.1], [0, 0.1, 0], [5, 0, 0]]])

        def build(device):
            return build_true3d_scene(query_count=6, device=device)
    else:
        path = Path(__file__).resolve().parents[1] / "out/pfnn_mixed_20261015/pfnn_map_catalog_v1/011/scene.npz"
        with np.load(path, allow_pickle=False) as source:
            height, origin, dx = source["heightfield_y_m"], source["min_point_world_xyz_m"], float(source["dx_m"])
            iz, ix = np.array(height.shape) // 2
            x, z = origin[0] + ix * dx, origin[2] + iz * dx
            y = height[iz, ix]
            points = torch.tensor([[[x, y + 0.01, z], [x, y - 0.01, z],
                                    [origin[0] - dx, y, z], [x, y + 0.02, z + dx]]], dtype=torch.float32)

        def build(device):
            return build_pfnn_scene(path, query_count=4, device=device)

    cpu_scene, gpu_scene = build("cpu"), build("cuda:0")
    cpu_points = points.detach().requires_grad_(True)
    gpu_points = points.to("cuda:0").detach().requires_grad_(True)
    cpu_out, gpu_out = query(cpu_scene, cpu_points), query(gpu_scene, gpu_points)
    for key in ("valid", "scene_known", "object_id", "distance_signed", "distance_gradient_valid"):
        assert torch.equal(cpu_out[key], gpu_out[key].cpu())
    for key in ("distance_m", "closest_point_world_m", "distance_gradient_world"):
        torch.testing.assert_close(cpu_out[key], gpu_out[key].cpu(), atol=2e-5, rtol=2e-4,
                                   msg=lambda message: f"{key}: {message}")
    # Face normals can select either tied triangle at exact box edges/corners.
    smooth = [2, 3] if kind == "box" else ([0, 1, 2, 4] if kind == "true3d" else [])
    torch.testing.assert_close(cpu_out["normal_world"][:, smooth], gpu_out["normal_world"][:, smooth].cpu(),
                               atol=2e-5, rtol=2e-4)
    cpu_out["distance_m"].sum().backward()
    gpu_out["distance_m"].sum().backward()
    torch.testing.assert_close(cpu_points.grad, gpu_points.grad.cpu(), atol=2e-5, rtol=2e-4)
    if kind == "box":
        for axis in range(3):
            step = torch.zeros_like(gpu_points)
            step[..., axis] = 0.001
            plus = query(gpu_scene, gpu_points.detach() + step)["distance_m"]
            minus = query(gpu_scene, gpu_points.detach() - step)["distance_m"]
            torch.testing.assert_close((plus - minus)[:, :4] / 0.002,
                                       gpu_points.grad[:, :4, axis], atol=2e-4, rtol=0)
