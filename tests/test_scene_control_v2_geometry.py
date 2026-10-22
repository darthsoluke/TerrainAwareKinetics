import numpy as np
import pytest
import trimesh

from motionbricks.scene_control_v2.geometry import (
    LegacyHeightfieldField, MeshField, SceneObject, SceneQuery, VolumeField,
)


def object_at(field, xyz=(0, 0, 0), *, object_id=0, frames=2):
    transform = np.tile(np.eye(4), (frames, 1, 1))
    transform[:, :3, 3] = xyz
    return SceneObject(object_id, str(object_id), field, transform, np.ones(frames, bool))


def test_signed_box_distance_and_outward_normal():
    field = MeshField(trimesh.creation.box(extents=[2, 2, 2]))
    out = field.query_local(np.array([[0, 1.3, 0], [0, 0.8, 0], [0, -1.4, 0]]))
    np.testing.assert_allclose(out["distance_m"], [0.3, -0.2, 0.4], atol=1e-6)
    np.testing.assert_allclose(out["normal_world"], [[0, 1, 0], [0, 1, 0], [0, -1, 0]])
    assert out["valid"].all()


def test_ceiling_overhang_and_hole_are_true_3d():
    # A closed annular prism has an actual through-hole, not a heightfield hole.
    ring = trimesh.creation.annulus(r_min=0.4, r_max=1.0, height=0.2, sections=32)
    scene = SceneQuery((object_at(MeshField(ring)),), [[-3, -3, -3], [3, 3, 3]])
    out = scene.query_world([[0, 0, 0], [0.7, 0, 0], [0.7, 0, 0.4]], np.array([0, 0, 0]))
    assert out["distance_m"][0] > 0.39  # air in the hole
    assert out["distance_m"][1] < -0.09  # annular solid
    assert out["distance_m"][2] > 0.29  # air above exactly same horizontal location
    ceiling = object_at(MeshField(trimesh.creation.box(extents=[2, 0.2, 2])), xyz=(0, 1.2, 0))
    low = SceneQuery((ceiling,), [[-3, -1, -3], [3, 3, 3]])
    result = low.query_world([[0, 1.05, 0], [0, 1.35, 0]], np.array([0, 0]))
    np.testing.assert_allclose(result["distance_m"], [0.05, 0.05], atol=1e-6)
    np.testing.assert_allclose(result["normal_world"], [[0, -1, 0], [0, 1, 0]])


def test_dynamic_rotation_translation_anchor_and_normal():
    field = MeshField(trimesh.creation.box(extents=[2, 1, 1]))
    tracks = np.tile(np.eye(4), (2, 1, 1))
    tracks[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    tracks[1, :3, 3] = [2, 3, 4]
    obj = SceneObject(7, "moving", field, tracks, np.array([True, True]))
    scene = SceneQuery((obj,), [[-10, -10, -10], [10, 10, 10]])
    anchors = scene.anchors_world([7, 7], [0, 1], [[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(anchors["world_points_m"], [[1, 0, 0], [2, 4, 4]])
    np.testing.assert_allclose(anchors["normal_world"], [[1, 0, 0], [0, 1, 0]])
    out = scene.query_world([[1.25, 0, 0], [2, 4.25, 4]], np.array([0, 1]))
    np.testing.assert_allclose(out["distance_m"], [0.25, 0.25], atol=1e-6)
    np.testing.assert_allclose(out["normal_world"], [[1, 0, 0], [0, 1, 0]])
    assert out["object_id"].tolist() == [7, 7]


def test_invalid_world_region_object_and_range():
    obj = object_at(MeshField(trimesh.creation.box()), frames=2)
    obj.valid[1] = False
    scene = SceneQuery((obj,), [[-2, -2, -2], [2, 2, 2]],
                       validity_fn=lambda p, t: p[:, 2] <= 0)
    out = scene.query_world([[0, 0.8, -0.2], [0, 0.8, 0.2], [0, 0.8, -0.2],
                             [3, 0, 0], [np.nan, 0, 0]], np.array([0, 0, 1, 0, 0]))
    assert out["valid"].tolist() == [True, False, False, False, False]
    assert (out["distance_m"][~out["valid"]] == 0).all()
    assert (out["object_id"][~out["valid"]] == -1).all()
    bounded = MeshField(trimesh.creation.box(), max_distance_m=0.1)
    assert not bounded.query_local(np.array([[0, 2, 0]]))["valid"].any()


def test_volume_interpolation_and_unknown_nodes():
    axes = (np.arange(3.0),) * 3
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    normals = np.zeros((*xyz.shape[:-1], 3))
    normals[..., 1] = 1
    valid = np.ones(xyz.shape[:-1], bool)
    valid[2, 2, 2] = False
    volume = VolumeField(axes, xyz[..., 1] - 1, normals, valid)
    out = volume.query_local(np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [3, 0, 0]]))
    np.testing.assert_allclose(out["distance_m"], [-0.5, 0, 0])
    assert out["valid"].tolist() == [True, False, False]
    assert "trilinear" in volume.distance_semantics


def test_no_silent_heightfield_or_open_mesh_sign():
    plane = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 0, 1]], faces=[[0, 1, 2]])
    with pytest.raises(ValueError, match="watertight"):
        MeshField(plane)
    unsigned = MeshField(plane, signed=False)
    assert unsigned.query_local(np.array([[0.2, -0.2, 0.2]]))["distance_m"][0] > 0
    legacy = LegacyHeightfieldField([0, 1], [0, 1], np.zeros((2, 2)), np.ones((2, 2), bool))
    assert "vertical_clearance" in legacy.distance_semantics
    np.testing.assert_allclose(legacy.query_local(np.array([[0.5, 0.3, 0.5]]))["distance_m"], [0.3])


def test_overlapping_objects_report_component_not_exact_union():
    field = MeshField(trimesh.creation.box(extents=[2, 2, 2]))
    scene = SceneQuery((object_at(field, object_id=3), object_at(field, (0.5, 0, 0), object_id=9)),
                       [[-4, -4, -4], [4, 4, 4]])
    out = scene.query_world([[0.25, 0.0, 0.0]], np.array([0]))
    assert out["distance_m"][0] < 0
    assert out["distance_semantics"].startswith("minimum_component:")
    assert out["object_id"][0] in (3, 9)


def test_nonrigid_transform_is_rejected():
    field = MeshField(trimesh.creation.box())
    transform = np.eye(4)[None]
    transform[0, 0, 0] = 2
    with pytest.raises(ValueError, match="SE"):
        SceneObject(0, "scaled", field, transform, np.array([True]))


def test_corner_distance_gradient_is_not_triangle_face_normal():
    import torch

    field = MeshField(trimesh.creation.box(extents=[2, 2, 2]))
    scene = SceneQuery((object_at(field),), [[-4, -4, -4], [4, 4, 4]])
    points = torch.tensor([[1.2, 1.3, 1.4], [1.2, 1.3, 0.1], [0.8, 0.2, 0.1]], requires_grad=True)
    out = scene.query_world_torch(points, torch.zeros(3, dtype=torch.long))
    out["distance_m"].sum().backward()
    expected = np.array([[0.2, 0.3, 0.4], [0.2, 0.3, 0], [1, 0, 0]])
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(points.grad.numpy(), expected, atol=2e-6)
    assert not np.allclose(out["normal_world"][0].numpy(), expected[0])
    np.testing.assert_allclose(out["closest_point_world_m"][0].numpy(), [1, 1, 1], atol=1e-6)
    assert out["closest_point_valid"].all() and out["distance_gradient_valid"].all()
    p = points.detach().numpy()
    finite_difference = np.zeros_like(p)
    for axis in range(3):
        step = np.zeros_like(p)
        step[:, axis] = 1e-3
        plus = scene.query_world(p + step, np.zeros(3, dtype=np.int32))["distance_m"]
        minus = scene.query_world(p - step, np.zeros(3, dtype=np.int32))["distance_m"]
        finite_difference[:, axis] = (plus - minus) / 2e-3
    np.testing.assert_allclose(points.grad.numpy(), finite_difference, atol=8e-5)


def test_torch_gradient_transforms_and_invalid_slots():
    import torch

    field = MeshField(trimesh.creation.box(extents=[2, 1, 1]))
    obj = object_at(field, (1, 2, 0))
    obj.object_to_world[:, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    scene = SceneQuery((obj,), [[-4, -4, -4], [4, 4, 4]])
    points = torch.tensor([[1.0, 3.2, 0.0], [9.0, 0.0, 0.0]], requires_grad=True)
    out = scene.query_world_torch(points, torch.tensor([0, 0]))
    out["distance_m"].sum().backward()
    np.testing.assert_allclose(points.grad.numpy(), [[0, 1, 0], [0, 0, 0]], atol=1e-6)
    assert out["valid"].tolist() == [True, False]


def test_torch_volume_gradient_is_interpolant_derivative():
    import torch

    axes = (np.arange(3.0),) * 3
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    distance = xyz[..., 0] * xyz[..., 1] + xyz[..., 2] * 0.5
    normals = np.zeros((*xyz.shape[:-1], 3))
    normals[..., 1] = 1  # Deliberately differs from the distance derivative.
    volume = VolumeField(axes, distance, normals, np.ones(xyz.shape[:-1], bool))
    scene = SceneQuery((object_at(volume),), [[-1, -1, -1], [4, 4, 4]])
    points = torch.tensor([[0.3, 0.4, 0.5]], requires_grad=True)
    out = scene.query_world_torch(points, torch.tensor([0]))
    out["distance_m"].sum().backward()
    np.testing.assert_allclose(points.grad.numpy(), [[0.4, 0.3, 0.5]], atol=1e-6)
    assert not out["closest_point_valid"].any()


def test_unknown_object_transform_does_not_mean_empty_space():
    import torch

    floor = object_at(MeshField(trimesh.creation.box(extents=[8, 0.2, 8])),
                      (0, -0.1, 0), object_id=0)
    unknown = object_at(MeshField(trimesh.creation.box()), (2, 1, 0), object_id=1)
    unknown.valid[1] = False
    unknown.object_to_world[1] = np.nan
    scene = SceneQuery((floor, unknown), [[-4, -4, -4], [4, 4, 4]])
    points = torch.tensor([[0, 0.05, 0], [0, 0.05, 0]], requires_grad=True)
    out = scene.query_world_torch(points, torch.tensor([0, 1]))
    assert out["valid"].tolist() == [True, False]
    assert out["object_id"].tolist() == [0, -1]
    assert out["valid_semantics"] == "all_declared_components_known"
    out["distance_m"].sum().backward()
    np.testing.assert_allclose(points.grad.numpy(), [[0, 1, 0], [0, 0, 0]], atol=1e-6)


@pytest.mark.parametrize("unknown_first", [False, True])
def test_unknown_component_geometry_invalidates_other_valid_distance(unknown_first):
    axes = (np.arange(3.0),) * 3
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    normals = np.zeros((*xyz.shape[:-1], 3))
    normals[..., 1] = 1
    fully_known = np.ones(xyz.shape[:-1], bool)
    unknown_nodes = fully_known.copy()
    unknown_nodes[2, 2, 2] = False
    floor = object_at(VolumeField(axes, xyz[..., 1], normals, fully_known), object_id=0)
    partial = object_at(VolumeField(axes, xyz[..., 1] + 3, normals, unknown_nodes), object_id=1)
    objects = (partial, floor) if unknown_first else (floor, partial)
    scene = SceneQuery(objects, [[-1, -1, -1], [4, 4, 4]])
    result = scene.query_world([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], np.array([0, 0]))
    assert result["valid"].tolist() == [True, False]
    assert result["object_id"].tolist() == [0, -1]
    assert not result["distance_gradient_valid"][1] and not result["closest_point_valid"][1]
    assert result["distance_m"][1] == 0
    assert not result["normal_world"][1].any() and not result["distance_gradient_world"][1].any()
