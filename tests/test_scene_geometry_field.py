import json
import math
from dataclasses import asdict, replace

import pytest
import torch

from motionbricks.data.scene_geometry_field import (
    FACE_NAMES, SceneBox, SceneGeometryField, SceneManifest, example_scene_manifest,
)
from motionbricks.data.unified_scene_field import VolumetricEgoSampler


def single_box(box=None, bounds=10.):
    return SceneManifest('single', (-bounds,) * 3, (bounds,) * 3,
                         (box or SceneBox(1, 'box', 'solid', (0., 0., 0.), (1., 1., 1.)),))


def rotation():
    c, s, a, b = math.cos(.7), math.sin(.7), math.cos(.4), math.sin(.4)
    yaw = torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]], dtype=torch.float64)
    pitch = torch.tensor([[1., 0., 0.], [0., a, -b], [0., b, a]], dtype=torch.float64)
    return yaw @ pitch


def test_bridge_has_free_lower_and_upper_paths_and_a_solid_deck():
    field = SceneGeometryField(example_scene_manifest('bridge'))
    points = torch.tensor([[0., .5, 0.], [0., 1., 0.], [0., 2., 0.], [0., 2.5, 0.],
                           [0., -1.5, 0.]])
    d, n, valid, sid = field.query(points)
    assert valid.all()
    torch.testing.assert_close(d, torch.tensor([.5, .85, -.15, .35, .5]))
    torch.testing.assert_close(n, torch.tensor([[0., 1., 0.], [0., -1., 0.], [0., -1., 0.],
                                               [0., 1., 0.], [0., -1., 0.]]))
    assert sid.tolist() == [9, 14, 14, 15, 8]
    lower = torch.stack((torch.zeros(17), torch.full((17,), .8), torch.linspace(-2., 2., 17)), -1)
    upper = torch.stack((torch.linspace(-2., 2., 17), torch.full((17,), 2.35), torch.zeros(17)), -1)
    for path in (lower, upper):
        result = field.query(path)
        assert result.valid.all() and (result.distance > 0).all()
    assert (field.query(upper).surface_id == 15).all()


def test_low_ceiling_and_uneven_ground_are_both_observed():
    scene = example_scene_manifest('low_ceiling_uneven')
    field = SceneGeometryField(scene)
    ramp = scene.boxes[1]
    r = torch.tensor(ramp.rotation_local_to_world, dtype=torch.float64)
    local = torch.tensor([[-1., .4, 0.], [1., .4, 0.]], dtype=torch.float64)
    normal = r[:, 1]
    surface = local @ r.T + torch.tensor(ramp.center_xyz_m)
    assert surface[1, 1] - surface[0, 1] > .2
    result = field.query(surface + .03 * normal)
    assert result.valid.all() and result.surface_id.eq(15).all()
    torch.testing.assert_close(result.distance, torch.full((2,), .03, dtype=torch.float64))
    torch.testing.assert_close(result.normal, normal.expand(2, 3))
    ceiling = field.query(torch.tensor([[0., 1.40, 0.], [0., 1.50, 0.]]))
    assert ceiling.valid.all() and ceiling.surface_id.eq(20).all()
    torch.testing.assert_close(ceiling.distance, torch.tensor([.03, -.07]))
    torch.testing.assert_close(ceiling.normal, torch.tensor([[0., -1., 0.]]).expand(2, 3))


def test_oriented_wall_and_step_keep_distinct_surfaces():
    scene = example_scene_manifest('side_wall_step')
    wall = scene.boxes[1]
    r = torch.tensor(wall.rotation_local_to_world, dtype=torch.float64)
    point = torch.tensor([-.15, 0., 0.], dtype=torch.float64) @ r.T + torch.tensor(wall.center_xyz_m)
    field = SceneGeometryField(scene)
    result = field.query(point)
    assert result.valid and result.surface_id == 12
    torch.testing.assert_close(result.distance, torch.tensor(.03, dtype=torch.float64))
    torch.testing.assert_close(result.normal, -r[:, 0])
    step = field.query(torch.tensor([-.2, .45, .5]))
    assert step.valid and step.surface_id == 21
    torch.testing.assert_close(step.distance, torch.tensor(.05))
    torch.testing.assert_close(step.normal, torch.tensor([0., 1., 0.]))


@pytest.mark.parametrize('face', range(6), ids=FACE_NAMES)
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_all_faces_rotate_and_translate_with_the_box(face, dtype):
    r = rotation().to(dtype)
    box = SceneBox(7, 'rotated', 'solid', (3., 1., -4.), (.6, .8, 1.1), tuple(map(tuple, rotation().tolist())))
    field = SceneGeometryField(single_box(box, 20.))
    axis, sign = face // 2, -1 if face % 2 == 0 else 1
    half = torch.tensor(box.half_extent_xyz_m, dtype=dtype)
    local = torch.zeros(2, 3, dtype=dtype)
    local[:, axis] = sign * (half[axis] + torch.tensor([.2, -.1], dtype=dtype))
    points = local @ r.T + torch.tensor(box.center_xyz_m, dtype=dtype)
    result = field.query(points)
    assert result.valid.all() and result.surface_id.eq(42 + face).all()
    torch.testing.assert_close(result.distance, torch.tensor([.2, -.1], dtype=dtype), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result.normal, (sign * r[:, axis]).expand(2, 3), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('order', ((0, 1), (1, 0)))
def test_overlapping_solids_use_min_composition_and_stable_ids(order):
    # The union's distance to its boundary at the origin is 1, not .5.
    boxes = (SceneBox(9, 'left', 'solid', (-.5, 0., 0.), (1., 1., 1.)),
             SceneBox(2, 'right', 'solid', (.5, 0., 0.), (1., 1., 1.)))
    manifest = replace(single_box(), boxes=tuple(boxes[i] for i in order))
    result = SceneGeometryField(manifest).query(torch.tensor([[0., 0., 0.], [0., 2., 0.]]))
    assert result.valid.all()
    torch.testing.assert_close(result.distance, torch.tensor([-.5, 1.]))
    torch.testing.assert_close(result.normal, torch.tensor([[-1., 0., 0.], [0., 1., 0.]]))
    assert result.surface_id.tolist() == [12, 15]


def test_face_and_corner_ties_have_explicit_canonical_identity():
    result = SceneGeometryField(single_box()).query(torch.tensor([[0., 0., 0.], [-2., -2., 0.], [-1., -1., 0.]]))
    assert result.surface_id.tolist() == [6, 6, 6]
    assert result.valid.all()
    torch.testing.assert_close(result.normal, torch.tensor([[-1., 0., 0.],
                                                           [-math.sqrt(.5), -math.sqrt(.5), 0.],
                                                           [-1., 0., 0.]]))
    torch.testing.assert_close(result.distance, torch.tensor([-1., math.sqrt(2.), 0.]))


def test_normal_is_the_distance_gradient_away_from_ties():
    points = torch.tensor([[1.3, 1.4, .2], [.7, .2, .1]], dtype=torch.float64, requires_grad=True)
    result = SceneGeometryField(single_box()).query(points)
    gradient, = torch.autograd.grad(result.distance.sum(), points)
    torch.testing.assert_close(result.normal, gradient, atol=1e-12, rtol=0)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_arbitrary_batch_axes_scalar_and_empty_queries(dtype):
    field = SceneGeometryField(example_scene_manifest('bridge'))
    points = torch.tensor([[0., .5, 0.], [0., 2., 0.], [7., 0., 0.], [0., 2.5, 0.]], dtype=dtype)
    batched = points.reshape(1, 1, 2, 2, 3).expand(3, -1, -1, -1, -1)
    result = field.query(batched)
    assert result.distance.shape == result.valid.shape == result.surface_id.shape == (3, 1, 2, 2)
    assert result.normal.shape == batched.shape
    assert result.distance.dtype == result.normal.dtype == dtype
    assert result.valid.dtype == torch.bool and result.surface_id.dtype == torch.int64
    assert all(value.device.type == 'cpu' for value in result)
    for i in range(4):
        scalar = field.query(points[i])
        for batch_value, scalar_value in zip(result, scalar):
            flattened = batch_value.reshape(-1, 3) if batch_value.ndim == 5 else batch_value.reshape(-1)
            torch.testing.assert_close(flattened[i], scalar_value)
    empty = field.query(torch.empty(2, 0, 3, dtype=dtype))
    assert empty.distance.shape == (2, 0) and empty.normal.shape == (2, 0, 3)


def test_unknown_domain_includes_uncertified_boundary_distances():
    result = SceneGeometryField(single_box(bounds=4.)).query(torch.tensor([
        [0., 1.1, 0.], [3.9, 0., 0.], [4., 0., 0.], [4.1, 0., 0.], [0., -4.1, 0.]]))
    assert result.valid.tolist() == [True, False, False, False, False]
    assert not result.distance[1:].any() and not result.normal[1:].any()
    assert result.surface_id[1:].eq(-1).all()


def test_no_ground_is_inserted_when_manifest_contains_only_a_ceiling():
    ceiling = SceneBox(1, 'ceiling', 'ceiling', (0., 2., 0.), (1., .1, 1.))
    result = SceneGeometryField(single_box(ceiling)).query(torch.tensor([0., -.5, 0.]))
    assert result.valid and result.surface_id == 8
    torch.testing.assert_close(result.distance, torch.tensor(2.4))
    torch.testing.assert_close(result.normal, torch.tensor([0., -1., 0.]))


def test_mesh_winding_normals_and_face_ids_agree_with_queries():
    box = SceneBox(3, 'rotated', 'solid', (1., .5, -2.), (.7, .9, 1.2), tuple(map(tuple, rotation().tolist())))
    scene = single_box(box)
    mesh, = scene.meshes(dtype=torch.float64)
    assert (mesh.box_id, mesh.name, mesh.role) == (3, 'rotated', 'solid')
    assert mesh.vertices_xyz_m.shape == (8, 3) and mesh.triangles.shape == (12, 3)
    triangles = mesh.vertices_xyz_m[mesh.triangles]
    centers = triangles.mean(1)
    normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = normals / normals.norm(dim=-1, keepdim=True)
    assert ((centers - centers.new_tensor(box.center_xyz_m)) * normals).sum(-1).gt(0).all()
    result = SceneGeometryField(scene).query(centers + .02 * normals)
    assert result.valid.all()
    torch.testing.assert_close(result.distance, torch.full((12,), .02, dtype=torch.float64))
    torch.testing.assert_close(result.normal, normals)
    torch.testing.assert_close(result.surface_id, mesh.surface_ids)


@pytest.mark.parametrize('name', ('bridge', 'low_ceiling_uneven', 'side_wall_step'))
def test_manifest_is_json_serializable_and_meshes_cover_all_authored_solids(name):
    scene = example_scene_manifest(name)
    data = json.loads(json.dumps(asdict(scene)))
    data['boxes'] = tuple(SceneBox(**box) for box in data['boxes'])
    restored = SceneManifest(**data)
    assert restored == scene
    points = torch.tensor([[0., .5, 0.], [0., 1., 0.], [1., .1, 0.], [-1., 2., 0.]])
    for actual, expected in zip(SceneGeometryField(restored).query(points), SceneGeometryField(scene).query(points)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert [mesh.box_id for mesh in restored.meshes()] == sorted(box.box_id for box in scene.boxes)
    assert all(mesh.vertices_xyz_m.device.type == 'cpu' for mesh in restored.meshes())


def test_sampler_adapter_preserves_samples_and_masks_unknowns():
    field = SceneGeometryField(example_scene_manifest('bridge'))
    history = torch.zeros(1, 4, 5)
    history[..., 1], history[..., 3] = 1., 1.
    points = torch.tensor([[0., .5, 0.], [0., 1., 0.], [0., 2., 0.], [7., 0., 0.]])
    points = points[None, None].expand(1, 3, 4, 3)
    samples = field.query(points)
    sampler = VolumetricEgoSampler()
    observation = sampler.observe_points(history, points, field.observation_field())
    direct = sampler.observe_points(history, points, None, samples=samples[:3])
    for key in ('distance', 'normal', 'valid', 'features'):
        torch.testing.assert_close(observation[key], direct[key], atol=0, rtol=0)
    assert observation['features'].shape == (1, 3, 4, 15)
    assert not observation['features'][..., -1, :].any()
    full = sampler.observe(history, torch.zeros(1, 2), field.observation_field(), .8)
    assert full['features'].shape == (1, 3, 512, 15)
    assert torch.isfinite(full['features']).all() and full['valid'].any()
    with pytest.raises(ValueError, match='unknown'):
        sampler.observe_points(history, torch.full((1, 3, 4, 3), 100.), field.observation_field())


@pytest.mark.parametrize('points', (torch.ones(2, 4), torch.tensor([1, 2, 3]),
                                   torch.tensor([float('nan'), 0., 0.]), torch.tensor(1.)))
def test_invalid_points_are_rejected(points):
    with pytest.raises(ValueError, match='XYZ'):
        SceneGeometryField(single_box()).query(points)


def test_malformed_manifest_is_rejected():
    box = single_box().boxes[0]
    for update in ({'half_extent_xyz_m': (1., 0., 1.)}, {'center_xyz_m': (math.inf, 0., 0.)},
                   {'rotation_local_to_world': ((1., 0., 0.), (0., 1., 0.), (0., 0., -1.))},
                   {'rotation_local_to_world': ((1., 0., 0.), (0., 2., 0.), (0., 0., 1.))},
                   {'box_id': 0}):
        with pytest.raises(ValueError):
            replace(box, **update)
    for update in ({'boxes': ()}, {'boxes': (box, box)}, {'known_max_xyz_m': (-20., 10., 10.)}):
        with pytest.raises(ValueError):
            replace(single_box(), **update)
    with pytest.raises(ValueError, match='Unknown authored scene'):
        example_scene_manifest('unlisted')


def test_manifest_detaches_mutable_json_inputs():
    data = json.loads(json.dumps(asdict(example_scene_manifest('bridge'))))
    boxes = [SceneBox(**box) for box in data['boxes']]
    manifest = SceneManifest(data['name'], data['known_min_xyz_m'], data['known_max_xyz_m'], boxes)
    field = SceneGeometryField(manifest)
    data['boxes'][1]['center_xyz_m'][1] = 20.
    data['known_min_xyz_m'][0] = 100.
    boxes.clear()
    assert len(manifest.boxes) == 4
    assert manifest.known_min_xyz_m[0] == -6.
    mesh = manifest.meshes()[1]
    assert mesh.vertices_xyz_m[:, 1].min() < 2.
    assert field.query(torch.tensor([0., 2., 0.])).distance < 0
