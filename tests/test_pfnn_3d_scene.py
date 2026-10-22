import math

import pytest
import torch

from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF, PackedPrismSDF
from motionbricks.data.pfnn_3d_scene import (
    BatchedSceneField, CompositeSceneField, cpu_reference_query, path_observation,
)
from motionbricks.data.scene_geometry_field import SceneBox, SceneGeometryField, SceneManifest
from motionbricks.data.unified_scene_field import VolumetricEgoSampler, field_from_heightfield


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def terrain(*, step=False, hole=False, offset=0.):
    height = torch.full((21, 21), offset)
    if step:
        height[:, :11] += .4
    valid = torch.ones_like(height, dtype=torch.bool)
    if hole:
        valid[10, 10] = False
    # Absolute heights: this deliberately nonzero origin Y must not be added.
    return dict(heightfield_y_m=height, min_point_world_xyz_m=[-5., 42., -5.],
                dx_m=.5, heightfield_valid=valid)


def manifest(*boxes, bounds=7.):
    return SceneManifest('extras', (-bounds, -5., -bounds), (bounds, 8., bounds), tuple(boxes))


def deck():
    return SceneBox(3, 'deck', 'deck', (0., 2., 0.), (3., .15, 2.))


@pytest.mark.parametrize('step,hole', ((False, False), (True, False), (False, True), (True, True)))
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_no_extras_preserves_original_terrain_exactly(step, hole, dtype):
    scene = terrain(step=step, hole=hole)
    generator = torch.Generator().manual_seed(17)
    points = torch.rand(2, 3, 37, 3, generator=generator, dtype=dtype)
    points = points * torch.tensor([12., 4., 12.]) - torch.tensor([6., 1., 6.])
    expected = field_from_heightfield(scene).query(points)
    actual = CompositeSceneField(scene).query(points)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)


def test_ceiling_underside_and_bridge_have_distinct_vertical_surfaces_at_same_xz():
    field = CompositeSceneField(terrain(), manifest(deck()))
    points = torch.tensor([[0., .2, 0.], [0., 1.5, 0.], [0., 2., 0.], [0., 2.5, 0.]])
    distance, normal, valid = field.query(points)
    assert valid.all()
    torch.testing.assert_close(distance, torch.tensor([.2, .35, -.15, .35]))
    torch.testing.assert_close(normal, torch.tensor([[0., 1., 0.], [0., -1., 0.],
                                                   [0., -1., 0.], [0., 1., 0.]]))
    assert field.distance_semantics.endswith('nonexact_overlap_interior')
    assert field.eligible_for_penetration_gate is False


def test_sidewall_and_original_step_are_both_queryable():
    wall = SceneBox(2, 'wall', 'wall', (1.5, 1., 0.), (.1, 1., 2.))
    field = CompositeSceneField(terrain(step=True), manifest(wall))
    distance, normal, valid = field.query(torch.tensor([[1.3, 1., 0.], [-1., .45, 0.], [.35, .2, 0.]]))
    assert valid.all()
    torch.testing.assert_close(distance, torch.tensor([.1, .05, .1]), atol=1e-6, rtol=0)
    torch.testing.assert_close(normal, torch.tensor([[-1., 0., 0.], [0., 1., 0.], [1., 0., 0.]]))


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_query_gradients_are_distance_normals_for_both_components(dtype):
    angle = .3
    rotation = ((math.cos(angle), 0., math.sin(angle)), (0., 1., 0.),
                (-math.sin(angle), 0., math.cos(angle)))
    wall = SceneBox(2, 'wall', 'wall', (1.5, 1., 0.), (.1, 1., 2.), rotation)
    r = torch.tensor(rotation, dtype=dtype)
    wall_point = torch.tensor([-.22, 0., .3], dtype=dtype) @ r.T + torch.tensor(wall.center_xyz_m, dtype=dtype)
    points = torch.stack((wall_point, torch.tensor([-1., .47, 0.], dtype=dtype))).requires_grad_()
    distance, normal, valid = CompositeSceneField(terrain(step=True), manifest(wall)).query(points)
    assert valid.all()
    gradient, = torch.autograd.grad(distance.sum(), points)
    torch.testing.assert_close(gradient, normal, atol=1e-6, rtol=0)
    assert gradient.abs().sum() > 0


def test_overlap_is_component_distance_and_not_union_interior_depth():
    box = SceneBox(2, 'intersecting', 'solid', (0., .2, 0.), (1., .5, 1.))
    field = CompositeSceneField(terrain(), manifest(box))
    # At y=-.1 ground contributes -.1, box -.2; union boundary is y=.7.
    distance, normal, valid = field.query(torch.tensor([0., -.1, 0.]))
    assert valid
    torch.testing.assert_close(distance, torch.tensor(-.2))
    torch.testing.assert_close(normal, torch.tensor([0., -1., 0.]))
    assert not field.eligible_for_penetration_gate


def test_known_extra_does_not_certify_unknown_terrain_hole_or_map_exterior():
    wall = SceneBox(2, 'wall', 'wall', (.2, 1., 0.), (.1, 1., 2.))
    field = CompositeSceneField(terrain(hole=True), manifest(wall, bounds=15.))
    points = torch.tensor([[0., 1., 0.], [0., -1., 0.], [6., .01, 0.]])
    distance, normal, valid = field.query(points)
    assert not valid.any()
    assert not distance.any() and not normal.any()


def test_nearer_candidate_can_be_certified_by_other_components_empty_ball():
    wall = SceneBox(2, 'wall', 'wall', (.6, 1., 0.), (.1, 1., 2.))
    scene = terrain(hole=True)
    field = CompositeSceneField(scene, manifest(wall))
    point = torch.tensor([.4, 1., 0.])
    assert not field.terrain.query(point)[2]  # Ground is farther than the hole.
    distance, normal, valid = field.query(point)
    assert valid  # Wall .1 away, hole starts .15 away.
    torch.testing.assert_close(distance, torch.tensor(.1), atol=1e-6, rtol=0)
    torch.testing.assert_close(normal, torch.tensor([-1., 0., 0.]))
    assert not field.query(torch.tensor([.3, 1., 0.]))[2]  # Hole can hide nearer geometry.


def test_manifest_unknown_volume_does_not_disappear_behind_ground():
    box = SceneBox(2, 'remote', 'solid', (0., 2., 0.), (.2, .1, .2))
    extras = manifest(box, bounds=2.)
    field = CompositeSceneField(terrain(), extras)
    points = torch.tensor([[1.8, .1, 0.], [1.8, .3, 0.], [2.1, .01, 0.]])
    distance, normal, valid = field.query(points)
    assert valid.tolist() == [True, False, False]
    torch.testing.assert_close(distance, torch.tensor([.1, 0., 0.]))
    torch.testing.assert_close(normal[0], torch.tensor([0., 1., 0.]))


def test_missing_component_is_explicit_and_manifest_only_matches_original():
    extras = manifest(deck())
    points = torch.tensor([[0., 1., 0.], [0., 2., 0.], [8., 0., 0.]])
    result = CompositeSceneField(None, extras).query(points)
    reference = SceneGeometryField(extras).query(points)
    for actual, expected in zip(result, reference[:3]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(CompositeSceneField(None, extras).meshes()) == 1
    with pytest.raises(ValueError, match='Nonempty'):
        CompositeSceneField(None)
    for height in (torch.empty(0, 3), torch.full((2, 2), torch.nan)):
        with pytest.raises(ValueError, match='Known terrain'):
            CompositeSceneField(dict(terrain(), heightfield_y_m=height, heightfield_valid=torch.isfinite(height)))


def test_meshes_preserve_cell_bounds_steps_holes_and_manifest_ids():
    scene = terrain(step=True, hole=True)
    extras = manifest(deck())
    field = CompositeSceneField(scene, extras)
    meshes = field.meshes(dtype=torch.float64)
    ground, obstacle = meshes
    assert ground.box_id == 0 and ground.surface_ids.eq(0).all()
    assert field.terrain_bounds_xz_m == (-5.25, 5.25, -5.25, 5.25)
    torch.testing.assert_close(ground.vertices_xyz_m[:, [0, 2]].amin(0), torch.tensor([-5.25, -5.25], dtype=torch.float64))
    torch.testing.assert_close(ground.vertices_xyz_m[:, [0, 2]].amax(0), torch.tensor([5.25, 5.25], dtype=torch.float64))
    for a, b in zip(obstacle, extras.meshes(dtype=torch.float64)[0]):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        else:
            assert a == b
    triangles = ground.vertices_xyz_m[ground.triangles]
    normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= normals.norm(dim=-1, keepdim=True)
    assert torch.all((normals.abs() > 1e-10).sum(-1) == 1)  # No interpolated ramps.
    centers = triangles.mean(1)
    assert not ((centers[:, 0].abs() < .25) & (centers[:, 2].abs() < .25)).any()
    # Skip boundary distances which correctly remain unknown.
    distance, query_normal, valid = field.terrain.query(centers + .001 * normals)
    assert valid.any() and (normals[:, 0] > .9).any()
    torch.testing.assert_close(distance[valid], torch.full_like(distance[valid], .001), atol=2e-8, rtol=0)
    torch.testing.assert_close(query_normal[valid], normals[valid], atol=1e-6, rtol=0)


def test_float32_batch_uses_packed_prisms_and_preserves_plane_zero_normal():
    fields = [CompositeSceneField(terrain(step=True)), CompositeSceneField(terrain()),
              CompositeSceneField(terrain(step=True, hole=True), manifest(deck())),
              CompositeSceneField(None, manifest(deck()))]
    batch = BatchedSceneField(fields)
    indices, packed = batch._buffers[(torch.device('cpu'), torch.float32)]
    assert isinstance(packed, PackedPrismSDF) and indices.tolist() == [0, 2]
    generator = torch.Generator().manual_seed(25)
    points = (torch.rand(4, 3, 29, 3, generator=generator) * torch.tensor([12., 3., 12.])
              - torch.tensor([6., .2, 6.])).requires_grad_()
    actual = batch.query(points)
    expected = [f.query(p) for f, p in zip(fields, points)]
    for j in range(3):
        torch.testing.assert_close(actual[j], torch.stack([r[j] for r in expected]), atol=1e-6, rtol=0)
    actual_gradient, = torch.autograd.grad(actual[0].sum(), points, retain_graph=True)
    expected_gradient, = torch.autograd.grad(sum(r[0].sum() for r in expected), points)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=0)
    zeros = batch.query(torch.zeros(4, 1, 3))
    torch.testing.assert_close(zeros[1][1], torch.tensor([[0., 1., 0.]]), atol=0, rtol=0)
    assert zeros[2][1].all()


def test_double_batch_matches_cpu_reference_and_reuses_prepared_buffers():
    scenes = [terrain(step=True), terrain(step=True, hole=True)]
    extras = [None, manifest(deck())]
    batch = BatchedSceneField([CompositeSceneField(s, m) for s, m in zip(scenes, extras)], dtype=torch.float64)
    before = [f._buffers[(torch.device('cpu'), torch.float64)] for f in batch.fields]
    points = torch.tensor([[[-1., .7, 0.], [1., 1.5, 0.]],
                           [[0., 1., 0.], [1., 1.5, 0.]]], dtype=torch.float64)
    result = batch.query(points)
    for i in range(2):
        reference = cpu_reference_query(scenes[i], extras[i], points[i])
        for actual, expected in zip(result, reference):
            torch.testing.assert_close(actual[i], expected, atol=0, rtol=0)
        assert batch.fields[i]._buffers[(torch.device('cpu'), torch.float64)] is before[i]


def test_packed_terrain_honors_finite_but_unknown_cells():
    scene = terrain(step=True, hole=True)
    field = CompositeSceneField(scene)
    batch = BatchedSceneField([field, field])
    points = torch.tensor([[[0., .2, 0.], [.4, .7, 0.], [-1., .6, 0.]]]).expand(2, -1, -1)
    result = batch.query(points)
    assert result[2][:, 0].eq(False).all()
    for i in range(2):
        for actual, expected in zip(result, field.query(points[i])):
            torch.testing.assert_close(actual[i], expected, atol=1e-6, rtol=0)


def test_batch16_surface_zero_distances_and_normals_have_exact_serial_parity():
    box = SceneBox(3, 'deck', 'deck', (0., 2., 0.), (3., .125, 2.))
    variants = [CompositeSceneField(terrain()), CompositeSceneField(terrain(step=True)),
                CompositeSceneField(terrain(step=True, hole=True)),
                CompositeSceneField(terrain(), manifest(box))]
    fields = variants * 4
    points = torch.tensor([[-1., .4, 1.], [1., 0., 1.], [.25, .2, 1.],
                           [0., 1.875, 1.], [0., 2.125, 1.], [3., 2., 1.]])
    points = points[None].expand(16, -1, -1)
    batch = BatchedSceneField(fields)
    actual = batch.query(points)
    serial = [field.query(points[i]) for i, field in enumerate(fields)]
    for j in range(3):
        torch.testing.assert_close(actual[j], torch.stack([row[j] for row in serial]), atol=0, rtol=0)
    assert actual[2][:, :3].all()
    assert actual[2][0::4].all() and actual[2][1::4].all() and actual[2][3::4].all()
    # High queries over a nearby hole stay uncertified in both backends.
    assert not actual[2][2::4, 3:5].any()
    # The old analytic plane and old prism deliberately differ at zero distance.
    assert actual[0][0, 1] == 0
    torch.testing.assert_close(actual[1][0, 1], torch.tensor([0., 1., 0.]), atol=0, rtol=0)
    assert actual[0][1, :3].eq(0).all() and actual[1][1, :3].eq(0).all()
    assert actual[0][3, 3:].eq(0).all()
    torch.testing.assert_close(actual[1][3, 3:], torch.tensor([[0., -1., 0.], [0., 1., 0.], [1., 0., 0.]]), atol=0, rtol=0)


def test_scene_buffers_are_snapshots_of_mutable_terrain_inputs():
    scene = terrain(step=True)
    field = CompositeSceneField(scene)
    points = torch.tensor([[-1., .6, 0.], [1., .6, 0.]])
    before = field.query(points)
    scene['heightfield_y_m'].add_(3.)
    scene['heightfield_valid'].zero_()
    scene['min_point_world_xyz_m'][0] += 10.
    for actual, expected in zip(field.query(points), before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize('shape', ((3,), (2, 0, 3), (1, 2, 3, 4, 3)))
def test_scalar_empty_and_arbitrary_leading_shapes(shape):
    points = torch.zeros(shape)
    field = CompositeSceneField(terrain(), manifest(deck()))
    distance, normal, valid = field.query(points)
    assert distance.shape == valid.shape == points.shape[:-1] and normal.shape == points.shape
    batched = BatchedSceneField([field, field]).query(points[None].expand(2, *shape))
    assert batched[0].shape == (2, *shape[:-1])
    assert batched[1].shape == (2, *shape)


@pytest.mark.parametrize('points', (torch.tensor([1, 2, 3]), torch.ones(2, 4),
                                   torch.tensor([0., math.inf, 0.]), torch.tensor(1.)))
def test_rejects_invalid_queries(points):
    field = CompositeSceneField(terrain())
    with pytest.raises(ValueError, match='XYZ'):
        field.query(points)


def test_rejects_empty_batch_or_wrong_batch_shape():
    with pytest.raises(ValueError, match='Nonempty'):
        BatchedSceneField([])
    with pytest.raises(ValueError, match='row per scene'):
        BatchedSceneField([CompositeSceneField(terrain())]).query(torch.zeros(2, 7, 3))


class CountingField:
    def __init__(self, field):
        self.field, self.shapes = field, []

    def query(self, points):
        self.shapes.append(tuple(points.shape))
        return self.field.query(points)


@pytest.mark.parametrize('stationary', (False, True))
def test_path_requeries_only_corridor_and_matches_existing_curve(stationary):
    from motionbricks.data.root_path_terrain import PredictedRootObservation

    sampler = VolumetricEgoSampler()
    field = CountingField(BatchedSceneField([CompositeSceneField(terrain(step=True), manifest(deck())),
                                             CompositeSceneField(terrain())]))
    history = torch.zeros(2, 4, 5)
    history[..., 1], history[..., 3] = 1., 1.
    history[1, :, 3:5] = torch.tensor([0., 1.])
    history[:, :, 0] = torch.arange(4) * .01
    command = torch.tensor([[0., 1.], [.8, 0.]])
    coarse = sampler.observe(history, command, field, .8)
    before = {k: v.clone() for k, v in coarse.items()}
    theta = torch.linspace(0., math.pi / 2, 8)
    path = torch.stack((1 - theta.cos(), theta.sin()), -1)[None].expand(2, -1, -1).clone()
    path += history[:, -1:, [0, 2]]
    if stationary:
        path[:] = history[:, -1:, [0, 2]]
    history.requires_grad_()
    path.requires_grad_()
    result = path_observation(sampler, history, path, coarse, field)
    assert field.shapes == [(2, 3, 512, 3), (2, 1, 512, 3)]
    centers, right = PredictedRootObservation._curve(None, path, 8, history)
    ids = torch.arange(512)
    lateral = torch.linspace(-sampler.config.corridor_halfwidth_m, sampler.config.corridor_halfwidth_m, 8)
    expected = coarse['points'].clone()
    xz = centers[:, ids % 8] + right[:, ids % 8] * lateral[ids // 64][None, :, None]
    expected[:, 1, :, 0], expected[:, 1, :, 2] = xz[..., 0], xz[..., 1]
    torch.testing.assert_close(result['points'], expected, atol=0, rtol=0)
    for key in before:
        torch.testing.assert_close(coarse[key], before[key], atol=0, rtol=0)
        torch.testing.assert_close(result[key][:, [0, 2]], before[key][:, [0, 2]], atol=0, rtol=0)
        assert not result[key].requires_grad
    full = sampler.observe_points(history, expected, field)
    for key in full:
        torch.testing.assert_close(result[key], full[key], atol=0, rtol=0)


def test_curved_predicted_path_observes_wall_missed_by_command_corridor():
    wall = SceneBox(2, 'wall', 'wall', (1.8, 1., .6), (.1, 1., .6))
    field = BatchedSceneField([CompositeSceneField(terrain(), manifest(wall))])
    sampler = VolumetricEgoSampler()
    history = torch.tensor([[[0., 1., 0., 1., 0.]]]).expand(1, 4, 5)
    coarse = sampler.observe(history, torch.tensor([[0., 1.]]), field, .8)
    path = torch.stack((torch.linspace(0., 1.65, 8), torch.linspace(0., .6, 8)), -1)[None]
    result = path_observation(sampler, history, path, coarse, field)
    assert result['distance'][:, 1].abs().min() < coarse['distance'][:, 1].abs().min()
    assert (result['normal'][:, 1, :, 0].abs() > .9).any()


def test_all_unknown_observation_is_rejected():
    field = BatchedSceneField([CompositeSceneField(terrain())])
    history = torch.tensor([[[100., 1., 100., 1., 0.]]]).expand(1, 4, 5)
    with pytest.raises(ValueError, match='unknown'):
        VolumetricEgoSampler().observe(history, torch.zeros(1, 2), field, .8)
