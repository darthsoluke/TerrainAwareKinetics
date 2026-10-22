"""Bounded, authored solid geometry; independent of motion and height fields."""

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch


IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
FACE_NAMES = ('-X', '+X', '-Y', '+Y', '-Z', '+Z')
_CORNERS = ((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1))
_TRIANGLES = ((0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
              (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
              (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3))


def _vector3(value):
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.shape != (3,) or not torch.isfinite(tensor).all():
        raise ValueError('A finite XYZ vector is required')
    return tensor


@dataclass(frozen=True)
class SceneBox:
    """Closed OBB, metres, Y up; rotation columns are local axes in world space.

    Face IDs are 6 * box_id + the index in FACE_NAMES, independent of list order.
    """

    box_id: int
    name: str
    role: str
    center_xyz_m: tuple[float, float, float]
    half_extent_xyz_m: tuple[float, float, float]
    rotation_local_to_world: tuple = IDENTITY

    def __post_init__(self):
        center = _vector3(self.center_xyz_m)
        half = _vector3(self.half_extent_xyz_m)
        if type(self.box_id) is not int or self.box_id < 1 or not self.name or not self.role:
            raise ValueError('Positive integer box ID, name and role required')
        if not (half > 0).all():
            raise ValueError('Box half extents must be positive')
        rotation = torch.as_tensor(self.rotation_local_to_world, dtype=torch.float64)
        if (rotation.shape != (3, 3) or not torch.isfinite(rotation).all()
                or not torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=rotation.dtype),
                                      atol=1e-8, rtol=0)
                or not torch.isclose(rotation.det(), rotation.new_tensor(1.), atol=1e-8, rtol=0)):
            raise ValueError('A proper orthonormal local-to-world rotation is required')
        object.__setattr__(self, 'center_xyz_m', tuple(center.tolist()))
        object.__setattr__(self, 'half_extent_xyz_m', tuple(half.tolist()))
        object.__setattr__(self, 'rotation_local_to_world', tuple(map(tuple, rotation.tolist())))


class SceneMesh(NamedTuple):
    box_id: int
    name: str
    role: str
    vertices_xyz_m: torch.Tensor
    triangles: torch.Tensor
    surface_ids: torch.Tensor


@dataclass(frozen=True)
class SceneManifest:
    """Complete list of known solids and known empty space inside an XYZ AABB."""

    name: str
    known_min_xyz_m: tuple[float, float, float]
    known_max_xyz_m: tuple[float, float, float]
    boxes: tuple[SceneBox, ...]

    def __post_init__(self):
        lower, upper = _vector3(self.known_min_xyz_m), _vector3(self.known_max_xyz_m)
        if not (lower < upper).all():
            raise ValueError('Known bounds must have positive XYZ extent')
        if (not self.name or not self.boxes
                or len({b.box_id for b in self.boxes}) != len(self.boxes)
                or len({b.name for b in self.boxes}) != len(self.boxes)):
            raise ValueError('A named scene with nonempty, uniquely identified boxes is required')
        object.__setattr__(self, 'known_min_xyz_m', tuple(lower.tolist()))
        object.__setattr__(self, 'known_max_xyz_m', tuple(upper.tolist()))
        object.__setattr__(self, 'boxes', tuple(self.boxes))

    def meshes(self, *, dtype=torch.float32, device='cpu'):
        """Closed outward-wound box meshes; IDs are per triangle, not a union mesh."""
        corners = torch.tensor(_CORNERS, dtype=dtype, device=device)
        triangles = torch.tensor(_TRIANGLES, dtype=torch.int64, device=device)
        meshes = []
        for box in sorted(self.boxes, key=lambda b: b.box_id):
            rotation = torch.tensor(box.rotation_local_to_world, dtype=dtype, device=device)
            vertices = ((corners * corners.new_tensor(box.half_extent_xyz_m)) @ rotation.T
                        + corners.new_tensor(box.center_xyz_m))
            ids = torch.arange(6, dtype=torch.int64, device=device).repeat_interleave(2) + 6 * box.box_id
            meshes.append(SceneMesh(box.box_id, box.name, box.role, vertices, triangles, ids))
        return tuple(meshes)


class GeometryQuery(NamedTuple):
    distance: torch.Tensor
    normal: torch.Tensor
    valid: torch.Tensor
    surface_id: torch.Tensor


class SceneGeometryField:
    """Batched min of exact OBB SDFs, with an explicit known-domain certificate.

    Min composition is exact outside the solid union, but not generally the
    union's interior SDF. Normals belong to the winning component. Equal solid
    distances choose the smallest box ID; face ties use FACE_NAMES order.
    """

    def __init__(self, manifest: SceneManifest):
        self.manifest = manifest
        boxes = sorted(manifest.boxes, key=lambda b: b.box_id)
        self._centers = torch.tensor([b.center_xyz_m for b in boxes], dtype=torch.float64)
        self._half = torch.tensor([b.half_extent_xyz_m for b in boxes], dtype=torch.float64)
        self._rotations = torch.tensor([b.rotation_local_to_world for b in boxes], dtype=torch.float64)
        self._ids = torch.tensor([b.box_id for b in boxes], dtype=torch.int64)
        self._lower = _vector3(manifest.known_min_xyz_m)
        self._upper = _vector3(manifest.known_max_xyz_m)

    def query(self, points: torch.Tensor) -> GeometryQuery:
        """Accept [...,3] floating XYZ; preserve leading shape, dtype and device.

        Outside the known volume, or if the distance ball crosses its boundary,
        return valid=False, distance/normal=0, surface_id=-1.
        """
        if (points.ndim < 1 or points.shape[-1] != 3 or not points.is_floating_point()
                or not torch.isfinite(points).all()):
            raise ValueError('Finite floating world XYZ points with shape [...,3] required')
        rotation = self._rotations.to(points)
        local = torch.einsum('...si,sij->...sj', points[..., None, :] - self._centers.to(points), rotation)
        q = local.abs() - self._half.to(points)
        outside = q.clamp_min(0)
        length = torch.linalg.vector_norm(outside, dim=-1)
        component_distance = length + q.amax(-1).clamp_max(0)

        # Interior: nearest face. Exterior edge/corner: dominant incident face.
        half = self._half.to(points)
        planes = torch.stack((-local[..., 0] - half[:, 0], local[..., 0] - half[:, 0],
                              -local[..., 1] - half[:, 1], local[..., 1] - half[:, 1],
                              -local[..., 2] - half[:, 2], local[..., 2] - half[:, 2]), -1)
        face = planes.argmax(-1)
        face_normal = torch.zeros_like(local).scatter(-1, (face // 2)[..., None],
                                                     (2 * (face % 2) - 1).to(points)[..., None])
        exterior_normal = (outside * torch.where(local >= 0, 1., -1.)
                           / length.clamp_min(torch.finfo(points.dtype).tiny)[..., None])
        local_normal = torch.where((length > 0)[..., None], exterior_normal, face_normal)
        normals = torch.einsum('...sj,sij->...si', local_normal, rotation)
        distance, winner = component_distance.min(-1)
        normal = normals.gather(-2, winner[..., None, None].expand(*winner.shape, 1, 3)).squeeze(-2)
        surface_id = (6 * self._ids.to(points.device) + face).gather(-1, winner[..., None]).squeeze(-1)
        margin = torch.minimum(points - self._lower.to(points), self._upper.to(points) - points).amin(-1)
        valid = (margin >= 0) & (distance.abs() <= margin)
        return GeometryQuery(torch.where(valid, distance, 0), torch.where(valid[..., None], normal, 0),
                             valid, torch.where(valid, surface_id, -1))

    def observation_field(self):
        """Three-value field adapter for VolumetricEgoSampler.observe_points."""
        return _ObservationField(self)


class _ObservationField:
    def __init__(self, field):
        self.field = field

    def query(self, points):
        return self.field.query(points)[:3]


def example_scene_manifest(name: str) -> SceneManifest:
    """Three explicitly authored static fixtures; ground is a finite solid box."""
    ground = SceneBox(1, 'ground', 'ground', (0., -.5, 0.), (5., .5, 5.))
    if name == 'low_ceiling_uneven':
        c, s = math.cos(.12), math.sin(.12)
        boxes = (ground,
                 SceneBox(2, 'sloped_ground', 'ground', (0., -.1, 0.), (1.5, .4, 1.8),
                          ((c, -s, 0.), (s, c, 0.), (0., 0., 1.))),
                 SceneBox(3, 'low_ceiling', 'ceiling', (0., 1.55, 0.), (1.9, .12, 2.)))
    elif name == 'side_wall_step':
        c, s = math.cos(.25), math.sin(.25)
        boxes = (ground,
                 SceneBox(2, 'side_wall', 'wall', (1.5, 1.15, 0.), (.12, 1.15, 2.3),
                          ((c, 0., s), (0., 1., 0.), (-s, 0., c))),
                 SceneBox(3, 'step', 'step', (-.2, .2, .5), (.65, .2, .9)))
    elif name == 'bridge':
        boxes = (ground,
                 SceneBox(2, 'bridge_deck', 'deck', (0., 2., 0.), (2.4, .15, 1.2)),
                 SceneBox(3, 'left_support', 'support', (-2.2, .925, 0.), (.18, .925, 1.2)),
                 SceneBox(4, 'right_support', 'support', (2.2, .925, 0.), (.18, .925, 1.2)))
    else:
        raise ValueError(f'Unknown authored scene: {name}')
    return SceneManifest(name, (-6., -3., -6.), (6., 7., 6.), boxes)
